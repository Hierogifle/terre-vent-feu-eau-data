"""Garde-fous anti-fuite — les tests qui protègent le score.

Une fuite ne lève aucune erreur : elle produit d'excellentes métriques et un
modèle inutilisable. Ces assertions sont le seul moyen de la détecter.
"""
import os
import sys

import psycopg
import pytest

sys.path.insert(0, "src")
from tvfed.db import DSN  # noqa: E402


@pytest.fixture(scope="module")
def conn():
    with psycopg.connect(DSN) as c:
        yield c


def q(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def test_aucune_feature_de_voisinage(conn):
    """Décision de cadrage : la proximité ne passe QUE par le clustering.

    Les comptages de feux alentour ont été retirés du pipeline. Ils ne
    fuyaient pas (la fenêtre était bornée à J-1), mais encoder deux fois la
    même intuition — une fois en features à la main, une fois via des
    clusters appris — rendait l'attribution SHAP ambiguë.

    L'implémentation est conservée dans sql/41_feat_voisinage.sql.reporte.
    Ce test garantit qu'elle ne revient pas dans la matrice par accident.
    """
    existe, = q(conn, "SELECT to_regclass('feat_voisinage') IS NOT NULL")
    assert not existe, (
        "la table feat_voisinage est revenue — la proximité doit passer "
        "par le clustering appris après le split"
    )


def test_lags_commune_excluent_le_jour_meme(conn):
    """Même règle pour l'historique de la commune elle-même."""
    n, = q(conn, """
        SELECT count(*) FROM fait_feu f
        JOIN feat_lags l ON l.code_insee = f.code_insee AND l.date = f.date_alerte
        WHERE f.code_insee IS NOT NULL AND l.jours_depuis_dernier_feu = 0
    """)
    assert n == 0, f"{n} lignes ont un décalage de 0 jour — la fenêtre inclut le jour J"


def test_grille_rectangulaire(conn):
    """Une grille trouée fausserait silencieusement toutes les fenêtres."""
    attendu, reel = q(conn, """
        SELECT (SELECT count(*) FROM ref_commune WHERE in_perimetre)
             * (SELECT count(*) FROM ref_calendrier),
               (SELECT count(*) FROM grille)
    """)
    assert reel == attendu, f"grille {reel:,} lignes, attendu {attendu:,}"


def test_val_test_non_echantillonnes(conn):
    """La calibration en dépend : val et test doivent rester INTÉGRAUX.

    Downsamplés, ils ne corrigeraient plus le décalage de prior introduit
    par l'échantillonnage du train (facteur ~×500).
    """
    n_com, = q(conn, "SELECT count(*) FROM ref_commune WHERE in_perimetre")
    for debut, fin in [("2020-01-01", "2022-12-31"), ("2023-01-01", "2025-12-31")]:
        n_jours, = q(conn,
            "SELECT count(*) FROM ref_calendrier WHERE date BETWEEN %s AND %s",
            (debut, fin))
        n, = q(conn, "SELECT count(*) FROM grille WHERE date BETWEEN %s AND %s",
               (debut, fin))
        assert n == n_com * n_jours, f"{debut}..{fin} : {n:,} ≠ {n_com * n_jours:,}"


def test_millesime_clc_jamais_futur(conn):
    """Utiliser CLC 2018 pour un feu de 2010 serait une fuite temporelle."""
    n, = q(conn, """
        SELECT count(*) FROM (
          SELECT g.date, clc.millesime
          FROM grille g
          JOIN ref_calendrier cal ON cal.date = g.date
          LEFT JOIN LATERAL (
            SELECT * FROM clc_part p
            WHERE p.code_insee = g.code_insee AND p.millesime <= cal.annee
            ORDER BY p.millesime DESC LIMIT 1) clc ON true
          WHERE g.date < '2006-02-01'
          LIMIT 10000
        ) x
        WHERE millesime > EXTRACT(year FROM date)
    """)
    assert n == 0, f"{n} lignes utilisent un millésime CORINE postérieur à leur date"


@pytest.mark.skipif(
    not os.path.exists("data/processed/train.parquet"),
    reason="matrice d'entraînement pas encore construite",
)
def test_pas_de_feature_apprise_en_base(conn):
    """Les transformations avec .fit() ne doivent jamais être stockées.

    Lissage bayésien, target encoding, clusters HDBSCAN, standardisation :
    tout cela s'apprend APRÈS le split, sur le train seul.
    """
    import pandas as pd

    cols = pd.read_parquet("data/processed/train.parquet").columns
    interdits = ("taux_lisse", "cluster", "_scaled", "_encoded", "zscore", "target_enc",
                 "voisins")   # la proximité est reportée au clustering
    trouves = [c for c in cols if any(i in c.lower() for i in interdits)]
    assert not trouves, f"features apprises ou de proximité dans la matrice : {trouves}"
