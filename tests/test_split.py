"""Le split temporel — bornes, étanchéité, et cohérence entre les 3 définitions.

Le split est le garde-fou central du projet : c'est lui qui garantit qu'aucune
information postérieure à la période d'entraînement ne parvient au modèle.
"""
import re
import sys
from pathlib import Path

import psycopg
import pytest
import yaml

sys.path.insert(0, "src")
from tvfed.db import DSN  # noqa: E402
from tvfed.matrices import SPLITS  # noqa: E402
from tvfed.paths import RACINE  # noqa: E402


@pytest.fixture(scope="module")
def conn():
    with psycopg.connect(DSN) as c:
        yield c


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load((RACINE / "config" / "perimetre.yaml").read_text(encoding="utf-8"))


def q(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# ── cohérence des 3 sources de vérité ────────────────────────────────

def test_matrices_suit_la_config(cfg):
    """src/tvfed/matrices.py doit refléter config/perimetre.yaml."""
    for nom, (debut, fin) in cfg["split"].items():
        assert SPLITS[nom]["debut"] == debut, f"{nom} : début désynchronisé"
        assert SPLITS[nom]["fin"] == fin, f"{nom} : fin désynchronisée"


def test_vue_sql_suit_la_config(cfg):
    """sql/31_split.sql code les bornes en dur — elles doivent correspondre.

    La vue teste `annee <= X` ; X doit être l'année de fin de train, puis
    celle de fin de val.
    """
    sql = (RACINE / "sql" / "31_split.sql").read_text(encoding="utf-8")
    annees = [int(a) for a in re.findall(r"annee\s*<=\s*(\d{4})", sql)]
    attendu = [int(cfg["split"]["train"][1][:4]), int(cfg["split"]["val"][1][:4])]
    assert annees == attendu, (
        f"sql/31_split.sql utilise {annees}, la config dit {attendu} — "
        "les deux ont divergé"
    )


# ── étanchéité du split en base ──────────────────────────────────────

def test_bornes_contigues_sans_chevauchement(conn, cfg):
    r = {s: (d, f) for s, d, f in q(conn, """
        SELECT split, min(date), max(date) FROM grille_split GROUP BY split
    """)}
    for nom, (debut, fin) in cfg["split"].items():
        assert str(r[nom][0]) == debut, f"{nom} commence le {r[nom][0]}, attendu {debut}"
        assert str(r[nom][1]) == fin, f"{nom} finit le {r[nom][1]}, attendu {fin}"

    # aucun trou : la fin d'une partition touche le début de la suivante
    assert r["train"][1] < r["val"][0] <= r["train"][1] + __import__("datetime").timedelta(days=1)
    assert r["val"][1] < r["test"][0] <= r["val"][1] + __import__("datetime").timedelta(days=1)


def test_partition_couvre_toute_la_grille(conn):
    """Aucune ligne ne doit échapper au split."""
    total, = q(conn, "SELECT count(*) FROM grille")[0]
    somme, = q(conn, "SELECT count(*) FROM grille_split")[0]
    assert total == somme, f"{total - somme:,} lignes hors split"


def test_val_et_test_integraux(conn, cfg):
    """Ils ne doivent JAMAIS être échantillonnés.

    C'est la validation intégrale qui corrige le décalage de prior introduit
    par le downsampling du train (facteur ~×500). Downsamplée, elle ne
    jouerait plus ce rôle et la calibration serait fausse.
    """
    n_com, = q(conn, "SELECT count(*) FROM ref_commune WHERE in_perimetre")[0]
    for nom in ("val", "test"):
        debut, fin = cfg["split"][nom]
        n_jours, = q(conn,
            "SELECT count(*) FROM ref_calendrier WHERE date BETWEEN %s AND %s",
            (debut, fin))[0]
        n, = q(conn, "SELECT count(*) FROM grille WHERE date BETWEEN %s AND %s",
               (debut, fin))[0]
        assert n == n_com * n_jours, f"{nom} : {n:,} lignes, attendu {n_com * n_jours:,}"


# ── la matrice exportée ──────────────────────────────────────────────

@pytest.mark.skipif(not Path("data/processed/train.parquet").exists(),
                    reason="matrice pas encore générée")
def test_train_parquet_sans_date_future(cfg):
    """Le fichier livré au modèle ne doit contenir QUE des dates de train."""
    import pandas as pd

    df = pd.read_parquet("data/processed/train.parquet")
    d = pd.to_datetime(df.date)
    assert str(d.min().date()) >= cfg["split"]["train"][0]
    assert str(d.max().date()) <= cfg["split"]["train"][1], (
        f"train.parquet contient des dates jusqu'au {d.max().date()} — fuite"
    )
    assert set(df.split.unique()) == {"train"}
