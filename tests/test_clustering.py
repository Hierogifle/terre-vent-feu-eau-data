"""Garde-fous du clustering et du lissage bayésien.

Le lissage agrège `y`. C'est donc la feature la plus exposée du projet : une
erreur ici ne lève rien, elle produit un excellent score de train et un modèle
qui s'effondre en production. Ces tests sont le seul filet.
"""
import re
import sys


import pandas as pd
import pytest

sys.path.insert(0, "src")
from tvfed import clustering  # noqa: E402


# ════════════════════════════════════════════════════════════════════════
#  jeu de données synthétique — pas de base, les tests doivent tourner seuls
# ════════════════════════════════════════════════════════════════════════
@pytest.fixture
def sin():
    """3 communes × 4 années. La commune A brûle, B une seule fois, C jamais."""
    lignes = []
    for an in range(2006, 2010):
        lignes += [
            {"code_insee": "A0001", "an": an, "jours": 1000, "feux": 10},
            {"code_insee": "B0002", "an": an, "jours": 1000,
             "feux": 4 if an == 2007 else 0},
            {"code_insee": "C0003", "an": an, "jours": 1000, "feux": 0},
        ]
    return pd.DataFrame(lignes)


@pytest.fixture
def clusters():
    # A et B dans le même cluster, C seule dans le sien
    return pd.Series({"A0001": 0, "B0002": 0, "C0003": 1}, name="cluster_id")


@pytest.fixture
def taux(sin, clusters):
    return clustering.lisser(sin, clusters)


# ════════════════════════════════════════════════════════════════════════
#  1. l'exclusion de l'année — la fuite du target encoding
# ════════════════════════════════════════════════════════════════════════
def test_annee_exclue_retire_bien_ses_feux(taux, sin):
    """Le taux de B en excluant 2007 doit ignorer ses 4 feux de 2007.

    B n'a brûlé qu'en 2007. Son taux « hors 2007 » doit donc être STRICTEMENT
    inférieur à son taux toutes années confondues. Si les deux sont égaux,
    l'exclusion ne fait rien et chaque ligne de train contribue à sa propre
    feature.
    """
    hors = taux[(taux.code_insee == "B0002") & (taux.an_exclue == 2007)]
    tout = taux[(taux.code_insee == "B0002") & (taux.an_exclue == 0)]
    assert hors.taux_commune_lisse.iloc[0] < tout.taux_commune_lisse.iloc[0]


def test_annee_sans_feu_ne_change_presque_rien(taux):
    """Exclure 2006 (où B n'a pas brûlé) ne doit retirer aucun feu à B."""
    a = taux[(taux.code_insee == "B0002") & (taux.an_exclue == 2006)]
    b = taux[(taux.code_insee == "B0002") & (taux.an_exclue == 2008)]
    assert a.taux_commune_lisse.iloc[0] == pytest.approx(
        b.taux_commune_lisse.iloc[0], rel=1e-9)


def test_une_ligne_par_annee_de_train_plus_la_reference(taux):
    """14 années de train + la ligne 0 servant à val et test."""
    assert set(taux.an_exclue.unique()) == {0, *range(2006, 2010)}


def test_exposition_constante_entre_train_et_val(taux):
    """L'exclusion d'une année ne doit PAS décaler l'échelle de la feature.

    Sans remise à exposition constante, retirer une année fait tomber le
    dénominateur (5 113 jours → 4 748) et la même commune reçoit un taux ~4 %
    plus élevé sur une ligne de train que sur une ligne de val. Pire :
    `ratio_commune_cluster` prend deux plages DISJOINTES, 0,2964 en train
    contre 0,2812 en val, et tout seuil appris dessus devient faux au moment
    de prédire. C'est un décalage train/service, invisible dans les métriques
    de train — donc exactement le genre d'erreur qu'un test doit attraper.

    Ici C n'a jamais brûlé : quelle que soit l'année retirée, son taux ne peut
    pas bouger, puisqu'on ne lui retire aucun feu.
    """
    c = taux[taux.code_insee == "C0003"].set_index("an_exclue")
    for col in ("taux_commune_lisse", "ratio_commune_cluster"):
        assert c[col].nunique() == 1, (
            f"{col} varie selon l'année exclue pour une commune sans feu : "
            f"{sorted(c[col].unique())}"
        )


def test_pas_de_decalage_systematique_train_val(taux):
    """La ligne val ne doit pas être décalée hors de la plage des lignes train.

    A brûle 10 fois par an, chaque année : son taux ne peut pas bouger
    beaucoup selon l'année retirée. Un résidu subsiste — retirer 2007 fait
    sortir les 4 feux de B, donc le prior du cluster baisse — mais il doit
    rester d'un ordre de grandeur en dessous des ~4 % que corrigeait la
    remise à exposition constante.

    Et surtout : la valeur de val (`an_exclue = 0`) doit tomber DANS la plage
    des valeurs de train. Si elle est systématiquement au-dessus ou en
    dessous, le modèle apprend une échelle et en applique une autre.
    """
    a = taux[taux.code_insee == "A0001"].set_index("an_exclue").taux_commune_lisse
    val, train = a.loc[0], a.drop(0)

    etendue = (train.max() - train.min()) / train.mean()
    assert etendue < 0.01, f"le taux varie de {etendue:.1%} selon l'année exclue"
    assert train.min() <= val <= train.max(), (
        f"val ({val:.6f}) hors de la plage de train "
        f"[{train.min():.6f}, {train.max():.6f}]"
    )


# ════════════════════════════════════════════════════════════════════════
#  2. le lissage fait ce qu'il prétend
# ════════════════════════════════════════════════════════════════════════
def test_commune_muette_recoit_le_taux_de_son_cluster(taux):
    """C n'a jamais brûlé : son taux empirique est 0, son taux lissé non.

    C'est toute la raison d'être de l'étape. Sans lissage, les 80 % de
    communes qui n'ont jamais brûlé sont indiscernables.
    """
    c = taux[(taux.code_insee == "C0003") & (taux.an_exclue == 0)]
    assert c.taux_commune_lisse.iloc[0] > 0


def test_le_lissage_va_dans_le_bon_sens(taux):
    """Une commune qui brûle beaucoup reste au-dessus d'une commune muette."""
    ref = taux[taux.an_exclue == 0].set_index("code_insee")
    assert (ref.loc["A0001", "taux_commune_lisse"]
            > ref.loc["B0002", "taux_commune_lisse"]
            > ref.loc["C0003", "taux_commune_lisse"])


def test_le_taux_lisse_reste_entre_empirique_et_prior(taux, sin, clusters):
    """Le lissage TIRE vers le cluster, il ne dépasse jamais.

    Un taux lissé hors de l'intervalle [empirique, prior] signalerait une
    erreur d'arithmétique dans la pondération.
    """
    ref = taux[taux.an_exclue == 0].set_index("code_insee")
    emp = sin.groupby("code_insee").apply(
        lambda g: g.feux.sum() / g.jours.sum(), include_groups=False)
    for code in ref.index:
        lo, hi = sorted([emp[code], ref.loc[code, "taux_cluster_lisse"]])
        assert lo <= ref.loc[code, "taux_commune_lisse"] <= hi


# ════════════════════════════════════════════════════════════════════════
#  3. l'application aux matrices
# ════════════════════════════════════════════════════════════════════════
def test_ligne_de_train_recoit_son_annee_exclue(taux):
    """Une ligne de train de 2007 doit lire les taux calculés SANS 2007."""
    df = pd.DataFrame({"code_insee": ["B0002"], "date": ["2007-06-15"],
                       "split": ["train"]})
    out = clustering.appliquer(df, taux)
    attendu = taux[(taux.code_insee == "B0002")
                   & (taux.an_exclue == 2007)].taux_commune_lisse.iloc[0]
    assert out.taux_commune_lisse.iloc[0] == pytest.approx(attendu)


def test_ligne_de_val_utilise_tout_le_train(taux):
    """Une ligne de val n'appartient à aucune année de train : rien à exclure."""
    df = pd.DataFrame({"code_insee": ["B0002"], "date": ["2021-06-15"],
                       "split": ["val"]})
    out = clustering.appliquer(df, taux)
    attendu = taux[(taux.code_insee == "B0002")
                   & (taux.an_exclue == 0)].taux_commune_lisse.iloc[0]
    assert out.taux_commune_lisse.iloc[0] == pytest.approx(attendu)


def test_commune_absente_echoue_bruyamment(taux):
    """Une commune sans taux doit lever, pas produire un NaN silencieux.

    Un NaN serait ensuite imputé par la médiane et la commune passerait pour
    moyenne — une erreur invisible dans toutes les métriques.
    """
    df = pd.DataFrame({"code_insee": ["Z9999"], "date": ["2021-06-15"],
                       "split": ["val"]})
    with pytest.raises(ValueError, match="sans taux lissé"):
        clustering.appliquer(df, taux)


# ════════════════════════════════════════════════════════════════════════
#  4. le profil de clustering ne doit rien savoir du feu
# ════════════════════════════════════════════════════════════════════════
def test_le_profil_ignore_la_cible():
    """Aucune colonne du profil ne peut dériver de `y`.

    Un clustering construit sur la sinistralité serait circulaire : on
    prédirait le feu avec des groupes définis par le feu.
    """
    sql = (clustering.RACINE / "sql" / "60_profil_commune.sql").read_text(
        encoding="utf-8")
    corps = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    # bornes de mot obligatoires : sans elles, `grille` matcherait
    # `grille_densite`, qui est la typologie INSEE et n'a rien à voir
    for interdit in ("grille", "fait_feu", "nb_feux", "surface_m2"):
        assert not re.search(rf"\b{interdit}\b", corps), \
            f"le profil lit {interdit!r}"


def test_la_sinistralite_ne_lit_que_le_train():
    """Les bornes du train sont écrites en dur dans la requête."""
    sql = (clustering.RACINE / "sql" / "61_sinistralite.sql").read_text(
        encoding="utf-8")
    assert "'2006-01-01'" in sql and "'2019-12-31'" in sql
    assert "2020" not in sql and "2023" not in sql


def test_climatologie_fwi_bornee_au_train():
    """La climatologie du profil ne doit pas lire la météo de val ni de test."""
    sql = (clustering.RACINE / "sql" / "60_profil_commune.sql").read_text(
        encoding="utf-8")
    corps = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    assert "'2019-12-31'" in corps
