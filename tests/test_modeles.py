"""La préparation des features — le dernier rempart avant le modèle.

Deux colonnes de la matrice sont la cible déguisée. Les laisser entrer
donnerait une PR-AUC proche de 1,00 et un modèle sans aucune valeur en
production. Ces tests documentent le problème et garantissent l'exclusion.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, "src")
from tvfed.modeles import CATEGORIELLES, CIBLE, FUITE, IDENTIFIANTS, Preparation  # noqa: E402

PARQUET = Path("data/processed/train.parquet")
pytestmark = pytest.mark.skipif(not PARQUET.exists(), reason="matrice pas générée")


@pytest.fixture(scope="module")
def train():
    return pd.read_parquet(PARQUET)


def test_les_colonnes_de_fuite_sont_bien_des_fuites(train):
    """Documente POURQUOI ces colonnes sont interdites.

    Si ce test échoue un jour, c'est que la sémantique de la grille a changé —
    et il faudra revoir la liste FUITE en conséquence.
    """
    assert ((train.nb_feux > 0) == train[CIBLE]).all(), \
        "nb_feux > 0 devrait être exactement équivalent à y"
    assert ((train.surface_m2 > 0) == train[CIBLE]).all(), \
        "surface_m2 > 0 devrait être exactement équivalent à y"


def test_aucune_fuite_dans_les_features(train):
    prep = Preparation().fit(train)
    interdites = set(FUITE + IDENTIFIANTS + CATEGORIELLES + [CIBLE])
    fautives = set(prep.colonnes_) & interdites
    assert not fautives, f"colonnes interdites dans les features : {fautives}"


def test_fit_refuse_les_donnees_hors_train(train):
    """L'imputation a un .fit() : elle ne doit voir QUE le train."""
    faux = train.head(100).copy()
    faux["split"] = "val"
    with pytest.raises(AssertionError, match="fuite"):
        Preparation().fit(faux)


def test_transform_ne_laisse_aucun_nan(train):
    """Les modèles à base d'arbres de sklearn n'acceptent pas les NaN."""
    prep = Preparation().fit(train)
    X = prep.transform(train)
    assert not pd.isna(X).any(), "des NaN subsistent après imputation"
    assert X.shape[1] == len(prep.colonnes_)


def test_transform_est_stable_sur_lordre_des_colonnes(train):
    """Un bloc de validation peut arriver avec les colonnes dans un autre ordre.

    `reindex` garantit que la colonne 12 de l'entraînement reste la colonne 12
    à la prédiction — sinon le modèle lit les mauvaises valeurs, silencieusement.
    """
    prep = Preparation().fit(train)
    attendu = prep.transform(train.head(1000))
    melange = train.head(1000)[list(train.columns)[::-1]]
    assert (prep.transform(melange) == attendu).all()
