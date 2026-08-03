"""Garde-fous sur la comparaison de modèles.

Le bug qu'ils empêchent de revenir : `sql/50_matrice.sql` n'a pas d'`ORDER
BY`, donc l'ordre des lignes change d'une exécution à l'autre. Deux fichiers
de prédictions issus de deux exécutions ont la même taille et le même nombre
de feux, mais pas le même ordre — les comparer position par position donne
un écart faux sans lever la moindre erreur. La première comparaison LSTM ↔
XGBoost annonçait ainsi −97 % au lieu de −52 %.
"""
import sys

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

sys.path.insert(0, "src")
from tvfed.comparer import SOURCES, ApRapide  # noqa: E402
from tvfed.paths import PROCESSED  # noqa: E402

CLES = {"code_insee", "date"}


@pytest.mark.parametrize(
    "n,n_uniques,taux", [(5_000, 50, 0.02), (50_000, 500, 0.001),
                         (20_000, 20_000, 0.05)])
def test_ap_ponderee_reproduit_sklearn(n, n_uniques, taux):
    """L'AP par cumsum doit être identique à sklearn, ex æquo compris.

    Les ex æquo sont le cas qui piège : sklearn n'évalue la précision qu'aux
    seuils DISTINCTS. Avec 16,6 M de valeurs distinctes pour 38 M de lignes,
    ils sont massivement présents dans les vraies prédictions.
    """
    rng = np.random.default_rng(0)
    p = (rng.integers(0, n_uniques, n) / n_uniques).astype(np.float32)
    y = (rng.random(n) < taux).astype(np.int8)
    if y.sum() == 0:
        pytest.skip("aucun positif tiré")
    rapide = ApRapide(p, y)
    assert abs(rapide() - average_precision_score(y, p)) < 1e-12

    w = rng.integers(0, 4, n).astype(np.int32)
    assert abs(rapide(w)
               - average_precision_score(y, p, sample_weight=w)) < 1e-12


def test_ap_ponderee_equivaut_a_la_duplication():
    """Un poids de 2 doit valoir exactement deux copies de la ligne.

    C'est la propriété sur laquelle repose tout le bootstrap : une réplique
    n'est qu'un jeu de multiplicités entières.
    """
    rng = np.random.default_rng(1)
    p = (rng.integers(0, 100, 3_000) / 100).astype(np.float32)
    y = (rng.random(3_000) < 0.03).astype(np.int8)
    w = rng.integers(1, 4, 3_000).astype(np.int32)
    rep = np.repeat(np.arange(3_000), w)
    assert abs(ApRapide(p, y)(w) - ApRapide(p[rep], y[rep])()) < 1e-12


@pytest.mark.parametrize("nom", list(SOURCES))
def test_les_predictions_portent_leurs_cles(nom):
    """Un fichier de prédictions sans (code_insee, date) n'est comparable à rien."""
    import pyarrow.parquet as pq

    chemin = PROCESSED / SOURCES[nom][0]
    if not chemin.exists():
        pytest.skip(f"{chemin.name} pas encore produit")
    manquantes = CLES - set(pq.read_schema(chemin).names)
    assert not manquantes, (
        f"{chemin.name} n'a pas {manquantes} : l'ordre des lignes que renvoie "
        f"PostgreSQL n'étant garanti par rien, ce fichier ne peut pas être "
        f"comparé à un autre.")
