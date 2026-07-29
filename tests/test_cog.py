"""Résolution des fusions de communes — les garde-fous, figés.

Ces tests protègent contre deux erreurs silencieuses qui corrompraient
la géolocalisation de tous les feux historiques.
"""
import sys

import pandas as pd
import pytest

sys.path.insert(0, "src")
from tvfed.io.cog import codes_ambigus, remapper, table_passage  # noqa: E402
from tvfed.paths import COMMUNES_CSV  # noqa: E402

# Ces tests lisent le référentiel INSEE, qui vit dans `data/` — suivi par DVC,
# donc absent d'un dépôt fraîchement cloné. On saute plutôt que d'échouer :
# une donnée non versionnée qui manque n'est pas une régression du code.
pytestmark = pytest.mark.skipif(
    not COMMUNES_CSV.exists(),
    reason="référentiel INSEE absent — `dvc pull` pour le récupérer")


@pytest.fixture(scope="module")
def codes_actuels():
    return set(pd.read_csv(COMMUNES_CSV, usecols=["code_insee"], dtype=str).code_insee)


@pytest.fixture(scope="module")
def passage(codes_actuels):
    return table_passage(codes_actuels)


# Cas vérifiés un par un dans le fichier INSEE officiel.
# Chirac est le cas critique : le rapprochement par nom proposait
# « Saint-Bonnet-de-Chirac » (48138), la réalité est « Bourgs sur Colagne ».
CAS_CONNUS = [
    ("48022", "48050"),  # Bédouès          -> Bédouès-Cocurès
    ("48040", "48027"),  # Chasseradès      -> Mont Lozère et Goulet
    ("48172", "48116"),  # St-Maurice-de-Ventalon -> Pont de Montvert
    ("30190", "30339"),  # ND-de-la-Rouvière -> Val-d'Aigoual
    ("07016", "07011"),  # Asperjoc         -> Vallées-d'Antraigues-Asperjoc
    ("48049", "48099"),  # Chirac           -> Bourgs sur Colagne  ⚠️ piège du nom
]


@pytest.mark.parametrize("ancien,attendu", CAS_CONNUS)
def test_fusions_connues(passage, ancien, attendu):
    assert passage.get(ancien) == attendu


def test_aucun_code_valide_deplace(passage, codes_actuels):
    """Le fichier INSEE remonte à 1943 et des codes ont été réattribués.

    Sans ce garde-fou, 251 codes encore vivants seraient déplacés — par ex.
    78143 -> 91143 (éclatement de la Seine-et-Oise, 1968), alors que 78143
    désigne une autre commune aujourd'hui.
    """
    fautifs = {a: b for a, b in passage.items() if a in codes_actuels}
    assert not fautifs, f"{len(fautifs)} codes encore valides seraient déplacés"


def test_toutes_les_cibles_existent(passage, codes_actuels):
    orphelines = {a: b for a, b in passage.items() if b not in codes_actuels}
    assert not orphelines, f"{len(orphelines)} cibles absentes du COG 2026"


def test_remap_idempotent(passage, codes_actuels):
    """Remapper deux fois doit donner le même résultat qu'une fois."""
    s = pd.Series(sorted(passage)[:500])
    une = remapper(s, passage)
    deux = remapper(une, passage)
    assert une.equals(deux)


def test_scissions_non_resolues(passage):
    """MOD=21 : une commune se scinde — pas de cible unique, on ne devine pas."""
    assert codes_ambigus(), "les scissions doivent être identifiées"
