"""Code officiel géographique — résolution des fusions de communes.

La BDIFF porte le code INSEE en vigueur **au moment du feu**. Depuis 2006,
des centaines de communes ont fusionné dans des « communes nouvelles » et
leur code a disparu du COG. Sans remappage, ces feux sont orphelins.

⚠️ Ne JAMAIS deviner par le nom. Mesuré sur ce projet : « Chirac » (48049,
Lozère) ressemble à « Saint-Bonnet-de-Chirac » (48138) mais a en réalité
fusionné dans « Bourgs sur Colagne » (48099). Un feu attribué à la mauvaise
commune corrompt y, le voisinage spatial ET les features CORINE.

Source : https://www.insee.fr/fr/information/8740222
"""
from __future__ import annotations

import pandas as pd

from ..paths import MVT_COMMUNE_CSV

# Modalités qui déplacent réellement une commune vers un autre code.
# 10 (changement de nom) et 70/71/72 (réorganisations internes) sont exclus :
# le code ne change pas.
MOD_FUSION = {
    "31",  # fusion simple
    "32",  # création de commune nouvelle  ← cas majoritaire (2016-2019)
    "33",  # fusion-association
    "34",  # transformation fusion-association → fusion simple
    "35",  # suppression de commune déléguée → absorbée par la commune de rattachement
    "41",  # changement de code (changement de département)
    "50",  # changement de code (transfert de chef-lieu)
}

# 21 = rétablissement : une commune se scinde en plusieurs. Un feu historique
# ne peut PAS être attribué à une seule des filles → ambigu, jamais résolu ici.
MOD_AMBIGU = {"21"}

MAX_SAUTS = 10  # garde anti-boucle : une commune peut fusionner plusieurs fois


def charger_mouvements(chemin=None) -> pd.DataFrame:
    """Lit v_mvt_commune_2026.csv (séparateur virgule, UTF-8, tout en texte)."""
    return pd.read_csv(chemin or MVT_COMMUNE_CSV, dtype=str).fillna("")


def table_passage(
    codes_actuels: set[str], mvt: pd.DataFrame | None = None
) -> dict[str, str]:
    """Construit `ancien_code -> code actuel`, résolu transitivement.

    `codes_actuels` = les codes INSEE du COG en vigueur (référentiel communes).
    Il est **obligatoire** et joue deux rôles de sécurité :

    1. Un code encore valide n'est JAMAIS déplacé. Sans ce garde-fou, 251 codes
       seraient corrompus : le fichier INSEE remonte à 1943 et certains codes ont
       été réattribués depuis (ex. 78143 → 91143 lors de l'éclatement de la
       Seine-et-Oise en 1968, alors que 78143 désigne une autre commune aujourd'hui).
    2. Une cible qui n'existe pas dans le COG actuel est rejetée plutôt que
       propagée (17 cas).

    Chaque fusion produit DEUX lignes dans le fichier INSEE :
        COM 48022 -> COMD 48022   (devient commune déléguée, même code)
        COM 48022 -> COM  48050   (la commune qui l'absorbe)   ← celle qu'on veut
    D'où le filtre sur TYPECOM_AP == 'COM'.
    """
    mvt = charger_mouvements() if mvt is None else mvt

    saut = (
        mvt[
            mvt.MOD.isin(MOD_FUSION)
            & (mvt.TYPECOM_AP == "COM")          # cible = vraie commune, pas déléguée
            & (mvt.COM_AV != mvt.COM_AP)          # le code change effectivement
        ]
        .sort_values("DATE_EFF")
        .drop_duplicates("COM_AV", keep="last")   # le mouvement le plus récent gagne
        .set_index("COM_AV")["COM_AP"]
        .to_dict()
    )

    # résolution transitive : 48022 -> 48050, et si 48050 avait fusionné ensuite…
    passage: dict[str, str] = {}
    for depart in saut:
        if depart in codes_actuels:
            continue                              # garde-fou 1 : code encore vivant
        courant, vus = depart, {depart}
        for _ in range(MAX_SAUTS):
            suivant = saut.get(courant)
            if suivant is None or suivant in vus:
                break
            vus.add(suivant)
            courant = suivant
            if courant in codes_actuels:
                break                             # arrivé, inutile d'aller plus loin
        if courant != depart and courant in codes_actuels:   # garde-fou 2
            passage[depart] = courant
    return passage


def codes_ambigus(mvt: pd.DataFrame | None = None) -> set[str]:
    """Codes issus d'une scission — plusieurs cibles possibles, non résolvables."""
    mvt = charger_mouvements() if mvt is None else mvt
    return set(mvt.loc[mvt.MOD.isin(MOD_AMBIGU), "COM_AV"])


def remapper(codes: pd.Series, passage: dict[str, str]) -> pd.Series:
    """Applique la table de passage à une colonne de codes INSEE."""
    return codes.map(lambda c: passage.get(c, c))
