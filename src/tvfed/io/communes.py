"""Référentiel communal — communes-france-2026.csv (COG 2026)."""
from __future__ import annotations

import pandas as pd

from ..paths import COMMUNES_CSV

# ⚠️ dep_code / reg_code / code_insee DOIVENT être lus en texte :
# '2A' et '2B' font échouer le parsing numérique, et les départements à
# un chiffre perdraient leur zéro initial ('01' -> 1).
DTYPES = {"code_insee": str, "dep_code": str, "reg_code": str}

COLONNES = [
    "code_insee", "nom_standard", "dep_code", "reg_code",
    "latitude_centre", "longitude_centre",
    "superficie_km2", "population", "densite",
    "altitude_moyenne", "altitude_minimale", "altitude_maximale",
    "grille_densite",
]


def charger(chemin=None, metropole_seule: bool = True) -> pd.DataFrame:
    """Retourne le référentiel, colonnes renommées pour coller au schéma SQL."""
    df = pd.read_csv(chemin or COMMUNES_CSV, usecols=COLONNES, dtype=DTYPES)

    if metropole_seule:
        # DOM : hors bbox météo européenne et absents de CORINE
        df = df[~df.dep_code.str.startswith(("97", "98"), na=False)]

    # 5 communes sans coordonnées — inutilisables pour le rattachement météo
    df = df.dropna(subset=["latitude_centre", "longitude_centre"])

    df = df.rename(
        columns={
            "nom_standard": "nom",
            "latitude_centre": "lat",
            "longitude_centre": "lon",
            "altitude_moyenne": "altitude_moy",
            "altitude_minimale": "altitude_min",
            "altitude_maximale": "altitude_max",
        }
    ).reset_index(drop=True)

    # Le CSV donne les altitudes et la population en flottants ('242.0').
    # Int64 (nullable) évite à la fois le rejet par PostgreSQL sur les colonnes
    # smallint/integer et la perte des valeurs manquantes.
    for c in ("population", "altitude_moy", "altitude_min", "altitude_max", "grille_densite"):
        df[c] = df[c].round().astype("Int64")

    # ⚠️ 162 communes (0,47 %) ont des altitudes incohérentes à la source :
    # Sari-d'Orcino (2A) annonce un minimum de 9 589 m, plusieurs communes de
    # montagne ont un maximum à 0. Le triplet entier est donc douteux → NULL,
    # ce qui est honnête, plutôt qu'une amplitude négative qui polluerait
    # silencieusement le modèle.
    incoherent = df.altitude_max < df.altitude_min
    df.loc[incoherent, ["altitude_moy", "altitude_min", "altitude_max"]] = pd.NA

    return df
