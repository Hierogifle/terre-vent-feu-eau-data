"""Localisation des données — un seul endroit qui sait où sont les fichiers."""
from pathlib import Path


def racine() -> Path:
    """Remonte l'arborescence jusqu'à trouver la racine du projet.

    Marche depuis notebook/, src/, ou la racine — c'est ce qui permet aux
    notebooks et aux scripts de partager le même code sans chemin en dur.
    """
    for p in (Path.cwd(), *Path.cwd().parents):
        if (p / "data").is_dir() and (p / "notebook").is_dir():
            return p
    raise FileNotFoundError(
        f"Racine du projet introuvable depuis {Path.cwd()} "
        "(on cherche un dossier contenant à la fois data/ et notebook/)"
    )


RACINE = racine()
DATA = RACINE / "data"

BDIFF_DIR = DATA / "BDIFF"
CEMS_DIR = DATA / "copernicus" / "raw" / "cems"
CORINE_CSV = DATA / "corine" / "clc_etat_com_n3.csv"
COMMUNES_CSV = DATA / "ville-france" / "communes-france-2026.csv"
MVT_COMMUNE_CSV = DATA / "insee-cog" / "v_mvt_commune_2026.csv"

INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
APP_DIR = DATA / "app"
