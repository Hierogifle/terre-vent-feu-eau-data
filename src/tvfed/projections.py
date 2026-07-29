"""Étape 16 — la climatologie du feu en 2050, et la migration des territoires.

    python -m tvfed.projections --telecharger
    python -m tvfed.projections --migration

────────────────────────────────────────────────────────────────────────────
LA DONNÉE
────────────────────────────────────────────────────────────────────────────
« Fire danger indicators for Europe from 1970 to 2098 derived from climate
projections » — dataset `sis-tourism-fire-danger-indicators` du CDS, calculé
par le modèle GEFF, **le même que les données historiques du projet**. C'est
ce qui rend les deux comparables.

    experiment    historical · rcp2_6 · rcp4_5 · rcp8_5
    product_type  multi_model_mean_case  (moyenne de 6 GCM EURO-CORDEX)
    variable      seasonal_fire_weather_index
    résolution    0,11°  (contre 0,25° pour l'historique CEMS)

────────────────────────────────────────────────────────────────────────────
⚠️ LA CORRECTION DE BIAIS — LA MÉTHODE DES DELTAS
────────────────────────────────────────────────────────────────────────────
Le FWI d'un modèle climatique régional n'est PAS celui d'une réanalyse : il
porte le biais propre du modèle. Nourrir le modèle de feu avec du FWI de RCM
brut donnerait des résultats faux avec beaucoup d'aplomb.

On n'utilise donc jamais la valeur projetée telle quelle, mais **l'écart entre
le futur et l'historique DU MÊME MODÈLE** :

    FWI_2050_corrigé  =  FWI_observé(2006-2019)  +  Δ
    avec  Δ = FWI_rcm(2046-2050) − FWI_rcm(historique)

Le biais du RCM, présent des deux côtés de la soustraction, s'annule. On ne
retient du modèle climatique que ce qu'il sait faire — la VARIATION — et on
garde des observations pour le niveau.

C'est plus simple qu'un quantile mapping et plus robuste sur peu d'années.

────────────────────────────────────────────────────────────────────────────
CE QU'ON EN FAIT
────────────────────────────────────────────────────────────────────────────
Le modèle de projection retenu (C, physique pur) n'utilise PAS le cluster :
la migration n'est donc pas un mécanisme de prédiction, c'est un **outil de
lecture**. Elle répond à une question qu'un jury comprend immédiatement :

    « en 2050, à quel territoire d'aujourd'hui le Morbihan ressemblera-t-il ? »

Une commune dont la climatologie 2050 rejoint celle du Var actuel bascule
dans le cluster du Var. C'est la substitution espace-temps, rendue visible.
"""
from __future__ import annotations

import argparse
import zipfile

import numpy as np
import pandas as pd

from .paths import RACINE

DOSSIER = RACINE / "data" / "projections"
JEU = "sis-tourism-fire-danger-indicators"

# ⚠️ UNE CLIMATOLOGIE SUR 5 ANS EST DU BRUIT.
# La variabilité interannuelle du FWI est énorme — sur les données observées
# du projet, 4,34 en 2021 contre 6,90 en 2020, un facteur 1,6. Un premier
# essai sur la seule période 2046-2050 donnait RCP8.5 EN DESSOUS de RCP4.5,
# ce qui n'a aucun sens physique : c'était l'échantillon, pas le climat.
#
# On prend donc 15 saisons de part et d'autre, centrées sur 2048, contre
# 20 pour l'historique. Le delta devient une différence de climatologies,
# pas de deux tirages.
# ⚠️ TROIS HORIZONS, PAS UN — ET C'EST CE QUI REND LES SCÉNARIOS LISIBLES.
# À 2050 les RCP ne diffèrent que de 6 % : le CO₂ déjà émis détermine les
# vingt-cinq prochaines années quoi qu'on fasse. L'éventail ne s'ouvre
# qu'après, et le jeu de données va jusqu'en 2098.
#
# Trois fenêtres de 15 saisons chacune, pour capter la COURBURE des
# trajectoires — RCP4.5 s'aplatit, RCP8.5 accélère. Deux points seuls
# forceraient une droite et effaceraient précisément ce qui distingue les
# scénarios.
H_PROCHE = ["2031_2035", "2036_2040", "2041_2045"]     # centré 2038
H_MILIEU = ["2051_2055", "2056_2060", "2061_2065"]     # centré 2058
H_FIN = ["2086_2090", "2091_2095", "2096_2098"]        # centré 2092
HORIZONS = {"proche": (H_PROCHE, 2038), "milieu": (H_MILIEU, 2058),
            "fin": (H_FIN, 2092)}
SCENARIOS = ["rcp2_6", "rcp4_5", "rcp8_5"]

DEMANDES = {"historical": {"experiment": "historical",
                           "period": ["1986_1990", "1991_1995",
                                      "1996_2000", "2001_2005"]}}
for _s in SCENARIOS:
    for _h, (_p, _) in HORIZONS.items():
        DEMANDES[f"{_s}__{_h}"] = {"experiment": _s, "period": _p}


# ⚠️ DEUX PORTAILS DIFFÉRENTS, ET C'EST UN PIÈGE.
# Les données CEMS historiques du projet viennent de l'Early Warning Data
# Store (ewds.climate.copernicus.eu) — c'est ce que contient `.cdsapirc`.
# Les PROJECTIONS sont sur le Climate Data Store (cds.climate.copernicus.eu).
# Adresses distinctes, même identifiant ECMWF depuis l'unification des comptes.
# On force donc l'URL, en réutilisant la clé du fichier existant.
URL_CDS = "https://cds.climate.copernicus.eu/api"


def _client():
    import cdsapi

    from pathlib import Path

    rc = Path.home() / ".cdsapirc"
    cle = None
    if rc.exists():
        for l in rc.read_text(encoding="utf-8").splitlines():
            if l.strip().startswith("key:"):
                cle = l.split(":", 1)[1].strip()
    if not cle:
        raise RuntimeError("aucune clé trouvée dans ~/.cdsapirc")
    return cdsapi.Client(url=URL_CDS, key=cle)


def telecharger() -> None:
    """Récupère les indicateurs saisonniers, un fichier par scénario."""
    DOSSIER.mkdir(parents=True, exist_ok=True)
    c = _client()
    for nom, extra in DEMANDES.items():
        dest = DOSSIER / f"fwi_{nom}.zip"
        if dest.exists():
            print(f"  {dest.name} déjà présent")
            continue
        print(f"  {nom}…")
        c.retrieve(JEU, {
            "format": "zip",
            "time_aggregation": "seasonal_indicators",
            "product_type": "multi_model_mean_case",
            "variable": "seasonal_fire_weather_index",
            "version": "v2_0",
            **extra,
        }, str(dest))
        print(f"  ✅ {dest.name}  {dest.stat().st_size / 1e6:.1f} Mo")


def _ouvrir(nom: str):
    """Ouvre le NetCDF contenu dans l'archive du scénario."""
    import xarray as xr

    z = DOSSIER / f"fwi_{nom}.zip"
    if not z.exists():
        raise FileNotFoundError(
            f"{z.name} absent — lancer d'abord :\n"
            "    python -m tvfed.projections --telecharger")
    with zipfile.ZipFile(z) as a:
        ncs = [n for n in a.namelist() if n.endswith(".nc")]
        if not ncs:
            raise ValueError(f"aucun .nc dans {z.name} : {a.namelist()[:5]}")
        a.extractall(DOSSIER / nom)
    fichiers = sorted((DOSSIER / nom).rglob("*.nc"))
    return xr.open_mfdataset(fichiers, combine="by_coords") if len(fichiers) > 1 \
        else xr.open_dataset(fichiers[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--telecharger", action="store_true")
    ap.add_argument("--migration", action="store_true")
    args = ap.parse_args()

    if args.telecharger:
        telecharger()
    if args.migration:
        print("⚠️ étape suivante — nécessite les fichiers téléchargés")


if __name__ == "__main__":
    main()
