"""CEMS Fire danger indices — les 8 indices feu journaliers, grille 0,25°.

Produit officiel du Copernicus Emergency Management Service, celui qui
alimente l'EFFIS. On le prend tel quel plutôt que de recalculer le FWI :
il est déjà interpolé à midi heure locale, ce qui est le vrai problème.

⚠️ Les noms de variables dans le NetCDF ne sont PAS ceux de la requête API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..paths import CEMS_DIR

# nom NetCDF -> nom de colonne SQL
INDICES = {
    "fwinx": "fwi",      # Fire Weather Index — l'indice de référence EFFIS
    "ffmcode": "ffmc",   # litière fine, mémoire ~1 j
    "dufmcode": "dmc",   # couches lâches, ~12 j
    "drtcode": "dc",     # couches profondes, ~52 j
    "fbupinx": "bui",    # f(DMC, DC)
    "infsinx": "isi",    # f(FFMC, vent) — seul porteur du vent
    "kbdi": "kbdi",      # Keetch-Byram, sécheresse du sol
    "ercnfdr": "erc",    # Energy Release Component
}


def annees_disponibles(dossier=None) -> list[int]:
    return sorted(int(f.stem.split("_")[-1]) for f in (dossier or CEMS_DIR).glob("*.nc"))


def charger_annee(annee: int, cell_ids: set[int] | None = None, dossier=None) -> pd.DataFrame:
    """Aplatit un NetCDF annuel en table (cell_id, date, 8 indices).

    `cell_ids` restreint aux mailles réellement utilisées par des communes
    (1 131 sur 2 360) : inutile de stocker l'océan et l'étranger.
    """
    fichier = (dossier or CEMS_DIR) / f"cems_fire_{annee}.nc"
    with xr.open_dataset(fichier) as ds:
        dates = pd.to_datetime(ds.valid_time.values)
        n_lat, n_lon = ds.sizes["latitude"], ds.sizes["longitude"]

        # cell_id = ilat * n_lon + ilon, cohérent avec geo.table_mailles()
        cid = np.arange(n_lat * n_lon)
        bloc = {v: ds[v].values.reshape(len(dates), -1) for v in INDICES}

    df = pd.DataFrame(
        {
            "cell_id": np.tile(cid, len(dates)),
            "date": np.repeat(dates.date, len(cid)),
            **{col: bloc[v].ravel() for v, col in INDICES.items()},
        }
    )

    if cell_ids is not None:
        df = df[df.cell_id.isin(cell_ids)]

    # Les NaN restants sont de la MER, pas des trous de données : on les
    # retire au lieu de les imputer (vérifié : 0 cellule intermittente).
    return df.dropna(subset=["fwi"]).reset_index(drop=True)
