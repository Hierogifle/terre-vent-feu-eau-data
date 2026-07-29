"""Géométrie : rattachement commune → maille météo, voisinage, distance à la côte.

Le rattachement commune → maille est LA charnière du projet : c'est lui qui
permet à chaque commune d'aller chercher ses indices feu. Code repris de
notebook/data-copernicus.ipynb (cellule 4), où il a été vérifié : les 34 734
communes métropolitaines tombent toutes sur de la terre, aucune sur la mer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from .paths import CEMS_DIR

R_TERRE_KM = 6371.0


def grille_cems(fichier=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne (lats, lons, masque_mer) depuis un NetCDF CEMS quelconque.

    Le masque de mer est stable sur les 53 années : une cellule sans valeur
    est de l'océan, pas un trou de données (vérifié dans le notebook CEMS).
    """
    fichier = fichier or sorted(CEMS_DIR.glob("*.nc"))[0]
    with xr.open_dataset(fichier) as ds:
        return (
            ds.latitude.values,
            ds.longitude.values,
            ds.fwinx.isnull().all(dim="valid_time").values,
        )


def table_mailles(lats, lons, mer) -> pd.DataFrame:
    """Aplatit la grille 2D en table (cell_id, ilat, ilon, lat, lon, est_mer)."""
    ilat, ilon = np.meshgrid(np.arange(len(lats)), np.arange(len(lons)), indexing="ij")
    return pd.DataFrame(
        {
            "cell_id": np.arange(mer.size),
            "ilat": ilat.ravel(),
            "ilon": ilon.ravel(),
            "lat": lats[ilat.ravel()],
            "lon": lons[ilon.ravel()],
            "est_mer": mer.ravel(),
        }
    )


def rattacher_mailles(communes: pd.DataFrame, lats, lons) -> pd.Series:
    """cell_id de la maille 0,25° la plus proche du centroïde de chaque commune.

    Plus-proche-voisin sur chaque axe séparément : la grille est régulière,
    donc c'est exact et immédiat.
    """
    ilat = np.abs(communes.lat.values[:, None] - lats[None, :]).argmin(axis=1)
    ilon = np.abs(communes.lon.values[:, None] - lons[None, :]).argmin(axis=1)
    return pd.Series(ilat * len(lons) + ilon, index=communes.index, name="cell_id")


def _xyz(lat_deg, lon_deg) -> np.ndarray:
    """Coordonnées cartésiennes 3D — évite le repliement du méridien 180°
    et la distorsion en latitude d'un KDTree posé sur (lat, lon)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.c_[
        R_TERRE_KM * np.cos(lat) * np.cos(lon),
        R_TERRE_KM * np.cos(lat) * np.sin(lon),
        R_TERRE_KM * np.sin(lat),
    ]


def voisinage(communes: pd.DataFrame, rayon_km: float = 50.0) -> pd.DataFrame:
    """Toutes les paires de communes distantes de moins de `rayon_km`.

    ~19 M paires à 50 km sur la France entière, calculées en ~0,2 s.
    Retourne les DEUX sens (a→b et b→a) : les features de voisinage joignent
    sur `voisin`, il faut donc que chaque commune apparaisse comme cible.
    """
    xyz = _xyz(communes.lat.values, communes.lon.values)
    paires = cKDTree(xyz).query_pairs(rayon_km, output_type="ndarray")
    codes = communes.code_insee.values

    # distance de corde 3D -> distance de grand cercle
    d = np.linalg.norm(xyz[paires[:, 0]] - xyz[paires[:, 1]], axis=1)
    dist_m = 2 * R_TERRE_KM * np.arcsin(np.clip(d / (2 * R_TERRE_KM), 0, 1)) * 1000

    return pd.DataFrame(
        {
            "code_insee": np.r_[codes[paires[:, 0]], codes[paires[:, 1]]],
            "voisin": np.r_[codes[paires[:, 1]], codes[paires[:, 0]]],
            "dist_m": np.r_[dist_m, dist_m].astype("float32"),
        }
    )
