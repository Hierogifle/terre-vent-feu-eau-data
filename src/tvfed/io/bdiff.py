"""BDIFF — Base de Données sur les Incendies de Forêts en France.

Lecture directe des ZIP, sans décompression sur disque.

⚠️ Piège de format vérifié : la ligne d'en-tête n'est PAS à la même position
selon les fichiers (3ᵉ, 4ᵉ ou 7ᵉ). Un `skiprows` en dur casse sur 2 des 6
archives. On la détecte.
"""
from __future__ import annotations

import io
import re
import zipfile

import pandas as pd

from ..paths import BDIFF_DIR

COLONNES = {
    "Année": "annee",
    "Numéro": "numero",
    "Département": "dep_code",
    "Code INSEE": "code_insee_source",
    "Nom de la commune": "nom_commune",
    "Date de première alerte": "ts_alerte",
    "Surface parcourue (m2)": "surface_m2",
    "Surface forêt (m2)": "surface_foret_m2",
    "Surface maquis garrigues (m2)": "surface_maquis_m2",
    "Type de peuplement": "type_peuplement",
    "Nature": "nature",
}


def _lire_zip(zpath) -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(zpath) as z:
        nom = next(n for n in z.namelist() if n.lower().endswith("incendies.csv"))
        lignes = z.read(nom).decode("utf-8").splitlines()

    i_ent = next(i for i, l in enumerate(lignes) if l.startswith("Année;Numéro"))

    meta = {"fichier": zpath.name, "notes": []}
    for l in lignes[:i_ent]:
        if m := re.search(r"sélection\s*:\s*(\d+)", l):
            meta["annonce"] = int(m.group(1))
        # certains fichiers documentent leurs propres pertes de données
        if re.match(r'^"\d{4}\s*-\s*\d+\s*:', l):
            meta["notes"].append(l.replace('";"', " ").strip('"'))

    df = pd.read_csv(
        io.StringIO("\n".join(lignes[i_ent:])),
        sep=";",
        dtype={"Département": str, "Code INSEE": str},
        low_memory=False,
    )
    meta["lignes"] = len(df)
    return df, meta


def charger(dossier=None) -> tuple[pd.DataFrame, list[dict]]:
    """Concatène les 6 archives. Retourne (feux, métadonnées par fichier).

    Vérifié : les plages demandées se touchent à minuit mais la borne haute
    est exclusive — aucun doublon de frontière entre archives.
    """
    frames, metas = [], []
    for z in sorted((dossier or BDIFF_DIR).glob("*.zip")):
        df, meta = _lire_zip(z)
        df["src_zip"] = z.name
        frames.append(df)
        metas.append(meta)

    brut = pd.concat(frames, ignore_index=True)
    feux = brut.rename(columns=COLONNES)[list(COLONNES.values()) + ["src_zip"]].copy()

    feux["ts_alerte"] = pd.to_datetime(feux.ts_alerte, errors="coerce")
    feux["date_alerte"] = feux.ts_alerte.dt.date
    feux["heure_alerte"] = feux.ts_alerte.dt.hour.astype("Int64")

    # Doublons INTERNES à la BDIFF (même commune + même minute + même surface,
    # numéros d'ordre différents, dans un MÊME fichier source). ~1 800 lignes.
    # On les MARQUE sans les supprimer : y étant binaire, ils ne changent rien
    # à la cible — seulement à nb_feux et surface_m2.
    cle = ["code_insee_source", "ts_alerte", "surface_m2"]
    feux["doublon_suspect"] = feux.duplicated(subset=cle, keep=False)

    return feux, metas
