"""Étape 26 — les contours communaux, pour colorier la carte au polygone.

    python -m tvfed.contours

────────────────────────────────────────────────────────────────────────────
POURQUOI CE MODULE
────────────────────────────────────────────────────────────────────────────
La base ne contient que des **centroïdes** : `ref_commune.geom` est un
`geography(Point)`. Suffisant pour rattacher chaque commune à sa maille météo,
inutilisable pour peindre un territoire.

La carte affichait donc un semis de points. Le reproche est juste : la couleur
doit **recouvrir la commune**, pas la marquer d'un pixel. Il faut des
polygones, et ils viennent d'ailleurs.

Source : `france-geojson` (Grégoire David), version **simplifiée**, dérivée
des données IGN. 19 Mo, ~11 points par commune — déjà généralisée, ce qui
tombe bien : la version complète pèse 45 Mo pour une précision métrique dont
une carte nationale n'a aucun usage.

────────────────────────────────────────────────────────────────────────────
CE QUE CE MODULE PRODUIT, ET POURQUOI CE FORMAT
────────────────────────────────────────────────────────────────────────────
`app/donnees/contours.parquet` — **une ligne par anneau extérieur**, pas par
commune :

    code_insee   la clé, répétée pour les communes en plusieurs morceaux
    lon, lat     les sommets, en tableaux NumPy

deck.gl attend un anneau simple par polygone. Les 80 communes en
`MultiPolygon` — îles, enclaves — sont donc **éclatées** en autant de lignes,
plutôt que de ne garder que leur plus grand morceau : Marseille sans le
Frioul serait une carte fausse.

Les trous (anneaux intérieurs) sont ignorés : à l'échelle nationale, une
enclave de quelques hectares ne se voit pas, et la porter doublerait la
complexité du format pour rien.

⚠️ Les coordonnées sont **arrondies à 5 décimales**, soit environ un mètre.
La charge utile envoyée au navigateur à chaque interaction en dépend
directement, et personne ne distingue le mètre sur une carte de France.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .paths import RACINE

SOURCE = RACINE / "data" / "geo" / "communes-simplifiee.geojson"
APP = RACINE / "app" / "donnees"
DECIMALES = 5          # ~1 m ; au-delà, on transporte du bruit


def _anneaux(geom: dict):
    """Les anneaux EXTÉRIEURS d'une géométrie, quelle que soit sa forme."""
    if geom is None:
        return []
    if geom["type"] == "Polygon":
        parts = [geom["coordinates"]]
    elif geom["type"] == "MultiPolygon":
        parts = geom["coordinates"]
    else:
        return []
    # `p[0]` est l'anneau extérieur ; `p[1:]` sont les trous, ignorés
    return [p[0] for p in parts if p and len(p[0]) >= 4]


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"{SOURCE.relative_to(RACINE)} absent.\n"
            "    curl -sL -o data/geo/communes-simplifiee.geojson \\\n"
            "      https://raw.githubusercontent.com/gregoiredavid/"
            "france-geojson/master/communes-version-simplifiee.geojson")

    print("lecture du GeoJSON…")
    brut = json.loads(SOURCE.read_text(encoding="utf-8"))["features"]
    perimetre = set(pd.read_parquet(APP / "communes.parquet",
                                    columns=["code_insee"]).code_insee)
    print(f"  {len(brut):,} entités · périmètre de l'application "
          f"{len(perimetre):,} communes")

    lignes, vus = [], set()
    for f in brut:
        code = f["properties"]["code"]
        if code not in perimetre:
            continue
        for anneau in _anneaux(f.get("geometry")):
            a = np.asarray(anneau, dtype=np.float64).round(DECIMALES)
            lignes.append({"code_insee": code,
                           "lon": a[:, 0].astype(np.float32),
                           "lat": a[:, 1].astype(np.float32)})
            vus.add(code)

    d = pd.DataFrame(lignes)
    d.to_parquet(APP / "contours.parquet", index=False, compression="zstd")

    # ⚠️ ON COMPTE CE QU'ON PERD. Le GeoJSON et le COG 2026 n'ont pas le même
    # millésime : quelques communes fusionnées ou recréées n'ont pas de
    # contour. On ne devine pas — on les dénombre, et l'application le dit.
    manquantes = sorted(perimetre - vus)
    (APP / "contours_manquants.json").write_text(
        json.dumps(manquantes, indent=1), encoding="utf-8")

    n_pts = int(sum(len(x) for x in d.lon))
    taille = (APP / "contours.parquet").stat().st_size / 1e6
    print(f"\n  {len(d):,} anneaux pour {len(vus):,} communes")
    print(f"  {len(d) - len(vus):,} communes en plusieurs morceaux "
          f"(îles, enclaves)")
    print(f"  {n_pts:,} sommets, soit {n_pts / len(vus):.1f} par commune")
    print(f"  ⚠️ {len(manquantes)} communes SANS contour "
          f"({len(manquantes) / len(perimetre):.2%}) — millésime différent")
    print(f"\n✅ contours.parquet — {taille:.1f} Mo")


if __name__ == "__main__":
    main()
