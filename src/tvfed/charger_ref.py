"""Étape 1 du pipeline — chargement des référentiels.

    python -m tvfed.charger_ref

Ne dépend que des fichiers sources. Rejouable : vide les tables avant de charger.
"""
from __future__ import annotations

import time

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from . import calendrier, db, geo
from .io import cog, communes
from .paths import RACINE


def _chrono(label, t0):
    print(f"   {label:44s} {time.time() - t0:6.1f} s")


def distance_cote_km(communes_df: pd.DataFrame) -> pd.Series:
    """Distance au littoral, en projection Lambert-93 (mètres réels).

    Trait de côte Natural Earth 110m : très simplifié, à quelques km près.
    Suffisant ici — la feature sert à capter l'effet de brise marine, qui
    opère sur des dizaines de km, et la maille météo fait déjà 28 km.
    Ordre de grandeur vérifié : 0 km sur le littoral, 455 km à Gerstheim (67),
    le point de France le plus éloigné de toute mer.
    """
    try:
        monde = gpd.read_file(
            "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_coastline.zip"
        )
    except Exception as e:  # pas de réseau : on continue sans la feature
        print(f"   ⚠️  distance_cote indisponible ({type(e).__name__}) — colonne NULL")
        return pd.Series(np.nan, index=communes_df.index)

    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(communes_df.lon, communes_df.lat), crs=4326
    ).to_crs(2154)
    cote = monde.to_crs(2154).union_all()
    return pts.distance(cote) / 1000.0


def main() -> None:
    cfg = yaml.safe_load((RACINE / "config" / "perimetre.yaml").read_text(encoding="utf-8"))
    conn = db.connexion()

    print("Chargement des référentiels\n" + "=" * 58)
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE ref_voisinage, ref_passage_cog, ref_calendrier, "
            "ref_commune, ref_maille RESTART IDENTITY CASCADE"
        )
    conn.commit()

    # ── mailles météo ──
    t0 = time.time()
    lats, lons, mer = geo.grille_cems()
    mailles = geo.table_mailles(lats, lons, mer)
    db.copier(mailles, "ref_maille", conn)
    _chrono(f"ref_maille ({len(mailles):,} dont {int(mer.sum())} en mer)", t0)

    # ── communes ──
    t0 = time.time()
    com = communes.charger()
    com["cell_id"] = geo.rattacher_mailles(com, lats, lons)

    # garde-fou : une commune sur une cellule de mer signalerait un rattachement faux
    sur_mer = mailles.set_index("cell_id").est_mer.reindex(com.cell_id).values
    assert not sur_mer.any(), f"{sur_mer.sum()} communes rattachées à une maille de mer"

    com["distance_cote_km"] = distance_cote_km(com)
    com["in_perimetre"] = True
    cols = [
        "code_insee", "nom", "dep_code", "reg_code", "lat", "lon",
        "superficie_km2", "population", "densite",
        "altitude_moy", "altitude_min", "altitude_max",
        "grille_densite", "distance_cote_km", "cell_id", "in_perimetre",
    ]
    db.copier(com[cols], "ref_commune", conn)
    db.executer(
        "UPDATE ref_commune SET geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography",
        conn,
    )
    _chrono(f"ref_commune ({len(com):,})", t0)

    # ── calendrier ──
    t0 = time.time()
    cal = calendrier.construire(cfg["periode"]["debut"], cfg["periode"]["fin"])
    db.copier(cal, "ref_calendrier", conn)
    _chrono(f"ref_calendrier ({len(cal):,} jours)", t0)

    # ── table de passage COG ──
    t0 = time.time()
    codes_actuels = set(com.code_insee)
    passage = cog.table_passage(codes_actuels)
    db.copier(
        pd.DataFrame(
            {"code_avant": list(passage), "code_apres": list(passage.values()),
             "source": "INSEE v_mvt_commune_2026"}
        ),
        "ref_passage_cog",
        conn,
    )
    _chrono(f"ref_passage_cog ({len(passage):,} remappages)", t0)

    # ── voisinage ──
    t0 = time.time()
    vois = geo.voisinage(com, rayon_km=cfg["voisinage"]["rayon_max_km"])
    db.copier(vois, "ref_voisinage", conn)
    _chrono(f"ref_voisinage ({len(vois):,} paires ≤ {cfg['voisinage']['rayon_max_km']} km)", t0)

    conn.close()
    print("=" * 58 + "\n✅ référentiels chargés")


if __name__ == "__main__":
    main()
