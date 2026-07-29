"""Étape 2 du pipeline — chargement des faits observés.

    python -m tvfed.charger_faits

Dépend des référentiels (étape 1). Rejouable.
"""
from __future__ import annotations

import time

import pandas as pd
import yaml

from . import db
from .io import bdiff, cems, corine
from .paths import RACINE


def _chrono(label, t0):
    print(f"   {label:48s} {time.time() - t0:6.1f} s")


def main() -> None:
    cfg = yaml.safe_load((RACINE / "config" / "perimetre.yaml").read_text(encoding="utf-8"))
    an_min = int(cfg["periode"]["debut"][:4])
    an_max = int(cfg["periode"]["fin"][:4])
    conn = db.connexion()

    print("Chargement des faits\n" + "=" * 62)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE fait_meteo, fait_feu, fait_clc RESTART IDENTITY")
        cur.execute("SELECT code_insee FROM ref_commune")
        communes_ok = {r[0] for r in cur.fetchall()}
        cur.execute("SELECT DISTINCT cell_id FROM ref_commune")
        cells = {r[0] for r in cur.fetchall()}
        cur.execute("SELECT code_avant, code_apres FROM ref_passage_cog")
        passage = dict(cur.fetchall())
    conn.commit()

    # ── météo : année par année, seulement les mailles utilisées ──
    t0, total = time.time(), 0
    for an in cems.annees_disponibles():
        if not (an_min <= an <= an_max):
            continue
        m = cems.charger_annee(an, cell_ids=cells)
        total += db.copier(m, "fait_meteo", conn)
    _chrono(f"fait_meteo ({total:,} lignes, {len(cells)} mailles)", t0)

    # ── feux : remappés COG 2026 ──
    t0 = time.time()
    feux, metas = bdiff.charger()
    feux = feux[feux.annee.between(an_min, an_max)].copy()

    feux["code_insee"] = feux.code_insee_source.map(lambda c: passage.get(c, c))
    # un code toujours absent du référentiel est irrécupérable (commune
    # inconnue '83999', ou scission MOD=21) → NULL, compté, jamais deviné
    orphelins = ~feux.code_insee.isin(communes_ok)
    feux.loc[orphelins, "code_insee"] = None

    # Deux causes très différentes, à ne pas confondre dans le rapport :
    # les DOM sont exclus VOLONTAIREMENT du périmètre (hors bbox météo et
    # absents de CORINE) ; le reste est un échec de remappage.
    dom = orphelins & feux.dep_code.str.startswith(("97", "98"), na=False)
    irrecuperables = orphelins & ~dom

    cols = [
        "code_insee_source", "code_insee", "dep_code", "ts_alerte", "date_alerte",
        "heure_alerte", "surface_m2", "surface_foret_m2", "surface_maquis_m2",
        "nature", "type_peuplement", "src_zip", "doublon_suspect",
    ]
    db.copier(feux[cols], "fait_feu", conn)
    _chrono(f"fait_feu ({len(feux):,} lignes)", t0)
    print(f"      dont DOM, hors périmètre        : {dom.sum():>6,}")
    print(f"      dont irrécupérables (métropole) : {irrecuperables.sum():>6,}"
          f"  ({100 * irrecuperables.sum() / (~dom).sum():.2f} % du périmètre)")
    assert irrecuperables.sum() / max((~dom).sum(), 1) < 0.01, \
        "plus de 1 % de feux métropolitains non rattachés — vérifier la table de passage"

    # ── occupation du sol ──
    t0 = time.time()
    clc, _ = corine.charger()
    clc = clc[clc.code_insee.isin(communes_ok)]
    db.copier(clc[["code_insee", "millesime", "base", "poste", "surface_ha"]],
              "fait_clc", conn)
    _chrono(f"fait_clc ({len(clc):,} lignes, {clc.millesime.nunique()} millésimes)", t0)

    conn.close()
    print("=" * 62)
    notes = [n for m in metas for n in m.get("notes", [])]
    if notes:
        print("⚠️  Pertes de données documentées par la BDIFF :")
        for n in notes:
            print(f"      {n}")
    print("✅ faits chargés")


if __name__ == "__main__":
    main()
