"""Étape 21 — l'artefact que lit l'application.

    python -m tvfed.export_app

────────────────────────────────────────────────────────────────────────────
POURQUOI UN EXPORT ET NON UNE CONNEXION À LA BASE
────────────────────────────────────────────────────────────────────────────
La base `tvfed` pèse ~30 Go et vit dans un conteneur Docker local. Aucune
plateforme d'hébergement Streamlit ne peut l'embarquer, et une application qui
exige un PostgreSQL local n'est pas déployable.

On produit donc un **artefact reproductible** : quelques fichiers Parquet
compacts, régénérés par cette commande, versionnables. Ce n'est pas une
seconde base à maintenir — c'est une projection figée de la première.

────────────────────────────────────────────────────────────────────────────
CE QUE L'APPLICATION A BESOIN DE SAVOIR
────────────────────────────────────────────────────────────────────────────
Elle fait tourner le **modèle C** — le seul déployable, puisqu'il ne dépend
d'aucune donnée indisponible en temps réel. Ses 41 features se rangent en trois
familles, et une seule dépend de la date :

    STATIQUES     végétation, relief, densité, distance à la côte
                  → une ligne par commune, 34 734 lignes

    MÉTÉO         les 9 indices CEMS
                  → une ligne par maille et par jour

    CALENDRIER    doy, mois, jour de semaine, jours fériés, sin/cos
                  → calculées à la volée, elles ne dépendent que de la date

⚠️ La météo n'est exportée que sur une fenêtre de démonstration. En production
elle viendrait de l'API EFFIS, qui publie les prévisions à 9 jours — c'est
exactement le même schéma de colonnes.
"""
from __future__ import annotations

import shutil

import pandas as pd

from . import clustering, db
from .modele_v3 import K, METHODE
from .paths import PROCESSED, RACINE

APP = RACINE / "app" / "donnees"
# fenêtre de démonstration : une saison de feu complète, la dernière disponible
DEBUT, FIN = "2025-05-01", "2025-10-31"


def main() -> None:
    APP.mkdir(parents=True, exist_ok=True)

    # ── 1. les communes, avec tout ce qui ne dépend pas de la date ──────
    print("communes…")
    with db.connexion() as c:
        com = pd.read_sql("""
            SELECT c.code_insee, c.nom, c.dep_code, c.reg_code, c.lat, c.lon,
                   c.cell_id, c.population, c.densite, c.superficie_km2,
                   c.altitude_moy, c.altitude_max, c.altitude_min,
                   c.grille_densite, c.distance_cote_km
            FROM ref_commune c WHERE c.in_perimetre ORDER BY c.code_insee""", c)
        # CORINE : le millésime le plus récent, celui qu'on utilisera pour
        # toute date future — c'est l'hypothèse « végétation constante »
        clc = pd.read_sql(
            "SELECT * FROM clc_part WHERE millesime = 2018", c)

    com = com.merge(clc.drop(columns=["surface_tot_ha"]), on="code_insee",
                    how="left")
    com = com.rename(columns={"millesime": "clc_millesime"})

    # les colonnes dérivées, calculées ici une fois pour toutes plutôt que
    # dans l'application — elle doit rester une interface, pas un pipeline
    import numpy as np
    com["log_population"] = np.log1p(com.population.fillna(0))
    com["log_densite"] = np.log1p(com.densite.fillna(0))
    com["log_superficie"] = np.log1p(com.superficie_km2.fillna(0))
    com["amplitude_altitude"] = com.altitude_max - com.altitude_min

    # ── 2. le clustering, pour l'onglet 2050 ────────────────────────────
    print("clustering et risque de fond…")
    p = clustering.profil()
    sin = clustering.sinistralite()
    cl = clustering.ajuster(p, METHODE, K)
    manq = sorted(set(sin.code_insee) - set(cl.index))
    if manq:
        cl = pd.concat([cl, pd.Series(-1, index=manq, name="cluster_id")])
    taux = clustering.lisser(sin, cl)
    ref = taux[taux.an_exclue == 0].set_index("code_insee")
    com["cluster_id"] = com.code_insee.map(ref.cluster_id)
    com["risque_fond"] = com.code_insee.map(ref.taux_cluster_lisse)

    mig = pd.read_parquet(PROCESSED / "migration_clusters.parquet")
    for s in ("rcp4_5", "rcp8_5"):
        com[f"cluster_{s}"] = com.code_insee.map(mig[f"cluster_{s}"])
        com[f"risque_{s}"] = com.code_insee.map(mig[f"risque_{s}"])
        com[f"fiable_{s}"] = com.code_insee.map(mig[f"fiable_{s}"])
    com["jours_danger"] = com.code_insee.map(p.jours_fwi_sup_21)

    com.to_parquet(APP / "communes.parquet", index=False, compression="zstd")
    print(f"  {len(com):,} communes, {com.shape[1]} colonnes")

    # ── 3. la météo de la fenêtre de démonstration ──────────────────────
    print(f"météo {DEBUT} → {FIN}…")
    with db.connexion() as c:
        met = pd.read_sql(f"""
            SELECT cell_id, date, fwi, ffmc, dmc, dc, bui, isi, kbdi, erc
            FROM fait_meteo
            WHERE date BETWEEN '{DEBUT}' AND '{FIN}'
            ORDER BY cell_id, date""", c)
    # le décalage d'un jour, que le modèle attend
    g = met.groupby("cell_id")
    met["fwi_j1"] = g.fwi.shift(1)
    met["ffmc_j1"] = g.ffmc.shift(1)
    # le classement EFFIS, reconstruit à l'identique
    met["danger_effis"] = pd.cut(
        met.fwi, [-1, 5.2, 11.2, 21.3, 38, 50, 1e9], labels=False).astype("Int8")
    met = met.dropna(subset=["fwi_j1"])
    met.to_parquet(APP / "meteo.parquet", index=False, compression="zstd")
    print(f"  {len(met):,} lignes, {met.date.nunique()} jours, "
          f"{met.cell_id.nunique()} mailles")

    # ── 4. le modèle et ses métadonnées ─────────────────────────────────
    for f in ("modele_c.json", "importances_c.csv"):
        shutil.copy(PROCESSED / f, APP / f)

    # ⚠️ L'ORDRE DES FEATURES EST CELUI DE L'ENTRAÎNEMENT, PAS DE L'IMPORTANCE.
    # XGBoost vérifie les noms ET leur ordre à la prédiction. Exporter la liste
    # triée par importance — ce qui semblait naturel — faisait échouer l'appel
    # avec « feature_names mismatch ». La seule source fiable est le modèle
    # lui-même.
    from xgboost import XGBClassifier
    _m = XGBClassifier()
    _m.load_model(APP / "modele_c.json")
    ordre_features = list(_m.get_booster().feature_names)

    import json
    (APP / "meta.json").write_text(json.dumps({
        "modele": "XGBoost C — physique pur, 41 features",
        "pourquoi": "seul modèle déployable : ne dépend d'aucune donnée "
                    "indisponible en temps réel. La BDIFF ne publie pas "
                    "l'année en cours.",
        "test": {"pr_auc": 0.0106, "lift": 63.7, "periode": "2023-2025",
                 "lignes": 38068464, "feux": 6322},
        "modele_a": {"pr_auc": 0.0156, "lift": 93.8,
                     "note": "meilleur, mais exige l'historique récent des feux"},
        "fenetre_demo": [DEBUT, FIN],
        "features": ordre_features,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(f.stat().st_size for f in APP.glob("*"))
    print(f"\n✅ {APP.relative_to(RACINE)} — {total / 1e6:.1f} Mo")
    for f in sorted(APP.glob("*")):
        print(f"   {f.name:24s} {f.stat().st_size / 1e6:6.2f} Mo")


if __name__ == "__main__":
    main()
