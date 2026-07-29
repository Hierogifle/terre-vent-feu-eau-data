"""Étape 21 — l'artefact que lit l'application.

    python -m tvfed.export_app

────────────────────────────────────────────────────────────────────────────
POURQUOI UN EXPORT ET NON UNE CONNEXION À LA BASE
────────────────────────────────────────────────────────────────────────────
La base `tvfed` pèse ~30 Go et vit dans un conteneur Docker local. Aucune
plateforme d'hébergement Streamlit ne peut l'embarquer, et une application qui
exige un PostgreSQL local n'est pas déployable.

On produit donc un **artefact reproductible** : quelques fichiers Parquet
compacts, régénérés par cette commande. Ce n'est pas une seconde base à
maintenir — c'est une projection figée de la première.

────────────────────────────────────────────────────────────────────────────
LE POINT DE CONCEPTION : ON N'EXPORTE PAS DES PRÉDICTIONS, ON EXPORTE DE QUOI
LES CALCULER
────────────────────────────────────────────────────────────────────────────
L'application doit répondre pour **chaque commune, chaque jour, de 2026 à
2050**. Soit 34 734 × 365 × 25 ≈ **317 millions de valeurs** : impossible à
pré-calculer et à embarquer.

Mais toutes ces valeurs se reconstruisent à partir de trois briques minuscules :

    climatologie   FWI moyen de chaque maille pour chaque jour de l'année,
                   sur 2006-2019       →  1 131 × 366 = 414 k lignes

    facteur annuel réchauffement projeté de chaque maille, année par année
                   →  1 131 × 25 = 28 k lignes

    profil commune végétation, relief, densité — figés
                   →  34 734 lignes

L'application assemble à la volée. Une commune sur 25 ans : 9 125 lignes,
instantané. Une date sur toute la France : 34 734 lignes, instantané.

────────────────────────────────────────────────────────────────────────────
⚠️ CE QUE « LE 2 AOÛT 2032 » VEUT DIRE, ET CE QUE ÇA NE VEUT PAS DIRE
────────────────────────────────────────────────────────────────────────────
Personne ne connaît la météo du 2 août 2032. Les projections climatiques
donnent un CLIMAT, pas un bulletin.

La valeur produite est donc : « un 2 août typique, sous le climat de 2032 ».
Le cycle saisonnier vient des observations 2006-2019 ; seul son NIVEAU est
décalé par le facteur de l'année. L'application doit le dire — sans ça, elle
ment sur ce qu'elle sait.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from . import clustering, db
from .modele_v3 import K, METHODE
from .paths import PROCESSED, RACINE

APP = RACINE / "app" / "donnees"
CLIM_DEBUT, CLIM_FIN = 2006, 2019      # la période de référence du modèle
AN_DEBUT, AN_FIN = 2026, 2100          # l'horizon de projection
DECENNIES = [(1973, 1979), (1980, 1989), (1990, 1999),
             (2000, 2009), (2010, 2019), (2020, 2025)]


def _communes() -> pd.DataFrame:
    with db.connexion() as c:
        com = pd.read_sql("""
            SELECT c.code_insee, c.nom, c.dep_code, c.reg_code, c.lat, c.lon,
                   c.cell_id, c.population, c.densite, c.superficie_km2,
                   c.altitude_moy, c.altitude_max, c.altitude_min,
                   c.grille_densite, c.distance_cote_km
            FROM ref_commune c WHERE c.in_perimetre ORDER BY c.code_insee""", c)
        clc = pd.read_sql("SELECT * FROM clc_part WHERE millesime = 2018", c)
    com = com.merge(clc.drop(columns=["surface_tot_ha"]), on="code_insee",
                    how="left").rename(columns={"millesime": "clc_millesime"})

    # le code postal, pour la recherche — il n'est pas dans ref_commune
    ref = pd.read_csv(RACINE / "data" / "ville-france" / "communes-france-2026.csv",
                      usecols=["code_insee", "code_postal", "dep_nom", "reg_nom"],
                      dtype=str)
    com = com.merge(ref, on="code_insee", how="left")

    com["log_population"] = np.log1p(com.population.fillna(0))
    com["log_densite"] = np.log1p(com.densite.fillna(0))
    com["log_superficie"] = np.log1p(com.superficie_km2.fillna(0))
    com["amplitude_altitude"] = com.altitude_max - com.altitude_min
    return com


def _climatologie() -> pd.DataFrame:
    """Les 8 indices, moyennés par maille et par jour de l'année.

    C'est le cycle saisonnier de référence : ce que vaut un 2 août ordinaire
    dans cette maille. Il porte la forme ; les projections n'en décalent que
    le niveau.

    ⚠️ Bornée à 2006-2019 — la période d'entraînement. Utiliser 2006-2025
    ferait entrer la validation et le test dans une donnée servie par
    l'application, ce qui n'aurait pas de conséquence sur les métriques déjà
    publiées mais brouillerait la traçabilité.
    """
    with db.connexion() as c:
        m = pd.read_sql(f"""
            SELECT cell_id,
                   extract(doy FROM date)::int AS doy,
                   avg(fwi) AS fwi, avg(ffmc) AS ffmc, avg(dmc) AS dmc,
                   avg(dc) AS dc, avg(bui) AS bui, avg(isi) AS isi,
                   avg(kbdi) AS kbdi, avg(erc) AS erc
            FROM fait_meteo
            WHERE date BETWEEN '{CLIM_DEBUT}-01-01' AND '{CLIM_FIN}-12-31'
            GROUP BY 1, 2 ORDER BY 1, 2""", c)
    # ⚠️ LISSAGE OBLIGATOIRE, ET CE N'EST PAS COSMÉTIQUE.
    # Cette moyenne ne porte que sur 14 années : d'un jour à l'autre elle
    # saute de plus de 20 %. Mesuré sur une maille en août : doy 227 → 4,23
    # et doy 228 → 6,33.
    #
    # Conséquence visible : le 15 août tombe au doy 227 les années normales et
    # 228 les bissextiles. Une série annuelle affichait donc un DENTELÉ de
    # période 4 ans, qui n'a aucune réalité physique — c'était le calendrier,
    # pas le climat.
    #
    # Une fenêtre de 15 jours centrée, circulaire, retire ce bruit
    # d'échantillonnage sans toucher à la forme de la saison, qui varie sur
    # des semaines et non sur des jours.
    lisse = []
    for cid, g in m.groupby("cell_id", sort=False):
        g = g.sort_values("doy").copy()
        for c in ["fwi", "ffmc", "dmc", "dc", "bui", "isi", "kbdi", "erc"]:
            # concaténation circulaire : décembre précède janvier
            trois = pd.concat([g[c], g[c], g[c]], ignore_index=True)
            r = trois.rolling(15, center=True, min_periods=1).mean()
            g[c] = r.iloc[len(g):2 * len(g)].to_numpy()
        lisse.append(g)
    m = pd.concat(lisse, ignore_index=True)

    # le décalage d'un jour, en circulaire : le 1er janvier suit le 31 décembre
    g = m.groupby("cell_id")
    m["fwi_j1"] = g.fwi.shift(1).fillna(g.fwi.transform("last"))
    m["ffmc_j1"] = g.ffmc.shift(1).fillna(g.ffmc.transform("last"))
    return m


def _decennies() -> pd.DataFrame:
    """FWI moyen et jours de danger par maille et par décennie, 1973-2025.

    C'est ce qui permet de montrer une TENDANCE. Sur vingt ans la pente du
    FWI n'est pas significative (p = 0,13) ; sur cinquante-trois, elle se voit.
    """
    cas = " ".join(
        f"WHEN extract(year FROM date) BETWEEN {a} AND {b} THEN '{a}-{b}'"
        for a, b in DECENNIES)
    with db.connexion() as c:
        return pd.read_sql(f"""
            SELECT cell_id, CASE {cas} END AS periode,
                   avg(fwi) AS fwi_moyen,
                   percentile_cont(0.9) WITHIN GROUP (ORDER BY fwi) AS fwi_p90,
                   count(*) FILTER (WHERE fwi > 21.3)::float
                     / count(DISTINCT extract(year FROM date)) AS jours_danger,
                   count(*) FILTER (WHERE fwi > 38)::float
                     / count(DISTINCT extract(year FROM date)) AS jours_tres_eleve
            FROM fait_meteo
            WHERE CASE {cas} END IS NOT NULL
            GROUP BY 1, 2 ORDER BY 1, 2""", c)


def _facteurs() -> pd.DataFrame:
    """Le niveau de chaque année, rapporté au cycle saisonnier de référence.

    ────────────────────────────────────────────────────────────────────────
    UN SEUL MÉCANISME POUR LE PASSÉ ET POUR L'AVENIR
    ────────────────────────────────────────────────────────────────────────
    Servir la vraie météo journalière de 1973 à 2025 demanderait **759 Mo** —
    mesuré. Impossible à embarquer dans une application.

    On garde donc le cycle saisonnier de référence (2006-2019) et on lui
    applique, pour chaque année, un **facteur de niveau** :

        1973-2025   mesuré — le FWI moyen de cette année-là dans cette maille,
                    divisé par le FWI moyen de la référence. 2003 et 2022
                    ressortent au-dessus de 1, 2021 en dessous.

        2026-2050   projeté — interpolé entre 1,0 en 2025 et le facteur des
                    projections climatiques à 2048, centre de la fenêtre
                    2041-2055.

    ⚠️ CE QUE ÇA PERD, ET IL FAUT LE DIRE : la variation d'un jour à l'autre
    à l'intérieur d'une année. Un 2 août 2003 ressort comme « un 2 août sous
    le climat de 2003 », pas comme le pic de la canicule. La FORME de la
    saison vient de la référence, seul son NIVEAU est celui de l'année.

    ⚠️ L'interpolation vers 2050 est LINÉAIRE, ce qui est une simplification.
    Le réchauffement ne l'est pas — mais à cet horizon l'écart à la vraie
    trajectoire est petit devant l'incertitude des modèles, et les scénarios
    RCP ne divergent qu'après 2050 (mesuré : +3,73 pour RCP4.5 contre +3,13
    pour RCP8.5 sur la France).
    """
    from .migration import ratios

    # ── le passé : mesuré ───────────────────────────────────────────────
    with db.connexion() as c:
        an_obs = pd.read_sql(f"""
            WITH par_an AS (
              SELECT cell_id, extract(year FROM date)::int AS annee,
                     avg(fwi) AS fwi
              FROM fait_meteo GROUP BY 1, 2),
            reference AS (
              SELECT cell_id, avg(fwi) AS fwi_ref
              FROM fait_meteo
              WHERE date BETWEEN '{CLIM_DEBUT}-01-01' AND '{CLIM_FIN}-12-31'
              GROUP BY 1)
            SELECT p.cell_id, p.annee,
                   CASE WHEN r.fwi_ref > 0 THEN p.fwi / r.fwi_ref ELSE 1 END AS k
            FROM par_an p JOIN reference r USING (cell_id)
            ORDER BY 1, 2""", c)
    # le passé est le même quel que soit le scénario : il a eu lieu
    from .projections import SCENARIOS as _SC

    # ⚠️ LE PASSÉ EST RÉPLIQUÉ POUR CHAQUE SCÉNARIO. Il a eu lieu, il ne
    # dépend d'aucune trajectoire d'émissions — mais l'application filtre par
    # scénario, et un scénario sans passé produit une série vide. Une version
    # antérieure n'en gardait que deux sur trois : le graphique de RCP 2.6
    # s'affichait sans aucun point observé.
    passe = pd.concat([an_obs.assign(scenario=s) for s in _SC],
                      ignore_index=True)

    # ── l'avenir : projeté, ancré sur le PRÉSENT OBSERVÉ ────────────────
    # ⚠️ INTERPOLATION SUR TROIS ANCRAGES, PAS UNE DROITE.
    # Les scénarios se distinguent par leur COURBURE : RCP2.6 plafonne puis
    # redescend, RCP4.5 s'aplatit, RCP8.5 accélère. Deux points forceraient
    # une droite et effaceraient exactement ce qui les sépare — or c'est tout
    # l'intérêt de les comparer.
    #
    # On dispose donc de trois climatologies par scénario (centrées 2038,
    # 2058, 2092) et on interpole entre elles, en partant de 1,0 en 2025.
    # ⚠️ L'ANCRAGE EST LE POINT DÉLICAT, ET UNE PREMIÈRE VERSION L'AVAIT RATÉ.
    #
    # Partir de k = 1,0 en 2025 revient à dire « le climat de 2025 est celui
    # de la référence 2006-2019 ». C'est faux : la référence est déjà datée,
    # et le facteur observé de 2025 vaut ~1,3 sur beaucoup de mailles. La
    # projection démarrait donc SOUS le niveau observé, et la courbe plongeait
    # au passage 2025 → 2026 avant de remonter. Visuellement, le réchauffement
    # semblait s'annuler.
    #
    # On ancre donc sur la moyenne observée des quinze dernières années —
    # assez longue pour lisser la variabilité, assez courte pour représenter
    # le climat d'aujourd'hui — et on applique par-dessus la hausse relative
    # projetée par les modèles climatiques.
    recent = (an_obs[an_obs.annee.between(2011, 2025)]
              .groupby("cell_id").k.mean().rename("k_actuel"))
    k = ratios().merge(recent, on="cell_id", how="left")
    k["k_actuel"] = k.k_actuel.fillna(1.0)

    # ── la variabilité d'une année à l'autre, qui ne disparaîtra pas ────
    # Une projection est une MOYENNE. Les années continueront de varier
    # autour d'elle autant qu'aujourd'hui — l'écart p10-p90 observé vaut 0,64
    # sur certaines mailles, soit le double du réchauffement projeté sur
    # 25 ans. Tracer un trait fin ferait croire à une trajectoire connue.
    # On exporte donc la fourchette, et l'application dessine une bande.
    spread = (an_obs[an_obs.annee.between(2011, 2025)]
              .groupby("cell_id").k
              .agg(p10=lambda v: v.quantile(.1), p90=lambda v: v.quantile(.9),
                   moy="mean"))
    spread["r_bas"] = (spread.p10 / spread.moy).clip(.3, 1)
    spread["r_haut"] = (spread.p90 / spread.moy).clip(1, 3)
    k = k.merge(spread[["r_bas", "r_haut"]], on="cell_id", how="left")
    k[["r_bas", "r_haut"]] = k[["r_bas", "r_haut"]].fillna(1.0)

    from .projections import HORIZONS, SCENARIOS

    centres = sorted(c for _, c in HORIZONS.values())
    annees = np.arange(AN_DEBUT, AN_FIN + 1)
    futur = []
    for s in SCENARIOS:
        cols = [f"k_{s}_{c}" for c in centres if f"k_{s}_{c}" in k.columns]
        if not cols:
            continue
        # les points d'appui : 1,0 en 2025, puis une climatologie par horizon
        xs = np.array([2025] + [c for c in centres if f"k_{s}_{c}" in k.columns])
        ys = np.column_stack([np.ones(len(k))] + [k[c].to_numpy() for c in cols])
        for an in annees:
            # np.interp par ligne : chaque maille a sa propre trajectoire
            rel = np.array([np.interp(an, xs, y) for y in ys])
            centre = k.k_actuel.to_numpy() * rel
            futur.append(pd.DataFrame({
                "cell_id": k.cell_id, "annee": an, "scenario": s,
                "k": centre, "k_bas": centre * k.r_bas.to_numpy(),
                "k_haut": centre * k.r_haut.to_numpy()}))

    out = pd.concat([passe] + futur, ignore_index=True)
    out[["k_bas", "k_haut"]] = out[["k_bas", "k_haut"]].astype("float32")
    out["cell_id"] = out.cell_id.astype("int16")
    out["annee"] = out.annee.astype("int16")
    out["k"] = out.k.astype("float32")
    return out


def main() -> None:
    APP.mkdir(parents=True, exist_ok=True)

    print("communes…")
    com = _communes()

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
    com["risque_commune"] = com.code_insee.map(ref.taux_commune_lisse)

    mig = pd.read_parquet(PROCESSED / "migration_clusters.parquet")
    for s in ("rcp4_5", "rcp8_5"):
        com[f"cluster_{s}"] = com.code_insee.map(mig[f"cluster_{s}"])
        com[f"risque_{s}"] = com.code_insee.map(mig[f"risque_{s}"])
        com[f"fiable_{s}"] = com.code_insee.map(mig[f"fiable_{s}"])

    # les feux réellement observés, par commune — pour la fiche
    with db.connexion() as c:
        feux = pd.read_sql("""
            SELECT code_insee, count(*) FILTER (WHERE y) AS feux,
                   sum(surface_m2) / 10000.0 AS ha,
                   max(date) FILTER (WHERE y) AS dernier
            FROM grille GROUP BY 1""", c)
    com = com.merge(feux, on="code_insee", how="left")
    com["feux"] = com.feux.fillna(0).astype(int)
    com.to_parquet(APP / "communes.parquet", index=False, compression="zstd")
    print(f"  {len(com):,} communes, {com.shape[1]} colonnes")

    print("climatologie journalière…")
    clim = _climatologie()
    clim.to_parquet(APP / "climatologie.parquet", index=False, compression="zstd")
    print(f"  {len(clim):,} lignes ({clim.cell_id.nunique()} mailles × "
          f"{clim.doy.nunique()} jours)")

    print("évolution par décennie…")
    dec = _decennies()
    dec.to_parquet(APP / "decennies.parquet", index=False, compression="zstd")
    print(f"  {len(dec):,} lignes, périodes : "
          f"{', '.join(sorted(dec.periode.dropna().unique()))}")

    print("facteurs de réchauffement…")
    fac = _facteurs()
    fac.to_parquet(APP / "facteurs.parquet", index=False, compression="zstd")
    print(f"  {len(fac):,} lignes ({AN_DEBUT}-{AN_FIN}, 2 scénarios)")

    shutil.copy(PROCESSED / "modele_c.json", APP / "modele_c.json")
    shutil.copy(PROCESSED / "importances_c.csv", APP / "importances_c.csv")

    # ⚠️ L'ORDRE DES FEATURES EST CELUI DE L'ENTRAÎNEMENT, PAS DE L'IMPORTANCE.
    # XGBoost vérifie les noms ET leur ordre. Exporter la liste triée par
    # importance faisait échouer la prédiction avec « feature_names mismatch ».
    from xgboost import XGBClassifier
    _m = XGBClassifier()
    _m.load_model(APP / "modele_c.json")

    (APP / "meta.json").write_text(json.dumps({
        "modele": "XGBoost C — physique pur, 41 features",
        "pourquoi": "seul modèle déployable : ne dépend d'aucune donnée "
                    "indisponible en temps réel. La BDIFF ne publie pas "
                    "l'année en cours.",
        "test": {"pr_auc": 0.0106, "lift": 63.7, "periode": "2023-2025",
                 "lignes": 38068464, "feux": 6322},
        "modele_a": {"pr_auc": 0.0156, "lift": 93.8,
                     "note": "meilleur, mais exige l'historique récent des feux"},
        "climatologie": [CLIM_DEBUT, CLIM_FIN],
        "horizon": [AN_DEBUT, AN_FIN],
        "decennies": [f"{a}-{b}" for a, b in DECENNIES],
        "features": list(_m.get_booster().feature_names),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(f.stat().st_size for f in APP.glob("*") if f.is_file())
    print(f"\n✅ app/donnees — {total / 1e6:.1f} Mo")
    for f in sorted(APP.glob("*")):
        if f.is_file():
            print(f"   {f.name:24s} {f.stat().st_size / 1e6:6.2f} Mo")


if __name__ == "__main__":
    main()
