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

    C'est ce qui permet de montrer une TENDANCE, et la longueur compte : la
    variabilité interannuelle noie le signal sur une fenêtre courte.

    ⚠️ Les CHIFFRES de cette tendance sont calculés par `_tendances()` et
    exportés. Ne pas les recopier ici : une version antérieure de cette
    docstring et de l'application annonçait « +45 %, p < 0,0001 », valeur qui
    ne correspondait à aucune agrégation réelle.
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


def _tendances() -> pd.DataFrame:
    """Les pentes de fond, calculées — jamais recopiées.

    ⚠️ POURQUOI CETTE FONCTION EXISTE.
    L'application affichait « **+45 % de FWI moyen (p < 0,0001)** » et « sur
    2006-2025 la pente est non significative (p = 0,13) ». Ces deux valeurs
    étaient écrites en dur, et fausses : le calcul donne **+58 % en moyenne
    annuelle** et **+62 % en juin-septembre**, sur 1973-2025.

    Un chiffre recopié à la main finit toujours par diverger de son calcul.
    On l'exporte donc, et l'application le lit.

    ⚠️ L'AMPLEUR DÉPEND DE L'AGRÉGATION, et c'est précisément ce que l'absence
    de définition avait permis d'oublier : l'été se réchauffe plus vite que
    l'année entière. On calcule les deux et on dit laquelle on retient.
    """
    from scipy import stats

    requetes = {
        "FWI moyen annuel": (
            "SELECT extract(year FROM date)::int AS x, avg(fwi) AS y "
            "FROM fait_meteo GROUP BY 1 ORDER BY 1"),
        "FWI moyen juin-septembre": (
            "SELECT extract(year FROM date)::int AS x, avg(fwi) AS y "
            "FROM fait_meteo WHERE extract(month FROM date) BETWEEN 6 AND 9 "
            "GROUP BY 1 ORDER BY 1"),
        "jours de danger élevé (FWI > 21,3)": (
            "SELECT extract(year FROM date)::int AS x, "
            "count(*) FILTER (WHERE fwi > 21.3)::float "
            "/ count(DISTINCT cell_id) AS y "
            "FROM fait_meteo GROUP BY 1 ORDER BY 1"),
        "communes-jours en feu": (
            "SELECT extract(year FROM date)::int AS x, "
            "count(*) FILTER (WHERE y)::float AS y FROM grille GROUP BY 1 "
            "ORDER BY 1"),
    }
    lignes = []
    with db.connexion() as c:
        for nom, sql in requetes.items():
            d = pd.read_sql(sql, c)
            r = stats.linregress(d.x, d.y)
            debut = r.intercept + r.slope * d.x.iloc[0]
            span = int(d.x.iloc[-1] - d.x.iloc[0])
            lignes.append({
                "serie": nom, "an_min": int(d.x.iloc[0]),
                "an_max": int(d.x.iloc[-1]), "n_ans": span,
                "pente": r.slope, "p": r.pvalue,
                "valeur_debut": debut, "valeur_fin": debut + r.slope * span,
                "variation_pct": 100 * r.slope * span / debut if debut else np.nan,
                "significatif": bool(r.pvalue < 0.05)})
    return pd.DataFrame(lignes)


def _meteo_test() -> pd.DataFrame:
    """La météo RÉELLE du jeu de test, jour par jour.

    ⚠️ POURQUOI ELLE EST NÉCESSAIRE, ALORS QU'ON A DÉJÀ LA CLIMATOLOGIE.
    Le reste de l'application sert un cycle saisonnier dont seul le NIVEAU
    change d'une année à l'autre : « un 15 août ordinaire sous le climat de
    2024 ». C'est le bon objet pour parler d'avenir, où personne ne connaît
    la météo.

    Mais le mode rétrospectif prétend montrer *les prédictions du modèle sur
    le test* — celles qui valent 93,8× le hasard. Les produire à partir d'une
    météo lissée donnerait des scores qui ne correspondent à AUCUN modèle
    évalué : ni au test publié, ni à rien d'autre. On sert donc la vraie
    météo, et le mode rétrospectif devient exactement reproductible.

    On garde le **31 décembre 2022** : `fwi_j1` et `ffmc_j1` du 1er janvier
    2023 ont besoin de la veille. Ces deux colonnes ne sont pas stockées —
    un décalage d'un jour se recalcule au chargement en une ligne, et les
    écrire coûterait 9 Mo dans un dépôt que d'autres vont cloner.
    """
    with db.connexion() as c:
        m = pd.read_sql("""
            SELECT cell_id, date, fwi, ffmc, dmc, dc, bui, isi, kbdi, erc
            FROM fait_meteo
            WHERE date BETWEEN '2022-12-31' AND '2025-12-31'
            ORDER BY cell_id, date""", c)
    m["cell_id"] = m.cell_id.astype("int16")
    for col in ("fwi", "ffmc", "dmc", "dc", "bui", "isi", "kbdi", "erc"):
        m[col] = m[col].astype("float32")
    return m


def _operationnel() -> pd.DataFrame:
    """La courbe « budget de surveillance » sur le test, pour C et pour v3.

    Question opérationnelle : si on surveille les X % du territoire les mieux
    classés, quelle part des départs attrape-t-on ?

    ⚠️ DEUX MODÈLES, ET IL FAUT LES DEUX.
    `operationnel_test.csv`, produit par `evaluation_test`, porte sur
    **xgb_v3** — `MODELE = "xgb_v3"` en tête de ce module. Ses 41,9 % à 1 % de
    budget décrivent donc le modèle qu'on NE déploie PAS. Le modèle C, lui,
    attrape 38,5 %. Servir le premier chiffre pour vanter le second serait la
    même erreur que le SHAP calculé sur le mauvais modèle.

    On calcule donc les deux, et l'écart devient lisible : c'est le prix de la
    déployabilité, exprimé en départs de feu plutôt qu'en PR-AUC.

    ⚠️ Grille LOGARITHMIQUE. Tout se joue entre 0,05 % et 5 % ; une grille
    linéaire y placerait trois points sur deux cents.
    """
    budgets = None
    out = []
    for nom, fichier, col in (
            ("C", "scores_c_test.parquet", "p_c"),
            ("v3", "scores_test.parquet", "xgb_v3")):
        f = PROCESSED / fichier
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=[col, "y"])
        y = d.y.to_numpy()[np.argsort(-d[col].to_numpy())]
        cum, n, tot = np.cumsum(y), len(y), int(y.sum())
        if budgets is None:
            budgets = np.unique(
                np.round(np.logspace(np.log10(2e-4), 0, 200) * n)
                .astype(int).clip(1, n))
        pris = cum[budgets - 1]
        out.append(pd.DataFrame({
            "modele": nom, "budget": budgets / n, "lignes": budgets,
            "feux": pris, "rappel": pris / tot, "precision": pris / budgets}))
    return pd.concat(out, ignore_index=True)


def _jours_feu() -> pd.DataFrame:
    """Un jour-feu = (commune, date, nombre de feux déclarés ce jour-là).

    C'est la brique qui permet de reconstruire à la volée les 5 features
    d'historique du modèle v3 — `feux_commune_7j/30j/90j/365j` et
    `jours_depuis_dernier_feu`. Réplique exacte du CTE `jours_feu` de
    `sql/40_feat_lags.sql` : on compte les LIGNES BDIFF, donc deux feux le
    même jour dans la même commune comptent deux.

    Quelques dizaines de milliers de lignes — rien, comparé aux 253 M de la
    grille qu'il aurait fallu embarquer pour servir les mêmes colonnes.
    """
    with db.connexion() as c:
        return pd.read_sql("""
            SELECT code_insee, date_alerte AS date, count(*)::int AS n
            FROM fait_feu WHERE code_insee IS NOT NULL
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

    # ⚠️ LES NOMS EXACTS ATTENDUS PAR LE MODÈLE v3. `risque_fond` et
    # `risque_commune` portent déjà ces valeurs, mais XGBoost vérifie les NOMS
    # des colonnes : servir v3 exige les libellés d'entraînement. Ce sont les
    # taux `an_exclue == 0`, c'est-à-dire ajustés sur le train complet —
    # exactement ceux qu'a utilisés l'évaluation test.
    for col in ("taux_cluster_lisse", "taux_commune_lisse",
                "ratio_commune_cluster"):
        com[col] = com.code_insee.map(ref[col])

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

    print("tendances de fond…")
    ten = _tendances()
    ten.to_csv(APP / "tendances.csv", index=False)
    for _, r in ten.iterrows():
        print(f"   {r.serie:36s} {r.variation_pct:+6.0f} % sur {r.n_ans} ans   "
              f"p = {r.p:.1e}  {'' if r.significatif else '(non significatif)'}")

    # ── SHAP du modèle C, sous-échantillonné ────────────────────────────
    # 60 000 lignes × 41 features font 10 Mo par tableau. Un nuage de points
    # de 60 000 valeurs est illisible de toute façon : 15 000 suffisent à
    # dessiner la même distribution, pour un quart du poids.
    N_SHAP = 15_000
    src = PROCESSED / "shap_c_alea.npy"
    if src.exists():
        rng = np.random.default_rng(0)
        for nom in ("alea", "sommet"):
            v = np.load(PROCESSED / f"shap_c_{nom}.npy")
            X = pd.read_parquet(PROCESSED / f"shap_c_{nom}_X.parquet")
            idx = rng.choice(len(v), min(N_SHAP, len(v)), replace=False)
            np.save(APP / f"shap_c_{nom}.npy", v[idx])
            X.iloc[idx].reset_index(drop=True).to_parquet(
                APP / f"shap_c_{nom}_X.parquet", index=False,
                compression="zstd")
            print(f"  SHAP {nom:7s} {len(idx):,} lignes sur {len(v):,}")
        shutil.copy(PROCESSED / "shap_c_colonnes.json",
                    APP / "shap_c_colonnes.json")
        shutil.copy(PROCESSED / "fond_dice.parquet", APP / "fond_dice.parquet")
    else:
        print("   ⚠️ SHAP du modèle C absent — lancer tvfed.explications")

    print("météo observée du test, pour le mode rétrospectif…")
    mt_test = _meteo_test()
    mt_test.to_parquet(APP / "meteo_test.parquet", index=False,
                       compression="zstd")
    print(f"  {len(mt_test):,} lignes ({mt_test.cell_id.nunique()} mailles × "
          f"{mt_test.date.nunique()} jours de 2023-2025)")

    if (PROCESSED / "scores_c_test.parquet").exists():
        print("courbe du budget de surveillance…")
        op = _operationnel()
        op.to_csv(APP / "operationnel_courbe.csv", index=False)
        for b in (0.01, 0.05, 0.10):
            r = op.iloc[(op.budget - b).abs().idxmin()]
            print(f"   {r.budget:6.2%} du territoire → {r.rappel:5.1%} "
                  f"des départs, précision {r.precision:.3%}")
    else:
        print("   ⚠️ scores_c_test.parquet absent — pas de courbe opérationnelle")

    print("jours-feu, pour reconstruire l'historique du modèle v3…")
    jf = _jours_feu()
    jf.to_parquet(APP / "jours_feu.parquet", index=False, compression="zstd")
    print(f"  {len(jf):,} jours-feu, {int(jf.n.sum()):,} feux, "
          f"{jf.date.min()} → {jf.date.max()}")

    for f in ("modele_c.json", "importances_c.csv", "modele_v3.json",
              "importances_v3.csv", "comparaison_appariee.csv",
              "pr_auc_val.csv", "transfert_spatial.csv", "series_adf.csv",
              "series_sarimax.csv", "test_par_annee.csv", "modeles_lstm.csv",
              "best_params_lstm.json", "calibration_v3.csv",
              "baselines.csv", "modeles_ensemble.csv", "resultat_test.csv",
              "modele_c_test.csv", "modele_taille.csv",
              "fiabilite_brut.csv", "fiabilite_platt.csv",
              "fiabilite_isotonic.csv", "courbe_apprentissage.csv",
              "operationnel_test.csv"):
        if (PROCESSED / f).exists():
            shutil.copy(PROCESSED / f, APP / f)
        else:
            print(f"   ⚠️ {f} absent — la page qui l'utilise devra le gérer")

    # ⚠️ L'ORDRE DES FEATURES EST CELUI DE L'ENTRAÎNEMENT, PAS DE L'IMPORTANCE.
    # XGBoost vérifie les noms ET leur ordre. Exporter la liste triée par
    # importance faisait échouer la prédiction avec « feature_names mismatch ».
    from xgboost import XGBClassifier
    _m = XGBClassifier()
    _m.load_model(APP / "modele_c.json")

    # ⚠️ LE MODÈLE v3 N'A PAS DE NOMS DE FEATURES.
    # `modele_v3.py` l'entraîne sur `prep.transform(train)`, qui rend un
    # tableau NumPy : XGBoost n'a donc mémorisé aucun nom, et
    # `get_booster().feature_names` vaut None. Le modèle C, lui, est entraîné
    # sur un DataFrame et porte ses noms.
    #
    # Conséquence pour l'application : servir v3 exige de présenter les
    # colonnes dans l'ORDRE EXACT de l'entraînement, sans quoi la prédiction
    # sera silencieusement fausse — aucune vérification de nom ne la
    # rattrapera. On exporte donc cet ordre, pris à sa source : la
    # `Preparation` ajustée sur le train.
    from .modeles import Preparation

    train = clustering.appliquer(
        pd.read_parquet(PROCESSED / "train.parquet"), taux)
    colonnes_v3 = list(Preparation().fit(train).colonnes_)
    _v3 = XGBClassifier()
    _v3.load_model(APP / "modele_v3.json")
    if _v3.n_features_in_ != len(colonnes_v3):
        raise ValueError(
            f"modele_v3 attend {_v3.n_features_in_} features, la préparation "
            f"en produit {len(colonnes_v3)} — relancer tvfed.modele_v3")
    print(f"  v3 : {len(colonnes_v3)} features, ordre d'entraînement conservé")

    (APP / "meta.json").write_text(json.dumps({
        "modele": "XGBoost C — physique pur, 41 features",
        "pourquoi": "seul modèle déployable : ne dépend d'aucune donnée "
                    "indisponible en temps réel. La BDIFF ne publie pas "
                    "l'année en cours.",
        "test": {"pr_auc": 0.0106, "lift": 63.7, "periode": "2023-2025",
                 "lignes": 38068464, "feux": 6322},
        "modele_a": {"pr_auc": 0.0156, "lift": 93.8,
                     "note": "meilleur, mais exige l'historique récent des feux"},
        # ⚠️ LES BORNES DES PARTITIONS SONT SERVIES, PAS ÉCRITES EN DUR DANS
        # L'APPLICATION. C'est elles qui décident quand le modèle v3 a le
        # droit d'être affiché : jamais sur le train (il a appris ces lignes),
        # jamais sur la validation (elle a servi à choisir les
        # hyperparamètres, le modèle et la calibration), uniquement sur le
        # test — la seule fenêtre qui n'a informé aucune décision.
        "splits": {"train": [2006, 2019], "val": [2020, 2022],
                   "test": [2023, 2025]},
        "climatologie": [CLIM_DEBUT, CLIM_FIN],
        "horizon": [AN_DEBUT, AN_FIN],
        "decennies": [f"{a}-{b}" for a, b in DECENNIES],
        "features": list(_m.get_booster().feature_names),
        # ⚠️ ORDRE D'ENTRAÎNEMENT, pas ordre alphabétique ni d'importance.
        # v3 ne porte pas ses noms : c'est cette liste qui fait foi.
        "features_v3": colonnes_v3,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(f.stat().st_size for f in APP.glob("*") if f.is_file())
    print(f"\n✅ app/donnees — {total / 1e6:.1f} Mo")
    for f in sorted(APP.glob("*")):
        if f.is_file():
            print(f"   {f.name:24s} {f.stat().st_size / 1e6:6.2f} Mo")


if __name__ == "__main__":
    main()
