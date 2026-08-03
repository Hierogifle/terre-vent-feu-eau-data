"""Le socle partagé par les pages : chargement, assemblage, prédiction.

────────────────────────────────────────────────────────────────────────────
CE QUE CE MODULE FAIT, ET POURQUOI IL EXISTE
────────────────────────────────────────────────────────────────────────────
L'application doit répondre pour n'importe quelle commune et n'importe quel
jour entre 1973 et 2100. Un seul mécanisme, deux régimes :

    1973-2025   le cycle saisonnier de référence, dont le niveau est celui
                MESURÉ cette année-là
    2026-2100   le même cycle, dont le niveau est celui PROJETÉ par les
                modèles climatiques, selon trois scénarios du GIEC

⚠️ La distinction est essentielle et l'interface doit la porter. Pour 2032 on
ne prédit pas la météo — personne ne le peut. On dit ce que vaudrait un jour
ORDINAIRE de cette saison sous le climat de 2032. C'est ce que donnent les
projections climatiques, et rien de plus.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DON = Path(__file__).parent / "donnees"

# ── charte, reprise des notebooks ────────────────────────────────────────
INK, MUTED, GRID = "#0b0b0b", "#6b6963", "#e1e0d9"
FOND, GRIS = "#fcfcfb", "#c3c2b7"
BLEU, ORANGE, ROUGE, VERT, VIOLET = "#2a78d6", "#eb6834", "#e34948", "#1baf7a", "#4a3aa7"

MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
        "août", "septembre", "octobre", "novembre", "décembre"]


def nb(x, dec: int = 0) -> str:
    """38068464 → « 38 068 464 ».

    ⚠️ POURQUOI CE HELPER EXISTE.
    Le raccourci `f"…{n:,}…".replace(",", " ")` était appliqué à des PHRASES
    entières, et mangeait les virgules légitimes. Le bandeau affichait
    « physique pur 41 features » au lieu de « physique pur, 41 features », et
    une légende « 38 068 464 communes-jours 9 176 feux ». On formate le NOMBRE,
    jamais le texte qui l'entoure.
    """
    return f"{x:,.{dec}f}".replace(",", " ")


def date_fr(d: pd.Timestamp) -> str:
    """« 15 août 2024 ». `strftime('%B')` rendrait « August » : la locale du
    serveur n'est pas celle du lecteur, et on ne la configure pas depuis une
    application web."""
    return f"{d.day} {MOIS[d.month - 1]} {d.year}"

# seuils et couleurs officiels EFFIS
SEUILS = [5.2, 11.2, 21.3, 38, 50]
CLASSES = ["très faible", "faible", "modéré", "élevé", "très élevé", "extrême"]
COUL_EFFIS = ["#84F07F", "#FFEB3C", "#FFB00C", "#FA4F00", "#B40000", "#280923"]

AN_OBS_MIN, AN_OBS_MAX = 1973, 2025
AN_PROJ_MIN, AN_PROJ_MAX = 2026, 2100
SCENARIOS = ["rcp2_6", "rcp4_5", "rcp8_5"]
ETIQ_SC = {"rcp2_6": "RCP 2.6 — neutralité carbone ~2070",
           "rcp4_5": "RCP 4.5 — émissions plafonnées vers 2040",
           "rcp8_5": "RCP 8.5 — aucune politique climatique"}
COUL_SC = {"rcp2_6": "#2a78d6", "rcp4_5": "#eb6834", "rcp8_5": "#b3121a"}


def hex_rgb(h: str) -> list[int]:
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


# ════════════════════════════════════════════════════════════════════════
#  chargement — en cache, l'artefact ne change jamais en cours de session
# ════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def communes() -> pd.DataFrame:
    return pd.read_parquet(DON / "communes.parquet")


@st.cache_data(show_spinner=False)
def climatologie() -> pd.DataFrame:
    return pd.read_parquet(DON / "climatologie.parquet")


@st.cache_data(show_spinner=False)
def decennies() -> pd.DataFrame:
    return pd.read_parquet(DON / "decennies.parquet")


@st.cache_data(show_spinner=False)
def facteurs() -> pd.DataFrame:
    return pd.read_parquet(DON / "facteurs.parquet")


@st.cache_data(show_spinner=False)
def meta() -> dict:
    return json.loads((DON / "meta.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner="chargement de la météo observée…")
def meteo_observee() -> pd.DataFrame:
    """La météo RÉELLE de 2023-2025, chargée seulement si on la demande.

    ⚠️ 33 Mo sur disque, ~92 Mo en mémoire. Le mode courant de l'application
    n'en a pas besoin : il sert la climatologie décalée, qui suffit à parler
    d'un jour ordinaire sous un climat donné. Seul le mode **rétrospectif** la
    réclame, parce qu'il prétend montrer les prédictions réellement évaluées
    sur le test. `@st.cache_data` la charge donc au premier usage, pas au
    démarrage.
    """
    d = pd.read_parquet(DON / "meteo_test.parquet")
    d["date"] = pd.to_datetime(d.date)
    # le décalage d'un jour se recalcule ici plutôt que d'être stocké : deux
    # colonnes de plus auraient coûté 9 Mo dans le dépôt pour une opération
    # qui prend une milliseconde. Le 31/12/2022 est présent exprès, pour que
    # le 1er janvier 2023 ait bien une veille.
    d = d.sort_values(["cell_id", "date"])
    g = d.groupby("cell_id")
    d["fwi_j1"] = g.fwi.shift(1).astype("float32")
    d["ffmc_j1"] = g.ffmc.shift(1).astype("float32")
    return d[d.date >= "2023-01-01"].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def jours_feu() -> pd.DataFrame:
    """(commune, date, nombre de feux) — la brique de l'historique du v3."""
    d = pd.read_parquet(DON / "jours_feu.parquet")
    d["date"] = pd.to_datetime(d.date)
    return d


@st.cache_data(show_spinner=False)
def tendances() -> pd.DataFrame:
    """Les pentes de fond, CALCULÉES puis exportées.

    ⚠️ Ne jamais réécrire ces chiffres à la main dans une page. Une version
    antérieure affichait « +45 %, p < 0,0001 », valeur qui ne correspondait à
    aucune agrégation réelle et a survécu des semaines parce qu'elle était en
    dur dans une chaîne de caractères.
    """
    return pd.read_csv(DON / "tendances.csv")


@st.cache_resource(show_spinner=False)
def modele(nom: str = "C"):
    """Le modèle demandé. « C » est le déployable, « v3 » le rétrospectif."""
    from xgboost import XGBClassifier

    m = XGBClassifier()
    m.load_model(DON / ("modele_c.json" if nom == "C" else "modele_v3.json"))
    return m


# ════════════════════════════════════════════════════════════════════════
#  quand a-t-on le DROIT d'afficher le modèle v3 ?
# ════════════════════════════════════════════════════════════════════════
# Le v3 est meilleur que C sur le test — 93,8× contre 63,7×. Mais son score
# ne veut dire quelque chose que sur une période qui n'a informé AUCUNE
# décision. Il n'y en a qu'une : le test.
#
# Afficher v3 ailleurs produirait des cartes flatteuses et malhonnêtes, sans
# qu'aucune erreur ne se déclenche. D'où cette fonction, et le fait que les
# bornes viennent de `meta.json` plutôt que d'être écrites ici.

MOTIFS = {
    "train": (
        "**{a}-{b} — le modèle v3 a appris ces années.** Afficher son score "
        "ici montrerait ce qu'il a mémorisé, pas ce qu'il sait prédire. "
        "Les cartes paraîtraient excellentes, pour la pire des raisons."),
    "val": (
        "**{a}-{b} — la validation.** Elle a servi à choisir les "
        "hyperparamètres par Optuna, à sélectionner le modèle, à ajuster la "
        "calibration et à faire tourner le bootstrap apparié. Un score "
        "affiché ici serait optimiste pour une raison invisible à l'œil."),
    "avenir": (
        "**Au-delà de {b} — le modèle v3 est impossible.** Il a besoin des "
        "feux des 365 derniers jours ; la BDIFF ne publie pas l'année en "
        "cours, et personne ne connaîtra jamais les feux de 2049."),
    "avant": (
        "**Avant {a} — pas d'historique exploitable.** Le v3 a besoin de "
        "365 jours de feux antérieurs, et la BDIFF n'est pas homogène avant "
        "2006."),
}


def periode(an: int) -> str:
    """Dans quelle partition tombe cette année ? train, val, test, avenir…"""
    s = meta()["splits"]
    for nom in ("train", "val", "test"):
        if s[nom][0] <= an <= s[nom][1]:
            return nom
    return "avenir" if an > s["test"][1] else "avant"


def v3_autorise(an: int) -> tuple[bool, str]:
    """(autorisé, message). Le refus est l'argument, pas une limitation."""
    p = periode(an)
    if p == "test":
        return True, ""
    s = meta()["splits"]
    a, b = (s["train"][0], s["test"][1])
    return False, MOTIFS[p].format(a=s.get(p, [a, b])[0] if p in s else a,
                                   b=s.get(p, [a, b])[1] if p in s else b)


# ════════════════════════════════════════════════════════════════════════
#  assemblage des features
# ════════════════════════════════════════════════════════════════════════
def calendrier(d: pd.Timestamp) -> dict:
    """Les 11 features de date. Row-local : elles ne dépendent que du jour."""
    import holidays

    fr = holidays.France(years=d.year)
    doy, mois = int(d.dayofyear), int(d.month)
    return {
        "doy": doy, "mois": mois, "jour_semaine": int(d.dayofweek),
        "est_weekend": int(d.dayofweek >= 5),
        "est_ferie": int(d.date() in fr),
        "est_14_juillet": int(mois == 7 and d.day == 14),
        "est_15_aout": int(mois == 8 and d.day == 15),
        "sin_doy": np.sin(2 * np.pi * doy / 365.25),
        "cos_doy": np.cos(2 * np.pi * doy / 365.25),
        "sin_mois": np.sin(2 * np.pi * mois / 12),
        "cos_mois": np.cos(2 * np.pi * mois / 12),
    }


INDICES = ["fwi", "ffmc", "dmc", "dc", "bui", "isi", "kbdi", "erc",
           "fwi_j1", "ffmc_j1"]


def meteo_du_jour(d: pd.Timestamp, scenario: str = "rcp8_5",
                  variante: str = "k", observee: bool = False) -> pd.DataFrame:
    """La météo de chaque maille pour cette date.

    Deux régimes, et la distinction est essentielle :

    `observee=False` (défaut) — le cycle saisonnier 2006-2019, dont le NIVEAU
        est multiplié par le facteur de l'année. Un « 2 août 2032 » est un
        2 août ORDINAIRE sous le climat de 2032. C'est le seul objet
        défendable pour parler d'avenir.

    `observee=True` — la météo RÉELLEMENT mesurée ce jour-là. Disponible
        uniquement sur 2023-2025, et réservée au mode rétrospectif : c'est
        elle qu'ont vue les modèles pendant l'évaluation test.
    """
    if observee:
        m = meteo_observee()
        j = m[m.date == d].copy()
        if j.empty:
            raise ValueError(
                f"la météo observée n'est exportée que pour 2023-2025 — "
                f"{d.date()} est hors de cette fenêtre")
        j["danger_effis"] = np.digitize(j.fwi, SEUILS).astype(int)
        return j.drop(columns=["date"])

    clim = climatologie()
    doy = min(int(d.dayofyear), 366)
    j = clim[clim.doy == doy].copy()

    # ⚠️ LE FACTEUR S'APPLIQUE AUSSI AU PASSÉ. Une première version ne le
    # posait qu'au-delà de 2025 : 1990 et 2025 sortaient alors IDENTIQUES,
    # puisque tous deux servaient la climatologie brute. Le facteur du passé
    # est mesuré (le FWI de cette année-là rapporté à la référence), celui de
    # l'avenir est projeté — mais le mécanisme est le même.
    f = facteurs()
    col = variante if variante in f.columns else "k"
    k = f[(f.annee == d.year) & (f.scenario == scenario)][["cell_id", col]]
    k = k.rename(columns={col: "k"})
    j = j.merge(k, on="cell_id", how="left")
    j["k"] = j.k.fillna(1.0)
    for c in INDICES:
        j[c] = j[c] * j.k
    j = j.drop(columns=["k"])

    # le classement EFFIS se recalcule après décalage, pas avant
    j["danger_effis"] = np.digitize(j.fwi, SEUILS).astype(int)
    return j


LAGS = ["feux_commune_7j", "feux_commune_30j", "feux_commune_90j",
        "feux_commune_365j", "jours_depuis_dernier_feu"]


@st.cache_data(show_spinner=False)
def historique(d: pd.Timestamp) -> pd.DataFrame:
    """Les 5 features d'historique du modèle v3, au jour `d`, PASSÉ STRICT.

    Réplique exacte de `sql/40_feat_lags.sql` :

        fenêtre    les jours-feu de [d − 365, d − 1]. Le jour d lui-même
                   n'entre JAMAIS — sans cette borne, `feux_commune_7j`
                   contiendrait le feu qu'on cherche à prédire, et c'est la
                   fuite la plus classique du domaine
        comptage   on somme les FEUX déclarés, pas les communes-jours : deux
                   feux le même jour dans la même commune comptent deux
        défauts    0 pour les compteurs, **9999** pour
                   `jours_depuis_dernier_feu` — la valeur qu'utilise le
                   COALESCE de `sql/50_matrice.sql`. Mettre 0 signifierait
                   « il a brûlé aujourd'hui », soit l'inverse du sens voulu.

    C'est ce recalcul qui évite d'embarquer les 253 M lignes de la grille :
    49 130 jours-feu suffisent à reconstruire la colonne pour n'importe quelle
    date.
    """
    jf = jours_feu()
    fen = jf[(jf.date >= d - pd.Timedelta(days=365))
             & (jf.date <= d - pd.Timedelta(days=1))].copy()

    codes = communes().code_insee
    if fen.empty:
        h = pd.DataFrame(0, index=codes, columns=LAGS, dtype=float)
        h["jours_depuis_dernier_feu"] = 9999.0
        return h.rename_axis("code_insee").reset_index()

    fen["decalage"] = (d - fen.date).dt.days
    g = fen.groupby("code_insee")
    h = pd.DataFrame({
        "feux_commune_7j": fen[fen.decalage <= 7].groupby("code_insee").n.sum(),
        "feux_commune_30j": fen[fen.decalage <= 30].groupby("code_insee").n.sum(),
        "feux_commune_90j": fen[fen.decalage <= 90].groupby("code_insee").n.sum(),
        "feux_commune_365j": g.n.sum(),
        "jours_depuis_dernier_feu": g.decalage.min(),
    }).reindex(codes)
    h[LAGS[:4]] = h[LAGS[:4]].fillna(0)
    h["jours_depuis_dernier_feu"] = h.jours_depuis_dernier_feu.fillna(9999)
    return h.rename_axis("code_insee").reset_index()


def predire(d: pd.Timestamp, scenario: str = "rcp8_5", nom: str = "C",
            observee: bool = False) -> pd.DataFrame:
    """Score de chaque commune pour cette date, par le modèle demandé.

    ⚠️ `nom="v3"` n'est légitime que sur le jeu de TEST — voir `v3_autorise`.
    On lève plutôt que de rendre une carte flatteuse et fausse.

    ⚠️ `observee=True` sert la météo réellement mesurée. Le mode rétrospectif
    doit l'utiliser POUR LES DEUX MODÈLES : comparer v3 sur météo réelle à C
    sur climatologie mesurerait la différence de météo, pas de modèle.
    """
    com, mt = communes(), meta()
    X = com.merge(meteo_du_jour(d, scenario, observee=observee),
                  on="cell_id", how="inner")
    for k, v in calendrier(d).items():
        X[k] = v

    if nom == "v3":
        ok, motif = v3_autorise(d.year)
        if not ok:
            raise ValueError(motif)
        X = X.merge(historique(d), on="code_insee", how="left")
        # ⚠️ ORDRE, PAS NOMS. Le v3 a été entraîné sur un tableau NumPy et ne
        # porte aucun nom de colonne : XGBoost ne vérifiera RIEN. Servir les
        # features dans un autre ordre produirait des scores plausibles et
        # entièrement faux. `features_v3` est l'ordre d'entraînement exact.
        colonnes = mt["features_v3"]
    else:
        colonnes = mt["features"]

    X["score"] = modele(nom).predict_proba(X[colonnes].astype(float))[:, 1]
    X["rang"] = X.score.rank(pct=True)
    return X


def serie_commune(code: str, annees, mois: int = 8, jour: int = 15,
                  scenarios=None) -> pd.DataFrame:
    """L'évolution d'une commune, année par année, pour un ou plusieurs scénarios.

    Prendre la même date chaque année isole l'effet du climat : la saison, le
    jour de la semaine et la végétation ne varient pas.

    ⚠️ TROIS SÉRIES PAR SCÉNARIO. Une projection est une MOYENNE, et les
    années continueront de varier autour d'elle autant qu'aujourd'hui. Sur
    certaines mailles l'écart p10-p90 observé vaut le DOUBLE du réchauffement
    projeté sur vingt-cinq ans. Un trait fin ferait croire à une trajectoire
    connue : on rend la fourchette, l'application dessine une bande.

    ⚠️ VECTORISÉ. Une première version appelait `meteo_du_jour` une fois par
    année et par variante — 128 ans × 3 scénarios × 3 variantes = plus de mille
    filtrages d'une table de 414 000 lignes. On lit la climatologie de la
    maille UNE fois, puis on applique tous les facteurs d'un coup.
    """
    com, m, mt = communes(), modele(), meta()
    c = com[com.code_insee == code]
    if c.empty:
        return pd.DataFrame()
    cid = int(c.cell_id.iloc[0])
    scenarios = scenarios or SCENARIOS
    annees = list(annees)

    # la climatologie de cette maille, pour le jour choisi
    ref = pd.Timestamp(year=2001, month=mois, day=min(jour, 28))
    clim = climatologie()
    base = clim[(clim.cell_id == cid) & (clim.doy == ref.dayofyear)]
    if base.empty:
        return pd.DataFrame()
    base = base.iloc[0]

    f = facteurs()
    f = f[(f.cell_id == cid) & (f.scenario.isin(scenarios))]

    lignes = []
    for _, r in f[f.annee.isin(annees)].iterrows():
        obs = r.annee <= AN_OBS_MAX
        for var, etiq in (("k", "centre"), ("k_bas", "bas"), ("k_haut", "haut")):
            if obs and var != "k":
                continue           # le passé est mesuré : pas de fourchette
            kk = r.get(var)
            if pd.isna(kk):
                kk = r.k
            d = {"annee": int(r.annee), "scenario": r.scenario,
                 "variante": etiq, "observe": bool(obs)}
            for col in INDICES:
                d[col] = float(base[col]) * float(kk)
            lignes.append(d)
    if not lignes:
        return pd.DataFrame()

    S = pd.DataFrame(lignes)
    S["danger_effis"] = np.digitize(S.fwi, SEUILS).astype(int)
    # les colonnes statiques de la commune, identiques sur toutes les lignes
    for col in c.columns:
        if col in mt["features"] and col not in S.columns:
            S[col] = c[col].iloc[0]
    # ⚠️ le calendrier dépend de l'ANNÉE (jour de semaine, férié) : on le
    # recalcule par ligne plutôt que de le figer.
    cal = pd.DataFrame([calendrier(pd.Timestamp(year=a, month=mois,
                                                day=min(jour, 28)))
                        for a in S.annee])
    for col in cal.columns:
        S[col] = cal[col].to_numpy()
    S["score"] = m.predict_proba(S[mt["features"]].astype(float))[:, 1]
    return S


def chercher(q: str, limite: int = 12) -> pd.DataFrame:
    """Recherche par nom, code postal ou code INSEE.

    Sans accent et sans casse : « st etienne » doit trouver « Saint-Étienne ».
    """
    import unicodedata

    com = communes()
    q = q.strip()
    if not q:
        return com.head(0)

    def sans_accent(s):
        return "".join(c for c in unicodedata.normalize("NFD", str(s))
                       if unicodedata.category(c) != "Mn").lower()

    # ⚠️ Le code INSEE corse contient une LETTRE — « 2A004 » n'est pas
    # `isdigit()`. On tente donc toujours la piste code, avant celle du nom.
    par_code = com[com.code_postal.fillna("").str.startswith(q.upper())
                   | com.code_insee.str.upper().str.startswith(q.upper())]
    if len(par_code):
        return par_code.nlargest(limite, "population")

    # « st etienne » doit trouver « Saint-Étienne » : les abréviations
    # courantes sont dépliées avant comparaison.
    cible = sans_accent(q)
    for court, long in (("st ", "saint "), ("ste ", "sainte "),
                        ("st-", "saint-"), ("ste-", "sainte-")):
        if cible.startswith(court):
            cible = long + cible[len(court):]
    cible = cible.replace(" ", "-")
    nn = com.nom.map(sans_accent).str.replace(" ", "-", regex=False)
    m = pd.concat([com[nn.str.startswith(cible)],
                   com[nn.str.contains(cible, regex=False)]]).drop_duplicates()
    return m.nlargest(limite, "population")


def entete():
    """Bandeau commun, et l'avertissement qui doit apparaître partout."""
    mt = meta()
    st.markdown(
        f"<div style='background:{FOND};border-left:4px solid {ROUGE};"
        f"padding:.6rem 1rem;margin-bottom:1rem'>"
        f"<b style='color:{INK}'>Risque de feu de forêt · France</b><br>"
        f"<span style='color:{MUTED};font-size:.88rem'>"
        f"{mt['modele']} — {mt['test']['lift']:.0f}× le hasard, mesuré sur "
        f"{nb(mt['test']['feux'])} feux de {mt['test']['periode']}"
        f"</span></div>",
        unsafe_allow_html=True)
