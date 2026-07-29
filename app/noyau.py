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
FOND = "#fcfcfb"
BLEU, ORANGE, ROUGE, VERT, VIOLET = "#2a78d6", "#eb6834", "#e34948", "#1baf7a", "#4a3aa7"

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


@st.cache_resource(show_spinner=False)
def modele():
    from xgboost import XGBClassifier

    m = XGBClassifier()
    m.load_model(DON / "modele_c.json")
    return m


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
                  variante: str = "k") -> pd.DataFrame:
    """La météo de chaque maille pour cette date. Observée ou projetée.

    ⚠️ Au-delà de 2025 ce n'est PAS une prévision : c'est le cycle saisonnier
    de 2006-2019, dont le niveau est multiplié par le facteur de l'année. Un
    « 2 août 2032 » est donc un 2 août ORDINAIRE sous le climat de 2032.
    """
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


def predire(d: pd.Timestamp, scenario: str = "rcp8_5") -> pd.DataFrame:
    """Score de chaque commune pour cette date."""
    com, m, mt = communes(), modele(), meta()
    X = com.merge(meteo_du_jour(d, scenario), on="cell_id", how="inner")
    for k, v in calendrier(d).items():
        X[k] = v
    X["score"] = m.predict_proba(X[mt["features"]].astype(float))[:, 1]
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
        f"{mt['test']['feux']:,} feux de {mt['test']['periode']}"
        f"</span></div>".replace(",", " "),
        unsafe_allow_html=True)
