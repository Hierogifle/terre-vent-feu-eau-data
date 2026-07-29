"""Terre, Vent, Feu, Eau, Data — application de risque incendie.

    streamlit run app/risque.py

────────────────────────────────────────────────────────────────────────────
CE QUE L'APPLICATION FAIT TOURNER, ET POURQUOI CELUI-LÀ
────────────────────────────────────────────────────────────────────────────
Le **modèle C**, 41 features physiques. Pas le meilleur — le modèle A fait
93,8× le hasard contre 63,7× — mais **le seul déployable**.

Le modèle A tire 29 % de son importance de l'historique récent des feux, et la
BDIFF ne publie pas l'année en cours : les données 2026 ne sortiront qu'au
printemps 2027. Un modèle qui a besoin des feux de la semaine dernière ne peut
pas prédire la semaine prochaine.

Le modèle C n'utilise que ce qui est connu à l'avance : les indices météo
(prévus par EFFIS à 9 jours), la végétation et le relief.

⚠️ Le score affiché est un RANG, pas une probabilité. La calibration a été
ajustée sur le modèle A et sur une période à taux de base différent : afficher
« 3 % de risque » serait faux d'un facteur ~2. Un classement est honnête et
répond à la vraie question opérationnelle — où regarder en premier.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RACINE = Path(__file__).parent
DON = RACINE / "donnees"

# nomenclature EFFIS, couleurs officielles
CLASSES = ["très faible", "faible", "modéré", "élevé", "très élevé", "extrême"]
COUL = ["#84F07F", "#FFEB3C", "#FFB00C", "#FA4F00", "#B40000", "#280923"]

st.set_page_config(page_title="Risque incendie · TVFED", page_icon="🔥",
                   layout="wide")


# ════════════════════════════════════════════════════════════════════════
#  chargement — mis en cache, l'artefact ne change pas
# ════════════════════════════════════════════════════════════════════════
@st.cache_data
def charger():
    com = pd.read_parquet(DON / "communes.parquet")
    met = pd.read_parquet(DON / "meteo.parquet")
    met["date"] = pd.to_datetime(met.date)
    meta = json.loads((DON / "meta.json").read_text(encoding="utf-8"))
    return com, met, meta


@st.cache_resource
def charger_modele():
    from xgboost import XGBClassifier

    m = XGBClassifier()
    m.load_model(DON / "modele_c.json")
    return m


def calendrier(d: pd.Timestamp) -> dict:
    """Les 11 features de date. Row-local : aucun risque de fuite."""
    import holidays

    fr = holidays.France(years=d.year)
    doy, mois = d.dayofyear, d.month
    return {
        "doy": doy, "mois": mois, "jour_semaine": d.dayofweek,
        "est_weekend": int(d.dayofweek >= 5),
        "est_ferie": int(d.date() in fr),
        "est_14_juillet": int(mois == 7 and d.day == 14),
        "est_15_aout": int(mois == 8 and d.day == 15),
        "sin_doy": np.sin(2 * np.pi * doy / 365.25),
        "cos_doy": np.cos(2 * np.pi * doy / 365.25),
        "sin_mois": np.sin(2 * np.pi * mois / 12),
        "cos_mois": np.cos(2 * np.pi * mois / 12),
    }


def predire(com, met, meta, d):
    """Assemble les 41 features pour une date et rend le score de chaque commune."""
    m = charger_modele()
    jour = met[met.date == d]
    if jour.empty:
        return None
    X = com.merge(jour.drop(columns=["date"]), on="cell_id", how="inner")
    for k, v in calendrier(d).items():
        X[k] = v
    feats = meta["features"]
    manquantes = [f for f in feats if f not in X.columns]
    if manquantes:
        st.error(f"features absentes de l'artefact : {manquantes}")
        st.stop()
    X["score"] = m.predict_proba(X[feats].astype(float))[:, 1]
    # rang plutôt que probabilité — voir l'en-tête du module
    X["rang"] = X.score.rank(pct=True)
    X["classe"] = np.clip((X.rang * 6).astype(int), 0, 5)
    return X


# ════════════════════════════════════════════════════════════════════════
COM, MET, META = charger()
JOURS = sorted(MET.date.unique())

st.title("🔥 Risque de feu de forêt — France")
st.caption(
    f"{META['modele']} · {META['test']['lift']:.0f}× le hasard sur "
    f"{META['test']['feux']:,} feux de {META['test']['periode']}".replace(",", " "))

t1, t2, t3 = st.tabs(["Carte du jour", "Projection 2050", "Le modèle"])

# ── ONGLET 1 ────────────────────────────────────────────────────────────
with t1:
    c1, c2 = st.columns([2, 1])
    with c1:
        d = st.select_slider(
            "Date", options=JOURS, value=JOURS[len(JOURS) // 2],
            format_func=lambda x: pd.Timestamp(x).strftime("%d %B %Y"))
    with c2:
        seuil = st.slider("Communes affichées (% les mieux notées)", 1, 100, 20)

    R = predire(COM, MET, META, pd.Timestamp(d))
    if R is None:
        st.warning("pas de météo pour cette date")
        st.stop()

    montre = R[R.rang >= 1 - seuil / 100]
    a, b, c, e = st.columns(4)
    a.metric("Communes affichées", f"{len(montre):,}".replace(",", " "))
    b.metric("FWI moyen du jour", f"{R.fwi.mean():.1f}")
    c.metric("FWI maximum", f"{R.fwi.max():.1f}")
    e.metric("Communes en danger EFFIS ≥ élevé",
             f"{(R.danger_effis >= 3).sum():,}".replace(",", " "))

    import pydeck as pdk

    montre = montre.copy()
    montre["couleur"] = montre.classe.map(
        lambda i: [int(COUL[i][j:j + 2], 16) for j in (1, 3, 5)])
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=46.6, longitude=2.5, zoom=4.9),
        layers=[pdk.Layer(
            "ScatterplotLayer", montre,
            get_position=["lon", "lat"], get_fill_color="couleur",
            get_radius=2600, pickable=True, opacity=.75)],
        tooltip={"text": "{nom} ({dep_code})\nFWI {fwi}\nrang {rang}"}))

    st.markdown("##### Les 15 communes les plus exposées ce jour-là")
    top = R.nlargest(15, "score")[
        ["nom", "dep_code", "fwi", "danger_effis", "part_maquis",
         "part_combustible", "rang"]].copy()
    top["part_maquis"] = (100 * top.part_maquis).round(1)
    top["part_combustible"] = (100 * top.part_combustible).round(1)
    top["rang"] = (100 * top.rang).round(2)
    st.dataframe(top.rename(columns={
        "nom": "Commune", "dep_code": "Dép", "fwi": "FWI",
        "danger_effis": "Danger EFFIS", "part_maquis": "Maquis %",
        "part_combustible": "Combustible %", "rang": "Percentile"}),
        use_container_width=True, hide_index=True)

    st.info(
        "Le **percentile** situe la commune parmi les 34 734 du jour. Ce n'est "
        "volontairement pas une probabilité : la calibration disponible a été "
        "ajustée sur un autre modèle et une autre période, elle serait fausse "
        "d'un facteur ~2. Un classement répond à la question opérationnelle — "
        "où regarder en premier.")

# ── ONGLET 2 ────────────────────────────────────────────────────────────
with t2:
    st.markdown("""
Le clustering range les communes par ressemblance physique, et trois de ses
vingt variables sont climatiques. En les remplaçant par leur valeur projetée
pour **2041-2055** et en demandant au *même* clustering où ranger la commune,
certaines changent de groupe — et héritent du risque qu'on observe
**aujourd'hui** chez leurs nouveaux semblables.

C'est la *substitution espace-temps*. La végétation, le relief et la prévention
sont figés : **seul le climat bouge**.
""")
    sc = st.radio("Scénario", ["rcp4_5", "rcp8_5"], horizontal=True,
                  format_func=lambda s: s.upper().replace("_", "."))
    M = COM.dropna(subset=[f"risque_{sc}"]).copy()
    M["evolution"] = M[f"risque_{sc}"] / M.risque_fond

    a, b, c = st.columns(3)
    a.metric("Risque de fond moyen",
             f"{M[f'risque_{sc}'].mean():.5%}",
             f"{100 * (M[f'risque_{sc}'].mean() / M.risque_fond.mean() - 1):+.0f} %")
    b.metric("Communes changeant de catégorie",
             f"{(M[f'cluster_{sc}'] != M.cluster_id).sum():,}".replace(",", " "))
    c.metric("Dont hors analogie ⚠",
             f"{(~M[f'fiable_{sc}'].astype(bool)).sum():,}".replace(",", " "))

    M["couleur"] = M.evolution.map(
        lambda r: [40, 120, 200] if r < .99 else
                  [200, 200, 195] if r < 1.01 else
                  [247, 197, 159] if r < 2 else
                  [235, 104, 52] if r < 5 else
                  [227, 73, 72] if r < 20 else [139, 26, 26])
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=46.6, longitude=2.5, zoom=4.9),
        layers=[pdk.Layer("ScatterplotLayer", M,
                          get_position=["lon", "lat"], get_fill_color="couleur",
                          get_radius=2600, pickable=True, opacity=.75)],
        tooltip={"text": "{nom} ({dep_code})\nrisque ×{evolution}"}))

    st.warning(
        "**Ce que ces chiffres disent** : à végétation constante, le climat de "
        "2050 rapproche ces communes du profil de risque de territoires "
        "aujourd'hui plus exposés.\n\n"
        "**Ce qu'ils ne disent pas** : qu'elles auront ce risque. Le maquis ne "
        "pousse pas en dix ans. Et pour 7,2 % des communes, la combinaison "
        "climat futur + végétation actuelle n'a **aucun équivalent réel** "
        "aujourd'hui — le chiffre y est une extrapolation.\n\n"
        "À cet horizon, **RCP 4.5 et RCP 8.5 ne se départagent pas** : les "
        "forçages ne divergent qu'après 2050. À lire comme une fourchette.")

# ── ONGLET 3 ────────────────────────────────────────────────────────────
with t3:
    st.markdown(f"""
### Le modèle

**{META['modele']}**

{META['pourquoi']}

| | PR-AUC | lift | déployable |
|---|---|---|---|
| A · 52 features | {META['modele_a']['pr_auc']} | {META['modele_a']['lift']}× | non |
| **C · physique pur** | **{META['test']['pr_auc']}** | **{META['test']['lift']}×** | **oui** |

Mesuré sur **{META['test']['lignes']:,} communes-jours** de
{META['test']['periode']}, jamais utilisés pendant la construction.

### Ce qui pèse dans la décision
""".replace(",", " "))
    imp = pd.read_csv(DON / "importances_c.csv", index_col=0).squeeze("columns")
    st.bar_chart((100 * imp.head(12)).rename("importance (%)"))

    st.markdown("""
La feature n°1 est **la part de maquis**, devant le danger EFFIS et l'ERC.
C'est la thèse du projet, mesurée : *la météo dit quand, le territoire dit où.*

### Les limites, en clair

- **la surface brûlée n'est pas prédictible** — R² de 0,14, moins bon que de
  toujours annoncer la médiane. La taille dépend surtout de ce qui se passe
  *après* le départ : vent, délai d'intervention, relief. En revanche
  « sera-ce un grand feu (> 5 ha) ? » se prédit à 0,77 de ROC-AUC ;
- **une commune-jour n'est pas un incendie** — un feu traversant cinq communes
  compte cinq fois ;
- **la portée est celle de la prévision météo**, soit ~9 jours avec EFFIS ;
- **le modèle suppose stable tout ce qui n'est pas la météo** : prévention,
  pratiques agricoles, déprise rurale. Mesuré sur 20 ans, cette dérive
  résiduelle n'est pas significative (p = 0,27), mais elle n'est pas nulle.
""")
