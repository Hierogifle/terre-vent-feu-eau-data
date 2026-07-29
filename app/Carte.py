"""Page d'accueil — la carte de chaleur.

    streamlit run app/Carte.py

────────────────────────────────────────────────────────────────────────────
POURQUOI UNE CARTE DE CHALEUR ET NON UN POINT PAR COMMUNE
────────────────────────────────────────────────────────────────────────────
34 734 points sur une carte de France, c'est un aplat. Les communes rurales
sont grandes et peu nombreuses, les communes urbaines minuscules et serrées :
un point par commune donne un poids visuel inverse à la surface réelle.

Une carte de chaleur agrège par zone. Elle montre les MASSIFS — l'arc
méditerranéen, les Landes, la Corse — au lieu de noyer l'œil dans le semis
administratif.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import noyau as N

st.set_page_config(page_title="Carte · Risque incendie", page_icon="🔥",
                   layout="wide", initial_sidebar_state="expanded")
N.entete()

MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
        "août", "septembre", "octobre", "novembre", "décembre"]

# ════════════════════════════════════════════════════════════════════════
#  les réglages
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Quoi afficher")
    couche = st.radio(
        "Couche", ["Risque prédit", "Danger météo (FWI)"], index=0,
        help="Le risque croise météo, végétation et relief — c'est la sortie "
             "du modèle. Le FWI est l'indice météo officiel EFFIS, sans "
             "modèle. Les comparer montre ce que le territoire ajoute.")

    st.markdown("### Quand")
    mode = st.radio("Période",
                    ["Une date précise", "Une année entière", "Comparer deux dates"],
                    index=0, label_visibility="collapsed")


    def choix_date(cle: str, defaut_an: int):
        """⚠️ Un SÉLECTEUR de date, pas un curseur. On ne choisit pas un jour
        en faisant glisser un curseur sur 28 000 positions."""
        c1, c2 = st.columns(2)
        a = c1.number_input("Année", N.AN_OBS_MIN, N.AN_PROJ_MAX, defaut_an,
                            step=1, key=f"an{cle}")
        m = c2.selectbox("Mois", range(1, 13), index=7,
                         format_func=lambda x: MOIS[x - 1], key=f"mo{cle}")
        j = st.number_input("Jour", 1, 31, 15, step=1, key=f"jo{cle}")
        try:
            return pd.Timestamp(year=int(a), month=int(m), day=int(j))
        except ValueError:
            st.caption(f"{MOIS[m - 1]} n'a pas {j} jours — ramené au 28")
            return pd.Timestamp(year=int(a), month=int(m), day=28)


    date_b = None
    if mode == "Une date précise":
        date = choix_date("a", 2025)
    elif mode == "Une année entière":
        an = st.slider("Année", N.AN_OBS_MIN, N.AN_PROJ_MAX, 2025, step=1)
        date = pd.Timestamp(year=int(an), month=8, day=15)
        st.caption("l'année est représentée par le 15 août, "
                   "cœur de la saison de feu")
    else:
        st.caption("**Référence**")
        date = choix_date("a", 1990)
        st.caption("**Comparaison**")
        date_b = choix_date("b", 2100)

    scenario = "rcp8_5"
    if max(date.year, date_b.year if date_b else 0) > N.AN_OBS_MAX:
        scenario = st.radio(
            "Scénario climatique", N.SCENARIOS, index=2,
            format_func=lambda s: N.ETIQ_SC[s].split(" — ")[0],
            help="RCP = trajectoire d'émissions du GIEC. Le chiffre est le "
                 "forçage radiatif en 2100, en W/m².\n\n"
                 + "\n\n".join(f"**{v.split(' — ')[0]}** : {v.split(' — ')[1]}"
                               for v in N.ETIQ_SC.values()))
        st.caption(N.ETIQ_SC[scenario].split(" — ")[1])

    st.markdown("### Recherche")
    q = st.text_input("Commune ou code postal", placeholder="Marseille · 13001",
                      label_visibility="collapsed")

# ════════════════════════════════════════════════════════════════════════
R = N.predire(date, scenario)
projete = date.year > N.AN_OBS_MAX


def carte(D: pd.DataFrame, titre: str, vue=None):
    """Une carte de chaleur, aux couleurs EFFIS."""
    if couche.startswith("Risque"):
        D = D.assign(poids=D.score / D.score.max())
    else:
        D = D.assign(poids=(D.fwi / 50).clip(0, 1))
    st.markdown(f"**{titre}**")
    st.pydeck_chart(pdk.Deck(
        map_style="light",
        initial_view_state=vue or pdk.ViewState(latitude=46.6, longitude=2.4,
                                                zoom=4.5),
        layers=[pdk.Layer(
            "HeatmapLayer", D[["lon", "lat", "poids"]],
            get_position=["lon", "lat"], get_weight="poids",
            radius_pixels=30, intensity=1.0, threshold=0.03,
            color_range=[N.hex_rgb(c) for c in N.COUL_EFFIS], opacity=.85)]),
        height=420, width="stretch")


# ── mode comparaison : deux cartes côte à côte, même échelle ────────────
if date_b is not None:
    Rb = N.predire(date_b, scenario)
    st.markdown(f"## {date.strftime('%d/%m/%Y')} vs {date_b.strftime('%d/%m/%Y')}")

    g, d_ = st.columns(2)
    with g:
        carte(R, date.strftime("%d %B %Y"))
    with d_:
        carte(Rb, date_b.strftime("%d %B %Y"))

    st.markdown("##### Ce qui change entre les deux")
    j = R[["code_insee", "nom", "dep_nom", "fwi", "score", "danger_effis"]].merge(
        Rb[["code_insee", "fwi", "score", "danger_effis"]], on="code_insee",
        suffixes=("_a", "_b"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("FWI moyen", f"{j.fwi_b.mean():.1f}",
              f"{j.fwi_b.mean() - j.fwi_a.mean():+.1f}")
    m2.metric("Communes en danger ≥ élevé",
              f"{(j.danger_effis_b >= 3).sum():,}".replace(",", " "),
              f"{int((j.danger_effis_b >= 3).sum() - (j.danger_effis_a >= 3).sum()):+,}"
              .replace(",", " "))
    m3.metric("Communes dont le FWI monte",
              f"{(j.fwi_b > j.fwi_a).sum():,}".replace(",", " "),
              f"{100 * (j.fwi_b > j.fwi_a).mean():.0f} % du territoire")
    m4.metric("Risque médian", f"×{(j.score_b / j.score_a).median():.2f}")

    j["hausse"] = j.fwi_b - j.fwi_a
    st.markdown("Les 10 communes où le danger progresse le plus :")
    t = j.nlargest(10, "hausse")[["nom", "dep_nom", "fwi_a", "fwi_b", "hausse"]]
    st.dataframe(t.rename(columns={
        "nom": "Commune", "dep_nom": "Département",
        "fwi_a": f"FWI {date.year}", "fwi_b": f"FWI {date_b.year}",
        "hausse": "Écart"}).round(1), width="stretch", hide_index=True)
    st.stop()

if projete:
    st.warning(
        f"**{date.strftime('%d %B %Y')} — ce n'est pas une prévision météo.** "
        f"Personne ne connaît le temps qu'il fera ce jour-là. La carte montre "
        f"ce que vaudrait un **{date.day} {MOIS[date.month - 1]} ordinaire "
        f"sous le climat de {date.year}** : le cycle saisonnier vient des "
        f"observations 2006-2019, seul son niveau est décalé par le "
        f"réchauffement projeté ({scenario.upper().replace('_', '.')}).")

a, b, c, d = st.columns(4)
a.metric("FWI moyen", f"{R.fwi.mean():.1f}")
b.metric("FWI maximum", f"{R.fwi.max():.1f}")
c.metric("Communes en danger ≥ élevé",
         f"{(R.danger_effis >= 3).sum():,}".replace(",", " "))
d.metric("Communes en danger extrême",
         f"{(R.danger_effis >= 5).sum():,}".replace(",", " "))

# ── la carte ─────────────────────────────────────────────────────────────
if couche.startswith("Risque"):
    poids = R.score
    # normalisé sur la journée : la carte répond à « où regarder aujourd'hui »
    R["poids"] = poids / poids.max()
    legende = "risque prédit par le modèle, relatif à la journée"
else:
    R["poids"] = (R.fwi / 50).clip(0, 1)
    legende = "FWI — 50 correspond au seuil « extrême » d'EFFIS"

couches = [pdk.Layer(
    "HeatmapLayer", R[["lon", "lat", "poids"]],
    get_position=["lon", "lat"], get_weight="poids",
    radius_pixels=34, intensity=1.0, threshold=0.03,
    color_range=[N.hex_rgb(c) for c in N.COUL_EFFIS], opacity=.85)]

# la commune cherchée, épinglée par-dessus
trouvees = N.chercher(q) if q else None
vue = pdk.ViewState(latitude=46.6, longitude=2.4, zoom=4.7)
if trouvees is not None and len(trouvees):
    sel = trouvees.iloc[0]
    ligne = R[R.code_insee == sel.code_insee]
    couches.append(pdk.Layer(
        "ScatterplotLayer", ligne, get_position=["lon", "lat"],
        get_fill_color=[20, 20, 20], get_radius=7000, opacity=1,
        stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2))
    vue = pdk.ViewState(latitude=float(sel.lat), longitude=float(sel.lon), zoom=8)

st.pydeck_chart(pdk.Deck(map_style="light", layers=couches,
                         initial_view_state=vue), width='stretch')
st.caption(f"Carte de chaleur — {legende}. "
           f"Échelle de couleurs EFFIS : vert → jaune → orange → rouge → noir.")

# ── le résultat de la recherche ─────────────────────────────────────────
if q:
    if trouvees is None or not len(trouvees):
        st.info(f"Aucune commune ne correspond à « {q} ».")
    else:
        st.markdown(f"##### {len(trouvees)} résultat(s) pour « {q} »")
        for _, t in trouvees.iterrows():
            l = R[R.code_insee == t.code_insee]
            if l.empty:
                continue
            l = l.iloc[0]
            cl = int(l.danger_effis)
            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
            c1.markdown(f"**{t.nom}**  \n"
                        f"<span style='color:{N.MUTED};font-size:.85rem'>"
                        f"{t.code_postal or ''} · {t.dep_nom}</span>",
                        unsafe_allow_html=True)
            c2.markdown(f"FWI **{l.fwi:.1f}**  \n"
                        f"<span style='color:{N.COUL_EFFIS[cl]};font-weight:600'>"
                        f"{N.CLASSES[cl]}</span>", unsafe_allow_html=True)
            c3.markdown(f"Rang  \n**{100 * l.rang:.1f}ᵉ** percentile")
            if c4.button("Voir la fiche →", key=f"f{t.code_insee}",
                         width='stretch'):
                st.session_state["commune"] = t.code_insee
                st.switch_page("pages/1_Commune.py")

st.divider()
st.markdown(f"""
<span style='color:{N.MUTED};font-size:.85rem'>
Le <b>risque prédit</b> est un classement, pas une probabilité : la
calibration disponible a été ajustée sur un autre modèle et une autre période,
elle serait fausse d'un facteur ~2. Le <b>FWI</b> est l'indice officiel, il se
lit directement sur l'échelle EFFIS.
</span>""", unsafe_allow_html=True)
