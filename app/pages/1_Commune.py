"""Fiche d'une commune — ce qu'elle est, ce qu'elle a vécu, ce qui l'attend.

Trois questions, dans cet ordre :

    1. de quoi cette commune est-elle faite ?     végétation, relief, densité
    2. le danger y a-t-il augmenté ?              FWI décennie par décennie,
                                                  sur 53 ans de mesures
    3. et jusqu'en 2100 ?                          les trois scénarios du GIEC
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import noyau as N

st.set_page_config(page_title="Commune · Risque incendie", page_icon="🔥",
                   layout="wide")
plt.rcParams.update({"figure.facecolor": N.FOND, "axes.facecolor": N.FOND,
                     "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "text.color": N.INK, "xtick.color": N.MUTED,
                     "ytick.color": N.MUTED, "axes.labelcolor": N.INK})
N.entete()

COM = N.communes()
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
        "août", "septembre", "octobre", "novembre", "décembre"]

# ── choisir la commune ───────────────────────────────────────────────────
# L'URL fait autorité : ?commune=13055 ouvre directement la fiche. Ça rend
# les fiches PARTAGEABLES — on peut envoyer un lien vers une commune précise —
# et ça survit à un rechargement, contrairement au seul `session_state`.
params = st.query_params
code = params.get("commune") or st.session_state.get("commune")

c1, c2 = st.columns([3, 1])
q = c1.text_input("Chercher une commune",
                  placeholder="Marseille · 13001 · 2A004 · st etienne")
if q:
    res = N.chercher(q)
    if len(res):
        code = c2.selectbox("Résultat", res.code_insee.to_list(), index=0,
                            format_func=lambda k: (
                                f"{res.set_index('code_insee').nom[k]} "
                                f"({res.set_index('code_insee').dep_code[k]})"))
    else:
        st.info(f"Aucune commune ne correspond à « {q} ».")

if code:
    st.session_state["commune"] = code
    if params.get("commune") != code:
        st.query_params["commune"] = code

if not code:
    st.info("Cherchez une commune ci-dessus, ou passez par la carte.")
    st.stop()

C = COM[COM.code_insee == code]
if C.empty:
    st.error(f"commune {code} inconnue")
    st.stop()
C = C.iloc[0]

st.markdown(f"## {C.nom}")
st.caption(f"{C.code_postal or '—'} · {C.dep_nom} ({C.dep_code}) · "
           f"{C.reg_nom} · code INSEE {C.code_insee}")

# ════════════════════════════════════════════════════════════════════════
#  1. de quoi cette commune est-elle faite ?
# ════════════════════════════════════════════════════════════════════════
g1, g2 = st.columns([1, 1.6])

with g1:
    st.pydeck_chart(pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(latitude=float(C.lat),
                                         longitude=float(C.lon), zoom=9),
        layers=[pdk.Layer("ScatterplotLayer",
                          pd.DataFrame([{"lat": C.lat, "lon": C.lon}]),
                          get_position=["lon", "lat"],
                          get_fill_color=[227, 73, 72], get_radius=2200,
                          stroked=True, get_line_color=[255, 255, 255],
                          line_width_min_pixels=2)]),
        height=260, width='stretch')

with g2:
    a, b, c = st.columns(3)
    a.metric("Population", f"{int(C.population or 0):,}".replace(",", " "))
    b.metric("Superficie", f"{C.superficie_km2:.1f} km²")
    c.metric("Altitude moyenne",
             "—" if pd.isna(C.altitude_moy) else f"{int(C.altitude_moy)} m")
    a.metric("Feux enregistrés (2006-2025)", f"{int(C.feux)}")
    b.metric("Surface brûlée cumulée",
             "—" if pd.isna(C.ha) else f"{C.ha:,.1f} ha".replace(",", " "))
    c.metric("Distance à la côte", f"{C.distance_cote_km:.0f} km")

    # la composition du sol — ce qui fait la différence dans le modèle
    parts = {"maquis": C.part_maquis, "conifères": C.part_coniferes,
             "feuillus": C.part_feuillus, "landes": C.part_landes,
             "mixtes": C.part_melangees, "agricole": C.part_agricole,
             "urbanisé": C.part_artificialise}
    parts = {k: v for k, v in parts.items() if pd.notna(v) and v > 0.005}
    if parts:
        fig, ax = plt.subplots(figsize=(6, 1.5))
        gauche = 0
        coul = {"maquis": "#8b5a2b", "conifères": "#1b5e20", "feuillus": "#66bb6a",
                "landes": "#c0a060", "mixtes": "#43a047", "agricole": "#f0e68c",
                "urbanisé": "#9e9e9e"}
        for k, v in sorted(parts.items(), key=lambda x: -x[1]):
            ax.barh(0, v, left=gauche, color=coul.get(k, N.GRID),
                    edgecolor=N.FOND, linewidth=1.4)
            if v > .07:
                ax.text(gauche + v / 2, 0, f"{k}\n{100 * v:.0f} %", ha="center",
                        va="center", fontsize=7.5,
                        color="white" if k in ("maquis", "conifères") else N.INK)
            gauche += v
        ax.set_xlim(0, 1); ax.axis("off")
        ax.set_title("Occupation du sol (CORINE 2018)", fontsize=9.5,
                     weight="bold", loc="left", color=N.INK)
        st.pyplot(fig, width='stretch')
        plt.close(fig)

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  2. le danger a-t-il augmenté ? — 53 ans de mesures
# ════════════════════════════════════════════════════════════════════════
st.markdown("### Le danger météo a-t-il augmenté ici ?")
st.caption("FWI observé, décennie par décennie, sur la maille météo de la "
           "commune. Ce sont des **mesures**, pas des projections — le CEMS "
           "couvre 1973-2025.")

D = N.decennies()
D = D[D.cell_id == C.cell_id].dropna(subset=["periode"]).sort_values("periode")

if len(D) >= 2:
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.4))
    x = np.arange(len(D))

    ax[0].bar(x, D.fwi_moyen, color=N.ORANGE, edgecolor=N.FOND, linewidth=1.4)
    for i, v in enumerate(D.fwi_moyen):
        ax[0].text(i, v * 1.02, f"{v:.1f}", ha="center", fontsize=8.5,
                   weight="bold")
    ax[0].set_xticks(x); ax[0].set_xticklabels(D.periode, fontsize=8, rotation=20)
    ax[0].set_ylabel("FWI moyen annuel")
    ax[0].set_title("Le niveau de danger", fontsize=10.5, weight="bold",
                    loc="left")

    ax[1].bar(x - .2, D.jours_danger, width=.38, color=N.ORANGE,
              edgecolor=N.FOND, linewidth=1.2, label="danger élevé (FWI > 21,3)")
    ax[1].bar(x + .2, D.jours_tres_eleve, width=.38, color=N.ROUGE,
              edgecolor=N.FOND, linewidth=1.2, label="très élevé (FWI > 38)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(D.periode, fontsize=8, rotation=20)
    ax[1].set_ylabel("jours par an")
    ax[1].set_title("Le nombre de jours à risque", fontsize=10.5,
                    weight="bold", loc="left")
    ax[1].legend(frameon=False, fontsize=8)

    for a in ax:
        a.grid(axis="y", color=N.GRID, lw=.7); a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close(fig)

    d0, d1 = D.iloc[0], D.iloc[-1]
    ev = 100 * (d1.fwi_moyen / d0.fwi_moyen - 1)
    jv = d1.jours_danger - d0.jours_danger
    s1, s2 = st.columns(2)
    s1.metric(f"FWI moyen · {d0.periode} → {d1.periode}",
              f"{d1.fwi_moyen:.2f}", f"{ev:+.0f} %")
    s2.metric("Jours de danger élevé par an",
              f"{d1.jours_danger:.1f}", f"{jv:+.1f} j")
    st.caption(
        "⚠️ Une commune n'est pas une tendance climatique : la variabilité "
        "d'une décennie à l'autre est forte, et une hausse ici ne prouve rien "
        "à elle seule. Sur l'ensemble de la France et sur 20 ans, la pente du "
        "FWI n'est pas significative (p = 0,13).")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  3. et jusqu'en 2100 ?
# ════════════════════════════════════════════════════════════════════════
st.markdown("### Et jusqu'en 2100 ?")
st.caption(
    "Trois trajectoires d'émissions du GIEC. Elles sont **collées jusqu'en "
    "2040** — le CO₂ déjà émis détermine les vingt prochaines années quoi "
    "qu'on fasse — puis l'éventail s'ouvre. C'est la seconde moitié du siècle "
    "que nos choix décident.")

r1, r2, r3 = st.columns([1, 1, 2])
mo = r1.selectbox("Mois de référence", range(1, 13), index=7,
                  format_func=lambda m: MOIS[m - 1])
jo = r2.number_input("Jour", 1, 28, 15, step=1)
fin_an = r3.select_slider("Horizon", [2050, 2070, 2085, 2100], value=2100)

with st.spinner("calcul des trois trajectoires…"):
    S = N.serie_commune(code, range(N.AN_OBS_MIN, fin_an + 1),
                        mois=int(mo), jour=int(jo))

if len(S) and "variante" not in S.columns:
    st.error("Le module de calcul a changé depuis le démarrage. "
             "Relancez l'application (Ctrl+C puis `streamlit run app/Carte.py`).")
    st.stop()

if len(S):
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.2))
    # Le passé ne dépend d'aucun scénario : on prend le premier qui en a,
    # plutôt que d'exiger qu'un scénario précis le porte.
    obs = S[S.observe].drop_duplicates("annee").sort_values("annee")

    for a, col, lab in ((ax[0], "fwi", "FWI"),
                        (ax[1], "score", "risque prédit")):
        # ── ce qui a été mesuré ──
        a.plot(obs.annee, obs[col], lw=.9, color=N.MUTED, alpha=.40, zorder=2)
        a.scatter(obs.annee, obs[col], s=11, color="#4a4a4a", zorder=3,
                  label="mesuré, année par année")

        # ── l'éventail : une trajectoire par scénario ──
        for sc in N.SCENARIOS:
            ctr = S[(~S.observe) & (S.scenario == sc)
                    & (S.variante == "centre")].sort_values("annee")
            bas = S[(~S.observe) & (S.scenario == sc)
                    & (S.variante == "bas")].sort_values("annee")
            haut = S[(~S.observe) & (S.scenario == sc)
                     & (S.variante == "haut")].sort_values("annee")
            if not len(ctr):
                continue
            # la bande n'est dessinée que pour les deux extrêmes : trois
            # bandes superposées deviennent illisibles
            if sc in ("rcp2_6", "rcp8_5") and len(bas) and len(haut):
                a.fill_between(ctr.annee, bas[col].to_numpy(),
                               haut[col].to_numpy(), color=N.COUL_SC[sc],
                               alpha=.10, zorder=1, lw=0)
            a.plot(ctr.annee, ctr[col], lw=2.6, color=N.COUL_SC[sc], zorder=4,
                   label=sc.upper().replace("_", "."))

        a.axvline(N.AN_OBS_MAX + .5, color=N.INK, ls="--", lw=1.1, zorder=6)
        a.set_ylabel(lab)
        a.set_ylim(bottom=0)
        a.grid(color=N.GRID, lw=.7); a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(frameon=False, fontsize=7.5, loc="upper left")

    ax[0].set_title(f"Le danger météo un {jo} {MOIS[mo - 1]}", fontsize=10.5,
                    weight="bold", loc="left")
    ax[1].set_title("Le risque prédit par le modèle", fontsize=10.5,
                    weight="bold", loc="left")
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown(f"##### Le danger météo un {jo} {MOIS[mo - 1]}, selon le scénario")
    rec = obs[obs.annee.between(2011, 2025)]
    cols = st.columns(len(N.SCENARIOS))
    for i, sc in enumerate(N.SCENARIOS):
        v = S[(S.annee == fin_an) & (S.scenario == sc) & (S.variante == "centre")]
        if len(v) and len(rec):
            cols[i].metric(N.ETIQ_SC[sc].split(" — ")[0] + f" · {fin_an}",
                           f"{v.fwi.iloc[0]:.1f}",
                           f"{100 * (v.fwi.iloc[0] / rec.fwi.mean() - 1):+.0f} % "
                           f"vs 2011-2025")
            cols[i].caption(N.ETIQ_SC[sc].split(" — ")[1])

    st.info(
        f"**Comment lire.** À gauche du trait : ce qui a été **mesuré**, un "
        f"{jo} {MOIS[mo - 1]} de chaque année depuis {N.AN_OBS_MIN}. À droite : "
        f"ce que vaudrait un {jo} {MOIS[mo - 1]} **ordinaire** sous le climat "
        f"de chaque année — pas une prévision de la météo de ce jour-là, qui "
        f"est inconnaissable.\n\n"
        f"Les **bandes** portent la variabilité d'une année à l'autre, qui ne "
        f"disparaîtra pas : un {jo} {MOIS[mo - 1]} a valu entre "
        f"{obs.fwi.min():.1f} et {obs.fwi.max():.1f} selon les années. "
        f"Comparer une année exceptionnelle à une trajectoire moyenne n'aurait "
        f"aucun sens.")

sc = "rcp8_5"

# ── la migration de territoire ───────────────────────────────────────────
if pd.notna(C.get(f"risque_{sc}")) and pd.notna(C.risque_fond):
    bouge = C.get(f"cluster_{sc}") != C.cluster_id
    fiable = bool(C.get(f"fiable_{sc}", True))
    st.markdown("##### Le type de territoire")
    if bouge:
        r = C[f"risque_{sc}"] / C.risque_fond
        st.markdown(
            f"Sous le climat de 2041-2055, cette commune rejoint un **type de "
            f"territoire différent** — celui de communes aujourd'hui "
            f"{'plus' if r > 1 else 'moins'} exposées. "
            f"Son risque de fond passerait de **{C.risque_fond:.5%}** à "
            f"**{C[f'risque_{sc}']:.5%}**, soit **×{r:.1f}**.")
        if not fiable:
            st.warning(
                "⚠️ **Hors domaine d'analogie.** La combinaison « climat 2050 "
                "+ végétation actuelle » de cette commune n'a aucun équivalent "
                "réel aujourd'hui. Le chiffre ci-dessus est une extrapolation "
                "du regroupement, pas une comparaison à un territoire existant.")
    else:
        st.markdown(
            f"Cette commune **reste dans le même type de territoire** sous le "
            f"climat de 2041-2055. Son risque de fond est de "
            f"**{C.risque_fond:.5%}** par jour.")
