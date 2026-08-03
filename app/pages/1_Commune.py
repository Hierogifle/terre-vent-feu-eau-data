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
    a.metric("Population", N.nb(int(C.population or 0)))
    b.metric("Superficie", f"{N.dec(C.superficie_km2)} km²")
    c.metric("Altitude moyenne",
             "—" if pd.isna(C.altitude_moy) else f"{int(C.altitude_moy)} m")
    a.metric("Feux enregistrés (2006-2025)", f"{int(C.feux)}")
    b.metric("Surface brûlée cumulée",
             "—" if pd.isna(C.ha) else f"{N.nb(C.ha, 1)} ha")
    c.metric("Distance à la côte", f"{N.nb(C.distance_cote_km)} km")

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
           "commune. Ce sont des mesures, pas des projections : le CEMS "
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
              N.dec(d1.fwi_moyen, 2), f"{ev:+.0f} %")
    s2.metric("Jours de danger élevé par an",
              N.dec(d1.jours_danger), f"{jv:+.1f} j")
    # ⚠️ CES CHIFFRES SONT LUS, PAS RECOPIÉS. Cette légende annonçait
    # « +45 %, p < 0,0001 » et « p = 0,13 », deux valeurs fausses restées en
    # place parce qu'elles vivaient dans une chaîne de caractères.
    _t = N.tendances()
    _fwi = _t[_t.serie == "FWI moyen annuel"].iloc[0]
    st.caption(
        f"Une commune ne fait pas une tendance climatique : la variabilité "
        f"d'une décennie à l'autre est forte, et une hausse ici ne prouve "
        f"rien à elle seule. Sur l'ensemble de la France en revanche, la "
        f"hausse est établie : {N.dec(_fwi.variation_pct, 0)} % de FWI moyen "
        f"entre {int(_fwi.an_min)} et {int(_fwi.an_max)}, avec "
        f"p = {_fwi.p:.1e}. Il a fallu cinquante ans de mesures pour "
        f"l'établir.")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  3. et jusqu'en 2100 ?
# ════════════════════════════════════════════════════════════════════════
st.markdown("### Et jusqu'en 2100 ?")
st.caption("Trois trajectoires d'émissions du GIEC, appliquées au danger "
           "météo de cette commune.")

with st.expander("Que veulent dire RCP 2.6, 4.5 et 8.5 ?"):
    st.markdown("""
Un RCP, pour *Representative Concentration Pathway*, est une trajectoire
d'émissions de gaz à effet de serre. Le chiffre est le **forçage radiatif en
2100**, en watts par mètre carré : l'énergie que la Terre reçoit en trop du
fait de ces émissions. Plus il est élevé, plus elle se réchauffe.

Ce ne sont pas des prévisions. Ce sont des hypothèses cohérentes, faites pour
que tous les laboratoires du monde calculent la même chose et puissent
comparer leurs résultats.
""")
    for sc in N.SCENARIOS:
        d = N.RCP[sc]
        st.markdown(
            f"**{d['titre']}** — {d['forcage']}  \n"
            f"{d['hypothese']}  \n"
            f"Réchauffement attendu : {d['rechauffement']}.  \n"
            f"<span style='color:{N.MUTED}'>{d['credibilite']}</span>",
            unsafe_allow_html=True)
    st.markdown(f"""
**Pourquoi les trois courbes se confondent avant {N.AN_DIVERGENCE}.** Le CO₂
déjà émis fixe les vingt prochaines années quoi qu'on fasse, et sur cette
période l'écart entre trajectoires reste plus petit que la variabilité
naturelle d'une année à l'autre. Sur nos propres facteurs, RCP 2.6 passe même
au-dessus de RCP 8.5 en 2030 dans 59 % des mailles, l'écart moyen entre les
deux valant −0,015. Ils ne se séparent vraiment qu'à partir de 2046.

Ce n'est donc pas une anomalie de calcul. C'est la seconde moitié du siècle
que nos choix décident.
""")

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
    obs = S[S.observe].drop_duplicates("annee").sort_values("annee")

    fig, a = plt.subplots(figsize=(13.5, 5))

    # ⚠️ ÉCHELLE FIXE, IDENTIQUE POUR TOUTES LES COMMUNES.
    # Un axe ajusté à chaque commune leur donnait toutes la même allure :
    # Lille et Bormes-les-Mimosas paraissaient aussi menacées l'une que
    # l'autre. On cale l'axe sur les classes officielles EFFIS, qui sont
    # absolues et se lisent d'un coup d'œil.
    HAUT = 60
    bornes = [0] + N.SEUILS + [HAUT]
    for i in range(len(bornes) - 1):
        a.axhspan(bornes[i], min(bornes[i + 1], HAUT),
                  color=N.COUL_EFFIS[i], alpha=.13, lw=0, zorder=0)
        if bornes[i] < HAUT:
            a.text(N.AN_OBS_MIN + .5, (bornes[i] + min(bornes[i + 1], HAUT)) / 2,
                   N.CLASSES[i], fontsize=7.5, color=N.MUTED, va="center")

    a.plot(obs.annee, obs.fwi, lw=.9, color=N.MUTED, alpha=.45, zorder=2)
    a.scatter(obs.annee, obs.fwi, s=13, color="#3a3a3a", zorder=3,
              label="mesuré, année par année")

    for sc in N.SCENARIOS:
        p = {v: S[(~S.observe) & (S.scenario == sc) & (S.variante == v)]
             .sort_values("annee") for v in ("centre", "bas", "haut")}
        if not len(p["centre"]):
            continue
        if len(p["bas"]) and len(p["haut"]):
            a.fill_between(p["centre"].annee, p["bas"].fwi.to_numpy(),
                           p["haut"].fwi.to_numpy(), color=N.COUL_SC[sc],
                           alpha=.16, zorder=1, lw=0)
        a.plot(p["centre"].annee, p["centre"].fwi, lw=2.6,
               color=N.COUL_SC[sc], zorder=4,
               label=sc.upper().replace("_", "."))

    a.axvline(N.AN_OBS_MAX + .5, color=N.INK, ls="--", lw=1.1, zorder=6)
    a.text(N.AN_OBS_MAX + 1, HAUT * .96, " projeté →", fontsize=8.5,
           color=N.MUTED, va="top")

    # la zone où les scénarios ne se distinguent pas encore
    if fin_an > N.AN_OBS_MAX:
        a.axvspan(N.AN_OBS_MAX + .5, N.AN_DIVERGENCE, color="#8a8a8a",
                  alpha=.10, lw=0, zorder=1)
        a.text((N.AN_OBS_MAX + N.AN_DIVERGENCE) / 2, HAUT * .06,
               "les trois scénarios\nsont indiscernables", fontsize=8,
               color=N.MUTED, ha="center", va="bottom", style="italic")

    a.set_ylim(0, HAUT)
    a.set_xlim(N.AN_OBS_MIN, fin_an)
    a.set_ylabel("FWI, indice de danger météo")
    a.set_title(f"Un {jo} {MOIS[mo - 1]} ordinaire, de {N.AN_OBS_MIN} à "
                f"{fin_an}", fontsize=11.5, weight="bold", loc="left")
    a.grid(color=N.GRID, lw=.7, axis="x")
    a.set_axisbelow(True)
    a.spines[["top", "right"]].set_visible(False)
    a.legend(frameon=False, fontsize=9, loc="upper left", ncols=4)
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

    st.info(f"""
À gauche du trait pointillé, ce qui a été mesuré : un {jo} {MOIS[mo - 1]} de
chaque année depuis {N.AN_OBS_MIN}. À droite, ce que vaudrait un
{jo} {MOIS[mo - 1]} ordinaire sous le climat de chaque année. Ce n'est pas une
prévision de la météo de ce jour-là, qui restera inconnaissable.

Les bandes colorées portent la variabilité d'une année sur l'autre, qui ne
disparaîtra pas. Ici, un {jo} {MOIS[mo - 1]} a valu entre
{N.dec(obs.fwi.min())} et {N.dec(obs.fwi.max())} selon les années : comparer
une année exceptionnelle à une trajectoire moyenne n'aurait pas de sens.

L'échelle va de 0 à {HAUT} pour toutes les communes, avec les classes
officielles EFFIS en fond. Vous pouvez donc comparer deux fiches directement.
""")

sc = "rcp8_5"

# ── la migration de territoire ───────────────────────────────────────────
if pd.notna(C.get(f"risque_{sc}")) and pd.notna(C.risque_fond):
    bouge = C.get(f"cluster_{sc}") != C.cluster_id
    fiable = bool(C.get(f"fiable_{sc}", True))
    st.markdown("##### Le type de territoire")
    st.caption("Les 34 734 communes sont regroupées en 30 types, sur leur "
               "végétation, leur relief et leur climatologie. Jamais sur les "
               "feux. Un réchauffement peut faire passer une commune d'un "
               "type à un autre.")
    if bouge:
        r = C[f"risque_{sc}"] / C.risque_fond
        st.markdown(
            f"Sous le climat de 2041-2055 (RCP 8.5), cette commune rejoint un "
            f"type de territoire différent, celui de communes aujourd'hui "
            f"{'plus' if r > 1 else 'moins'} exposées. Son risque de fond "
            f"passerait de {N.pct(C.risque_fond, 4)} à "
            f"{N.pct(C[f'risque_{sc}'], 4)} par jour, soit **{N.dec(r)} fois** "
            f"{'plus' if r > 1 else 'moins'}.")
        if not fiable:
            st.warning(
                "**Hors domaine d'analogie.** La combinaison « climat 2050 "
                "+ végétation actuelle » de cette commune n'a aucun équivalent "
                "réel aujourd'hui. Le chiffre ci-dessus est une extrapolation "
                "du regroupement, pas une comparaison à un territoire existant.")
    else:
        st.markdown(
            f"Cette commune reste dans le même type de territoire sous le "
            f"climat de 2041-2055. Son risque de fond est de "
            f"{N.pct(C.risque_fond, 4)} par jour.")
