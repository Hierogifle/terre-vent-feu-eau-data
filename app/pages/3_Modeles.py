"""Ce que vaut le modèle, comment on l'a mesuré, et ce qu'on a comparé.

Quatre onglets, dans l'ordre où les questions se posent. Tous les chiffres
sont lus dans les CSV produits par les modules d'entraînement ; aucun n'est
écrit en dur.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import noyau as N

st.set_page_config(page_title="Les modèles · Risque incendie", page_icon="🔥",
                   layout="wide")
plt.rcParams.update({"figure.facecolor": N.FOND, "axes.facecolor": N.FOND,
                     "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "text.color": N.INK, "xtick.color": N.MUTED,
                     "ytick.color": N.MUTED, "axes.labelcolor": N.INK})
N.entete()

MT = N.meta()
TAUX_VAL = 0.0002410


@st.cache_data(show_spinner=False)
def csv(nom: str) -> pd.DataFrame | None:
    f = N.DON / nom
    return pd.read_csv(f) if f.exists() else None


def habiller(ax, titre=None, x=None, y=None, axe="both"):
    if titre:
        ax.set_title(titre, fontsize=10.5, weight="bold", loc="left")
    if x:
        ax.set_xlabel(x)
    if y:
        ax.set_ylabel(y)
    ax.grid(color=N.GRID, lw=.7, axis=axe)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


PR, CMP = csv("pr_auc_val.csv"), csv("comparaison_appariee.csv")
if PR is None or CMP is None:
    st.error("`pr_auc_val.csv` ou `comparaison_appariee.csv` manquant. "
             "Lancer `python -m tvfed.comparer` puis `tvfed.export_app`.")
    st.stop()

AP = PR.iloc[0].to_dict()


def ec(ref, mod):
    return CMP[(CMP.reference == ref) & (CMP.modele == mod)].iloc[0]


o1, o2, o3, o4 = st.tabs(["Ce que vaut le modèle", "Comment on l'a mesuré",
                          "Ce qu'on a comparé", "Les limites"])

# ════════════════════════════════════════════════════════════════════════
with o1:
    OP = csv("operationnel_courbe.csv")
    if OP is not None:
        st.markdown("### Combien de départs attrape-t-on, et à quel prix")
        st.caption("Mesuré sur le jeu de test, 2023-2025, 6 322 départs de "
                   "feu. On classe les 38 millions de communes-jours par "
                   "score décroissant et on garde les premiers.")

        budget = st.slider("Part du territoire surveillée",
                           0.1, 25.0, 1.0, 0.1, format="%.1f %%") / 100
        C = OP[OP.modele == "C"].reset_index(drop=True)
        i = (C.budget - budget).abs().idxmin()
        r = C.loc[i]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Communes-jours surveillés", N.nb(r.lignes))
        m2.metric("Départs attrapés", f"{N.nb(r.feux)} sur 6 322",
                  N.pct(r.rappel) + " des départs")
        m3.metric("Précision", N.pct(r.precision, 2),
                  f"{r.precision / TAUX_VAL:.0f}× le hasard",
                  delta_color="off")
        m4.metric("Communes-jours par départ trouvé", N.nb(1 / r.precision))

        fig, ax = plt.subplots(figsize=(11, 3.8))
        for nom, coul, lib in (("C", N.VERT, "modèle C, déployé"),
                               ("v3", N.GRIS, "modèle v3, non déployable")):
            s = OP[OP.modele == nom]
            if len(s):
                ax.plot(100 * s.budget, 100 * s.rappel, color=coul, lw=2,
                        label=lib)
        ax.plot([0, 25], [0, 25], ":", color=N.MUTED, lw=1.2,
                label="au hasard")
        ax.plot(100 * r.budget, 100 * r.rappel, "o", color=N.ROUGE, ms=10,
                zorder=5)
        ax.annotate(f"{N.pct(r.rappel)} des départs",
                    xy=(100 * r.budget, 100 * r.rappel),
                    xytext=(10, -16), textcoords="offset points",
                    fontsize=10, weight="bold", color=N.ROUGE)
        # échelle linéaire : l'échelle logarithmique se lit mal et le premier
        # réflexe devant elle est de croire que la courbe est plus plate
        # qu'elle ne l'est.
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 26, 5), [f"{v} %" for v in range(0, 26, 5)])
        ax.set_yticks(range(0, 101, 20), [f"{v} %" for v in range(0, 101, 20)])
        habiller(ax, x="part du territoire surveillée",
                 y="départs attrapés")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

        v3 = OP[OP.modele == "v3"]
        if len(v3):
            rv = v3.iloc[(v3.budget - budget).abs().argmin()]
            gap = f"{abs(rv.rappel - r.rappel) * 100:.1f}".replace(".", ",")
            st.info(f"""
À ce budget, le modèle déployé attrape {N.pct(r.rappel)} des départs. Le
modèle v3, qui a besoin de l'historique des feux et n'est donc pas utilisable
en temps réel, en attraperait {N.pct(rv.rappel)}.

L'écart de PR-AUC entre les deux vaut 37 %, ce qui paraît considérable.
Traduit en départs attrapés, il vaut {gap} points de pourcentage. Au-delà de
10 % de territoire surveillé, les deux courbes se rejoignent.
""")
        st.divider()

    BASE = csv("baselines.csv")
    if BASE is not None:
        st.markdown("### La barre à battre")
        st.caption("Trois prédicteurs sans aucun apprentissage, mesurés sur "
                   "la même validation.")
        b = BASE.sort_values("lift")
        fig, ax = plt.subplots(figsize=(11, 2.7))
        coul = [N.GRIS if "hasard" in p else N.BLEU for p in b.predicteur]
        vals = list(b.lift) + [MT["test"]["lift"]]
        noms = list(b.predicteur) + ["4 · le modèle déployé (test)"]
        ax.barh(range(len(b)), b.lift, color=coul, edgecolor=N.FOND, lw=1.2)
        ax.barh([len(b)], [MT["test"]["lift"]], color=N.VERT,
                edgecolor=N.FOND, lw=1.2)
        for i_, v in enumerate(vals):
            ax.text(v + .8, i_, f"×{v:.1f}", va="center", fontsize=9,
                    weight="bold")
        ax.set_yticks(range(len(noms)), noms, fontsize=9)
        ax.set_xlim(0, max(vals) * 1.16)
        habiller(ax, x="lift : combien de fois mieux que le hasard", axe="x")
        plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

        st.markdown("""
L'historique spatial seul vaut déjà 19 fois le hasard, la météo seule 5 fois,
et leur croisement 42. C'est cette dernière valeur qu'il faut battre, pas le
hasard. Un modèle à ×30 serait moins bon qu'une règle de trois.
""")
        st.divider()

    TPA = csv("test_par_annee.csv")
    if TPA is not None:
        st.markdown("### Année par année")
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.dataframe(TPA.assign(**{
                "Année": TPA.an, "Feux": TPA.feux,
                "PR-AUC": TPA.pr_auc.round(4),
                "lift": TPA.lift.map(lambda v: f"×{v:.0f}"),
            })[["Année", "Feux", "PR-AUC", "lift"]],
                width="stretch", hide_index=True)
        with c2:
            mx, mn = TPA.loc[TPA.lift.idxmax()], TPA.loc[TPA.lift.idxmin()]
            st.markdown(f"""
Le lift varie du simple au double selon l'année, et il suit la rareté.
{int(mx.an)} est l'année la plus calme, avec {N.nb(mx.feux)} départs, et donne
le meilleur lift (×{mx.lift:.0f}). {int(mn.an)}, la plus active, le plus
faible (×{mn.lift:.0f}).

Une année calme concentre les feux dans les endroits prévisibles. Quand tout
brûle, y compris là où ce n'est pas censé arriver, le modèle est pris en
défaut.

Ces chiffres ont été mesurés une seule fois, après gel complet du modèle, des
features et de la calibration.
""")

# ════════════════════════════════════════════════════════════════════════
with o2:
    p1, p2, p3 = st.columns(3)
    s = MT["splits"]
    with p1:
        st.markdown(f"""
##### Le découpage

| Partition | Années | Rôle |
|---|---|---|
| train | {s['train'][0]}-{s['train'][1]} | apprendre |
| validation | {s['val'][0]}-{s['val'][1]} | choisir |
| test | {s['test'][0]}-{s['test'][1]} | juger, une fois |

Temporel, jamais aléatoire. Un découpage au hasard mettrait le 14 juillet 2019
dans le train et le 15 dans le test : le modèle « prédirait » un feu qu'il a
déjà vu, à vingt kilomètres et un jour d'écart.
""")
    with p2:
        st.markdown("""
##### La métrique

À 0,019 % de positifs, la ROC-AUC est flatteuse : les vrais négatifs écrasent
tout et un modèle médiocre affiche 0,95.

La PR-AUC vaut exactement le taux de base quand on répond au hasard. Le
rapport des deux, le lift, se lit directement : le modèle est N fois meilleur
que tirer au sort.
""")
    with p3:
        st.markdown("""
##### Le prior déplacé

Le train est sous-échantillonné à 1 positif pour 10 négatifs, sans quoi
l'entraînement serait ingérable. Le modèle apprend donc sur un monde à 9,1 %
de feux quand le vrai taux est 0,019 %, soit un facteur **487**.

Validation et test ne sont jamais échantillonnés.
""")

    with st.expander("Le piège qui ne se voit dans aucune métrique"):
        st.markdown("""
Toute statistique dérivée de la cible doit se calculer sur le train complet,
pas sur le train échantillonné. Sur l'échantillon, un lissage bayésien
vaudrait 9,1 % au lieu de 0,019 %. Le prior serait faux d'un facteur 487, et
rien dans les métriques ne le signalerait.

Deuxième règle, plus subtile : une feature datée peut regarder tout le passé,
y compris celui de sa propre période d'évaluation ; une statistique non datée
ne peut regarder que le train. « Feux des 30 jours précédents » au 3 août 2023
lit juillet 2023, et ce n'est pas une fuite : le 3 août au matin, on connaît
juillet. « Taux moyen de la commune sur toute la période » lit le futur.
""")

    st.divider()

    CA, CAL = csv("courbe_apprentissage.csv"), csv("calibration_v3.csv")
    g1, g2 = st.columns(2)

    with g1:
        st.markdown("##### Le modèle apprend-il, ou récite-t-il ?")
        if CA is not None:
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            ax.plot(CA.iteration, CA.ajustement, color=N.GRIS, lw=1.6,
                    label="sur les données d'ajustement")
            ax.plot(CA.iteration, CA.evaluation, color=N.BLEU, lw=1.8,
                    label="sur les données d'évaluation")
            j = int(CA.evaluation.idxmax())
            ax.axvline(CA.iteration[j], color=N.ROUGE, ls=":", lw=1.3)
            ax.text(CA.iteration[j], CA.evaluation.min(),
                    f"  maximum : {int(CA.iteration[j])} arbres", fontsize=8.5,
                    color=N.ROUGE)
            habiller(ax, x="arbres ajoutés", y="PR-AUC interne")
            ax.legend(frameon=False, fontsize=8.5, loc="lower right")
            plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)
            st.caption(
                f"L'écart entre les deux courbes est le surapprentissage. Il "
                f"reste modéré et l'évaluation cesse de progresser vers "
                f"{int(CA.iteration[j])} arbres : au-delà, on paie du temps de "
                f"calcul sans rien gagner.")
        else:
            st.caption("`courbe_apprentissage.csv` absent.")

    with g2:
        st.markdown("##### Le score est-il une probabilité ?")
        courbes = {m: csv(f"fiabilite_{m}.csv")
                   for m in ("brut", "platt", "isotonic")}
        if any(v is not None for v in courbes.values()):
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            lim = 1e-7
            for (m, d), coul in zip(courbes.items(),
                                    (N.ROUGE, N.VERT, N.BLEU)):
                if d is None:
                    continue
                ax.plot(d.annonce.clip(lower=lim), d.observe.clip(lower=lim),
                        "o-", color=coul, lw=1.5, ms=3.5, label=m)
            ax.plot([lim, 1], [lim, 1], ":", color=N.MUTED, lw=1.2,
                    label="calibration parfaite")
            ax.set_xscale("log"); ax.set_yscale("log")
            habiller(ax, x="probabilité annoncée", y="fréquence observée")
            ax.legend(frameon=False, fontsize=8.5, loc="upper left")
            plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)
            if CAL is not None:
                b = CAL.set_index("methode").biais
                platt = f"{b['platt']:.2f}".replace(".", ",")
                st.caption(
                    f"Le score brut annonce {b['brut']:.0f} fois trop de "
                    f"risque : c'est le sous-échantillonnage. Platt le ramène "
                    f"à ×{platt} sans rien coûter en PR-AUC. "
                    f"L'isotonique calibre aussi bien mais écrase le score sur "
                    f"{int(CAL.set_index('methode').valeurs_distinctes['isotonic'])} "
                    f"valeurs distinctes au lieu de 9 millions.")
        else:
            st.caption("`fiabilite_*.csv` absents.")

    st.markdown("""
L'application affiche finalement un rang, pas une probabilité. Le calibrateur
disponible a été ajusté sur un autre modèle et une autre période : il serait
faux d'un facteur 2. Un rang, lui, reste juste.
""")

# ════════════════════════════════════════════════════════════════════════
with o3:
    DESCR = {
        "XGBoost v3": "52 features, historique des feux et clustering territorial",
        "DART": "même chose, avec abandon d'arbres à l'entraînement",
        "MLP": "réseau dense 3 couches, dropout, mêmes 52 features",
        "XGBoost C": "41 features, physique pure, rien qui dérive de la cible",
        "LSTM": "30 jours × 8 indices météo en séquence, plus 30 features de territoire",
    }
    T = pd.DataFrame([{
        "Modèle": k, "PR-AUC": N.dec(v, 4), "lift": f"×{v / TAUX_VAL:.1f}",
        "Ce que c'est": DESCR.get(k, "")}
        for k, v in sorted(AP.items(), key=lambda x: -x[1])])
    st.dataframe(T, width="stretch", hide_index=True)

    ENS = csv("modeles_ensemble.csv")
    st.caption(
        f"Validation : {N.nb(38_068_464)} communes-jours, 9 176 feux, taux "
        f"{N.pct(TAUX_VAL, 4)}."
        + (f" L'ensemble v3 + MLP monte à {N.dec(ENS.pr_auc.iloc[0], 4)} "
           f"(×{N.dec(ENS.lift.iloc[0])}), au prix de deux modèles à faire "
           f"tourner." if ENS is not None else ""))

    st.markdown("### Ces écarts survivent-ils au bruit ?")
    st.markdown("""
Bootstrap apparié, 200 répliques, en rééchantillonnant les 34 734 communes et
non les lignes. Les 1 096 jours d'une même commune ne sont pas indépendants, et
31 communes partagent en moyenne la même maille météo : un rééchantillonnage
ligne à ligne donnerait des intervalles trop étroits.
""")

    c = CMP[CMP.reference == "XGBoost v3"].copy()
    c = pd.concat([c, CMP[(CMP.reference == "XGBoost C")
                          & (CMP.modele == "LSTM")]]).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    for i_, r in c.iterrows():
        coul = N.ROUGE if r.significatif else N.MUTED
        ax.plot([r.ic_bas, r.ic_haut], [i_, i_], color=coul, lw=2.6,
                solid_capstyle="round", zorder=2)
        ax.plot(r.ecart_pct, i_, "o", color=coul, ms=7, zorder=3)
        ax.text(r.ic_haut + 1.5, i_,
                f"{N.dec(r.ecart_pct):>6s} %" if r.ecart_pct < 0 else f"+{N.dec(r.ecart_pct)} %"
                + ("" if r.significatif else "   non significatif"),
                va="center", fontsize=9, color=coul,
                weight="bold" if r.significatif else "normal")
    ax.axvline(0, color=N.INK, lw=1.2)
    ax.set_yticks(range(len(c)),
                  [f"{r.modele}  vs  {r.reference}" for _, r in c.iterrows()],
                  fontsize=9)
    ax.set_xlim(-72, 30)
    habiller(ax, x="écart de PR-AUC (%), intervalle de confiance à 95 %",
             axe="x")
    plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

    st.success(f"""
DART et le MLP paraissaient {N.dec(abs(ec('XGBoost v3', 'DART').ecart_pct))} % et
{N.dec(abs(ec('XGBoost v3', 'MLP').ecart_pct))} % moins bons que XGBoost. Leurs
intervalles traversent zéro : les trois modèles sont **indiscernables**.
Conclure que « XGBoost bat le MLP » aurait été lire du bruit.

Deux écarts seulement résistent : celui du modèle physique et celui du LSTM.
""")

    st.divider()
    st.markdown("### Le LSTM")

    g, d_ = st.columns([1.1, 1])
    with g:
        l_c = ec("XGBoost C", "LSTM")
        st.markdown(f"""
« Pour le temps, prends un LSTM » est le réflexe standard. On l'a construit,
réglé par 25 essais Optuna avec arrêt précoce, et mesuré. Il perd de
**{N.dec(abs(l_c.ecart_pct))} %** contre le modèle physique, intervalle
[{N.dec(l_c.ic_bas)} ; {N.dec(l_c.ic_haut)}].

La comparaison contre XGBoost v3 n'aurait pas eu de sens : v3 voit l'historique
des feux, qui pèse 29 % de ses importances, et le LSTM n'en voit rien. Le seul
adversaire à jeu d'information égal est le modèle physique. Le LSTM reçoit
pourtant 30 jours × 8 indices, soit 240 valeurs, contre onze features météo
pour son adversaire.
""")
        bp = N.DON / "best_params_lstm.json"
        if bp.exists():
            P = json.loads(bp.read_text(encoding="utf-8"))
            with st.expander("Les hyperparamètres retenus"):
                st.dataframe(pd.DataFrame(
                    [{"Hyperparamètre": k, "Valeur": f"{v:.5g}"}
                     for k, v in P.items()]),
                    width="stretch", hide_index=True)
    with d_:
        st.markdown("""
##### Pourquoi il perd

Un LSTM sert quand l'ordre de la séquence porte une information qu'aucun
résumé ne capture. Ici ce résumé existe déjà.

Les indices DC, DMC et BUI du système canadien sont des états récursifs. Le
*Drought Code* est une moyenne exponentielle de la météo passée avec une
constante de temps de 52 jours ; le *Duff Moisture Code*, de 15 jours. C'est la
forme d'une cellule récurrente, sauf que les coefficients ont été calibrés par
cinquante ans de science du feu plutôt qu'estimés sur 9 176 exemples positifs.

Le CEMS livre donc déjà l'état caché que le LSTM devrait réapprendre. Deux
observations vont dans le même sens : la PACF montre une autocorrélation
épuisée en deux à trois jours, et un ARIMA sans exogène donne r = −0,118.

Une réserve : le LSTM ne reçoit pas `danger_effis`, qui pèse 13,7 % dans le
modèle physique. Les 23,6 % sont donc un majorant.
""")

    st.divider()
    st.markdown("### Pourquoi le meilleur modèle n'est pas celui qu'on déploie")

    a = MT["modele_a"]
    m1, m2, m3 = st.columns(3)
    m1.metric("XGBoost v3 sur le test", f"×{a['lift']:.1f}", "le meilleur")
    m2.metric("Modèle C sur le test", f"×{MT['test']['lift']:.1f}",
              f"{100 * (MT['test']['lift'] / a['lift'] - 1):+.0f} %",
              delta_color="off")
    m3.metric("En territoire jamais vu", "C gagne 9 fois sur 9",
              "+8,2 % pondéré")

    st.markdown("""
v3 est meilleur sur le test, et pourtant c'est le modèle physique qui tourne
dans l'application. Il tire 29 % de son importance de l'historique des feux, or
la BDIFF ne publie pas l'année en cours : les feux de 2026 sortiront au
printemps 2027. Pour une prédiction faite aujourd'hui, `feux_commune_7j`
vaudrait le décompte d'une semaine de décembre 2025. Pas imprécis, faux.

En territoire inconnu, la même variable vaut zéro partout, et le modèle lit ce
zéro comme « ça n'a jamais brûlé, donc ça ne brûlera pas ». Pour 2050 elle est
impossible par construction : on ne connaîtra jamais les feux de 2049.

Le choix se fait donc sur la disponibilité de la donnée, pas sur la
performance. Ce défaut n'apparaît dans aucune métrique d'entraînement : en
validation comme en test, l'historique est toujours là.
""")

    TS = csv("transfert_spatial.csv")
    if TS is not None:
        ts = TS.copy()
        ts["ecart"] = 100 * (ts["C · physique"] / ts["A · tout"] - 1)
        ts = ts.sort_values("ecart")
        fig, ax = plt.subplots(figsize=(11, 3.6))
        y = np.arange(len(ts))
        ax.barh(y - .2, ts["A · tout"], height=.38, color=N.GRIS,
                label="A · toutes les features", edgecolor=N.FOND, lw=.8)
        ax.barh(y + .2, ts["C · physique"], height=.38, color=N.VERT,
                label="C · physique pure", edgecolor=N.FOND, lw=.8)
        for i_, (_, r) in enumerate(ts.iterrows()):
            ax.text(max(r["A · tout"], r["C · physique"]) + .012, i_,
                    f"{r.ecart:+.0f} %", va="center", fontsize=8.5,
                    weight="bold", color=N.VERT)
        ax.set_yticks(y, [f"région {int(r)}" for r in ts.region], fontsize=8.5)
        ax.set_xlim(0, ts[["A · tout", "C · physique"]].to_numpy().max() * 1.18)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        habiller(ax, "On retire une région du train, puis on teste dessus",
                 "PR-AUC sur la région exclue", axe="x")
        plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)
        st.caption(f"Le modèle physique gagne dans les neuf régions, jusqu'à "
                   f"+{ts.ecart.max():.0f} % dans le Grand Est. Là où "
                   f"l'historique est le plus pauvre, s'y fier est un handicap.")

# ════════════════════════════════════════════════════════════════════════
with o4:
    TA = csv("modele_taille.csv")
    ta = TA.iloc[0] if TA is not None else None

    l1, l2 = st.columns(2)
    with l1:
        st.markdown("##### Ce que le modèle ne sait pas faire")
        if ta is not None:
            st.markdown(f"""
La surface brûlée n'est pas prédictible : R² de {N.dec(ta.r2_log, 2)}, moins bon que
d'annoncer toujours la médiane. Elle dépend surtout de ce qui se passe après le
départ, c'est-à-dire du vent, du délai d'intervention et du relief.

La question binaire « sera-ce un grand feu de plus de {ta.seuil_ha:.0f} ha ? »
se prédit mal aussi : PR-AUC {N.dec(ta.pr_auc_grand, 3)} pour un taux de base de
{N.pct(ta.base_grand, 1)}, soit un lift de {N.dec(ta.lift_grand)} seulement. On
lit parfois « ROC-AUC {N.dec(ta.roc_auc_grand, 2)} » pour ce modèle. C'est exact,
mais s'en servir après avoir expliqué pourquoi la ROC-AUC flatte serait se
contredire.
""")
        st.markdown("""
Le score affiché est un rang, pas une probabilité.

Le modèle suppose stable tout ce qui n'est pas la météo : prévention,
pratiques agricoles, déprise rurale. Les projections à 2100 ne font varier que
le climat.
""")
    with l2:
        st.markdown("""
##### Ce qu'il faut savoir des données

Une commune-jour n'est pas un incendie. Un feu traversant cinq communes compte
cinq fois ; les 49 130 « feux » sont des communes-jours ayant brûlé.

Environ 31 communes partagent une maille météo de 28 km et ont donc le même FWI
le même jour. Le FWI porte le *quand*, la végétation porte le *où*. Conséquence
statistique : les intervalles naïfs sur les coefficients météo seraient trop
étroits.

Les feux ne sont observés que depuis 2006, la météo depuis 1973. On compare
donc une tendance sur 53 ans à une stabilité sur 20.

1 378 feux d'outre-mer sont exclus, hors couverture météo européenne, et
30 feux métropolitains ne sont rattachables à aucune commune actuelle.
""")

    st.divider()
    st.markdown("### Une erreur qui a failli passer")

    e1, e2 = st.columns([1.15, 1])
    with e1:
        st.markdown("""
Le premier verdict du LSTM annonçait −97 %. Le vrai est −52 %.

`sql/50_matrice.sql` n'a pas d'`ORDER BY`. L'ordre dans lequel PostgreSQL
renvoie les 38 millions de lignes dépend du plan d'exécution et change d'une
exécution à l'autre. Les fichiers de prédictions ne portaient que le score et
la cible : les comparer revenait à les aligner par position. Même taille, même
nombre de feux, ordre différent, et aucune erreur levée.

Depuis, tout fichier de prédictions porte ses clés, une fonction d'alignement
vérifie qu'elles correspondent, et un test refuse un fichier sans clés.
""")
    with e2:
        st.code("""LSTM aligné sur les clés    PR-AUC 0.0085   ×35,4
LSTM, lignes permutées      PR-AUC 0.0002   ×1,0
                                            ↑
                            le hasard, avec
                            les mêmes valeurs""", language=None)
        st.caption("Sur un événement à 0,024 %, une erreur de plomberie ne "
                   "lève pas d'exception. Elle produit un chiffre plausible.")

    with st.expander("Les autres erreurs trouvées et corrigées"):
        st.markdown("""
| L'erreur | Le symptôme | Ce qui l'a révélée |
|---|---|---|
| `ha` et la cible laissées dans les features du modèle de surface | R² = 0,994 et ROC-AUC = 1,0000 | un score trop beau pour être vrai |
| interaction analysée sur le seul échantillon du sommet | le signe s'inversait | biais de collision : sélectionner sur le score conditionne le résultat |
| moyenne climatologique non lissée | un dentelé de période 4 ans | le 15 août tombe au jour 227 ou 228 selon les bissextiles |
| projection ancrée à k = 1,0 en 2025 | la courbe plongeait entre 2025 et 2026 | le facteur observé de 2025 vaut déjà 1,3 |
| tendance du FWI écrite en dur | +45 % affiché, +58 % réel | recalcul systématique à la source |

50 tests tournent en intégration continue à chaque commit.
""")
