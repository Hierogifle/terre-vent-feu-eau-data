"""Ce qu'on a compris des données.

Quatre blocs : d'où elles viennent, ce qu'elles disent du danger sur 53 ans,
où ça brûle et sur quoi, et ce que la dimension TEMPS apporte — ou n'apporte
pas.

⚠️ AUCUN CHIFFRE DE TENDANCE N'EST ÉCRIT EN DUR DANS CE FICHIER.
Une version antérieure annonçait « +45 % de FWI moyen (p < 0,0001) » et « sur
2006-2025 la pente n'est pas significative (p = 0,13) ». Les deux étaient
fausses, et ont survécu des semaines précisément parce qu'elles étaient dans
une chaîne de caractères que rien ne recalculait. Tout vient désormais de
`tendances.csv`, produit par `tvfed.export_app`.
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
import streamlit as st

import noyau as N

st.set_page_config(page_title="Les données · Risque incendie", page_icon="🔥",
                   layout="wide")
plt.rcParams.update({"figure.facecolor": N.FOND, "axes.facecolor": N.FOND,
                     "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "text.color": N.INK, "xtick.color": N.MUTED,
                     "ytick.color": N.MUTED, "axes.labelcolor": N.INK})
N.entete()

COM, MT, TEN = N.communes(), N.meta(), N.tendances()


def ten(nom: str) -> pd.Series:
    """Une ligne de tendances.csv, par son nom de série."""
    return TEN[TEN.serie == nom].iloc[0]


FWI_AN = ten("FWI moyen annuel")
FWI_ETE = ten("FWI moyen juin-septembre")
JOURS = ten("jours de danger élevé (FWI > 21,3)")
FEUX = ten("communes-jours en feu")


# ════════════════════════════════════════════════════════════════════════
st.markdown("## Le jeu de données")
a, b, c, d = st.columns(4)
a.metric("Communes", f"{len(COM):,}".replace(",", " "))
b.metric("Jours couverts", "19 358", "1973 → 2025")
c.metric("Communes-jours en feu", f"{int(COM.feux.sum()):,}".replace(",", " "),
         "2006-2025")
d.metric("Surface brûlée",
         f"{COM.ha.sum() / 1000:,.0f} k ha".replace(",", " "))

st.markdown("""
Quatre sources, croisées sur le **code INSEE** et sur une **grille météo de
0,25°** :

| Source | Ce qu'elle apporte | Volume |
|---|---|---|
| **CEMS** (Copernicus) | 8 indices de danger, chaque jour, chaque maille | 21,9 M lignes, 1973-2025 |
| **BDIFF** (IGN) | les feux déclarés, commune par commune | 142 787 feux au total, 52 809 sur le périmètre |
| **CORINE** (Copernicus) | l'occupation du sol, 44 postes | 1,08 M lignes |
| **INSEE** | référentiel des communes et fusions | 34 734 communes |

La table centrale est une grille **commune × jour** de
**253 731 870 lignes** sur 2006-2025, avec une cible binaire : *y a-t-il eu un
départ de feu ce jour-là dans cette commune ?*

**49 130 fois oui, soit 0,0194 %.** C'est cette rareté qui commande toute la
méthode — à ce niveau, un modèle qui répond toujours « non » a 99,98 % de
justesse et ne sert à rien.
""")

with st.expander("Pourquoi une grille dense, et non la seule liste des feux ?"):
    st.markdown("""
Parce qu'une série creuse rendrait les fenêtres glissantes **silencieusement
fausses**. « Nombre de feux dans les 30 jours précédents » se calcule en
remontant 30 lignes — si les jours sans feu sont absents, on remonte en
réalité plusieurs années.

C'est la raison technique n°1 d'avoir matérialisé 253 millions de lignes
plutôt que 52 809.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  le danger augmente-t-il ?
# ════════════════════════════════════════════════════════════════════════
st.markdown("## Le danger météo augmente-t-il ?")
st.caption("Moyenne nationale des 1 131 mailles, par décennie. Ce sont des "
           "mesures issues des réanalyses Copernicus, pas des projections.")

D = N.decennies().dropna(subset=["periode"])
nat = D.groupby("periode").agg(
    fwi=("fwi_moyen", "mean"), p90=("fwi_p90", "mean"),
    jours=("jours_danger", "mean"), extremes=("jours_tres_eleve", "mean")
).sort_index()

fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
x = np.arange(len(nat))
for a_, col, titre, coul in (
        (ax[0], "fwi", "FWI moyen annuel", N.ORANGE),
        (ax[1], "jours", "Jours de danger élevé (FWI > 21,3)", N.ROUGE),
        (ax[2], "extremes", "Jours très élevés (FWI > 38)", "#8b1a1a")):
    a_.bar(x, nat[col], color=coul, edgecolor=N.FOND, linewidth=1.4)
    for i, v in enumerate(nat[col]):
        a_.text(i, v * 1.02, f"{v:.1f}", ha="center", fontsize=8.5, weight="bold")
    a_.set_xticks(x)
    a_.set_xticklabels(nat.index, fontsize=7.5, rotation=25, ha="right")
    a_.set_title(titre, fontsize=10.5, weight="bold", loc="left")
    a_.grid(axis="y", color=N.GRID, lw=.7); a_.set_axisbelow(True)
    a_.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
st.pyplot(fig, width='stretch')
plt.close(fig)

d0, d1 = nat.iloc[0], nat.iloc[-1]
m1, m2, m3 = st.columns(3)
m1.metric(f"FWI moyen · {nat.index[0]} → {nat.index[-1]}", f"{d1.fwi:.2f}",
          f"{100 * (d1.fwi / d0.fwi - 1):+.0f} %")
m2.metric("Jours de danger élevé", f"{d1.jours:.1f} j/an",
          f"{d1.jours - d0.jours:+.1f} j")
m3.metric("Jours très élevés", f"{d1.extremes:.1f} j/an",
          f"{d1.extremes - d0.extremes:+.1f} j")

st.caption(
    f"⚠️ **Ces trois écarts ne sont pas ceux du tableau ci-dessous, et c'est "
    f"normal.** Ici on compare la dernière décennie à la première "
    f"({nat.index[-1]} contre {nat.index[0]}) — deux fenêtres de 6 à 10 ans. "
    f"En dessous, une droite est ajustée sur les {FWI_AN.n_ans} années. Deux "
    f"façons légitimes de mesurer la même hausse, deux nombres différents : "
    f"c'est précisément pourquoi un chiffre ne doit jamais circuler sans sa "
    f"définition.")

# ── les pentes, CALCULÉES ────────────────────────────────────────────────
st.markdown("##### Les pentes de fond, par régression linéaire")
T = TEN.copy()
# ⚠️ pas de markdown ici : `st.dataframe` affiche le texte brut, les
# astérisques apparaîtraient telles quelles.
T["Conclusion"] = np.where(T.significatif, "significatif", "NON significatif")
st.dataframe(
    T.assign(**{
        "Série": T.serie,
        "Période": T.an_min.astype(str) + "-" + T.an_max.astype(str),
        "Pente / an": T.pente.round(4),
        "Variation": T.variation_pct.round(0).astype(int).astype(str) + " %",
        "p": T.p.map(lambda v: f"{v:.1e}"),
    })[["Série", "Période", "Pente / an", "Variation", "p", "Conclusion"]],
    width="stretch", hide_index=True)

st.info(f"""
**Le danger monte, et c'est établi.** Sur les {FWI_AN.n_ans} années de mesures,
le FWI moyen gagne **{FWI_AN.variation_pct:+.0f} %** (p = {FWI_AN.p:.1e}), et les
jours de danger élevé **{JOURS.variation_pct:+.0f} %** (p = {JOURS.p:.1e}).

⚠️ **L'ampleur dépend de l'agrégation, et il faut le dire.** La moyenne
**juin-septembre** monte de **{FWI_ETE.variation_pct:+.0f} %**, plus que la moyenne
annuelle : c'est l'été qui se réchauffe et s'assèche le plus. Citer un chiffre
sans son périmètre est ce qui permet à deux valeurs incompatibles de coexister.

**Et pourtant le nombre de feux ne monte pas** : {FEUX.variation_pct:+.0f} % sur
{FEUX.n_ans} ans, p = {FEUX.p:.2f} — rigoureusement rien.
""")

with st.expander("Comment deux constats opposés peuvent-ils être vrais tous les deux ?"):
    st.markdown(f"""
Ce n'est pas une contradiction, et il serait malhonnête de ne montrer que le
premier. Trois lectures cohabitent :

1. **La puissance statistique.** Les feux ne sont observés que depuis 2006 :
   {FEUX.n_ans} points annuels très bruités ne peuvent pas détecter une tendance
   modérée. *L'absence de preuve n'est pas une preuve d'absence.*
2. **La prévention fonctionne.** Le nombre de départs dépend autant des moyens
   de lutte, des interdictions d'accès aux massifs et du débroussaillement que
   du climat. Un aléa qui monte à sinistralité constante est le résultat
   **attendu** d'une politique de prévention efficace.
3. **Ce que le modèle projette, c'est l'aléa, pas le bilan.** Les projections à
   2100 transportent l'évolution du **FWI** sous les scénarios RCP, jamais une
   extrapolation du décompte de feux. C'est la seule des deux quantités qui
   montre un signal, et la seule qu'un modèle climatique sache fournir.

**À retenir** : ne pas dire « les feux augmentent » — les données ne le montrent
pas. Dire « les conditions favorables aux feux augmentent très
significativement, et le nombre de départs reste stable, ce qui est cohérent
avec une prévention qui absorbe pour l'instant la hausse de l'aléa ».
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  où ça brûle
# ════════════════════════════════════════════════════════════════════════
st.markdown("## Où ça brûle, et sur quoi")

g1, g2 = st.columns(2)

with g1:
    top = (COM.groupby("dep_nom")
             .agg(feux=("feux", "sum"), ha=("ha", "sum"))
             .nlargest(12, "feux").sort_values("feux"))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(range(len(top)), top.feux, color=N.ROUGE, edgecolor=N.FOND, lw=1.2)
    for i, v in enumerate(top.feux):
        ax.text(v * 1.01, i, f"{int(v):,}".replace(",", " "), va="center",
                fontsize=8.5, weight="bold")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8.5)
    ax.set_xlim(0, top.feux.max() * 1.15)
    ax.set_xlabel("communes-jours ayant brûlé, 2006-2025")
    ax.set_title("Les 12 départements les plus touchés", fontsize=10.5,
                 weight="bold", loc="left")
    ax.grid(axis="x", color=N.GRID, lw=.7); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

with g2:
    # le taux de feu par tranche de maquis — le résultat central du projet
    C = COM[COM.part_maquis.notna()].copy()
    C["tranche"] = pd.cut(100 * C.part_maquis, [-.01, .5, 5, 15, 30, 101],
                          labels=["< 0,5 %", "0,5-5 %", "5-15 %", "15-30 %",
                                  "> 30 %"])
    t = C.groupby("tranche", observed=True).agg(
        feux=("feux", "sum"), n=("code_insee", "size"))
    t["taux"] = t.feux / (t.n * 7305)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(range(len(t)), 100 * t.taux, color="#8b5a2b", edgecolor=N.FOND, lw=1.4)
    for i, (v, n) in enumerate(zip(100 * t.taux, t.n)):
        ax.text(i, v * 1.03, f"{v:.3f} %", ha="center", fontsize=8.5,
                weight="bold")
        ax.text(i, v * .5, f"{n:,}".replace(",", " ") + "\ncommunes",
                ha="center", fontsize=7.5, color="white")
    ax.set_xticks(range(len(t))); ax.set_xticklabels(t.index, fontsize=8.5)
    ax.set_xlabel("part de maquis dans la commune")
    ax.set_ylabel("probabilité de feu un jour donné (%)")
    ax.set_title("Le maquis, seul, multiplie le risque", fontsize=10.5,
                 weight="bold", loc="left")
    ax.grid(axis="y", color=N.GRID, lw=.7); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

ratio = (100 * t.taux).iloc[-1] / (100 * t.taux).iloc[0]
st.markdown(f"""
Une commune couverte à plus de 30 % de maquis brûle **×{ratio:.0f}** plus
souvent qu'une commune qui n'en a pas. C'est le résultat central du projet, et
c'est ce que le modèle exploite : le maquis arrive **en tête de l'importance
par gain, à 26,2 %**, devant le danger météo.

⚠️ « En tête » selon *quelle* mesure, et sur *quelle* population ? La question
n'est pas rhétorique : sur un échantillon aléatoire de communes-jours, SHAP
place le maquis **10ᵉ** — il ne change rien là où il n'y en a pas. Il remonte
**2ᵉ** dès qu'on regarde les communes que le modèle juge à risque. Les trois
mesures sont détaillées page *Pourquoi un feu part*.

*La météo dit quand, le territoire dit où.*
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  le temps
# ════════════════════════════════════════════════════════════════════════
st.markdown("## Ce que le temps apporte — et ce qu'il n'apporte pas")
st.caption("Analyse de la série nationale : nombre de communes-jours en feu "
           "par jour, 7 305 points de 2006 à 2025.")

st.markdown("""
Le modèle répond à **où**. Il ne répond jamais à **combien de feux demain en
France**, qui est pourtant la question qui dimensionne les moyens nationaux.
On a donc traité cet axe séparément — et il apprend surtout quelque chose sur
les limites de l'approche temporelle.
""")

c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("##### Stationnarité — test de Dickey-Fuller augmenté")
    try:
        ADF = pd.read_csv(N.DON / "series_adf.csv")
        st.dataframe(ADF.assign(**{
            "Série": ADF.serie,
            "stat ADF": ADF.adf.round(2),
            "p": ADF.p.map(lambda v: f"{v:.1e}"),
            "Conclusion": np.where(ADF.stationnaire, "stationnaire",
                                   "non stationnaire"),
        })[["Série", "stat ADF", "p", "Conclusion"]],
            width="stretch", hide_index=True)
        st.caption("⚠️ H₀ = « la série a une racine unitaire », donc "
                   "**non** stationnaire. Rejeter H₀ (p < 0,05) signifie "
                   "STATIONNAIRE — c'est l'inverse de l'intuition, et la "
                   "confusion la plus fréquente sur ce test.")
    except FileNotFoundError:
        st.caption("`series_adf.csv` absent — lancer `python -m tvfed.series`.")

with c2:
    st.markdown("##### Prévoir « combien » — SARIMAX")
    try:
        SAR = pd.read_csv(N.DON / "series_sarimax.csv")
        st.dataframe(SAR.assign(**{
            "Modèle": SAR.modele, "MAE": SAR.mae.round(2),
            "r": SAR.correlation.round(3),
        })[["Modèle", "MAE", "r"]], width="stretch", hide_index=True)
        st.caption("Ajusté sur 2006-2019, évalué sur **2020-2022**. Le test "
                   "2023-2025 n'est pas touché — même pour une cible "
                   "différente, on ne l'entrouvre pas.")
    except FileNotFoundError:
        st.caption("`series_sarimax.csv` absent — lancer `python -m tvfed.series`.")

st.warning("""
**La ligne qui compte est la dernière : ARIMA sans exogène donne r = −0,118.**
La corrélation est **négative**.

À 1 096 pas d'horizon, un modèle autorégressif dont la mémoire utile est de
deux à trois jours a totalement oublié son point de départ : il converge vers
la moyenne, et la ligne plate qu'il produit se trouve légèrement
anti-corrélée à l'observé, par hasard. Ce n'est pas un mauvais réglage, c'est
**structurel**.

Ajouter le FWI fait tomber l'erreur de 37 %. **La prévisibilité du feu n'est
pas dans son propre passé — elle est dans la météo.** C'est exactement ce que
le modèle principal exploite, et c'est aussi pourquoi un LSTM n'y change rien
(voir la page *Les modèles*).
""")

with st.expander("Pourquoi pas de composante saisonnière SARIMA ?"):
    st.markdown("""
Une saisonnalité annuelle sur données journalières donnerait *s* = 365. Un
SARIMA(p,d,q)(P,D,Q)₃₆₅ demanderait d'estimer des coefficients à 365 pas de
distance sur 5 113 points d'ajustement : le modèle serait ingérable et
instable.

La pratique établie sur données journalières est de porter la saisonnalité par
des **termes de Fourier en exogène** — quelques harmoniques suffisent à décrire
un cycle annuel lisse. C'est le « X » de SARIMAX qui travaille.
""")

st.caption("Le détail complet — ACF, PACF, choix des ordres, tendance sur "
           "53 ans — est dans `notebook/series-lstm.ipynb` et le cours "
           "`docs/series-temporelles.md`.")
