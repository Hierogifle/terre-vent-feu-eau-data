"""Le jeu de données, ce qu'il contient et ce qu'il montre.

Trois blocs : d'où viennent les données, ce qu'elles disent de l'évolution du
danger sur 53 ans, et ce que vaut le modèle.
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

st.set_page_config(page_title="Statistiques · Risque incendie", page_icon="🔥",
                   layout="wide")
plt.rcParams.update({"figure.facecolor": N.FOND, "axes.facecolor": N.FOND,
                     "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "text.color": N.INK, "xtick.color": N.MUTED,
                     "ytick.color": N.MUTED, "axes.labelcolor": N.INK})
N.entete()

COM, MT = N.communes(), N.meta()

st.markdown("## Le jeu de données")
a, b, c, d = st.columns(4)
a.metric("Communes", f"{len(COM):,}".replace(",", " "))
b.metric("Jours couverts", "19 358", "1973 → 2025")
c.metric("Feux enregistrés", f"{int(COM.feux.sum()):,}".replace(",", " "),
         "2006-2025")
d.metric("Surface brûlée",
         f"{COM.ha.sum() / 1000:,.0f} k ha".replace(",", " "))

st.markdown(f"""
Quatre sources, croisées sur le **code INSEE** et sur une **grille météo de
0,25°** :

| Source | Ce qu'elle apporte | Volume |
|---|---|---|
| **CEMS** (Copernicus) | 8 indices de danger, chaque jour, chaque maille | 21,9 M lignes, 1973-2025 |
| **BDIFF** (IGN) | les feux déclarés, commune par commune | 142 787 feux |
| **CORINE** (Copernicus) | l'occupation du sol, 44 postes | 1,08 M lignes |
| **INSEE** | référentiel des communes et fusions | 34 734 communes |

La table centrale est une grille **commune × jour** de
**253 731 870 lignes** sur 2006-2025, avec une cible binaire : *y a-t-il eu un
départ de feu ce jour-là dans cette commune ?*

**49 130 fois oui, soit 0,0194 %.** C'est cette rareté qui commande toute la
méthode — à ce niveau, un modèle qui répond toujours « non » a 99,98 % de
justesse.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  l'évolution du danger sur 53 ans
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

st.info(
    "**Et pourtant le nombre de feux ne monte pas.** Mesuré sur 2006-2025 : "
    "le FWI progresse de +24 % et les jours de danger de +59 %, mais les feux "
    "restent stables (pente +3 % sur 20 ans, p = 0,89). La prévention et les "
    "moyens de lutte absorbent la dégradation climatique. La dérive résiduelle "
    "— ce que la météo n'explique pas — va dans le sens d'une amélioration "
    "(−20 % sur 20 ans) mais n'est pas significative (p = 0,27).")

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
c'est ce que le modèle exploite : **le maquis est sa feature n°1, à 26,2 %
d'importance**, devant le danger météo.

*La météo dit quand, le territoire dit où.*
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  le modèle
# ════════════════════════════════════════════════════════════════════════
st.markdown("## Ce que vaut le modèle")

st.markdown(f"""
| Modèle | PR-AUC | lift | déployable |
|---|---|---|---|
| A · 52 features, historique inclus | {MT['modele_a']['pr_auc']} | {MT['modele_a']['lift']}× | non |
| **C · physique pur, 41 features** | **{MT['test']['pr_auc']}** | **{MT['test']['lift']}×** | **oui** |

Mesuré sur **{MT['test']['lignes']:,} communes-jours** de {MT['test']['periode']},
jamais utilisés pendant la construction du modèle — ni pour choisir les
variables, ni les réglages, ni la méthode de calibration.

Le modèle A est meilleur, mais il tire 29 % de son importance de l'historique
récent des feux. Or **la BDIFF ne publie pas l'année en cours** : les données
2026 ne sortiront qu'au printemps 2027. Un modèle qui a besoin des feux de la
semaine dernière ne peut pas prédire la semaine prochaine.

C'est le modèle C qui tourne dans cette application.
""".replace(",", " "))

imp = pd.read_csv(N.DON / "importances_c.csv", index_col=0).squeeze("columns")
fig, ax = plt.subplots(figsize=(11, 3.4))
top = (100 * imp.head(14)).iloc[::-1]
coul = ["#8b5a2b" if "maquis" in f else
        N.ORANGE if f in ("fwi", "ffmc", "dmc", "dc", "bui", "isi", "kbdi",
                          "erc", "danger_effis", "fwi_j1", "ffmc_j1") else
        "#7cc4a0" if f.startswith("part_") else N.BLEU for f in top.index]
ax.barh(range(len(top)), top.to_numpy(), color=coul, edgecolor=N.FOND, lw=1.2)
for i, v in enumerate(top):
    ax.text(v + .3, i, f"{v:.1f} %", va="center", fontsize=8.5, weight="bold")
ax.set_yticks(range(len(top)))
ax.set_yticklabels([f.replace("_", " ") for f in top.index], fontsize=8.5)
ax.set_xlim(0, top.max() * 1.15)
ax.set_xlabel("importance (%)")
ax.set_title("Ce qui pèse dans la décision du modèle", fontsize=10.5,
             weight="bold", loc="left")
ax.grid(axis="x", color=N.GRID, lw=.7); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

st.markdown("""
### Les limites, en clair

- **la surface brûlée n'est pas prédictible** — R² de 0,14, moins bon que
  d'annoncer toujours la médiane. La taille dépend surtout de ce qui se passe
  *après* le départ : vent, délai d'intervention, relief. En revanche
  « sera-ce un grand feu de plus de 5 hectares ? » se prédit à 0,77 de ROC-AUC ;
- **une commune-jour n'est pas un incendie** — un feu traversant cinq communes
  compte cinq fois. Les 49 130 « feux » sont des communes-jours ayant brûlé ;
- **~31 communes partagent une maille météo** — elles ont le même FWI le même
  jour. Le FWI porte le *quand*, la végétation porte le *où* ;
- **le score affiché est un rang, pas une probabilité** — la calibration
  disponible serait fausse d'un facteur ~2 ;
- **le modèle suppose stable tout ce qui n'est pas la météo** : prévention,
  pratiques agricoles, déprise rurale.
""")
