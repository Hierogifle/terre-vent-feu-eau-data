"""Tout ce qui a été comparé, et pourquoi c'est ce modèle-là qui tourne.

Six modèles, un protocole, et deux conclusions contre-intuitives :

    1. le MEILLEUR modèle n'est pas le déployable
    2. un LSTM correctement optimisé perd contre un gradient boosting qui
       voit vingt fois moins d'historique météo

Aucune des deux ne s'admet sans preuve. Cette page les donne.
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
TAUX_VAL = 0.0002410          # taux de positifs de la validation


@st.cache_data(show_spinner=False)
def csv(nom: str) -> pd.DataFrame | None:
    f = N.DON / nom
    return pd.read_csv(f) if f.exists() else None


# ════════════════════════════════════════════════════════════════════════
#  1. le protocole
# ════════════════════════════════════════════════════════════════════════
st.markdown("## Le protocole, avant les résultats")

st.markdown(f"""
Un événement à **0,019 %** ne pardonne pas l'à-peu-près : une fuite de données
n'y produit pas d'erreur, elle produit d'excellentes métriques et un modèle
sans valeur. Trois décisions structurent tout.
""")

p1, p2, p3 = st.columns(3)

with p1:
    st.markdown("##### La barrière du split")
    s = MT["splits"]
    st.markdown(f"""
| Partition | Années | Rôle |
|---|---|---|
| train | {s['train'][0]}-{s['train'][1]} | apprendre |
| validation | {s['val'][0]}-{s['val'][1]} | choisir |
| **test** | **{s['test'][0]}-{s['test'][1]}** | **juger, une fois** |

Découpage **temporel**, jamais aléatoire : un découpage au hasard mettrait le
14 juillet 2019 dans le train et le 15 dans le test, et le modèle
« prédirait » un feu qu'il a déjà vu à 20 km.
""")

with p2:
    st.markdown("##### PR-AUC, et pas ROC-AUC")
    st.markdown("""
À 0,019 % de positifs, la ROC-AUC est **flatteuse et inutile** : les vrais
négatifs écrasent tout, et un modèle médiocre affiche 0,95.

La **PR-AUC** vaut exactement le taux de base quand on répond au hasard. Le
rapport des deux — le **lift** — se lit directement : « le modèle est N fois
meilleur que tirer au sort ».
""")

with p3:
    st.markdown("##### Le prior déplacé ×487")
    st.markdown("""
Le train est sous-échantillonné **1:10** — tous les positifs, dix négatifs
par positif. Le modèle apprend donc sur un monde à **9,1 %** de feux alors
que le vrai taux est **0,019 %**.

Validation et test, eux, ne sont **jamais** échantillonnés : c'est ce qui
rend les scores lisibles, et ce que la calibration doit absorber.
""")

with st.expander("Le piège qui ne se voit dans aucune métrique"):
    st.markdown("""
Toute statistique dérivée de `y` — lissage bayésien, taux par cluster — doit
se calculer sur le **train complet**, pas sur le train échantillonné. Sur
l'échantillon, le taux de positifs vaut 9,1 % au lieu de 0,019 % : un facteur
**487**. Le prior serait empoisonné, `k` n'aurait plus de sens, et **rien dans
les métriques ne le signalerait**.

Deuxième règle, plus subtile : *une feature datée peut regarder tout le passé,
y compris celui de sa propre période d'évaluation ; une statistique non datée
ne peut regarder que le train.* « Feux des 30 jours précédents » au 3 août 2023
lit juillet 2023 — ce n'est pas une fuite, le 3 août à 8 h on connaît juillet.
« Taux moyen de la commune sur toute la période » lit le futur.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  2. les baselines — savoir contre quoi on se bat
# ════════════════════════════════════════════════════════════════════════
BASE = csv("baselines.csv")
if BASE is not None:
    st.markdown("## Contre quoi se bat-on ?")
    st.caption("Trois prédicteurs sans apprentissage, mesurés sur la même "
               "validation. Sans eux, un lift de 73× ne veut rien dire.")

    fig, ax = plt.subplots(figsize=(11, 2.9))
    b = BASE.sort_values("lift")
    coul = [N.GRIS if "hasard" in p else N.BLEU for p in b.predicteur]
    ax.barh(range(len(b)), b.lift, color=coul, edgecolor=N.FOND, lw=1.2)
    ax.barh([len(b)], [N.meta()["test"]["lift"]], color=N.VERT,
            edgecolor=N.FOND, lw=1.2)
    noms = list(b.predicteur) + ["4 · le modèle déployé (test)"]
    for i, v in enumerate(list(b.lift) + [N.meta()["test"]["lift"]]):
        ax.text(v + .8, i, f"×{v:.1f}", va="center", fontsize=9, weight="bold")
    ax.set_yticks(range(len(noms)))
    ax.set_yticklabels([n.replace("·", "·") for n in noms], fontsize=9)
    ax.set_xlabel("lift — combien de fois mieux que le hasard")
    ax.set_xlim(0, max(list(b.lift) + [N.meta()["test"]["lift"]]) * 1.16)
    ax.grid(axis="x", color=N.GRID, lw=.7); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

    st.markdown("""
**L'histoire seule vaut déjà ×19, la météo seule ×5, et leur croisement ×42.**
C'est la barre à battre — pas le hasard. Un modèle qui ferait ×30 serait *moins
bon qu'une règle de trois*.
""")
    st.divider()

# ════════════════════════════════════════════════════════════════════════
#  3. les six modèles
# ════════════════════════════════════════════════════════════════════════
st.markdown("## Les six modèles, et ce qui les sépare vraiment")

PR, CMP = csv("pr_auc_val.csv"), csv("comparaison_appariee.csv")
ENS, LSTM = csv("modeles_ensemble.csv"), csv("modeles_lstm.csv")

if PR is None or CMP is None:
    st.error("`pr_auc_val.csv` ou `comparaison_appariee.csv` manquant — "
             "lancer `python -m tvfed.comparer` puis `tvfed.export_app`.")
    st.stop()

ap = PR.iloc[0].to_dict()

DESCR = {
    "XGBoost v3": "52 features, historique des feux inclus + clustering territorial",
    "DART": "même chose, avec abandon d'arbres (dropout) à l'entraînement",
    "MLP": "réseau dense 3 couches, dropout, mêmes 52 features",
    "XGBoost C": "**41 features, physique pure** — aucune donnée dérivée des feux",
    "LSTM": "30 jours × 8 indices météo en séquence + 30 features de territoire",
}
T = pd.DataFrame([{
    "Modèle": k, "PR-AUC": round(v, 4), "lift": f"×{v / TAUX_VAL:.1f}",
    "Ce que c'est": DESCR.get(k, "")} for k, v in
    sorted(ap.items(), key=lambda x: -x[1])])
st.dataframe(T, width="stretch", hide_index=True)

st.caption(f"Mesuré sur la **validation** — {38_068_464:,} communes-jours, "
           f"9 176 feux, taux {TAUX_VAL:.4%}. "
           f"L'ensemble v3 + MLP monte à {ENS.pr_auc.iloc[0]:.4f} "
           f"(×{ENS.lift.iloc[0]:.1f}) si l'on accepte de faire tourner deux "
           f"modèles.".replace(",", " ") if ENS is not None else "")

# ── le graphique en forêt ────────────────────────────────────────────────
st.markdown("### Ces écarts survivent-ils au bruit ?")
st.markdown("""
Un écart de PR-AUC ne veut **rien dire** sans intervalle de confiance. Ceux-ci
viennent d'un **bootstrap apparié** à 200 répliques qui rééchantillonne les
**34 734 communes**, pas les lignes : les 1 096 jours d'une même commune ne
sont pas indépendants, et 31 communes partagent en moyenne la même maille
météo. Un bootstrap ligne à ligne produirait des intervalles faussement
étroits.
""")

c = CMP[CMP.reference == "XGBoost v3"].copy()
sup = CMP[(CMP.reference == "XGBoost C") & (CMP.modele == "LSTM")]
c = pd.concat([c, sup]).iloc[::-1].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11.5, 3.6))
for i, r in c.iterrows():
    coul = N.ROUGE if r.significatif else N.MUTED
    ax.plot([r.ic_bas, r.ic_haut], [i, i], color=coul, lw=2.6,
            solid_capstyle="round", zorder=2)
    ax.plot(r.ecart_pct, i, "o", color=coul, ms=7, zorder=3)
    ax.text(r.ic_haut + 1.5, i,
            f"{r.ecart_pct:+.1f} %"
            + ("" if r.significatif else "   ← non significatif"),
            va="center", fontsize=9, color=coul,
            weight="bold" if r.significatif else "normal")
ax.axvline(0, color=N.INK, lw=1.2)
ax.set_yticks(range(len(c)))
ax.set_yticklabels([f"{r.modele}  vs  {r.reference}" for _, r in c.iterrows()],
                   fontsize=9)
ax.set_xlabel("écart de PR-AUC (%) — intervalle de confiance à 95 %")
ax.set_xlim(-72, 30)
ax.grid(axis="x", color=N.GRID, lw=.7); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

st.success("""
**Le résultat le plus utile de cette page est un résultat négatif.**

DART et le MLP paraissaient 1,8 % et 1,9 % moins bons que XGBoost v3. Leurs
intervalles **traversent zéro** : les trois modèles sont **indiscernables**.
Annoncer « XGBoost bat le MLP » aurait été une conclusion inventée à partir du
bruit — c'est exactement l'erreur que ce bootstrap existe pour empêcher.

Seuls deux écarts survivent : celui du modèle C, et celui du LSTM.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  4. le LSTM
# ════════════════════════════════════════════════════════════════════════
st.markdown("## Le LSTM — pourquoi « prends un LSTM » ne suffit pas")

g, d_ = st.columns([1.15, 1])

with g:
    st.markdown("""
« Pour le temps, prends un LSTM » est le réflexe standard. Le projet l'a donc
construit, optimisé, et mesuré. **Il perd de 23,6 %** — intervalle
[−33,5 ; −17,3], loin de zéro.

##### Ce n'est pas un LSTM bâclé
""")
    bp = N.DON / "best_params_lstm.json"
    if bp.exists():
        P = json.loads(bp.read_text(encoding="utf-8"))
        st.markdown("**25 essais Optuna** sur sept hyperparamètres, arrêt "
                    "précoce à l'époque 21 :")
        # tout en texte : une colonne mêlant entiers et chaînes ne se
        # sérialise pas en Arrow, et Streamlit la répare en émettant un
        # avertissement à chaque affichage
        st.dataframe(pd.DataFrame([{"Hyperparamètre": k,
                                    "Valeur retenue": f"{v:.5g}"}
                                   for k, v in P.items()]),
                     width="stretch", hide_index=True)

with d_:
    st.markdown("""
##### La comparaison loyale n'est pas celle qu'on croit

XGBoost v3 voit l'historique des feux — **29 % de ses importances**. Le LSTM
n'en voit rien. Les opposer mesurerait le prix de l'information retirée, pas
la valeur de la séquence.

La seule référence à jeu d'information égal est le **modèle C** :

| | modèle C | LSTM |
|---|---|---|
| territoire, calendrier | 30 features | **les mêmes 30** |
| météo | 8 indices du jour + `danger_effis` + 2 décalages | **30 jours × 8 = 240 valeurs** |

Le LSTM voit donc **vingt fois plus d'historique météo**, et perd quand même.
""")

st.error("""
##### Pourquoi il perd — l'explication est physique, pas informatique

Un LSTM sert quand **l'ordre de la séquence porte une information qu'aucun
résumé ne capture**. Ici, ce résumé existe déjà.

Les indices `DC`, `DMC` et `BUI` du système canadien **sont** des états
récursifs. Le *Drought Code* est littéralement une moyenne exponentielle de la
météo passée avec une constante de temps de **52 jours** ; le *Duff Moisture
Code*, de **15 jours**. C'est exactement la forme d'une cellule récurrente —
sauf que ses coefficients ont été calibrés par cinquante ans de science du feu
plutôt qu'estimés sur 9 176 exemples positifs.

**Le CEMS livre déjà l'état caché.** Demander au LSTM de le réapprendre depuis
30 jours de séries brutes, c'est lui demander de redécouvrir une solution
qu'on lui donne en entrée sous forme fermée.
""")

st.markdown("""
Trois observations indépendantes convergent — voir la page *Les données* :

- la **PACF** montre une autocorrélation épuisée en **deux à trois jours** ;
- l'**ARIMA sans exogène** est inutilisable, **r = −0,118** : le passé des feux
  ne prédit pas leur futur ;
- les trois premières features du modèle C sont `part_maquis` (26,2 %),
  `danger_effis` (13,7 %) et `erc` (11,1 %) — le signal dit **où il y a du
  combustible**, pas ce qui s'est passé le mois dernier.

Ce problème n'est pas une prévision de série temporelle. C'est une
**classification spatio-temporelle d'événement rare**, sur 34 734 séries
parallèles pilotées par un exogène déjà résumé par la physique du domaine.

*Réserve honnête* : le LSTM ne reçoit pas `danger_effis`, qui pèse 13,7 % dans
le modèle C. L'écart de 23,6 % est donc un **majorant**.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  5. v3 contre C
# ════════════════════════════════════════════════════════════════════════
st.markdown("## Le meilleur modèle n'est pas celui qu'on déploie")

t = MT["test"]
a = MT["modele_a"]
m1, m2, m3 = st.columns(3)
m1.metric("XGBoost v3 sur le test", f"×{a['lift']:.1f}", "le meilleur")
m2.metric("Modèle C sur le test", f"×{t['lift']:.1f}",
          f"{100 * (t['lift'] / a['lift'] - 1):+.0f} %", delta_color="off")
m3.metric("En territoire jamais vu", "C gagne 9 fois sur 9",
          "+8,2 % pondéré")

st.markdown("""
v3 est **nettement meilleur** sur le test. Et pourtant c'est C qui tourne dans
cette application. Trois raisons, dont aucune n'est visible dans une métrique
d'entraînement.
""")

r1, r2, r3 = st.columns(3)
r1.warning("**1. La donnée n'existe pas en temps réel.**\n\n"
           "v3 tire 29 % de son importance de l'historique récent des feux. "
           "Or la BDIFF ne publie pas l'année en cours : les feux de 2026 "
           "sortiront au printemps 2027. Pour une prédiction faite "
           "aujourd'hui, `feux_commune_7j` vaudrait le décompte d'une semaine "
           "de décembre 2025. Pas imprécis : **faux**.")
r2.warning("**2. En territoire inconnu, elle vaut zéro.**\n\n"
           "Et le modèle lit ce zéro comme « ça n'a jamais brûlé, donc ça ne "
           "brûlera pas ». C'est précisément là que le risque nouveau "
           "apparaît.")
r3.warning("**3. Pour 2050, elle est impossible.**\n\n"
           "On ne connaîtra jamais les feux de 2049. Un modèle qui en dépend "
           "ne peut rien dire de l'avenir — c'est-à-dire de la question "
           "centrale du projet.")

# ── la validation croisée spatiale ───────────────────────────────────────
TS = csv("transfert_spatial.csv")
if TS is not None:
    st.markdown("### La preuve : on retire une région entière, puis on teste dessus")
    st.caption("Chaque ligne : le modèle n'a JAMAIS vu cette région pendant "
               "l'entraînement. C'est la simulation d'un territoire nouveau — "
               "ou d'un climat qui déplace le risque.")

    ts = TS.copy()
    ts["ecart"] = 100 * (ts["C · physique"] / ts["A · tout"] - 1)
    ts = ts.sort_values("ecart")

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 3.8), width_ratios=[1.25, 1])
    y = np.arange(len(ts))
    ax[0].barh(y - .2, ts["A · tout"], height=.38, color=N.GRIS,
               label="A · toutes features", edgecolor=N.FOND, lw=.8)
    ax[0].barh(y + .2, ts["C · physique"], height=.38, color=N.VERT,
               label="C · physique pure", edgecolor=N.FOND, lw=.8)
    ax[0].set_yticks(y, [f"région {int(r)}" for r in ts.region], fontsize=8.5)
    ax[0].set_xlabel("PR-AUC sur la région exclue")
    ax[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    ax[0].set_title("Le modèle physique gagne partout", fontsize=10.5,
                    weight="bold", loc="left")

    ax[1].barh(y, ts.ecart, color=N.VERT, edgecolor=N.FOND, lw=.8)
    for i, v in enumerate(ts.ecart):
        ax[1].text(v + 2, i, f"{v:+.0f} %", va="center", fontsize=8.5,
                   weight="bold")
    ax[1].set_yticks(y, [""] * len(y))
    ax[1].axvline(0, color=N.INK, lw=1)
    ax[1].set_xlim(0, ts.ecart.max() * 1.22)
    ax[1].set_xlabel("avantage du modèle C (%)")
    ax[1].set_title("… et parfois très largement", fontsize=10.5,
                    weight="bold", loc="left")
    for a_ in ax:
        a_.grid(axis="x", color=N.GRID, lw=.7); a_.set_axisbelow(True)
        a_.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

    pire = ts.iloc[-1]
    st.markdown(f"""
**C gagne dans les 9 régions, sans exception**, et jusqu'à
**{pire.ecart:+.0f} %** dans la région {int(pire.region)} — là où l'historique
est le plus pauvre, s'y fier est un handicap.

Le choix ne se fait donc **pas sur la performance mais sur la disponibilité de
la donnée**. Et c'est ce qui autorise à mesurer les deux sur le test sans
corrompre le protocole : ils répondent à deux situations différentes, pas à la
même question.

> **Vous pouvez le voir vous-même** : la page *Carte* permet de basculer entre
> les deux modèles sur une date de {MT['splits']['test'][0]}-{MT['splits']['test'][1]},
> et refuse de le faire ailleurs.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  6. la calibration
# ════════════════════════════════════════════════════════════════════════
CAL = csv("calibration_v3.csv")
if CAL is not None:
    st.markdown("## Pourquoi l'application affiche un rang, pas une probabilité")
    g, d_ = st.columns([1, 1.1])
    with g:
        st.dataframe(CAL.assign(**{
            "Méthode": CAL.methode,
            "PR-AUC": CAL.pr_auc.round(4),
            "p moyen": CAL.p_moyen.map(lambda v: f"{v:.2e}"),
            "Biais": CAL.biais.map(lambda v: f"×{v:.1f}"),
        })[["Méthode", "PR-AUC", "p moyen", "Biais"]],
            width="stretch", hide_index=True)
    with d_:
        st.markdown(f"""
Le score brut est **{CAL.biais.iloc[0]:.0f} fois trop grand** : c'est le
sous-échantillonnage 1:10 qui remonte le prior de 0,019 % à 9,1 %.

Platt le ramène à ×{CAL.biais.iloc[1]:.2f} sans rien coûter en PR-AUC —
**le classement est intact**, seule l'échelle bouge. L'isotonique calibre
aussi bien mais écrase le score sur {int(CAL.valeurs_distinctes.iloc[2])}
valeurs distinctes et **perd du pouvoir de discrimination**.

L'application affiche donc un **rang**, pas une probabilité : le calibrateur
disponible a été ajusté sur un autre modèle et une autre période, et serait
faux d'un facteur ~2. Un rang, lui, reste juste.
""")
    st.divider()

# ════════════════════════════════════════════════════════════════════════
#  7. l'évaluation test
# ════════════════════════════════════════════════════════════════════════
TPA = csv("test_par_annee.csv")
st.markdown("## L'évaluation finale — ouverte une seule fois")

if TPA is not None:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.dataframe(TPA.assign(**{
            "Année": TPA.an, "Feux": TPA.feux,
            "PR-AUC": TPA.pr_auc.round(4),
            "lift": TPA.lift.map(lambda v: f"×{v:.0f}"),
        })[["Année", "Feux", "PR-AUC", "lift"]],
            width="stretch", hide_index=True)
    with c2:
        pire, meilleure = TPA.loc[TPA.lift.idxmin()], TPA.loc[TPA.lift.idxmax()]
        st.markdown(f"""
**Le lift varie du simple au double d'une année à l'autre**, et ce n'est pas
du bruit : il suit la **rareté**. {int(meilleure.an)} est l'année la plus calme
({int(meilleure.feux):,} feux) et donne le **meilleur** lift (×{meilleure.lift:.0f}) ;
{int(pire.an)}, la plus active, le plus faible (×{pire.lift:.0f}).

C'est contre-intuitif et c'est instructif : **une année calme concentre les
feux dans les endroits les plus prévisibles**. Quand tout brûle, y compris là
où ce n'est pas censé arriver, le modèle est pris en défaut.

⚠️ Ce chiffre a été mesuré **une seule fois**, après gel complet du modèle,
des features et de la calibration.
""".replace(",", " "))

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  8. l'erreur qu'on a faite
# ════════════════════════════════════════════════════════════════════════
st.markdown("## Une erreur qu'on a commise, et comment elle a été trouvée")

st.markdown("""
Le premier verdict annoncé pour le LSTM était **−97 %**. Le vrai est
**−51,7 %**. L'écart ne venait pas du modèle mais de la façon dont deux
fichiers de prédictions étaient comparés.

`sql/50_matrice.sql` n'a **aucun `ORDER BY`**. L'ordre dans lequel PostgreSQL
renvoie les 38 millions de lignes dépend du plan d'exécution et des workers
parallèles : il **change d'une exécution à l'autre**. Les fichiers de
prédictions ne portaient que `(score, cible)` — les comparer revenait à les
aligner **par position**.

Deux fichiers issus de deux exécutions ont la même taille, le même nombre de
feux, et un ordre différent. **Rien ne signale l'erreur.**
""")

e1, e2 = st.columns([1, 1])
with e1:
    st.code("""LSTM aligné sur les clés     PR-AUC 0.0085   ×35,4
LSTM, lignes permutées      PR-AUC 0.0002   ×1,0
                                            ↑
                            le hasard, avec les MÊMES valeurs""",
            language=None)
with e2:
    st.markdown("""
**La parade retenue** — un `ORDER BY` coûterait un tri de 38 M lignes larges à
chaque parcours :

1. tout fichier de prédictions porte ses clés `(code_insee, date)` ;
2. `tvfed.comparer.aligner()` trie dessus et **vérifie** ;
3. `tests/test_comparaison.py` **refuse** un fichier sans clés.
""")

st.info("""
**La leçon générale.** Sur un événement à 0,024 %, une erreur de plomberie ne
se manifeste jamais par une exception. Elle se manifeste par un **chiffre
plausible**. Les seules défenses sont les invariants explicites et les
assertions qui échouent bruyamment.

Les autres erreurs trouvées et corrigées : une fuite `ha`/`cible` qui donnait
R² = 0,994 sur la surface brûlée ; un biais de collision qui inversait le signe
d'une interaction ; un dentelé de période 4 ans dû aux années bissextiles ;
une tendance FWI annoncée à +45 % au lieu de +58 %.
""")

st.caption("Le détail complet est dans `notebook/series-lstm.ipynb`, "
           "`src/tvfed/comparer.py` et le README du dépôt.")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  9. les limites
# ════════════════════════════════════════════════════════════════════════
st.markdown("## Les limites, en clair")

l1, l2 = st.columns(2)
with l1:
    st.markdown("""
##### Ce que le modèle ne sait pas faire

- **La surface brûlée n'est pas prédictible.** R² de 0,14, moins bon que
  d'annoncer toujours la médiane. La taille dépend surtout de ce qui se passe
  *après* le départ : vent, délai d'intervention, relief. En revanche
  « sera-ce un grand feu de plus de 5 hectares ? » se prédit à **0,77 de
  ROC-AUC**.
- **Le score affiché est un rang, pas une probabilité.** La calibration
  disponible serait fausse d'un facteur ~2.
- **Le modèle suppose stable tout ce qui n'est pas la météo** : prévention,
  pratiques agricoles, déprise rurale. Les projections à 2100 ne font varier
  que le climat.
""")
with l2:
    st.markdown("""
##### Ce qu'il faut savoir des données

- **Une commune-jour n'est pas un incendie.** Un feu traversant cinq communes
  compte cinq fois. Les 49 130 « feux » sont des communes-jours ayant brûlé.
- **~31 communes partagent une maille météo** de 28 km : elles ont le même FWI
  le même jour. Le FWI porte le *quand*, la végétation porte le *où* —
  conséquence statistique, les intervalles de confiance naïfs sur les
  coefficients météo seraient trop étroits.
- **Les feux ne sont observés que depuis 2006** ; la météo depuis 1973. On
  compare donc une tendance sur 53 ans à une stabilité sur 20.
- **1 378 feux d'outre-mer exclus**, hors couverture météo européenne, et
  **30 feux métropolitains** non rattachables — comptés, jamais devinés.
""")
