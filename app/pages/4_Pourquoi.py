"""Pourquoi un feu part — et pourquoi trois outils d'explication ne disent
pas la même chose.

    SHAP   exact sur un modèle d'arbres, décompose CE score
    LIME   substitut linéaire local, approché — utile pour le comparer
    DiCE   le contrefactuel : qu'aurait-il fallu changer ?

⚠️ Le SHAP affiché ici porte sur le MODÈLE C, celui qui dessine la carte.
Le projet dispose aussi d'un SHAP du modèle v3 (52 features) : le montrer ici
décrirait un modèle que l'utilisateur ne voit jamais, et l'erreur serait
invisible — mêmes noms de features, même allure de graphique.
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

st.set_page_config(page_title="Pourquoi · Risque incendie", page_icon="🔥",
                   layout="wide")
plt.rcParams.update({"figure.facecolor": N.FOND, "axes.facecolor": N.FOND,
                     "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "text.color": N.INK, "xtick.color": N.MUTED,
                     "ytick.color": N.MUTED, "axes.labelcolor": N.INK})
N.entete()

MT = N.meta()
FEATURES = MT["features"]

# noms lisibles, pour ne pas afficher `part_veg_clairsemee` à un jury
JOLI = {
    "part_maquis": "part de maquis", "part_foret": "part de forêt",
    "part_coniferes": "part de conifères", "part_feuillus": "part de feuillus",
    "part_melangees": "forêt mélangée", "part_landes": "landes",
    "part_veg_mutation": "végétation en mutation",
    "part_veg_clairsemee": "végétation clairsemée",
    "part_combustible": "part combustible", "part_agricole": "part agricole",
    "part_artificialise": "part urbanisée", "clc_millesime": "millésime CORINE",
    "distance_cote_km": "distance à la côte (km)",
    "log_superficie": "superficie (log)", "log_population": "population (log)",
    "log_densite": "densité (log)", "altitude_moy": "altitude moyenne",
    "amplitude_altitude": "relief (max − min)",
    "grille_densite": "typologie de densité",
    "fwi": "FWI du jour", "ffmc": "FFMC — litière fine",
    "dmc": "DMC — humus (15 j)", "dc": "DC — sécheresse profonde (52 j)",
    "bui": "BUI — combustible disponible", "isi": "ISI — vitesse de propagation",
    "kbdi": "KBDI — sécheresse du sol", "erc": "ERC — énergie libérable",
    "danger_effis": "classe de danger EFFIS", "fwi_j1": "FWI de la veille",
    "ffmc_j1": "FFMC de la veille", "doy": "jour de l'année",
    "mois": "mois", "jour_semaine": "jour de la semaine",
    "est_weekend": "week-end", "est_ferie": "jour férié",
    "est_14_juillet": "14 juillet", "est_15_aout": "15 août",
    "sin_doy": "saison (sin)", "cos_doy": "saison (cos)",
    "sin_mois": "mois (sin)", "cos_mois": "mois (cos)",
}


def joli(f: str) -> str:
    return JOLI.get(f, f.replace("_", " "))


def rang(n) -> str:
    """« 1ᵉʳ », « 2ᵉ », « 30ᵉ » — le premier ne se dit pas comme les autres."""
    return "1ᵉʳ" if int(n) == 1 else f"{int(n)}ᵉ"


@st.cache_data(show_spinner=False)
def shap_c(nom: str):
    v = np.load(N.DON / f"shap_c_{nom}.npy")
    X = pd.read_parquet(N.DON / f"shap_c_{nom}_X.parquet")
    cols = json.loads((N.DON / "shap_c_colonnes.json").read_text(encoding="utf-8"))
    return v, X, cols


@st.cache_data(show_spinner=False)
def fond() -> pd.DataFrame:
    return pd.read_parquet(N.DON / "fond_dice.parquet")


if not (N.DON / "shap_c_alea.npy").exists():
    st.error("Les explications du modèle C ne sont pas encore produites.\n\n"
             "    python -m tvfed.explications\n"
             "    python -m tvfed.export_app")
    st.stop()

# ════════════════════════════════════════════════════════════════════════
#  1. trois façons de demander « qu'est-ce qui compte »
# ════════════════════════════════════════════════════════════════════════
st.markdown("## Qu'est-ce qui fait partir un feu ?")

st.markdown("""
Trois mesures répondent à cette question et ne donnent pas le même classement.
Ce n'est pas que l'une se trompe : elles ne mesurent pas la même chose. Autant
le voir avant de lire les graphiques.
""")

V_A, X_A, COLS = shap_c("alea")
V_S, X_S, _ = shap_c("sommet")

gain = 100 * pd.read_csv(N.DON / "importances_c.csv", index_col=0).squeeze("columns")
sh_a = pd.Series(np.abs(V_A).mean(0), index=COLS)
sh_s = pd.Series(np.abs(V_S).mean(0), index=COLS)

D = pd.DataFrame({"gain": gain, "alea": sh_a, "sommet": sh_s})
for c in ("gain", "alea", "sommet"):
    D[f"r_{c}"] = D[c].rank(ascending=False).astype(int)

c1, c2, c3 = st.columns(3)
c1.markdown("""
##### 1. L'importance par *gain*
Combien chaque variable a réduit la perte pendant l'entraînement.

C'est la sortie par défaut de XGBoost. Elle penche vers les variables qui
découpent proprement, et elle ne sait pas voir la redondance.
""")
c2.markdown("""
##### 2. SHAP sur un échantillon aléatoire
Combien chaque variable déplace le score sur le territoire tel qu'il est,
c'est-à-dire 99,97 % de communes-jours sans feu.

La lecture à retenir pour « en général ».
""")
c3.markdown("""
##### 3. SHAP sur le sommet du classement
Combien elle déplace le score là où le modèle s'engage.

La lecture à retenir pour « quand ça brûle ». Sélectionner sur le score
introduit un biais de collision : ce panneau ne sert qu'à cette question-là.
""")

# ── le graphique de comparaison ──────────────────────────────────────────
top = D.sort_values("sommet", ascending=False).head(14).iloc[::-1]
fig, ax = plt.subplots(1, 3, figsize=(15.5, 5), sharey=True)
y = np.arange(len(top))
for a_, col, titre, coul in (
        (ax[0], "gain", "1 · gain d'entraînement (%)", N.GRIS),
        (ax[1], "alea", "2 · SHAP, échantillon aléatoire", N.BLEU),
        (ax[2], "sommet", "3 · SHAP, sommet du classement", N.ROUGE)):
    a_.barh(y, top[col], color=coul, edgecolor=N.FOND, lw=1)
    a_.set_title(titre, fontsize=10.5, weight="bold", loc="left")
    a_.grid(axis="x", color=N.GRID, lw=.7); a_.set_axisbelow(True)
    a_.spines[["top", "right"]].set_visible(False)
ax[0].set_yticks(y, [joli(f) for f in top.index], fontsize=9)
plt.tight_layout()
st.pyplot(fig, width='stretch')
plt.close(fig)

de = D.loc["danger_effis"]
ma = D.loc["part_maquis"]
st.markdown(f"""
##### Deux désaccords à expliquer

La classe de danger EFFIS arrive {rang(de.r_gain)} par gain
({N.dec(de.gain, 1)} %) et seulement {rang(de.r_alea)} par SHAP. C'est une
discrétisation du FWI en six classes : XGBoost trouve ces seuils nets commodes
pour découper, ce qui lui vaut un gain élevé, mais l'information est déjà dans
le FWI continu et SHAP en attribue le crédit à ce dernier. Les deux mesures ne
se contredisent pas, elles traitent la redondance différemment.

La part de maquis arrive {rang(ma.r_gain)} par gain, {rang(ma.r_alea)} sur
l'échantillon aléatoire et {rang(ma.r_sommet)} au sommet du classement. Sur une
commune-jour tirée au hasard il n'y a pas de maquis, donc la variable ne déplace
rien ; là où le modèle voit du risque, elle devient déterminante.

Conséquence pratique : citer « la variable la plus importante » n'a pas de sens
sans préciser quelle mesure et sur quelle population.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  2. le nuage SHAP
# ════════════════════════════════════════════════════════════════════════
st.markdown("## Dans quel sens chaque variable pousse-t-elle ?")

pop = st.radio("Population", ["Le sommet du classement", "Un échantillon aléatoire"],
               horizontal=True, label_visibility="collapsed")
V, X = (V_S, X_S) if pop.startswith("Le sommet") else (V_A, X_A)

st.caption("Chaque point est une commune-jour. À droite du zéro la variable "
           "augmente le risque, à gauche elle le diminue. La couleur donne la "
           "valeur de la variable : rouge pour élevée, bleu pour basse.")

ordre = pd.Series(np.abs(V).mean(0), index=COLS).nlargest(12).index[::-1]
fig, ax = plt.subplots(figsize=(11.5, 5.4))
rng = np.random.default_rng(0)
for i, f in enumerate(ordre):
    k = COLS.index(f)
    s = V[:, k]
    val = X[f].to_numpy(dtype=float)
    # normalisation robuste : les queues écraseraient l'échelle de couleur
    lo, hi = np.nanpercentile(val, [5, 95])
    coul = np.clip((val - lo) / (hi - lo + 1e-9), 0, 1)
    ax.scatter(s, i + rng.uniform(-.28, .28, len(s)), c=coul, cmap="coolwarm",
               s=3.2, alpha=.45, edgecolors="none", rasterized=True)
ax.axvline(0, color=N.INK, lw=1)
ax.set_yticks(range(len(ordre)), [joli(f) for f in ordre], fontsize=9)
ax.set_xlabel("effet sur le score (valeur SHAP)")
ax.grid(axis="x", color=N.GRID, lw=.7)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
sm = plt.cm.ScalarMappable(cmap="coolwarm")
cb = fig.colorbar(sm, ax=ax, pad=.015, fraction=.022)
cb.set_ticks([0, 1]); cb.set_ticklabels(["basse", "élevée"])
cb.set_label("valeur de la variable", fontsize=8.5)
plt.tight_layout()
st.pyplot(fig, width='stretch')
plt.close(fig)

st.markdown("""
Les variables de territoire (maquis, distance à la côte, superficie, relief)
occupent le haut du classement à égalité avec les indices météo. Ni l'une ni
l'autre famille ne suffit seule : la météo situe le moment, le territoire situe
le lieu.

Les baselines allaient déjà dans ce sens. Le danger EFFIS seul vaut 5 fois le
hasard, l'historique spatial seul 19 fois, et leur croisement 42 fois.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  3. une commune, un jour — les trois explainers
# ════════════════════════════════════════════════════════════════════════
st.markdown("## Une commune, un jour : trois façons d'expliquer le même score")

s1, s2, s3 = st.columns([2, 1, 1])
with s1:
    q = st.text_input("Commune ou code postal", "Bormes-les-Mimosas",
                      key="q_expl")
with s2:
    mois = st.selectbox("Mois", range(1, 13), index=7,
                        format_func=lambda m: N.MOIS[m - 1])
with s3:
    jour = st.number_input("Jour", 1, 31, 12, step=1)

trouvees = N.chercher(q) if q else None
if trouvees is None or not len(trouvees):
    st.info("Aucune commune ne correspond. Essayez « Bormes » ou « 83230 ».")
    st.stop()

sel = trouvees.iloc[0]
date = pd.Timestamp(year=2024, month=int(mois), day=min(int(jour), 28))
st.caption(f"{sel.nom} ({sel.dep_nom}), {N.date_fr(date)}. "
           f"L'année est fixée à 2024 : on explique le modèle, pas une "
           f"projection.")

R = N.predire(date, nom="C")
ligne = R[R.code_insee == sel.code_insee]
if ligne.empty:
    st.error("Cette commune n'est pas dans le périmètre métropolitain.")
    st.stop()
ligne = ligne.iloc[0]

x = pd.DataFrame([ligne[FEATURES].to_numpy(dtype=float)], columns=FEATURES)

m1, m2, m3 = st.columns(3)
m1.metric("Score du modèle", f"{ligne.score:.4f}")
m2.metric("Rang national", f"{100 * ligne.rang:.1f}ᵉ percentile")
m3.metric("FWI ce jour-là", f"{ligne.fwi:.1f}",
          N.CLASSES[int(ligne.danger_effis)])

o1, o2, o3 = st.tabs(["SHAP — exact", "LIME — approché",
                      "DiCE — le contrefactuel"])

# ── SHAP local ───────────────────────────────────────────────────────────
with o1:
    import shap

    expl = shap.TreeExplainer(N.modele("C"))
    v = expl.shap_values(x)
    if isinstance(v, list):
        v = v[1]
    elif v.ndim == 3:
        v = v[:, :, 1]
    loc = pd.Series(v[0], index=FEATURES).sort_values(key=abs, ascending=False)
    t = loc.head(12).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.barh(range(len(t)), t.to_numpy(),
            color=[N.ROUGE if x_ > 0 else N.BLEU for x_ in t],
            edgecolor=N.FOND, lw=1)
    ax.axvline(0, color=N.INK, lw=1)
    ax.set_yticks(range(len(t)),
                  [f"{joli(f)} = {ligne[f]:.4g}" for f in t.index], fontsize=8.5)
    ax.set_xlabel("effet sur le score (log-odds)")
    ax.set_title("Ce qui pousse ce score vers le haut (rouge) ou le bas (bleu)",
                 fontsize=10.5, weight="bold", loc="left")
    ax.grid(axis="x", color=N.GRID, lw=.7); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

    st.markdown("""
Sur un modèle d'arbres, TreeSHAP ne procède pas par échantillonnage : il
parcourt la structure des arbres et calcule la valeur de Shapley exacte, en
temps polynomial. La somme des barres, ajoutée à la valeur de base, redonne le
score du modèle au dixième de millionième près (vérifié à 9,5 × 10⁻⁷).

C'est pourquoi SHAP sert ici de référence et LIME de point de comparaison.
""")

# ── LIME ─────────────────────────────────────────────────────────────────
with o2:
    try:
        from lime.lime_tabular import LimeTabularExplainer

        F = fond()
        expl_l = LimeTabularExplainer(
            F[FEATURES].to_numpy(dtype=float), feature_names=FEATURES,
            class_names=["pas de feu", "feu"], discretize_continuous=True,
            random_state=42, mode="classification")
        m = N.modele("C")
        e = expl_l.explain_instance(
            x.to_numpy(dtype=float)[0],
            lambda z: m.predict_proba(pd.DataFrame(z, columns=FEATURES)),
            num_features=12)
        L = pd.Series(dict(e.as_list())).sort_values(key=abs, ascending=False)
        t = L.iloc[::-1]

        fig, ax = plt.subplots(figsize=(9.5, 4.4))
        ax.barh(range(len(t)), t.to_numpy(),
                color=[N.ROUGE if x_ > 0 else N.BLEU for x_ in t],
                edgecolor=N.FOND, lw=1)
        ax.axvline(0, color=N.INK, lw=1)
        ax.set_yticks(range(len(t)), list(t.index), fontsize=8)
        ax.set_xlabel("poids dans le substitut linéaire local")
        ax.set_title("LIME — les règles qui décrivent le voisinage du point",
                     fontsize=10.5, weight="bold", loc="left")
        ax.grid(axis="x", color=N.GRID, lw=.7); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

        st.markdown("""
LIME ne répond pas à la même question. Il perturbe le point, interroge le
modèle des milliers de fois, puis ajuste une régression linéaire pondérée sur
ce voisinage. Ce qu'il renvoie n'est pas la contribution exacte mais les
coefficients d'un modèle de substitution.

Trois conséquences, visibles sur le graphique ci-dessus :

- les variables apparaissent sous forme de règles (`fwi > 12,4`), parce que
  LIME discrétise ;
- le résultat est stochastique : relancez, il bougera un peu ;
- il dépend du fond qu'on lui fournit, ici 30 000 lignes du train.

Sur un modèle d'arbres, LIME approxime ce que TreeSHAP calcule exactement : il
ne peut donc pas faire mieux. Il figure ici parce qu'il est très répandu et
qu'il vaut mieux savoir pourquoi on ne l'a pas retenu. Sur un modèle qu'on ne
peut pas ouvrir, une API ou un réseau profond, il redeviendrait le bon outil.
""")
    except ImportError:
        st.warning("`lime` n'est pas installé : `uv pip install lime`.")

# ── DiCE ─────────────────────────────────────────────────────────────────
with o3:
    st.markdown("""
SHAP et LIME répondent à « pourquoi ce score ». DiCE répond à « qu'aurait-il
fallu changer », qui est la seule des trois questions dont la réponse se
traduise en décision.
""")
    try:
        import dice_ml
        from dice_ml import Dice
        from raiutils.exceptions import UserConfigValidationException

        class SeuilPercentile:
            """Le modèle C, dont la « classe 1 » veut dire *dans le décile le
            plus à risque de la journée*.

            ⚠️ SANS CE HABILLAGE, LE CONTREFACTUEL N'AURAIT PAS DE SENS.
            DiCE cherche à faire basculer la classe prédite, donc à passer
            sous **0,5**. Or le score du modèle n'est pas calibré : le
            sous-échantillonnage 1:10 gonfle le prior d'un facteur 487, et
            0,5 correspond à un risque astronomique. Demander de descendre
            sous 0,5 revient à demander l'impossible, et DiCE ne renvoie
            rien — mesuré.

            On recentre donc le seuil sur le **décile**, par une
            transformation affine par morceaux, strictement croissante : le
            classement est inchangé, seule la frontière bouge. La question
            devient « que faudrait-il pour sortir des 10 % les plus à
            risque », qui est la vraie question opérationnelle.
            """

            def __init__(self, modele, seuil):
                self.m, self.s = modele, float(seuil)
                self.classes_ = np.array([0, 1])

            def _t(self, p):
                return np.where(p <= self.s, .5 * p / max(self.s, 1e-12),
                                .5 + .5 * (p - self.s) / max(1 - self.s, 1e-12))

            def predict_proba(self, X):
                p = self._t(self.m.predict_proba(X)[:, 1])
                return np.column_stack([1 - p, p])

            def predict(self, X):
                return (self.predict_proba(X)[:, 1] >= .5).astype(int)

        seuil = float(R.score.quantile(.90))
        F = fond().copy()
        cible = "y" if "y" in F.columns else F.columns[-1]

        donnees = dice_ml.Data(dataframe=F, continuous_features=FEATURES,
                               outcome_name=cible)
        PARTS = [f for f in FEATURES if f.startswith("part_")]
        enveloppe = SeuilPercentile(N.modele("C"), seuil)
        mod = dice_ml.Model(model=enveloppe, backend="sklearn")
        moteur = Dice(donnees, mod, method="random")
        BORNES = {f: (float(F[f].min()), float(F[f].max())) for f in FEATURES}

        def atteignable(lignes, leviers, baisser):
            """Un score favorable ATTEIGNABLE, pas le meilleur possible.

            On pousse chaque levier autorisé vers l'extrémité de son domaine
            qui déplace le score dans le sens voulu, puis on évalue le point
            combinant tous ces choix. `baisser` dit dans quel sens : vrai pour
            faire sortir du décile, faux pour l'y faire entrer.

            ⚠️ LA SONDE EST UNE BORNE, ET ELLE NE VAUT QUE DANS UN SENS.
            L'optimisation se fait levier par levier ; les variables
            interagissent, donc une COMBINAISON que cette recherche gloutonne
            ne visite pas peut faire mieux. Mesuré sur Ambres : la sonde
            plafonne à 0,0405 pour un seuil de 0,0688, et DiCE trouve pourtant
            cinq contrefactuels. Augmenter la résolution n'y change rien, le
            défaut est dans le principe, pas dans le pas d'échantillonnage.

            D'où l'asymétrie à respecter dans l'interface :
              franchit      → une solution EXISTE, on en tient une explicite ;
              ne franchit pas → probablement aucune, mais sans garantie.
            La sonde ne doit donc jamais empêcher de lancer la recherche.

            Vectorisé : `lignes` peut porter une commune comme les 34 734.
            """
            m = N.modele("C")
            base = lignes[FEATURES].astype(float).reset_index(drop=True)
            best = m.predict_proba(base)[:, 1]
            combine = base.copy()
            for f in leviers:
                lo, hi = BORNES[f]
                for v in (lo, hi):
                    essai = base.copy()
                    essai[f] = v
                    p = m.predict_proba(essai)[:, 1]
                    gagne = (p < best) if baisser else (p > best)
                    combine.loc[gagne, f] = v
                    best = np.where(gagne, p, best)
            p = m.predict_proba(combine)[:, 1]
            return np.minimum(best, p) if baisser else np.maximum(best, p)

        # ⚠️ Le sens de la question dépend d'où se trouve la commune. Pour
        # une commune du décile, on cherche à en SORTIR ; pour une autre, ce
        # qui l'y ferait ENTRER. Ne pas le dire ferait lire les signes à
        # l'envers — et un « +20 % de maquis » passerait pour une
        # recommandation de plantation.
        dedans = ligne.score >= seuil
        st.caption(
            f"Cette commune est au {N.dec(100 * ligne.rang, 1)}ᵉ percentile du "
            f"jour ; le décile le plus à risque commence à un score de "
            f"{N.dec(seuil, 4)}. "
            + ("Elle est dedans : DiCE cherche ce qu'il faudrait changer pour "
               "l'en faire sortir."
               if dedans else
               "Elle est en dehors : DiCE cherche à l'inverse ce qui l'y "
               "ferait entrer, et les signes se lisent dans ce sens-là."))

        cg, cd = st.columns([1, 3])
        n_cf = cg.slider("Combien de scénarios", 1, 5, 3)
        modifiables = cd.multiselect(
            "Variables que l'on s'autorise à changer", list(FEATURES),
            format_func=joli,
            default=["part_maquis", "part_foret", "part_combustible",
                     "part_agricole"],
            help="Le débroussaillement se décide ; la distance à la côte non. "
                 "Un contrefactuel n'a de sens que sur des leviers réels.")

        # ── la sonde, AVANT de lancer quoi que ce soit ──────────────────
        # DiCE met plusieurs secondes puis renvoie parfois rien. On sait
        # d'avance si la recherche a une chance : il suffit de regarder si
        # l'extrême atteignable franchit le seuil.
        extreme = float(atteignable(pd.DataFrame([ligne]), modifiables,
                                    baisser=dedans)[0]) if modifiables else \
            float(ligne.score)
        franchit = (extreme < seuil) if dedans else (extreme >= seuil)

        sens = "descendrait" if dedans else "monterait"
        cote = "au-dessus du" if dedans else "en dessous du"
        if franchit:
            st.success(f"Une solution existe : en poussant les leviers "
                       f"autorisés, le score {sens} à {N.dec(extreme, 4)}, de "
                       f"l'autre côté du seuil de {N.dec(seuil, 4)}. DiCE va "
                       f"en chercher des variantes proches de l'état actuel.")
        else:
            st.warning(f"Les leviers autorisés semblent insuffisants : en les "
                       f"poussant un par un, le score {sens} au mieux à "
                       f"{N.dec(extreme, 4)}, soit {cote} seuil de "
                       f"{N.dec(seuil, 4)}.\n\n"
                       f"Ce n'est pas une certitude. Cette sonde teste les "
                       f"leviers isolément quand DiCE explore leurs "
                       f"combinaisons, et il lui arrive de trouver ce qu'elle "
                       f"manque. La recherche reste donc possible.")
            if st.button("Trouver une commune où la recherche aboutit"):
                cand = R[["code_insee", "nom", "dep_nom", "score"]].copy()
                lim = atteignable(R, modifiables, baisser=True)
                cand["ok"] = lim < seuil
                # On préfère une commune DANS le décile : « comment en
                # sortir » est la question la plus parlante.
                bons = cand[cand.ok & (cand.score >= seuil)]
                if not len(bons):
                    bons = cand[cand.ok]
                if len(bons):
                    # ⚠️ PRENDRE LA MÉDIANE, PAS LE SCORE LE PLUS HAUT.
                    # La sonde démontre qu'une solution existe quelque part ;
                    # DiCE, lui, cherche PRÈS du point de départ. Mesuré :
                    # Fontan, score 0,69 et sortie démontrée, rend zéro
                    # scénario, quand la commune médiane des candidates en
                    # rend cinq. Une commune trop enfoncée dans le décile est
                    # un mauvais candidat même quand une issue existe.
                    bons = bons.sort_values("score")
                    st.session_state["q_expl"] = \
                        bons.iloc[len(bons) // 2].nom
                    st.rerun()
                else:
                    st.info("Aucune commune ne convient pour cette date et ces "
                            "leviers. Essayez une date d'été.")

        # La sonde conseille, elle n'interdit pas : elle sous-estime ce qui est
        # atteignable, et bloquer le bouton priverait l'utilisateur des cas
        # qu'elle manque.
        if st.button("Chercher les contrefactuels" if franchit
                     else "Chercher quand même",
                     type="primary" if franchit else "secondary"):
            res, ecretes = None, 0
            with st.spinner("recherche…"):
                try:
                    cf = moteur.generate_counterfactuals(
                        x[FEATURES], total_CFs=n_cf, desired_class="opposite",
                        features_to_vary=modifiables or "all")
                    res = cf.cf_examples_list[0].final_cfs_df
                except UserConfigValidationException:
                    res = None

            # ⚠️ DiCE RENVOIE DES PROPORTIONS SUPÉRIEURES À 100 %.
            # Mesuré sur L'Estréchure : 3 scénarios sur 5 portaient
            # `part_agricole` à 110 % du territoire communal. La bibliothèque
            # connaît pourtant le domaine [0 ; 1] — elle quantifie après
            # échantillonnage et l'arrondi déborde d'un cran. Ni
            # `permitted_range` sur Data ni sur generate_counterfactuals ne
            # l'en empêchent.
            #
            # On écrête, PUIS on revérifie avec le modèle : un point écrêté
            # n'est plus celui que DiCE a validé, et l'afficher sans contrôle
            # reviendrait à montrer un faux contrefactuel. Mesuré : les cinq
            # scénarios de L'Estréchure basculent encore après écrêtage.
            proposes = 0
            if res is not None and len(res):
                proposes = len(res)          # ce que DiCE a rendu, avant tri
                classe = int(enveloppe.predict(x[FEATURES].astype(float))[0])
                res = res.copy()
                hors = (res[PARTS].lt(0) | res[PARTS].gt(1)).any(axis=1)
                ecretes = int(hors.sum())
                res[PARTS] = res[PARTS].clip(0.0, 1.0)
                res = res[enveloppe.predict(
                    res[FEATURES].astype(float)) != classe]
            if res is None or not len(res):
                # ⚠️ DIRE POURQUOI, PAS SEULEMENT QUE ÇA A ÉCHOUÉ.
                # Le message se contentait d'annoncer l'absence de solution,
                # ce qui laisse croire à une panne. Trois quantités mesurables
                # expliquent l'échec ; on les affiche.
                # ⚠️ LES TROIS QUANTITÉS S'ÉNONCENT À L'ENVERS SELON LE SENS.
                # Une version affichait « Saint-Trivier est à 52,5 fois le
                # seuil » pour une commune située cinquante fois EN DESSOUS,
                # et présentait 93 % de couverture comme une pénurie. On
                # sépare franchement les deux formulations.
                # `part_combustible` est exclu des sommes : c'est un agrégat
                # qui recoupe maquis, forêt et landes, et l'inclure donnait
                # des « 156 % du territoire ».
                vg = [f for f in (modifiables or [])
                      if f.startswith("part_") and f != "part_combustible"]
                if dedans:
                    ecart = ligne.score / seuil
                    phrase_ecart = (
                        f"{sel.nom} est à **{N.dec(ecart, 1)} fois** le seuil "
                        f"du décile. Il faut faire descendre son score, et "
                        f"plus ce rapport est grand, moins la végétation y "
                        f"suffit.")
                    stock = sum(float(ligne[f]) for f in vg)
                    phrase_leviers = (
                        f"On ne peut retirer que ce qui est là : les variables "
                        f"autorisées représentent **{N.pct(stock, 0)}** du "
                        f"territoire communal.")
                    phrase_verrou = (
                        f"Le FWI du jour vaut **{N.dec(ligne.fwi)}** "
                        f"({N.CLASSES[int(ligne.danger_effis)].lower()}) et la "
                        f"commune est à {N.dec(ligne.distance_cote_km, 0)} km "
                        f"de la côte. Ces deux variables maintiennent le score "
                        f"haut et ne figurent pas dans les leviers.")
                else:
                    ecart = seuil / max(ligne.score, 1e-9)
                    phrase_ecart = (
                        f"{sel.nom} est **{N.dec(ecart, 1)} fois en dessous** "
                        f"du seuil. Il faudrait faire monter son score, ce qui "
                        f"est d'autant plus dur que l'écart est grand.")
                    marge = sum(1.0 - float(ligne[f]) for f in vg)
                    phrase_leviers = (
                        f"On ne peut ajouter que là où il reste de la place : "
                        f"la marge cumulée avant saturation des variables "
                        f"autorisées vaut **{N.pct(marge / max(len(vg), 1), 0)}** "
                        f"en moyenne par variable.")
                    phrase_verrou = (
                        f"Surtout, le FWI du jour vaut **{N.dec(ligne.fwi)}** "
                        f"({N.CLASSES[int(ligne.danger_effis)].lower()}). "
                        f"Aucune plantation ne fait entrer une commune dans le "
                        f"décile un jour où sa météo est clémente : c'est le "
                        f"vrai verrou ici.")
                st.error(f"""
Aucun contrefactuel trouvé, et ce n'est pas une panne. Trois quantités
l'expliquent.

**L'écart à franchir.** {phrase_ecart}

**Les leviers disponibles.** {phrase_leviers}

**Ce qu'on s'interdit de toucher.** {phrase_verrou}
""")
            else:
                cols = modifiables or FEATURES
                delta = res[cols].astype(float).reset_index(drop=True) \
                    - x[cols].to_numpy()
                st.markdown("##### Ce qu'il faudrait changer")
                aff = delta.round(4)
                aff.insert(0, "scénario", [f"#{i + 1}" for i in range(len(aff))])
                st.dataframe(
                    aff.rename(columns={f: joli(f) for f in cols}),
                    width="stretch", hide_index=True)
                st.caption(
                    "Chaque ligne est un scénario alternatif : la variation à "
                    "appliquer pour que la commune sorte du décile le plus à "
                    "risque."
                    + (f" {ecretes} des {proposes} scénarios rendus par DiCE "
                       f"portaient une part au-delà de 100 % du territoire : "
                       f"ramenés à 100 %, puis revérifiés comme basculant "
                       f"toujours." if ecretes else ""))
        st.warning("""
Un contrefactuel n'est pas une recommandation. DiCE trouve un point proche que
le modèle classe différemment, sans garantir qu'il soit réalisable : on ne
convertit pas 40 % de maquis en terres agricoles. Rien ne garantit non plus que
le lien soit causal, le modèle ayant appris des corrélations sur 2006-2019 et
non des mécanismes.

Une limite plus technique, visible dans les tableaux ci-dessus : DiCE fait
varier **chaque part indépendamment**, sans savoir qu'elles décrivent le même
territoire. Un scénario peut donc retirer 48 points de forêt et ajouter
99 points de terres agricoles sans que la somme reste cohérente. Les parts
sont désormais bornées à 100 %, mais leur cohérence mutuelle n'est pas
imposée. Ces scénarios se lisent comme des directions, pas comme des plans.
""")
    except ImportError:
        st.warning("`dice-ml` n'est pas installé : `uv pip install dice-ml`.")

st.divider()
st.caption("SHAP calculé sur le modèle C (41 features), celui qui dessine la "
           "carte. Le projet dispose aussi d'un SHAP du modèle v3, mais "
           "l'afficher ici décrirait un modèle que vous ne voyez jamais.")
