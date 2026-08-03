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
La question paraît simple. Elle a **trois réponses différentes**, et chacune
est juste — mais pas à la même question. C'est le premier piège de
l'interprétabilité, et il vaut la peine de s'y arrêter avant de regarder le
moindre graphique.
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
Combien chaque feature a réduit la perte **pendant l'entraînement**.

C'est ce que renvoie XGBoost par défaut. Biaisée vers les variables qui
découpent proprement, et **aveugle à la redondance**.
""")
c2.markdown("""
##### 2. SHAP sur un échantillon **aléatoire**
Combien chaque feature déplace le score **sur le territoire tel qu'il est** —
99,97 % de communes-jours sans feu.

C'est la bonne lecture pour « en général ».
""")
c3.markdown("""
##### 3. SHAP sur le **sommet** du classement
Combien elle déplace le score **là où le modèle s'engage**.

C'est la bonne lecture pour « quand ça brûle ». ⚠️ Sélectionner sur le score
introduit un **biais de collision** — on ne lit ce panneau que pour ça.
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
st.warning(f"""
**Deux désaccords qui valent d'être expliqués.**

**`classe de danger EFFIS`** est **{rang(de.r_gain)} par gain** ({de.gain:.1f} %) mais
**{rang(de.r_alea)} par SHAP**. Ce n'est pas une erreur : c'est une discrétisation du
FWI en six classes. XGBoost adore ces seuils nets pour découper, d'où un gain
élevé — mais l'information est déjà dans le `FWI` continu, et SHAP en attribue
donc le crédit au FWI. **L'importance par gain ne sait pas gérer la
redondance.**

**`part de maquis`** est **{rang(ma.r_gain)} par gain**, **{rang(ma.r_alea)}** sur
l'échantillon aléatoire, et **{rang(ma.r_sommet)} au sommet**. Là encore tout est
cohérent : le maquis ne change rien sur une commune-jour moyenne — il n'y en a
pas — mais il devient déterminant **là où le modèle voit du risque**.

*Morale* : ne jamais citer « la feature n°1 » sans dire de quelle mesure il
s'agit et sur quelle population.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  2. le nuage SHAP
# ════════════════════════════════════════════════════════════════════════
st.markdown("## Dans quel sens chaque variable pousse-t-elle ?")

pop = st.radio("Population", ["Le sommet du classement", "Un échantillon aléatoire"],
               horizontal=True, label_visibility="collapsed")
V, X = (V_S, X_S) if pop.startswith("Le sommet") else (V_A, X_A)

st.caption("Chaque point est une commune-jour. À droite du zéro, la variable "
           "**augmente** le risque ; à gauche, elle le diminue. La couleur "
           "donne la valeur de la variable — rouge = élevée, bleu = basse.")

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
**La lecture centrale du projet est visible ici.** Les variables de
**territoire** — maquis, distance à la côte, superficie, relief — occupent le
haut du classement à égalité avec les indices **météo**. Aucune des deux
familles ne suffit seule :

> *la météo dit **quand**, le territoire dit **où**.*

C'est aussi ce que montrait la baseline : le danger EFFIS seul vaut ×5 le
hasard, l'historique spatial seul ×19, et leur croisement ×42.
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
    st.info("Aucune commune ne correspond — essayez « Bormes », « 83230 »…")
    st.stop()

sel = trouvees.iloc[0]
date = pd.Timestamp(year=2024, month=int(mois), day=min(int(jour), 28))
st.caption(f"**{sel.nom}** ({sel.dep_nom}) — {N.date_fr(date)}. "
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

    st.success("""
**TreeSHAP est EXACT.** Sur un modèle à base d'arbres, il ne s'agit pas d'une
approximation par échantillonnage : l'algorithme parcourt la structure des
arbres et calcule la valeur de Shapley exacte, en temps polynomial. La somme
des barres, plus la valeur de base, redonne **exactement** le score du modèle.

C'est la raison pour laquelle SHAP est ici la référence, et LIME la
comparaison.
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

        st.info("""
**LIME répond à une autre question, et il faut le dire.** Il perturbe le point,
interroge le modèle des milliers de fois, et ajuste une **régression linéaire
pondérée** sur ce voisinage. Ce qu'il rend n'est pas la contribution exacte : ce
sont les coefficients d'un **modèle de substitution**.

Trois conséquences, toutes visibles ci-dessus :

- les variables apparaissent sous forme de **règles** (`fwi > 12,4`), parce que
  LIME discrétise ;
- le résultat est **stochastique** — relancez, il bougera un peu ;
- il dépend du **fond** qu'on lui donne, ici 30 000 lignes du train.

Sur un modèle d'arbres, **LIME ne peut pas être meilleur que SHAP** : il
approxime ce que TreeSHAP calcule exactement. On le montre parce qu'il est
partout dans la littérature, et parce que savoir *pourquoi on ne s'en sert pas*
vaut mieux que de l'ignorer. Sur un modèle qu'on ne peut pas ouvrir — une API,
un réseau profond — il redeviendrait le bon outil.
""")
    except ImportError:
        st.warning("`lime` n'est pas installé : `uv pip install lime`.")

# ── DiCE ─────────────────────────────────────────────────────────────────
with o3:
    st.markdown("""
SHAP et LIME répondent à *« pourquoi ce score ? »*. **DiCE répond à
*« qu'aurait-il fallu changer ? »*** — et c'est la seule question dont la
réponse soit actionnable.
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
        mod = dice_ml.Model(model=SeuilPercentile(N.modele("C"), seuil),
                            backend="sklearn")
        moteur = Dice(donnees, mod, method="random")

        # ⚠️ Le sens de la question dépend d'où se trouve la commune. Pour
        # une commune du décile, on cherche à en SORTIR ; pour une autre, ce
        # qui l'y ferait ENTRER. Ne pas le dire ferait lire les signes à
        # l'envers — et un « +20 % de maquis » passerait pour une
        # recommandation de plantation.
        dedans = ligne.score >= seuil
        st.caption(
            f"Cette commune est au **{100 * ligne.rang:.1f}ᵉ percentile** du "
            f"jour ; le décile le plus à risque commence à un score de "
            f"{seuil:.4f}. "
            + (f"Elle **est dedans** : DiCE cherche donc ce qu'il faudrait "
               f"changer pour l'en **faire sortir**."
               if dedans else
               f"Elle **est en dehors** : DiCE cherche donc, à l'inverse, ce "
               f"qui l'y **ferait entrer**. Les signes se lisent dans ce "
               f"sens-là."))

        cg, cd = st.columns([1, 3])
        n_cf = cg.slider("Combien de scénarios", 1, 5, 3)
        modifiables = cd.multiselect(
            "Variables que l'on s'autorise à changer", list(FEATURES),
            format_func=joli,
            default=["part_maquis", "part_foret", "part_combustible",
                     "part_agricole"],
            help="Le débroussaillement se décide ; la distance à la côte non. "
                 "Un contrefactuel n'a de sens que sur des leviers réels.")

        if st.button("Chercher les contrefactuels", type="primary"):
            res = None
            with st.spinner("recherche…"):
                try:
                    cf = moteur.generate_counterfactuals(
                        x[FEATURES], total_CFs=n_cf, desired_class="opposite",
                        features_to_vary=modifiables or "all")
                    res = cf.cf_examples_list[0].final_cfs_df
                except UserConfigValidationException:
                    res = None
            if res is None or not len(res):
                st.error(f"""
**Aucun contrefactuel trouvé** — et c'est un résultat, pas un échec.

Sur ces leviers-là, rien ne fait
{'sortir' if dedans else 'entrer'} **{sel.nom}**
{'du' if dedans else 'dans le'} décile à risque. Son exposition ne tient pas à
ce qu'on peut modifier : elle tient à sa position, à son relief, à sa
superficie. Le risque est **structurel**.

Élargissez la liste des variables pour voir ce qu'il faudrait vraiment
changer — et mesurez à quel point ce serait irréaliste. C'est précisément ce
qu'un contrefactuel apprend.
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
                st.caption("Chaque ligne est un scénario alternatif : la "
                           "variation à appliquer pour que la commune sorte "
                           "du décile le plus à risque.")
        st.warning("""
⚠️ **Un contrefactuel n'est pas une recommandation.** DiCE trouve un point
proche que le modèle classe différemment. Rien ne garantit que ce point soit
**réalisable** — on ne convertit pas 40 % de maquis en terres agricoles — ni
que le lien soit **causal** : le modèle a appris des corrélations sur
2006-2019, pas des mécanismes.

C'est pour cette raison que la liste des variables modifiables est un choix
explicite, laissé à l'utilisateur, et non un réglage caché.
""")
    except ImportError:
        st.warning("`dice-ml` n'est pas installé : `uv pip install dice-ml`.")

st.divider()
st.caption("SHAP calculé sur le modèle C (41 features) — celui qui dessine la "
           "carte. Le projet dispose aussi d'un SHAP du modèle v3 : "
           "l'afficher ici décrirait un modèle que vous ne voyez jamais.")
