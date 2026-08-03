"""Ce que les données disent.

Quatre onglets : d'où elles viennent, où et quand ça brûle, si le danger
augmente, et ce que la dimension temporelle permet de prévoir.

Aucun chiffre de tendance n'est écrit en dur ici. Tout vient de
`tendances.csv`, produit par `tvfed.export_app`. Une version antérieure
annonçait « +45 % de FWI » dans une chaîne de caractères que rien ne
recalculait, et la valeur était fausse.
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

COM, TEN = N.communes(), N.tendances()
MOIS_DOY = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]


def ten(nom: str) -> pd.Series:
    return TEN[TEN.serie == nom].iloc[0]


FWI_AN = ten("FWI moyen annuel")
FWI_ETE = ten("FWI moyen juin-septembre")
JOURS = ten("jours de danger élevé (FWI > 21,3)")
FEUX = ten("communes-jours en feu")


@st.cache_data(show_spinner=False)
def saison() -> pd.DataFrame:
    """Feux par département et par jour de l'année, 2006-2025.

    Calculé à la volée : `jours_feu.parquet` fait 49 130 lignes et
    `communes.parquet` porte déjà le département. Aucun artefact à produire.
    """
    d = N.jours_feu().merge(COM[["code_insee", "dep_nom"]], on="code_insee")
    d["doy"] = d.date.dt.dayofyear
    return d.groupby(["dep_nom", "doy"], as_index=False).n.sum()


def liste(s: pd.Series) -> str:
    """« Cantal (52 %), Dordogne (44 %) » — sans article, donc utilisable
    après un deux-points quel que soit le genre du département."""
    return ", ".join(f"{d} ({N.pct(p)})" for d, p in s.items())


def habiller(ax, titre=None, y=None):
    if titre:
        ax.set_title(titre, fontsize=10.5, weight="bold", loc="left")
    if y:
        ax.set_ylabel(y)
    ax.grid(color=N.GRID, lw=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


o1, o2, o3, o4 = st.tabs(["Le jeu de données", "Où et quand ça brûle",
                          "Le danger augmente-t-il", "Ce que le temps prédit"])

# ════════════════════════════════════════════════════════════════════════
with o1:
    a, b, c, d = st.columns(4)
    a.metric("Communes", N.nb(len(COM)))
    b.metric("Jours couverts", N.nb(19358), "1973 → 2025")
    c.metric("Communes-jours en feu", N.nb(int(COM.feux.sum())), "2006-2025")
    d.metric("Surface brûlée", f"{N.nb(COM.ha.sum() / 1000)} k ha")

    st.markdown("""
Quatre sources publiques, croisées sur le code INSEE et sur une grille météo
de 0,25°.

| Source | Ce qu'elle apporte | Volume |
|---|---|---|
| CEMS · Copernicus | 8 indices de danger, par jour et par maille | 21,9 M lignes, 1973-2025 |
| BDIFF · IGN | les feux déclarés, commune par commune | 52 809 sur le périmètre |
| CORINE Land Cover | l'occupation du sol, 44 postes | 1,08 M lignes |
| INSEE | référentiel des communes et leurs fusions | 34 734 communes |

La table centrale croise les communes et les jours : une ligne par commune et
par jour, qu'il y ait eu un feu ou non. **253 731 870 lignes** sur 2006-2025.

La cible est binaire : y a-t-il eu un départ de feu ce jour-là dans cette
commune ? **49 130 fois oui, soit 0,0194 %.** Cette rareté commande le reste
de la méthode. Un modèle qui répond toujours « non » a 99,98 % de justesse.
""")

    with st.expander("Pourquoi garder les jours sans feu ?"):
        st.markdown("""
Parce qu'une série creuse fausse les fenêtres glissantes. « Nombre de feux
dans les 30 jours précédents » se calcule en remontant 30 lignes. Si les jours
sans feu sont absents, on remonte en réalité plusieurs années, et le calcul
donne un résultat plausible mais faux.

C'est la raison d'avoir matérialisé 253 millions de lignes plutôt que 52 809.
""")

    with st.expander("Les fusions de communes"):
        st.markdown("""
965 feux portent un code INSEE qui n'existe plus dans le référentiel 2026.

On a d'abord essayé de les rattacher par le nom. Mauvaise idée : « Chirac »
en Lozère renvoyait vers une commune de Charente, et « Fraissinet-de-Lozère »
vers « Fraissinet-de-Fourques », qui est une autre commune. Trois propositions
fausses sur huit testées.

On est donc passé par le fichier officiel des mouvements de communes de
l'INSEE, en suivant les chaînes de fusion de façon transitive. Il reste
30 cas sans solution, essentiellement des scissions où un feu ancien ne peut
être attribué à aucune des communes filles. Ils sont écartés et comptés.
""")

# ════════════════════════════════════════════════════════════════════════
with o2:
    g1, g2 = st.columns(2)

    with g1:
        top = (COM.groupby("dep_nom").feux.sum().nlargest(12)
               .sort_values())
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.barh(range(len(top)), top, color=N.ROUGE, edgecolor=N.FOND, lw=1.2)
        for i, v in enumerate(top):
            ax.text(v * 1.01, i, N.nb(v), va="center", fontsize=8.5,
                    weight="bold")
        ax.set_yticks(range(len(top)), top.index, fontsize=8.5)
        ax.set_xlim(0, top.max() * 1.15)
        ax.set_xlabel("communes-jours ayant brûlé, 2006-2025")
        habiller(ax, "Les 12 départements les plus touchés")
        plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

    with g2:
        C = COM[COM.part_maquis.notna()].copy()
        C["tranche"] = pd.cut(100 * C.part_maquis, [-.01, .5, 5, 15, 30, 101],
                              labels=["< 0,5 %", "0,5-5 %", "5-15 %",
                                      "15-30 %", "> 30 %"])
        t = C.groupby("tranche", observed=True).agg(
            feux=("feux", "sum"), n=("code_insee", "size"))
        t["taux"] = t.feux / (t.n * 7305)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.bar(range(len(t)), 100 * t.taux, color="#8b5a2b",
               edgecolor=N.FOND, lw=1.4)
        for i, (v, n) in enumerate(zip(100 * t.taux, t.n)):
            ax.text(i, v * 1.03, f"{v:.3f} %", ha="center", fontsize=8.5,
                    weight="bold")
            ax.text(i, v * .5, N.nb(n) + "\ncommunes", ha="center",
                    fontsize=7.5, color="white")
        ax.set_xticks(range(len(t)), t.index, fontsize=8.5)
        ax.set_xlabel("part de maquis dans la commune")
        habiller(ax, "Le maquis, seul, multiplie le risque",
                 "probabilité de feu un jour donné (%)")
        plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

    ratio = (100 * t.taux).iloc[-1] / (100 * t.taux).iloc[0]
    st.markdown(f"""
Une commune couverte à plus de 30 % de maquis brûle **{ratio:.0f} fois** plus
souvent qu'une commune qui n'en a pas. C'est le signal le plus fort du jeu de
données, et le modèle s'en sert : le maquis arrive en tête de l'importance par
gain, à 26,2 %.

Cette mesure a ses limites. Sur un échantillon aléatoire de communes-jours,
SHAP place le maquis au 10ᵉ rang, parce qu'il n'y en a pas là où rien ne se
passe. Il remonte au 2ᵉ dès qu'on regarde les communes que le modèle juge à
risque. Les trois classements sont détaillés page *Pourquoi un feu part*.
""")

    st.divider()
    st.markdown("### Quand ça brûle")

    S = saison()
    ordre = (S.groupby("dep_nom").n.sum().sort_values(ascending=False).index)
    dep = st.selectbox("Département", ordre, index=0,
                       help="Classés par nombre de feux décroissant.")

    nat = S.groupby("doy", as_index=False).n.sum()
    loc = S[S.dep_nom == dep]

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 3.6), sharex=True)
    for a_, (d_, titre, coul) in zip(ax, [
            (nat, "France entière", N.ORANGE),
            (loc, dep, N.ROUGE)]):
        s = d_.set_index("doy").n.reindex(range(1, 367), fill_value=0)
        liss = s.rolling(7, center=True, min_periods=1).mean()
        a_.fill_between(liss.index, 0, liss.to_numpy(), color=coul, lw=0,
                        alpha=.85)
        a_.set_xticks(MOIS_DOY, list("JFMAMJJASOND"))
        a_.set_xlim(1, 366)
        habiller(a_, titre, "feux (lissé sur 7 jours)")
    plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

    # ⚠️ Tout ce qui est affirmé ici est CALCULÉ. Une première rédaction
    # attribuait le pic d'hiver au Sud-Ouest et aux Pyrénées : faux. En valeur
    # absolue il est mené par les départements méditerranéens, qui brûlent
    # simplement beaucoup. Ce sont les PARTS locales qui distinguent les deux
    # régimes, pas les totaux.
    HIV, ETE = slice(32, 105), slice(182, 258)
    sn = nat.set_index("doy").n.reindex(range(1, 367), fill_value=0)
    hiver, ete = int(sn.loc[HIV].sum()), int(sn.loc[ETE].sum())

    par_dep = S.pivot_table(index="dep_nom", columns="doy", values="n",
                            aggfunc="sum", fill_value=0)
    tot = par_dep.sum(axis=1)
    ph = par_dep.loc[:, HIV.start:HIV.stop].sum(axis=1) / tot
    pe = par_dep.loc[:, ETE.start:ETE.stop].sum(axis=1) / tot
    assez = tot >= 300                      # sous 300 feux, la part est bruitée
    hivernaux = ph[assez].nlargest(3)
    estivaux = pe[assez].nlargest(2)

    sl = loc.set_index("doy").n.reindex(range(1, 367), fill_value=0)
    part_h = sl.loc[HIV].sum() / max(sl.sum(), 1)
    part_e = sl.loc[ETE].sum() / max(sl.sum(), 1)

    st.markdown(f"""
La France a **deux saisons de feu**. Celle de fin d'hiver, de février à
mi-avril, totalise {N.nb(hiver)} départs ; celle d'été, de juillet à
mi-septembre, {N.nb(ete)}.

Les deux pics sont menés, en valeur absolue, par les mêmes départements
méditerranéens : ils brûlent beaucoup, toute l'année. La différence se voit
dans les proportions locales.

Là où l'hiver l'emporte : {liste(hivernaux)}. Ce sont les écobuages, ces feux
de pâture allumés volontairement en fin d'hiver pour rouvrir les parcours.
Là où l'été l'emporte : {liste(estivaux)}.

**{dep}** : {N.pct(part_h)} des départs en fin d'hiver, {N.pct(part_e)} en été.
""")

# ════════════════════════════════════════════════════════════════════════
with o3:
    st.caption("Moyenne nationale des 1 131 mailles, par décennie. Ce sont des "
               "mesures issues des réanalyses Copernicus, pas des projections.")

    D = N.decennies().dropna(subset=["periode"])
    nat_d = D.groupby("periode").agg(
        fwi=("fwi_moyen", "mean"), jours=("jours_danger", "mean"),
        extremes=("jours_tres_eleve", "mean")).sort_index()

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.4))
    x = np.arange(len(nat_d))
    for a_, col, titre, coul in (
            (ax[0], "fwi", "FWI moyen annuel", N.ORANGE),
            (ax[1], "jours", "Jours de danger élevé (FWI > 21,3)", N.ROUGE),
            (ax[2], "extremes", "Jours très élevés (FWI > 38)", "#8b1a1a")):
        a_.bar(x, nat_d[col], color=coul, edgecolor=N.FOND, linewidth=1.4)
        for i, v in enumerate(nat_d[col]):
            a_.text(i, v * 1.02, f"{v:.1f}", ha="center", fontsize=8.5,
                    weight="bold")
        a_.set_xticks(x, nat_d.index, fontsize=7.5, rotation=25, ha="right")
        habiller(a_, titre)
    plt.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

    st.markdown("##### Les pentes, par régression linéaire sur toute la période")
    T = TEN.copy()
    T["Conclusion"] = np.where(T.significatif, "significatif",
                               "NON significatif")
    st.dataframe(T.assign(**{
        "Série": T.serie,
        "Période": T.an_min.astype(str) + "-" + T.an_max.astype(str),
        "Pente / an": T.pente.round(4),
        "Variation": T.variation_pct.round(0).astype(int).astype(str) + " %",
        "p": T.p.map(lambda v: f"{v:.1e}"),
    })[["Série", "Période", "Pente / an", "Variation", "p", "Conclusion"]],
        width="stretch", hide_index=True)

    st.markdown(f"""
Le FWI moyen gagne **{FWI_AN.variation_pct:+.0f} %** sur {FWI_AN.n_ans} ans, et
les jours de danger élevé {JOURS.variation_pct:+.0f} %. La moyenne de
juin à septembre monte plus vite que la moyenne annuelle
({FWI_ETE.variation_pct:+.0f} %) : c'est l'été qui se réchauffe et s'assèche le
plus. Citer l'un pour l'autre change le message, il faut donc préciser lequel.

Le nombre de feux, lui, ne bouge pas : {FEUX.variation_pct:+.0f} % sur
{FEUX.n_ans} ans, avec p = {FEUX.p:.2f}.
""")

    with st.expander("Comment deux constats opposés peuvent-ils être vrais ?"):
        st.markdown(f"""
Trois explications, compatibles entre elles.

Les feux ne sont observés que depuis 2006. {FEUX.n_ans} points annuels très
bruités ne suffisent pas à détecter une tendance modérée : l'absence de preuve
n'est pas une preuve d'absence.

Le nombre de départs dépend aussi des moyens de lutte, des interdictions
d'accès aux massifs et du débroussaillement. Un aléa qui monte à sinistralité
constante est ce qu'on attend d'une prévention efficace.

Enfin, ce que le modèle projette est l'aléa, pas le bilan. Les projections à
2100 transportent l'évolution du FWI sous les scénarios RCP, jamais une
extrapolation du décompte de feux.

La formulation juste est donc : les conditions favorables aux feux augmentent
très significativement, et le nombre de départs reste stable.
""")

# ════════════════════════════════════════════════════════════════════════
with o4:
    st.caption("Série nationale : nombre de communes-jours en feu par jour, "
               "7 305 points de 2006 à 2025.")

    st.markdown("""
Le modèle principal répond à « où ». Il ne dit rien de « combien de feux
demain en France », qui est pourtant la question qui dimensionne les moyens
nationaux. On a traité cet axe séparément, avec les outils classiques des
séries temporelles.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Stationnarité (Dickey-Fuller augmenté)")
        try:
            ADF = pd.read_csv(N.DON / "series_adf.csv")
            st.dataframe(ADF.assign(**{
                "Série": ADF.serie, "stat ADF": ADF.adf.round(2),
                "p": ADF.p.map(lambda v: f"{v:.1e}"),
                "Conclusion": np.where(ADF.stationnaire, "stationnaire",
                                       "non stationnaire"),
            })[["Série", "stat ADF", "p", "Conclusion"]],
                width="stretch", hide_index=True)
            st.caption("H₀ est « la série a une racine unitaire », donc non "
                       "stationnaire. Rejeter H₀ signifie stationnaire, ce qui "
                       "est l'inverse de l'intuition.")
        except FileNotFoundError:
            st.caption("`series_adf.csv` absent : lancer `python -m tvfed.series`.")

    with c2:
        st.markdown("##### Prévoir « combien » (SARIMAX)")
        try:
            SAR = pd.read_csv(N.DON / "series_sarimax.csv")
            st.dataframe(SAR.assign(**{
                "Modèle": SAR.modele, "MAE": SAR.mae.round(2),
                "r": SAR.correlation.round(3),
            })[["Modèle", "MAE", "r"]], width="stretch", hide_index=True)
            st.caption("Ajusté sur 2006-2019, évalué sur 2020-2022. Le test "
                       "2023-2025 n'est pas touché, même pour une cible "
                       "différente.")
        except FileNotFoundError:
            st.caption("`series_sarimax.csv` absent : lancer "
                       "`python -m tvfed.series`.")

    st.warning("""
La ligne qui compte est la dernière : un ARIMA sans variable exogène donne
**r = −0,118**. La corrélation est négative.

À 1 096 pas d'horizon, un modèle autorégressif dont la mémoire utile est de
deux à trois jours a oublié son point de départ. Il converge vers la moyenne,
et la ligne plate qu'il produit se trouve légèrement anti-corrélée à l'observé.
Ce n'est pas un défaut de réglage.

Ajouter le FWI fait tomber l'erreur de 37 %. La prévisibilité du feu est dans
la météo, pas dans son propre passé. C'est aussi pourquoi un LSTM n'y change
rien, ce que détaille la page *Les modèles*.
""")

    with st.expander("Pourquoi pas de composante saisonnière SARIMA ?"):
        st.markdown("""
Une saisonnalité annuelle sur données journalières donnerait *s* = 365. Il
faudrait estimer des coefficients à 365 pas de distance sur 5 113 points
d'ajustement : le modèle serait instable et très lent.

La pratique établie sur données journalières est de porter la saisonnalité par
des termes de Fourier en variable exogène. Quelques harmoniques suffisent pour
un cycle annuel lisse. C'est le « X » de SARIMAX qui fait ce travail.
""")

    st.caption("Le détail, avec les corrélogrammes ACF et PACF et le choix des "
               "ordres, est dans `notebook/series-lstm.ipynb` et le cours "
               "`docs/series-temporelles.md`.")
