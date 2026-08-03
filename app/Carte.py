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

MOIS = N.MOIS

# ════════════════════════════════════════════════════════════════════════
#  les réglages
# ════════════════════════════════════════════════════════════════════════
TEST_MIN, TEST_MAX = N.meta()["splits"]["test"]

with st.sidebar:
    st.markdown("### Le modèle")
    mode_modele = st.radio(
        "Modèle",
        ["Temps réel et projections",
         f"Rétrospectif — C contre v3 ({TEST_MIN}-{TEST_MAX})"],
        index=0, label_visibility="collapsed",
        help="Le modèle **C** (physique pure) est le seul déployable : il ne "
             "dépend d'aucune donnée indisponible à l'avance.\n\n"
             "Le modèle **v3** est meilleur — 93,8× contre 63,7× sur le test — "
             "mais il a besoin de l'historique des feux, que la BDIFF ne "
             "publie qu'un an plus tard. Le mode rétrospectif permet de les "
             "comparer, **uniquement sur le jeu de test**.")
    retro = mode_modele.startswith("Rétrospectif")

    if retro:
        st.caption(f"Les deux modèles sont servis avec la **météo réellement "
                   f"observée** ce jour-là — celle qu'ils ont vue pendant "
                   f"l'évaluation.")

    st.markdown("### Quoi afficher")
    couche = st.radio(
        "Couche", ["Risque prédit", "Danger météo (FWI)"], index=0,
        disabled=retro,
        help="Le risque croise météo, végétation et relief — c'est la sortie "
             "du modèle. Le FWI est l'indice météo officiel EFFIS, sans "
             "modèle. Les comparer montre ce que le territoire ajoute.")

    st.markdown("### Quand")
    mode = "Une date précise" if retro else st.radio(
        "Période",
        ["Une date précise", "Une année entière", "Comparer deux dates"],
        index=0, label_visibility="collapsed")


    def choix_date(cle: str, defaut_an: int, an_min=None, an_max=None):
        """⚠️ Un SÉLECTEUR de date, pas un curseur. On ne choisit pas un jour
        en faisant glisser un curseur sur 28 000 positions."""
        c1, c2 = st.columns(2)
        a = c1.number_input("Année", an_min or N.AN_OBS_MIN,
                            an_max or N.AN_PROJ_MAX, defaut_an,
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
    if retro:
        # ⚠️ Bornes IMPOSÉES au jeu de test. Le sélecteur ne propose même pas
        # les autres années : un message d'erreur après coup serait moins
        # clair qu'une plage qui ne les contient pas.
        date = choix_date("r", TEST_MIN + 1, TEST_MIN, TEST_MAX)
    elif mode == "Une date précise":
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
def poids_de(D: pd.DataFrame) -> pd.Series:
    """La grandeur à colorier, normalisée dans [0,1].

    ⚠️ Le risque est normalisé **par rapport à la journée**, pas dans
    l'absolu : la carte répond à « où regarder aujourd'hui ». Le FWI, lui, se
    lit sur l'échelle EFFIS, dont 50 est le seuil extrême — donc pas de
    normalisation relative, sinon un jour calme paraîtrait alarmant.
    """
    if couche.startswith("Risque"):
        return D.score / D.score.max()
    return (D.fwi / 50).clip(0, 1)


def couche_communes(D: pd.DataFrame):
    """Les communes en APLAT, chacune à la couleur de son risque.

    ⚠️ On joint les contours aux scores par `code_insee`, jamais par
    position : une commune en plusieurs morceaux occupe plusieurs lignes de
    contour, et un `merge` sur l'index les décalerait silencieusement.
    """
    C = N.contours().merge(
        D.assign(poids=poids_de(D))[["code_insee", "poids", "nom"]],
        on="code_insee", how="inner")
    # ⚠️ `.tolist()`, PAS `list()`. `list()` d'un tableau NumPy donne des
    # tableaux NumPy par cellule ; pydeck sérialise la table en JSON et
    # échoue dessus sans rien dire. Résultat : deck.gl ne reçoit aucune
    # couleur et la carte s'affiche vide. `.tolist()` rend des entiers
    # Python, qui passent.
    C["couleur"] = N.couleur_effis(C.poids.to_numpy()).tolist()
    return pdk.Layer(
        "PolygonLayer", C[["polygone", "couleur", "nom"]],
        get_polygon="polygone", get_fill_color="couleur",
        stroked=False, filled=True, extruded=False,
        pickable=True, auto_highlight=True)


def carte(D: pd.DataFrame, titre: str, vue=None, hauteur=460):
    """La France, commune par commune, aux couleurs EFFIS."""
    st.markdown(f"**{titre}**")
    st.pydeck_chart(pdk.Deck(
        map_style="light",
        initial_view_state=vue or pdk.ViewState(latitude=46.6, longitude=2.4,
                                                zoom=4.5),
        layers=[couche_communes(D)],
        tooltip={"text": "{nom}"}),
        height=hauteur, width="stretch")


# ════════════════════════════════════════════════════════════════════════
#  MODE RÉTROSPECTIF — le seul endroit où le modèle v3 a le droit d'exister
# ════════════════════════════════════════════════════════════════════════
if retro:
    autorise, motif = N.v3_autorise(date.year)
    if not autorise:                       # ceinture et bretelles
        st.error(motif)
        st.stop()

    st.markdown(f"## {N.date_fr(date)} — les deux modèles, "
                f"et ce qui a réellement brûlé")
    st.caption("Météo réellement observée ce jour-là, pour les deux modèles. "
               "Comparer v3 sur météo réelle à C sur climatologie mesurerait "
               "la différence de météo, pas de modèle.")

    Rc = N.predire(date, nom="C", observee=True)
    Rv = N.predire(date, nom="v3", observee=True)

    # ── ce qui a effectivement brûlé ────────────────────────────────────
    jf = N.jours_feu()
    brules = set(jf[jf.date == date].code_insee)

    def classement(R):
        """Rang décroissant : 1 = la commune la plus à risque selon le modèle."""
        return R.assign(pos=R.score.rank(ascending=False, method="min"))

    Rc, Rv = classement(Rc), classement(Rv)

    if brules:
        pc = Rc[Rc.code_insee.isin(brules)].pos
        pv = Rv[Rv.code_insee.isin(brules)].pos
        n = len(Rc)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Communes ayant brûlé", f"{len(brules)}")
        m2.metric("Rang médian — modèle C", f"{pc.median():,.0f}ᵉ"
                  .replace(",", " "), f"sur {n:,}".replace(",", " "),
                  delta_color="off")
        m3.metric("Rang médian — modèle v3", f"{pv.median():,.0f}ᵉ"
                  .replace(",", " "),
                  f"{100 * (pv.median() / pc.median() - 1):+.0f} % vs C",
                  delta_color="inverse")
        m4.metric("Dans le top 1 % (347 communes)",
                  f"C {100 * (pc <= n * .01).mean():.0f} %  ·  "
                  f"v3 {100 * (pv <= n * .01).mean():.0f} %")
        st.caption("Un rang médian **plus bas** est meilleur : le feu était "
                   "mieux classé. Sur une seule journée, ces chiffres sont "
                   "très bruités — c'est une illustration, pas une mesure. "
                   "La mesure est page *Les modèles*, sur 6 322 feux.")
    else:
        st.info("Aucun départ de feu déclaré ce jour-là dans la BDIFF. "
                "Choisissez une date d'été pour que la comparaison ait du "
                "relief.")

    # ── les deux cartes ─────────────────────────────────────────────────
    g, d_ = st.columns(2)
    with g:
        carte(Rc, "Modèle C — physique pure, déployable")
    with d_:
        carte(Rv, "Modèle v3 — avec l'historique des feux")

    # ── où sont-ils en désaccord ? ──────────────────────────────────────
    st.markdown("##### Là où les deux modèles ne sont pas d'accord")
    J = Rc[["code_insee", "nom", "dep_nom", "pos", "fwi", "part_maquis"]].merge(
        Rv[["code_insee", "pos", "feux_commune_365j",
            "jours_depuis_dernier_feu"]],
        on="code_insee", suffixes=("_c", "_v3"))
    J["ecart"] = J.pos_c - J.pos_v3          # > 0 : v3 la classe plus haut
    J["a_brule"] = J.code_insee.isin(brules)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**v3 alerte, C non** — l'historique parle")
        h = J.nlargest(8, "ecart")[
            ["nom", "dep_nom", "pos_c", "pos_v3", "feux_commune_365j",
             "jours_depuis_dernier_feu", "a_brule"]]
        st.dataframe(h.rename(columns={
            "nom": "Commune", "dep_nom": "Département", "pos_c": "rang C",
            "pos_v3": "rang v3", "feux_commune_365j": "feux 365 j",
            "jours_depuis_dernier_feu": "j. depuis",
            "a_brule": "a brûlé"}), width="stretch", hide_index=True)
    with t2:
        st.markdown("**C alerte, v3 non** — le territoire parle")
        b = J.nsmallest(8, "ecart")[
            ["nom", "dep_nom", "pos_c", "pos_v3", "fwi", "part_maquis",
             "a_brule"]]
        st.dataframe(b.assign(part_maquis=(100 * b.part_maquis).round(1)).rename(
            columns={"nom": "Commune", "dep_nom": "Département",
                     "pos_c": "rang C", "pos_v3": "rang v3", "fwi": "FWI",
                     "part_maquis": "maquis %", "a_brule": "a brûlé"}),
            width="stretch", hide_index=True)

    st.info(f"""
**Ce que ce tableau montre.** À gauche, v3 place haut des communes que C ignore
— regardez les colonnes `feux 365 j` et `j. depuis` : v3 dit *« ça a brûlé ici
récemment »*. À droite, C place haut des communes que v3 ignore, sur la
végétation et la météo.

Les deux ont raison à leur manière. Mais **la colonne de gauche est
inaccessible en temps réel** : le {date.strftime('%d/%m/%Y')}, la BDIFF
n'aurait publié aucun feu de {date.year}. C'est tout l'argument, et c'est
pourquoi l'application tourne sur C.
""")
    st.stop()

# ════════════════════════════════════════════════════════════════════════
R = N.predire(date, scenario)
projete = date.year > N.AN_OBS_MAX

# ── mode comparaison : deux cartes côte à côte, même échelle ────────────
if date_b is not None:
    Rb = N.predire(date_b, scenario)
    st.markdown(f"## {date.strftime('%d/%m/%Y')} vs {date_b.strftime('%d/%m/%Y')}")

    g, d_ = st.columns(2)
    with g:
        carte(R, N.date_fr(date))
    with d_:
        carte(Rb, N.date_fr(date_b))

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
        f"**{N.date_fr(date)} — ce n'est pas une prévision météo.** "
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
legende = ("risque prédit par le modèle, relatif à la journée"
           if couche.startswith("Risque")
           else "FWI — 50 correspond au seuil « extrême » d'EFFIS")

couches = [couche_communes(R)]

# la commune cherchée, cerclée par-dessus l'aplat
trouvees = N.chercher(q) if q else None
vue = pdk.ViewState(latitude=46.6, longitude=2.4, zoom=4.7)
if trouvees is not None and len(trouvees):
    sel = trouvees.iloc[0]
    ligne = R[R.code_insee == sel.code_insee]
    couches.append(pdk.Layer(
        "ScatterplotLayer", ligne, get_position=["lon", "lat"],
        get_fill_color=[0, 0, 0, 0], get_radius=4000,
        stroked=True, get_line_color=[20, 20, 20], line_width_min_pixels=2.5))
    vue = pdk.ViewState(latitude=float(sel.lat), longitude=float(sel.lon),
                        zoom=9)

st.pydeck_chart(pdk.Deck(map_style="light", layers=couches,
                         initial_view_state=vue, tooltip={"text": "{nom}"}),
                height=560, width='stretch')

_manq = len(N.contours_manquants())
st.caption(
    f"Chaque commune est colorée sur **toute sa surface** — {legende}. "
    f"Échelle EFFIS : vert → jaune → orange → rouge → noir. "
    f"Survolez une commune pour la nommer."
    + (f" ⚠️ {_manq} communes sur {N.nb(len(R))} n'ont pas de contour dans le "
       f"référentiel géographique et n'apparaissent pas : leur code a changé "
       f"entre le millésime du fond de carte et le COG 2026."
       if _manq else ""))

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
