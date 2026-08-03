"""Page d'accueil : la carte du risque, commune par commune.

    streamlit run app/Carte.py

Chaque commune est peinte sur toute sa surface, à partir des contours réels.
La base ne contient que des centroïdes ; les polygones viennent de
`app/donnees/contours.parquet`, produit par `tvfed.contours`.

Organisation de la page : les réglages qu'on change souvent sont au-dessus de
la carte, ceux qu'on règle une fois restent dans la barre latérale.
"""
from __future__ import annotations

import sys
from datetime import date as _date
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
TEST_MIN, TEST_MAX = N.meta()["splits"]["test"]
VUE_FRANCE = dict(latitude=46.6, longitude=2.4, zoom=4.7)

# ════════════════════════════════════════════════════════════════════════
#  barre latérale : ce qu'on règle une fois
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Le modèle")
    mode_modele = st.radio(
        "Modèle",
        ["Temps réel et projections",
         f"Rétrospectif · C contre v3 ({TEST_MIN}-{TEST_MAX})"],
        index=0, label_visibility="collapsed",
        help="Le modèle C, physique pure, est le seul déployable : il ne "
             "dépend d'aucune donnée indisponible à l'avance.\n\n"
             "Le modèle v3 est meilleur, 93,8× contre 63,7× sur le test, mais "
             "il a besoin de l'historique des feux que la BDIFF ne publie "
             "qu'un an plus tard. Le mode rétrospectif les compare, "
             "uniquement sur le jeu de test.")
    retro = mode_modele.startswith("Rétrospectif")

    st.markdown("### La période")
    mode = "Une date précise" if retro else st.radio(
        "Période", ["Une date précise", "Comparer deux dates"],
        index=0, label_visibility="collapsed")

    scenario = st.radio(
        "Scénario climatique", N.SCENARIOS, index=2,
        format_func=lambda s: N.ETIQ_SC[s].split(" — ")[0],
        help="Un RCP est une trajectoire d'émissions du GIEC. Le chiffre est "
             "le forçage radiatif en 2100, en W/m². Sans effet avant 2026.\n\n"
             + "\n\n".join(f"**{v.split(' — ')[0]}** : {v.split(' — ')[1]}"
                           for v in N.ETIQ_SC.values()))

    st.markdown("### L'affichage")
    palette = st.radio(
        "Palette", list(N.PALETTES), index=0,
        format_func=lambda p: N.PALETTES[p]["titre"],
        help="La palette EFFIS est la norme européenne, mais ses deux "
             "premières classes sont inversées en luminance : le jaune "
             "« faible » est plus clair que le vert « très faible ». En "
             "vision des couleurs déficiente, elles se lisent donc à "
             "l'envers.")
    st.caption(N.PALETTES[palette]["note"])


# ════════════════════════════════════════════════════════════════════════
#  les briques d'affichage
# ════════════════════════════════════════════════════════════════════════
def poids_de(D: pd.DataFrame, couche: str) -> pd.Series:
    """La grandeur à colorier, ramenée dans [0,1].

    Le risque est normalisé par rapport à la journée : la carte répond à « où
    regarder aujourd'hui ». Le FWI se lit sur l'échelle EFFIS, dont 50 est le
    seuil extrême, donc pas de normalisation relative — sinon un jour calme
    paraîtrait alarmant.

    ⚠️ POURQUOI UNE VUE « PAR KM² » EXISTE.
    La cible du modèle est « cette commune a-t-elle AU MOINS UN feu ce
    jour-là ». Une commune de 172 km² a mécaniquement plus de chances d'en
    contenir un qu'une de 6 km², et le modèle l'a appris : la corrélation
    entre le score et la superficie vaut 0,55. Fontainebleau ressort donc très
    sombre au milieu de communes vertes qui partagent sa météo.

    Le rang plutôt que la valeur brute : la distribution du score par km² est
    trop écrasée pour une normalisation linéaire.
    """
    if couche.startswith("Risque par"):
        return (D.score / D.superficie_km2.clip(lower=.5)).rank(pct=True)
    if couche.startswith("Risque"):
        return D.score / D.score.max()
    return (D.fwi / 50).clip(0, 1)


def couche_communes(D: pd.DataFrame, couche: str):
    """Les communes en aplat, chacune à la couleur de sa valeur.

    ⚠️ On joint les contours aux scores par `code_insee`, jamais par
    position : une commune en plusieurs morceaux occupe plusieurs lignes de
    contour, et un `merge` sur l'index les décalerait silencieusement.
    """
    src = D.assign(poids=poids_de(D, couche))
    src["classe"] = [N.CLASSES[int(c)] for c in src.danger_effis]
    src["rang_txt"] = (100 * src.score.rank(pct=True)).round(1).astype(str) + "ᵉ"
    src["fwi_txt"] = src.fwi.round(1).astype(str)
    src["val"] = (src.poids.rank(pct=True) * 100).round(0).astype(int).astype(str)
    C = N.contours().merge(
        src[["code_insee", "poids", "nom", "dep_nom", "classe", "rang_txt",
             "fwi_txt", "val"]], on="code_insee", how="inner")
    # ⚠️ `.tolist()`, PAS `list()`. `list()` d'un tableau NumPy laisse des
    # tableaux NumPy dans les cellules ; pydeck écrit alors « [136 240 125] »,
    # sans virgules, et deck.gl ne colorie rien sans lever d'erreur.
    C["couleur"] = N.couleur(C.poids.to_numpy(), palette).tolist()
    return pdk.Layer(
        "PolygonLayer",
        C[["polygone", "couleur", "nom", "dep_nom", "classe", "rang_txt",
           "fwi_txt", "val", "code_insee"]],
        get_polygon="polygone", get_fill_color="couleur",
        stroked=False, filled=True, extruded=False,
        pickable=True, auto_highlight=True)


INFOBULLE = {
    "html": "<b style='font-size:13px'>{nom}</b><br/>"
            "<span style='opacity:.75'>{dep_nom}</span>"
            "<hr style='margin:5px 0;border:0;border-top:1px solid #555'/>"
            "FWI <b>{fwi_txt}</b> · {classe}<br/>"
            "rang national <b>{rang_txt}</b><br/>"
            "<span style='opacity:.6;font-size:11px'>cliquez pour la fiche"
            "</span>",
    "style": {"backgroundColor": "#141412", "color": "#f4f3ef",
              "fontSize": "12px", "padding": "9px 11px",
              "borderRadius": "7px", "maxWidth": "230px"},
}


def legende(couche: str) -> None:
    """Le bandeau de classes, au-dessus de la carte.

    pydeck ne sait pas dessiner de légende, et l'échelle n'était jusqu'ici
    décrite qu'en mots sous la carte. On la construit en HTML, à partir de la
    MÊME palette que la carte : les deux ne peuvent donc pas diverger.
    """
    pal = N.PALETTES[palette]["couleurs"]
    if couche.startswith("Danger"):
        titre = "FWI, indice de danger météo"
        etiq = ["0", "5,2", "11,2", "21,3", "38", "50 +"]
        sous = [c.replace("très ", "très ") for c in N.CLASSES]
    else:
        titre = ("risque par km², rang national" if couche.startswith("Risque par")
                 else "risque prédit, relatif à la journée")
        etiq = ["", "", "", "", "", ""]
        sous = ["le plus faible", "", "", "", "", "le plus élevé"]

    blocs = "".join(
        f"<div style='flex:1;text-align:center'>"
        f"<div style='height:15px;background:{c}'></div>"
        f"<div style='font-size:10px;color:#6b6963;margin-top:3px'>{e}</div>"
        f"<div style='font-size:9.5px;color:#9b9992'>{s}</div></div>"
        for c, e, s in zip(pal, etiq, sous))
    st.markdown(
        f"<div style='margin:2px 0 8px'>"
        f"<div style='font-size:11px;color:#6b6963;margin-bottom:3px'>"
        f"{titre}</div><div style='display:flex;gap:2px'>{blocs}</div></div>",
        unsafe_allow_html=True)


def indicateurs(D: pd.DataFrame, couche: str) -> None:
    """Quatre chiffres, cohérents avec ce que la carte montre.

    Ils parlaient toujours de FWI, même quand la carte affichait le risque.
    """
    a, b, c, d = st.columns(4)
    if couche.startswith("Danger"):
        a.metric("FWI moyen", N.dec(D.fwi.mean()))
        b.metric("FWI maximum", N.dec(D.fwi.max()))
        c.metric("Communes en danger ≥ élevé",
                 N.nb((D.danger_effis >= 3).sum()))
        d.metric("Communes en danger extrême",
                 N.nb((D.danger_effis >= 5).sum()))
    else:
        seuil = D.score.quantile(.90)
        pire = D.loc[D.score.idxmax()]
        a.metric("Communes dans le décile à risque", N.nb((D.score >= seuil).sum()))
        b.metric("La plus exposée", pire.nom, pire.dep_nom, delta_color="off")
        c.metric("FWI moyen du jour", N.dec(D.fwi.mean()))
        d.metric("Communes en danger ≥ élevé",
                 N.nb((D.danger_effis >= 3).sum()))


def ouvrir_fiche(etat) -> None:
    """Un clic sur une commune ouvre sa fiche."""
    objs = (etat or {}).get("selection", {}).get("objects", {})
    for lignes in objs.values():
        if lignes:
            st.session_state["commune"] = lignes[0]["code_insee"]
            st.switch_page("pages/1_Commune.py")


def aller_a(code: str) -> None:
    st.session_state["commune"] = code
    st.switch_page("pages/1_Commune.py")


def carte(D: pd.DataFrame, titre: str, couche: str, vue=None, hauteur=460,
          cle=None):
    """Une carte, pour les affichages côte à côte."""
    st.markdown(f"**{titre}**")
    st.pydeck_chart(pdk.Deck(
        map_style="light",
        initial_view_state=vue or pdk.ViewState(**VUE_FRANCE),
        layers=[couche_communes(D, couche)], tooltip=INFOBULLE),
        height=hauteur, width="stretch")


# ════════════════════════════════════════════════════════════════════════
#  les réglages fréquents, au-dessus de la carte
# ════════════════════════════════════════════════════════════════════════
if "date_carte" not in st.session_state:
    st.session_state["date_carte"] = _date(2025, 8, 15)


def _appliquer_repere():
    """Un raccourci écrit la date ; `st.date_input` la relit ensuite."""
    lab = st.session_state.get("repere")
    if not lab:
        return
    v = N.DATES_REPERES[lab]
    d = _date.today() if v is None else _date(*v)
    st.session_state["date_carte"] = min(max(d, _date(N.AN_OBS_MIN, 1, 1)),
                                         _date(N.AN_PROJ_MAX, 12, 31))


haut = st.container()
with haut:
    c1, c2 = st.columns([2.6, 1.4])
    with c1:
        couche = st.segmented_control(
            "Couche", ["Risque prédit", "Risque par km²", "Danger météo (FWI)"],
            default="Risque prédit", label_visibility="collapsed",
            disabled=retro) or "Risque prédit"
    with c2:
        q = st.text_input("Rechercher", placeholder="Marseille · 13001 · 2A004",
                          label_visibility="collapsed")

    d1, d2 = st.columns([2.6, 1.4])
    with d1:
        if not retro:
            st.pills("Raccourcis", list(N.DATES_REPERES), key="repere",
                     on_change=_appliquer_repere, label_visibility="collapsed")
    with d2:
        pass

    date_b = None
    if retro:
        # ⚠️ Bornes imposées au jeu de test. Le sélecteur ne propose même pas
        # les autres années : une plage qui ne les contient pas est plus
        # claire qu'un message d'erreur après coup.
        e1, e2 = st.columns([1, 3])
        d = e1.date_input("Date", _date(TEST_MIN + 1, 8, 15),
                          min_value=_date(TEST_MIN, 1, 1),
                          max_value=_date(TEST_MAX, 12, 31), format="DD/MM/YYYY")
        date = pd.Timestamp(d)
        e2.caption("Les deux modèles sont servis avec la météo réellement "
                   "observée ce jour-là, celle qu'ils ont vue pendant "
                   "l'évaluation.")
    elif mode == "Une date précise":
        e1, e2 = st.columns([1, 3])
        d = e1.date_input("Date", key="date_carte",
                          min_value=_date(N.AN_OBS_MIN, 1, 1),
                          max_value=_date(N.AN_PROJ_MAX, 12, 31),
                          format="DD/MM/YYYY")
        date = pd.Timestamp(d)
    else:
        e1, e2, e3 = st.columns([1, 1, 2])
        da = e1.date_input("Référence", _date(1990, 8, 15),
                           min_value=_date(N.AN_OBS_MIN, 1, 1),
                           max_value=_date(N.AN_PROJ_MAX, 12, 31),
                           format="DD/MM/YYYY")
        db = e2.date_input("Comparaison", _date(2100, 8, 15),
                           min_value=_date(N.AN_OBS_MIN, 1, 1),
                           max_value=_date(N.AN_PROJ_MAX, 12, 31),
                           format="DD/MM/YYYY")
        date, date_b = pd.Timestamp(da), pd.Timestamp(db)

# ── la recherche, juste sous les réglages ────────────────────────────────
trouvees = N.chercher(q) if q else None
cible = None
if trouvees is not None:
    if not len(trouvees):
        st.info(f"Aucune commune ne correspond à « {q} ».")
    else:
        st.caption(f"{len(trouvees)} résultat(s)")
        for _, t in trouvees.head(6).iterrows():
            r1, r2, r3 = st.columns([3, 3, 1.4])
            r1.markdown(f"**{t.nom}**")
            r2.caption(f"{t.code_postal or ''} · {t.dep_nom}")
            if r3.button("Voir la fiche", key=f"f{t.code_insee}",
                         width="stretch"):
                aller_a(t.code_insee)
        cible = trouvees.iloc[0]

st.divider()

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
        carte(Rc, "Modèle C · physique pure, déployable", couche)
    with d_:
        carte(Rv, "Modèle v3 · avec l'historique des feux", couche)

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
#  la carte principale
# ════════════════════════════════════════════════════════════════════════
R = N.predire(date, scenario)
projete = date.year > N.AN_OBS_MAX

# ── comparer deux dates : deux cartes côte à côte ───────────────────────
if date_b is not None:
    Rb = N.predire(date_b, scenario)
    st.markdown(f"## {N.date_fr(date)} contre {N.date_fr(date_b)}")
    legende(couche)
    g, d_ = st.columns(2)
    with g:
        carte(R, N.date_fr(date), couche)
    with d_:
        carte(Rb, N.date_fr(date_b), couche)

    j = R[["code_insee", "nom", "dep_nom", "fwi", "score", "danger_effis"]].merge(
        Rb[["code_insee", "fwi", "score", "danger_effis"]], on="code_insee",
        suffixes=("_a", "_b"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("FWI moyen", N.dec(j.fwi_b.mean()),
              N.dec(j.fwi_b.mean() - j.fwi_a.mean()))
    m2.metric("Communes en danger ≥ élevé", N.nb((j.danger_effis_b >= 3).sum()),
              N.nb((j.danger_effis_b >= 3).sum() - (j.danger_effis_a >= 3).sum()))
    m3.metric("Communes dont le FWI monte", N.nb((j.fwi_b > j.fwi_a).sum()),
              f"{N.pct((j.fwi_b > j.fwi_a).mean())} du territoire")
    m4.metric("Risque médian", f"×{N.dec((j.score_b / j.score_a).median(), 2)}")

    j["hausse"] = j.fwi_b - j.fwi_a
    st.markdown("##### Les 10 communes où le danger progresse le plus")
    t = j.nlargest(10, "hausse")[["nom", "dep_nom", "fwi_a", "fwi_b", "hausse"]]
    st.dataframe(t.rename(columns={
        "nom": "Commune", "dep_nom": "Département",
        "fwi_a": f"FWI {date.year}", "fwi_b": f"FWI {date_b.year}",
        "hausse": "Écart"}).round(1), width="stretch", hide_index=True)
    st.stop()

# ── une date ────────────────────────────────────────────────────────────
if projete:
    st.warning(
        f"**{N.date_fr(date)} n'est pas une prévision météo.** Personne ne "
        f"connaît le temps qu'il fera ce jour-là. La carte montre ce que "
        f"vaudrait un {date.day} {MOIS[date.month - 1]} ordinaire sous le "
        f"climat de {date.year} : le cycle saisonnier vient des observations "
        f"2006-2019, seul son niveau est décalé par le réchauffement projeté "
        f"({scenario.upper().replace('_', '.')}).")

indicateurs(R, couche)
legende(couche)

# ⚠️ `st.pydeck_chart` GARDE LA CAMÉRA entre deux exécutions quand il porte
# une `key` : changer `initial_view_state` ne suffit donc pas à recentrer.
# On fait varier la clé pour obtenir un widget neuf. Effet de bord accepté :
# la sélection en cours est perdue au recentrage.
vue, cle = pdk.ViewState(**VUE_FRANCE), "france"
couches = [couche_communes(R, couche)]
if cible is not None:
    ligne = R[R.code_insee == cible.code_insee]
    couches.append(pdk.Layer(
        "ScatterplotLayer", ligne, get_position=["lon", "lat"],
        get_fill_color=[0, 0, 0, 0], get_radius=4000, stroked=True,
        get_line_color=[20, 20, 20], line_width_min_pixels=2.5))
    vue = pdk.ViewState(latitude=float(cible.lat), longitude=float(cible.lon),
                        zoom=9)
    cle = f"z{cible.code_insee}"

gauche, droite = st.columns([3, 1])

with gauche:
    etat = st.pydeck_chart(
        pdk.Deck(map_style="light", layers=couches, initial_view_state=vue,
                 tooltip=INFOBULLE),
        height=580, width="stretch", on_select="rerun",
        selection_mode="single-object", key=f"carte-{cle}")
    ouvrir_fiche(etat)

    _manq = len(N.contours_manquants())
    st.caption(
        f"Survolez une commune pour la lire, cliquez pour ouvrir sa fiche."
        + (f" {_manq} communes sur {N.nb(len(R))} n'ont pas de contour dans le "
           f"référentiel géographique et n'apparaissent pas : leur code a "
           f"changé entre le millésime du fond de carte et le COG 2026."
           if _manq else ""))

with droite:
    st.markdown("##### Où regarder ce jour-là")
    top = R.nlargest(10, "score")
    for i, (_, t) in enumerate(top.iterrows(), 1):
        cl = int(t.danger_effis)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:7px;"
            f"margin-bottom:1px'>"
            f"<span style='color:#9b9992;font-size:11px;width:14px'>{i}</span>"
            f"<span style='width:9px;height:9px;border-radius:50%;"
            f"background:{N.COUL_EFFIS[cl]};display:inline-block'></span>"
            f"<span style='font-size:13px'><b>{t.nom}</b></span></div>"
            f"<div style='color:#9b9992;font-size:10.5px;margin:0 0 4px 30px'>"
            f"{t.dep_nom} · FWI {N.dec(t.fwi)}</div>",
            unsafe_allow_html=True)
        if st.button("fiche", key=f"t{t.code_insee}", width="stretch"):
            aller_a(t.code_insee)

if couche.startswith("Risque par"):
    st.info("""
Cette vue divise le score par la surface de la commune.

Le modèle prédit « cette commune aura-t-elle au moins un feu aujourd'hui ». Une
commune vaste a mécaniquement plus de chances d'en contenir un : la corrélation
entre le score et la superficie vaut 0,55. Fontainebleau, 172 km² de forêt,
ressort donc très au-dessus de Melun et Barbizon, qui partagent pourtant sa
météo et son FWI de 5,1.

Le modèle a raison pour sa question. Dans les données observées, les plus
grandes communes brûlent 20 fois plus souvent que les plus petites, mais
seulement 3 fois plus par km². Cette vue répond au « le sol est-il dangereux
ici », l'autre au « quelle commune surveiller ».
""")

st.divider()
st.caption("Le risque prédit est un classement, pas une probabilité : la "
           "calibration disponible a été ajustée sur un autre modèle et une "
           "autre période, elle serait fausse d'un facteur 2. Le FWI est "
           "l'indice officiel, il se lit directement sur l'échelle EFFIS.")
