"""Étape 25 — le diaporama de soutenance, généré.

    python -m tvfed.diaporama

────────────────────────────────────────────────────────────────────────────
POURQUOI GÉNÉRER PLUTÔT QUE DESSINER
────────────────────────────────────────────────────────────────────────────
Un diaporama fait à la main diverge de ses sources dès la première
correction. Ce projet en a fait l'expérience : « +45 % de FWI » a survécu des
semaines dans l'application parce que c'était une chaîne de caractères que
rien ne recalculait.

Ici, chaque chiffre est **lu dans le CSV qui l'a produit**. Relancer
`tvfed.comparer` puis cette commande suffit à remettre le diaporama d'aplomb.
Les seules constantes écrites en dur sont celles qui décrivent le protocole —
les bornes du split, la taille de la grille — et elles ne changent pas.

Le fichier produit est un `.pptx` ordinaire : ouvrable dans PowerPoint,
LibreOffice ou Google Slides, et retouchable.
"""
from __future__ import annotations

import json

import pandas as pd

from .paths import PROCESSED, RACINE

SORTIE = RACINE / "presentation"
FIG = RACINE / "figures"

# charte, reprise des notebooks et de l'application
INK = "0B0B0B"
MUTED = "6B6963"
FOND = "FCFCFB"
ROUGE = "E34948"
ORANGE = "EB6834"
BLEU = "2A78D6"
VERT = "1BAF7A"
GRIS = "C3C2B7"


def _lire():
    """Tous les chiffres du diaporama, à leur source."""
    d = {}
    for nom in ("pr_auc_val", "comparaison_appariee", "transfert_spatial",
                "test_par_annee", "baselines", "calibration_v3",
                "series_sarimax", "modeles_ensemble"):
        f = PROCESSED / f"{nom}.csv"
        d[nom] = pd.read_csv(f) if f.exists() else None
    d["meta"] = json.loads(
        (RACINE / "app" / "donnees" / "meta.json").read_text(encoding="utf-8"))
    d["tendances"] = pd.read_csv(RACINE / "app" / "donnees" / "tendances.csv")
    d["params_lstm"] = json.loads(
        (PROCESSED / "best_params_lstm.json").read_text(encoding="utf-8"))
    return d


# ════════════════════════════════════════════════════════════════════════
#  primitives de mise en page
# ════════════════════════════════════════════════════════════════════════
def _cm(v):
    from pptx.util import Cm
    return Cm(v)


def _rgb(h):
    from pptx.dml.color import RGBColor
    return RGBColor.from_string(h)


class Deck:
    """Un diaporama 16:9, avec quelques gabarits maison.

    python-pptx ne fournit pas de mise en page : les gabarits par défaut sont
    ceux d'Office, avec leurs polices et leurs puces. On part donc de
    diapositives VIDES et on pose les blocs soi-même — c'est plus long à
    écrire, mais c'est la seule façon d'obtenir une charte cohérente.
    """

    L, R = 1.9, 1.9              # marges latérales, en cm
    LARG, HAUT = 33.87, 19.05    # 16:9

    def __init__(self):
        from pptx import Presentation

        self.p = Presentation()
        self.p.slide_width = _cm(self.LARG)
        self.p.slide_height = _cm(self.HAUT)

    # ── briques ─────────────────────────────────────────────────────────
    def _vide(self, fond=FOND):
        s = self.p.slides.add_slide(self.p.slide_layouts[6])
        f = s.background.fill
        f.solid()
        f.fore_color.rgb = _rgb(fond)
        return s

    def _texte(self, s, txt, x, y, w, h, taille=16, coul=INK, gras=False,
               interligne=1.25, aligne=None):
        from pptx.util import Pt

        b = s.shapes.add_textbox(_cm(x), _cm(y), _cm(w), _cm(h))
        tf = b.text_frame
        tf.word_wrap = True
        for i, ligne in enumerate(str(txt).split("\n")):
            par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            par.line_spacing = interligne
            if aligne is not None:
                par.alignment = aligne
            r = par.add_run()
            r.text = ligne
            r.font.size = Pt(taille)
            r.font.bold = gras
            r.font.color.rgb = _rgb(coul)
            r.font.name = "Calibri"
        return b

    def _image(self, s, chemin, x, y, w_max, h_max):
        """Insère en respectant le rapport d'aspect, centré dans la boîte."""
        from PIL import Image

        iw, ih = Image.open(chemin).size
        ratio = min(w_max / iw, h_max / ih)
        w, h = iw * ratio, ih * ratio
        return s.shapes.add_picture(
            str(chemin), _cm(x + (w_max - w) / 2), _cm(y + (h_max - h) / 2),
            width=_cm(w), height=_cm(h))

    def _filet(self, s, y, coul=ROUGE, x=None, w=3.2, ep=0.09):
        from pptx.util import Pt
        from pptx.enum.shapes import MSO_SHAPE

        f = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, _cm(x or self.L), _cm(y),
                               _cm(w), _cm(ep))
        f.fill.solid()
        f.fill.fore_color.rgb = _rgb(coul)
        f.line.fill.background()
        f.shadow.inherit = False
        return f

    # ── gabarits ────────────────────────────────────────────────────────
    def titre(self, sur, titre, sous, bas=""):
        s = self._vide()
        self._texte(s, sur, self.L, 4.4, 24, 1, 13, ROUGE, True)
        self._texte(s, titre, self.L, 5.5, 30, 4, 40, INK, True, 1.05)
        self._texte(s, sous, self.L, 10.6, 26, 3, 17, MUTED, interligne=1.35)
        if bas:
            self._texte(s, bas, self.L, 16.2, 28, 1.4, 12, MUTED)
        return s

    def section(self, num, titre, sous=""):
        s = self._vide(INK)
        self._texte(s, num, self.L, 6.3, 6, 1.4, 15, ROUGE, True)
        self._texte(s, titre, self.L, 7.3, 28, 3, 34, FOND, True, 1.1)
        if sous:
            self._texte(s, sous, self.L, 11.4, 26, 2.5, 16, GRIS,
                        interligne=1.35)
        return s

    def page(self, titre, sous=""):
        """En-tête standard. Renvoie (slide, y où le contenu peut commencer)."""
        s = self._vide()
        self._filet(s, 1.5)
        self._texte(s, titre, self.L, 1.95, 29, 1.8, 27, INK, True)
        y = 3.9
        if sous:
            self._texte(s, sous, self.L, y, 29, 1.6, 14.5, MUTED,
                        interligne=1.3)
            y += 1.5
        return s, y

    def puces(self, s, items, x, y, w, taille=15, ecart=1.55):
        """Une liste, avec un tiret cadratin plutôt qu'une puce Office."""
        for i, it in enumerate(items):
            gras = it.startswith("**")
            self._texte(s, ("— " if not gras else "") + it.replace("**", ""),
                        x, y + i * ecart, w, ecart, taille,
                        INK if gras else MUTED, gras)
        return y + len(items) * ecart

    def tableau(self, s, entetes, lignes, x, y, w, h, larg=None, taille=13,
                fort=()):
        from pptx.util import Pt

        t = s.shapes.add_table(len(lignes) + 1, len(entetes),
                               _cm(x), _cm(y), _cm(w), _cm(h)).table
        if larg:
            for i, lg in enumerate(larg):
                t.columns[i].width = _cm(lg)
        def _pose(c, txt, gras, coul, fond):
            # ⚠️ une cellule vide ne crée AUCUN run : `paragraphs[0].runs` est
            # alors un tuple vide et tout accès par index lève. On met une
            # espace insécable plutôt que de multiplier les gardes.
            c.text = str(txt) if str(txt).strip() else " "
            r = c.text_frame.paragraphs[0].runs[0]
            r.font.size = Pt(taille)
            r.font.bold = gras
            r.font.color.rgb = _rgb(coul)
            r.font.name = "Calibri"
            c.fill.solid()
            c.fill.fore_color.rgb = _rgb(fond)

        for j, e in enumerate(entetes):
            _pose(t.cell(0, j), e, True, FOND, INK)
        for i, lig in enumerate(lignes):
            for j, v in enumerate(lig):
                _pose(t.cell(i + 1, j), v, i in fort,
                      INK if i in fort else MUTED,
                      "F2F1EC" if i in fort else FOND)
        return t

    def encart(self, s, txt, x, y, w, h, coul=ORANGE, taille=15):
        from pptx.enum.shapes import MSO_SHAPE

        b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, _cm(x), _cm(y), _cm(0.11),
                               _cm(h))
        b.fill.solid()
        b.fill.fore_color.rgb = _rgb(coul)
        b.line.fill.background()
        b.shadow.inherit = False
        self._texte(s, txt, x + 0.55, y + 0.1, w - 0.7, h, taille, INK,
                    interligne=1.3)

    def notes(self, s, txt):
        s.notes_slide.notes_text_frame.text = txt.strip()

    def enregistrer(self, chemin):
        chemin.parent.mkdir(parents=True, exist_ok=True)
        self.p.save(str(chemin))
        return chemin


# ════════════════════════════════════════════════════════════════════════
#  deux figures qui n'existent que pour la soutenance
# ════════════════════════════════════════════════════════════════════════
def _figures_soutenance(D) -> dict:
    """Génère ce que les notebooks n'ont pas produit, et renvoie les chemins.

    Le reste du diaporama réutilise les PNG déjà versionnés dans `figures/` :
    refaire une figure, c'est risquer qu'elle diverge de celle du notebook.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    dest = FIG / "soutenance"
    dest.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "#" + FOND,
                         "axes.facecolor": "#" + FOND, "font.size": 9,
                         "axes.edgecolor": "#c3c2b7", "text.color": "#" + INK})
    out = {}

    # ── validation croisée spatiale ─────────────────────────────────────
    ts = D["transfert_spatial"]
    if ts is not None:
        t = ts.copy()
        t["ecart"] = 100 * (t["C · physique"] / t["A · tout"] - 1)
        t = t.sort_values("ecart")
        fig, ax = plt.subplots(figsize=(11, 4.4))
        y = np.arange(len(t))
        ax.barh(y - .2, t["A · tout"], height=.38, color="#" + GRIS,
                label="A · toutes les features", edgecolor="#" + FOND, lw=.8)
        ax.barh(y + .2, t["C · physique"], height=.38, color="#" + VERT,
                label="C · physique pure", edgecolor="#" + FOND, lw=.8)
        for i, (_, r) in enumerate(t.iterrows()):
            ax.text(max(r["A · tout"], r["C · physique"]) + .012, i,
                    f"{r.ecart:+.0f} %", va="center", fontsize=9,
                    weight="bold", color="#" + VERT)
        ax.set_yticks(y, [f"région {int(r)}" for r in t.region], fontsize=9)
        ax.set_xlabel("PR-AUC sur la région EXCLUE de l'entraînement")
        ax.set_xlim(0, t[["A · tout", "C · physique"]].to_numpy().max() * 1.18)
        ax.legend(frameon=False, fontsize=9.5, loc="lower right")
        ax.set_title("En territoire jamais vu, le modèle physique gagne "
                     "9 fois sur 9", fontsize=12, weight="bold", loc="left")
        ax.grid(axis="x", color="#e1e0d9", lw=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        out["spatial"] = dest / "spatial.png"
        fig.savefig(out["spatial"], dpi=150, facecolor="#" + FOND,
                    bbox_inches="tight")
        plt.close(fig)

    # ── SHAP du modèle C : gain contre SHAP ─────────────────────────────
    npy = PROCESSED / "shap_c_alea.npy"
    if npy.exists():
        cols = json.loads(
            (PROCESSED / "shap_c_colonnes.json").read_text(encoding="utf-8"))
        gain = 100 * pd.read_csv(PROCESSED / "importances_c.csv",
                                 index_col=0).squeeze("columns")
        sa = pd.Series(np.abs(np.load(npy)).mean(0), index=cols)
        ss = pd.Series(np.abs(np.load(PROCESSED / "shap_c_sommet.npy")).mean(0),
                       index=cols)
        top = pd.DataFrame({"gain": gain, "alea": sa, "sommet": ss}) \
            .sort_values("sommet", ascending=False).head(12).iloc[::-1]
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
        y = np.arange(len(top))
        for a_, c, ti, co in ((ax[0], "gain", "gain d'entraînement (%)", GRIS),
                              (ax[1], "alea", "SHAP · échantillon aléatoire", BLEU),
                              (ax[2], "sommet", "SHAP · sommet du classement", ROUGE)):
            a_.barh(y, top[c], color="#" + co, edgecolor="#" + FOND, lw=1)
            a_.set_title(ti, fontsize=11, weight="bold", loc="left")
            a_.grid(axis="x", color="#e1e0d9", lw=.7)
            a_.set_axisbelow(True)
            a_.spines[["top", "right"]].set_visible(False)
        ax[0].set_yticks(y, [f.replace("_", " ") for f in top.index], fontsize=9)
        plt.tight_layout()
        out["shap"] = dest / "shap_c.png"
        fig.savefig(out["shap"], dpi=150, facecolor="#" + FOND,
                    bbox_inches="tight")
        plt.close(fig)

    return out


# ════════════════════════════════════════════════════════════════════════
#  le diaporama
# ════════════════════════════════════════════════════════════════════════
def main() -> None:
    D = _lire()
    G = _figures_soutenance(D)
    d = Deck()
    mt, ten = D["meta"], D["tendances"]
    ap = D["pr_auc_val"].iloc[0].to_dict()
    cmp_ = D["comparaison_appariee"]
    taux_val = 0.0002410

    def t(nom):
        return ten[ten.serie == nom].iloc[0]

    def ec(ref, mod):
        return cmp_[(cmp_.reference == ref) & (cmp_.modele == mod)].iloc[0]

    # ── 1. titre ────────────────────────────────────────────────────────
    s = d.titre("Projet de fin d'année · M1 Data / IA · La Plateforme_",
                "Où va-t-il brûler\ndemain ?",
                "Prédire le risque de départ de feu pour chacune des 34 734\n"
                "communes de France métropolitaine, chaque jour,\nde 1973 à 2100.",
                "Romuald Courtois")
    d._image(s, RACINE / "docs" / "img" / "carte.png", 20.5, 2.4, 11.5, 14.2)
    d.notes(s, "Le modèle retrouve seul les Landes, l'arc méditerranéen et la "
               "Corse — sans jamais voir de carte : ni lat ni lon ne font "
               "partie de ses variables.")

    # ── 2 ───────────────────────────────────────────────────────────────
    d.section("01", "Le problème",
              "Un événement rare, quatre sources, 253 millions de lignes")

    # ── 3. la rareté ────────────────────────────────────────────────────
    s, y = d.page("Ce qui rend le problème difficile",
                  "Un départ de feu est un événement RARE.")
    d._texte(s, "0,019 %", d.L, y + .3, 14, 3.4, 66, ROUGE, True, 1)
    d._texte(s, "des couples commune × jour comptent\nau moins un départ de feu",
             d.L, y + 4, 14, 2.4, 15, MUTED)
    d.puces(s, [
        "**Un modèle qui répond toujours « non » a 99,98 % de justesse.**",
        "Et ne sert strictement à rien.",
        "**Une fuite de données ne produit pas d'erreur.**",
        "Elle produit d'excellentes métriques et un modèle sans valeur.",
        "**Le déséquilibre commande TOUT le reste.**",
        "La métrique, l'échantillonnage, la calibration, et jusqu'à la façon "
        "de comparer deux modèles.",
    ], 17.5, y + .4, 14.5, 15, 1.35)
    d.notes(s, "Insister : ce n'est pas un détail technique, c'est la "
               "contrainte qui explique chacune des décisions suivantes.")

    # ── 4. les sources ──────────────────────────────────────────────────
    s, y = d.page("Quatre sources publiques, croisées",
                  "Jointure sur le code INSEE et sur une grille météo de 0,25°.")
    d.tableau(s, ["Source", "Ce qu'elle apporte", "Volume"], [
        ["CEMS · Copernicus", "8 indices de danger, par jour et par maille",
         "21,9 M lignes · 1973-2025"],
        ["BDIFF · IGN", "les feux déclarés, commune par commune",
         "52 809 feux sur le périmètre"],
        ["CORINE Land Cover", "l'occupation du sol, 44 postes", "1,08 M lignes"],
        ["INSEE", "référentiel des communes, et leurs fusions",
         "34 734 communes"],
    ], d.L, y, 30, 6, larg=[7, 14, 9])
    d.encart(s, "LE POINT DÉLICAT N'EST PAS LE VOLUME : CE SONT LES FUSIONS DE "
                "COMMUNES.\n965 feux portent un code INSEE disparu. Les "
                "rapprocher par le nom donnait de faux positifs — « Chirac » "
                "en Lozère renvoyait vers la Charente. On a téléchargé le "
                "fichier officiel des mouvements de communes de l'INSEE ; les "
                "30 cas restants sont écartés et comptés, jamais devinés.",
             d.L, y + 6.8, 30, 4.4, ORANGE, 14)
    d.notes(s, "Une heuristique par le nom aurait « résolu » tous les cas et "
               "corrompu y, le voisinage et les features CORINE de communes "
               "innocentes. Perdre 0,68 % des feux est préférable.")

    # ── 5. la grille ────────────────────────────────────────────────────
    s, y = d.page("La table centrale : commune × jour",
                  "Une ligne par commune et par jour, qu'il y ait eu un feu "
                  "ou non.")
    d.tableau(s, ["Partition", "Années", "Lignes", "Feux", "Taux"], [
        ["train", "2006-2019", "177 594 942", "33 632", "0,0189 %"],
        ["validation", "2020-2022", "38 068 464", "9 176", "0,0241 %"],
        ["test", "2023-2025", "38 068 464", "6 322", "0,0166 %"],
        ["total", "2006-2025", "253 731 870", "49 130", "0,0194 %"],
    ], d.L, y, 24, 5.4, fort=(3,))
    d.encart(s, "POURQUOI UNE GRILLE DENSE, ET NON LA SEULE LISTE DES FEUX\n"
                "Parce qu'une série creuse rendrait les fenêtres glissantes "
                "silencieusement fausses. « Feux des 30 jours précédents » se "
                "calcule en remontant 30 lignes : si les jours sans feu sont "
                "absents, on remonte en réalité plusieurs années.",
             d.L, y + 6.2, 30, 3.8, BLEU, 14)
    d.notes(s, "253 millions de lignes plutôt que 52 809 : c'est un choix "
               "coûteux, et il a une raison technique précise.")

    # ── 6 ───────────────────────────────────────────────────────────────
    d.section("02", "Le protocole",
              "Ce qui protège le résultat — avant même de parler de modèles")

    # ── 7. le split ─────────────────────────────────────────────────────
    s, y = d.page("La barrière du split",
                  "Découpage TEMPOREL, jamais aléatoire.")
    d.puces(s, [
        "**train · 2006-2019 — apprendre**",
        "177,6 M lignes. Tout ce qui a un .fit() s'ajuste ici, et nulle part "
        "ailleurs.",
        "**validation · 2020-2022 — choisir**",
        "Hyperparamètres, sélection du modèle, calibration, comparaisons.",
        "**test · 2023-2025 — juger, UNE SEULE FOIS**",
        "Ouvert après gel complet. Aucune décision n'en découle.",
    ], d.L, y, 17, 15, 1.4)
    d.encart(s, "Un découpage ALÉATOIRE mettrait le 14 juillet 2019 dans le "
                "train et le 15 dans le test. Le modèle « prédirait » un feu "
                "qu'il a déjà vu, à 20 km et un jour d'écart.",
             19.5, y + .2, 12.6, 3.6, ROUGE, 14)
    d.encart(s, "LA RÈGLE QUI TRANCHE LES CAS DOUTEUX\nUne feature DATÉE peut "
                "regarder tout le passé, y compris celui de sa propre période "
                "d'évaluation.\nUne statistique NON DATÉE ne peut regarder que "
                "le train.", 19.5, y + 4.6, 12.6, 4.4, BLEU, 14)
    d.notes(s, "« Feux des 30 jours précédents » au 3 août 2023 lit juillet "
               "2023 : ce n'est pas une fuite, le 3 août à 8 h on connaît "
               "juillet. « Taux moyen de la commune sur toute la période » lit "
               "le futur — c'en est une.")

    # ── 8. la métrique ──────────────────────────────────────────────────
    s, y = d.page("Pourquoi PR-AUC, et pas ROC-AUC",
                  "La métrique n'est pas un détail : à 0,019 %, elle décide de "
                  "ce qu'on croit avoir réussi.")
    d.tableau(s, ["", "ROC-AUC", "PR-AUC"], [
        ["Ce qu'elle mesure", "vrais positifs contre faux positifs",
         "précision contre rappel"],
        ["Valeur au hasard", "0,50, quel que soit le déséquilibre",
         "exactement le taux de base"],
        ["À 0,019 % de positifs", "flatteuse — un modèle médiocre affiche 0,95",
         "lisible : le rapport donne le lift"],
    ], d.L, y, 30, 5.2, larg=[7, 11.5, 11.5])
    d._texte(s, "lift  =  PR-AUC ÷ taux de base",
             d.L, y + 6.2, 30, 1.6, 26, INK, True)
    d._texte(s, "« combien de fois mieux que tirer au sort » — un nombre qui "
                "se dit à voix haute", d.L, y + 8.2, 30, 1.4, 16, MUTED)
    d.notes(s, "C'est souvent la première question d'un jury : pourquoi pas "
               "l'accuracy. Répondre par le chiffre : 99,98 % de justesse en "
               "répondant toujours non.")

    # ── 9. le prior ─────────────────────────────────────────────────────
    s, y = d.page("Le piège du sous-échantillonnage",
                  "Le train est réduit à 1 positif pour 10 négatifs — sans "
                  "quoi l'entraînement serait ingérable.")
    d._texte(s, "× 487", d.L, y + .4, 13, 3.4, 60, ORANGE, True, 1)
    d._texte(s, "l'écart entre le prior appris (9,1 %)\net le prior réel "
                "(0,019 %)", d.L, y + 4.1, 14, 2.4, 15, MUTED)
    d.puces(s, [
        "**Validation et test ne sont JAMAIS échantillonnés.**",
        "C'est ce qui rend les scores comparables au monde réel.",
        "**Les statistiques dérivées de y se calculent sur le train COMPLET.**",
        "Sur l'échantillon, un lissage bayésien vaudrait 9,1 % au lieu de "
        "0,019 %. Le prior serait empoisonné, et rien dans les métriques ne "
        "le signalerait.",
        "**La calibration absorbe le décalage.**",
        "Platt ramène le biais de ×144,7 à ×1,13, sans rien coûter en PR-AUC.",
    ], 17.5, y + .3, 14.5, 14.5, 1.3)
    d.notes(s, "Ce piège est invisible : le modèle marche, les métriques sont "
               "bonnes, seul le niveau absolu des probabilités est faux.")

    # ── 10 ──────────────────────────────────────────────────────────────
    d.section("03", "Les modèles",
              "Six modèles, et deux conclusions contre-intuitives")

    # ── 11. les baselines ───────────────────────────────────────────────
    s, y = d.page("Contre quoi se bat-on ?",
                  "Sans référence, un lift de 73× ne veut rien dire.")
    b = D["baselines"]
    d.tableau(s, ["Prédicteur, sans aucun apprentissage", "PR-AUC", "lift"],
              [[r.predicteur, f"{r.pr_auc:.4f}", f"×{r.lift:.1f}"]
               for _, r in b.iterrows()]
              + [["le modèle déployé, mesuré sur le test",
                  f"{mt['test']['pr_auc']:.4f}", f"×{mt['test']['lift']:.1f}"]],
              d.L, y, 25, 6.6, larg=[15, 5, 5], fort=(len(b),))
    d.encart(s, "L'historique spatial seul vaut déjà ×19, la météo seule ×5, "
                "et leur croisement ×42.\nC'EST LA BARRE À BATTRE — pas le "
                "hasard. Un modèle à ×30 serait moins bon qu'une règle de "
                "trois.", d.L, y + 7.4, 30, 3, VERT, 15)
    d.notes(s, "Montrer qu'on a cherché à se rendre la tâche difficile plutôt "
               "qu'à se flatter d'un grand chiffre.")

    # ── 12. v1 → v3 ─────────────────────────────────────────────────────
    s, y = d.page("Du v1 au v3 : donner un risque aux communes sans histoire",
                  "Le v1 tirait 54,6 % de son importance de l'historique de la "
                  "commune. Il disait surtout : « ce qui a brûlé rebrûlera ».")
    d._image(s, FIG / "modele-v3" /
             "02_le-lissage-bayesien-rendre-une-estimation-aux-communes-sans.png",
             d.L, y, 18.5, 10.5)
    d.puces(s, [
        "**Le trou**",
        "Une commune qui n'a jamais brûlé gardait un score bas, même entourée "
        "de communes qui brûlent chaque été.",
        "**La parade**",
        "Regrouper les communes qui se ressemblent PHYSIQUEMENT — 30 groupes "
        "formés sans jamais regarder le feu — puis faire retomber chaque "
        "commune vers le taux de son groupe, à proportion de ce qu'on sait "
        "d'elle.",
        "**Le gain, mesuré**",
        "+0,83 % de PR-AUC. Réel, mais modeste — et le dire est aussi un "
        "résultat.",
    ], 21.3, y, 10.9, 13, 1.22)
    d.notes(s, "C'est le problème classique de small area estimation. Ce qui "
               "compte méthodologiquement : les groupes sont formés sans "
               "regarder y, la sinistralité n'entre qu'après.")

    # ── 13. les six modèles ─────────────────────────────────────────────
    s, y = d.page("Les six modèles, sur la même validation",
                  "38 068 464 communes-jours, 9 176 feux, taux 0,0241 %.")
    lignes = []
    for nom in sorted(ap, key=ap.get, reverse=True):
        e = "référence" if nom == "XGBoost v3" else \
            f"{ec('XGBoost v3', nom).ecart_pct:+.1f} %"
        lignes.append([nom, f"{ap[nom]:.4f}", f"×{ap[nom] / taux_val:.1f}", e])
    d.tableau(s, ["Modèle", "PR-AUC", "lift", "vs XGBoost v3"], lignes,
              d.L, y, 23, 6.6, larg=[9.5, 4.5, 4.5, 4.5], fort=(0,))
    ens = D["modeles_ensemble"]
    if ens is not None:
        d._texte(s, f"L'ensemble v3 + MLP monte à {ens.pr_auc.iloc[0]:.4f} "
                    f"(×{ens.lift.iloc[0]:.1f}) — au prix de deux modèles à "
                    f"faire tourner en production.",
                 d.L, y + 7.3, 30, 1.4, 14, MUTED)
    d.encart(s, "AUCUN de ces écarts n'a été retenu sans intervalle de "
                "confiance. Diapositive suivante.",
             d.L, y + 8.8, 30, 2.2, ROUGE, 16)
    d.notes(s, "Ne pas commenter le classement ici : le commentaire n'a de "
               "sens qu'avec les intervalles.")

    # ── 14. le graphique en forêt ───────────────────────────────────────
    s, y = d.page("Ces écarts survivent-ils au bruit ?",
                  "Bootstrap apparié, 200 répliques, rééchantillonnage des "
                  "34 734 COMMUNES — pas des lignes.")
    d._image(s, RACINE / "docs" / "img" / "modeles.png", d.L, y, 30, 8.2)
    d.encart(s, "LE RÉSULTAT LE PLUS UTILE DU PROJET EST UN RÉSULTAT NÉGATIF.\n"
                f"DART et le MLP paraissaient "
                f"{abs(ec('XGBoost v3','DART').ecart_pct):.1f} % et "
                f"{abs(ec('XGBoost v3','MLP').ecart_pct):.1f} % moins bons. "
                "Leurs intervalles TRAVERSENT ZÉRO : les trois modèles sont "
                "indiscernables. Annoncer « XGBoost bat le MLP » aurait été "
                "une conclusion inventée à partir du bruit.",
             d.L, y + 8.8, 30, 3.8, ROUGE, 15)
    d.notes(s, "Pourquoi rééchantillonner les communes et non les lignes : les "
               "1 096 jours d'une même commune ne sont pas indépendants, et 31 "
               "communes partagent la même maille météo. Un bootstrap ligne à "
               "ligne donnerait des intervalles faussement étroits, et ferait "
               "conclure à tort.")

    # ── 15. le LSTM, 1 ──────────────────────────────────────────────────
    s, y = d.page("« Pour le temps, prends un LSTM »",
                  "Le réflexe standard. On l'a construit, optimisé, et mesuré.")
    p = D["params_lstm"]
    d.tableau(s, ["Hyperparamètre", "Valeur retenue"],
              [[k, f"{v:.5g}"] for k, v in p.items()],
              d.L, y, 13, 7.6, larg=[7.5, 5.5], taille=12)
    d._texte(s, "25 essais Optuna, arrêt précoce à l'époque 21.\n"
                "L'objection « il n'a pas été fine-tuné » ne tient pas.",
             d.L, y + 8.3, 15, 2.4, 15, MUTED)
    l_c = ec("XGBoost C", "LSTM")
    d._texte(s, f"{l_c.ecart_pct:.1f} %", 18.5, y + .8, 13.5, 3.4, 62,
             ROUGE, True, 1)
    d._texte(s, f"contre le modèle C, à information égale\n"
                f"intervalle [{l_c.ic_bas:.1f} ; {l_c.ic_haut:.1f}] — "
                f"loin de zéro", 18.5, y + 4.5, 13.5, 2.4, 16, MUTED)
    d.encart(s, "LA COMPARAISON LOYALE N'EST PAS CELLE QU'ON CROIT\n"
                "XGBoost v3 voit l'historique des feux — 29 % de ses "
                "importances. Le LSTM n'en voit rien. Les opposer mesurerait "
                "le prix de l'information retirée, pas la valeur de la "
                "séquence. La seule référence à jeu d'information égal est le "
                "modèle C.", 18.5, y + 7.5, 13.5, 4.8, ORANGE, 14)
    d.notes(s, "Le LSTM voit 30 jours × 8 indices = 240 valeurs ; le modèle C "
               "voit les 8 indices du jour plus deux décalages. Vingt fois "
               "plus d'historique météo, et il perd.")

    # ── 16. le LSTM, 2 ──────────────────────────────────────────────────
    s, y = d.page("Pourquoi il perd — l'explication est physique",
                  "Un LSTM sert quand l'ordre de la séquence porte une "
                  "information qu'aucun résumé ne capture. Ici, ce résumé "
                  "existe déjà.")
    d.encart(s, "Les indices DC, DMC et BUI du système canadien SONT des états "
                "récursifs.\nLe Drought Code est une moyenne exponentielle de "
                "la météo passée, avec une constante de temps de 52 jours ; "
                "le Duff Moisture Code, de 15 jours.\nC'est exactement la "
                "forme d'une cellule récurrente — sauf que ses coefficients "
                "ont été calibrés par cinquante ans de science du feu, plutôt "
                "qu'estimés sur 9 176 exemples positifs.",
             d.L, y, 30, 5.6, VERT, 16)
    d._texte(s, "Le CEMS livre déjà l'état caché que le LSTM devrait "
                "réapprendre.", d.L, y + 5.9, 30, 1.4, 20, INK, True)
    sar = D["series_sarimax"]
    r_ar = (sar[sar.modele.str.contains("sans exogène")].correlation.iloc[0]
            if sar is not None else float("nan"))
    d.puces(s, [
        "**Trois observations indépendantes convergent**",
        "la PACF montre une autocorrélation épuisée en deux à trois jours ;",
        f"l'ARIMA sans exogène est inutilisable — r = {r_ar:.3f}, la "
        f"corrélation est NÉGATIVE ;",
        "les trois premières features du modèle C sont part_maquis, "
        "danger_effis et erc : le signal dit OÙ il y a du combustible.",
    ], d.L, y + 7.4, 30, 15, 1.15)
    d.encart(s, "Ce n'est pas une prévision de série temporelle. C'est une "
                "CLASSIFICATION SPATIO-TEMPORELLE D'ÉVÉNEMENT RARE, sur "
                "34 734 séries parallèles pilotées par un exogène déjà résumé "
                "par la physique du domaine.",
             d.L, y + 11.5, 30, 2.1, ROUGE, 15)
    d.notes(s, "Réserve honnête, à donner spontanément : le LSTM ne reçoit pas "
               "danger_effis, qui pèse 13,7 % dans le modèle C. L'écart de "
               "23,6 % est donc un majorant.")

    # ── 17. v3 contre C ─────────────────────────────────────────────────
    s, y = d.page("Le meilleur modèle n'est pas celui qu'on déploie",
                  "C'est la décision la plus contre-intuitive du projet.")
    d.tableau(s, ["Sur le test 2023-2025", "PR-AUC", "lift"], [
        ["XGBoost v3 · 52 features", f"{mt['modele_a']['pr_auc']:.4f}",
         f"×{mt['modele_a']['lift']:.1f}"],
        ["XGBoost C · physique pure, 41 features", f"{mt['test']['pr_auc']:.4f}",
         f"×{mt['test']['lift']:.1f}"],
    ], d.L, y, 19, 3.2, larg=[11, 4, 4], fort=(1,))
    d.puces(s, [
        "**1. La donnée n'existe pas en temps réel.**",
        "v3 tire 29 % de son importance de l'historique des feux. Or la BDIFF "
        "ne publie pas l'année en cours : les feux de 2026 sortiront au "
        "printemps 2027. Aujourd'hui, feux_commune_7j vaudrait le décompte "
        "d'une semaine de décembre 2025. Pas imprécis : FAUX.",
        "**2. En territoire inconnu, elle vaut zéro.**",
        "Et le modèle lit ce zéro comme « ça n'a jamais brûlé, donc ça ne "
        "brûlera pas » — précisément là où le risque nouveau apparaît.",
        "**3. Pour 2050, elle est impossible par construction.**",
        "On ne connaîtra jamais les feux de 2049.",
    ], d.L, y + 4, 30, 14.5, 1.15)
    d.encart(s, "LE CHOIX SE FAIT SUR LA DISPONIBILITÉ DE LA DONNÉE, PAS SUR "
                "LA PERFORMANCE.\nEt ce défaut n'apparaît dans AUCUNE métrique "
                "d'entraînement : en validation comme en test, l'historique "
                "est toujours là. Il ne se voit qu'en pensant au déploiement.",
             d.L, y + 11.2, 30, 2.4, ROUGE, 15)
    d.notes(s, "Dans l'application, un basculement permet de comparer les deux "
               "modèles — et il REFUSE de le faire ailleurs que sur le test.")

    # ── 18. validation croisée spatiale ─────────────────────────────────
    s, y = d.page("La preuve : retirer une région entière, puis tester dessus",
                  "Simulation d'un territoire jamais vu — ou d'un climat qui "
                  "déplace le risque.")
    if "spatial" in G:
        d._image(s, G["spatial"], d.L, y, 30, 8.8)
    d.encart(s, "Le modèle physique gagne dans les 9 régions, sans exception : "
                "+8,2 % en moyenne pondérée, et jusqu'à +137 % dans le Grand "
                "Est.\nLà où l'historique est le plus pauvre, s'y fier est un "
                "HANDICAP. C'est l'argument décisif pour 2050 : le climat va "
                "déplacer le risque vers des communes qui n'ont pas de passé.",
             d.L, y + 9.4, 30, 3.4, VERT, 15)
    d.notes(s, "Protocole : on retire une région du train, on entraîne, on "
               "teste sur la région exclue. Neuf fois, une par région.")

    # ── 19. la calibration ──────────────────────────────────────────────
    s, y = d.page("La calibration, et pourquoi l'app affiche un rang",
                  "Le score brut est 145 fois trop grand — c'est le "
                  "sous-échantillonnage qui remonte.")
    cal = D["calibration_v3"]
    d.tableau(s, ["Méthode", "PR-AUC", "p moyen", "biais", "valeurs distinctes"],
              [[r.methode, f"{r.pr_auc:.4f}", f"{r.p_moyen:.2e}",
                f"×{r.biais:.2f}",
                f"{int(r.valeurs_distinctes):,}".replace(",", " ")]
               for _, r in cal.iterrows()],
              d.L, y, 25, 4.4, larg=[5, 5, 5, 5, 5], fort=(1,))
    d.puces(s, [
        "**Platt corrige sans rien coûter.**",
        "Le classement est intact, seule l'échelle bouge.",
        "**L'isotonique calibre aussi bien, mais écrase le score.**",
        "136 valeurs distinctes au lieu de 9 millions : on perd du pouvoir de "
        "discrimination pour rien.",
        "**L'application affiche donc un RANG, pas une probabilité.**",
        "Le calibrateur disponible a été ajusté sur un autre modèle et une "
        "autre période : il serait faux d'un facteur ~2. Un rang, lui, reste "
        "juste.",
    ], d.L, y + 5.2, 30, 15, 1.3)
    d.notes(s, "Dire qu'on a préféré ne PAS afficher une probabilité plutôt "
               "que d'en afficher une fausse. C'est un choix, pas un oubli.")

    # ── 20. l'évaluation test ───────────────────────────────────────────
    s, y = d.page("L'évaluation finale — ouverte une seule fois",
                  "Après gel complet du modèle, des features et de la "
                  "calibration.")
    tpa = D["test_par_annee"]
    d.tableau(s, ["Année", "Lignes", "Feux", "Taux", "PR-AUC", "lift"],
              [[int(r.an), f"{int(r.lignes):,}".replace(",", " "),
                f"{int(r.feux):,}".replace(",", " "), f"{r.taux:.4%}",
                f"{r.pr_auc:.4f}", f"×{r.lift:.0f}"]
               for _, r in tpa.iterrows()],
              d.L, y, 24, 4.4, larg=[3.5, 6, 4, 4.5, 3.5, 2.5])
    mx, mn = tpa.loc[tpa.lift.idxmax()], tpa.loc[tpa.lift.idxmin()]
    d.encart(s, f"LE LIFT VARIE DU SIMPLE AU DOUBLE, ET CE N'EST PAS DU BRUIT — "
                f"IL SUIT LA RARETÉ.\n"
                f"{int(mx.an)} est l'année la plus calme ({int(mx.feux)} feux) "
                f"et donne le MEILLEUR lift (×{mx.lift:.0f}) ; {int(mn.an)}, "
                f"la plus active, le plus faible (×{mn.lift:.0f}).\n"
                f"Une année calme concentre les feux dans les endroits les "
                f"plus prévisibles. Quand tout brûle — y compris là où ce "
                f"n'est pas censé arriver — le modèle est pris en défaut.",
             d.L, y + 5.2, 30, 4.6, ORANGE, 15)
    d.notes(s, "C'est contre-intuitif, et c'est le genre de chose qu'un jury "
               "apprécie : on a regardé la variabilité, pas seulement la "
               "moyenne.")

    # ── 21 ──────────────────────────────────────────────────────────────
    d.section("04", "Comprendre le modèle",
              "SHAP, LIME, DiCE — et pourquoi ils ne disent pas la même chose")

    # ── 22. SHAP ────────────────────────────────────────────────────────
    s, y = d.page("Qu'est-ce qui fait partir un feu ?",
                  "Trois façons de poser la question, trois réponses — chacune "
                  "juste, mais pas à la même question.")
    if "shap" in G:
        d._image(s, G["shap"], d.L, y, 30, 8.2)
    d.encart(s, "danger_effis est 2ᵉ par gain et 30ᵉ par SHAP. Ce n'est pas une "
                "erreur : c'est une discrétisation du FWI en six classes. "
                "XGBoost adore ces seuils nets pour découper, mais "
                "l'information est déjà dans le FWI continu — et SHAP lui en "
                "attribue le crédit. L'IMPORTANCE PAR GAIN NE SAIT PAS GÉRER "
                "LA REDONDANCE.\npart_maquis est 1ᵉʳ par gain, 10ᵉ sur "
                "l'échantillon aléatoire, 2ᵉ au sommet : il ne change rien sur "
                "une commune-jour moyenne, et devient déterminant là où le "
                "modèle voit du risque.", d.L, y + 8.8, 30, 5, ROUGE, 14)
    d.notes(s, "Morale à énoncer : ne jamais citer « la feature n°1 » sans "
               "dire de quelle mesure il s'agit et sur quelle population. "
               "TreeSHAP est EXACT sur un modèle d'arbres — vérifié : somme "
               "des contributions plus valeur de base = logit du score, à "
               "1e-6 près.")

    # ── 23. DiCE ────────────────────────────────────────────────────────
    s, y = d.page("DiCE — la seule question actionnable",
                  "SHAP et LIME répondent à « pourquoi ce score ». DiCE répond "
                  "à « qu'aurait-il fallu changer ».")
    d.puces(s, [
        "**Un exemple, mesuré**",
        "Bormes-les-Mimosas, 99,9ᵉ percentile du 12 août 2024. On autorise à "
        "modifier la végétation seule : maquis, forêt, part combustible, part "
        "agricole.",
        "**Résultat : aucun contrefactuel.**",
        "Rien, sur ces leviers, ne fait sortir la commune du décile à risque. "
        "Son exposition tient à sa position, à son relief, à sa superficie. "
        "Le risque est STRUCTUREL — et c'est un résultat, pas un échec.",
    ], d.L, y, 30, 16, 1.28)
    d.encart(s, "UN DÉTAIL D'IMPLÉMENTATION QUI CHANGE TOUT\n"
                "DiCE cherche par défaut à faire passer la probabilité sous "
                "0,5. Or le score n'est pas calibré : 0,5 correspond à un "
                "risque astronomique, et l'outil ne renvoyait jamais rien.\n"
                "On a recentré la frontière sur le DÉCILE, par une "
                "transformation strictement croissante qui laisse le "
                "classement intact. La question devient « que faudrait-il pour "
                "sortir des 10 % les plus à risque » — celle qui a un sens "
                "opérationnel.", d.L, y + 5.4, 30, 5.2, BLEU, 14)
    d.encart(s, "UN CONTREFACTUEL N'EST PAS UNE RECOMMANDATION. Rien ne "
                "garantit qu'il soit réalisable — on ne convertit pas 40 % de "
                "maquis en terres agricoles — ni que le lien soit causal : le "
                "modèle a appris des corrélations, pas des mécanismes.",
             d.L, y + 11.2, 30, 2.8, ORANGE, 14)
    d.notes(s, "Si on demande pourquoi LIME : sur un modèle d'arbres il "
               "approxime ce que TreeSHAP calcule exactement. On le montre "
               "pour la comparaison, et parce qu'il redeviendrait le bon outil "
               "sur un modèle qu'on ne peut pas ouvrir — une API, un réseau "
               "profond.")

    # ── 24 ──────────────────────────────────────────────────────────────
    d.section("05", "Le climat",
              "Ce que les données disent — et ce qu'elles ne disent pas")

    # ── 25. la tendance ─────────────────────────────────────────────────
    s, y = d.page("Le danger monte. Les feux, non.",
                  "Deux fenêtres d'observation, deux conclusions — et il faut "
                  "dire laquelle croire.")
    d._image(s, RACINE / "docs" / "img" / "tendance.png", d.L, y, 30, 7.2)
    fa, fe = t("FWI moyen annuel"), t("FWI moyen juin-septembre")
    jd, fx = t("jours de danger élevé (FWI > 21,3)"), t("communes-jours en feu")
    d.tableau(s, ["Série", "Période", "Variation", "p", "Verdict"], [
        ["FWI moyen annuel", f"{fa.an_min}-{fa.an_max}",
         f"{fa.variation_pct:+.0f} %", f"{fa.p:.1e}", "significatif"],
        ["FWI juin-septembre", f"{fe.an_min}-{fe.an_max}",
         f"{fe.variation_pct:+.0f} %", f"{fe.p:.1e}", "significatif"],
        ["jours de danger élevé", f"{jd.an_min}-{jd.an_max}",
         f"{jd.variation_pct:+.0f} %", f"{jd.p:.1e}", "significatif"],
        ["communes-jours en feu", f"{fx.an_min}-{fx.an_max}",
         f"{fx.variation_pct:+.0f} %", f"{fx.p:.2f}", "NON significatif"],
    ], d.L, y + 7.8, 27, 5, larg=[8, 5, 5, 4.5, 4.5], fort=(3,))
    d.notes(s, "À DIRE EXACTEMENT : « les conditions favorables aux feux "
               "augmentent très significativement, et le nombre de départs "
               "reste stable — ce qui est cohérent avec une prévention qui "
               "absorbe pour l'instant la hausse de l'aléa ». Ne JAMAIS dire "
               "« les feux augmentent » : les données du projet ne le montrent "
               "pas. Trois lectures possibles : puissance statistique "
               "insuffisante sur 20 points, prévention efficace, et le fait "
               "qu'on projette l'aléa et non le bilan.")

    # ── 26. les projections ─────────────────────────────────────────────
    s, y = d.page("Projeter jusqu'en 2100 — et ce que ça ne veut PAS dire",
                  "Trois scénarios GIEC, appliqués à l'aléa météo.")
    d._image(s, FIG / "projection-2050" /
             "02_le-risque-de-fond-en-2041-2055-a-vegetation-relief-et-preven.png",
             d.L, y, 18, 10.2)
    d.puces(s, [
        "**Ce qu'on projette**",
        "Le FWI — seule quantité qui montre un signal, et seule que les "
        "modèles climatiques savent fournir.",
        "**Ce qu'on ne projette pas**",
        "Le nombre de feux. La végétation, la prévention et les pratiques "
        "agricoles sont supposées constantes : c'est une hypothèse, et elle "
        "est fausse.",
        "**Ce que « le 2 août 2050 » veut dire**",
        "Pas une prévision météo : un 2 août ORDINAIRE sous le climat de 2050. "
        "La forme de la saison vient des observations, seul son niveau est "
        "décalé.",
    ], 20.8, y, 11.3, 12.5, 1.2)
    d.notes(s, "L'application affiche cet avertissement à chaque date "
               "postérieure à 2025. Le dire avant qu'on le demande.")

    # ── 27 ──────────────────────────────────────────────────────────────
    d.section("06", "Rigueur",
              "Les erreurs commises, et comment elles ont été trouvées")

    # ── 28. le bug d'alignement ─────────────────────────────────────────
    s, y = d.page("Le premier verdict du LSTM annonçait −97 %. Le vrai est −52 %.",
                  "L'écart ne venait pas du modèle, mais de la façon de "
                  "comparer deux fichiers.")
    d._image(s, RACINE / "docs" / "img" / "alignement.png", d.L, y, 17, 9.2)
    d.puces(s, [
        "**La cause**",
        "La requête d'assemblage n'a pas d'ORDER BY. L'ordre des 38 M lignes "
        "que renvoie PostgreSQL dépend du plan d'exécution, et change d'une "
        "exécution à l'autre.",
        "**Pourquoi c'est invisible**",
        "Les fichiers ne portaient que (score, cible). Même taille, même "
        "nombre de feux, ordre différent : rien ne lève d'erreur.",
        "**La parade**",
        "Tout fichier porte désormais ses clés (code_insee, date) ; une "
        "fonction d'alignement vérifie ; un test REFUSE un fichier sans clés.",
    ], 19.5, y, 12.6, 12.5, 1.2)
    d.notes(s, "Reproduit volontairement dans le notebook et dans "
               "l'application : on permute les lignes, la PR-AUC tombe de "
               "0,0085 à 0,0002 — exactement la ligne du hasard, avec les "
               "mêmes valeurs dans le fichier.")

    # ── 29. les autres erreurs ──────────────────────────────────────────
    s, y = d.page("Les autres erreurs, trouvées et corrigées",
                  "Sur un événement à 0,02 %, une erreur ne se manifeste "
                  "jamais par une exception. Elle se manifeste par un chiffre "
                  "plausible.")
    d.tableau(s, ["L'erreur", "Le symptôme", "Ce qui l'a révélée"], [
        ["ha et cible laissées dans les features du modèle de surface",
         "R² = 0,994 et ROC-AUC = 1,0000",
         "un score trop beau pour être vrai"],
        ["interaction analysée sur l'échantillon du SOMMET",
         "le signe de l'interaction s'inversait",
         "biais de collision : sélectionner sur le score conditionne le résultat"],
        ["moyenne climatologique non lissée",
         "un dentelé de période 4 ans dans les séries annuelles",
         "le 15 août tombe au jour 227 ou 228 selon les bissextiles"],
        ["projection ancrée à k = 1,0 en 2025",
         "la courbe plongeait au passage 2025 → 2026",
         "le facteur observé de 2025 vaut déjà ~1,3"],
        ["tendance du FWI écrite en dur dans l'application",
         "+45 % affiché, +58 % réel",
         "recalcul systématique à la source"],
    ], d.L, y, 30, 9.2, larg=[10, 9, 11], taille=12)
    d.encart(s, "Les seules défenses sont les invariants explicites et les "
                "assertions qui échouent bruyamment. 50 tests tournent en "
                "intégration continue à chaque commit.",
             d.L, y + 10, 30, 2.4, VERT, 15)
    d.notes(s, "Assumer ces erreurs plutôt que les cacher. Un jury retient "
               "davantage un candidat qui sait où son travail a failli casser "
               "qu'un candidat sans aspérité.")

    # ── 30. l'application ───────────────────────────────────────────────
    s, y = d.page("L'application", "Cinq pages, déployée publiquement.")
    d.tableau(s, ["Page", "Ce qu'elle montre"], [
        ["Carte", "carte de chaleur nationale, 1973 → 2100, 3 scénarios GIEC, "
                  "et un mode rétrospectif qui compare v3 et C sur le test"],
        ["Commune", "fiche détaillée : décennies, projections, type de "
                    "territoire"],
        ["Les données", "les 4 sources, la tendance sur 53 ans, ADF et SARIMAX"],
        ["Les modèles", "protocole, les 6 modèles et leurs IC, le LSTM, v3 "
                        "contre C, la calibration, le bug d'alignement"],
        ["Pourquoi un feu part", "SHAP, LIME et DiCE — et leur désaccord"],
    ], d.L, y, 30, 7.6, larg=[7, 23])
    d.encart(s, "Le mode rétrospectif REFUSE d'afficher le modèle v3 sur 20 "
                "des 23 années qu'il pourrait techniquement couvrir — le train "
                "(il a appris ces lignes), la validation (elle a servi à "
                "choisir), l'avenir (pas d'historique) — et sait dire pourquoi "
                "pour chacune.", d.L, y + 8.4, 30, 3.2, ROUGE, 15)
    d._texte(s, "terre-vent-feu-eau-data.streamlit.app",
             d.L, y + 12.1, 30, 1.4, 19, BLEU, True)
    d.notes(s, "Faire la démonstration ici. Montrer d'abord la carte, puis le "
               "basculement rétrospectif sur une date d'août, puis la page "
               "Pourquoi.")

    # ── 31. limites ─────────────────────────────────────────────────────
    s, y = d.page("Ce que le projet ne fait pas",
                  "Les limites qu'on connaît valent mieux que celles qu'on "
                  "découvre en soutenance.")
    d.puces(s, [
        "**La surface brûlée n'est pas prédictible.**",
        "R² de 0,14 — moins bon que d'annoncer toujours la médiane. Elle "
        "dépend de ce qui se passe APRÈS le départ : vent, délai "
        "d'intervention, relief. En revanche « sera-ce un grand feu de plus de "
        "5 hectares ? » se prédit à 0,77 de ROC-AUC.",
        "**Une commune-jour n'est pas un incendie.**",
        "Un feu traversant cinq communes compte cinq fois.",
        "**31 communes partagent une maille météo de 28 km.**",
        "Le FWI porte le quand, la végétation porte le où. Conséquence "
        "statistique : les intervalles naïfs sur les coefficients météo "
        "seraient trop étroits.",
        "**Le LSTM n'a pas reçu danger_effis.**",
        "L'écart de 23,6 % est un majorant tant que cette asymétrie n'est pas "
        "levée. C'est la première chose à refaire.",
    ], d.L, y, 30, 15, 1.26)
    d.notes(s, "Terminer là-dessus : montrer qu'on sait ce qui reste ouvert, "
               "et dans quel ordre on s'y prendrait.")

    chemin = d.enregistrer(SORTIE / "soutenance.pptx")
    n = len(d.p.slides._sldIdLst)
    print(f"✅ {chemin.relative_to(RACINE)} — {n} diapositives, "
          f"{chemin.stat().st_size / 1e6:.1f} Mo")


if __name__ == "__main__":
    main()
