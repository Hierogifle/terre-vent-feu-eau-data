"""Enregistrement automatique des figures.

Une ligne à placer en tête d'un notebook :

    from tvfed.figures import activer
    activer("data-all")

À partir de là, **chaque `plt.show()` enregistre aussi la figure** dans
`figures/data-all/`, sans rien changer au reste du code. Le nom est déduit du
titre de la figure, préfixé par son rang.

    figures/data-all/03_vegetation-le-brut-mesure-ce-qui-est-abondant.png

Sans ce helper, `plt.show()` affiche puis détruit la figure : elle ne vit que
dans les sorties du notebook, et disparaît au moindre « Clear outputs ».
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt

from .paths import RACINE

DOSSIER = RACINE / "figures"
_show_origine = None


def _slug(texte: str, longueur: int = 60) -> str:
    """« Végétation — le brut mesure… » → « vegetation-le-brut-mesure »"""
    t = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-zA-Z0-9]+", "-", t).strip("-").lower()
    return t[:longueur].rstrip("-")


def _titre(fig) -> str:
    """Cherche un titre : d'abord le suptitle, sinon le titre du premier axe.

    ⚠️ `ax.get_title()` ne lit que le slot CENTRAL. Un titre posé avec
    `set_title(..., loc="left")` — ce que fait toute la charte du projet —
    y est invisible et renvoie une chaîne vide. Il faut interroger les trois
    emplacements.
    """
    sup = getattr(fig, "_suptitle", None)
    if sup is not None and sup.get_text().strip():
        return sup.get_text()
    for ax in fig.axes:
        for loc in ("center", "left", "right"):
            t = ax.get_title(loc=loc)
            if t.strip():
                return t
    return "figure"


def activer(notebook: str, dpi: int = 150, fond: str = "#fcfcfb") -> Path:
    """Redirige `plt.show()` pour qu'il enregistre avant d'afficher."""
    global _show_origine
    if _show_origine is None:
        _show_origine = plt.show

    dest = DOSSIER / notebook
    dest.mkdir(parents=True, exist_ok=True)
    rang = [0]

    def show(*args, **kwargs):
        fig = plt.gcf()
        if fig.axes:                       # ignore les figures vides
            rang[0] += 1
            nom = f"{rang[0]:02d}_{_slug(_titre(fig))}.png"
            # bbox_inches='tight' : sans lui, les suptitles placés en y > 1
            # et les légendes hors cadre sont rognés
            fig.savefig(dest / nom, dpi=dpi, bbox_inches="tight", facecolor=fond)
        return _show_origine(*args, **kwargs)

    plt.show = show
    print(f"📁 figures enregistrées dans  {dest.relative_to(RACINE)}/")
    return dest


def desactiver() -> None:
    """Rend à `plt.show()` son comportement d'origine."""
    global _show_origine
    if _show_origine is not None:
        plt.show = _show_origine
        _show_origine = None
