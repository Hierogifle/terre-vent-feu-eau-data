"""CORINE Land Cover — occupation du sol par commune et millésime.

⚠️ Deux pièges de format vérifiés :
  1. Le fichier a QUATRE lignes d'en-tête (libellés, types, unités, codes
     machine). On prend la 4ᵉ comme noms de colonnes et on garde la 1ʳᵉ
     comme dictionnaire de libellés.
  2. Les millésimes 2000, 2006 et 2012 existent en DEUX versions, normale et
     « révisée ». Elles diffèrent pour 60 à 78 % des communes : ce ne sont
     pas des doublons anodins. On garde la révisée.

⚠️ La clé est un cog_commune_2010 : 1 021 codes ont disparu depuis.
"""
from __future__ import annotations

import csv

import pandas as pd

from ..paths import CORINE_CSV

# Les 8 postes qui décrivent de la végétation qui brûle
COMBUSTIBLE = {
    "CLC_311": "forêts de feuillus",
    "CLC_312": "forêts de conifères",
    "CLC_313": "forêts mélangées",
    "CLC_321": "pelouses et pâturages naturels",
    "CLC_322": "landes et broussailles",
    "CLC_323": "végétation sclérophylle",      # maquis et garrigue ⭐
    "CLC_324": "végétation arbustive en mutation",
    "CLC_333": "végétation clairsemée",
}


def charger(chemin=None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Retourne (table LONGUE, libellés). Une ligne = commune × millésime × poste."""
    chemin = chemin or CORINE_CSV
    with open(chemin, encoding="utf-8") as fh:
        r = csv.reader(fh, delimiter=";")
        libelles, _, _, codes = [next(r) for _ in range(4)]

    label = {
        c: (l.split(" - ", 1)[-1].replace(" (en ha)", "") if " - " in l else l)
        for c, l in zip(codes, libelles)
    }

    df = pd.read_csv(
        chemin, sep=";", skiprows=[1, 2, 3], header=0, names=codes,
        dtype={"NUM_COM": str}, low_memory=False,
    )
    postes = [c for c in df.columns if c.startswith("CLC_")]
    df[postes] = df[postes].apply(pd.to_numeric, errors="coerce")
    df["ANNEE"] = df.ANNEE.astype(int)

    # version révisée prioritaire quand elle existe
    df["_rev"] = df.base.str.contains("révisée")
    df = (
        df.sort_values(["NUM_COM", "ANNEE", "_rev"])
        .drop_duplicates(["NUM_COM", "ANNEE"], keep="last")
        .drop(columns="_rev")
    )

    long = df.melt(
        id_vars=["NUM_COM", "ANNEE", "base"],
        value_vars=postes,
        var_name="poste",
        value_name="surface_ha",
    ).rename(columns={"NUM_COM": "code_insee", "ANNEE": "millesime"})

    # un poste absent vaut 0 : inutile de stocker 44 lignes par commune
    long = long[long.surface_ha > 0].reset_index(drop=True)
    return long, label
