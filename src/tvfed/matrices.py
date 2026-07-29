"""Étape 4 — assemblage des matrices d'entraînement.

    python -m tvfed.matrices

Stratégie de volumétrie, imposée par une contrainte du plan :

  train  366 k lignes après downsampling 1:10  → Parquet, relu à chaque
                                                  itération d'entraînement
  val     38,1 M lignes, NON échantillonnées   → JAMAIS matérialisé
  test    38,1 M lignes, NON échantillonnées   → JAMAIS matérialisé

La validation doit rester intégrale (c'est elle qui corrige le décalage de
prior introduit par le downsampling, facteur ×500 environ). Mais 38 M lignes
× 50 features = ~6 Go : impossible à charger d'un bloc. On la parcourt donc
par curseur serveur, en ne gardant que (probabilité, y) — 300 Mo au lieu de 6 Go.
"""
from __future__ import annotations

import time
from typing import Iterator

import pandas as pd
import psycopg

from . import db
from .paths import PROCESSED, RACINE

TAILLE_BLOC = 500_000

# Bornes et taux d'échantillonnage par partition.
# `frac` = fraction des NÉGATIFS conservée. À 1.0, la partition est intégrale.
# Le train vise un ratio 1:10 sur un taux de positifs de ~0,019 %.
SPLITS = {
    "train": {"debut": "2006-01-01", "fin": "2019-12-31", "frac": 0.00189},
    "val":   {"debut": "2020-01-01", "fin": "2022-12-31", "frac": 1.0},
    "test":  {"debut": "2023-01-01", "fin": "2025-12-31", "frac": 1.0},
}


def _requete(split: str) -> str:
    p = SPLITS[split]
    return (
        (RACINE / "sql" / "50_matrice.sql")
        .read_text(encoding="utf-8")
        .replace(":date_min", f"'{p['debut']}'::date")
        .replace(":date_max", f"'{p['fin']}'::date")
        .replace(":split_nom", f"'{split}'")
        .replace(":frac", str(p["frac"]))
    )


def charger(split: str) -> pd.DataFrame:
    """Charge une partition entière en mémoire. À réserver au train."""
    with db.connexion() as conn:
        return pd.read_sql(_requete(split), conn)


def parcourir(split: str, taille: int = TAILLE_BLOC) -> Iterator[pd.DataFrame]:
    """Parcourt une partition par blocs, sans jamais tout charger.

    Curseur nommé = curseur côté serveur : PostgreSQL ne matérialise pas
    l'ensemble du résultat, il le produit à la demande.
    """
    with psycopg.connect(db.DSN) as conn:
        with conn.cursor(name=f"cur_{split}") as cur:
            cur.itersize = taille
            cur.execute(_requete(split))
            colonnes = None
            while lignes := cur.fetchmany(taille):
                if colonnes is None:
                    colonnes = [d.name for d in cur.description]
                yield pd.DataFrame(lignes, columns=colonnes)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print("Assemblage des matrices\n" + "=" * 62)

    # ── train : matérialisé, il sera relu des dizaines de fois ──
    t0 = time.time()
    train = charger("train")
    dest = PROCESSED / "train.parquet"
    train.to_parquet(dest, index=False)
    print(f"   train  {len(train):>9,} lignes  {train.y.sum():>6,} positifs  "
          f"({100 * train.y.mean():.2f} %)  {dest.stat().st_size / 1e6:.0f} Mo  "
          f"{time.time() - t0:.0f} s")
    print(f"          {train.shape[1]} colonnes")

    # ── val / test : jamais matérialisés ──
    # On vérifie seulement que la requête produit bien des lignes, et on
    # compte via la grille (bien moins cher qu'assembler 38 M lignes de
    # features juste pour les dénombrer).
    with db.connexion() as conn, conn.cursor() as cur:
        for split in ("val", "test"):
            p = SPLITS[split]
            t0 = time.time()
            cur.execute(
                "SELECT count(*), count(*) FILTER (WHERE y) FROM grille "
                "WHERE date BETWEEN %s AND %s",
                (p["debut"], p["fin"]),
            )
            n, pos = cur.fetchone()
            premier = next(parcourir(split, taille=50_000))
            print(f"   {split:5s}  {n:>9,} lignes  {pos:>6,} positifs  "
                  f"({100 * pos / n:.4f} %)  intégral, non matérialisé  "
                  f"[bloc test {len(premier):,} × {premier.shape[1]} OK]  "
                  f"{time.time() - t0:.0f} s")

    print("=" * 62)
    print(f"✅ train écrit dans {dest}")
    print("   val et test restent en base : ils se parcourent avec "
          "matrices.parcourir('val')")


if __name__ == "__main__":
    main()
