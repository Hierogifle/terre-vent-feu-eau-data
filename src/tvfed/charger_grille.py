"""Étape 3 du pipeline — construction de la grille commune × jour.

    python -m tvfed.charger_grille

~253 M lignes. Construite année par année : en une transaction unique, le WAL
PostgreSQL exploserait et la moindre erreur ferait tout recommencer.
Chaque année est indépendante et rejouable.
"""
from __future__ import annotations

import time

import yaml

from . import db
from .paths import RACINE


def main() -> None:
    cfg = yaml.safe_load((RACINE / "config" / "perimetre.yaml").read_text(encoding="utf-8"))
    an_min = int(cfg["periode"]["debut"][:4])
    an_max = int(cfg["periode"]["fin"][:4])
    gabarit = (RACINE / "sql" / "30_grille.sql").read_text(encoding="utf-8")

    conn = db.connexion()
    print(f"Construction de la grille {an_min}-{an_max}\n" + "=" * 62)
    t_total = time.time()
    total = pos_total = 0

    for annee in range(an_min, an_max + 1):
        t0 = time.time()
        with conn.cursor() as cur:
            # la partition annuelle est vidée d'abord : l'étape est rejouable
            cur.execute(f"TRUNCATE grille_{annee}")
            cur.execute(gabarit.replace(":annee", str(annee)))
            cur.execute(
                f"SELECT count(*), count(*) FILTER (WHERE y) FROM grille_{annee}"
            )
            n, pos = cur.fetchone()
        conn.commit()
        total += n
        pos_total += pos
        print(f"   {annee}  {n:>10,} lignes  {pos:>6,} positifs "
              f"({100 * pos / n:.4f} %)   {time.time() - t0:5.1f} s")

    print("=" * 62)
    print(f"   TOTAL {total:>10,} lignes  {pos_total:>6,} positifs "
          f"({100 * pos_total / total:.4f} %)   {time.time() - t_total:.0f} s")

    attendu = cfg["attendu"]
    ecart = abs(pos_total - attendu["positifs"]) / attendu["positifs"]
    print(f"\n   attendu : {attendu['positifs']:,} positifs — écart {100 * ecart:.1f} %")
    if ecart > 0.05:
        print("   ⚠️  écart > 5 % : vérifier le remappage COG et le périmètre")

    conn.close()
    print("\n✅ grille construite")


if __name__ == "__main__":
    main()
