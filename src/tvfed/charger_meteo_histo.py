"""Étape 22 — la météo d'avant le périmètre, pour montrer l'évolution.

    python -m tvfed.charger_meteo_histo

────────────────────────────────────────────────────────────────────────────
POURQUOI CHARGER 1973-2005 ALORS QUE LE MODÈLE S'ARRÊTE À 2006
────────────────────────────────────────────────────────────────────────────
Le périmètre de modélisation commence en 2006, et pour une bonne raison : huit
départements n'ont AUCUN feu enregistré avant cette date — la collecte BDIFF
n'était pas homogène. Entraîner sur ces années fabriquerait des zéros qui ne
sont pas des absences de feu mais des absences de saisie.

Mais la MÉTÉO, elle, ne souffre d'aucun de ces défauts : elle vient d'une
réanalyse, calculée uniformément sur toute l'Europe depuis 1970. On peut donc
la charger sans réserve, et elle sert à une chose que 2006-2025 ne permet pas :

    montrer l'ÉVOLUTION du danger météo, décennie par décennie.

Vingt ans ne font pas une tendance climatique — on l'a mesuré, la pente du FWI
sur 2006-2025 n'est pas significative (p = 0,13), la variabilité d'une année à
l'autre écrase le signal. Sur cinquante-trois ans, c'est une autre histoire.

⚠️ Ces années n'entrent NI dans la grille NI dans l'entraînement. Elles ne
servent qu'à l'affichage et au contexte. Le modèle reste celui du gel.
"""
from __future__ import annotations

import time

from . import db
from .io import cems

AN_MAX_HISTO = 2005     # 2006+ est déjà chargé par charger_faits


def main() -> None:
    conn = db.connexion()
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT cell_id FROM ref_commune")
        cells = {r[0] for r in cur.fetchall()}
        cur.execute("SELECT min(date), max(date), count(*) FROM fait_meteo")
        av = cur.fetchone()
    print(f"déjà en base : {av[0]} → {av[1]}, {av[2]:,} lignes")

    annees = [a for a in cems.annees_disponibles() if a <= AN_MAX_HISTO]
    if not annees:
        print("rien à charger")
        return
    print(f"à charger    : {annees[0]} → {annees[-1]}, "
          f"{len(annees)} années, {len(cells)} mailles\n")

    t0, total = time.time(), 0
    for an in annees:
        # ⚠️ ON NE CHARGE PAS DEUX FOIS. La clé primaire (cell_id, date)
        # rejetterait les doublons, mais un COPY qui échoue à mi-parcours
        # laisse la transaction dans un état sale. On vérifie avant.
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM fait_meteo "
                        "WHERE date = %s LIMIT 1", (f"{an}-07-01",))
            if cur.fetchone():
                print(f"   {an} déjà présente — ignorée")
                continue
        m = cems.charger_annee(an, cell_ids=cells)
        n = db.copier(m, "fait_meteo", conn)
        total += n
        if an % 5 == 0 or an == annees[-1]:
            print(f"   {an}   {total:>9,} lignes   {time.time() - t0:5.0f} s")

    with conn.cursor() as cur:
        cur.execute("SELECT min(date), max(date), count(*) FROM fait_meteo")
        ap = cur.fetchone()
    conn.commit()
    conn.close()

    print(f"\n{'═' * 62}")
    print(f"avant : {av[0]} → {av[1]}   {av[2]:>11,} lignes")
    print(f"après : {ap[0]} → {ap[1]}   {ap[2]:>11,} lignes  (+{total:,})")
    print("═" * 62)
    print("\n⚠️ La grille et le modèle ne changent PAS : ils sont bornés à")
    print("   2006-2025 par config/perimetre.yaml et par les requêtes SQL.")


if __name__ == "__main__":
    main()
