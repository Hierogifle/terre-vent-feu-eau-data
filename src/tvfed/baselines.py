"""Étape 5 — les baselines. Le score de référence à battre.

    python -m tvfed.baselines

Trois prédicteurs, du plus bête au plus sérieux :

  0. HASARD          — tout le monde au même risque. C'est le « niveau zéro »
                       de la performance : le plancher absolu.
  1. HISTORIQUE      — le risque d'une commune = sa fréquence passée.
                       Purement SPATIAL, aucune météo.
  2. DANGER EFFIS    — le risque d'un jour = celui de sa classe de danger.
                       Purement MÉTÉO, aucune connaissance du territoire.

⚠️ Les trois sont appris sur le TRAIN SEUL (≤ 2019) et évalués sur la
VALIDATION (2020-2022), intégrale. Apprendre sur la validation reviendrait à
réviser sur le corrigé.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from . import db, metriques
from .paths import PROCESSED, RACINE

# Le lissage bayésien : une commune sans feu observé n'a pas un risque nul,
# elle a un risque mal mesuré. On tire son estimation vers la moyenne du
# groupe, d'autant plus fort qu'on a peu d'observations.
# (C'est l'effet aléatoire d'un modèle mixte, appliqué à l'espace.)
K_LISSAGE = 500  # en jours-communes : ~0,2 an d'observation « empruntée »


def _cfg():
    return yaml.safe_load((RACINE / "config" / "perimetre.yaml").read_text(encoding="utf-8"))


def calculer(conn) -> tuple[pd.DataFrame, dict]:
    cfg = _cfg()
    tr_deb, tr_fin = cfg["split"]["train"]
    va_deb, va_fin = cfg["split"]["val"]

    # ── taux de base, appris sur le train ──
    base, = pd.read_sql(
        f"SELECT avg(y::int) t FROM grille WHERE date BETWEEN '{tr_deb}' AND '{tr_fin}'",
        conn).t.values
    print(f"Taux de base appris sur le train : {100 * base:.5f} %\n")

    lignes = []

    # ── BASELINE 0 : le hasard ────────────────────────────────────────
    # Un seul score pour tout le monde. Sert de plancher : c'est la PR-AUC
    # qu'on obtient sans aucune information.
    val0 = pd.read_sql(
        f"""SELECT count(*) n, count(*) FILTER (WHERE y) pos
            FROM grille WHERE date BETWEEN '{va_deb}' AND '{va_fin}'""", conn)
    taux_val = val0.pos[0] / val0.n[0]
    lignes.append(metriques.rapport("0 · hasard", [base], val0.n, val0.pos, taux_val))

    # ── BASELINE 1 : fréquence historique par commune (spatial pur) ───
    # ⚠️ le taux est appris sur le TRAIN, puis APPLIQUÉ à la validation.
    hist = pd.read_sql(f"""
        WITH tr AS (
            SELECT code_insee, count(*) n, count(*) FILTER (WHERE y) pos
            FROM grille WHERE date BETWEEN '{tr_deb}' AND '{tr_fin}'
            GROUP BY 1
        ),
        va AS (
            SELECT code_insee, count(*) n, count(*) FILTER (WHERE y) pos
            FROM grille WHERE date BETWEEN '{va_deb}' AND '{va_fin}'
            GROUP BY 1
        )
        SELECT tr.pos AS tr_pos, tr.n AS tr_n, va.n, va.pos
        FROM tr JOIN va USING (code_insee)""", conn)
    score_hist = (hist.tr_pos + K_LISSAGE * base) / (hist.tr_n + K_LISSAGE)
    lignes.append(metriques.rapport("1 · historique commune (spatial)",
                                    score_hist, hist.n, hist.pos, taux_val))

    # ── BASELINE 2 : danger EFFIS seul (météo pure) ───────────────────
    # Le taux de feu de chaque classe est appris sur le train, puis appliqué.
    effis = pd.read_sql(f"""
        WITH cl AS (
            SELECT g.date, g.y,
                   CASE WHEN m.fwi <  5.2 THEN 1 WHEN m.fwi < 11.2 THEN 2
                        WHEN m.fwi < 21.3 THEN 3 WHEN m.fwi < 38.0 THEN 4
                        WHEN m.fwi < 50.0 THEN 5 ELSE 6 END AS classe
            FROM grille g
            JOIN ref_commune c ON c.code_insee = g.code_insee
            JOIN fait_meteo  m ON m.cell_id = c.cell_id AND m.date = g.date
            WHERE g.date BETWEEN '{tr_deb}' AND '{va_fin}'
        )
        SELECT classe,
               count(*) FILTER (WHERE date <= '{tr_fin}')                  AS tr_n,
               count(*) FILTER (WHERE date <= '{tr_fin}' AND y)            AS tr_pos,
               count(*) FILTER (WHERE date >= '{va_deb}')                  AS n,
               count(*) FILTER (WHERE date >= '{va_deb}' AND y)            AS pos
        FROM cl GROUP BY 1 ORDER BY 1""", conn)
    score_effis = effis.tr_pos / effis.tr_n          # taux appris sur le train
    lignes.append(metriques.rapport("2 · danger EFFIS (météo)",
                                    score_effis, effis.n, effis.pos, taux_val))

    # ── BASELINE 3 : les deux combinées, en supposant l'indépendance ──
    # p ≈ p_commune × p_classe / p_base. Grossier, mais il montre ce qu'on
    # gagne juste en croisant les deux sources sans modèle.
    croise = pd.read_sql(f"""
        WITH tr_com AS (
            SELECT code_insee, count(*) n, count(*) FILTER (WHERE y) pos
            FROM grille WHERE date BETWEEN '{tr_deb}' AND '{tr_fin}' GROUP BY 1
        ),
        va AS (
            SELECT g.code_insee, g.y,
                   CASE WHEN m.fwi <  5.2 THEN 1 WHEN m.fwi < 11.2 THEN 2
                        WHEN m.fwi < 21.3 THEN 3 WHEN m.fwi < 38.0 THEN 4
                        WHEN m.fwi < 50.0 THEN 5 ELSE 6 END AS classe
            FROM grille g
            JOIN ref_commune c ON c.code_insee = g.code_insee
            JOIN fait_meteo  m ON m.cell_id = c.cell_id AND m.date = g.date
            WHERE g.date BETWEEN '{va_deb}' AND '{va_fin}'
        )
        SELECT t.pos AS tr_pos, t.n AS tr_n, v.classe,
               count(*) n, count(*) FILTER (WHERE v.y) pos
        FROM va v JOIN tr_com t USING (code_insee)
        GROUP BY 1, 2, 3""", conn)
    p_com = (croise.tr_pos + K_LISSAGE * base) / (croise.tr_n + K_LISSAGE)
    p_cls = croise.classe.map(dict(zip(effis.classe, score_effis)))
    score_x = np.clip(p_com * p_cls / base, 0, 1)
    lignes.append(metriques.rapport("3 · historique × EFFIS",
                                    score_x, croise.n, croise.pos, taux_val))

    return pd.DataFrame(lignes), {
        "taux_base_train": base, "taux_val": taux_val,
        "effis": effis.assign(score=score_effis),
    }


def main() -> None:
    with db.connexion() as conn:
        res, ctx = calculer(conn)

    print("═" * 74)
    print(f"{'prédicteur':36s} {'PR-AUC':>10s} {'lift':>8s} {'Brier':>12s}")
    print("─" * 74)
    for _, r in res.iterrows():
        print(f"{r.predicteur:36s} {r.pr_auc:10.4f} {r.lift:7.1f}× {r.brier:12.2e}")
    print("═" * 74)
    print(f"\nTaux de positifs de la validation : {100 * ctx['taux_val']:.5f} %")
    print("Le « lift » indique combien de fois le prédicteur fait mieux que le hasard.")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    dest = PROCESSED / "baselines.csv"
    res.to_csv(dest, index=False)
    print(f"\n✅ scores écrits dans {dest.name} — c'est le seuil que le modèle doit battre")


if __name__ == "__main__":
    main()
