"""Étape 14 — l'évaluation finale sur le test. UNE SEULE FOIS.

    python -m tvfed.evaluation_test

────────────────────────────────────────────────────────────────────────────
LA RÈGLE
────────────────────────────────────────────────────────────────────────────
Les 38 M lignes de 2023-2025 n'ont servi à RIEN jusqu'ici : ni à choisir les
features, ni les hyperparamètres, ni le nombre de clusters, ni la méthode de
calibration, ni le nombre d'arbres. C'est ce qui fait d'elles un juge.

Ce statut se détruit à la première retouche. Si le résultat déçoit et qu'on
change quoi que ce soit ensuite, le chiffre suivant ne mesurera plus la
performance mais la capacité à s'ajuster au test — et il n'y a pas de
troisième jeu pour s'en rendre compte.

La configuration gelée et la prédiction faite AVANT de regarder sont dans
`gel_avant_test.json`. Ce module les relit et confronte.

────────────────────────────────────────────────────────────────────────────
CE QUI EST MESURÉ
────────────────────────────────────────────────────────────────────────────
1. PR-AUC et lift — la performance de classement
2. Précision et rappel à budget de surveillance fixé — la lecture opérationnelle
3. Calibration — le pourcentage affiché est-il juste, avec le calibrateur
   ajusté sur la validation et jamais retouché
4. Par année — 2024 fut la plus calme des vingt dernières (FWI 3,38)
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from .paths import PROCESSED

BUDGETS = (0.001, 0.005, 0.01, 0.05, 0.10)
MODELE = "xgb_v3"


def main() -> None:
    from sklearn.metrics import average_precision_score, brier_score_loss

    f = PROCESSED / "scores_test.parquet"
    if not f.exists():
        raise FileNotFoundError(
            "scores_test.parquet absent — lancer d'abord :\n"
            "    python -m tvfed.scores --split test")

    gel = json.loads((PROCESSED / "gel_avant_test.json").read_text(encoding="utf-8"))
    D = pd.read_parquet(f)
    an = pd.to_datetime(D.date).dt.year
    y = D.y.to_numpy()
    p = D[MODELE].to_numpy(np.float64)
    base = y.mean()

    print("═" * 74)
    print("ÉVALUATION FINALE — TEST 2023-2025")
    print("═" * 74)
    print(f"gelé le {gel['date_gel']} · {gel['modele_retenu']}")
    print(f"prédiction consignée AVANT : PR-AUC "
          f"{gel['prediction_avant_test']['pr_auc_attendu']}, "
          f"lift {gel['prediction_avant_test']['lift_attendu']}")
    print(f"\n{len(D):,} lignes, {y.sum():,} feux ({base:.4%})")

    ap = average_precision_score(y, p)
    ref = gel["reference_validation"]
    print(f"\n{'─' * 74}")
    print(f"{'':22s} {'PR-AUC':>9s} {'lift':>9s} {'taux positifs':>15s}")
    print(f"{'validation 2020-2022':22s} {ref['pr_auc']:9.4f} {ref['lift']:8.1f}× "
          f"{ref['taux_positifs']:15.4%}")
    print(f"{'TEST 2023-2025':22s} {ap:9.4f} {ap / base:8.1f}× {base:15.4%}")
    print(f"{'hasard':22s} {base:9.4f} {1.0:8.1f}×")

    # ── lecture opérationnelle ──────────────────────────────────────────
    o = np.argsort(-p)
    yo = y[o]
    print(f"\n{'─' * 74}")
    print("LECTURE OPÉRATIONNELLE — que se passe-t-il si on surveille les N % "
          "les mieux notés")
    print(f"{'budget':>8s} {'communes-jours':>16s} {'précision':>11s} "
          f"{'rappel':>9s} {'feux couverts':>15s}")
    ops = []
    for b in BUDGETS:
        k = int(len(yo) * b)
        pris = int(yo[:k].sum())
        ops.append({"budget": b, "lignes": k, "precision": pris / k,
                    "rappel": pris / y.sum(), "feux": pris})
        print(f"{b:>7.1%} {k:>16,} {pris / k:>10.2%} {pris / y.sum():>8.1%} "
              f"{pris:>10,} / {y.sum():,}")

    # ── calibration, avec le calibrateur JAMAIS retouché ────────────────
    with open(PROCESSED / "calibrateur_platt.pkl", "rb") as fh:
        cal = pickle.load(fh)
    pc = cal.predict(p)
    print(f"\n{'─' * 74}")
    print("CALIBRATION — Platt ajusté sur la validation, appliqué tel quel")
    print(f"{'':12s} {'Brier':>12s} {'p moyen':>11s} {'réel':>11s} {'biais':>9s}")
    print(f"{'brut':12s} {brier_score_loss(y, p):12.4e} {p.mean():11.6f} "
          f"{base:11.6f} {p.mean() / base:8.1f}×")
    print(f"{'Platt':12s} {brier_score_loss(y, pc):12.4e} {pc.mean():11.6f} "
          f"{base:11.6f} {pc.mean() / base:8.2f}×")
    print(f"PR-AUC après calibration : {average_precision_score(y, pc):.4f} "
          f"(inchangée, Platt est monotone)")

    # ── par année ───────────────────────────────────────────────────────
    print(f"\n{'─' * 74}")
    print("PAR ANNÉE")
    print(f"{'année':>6s} {'lignes':>12s} {'feux':>7s} {'taux':>9s} "
          f"{'PR-AUC':>9s} {'lift':>8s}")
    lignes_an = []
    for a, g in D.groupby(an):
        api = average_precision_score(g.y, g[MODELE])
        lignes_an.append({"an": int(a), "lignes": len(g), "feux": int(g.y.sum()),
                          "taux": g.y.mean(), "pr_auc": api,
                          "lift": api / g.y.mean()})
        print(f"{a:>6} {len(g):>12,} {g.y.sum():>7,} {g.y.mean():>8.4%} "
              f"{api:9.4f} {api / g.y.mean():7.1f}×")

    # ── verdict ─────────────────────────────────────────────────────────
    lo, hi = [float(x.replace(",", ".")) for x in
              gel["prediction_avant_test"]["pr_auc_attendu"].split(" - ")]
    dans = lo <= ap <= hi
    print(f"\n{'═' * 74}")
    print(f"La prédiction annonçait {lo}-{hi}. Mesuré : {ap:.4f} — "
          f"{'DANS la fourchette' if dans else 'HORS fourchette'}.")
    print("═" * 74)

    pd.DataFrame([{"modele": MODELE, "pr_auc": ap, "lift": ap / base,
                   "taux_positifs": base, "n_lignes": len(D),
                   "n_feux": int(y.sum())}]).to_csv(
        PROCESSED / "resultat_test.csv", index=False)
    pd.DataFrame(ops).to_csv(PROCESSED / "operationnel_test.csv", index=False)
    pd.DataFrame(lignes_an).to_csv(PROCESSED / "test_par_annee.csv", index=False)
    print("\n✅ resultat_test.csv · operationnel_test.csv · test_par_annee.csv")


if __name__ == "__main__":
    main()
