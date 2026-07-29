"""Modèle v2 — XGBoost avec les hyperparamètres retenus par Optuna.

    python -m tvfed.modele_v2

Réentraîne sur le train COMPLET (2006-2019) avec les paramètres trouvés, puis
évalue sur la validation intégrale — la seule mesure comparable au v1 et aux
baselines.

⚠️ Optuna a cherché sur un découpage INTERNE au train (ajustement 2006-2017,
évaluation 2018-2019). La validation 2020-2022 n'a jamais servi à choisir quoi
que ce soit : elle reste un juge indépendant.

XGBoost seul, sans RandomForest : sa prédiction sur 38 M lignes coûte ~25 min
à elle seule, pour un modèle qui a déjà été mesuré au v1.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import matrices
from .modeles import CIBLE, Preparation
from .paths import PROCESSED


def main() -> None:
    fichier = PROCESSED / "best_params_xgb.json"
    if not fichier.exists():
        raise FileNotFoundError(
            "best_params_xgb.json absent — lancer d'abord :\n"
            "    python -m tvfed.optimisation --modele xgb --essais 60"
        )
    params = json.loads(fichier.read_text(encoding="utf-8"))

    # Garde-fou : XGBoost refuse un flottant là où il attend un entier
    # (« TypeError: 'float' object cannot be interpreted as an integer »).
    # Un fichier écrit par une version antérieure peut contenir 900.0.
    for k in ("n_estimators", "max_depth", "min_child_weight"):
        if k in params:
            params[k] = int(params[k])

    print("Hyperparamètres retenus par Optuna :")
    for k, v in params.items():
        print(f"   {k:20s} {v}")

    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    train = pd.read_parquet(PROCESSED / "train.parquet")
    prep = Preparation().fit(train)
    X, y = prep.transform(train), train[CIBLE].to_numpy()

    t0 = time.time()
    m = XGBClassifier(**params, tree_method="hist", eval_metric="aucpr",
                      n_jobs=-1, random_state=42).fit(X, y)
    print(f"\nEntraîné sur {len(y):,} lignes en {time.time() - t0:.0f} s")

    print("\nÉvaluation sur la validation intégrale…")
    scores, cibles, n, t0 = [], [], 0, time.time()
    for bloc in matrices.parcourir("val"):
        scores.append(m.predict_proba(prep.transform(bloc))[:, 1].astype(np.float32))
        cibles.append(bloc[CIBLE].to_numpy(np.int8))
        n += len(bloc)
        if n % 10_000_000 < len(bloc):
            print(f"   {n:>12,} lignes   {time.time() - t0:5.0f} s")

    p = np.concatenate(scores)
    yv = np.concatenate(cibles)
    ap = average_precision_score(yv, p)

    pd.DataFrame({"p_xgb_v2": p, "y": yv}).to_parquet(
        PROCESSED / "predictions_val_v2.parquet", index=False, compression="zstd")

    base = pd.read_csv(PROCESSED / "baselines.csv")
    v1 = pd.read_csv(PROCESSED / "modeles_v1.csv").set_index("modele")
    ref_base, ref_v1 = base.pr_auc.max(), v1.pr_auc["XGBoost"]

    print("\n" + "═" * 60)
    print(f"{'':28s} {'PR-AUC':>9s} {'lift':>8s}")
    print(f"{'meilleure baseline':28s} {ref_base:9.4f} {ref_base / yv.mean():7.1f}×")
    print(f"{'XGBoost v1 (réglé à la main)':28s} {ref_v1:9.4f} {ref_v1 / yv.mean():7.1f}×")
    print(f"{'XGBoost v2 (Optuna)':28s} {ap:9.4f} {ap / yv.mean():7.1f}×")
    print("═" * 60)
    print(f"\ngain d'Optuna sur le v1 : {100 * (ap / ref_v1 - 1):+.1f} %")
    print(f"gain total sur la baseline : ×{ap / ref_base:.2f}")

    pd.DataFrame([{"modele": "XGBoost v2 (Optuna)", "pr_auc": ap,
                   "lift": ap / yv.mean()}]).to_csv(
        PROCESSED / "modeles_v2.csv", index=False)
    print(f"\n✅ modeles_v2.csv · predictions_val_v2.parquet")


if __name__ == "__main__":
    main()
