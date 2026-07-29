"""Persiste les prédictions sur la validation.

    python -m tvfed.predictions

Le parcours des 38 M lignes coûte ~30 min. On ne le refait pas à chaque
figure : on enregistre (p_rf, p_xgb, y) une fois pour toutes.

38 M × 9 octets ≈ 340 Mo — à comparer aux ~6 Go qu'occuperait la matrice de
features complète. C'est ce fichier qui alimente ensuite les courbes
précision-rappel, la calibration et l'analyse de seuil opérationnel.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import matrices
from .modeles import CIBLE, entrainer
from .paths import PROCESSED


def main() -> None:
    print("Entraînement\n" + "=" * 60)
    train = pd.read_parquet(PROCESSED / "train.parquet")
    modeles, prep, _ = entrainer(train)

    print("\nPrédiction sur la validation intégrale\n" + "=" * 60)
    morceaux, n, t0 = [], 0, time.time()
    for bloc in matrices.parcourir("val"):
        X = prep.transform(bloc)
        morceaux.append(pd.DataFrame({
            "p_rf":  modeles["RandomForest"].predict_proba(X)[:, 1].astype(np.float32),
            "p_xgb": modeles["XGBoost"].predict_proba(X)[:, 1].astype(np.float32),
            "y":     bloc[CIBLE].to_numpy(np.int8),
        }))
        n += len(bloc)
        if n % 5_000_000 < len(bloc):
            print(f"   {n:>12,} lignes   {time.time() - t0:6.0f} s")

    pred = pd.concat(morceaux, ignore_index=True)
    dest = PROCESSED / "predictions_val.parquet"
    pred.to_parquet(dest, index=False, compression="zstd")

    print(f"\n✅ {dest.name} — {len(pred):,} lignes, "
          f"{dest.stat().st_size / 1e6:.0f} Mo, {time.time() - t0:.0f} s")
    print(f"   positifs : {int(pred.y.sum()):,} ({100 * pred.y.mean():.4f} %)")


if __name__ == "__main__":
    main()
