"""Courbe d'apprentissage — la performance itération par itération.

    python -m tvfed.apprentissage

XGBoost construit ses arbres EN SÉQUENCE : chacun corrige les erreurs des
précédents. On peut donc mesurer la PR-AUC après 1 arbre, après 2, … après 400.

C'est le diagnostic qui dit si le nombre d'itérations est bien choisi :

    les deux courbes montent encore    → trop peu d'arbres, on s'arrête tôt
    l'ajustement monte, l'éval stagne  → nombre correct
    l'ajustement monte, l'éval BAISSE  → sur-apprentissage, trop d'arbres

⚠️ Évalué sur le découpage INTERNE au train (2006-2017 / 2018-2019), jamais
sur la validation 2020-2022 — elle est réservée à la calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .optimisation import _decouper
from .paths import PROCESSED


def courbe(params: dict | None = None, n_arbres: int = 600) -> pd.DataFrame:
    from xgboost import XGBClassifier

    train = pd.read_parquet(PROCESSED / "train.parquet")
    Xa, ya, Xb, yb, _ = _decouper(train)
    print(f"  ajustement  ≤2017     : {len(ya):>7,} lignes, {ya.sum():>6,} positifs")
    print(f"  évaluation  2018-2019 : {len(yb):>7,} lignes, {yb.sum():>6,} positifs\n")

    base = dict(n_estimators=n_arbres, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8)
    base.update(params or {})

    m = XGBClassifier(**base, tree_method="hist", eval_metric="aucpr",
                      n_jobs=-1, random_state=42)
    # eval_set : c'est LUI qui déclenche l'enregistrement par itération
    m.fit(Xa, ya, eval_set=[(Xa, ya), (Xb, yb)], verbose=False)

    r = m.evals_result_
    return pd.DataFrame({
        "iteration": np.arange(1, n_arbres + 1),
        "ajustement": r["validation_0"]["aucpr"],
        "evaluation": r["validation_1"]["aucpr"],
    })


def main() -> None:
    print("Courbe d'apprentissage XGBoost\n" + "=" * 60)
    c = courbe()
    dest = PROCESSED / "courbe_apprentissage.csv"
    c.to_csv(dest, index=False)

    best = int(c.evaluation.idxmax()) + 1
    print(f"{'itération':>10s} {'ajustement':>12s} {'évaluation':>12s}")
    for i in (1, 25, 50, 100, 200, 300, 400, 500, 600):
        if i <= len(c):
            r = c.iloc[i - 1]
            print(f"{i:10d} {r.ajustement:12.4f} {r.evaluation:12.4f}")
    print("=" * 60)
    print(f"Meilleure évaluation : itération {best}  →  PR-AUC {c.evaluation.max():.4f}")
    print(f"À 400 itérations     : {c.evaluation.iloc[399]:.4f}"
          if len(c) >= 400 else "")
    ecart = c.ajustement.iloc[-1] - c.evaluation.iloc[-1]
    print(f"Écart final ajustement − évaluation : {ecart:.4f}  "
          f"{'← sur-apprentissage marqué' if ecart > .15 else ''}")
    print(f"\n✅ {dest.name}")


if __name__ == "__main__":
    main()
