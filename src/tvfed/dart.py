"""Étape 10b — DART : le dropout, transposé aux arbres.

    python -m tvfed.dart

────────────────────────────────────────────────────────────────────────────
CE QUE DART FAIT
────────────────────────────────────────────────────────────────────────────
*Dropouts meet Multiple Additive Regression Trees*. C'est le transfert
littéral de l'idée du dropout aux ensembles d'arbres.

Le boosting classique (`gbtree`) construit chaque arbre pour corriger les
erreurs de TOUS les précédents. Conséquence connue : les premiers arbres
prennent un poids démesuré, les derniers ne font plus que du réglage fin sur
des résidus minuscules — c'est exactement ce que montrait notre courbe
d'apprentissage, plateau à 106 arbres sur 400.

DART éteint une fraction des arbres DÉJÀ CONSTRUITS avant d'en ajouter un
nouveau. Le nouvel arbre doit donc corriger un ensemble amputé, et ne peut pas
se spécialiser sur ce qui reste après les autres. Aucun arbre ne devient
indispensable — le principe même du dropout.

────────────────────────────────────────────────────────────────────────────
LES TROIS MÉCANISMES DE MASQUAGE, CÔTE À CÔTE
────────────────────────────────────────────────────────────────────────────
    RandomForest    max_features        cache des COLONNES, à chaque nœud
    XGBoost gbtree  subsample           cache des LIGNES, à chaque arbre
                    colsample_bytree    cache des COLONNES, à chaque arbre
    XGBoost DART    rate_drop           cache des ARBRES ENTIERS
    MLP             dropout             cache des NEURONES

⚠️ DART est notablement plus LENT que gbtree : à chaque itération il doit
recalculer la prédiction de l'ensemble amputé, au lieu d'ajouter simplement le
nouvel arbre. Compter ×3 à ×5 sur le temps d'entraînement.

⚠️ Et un piège de prédiction : par défaut XGBoost applique le dropout aussi à
l'inférence, ce qui rend les prédictions ALÉATOIRES. `predict_proba` de
l'API scikit-learn passe `training=False`, donc tous les arbres votent — mais
c'est à vérifier, pas à supposer. Le contrôle est fait plus bas.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import clustering, matrices
from .modele_v3 import K, METHODE
from .modeles import CIBLE, Preparation
from .paths import PROCESSED

# GPU si disponible : DART est lent, autant ne pas s'en priver
DEVICE = "cuda"


def main() -> None:
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    params = json.loads(
        (PROCESSED / "best_params_xgb.json").read_text(encoding="utf-8"))
    for k in ("n_estimators", "max_depth", "min_child_weight"):
        params[k] = int(params[k])

    # ── mêmes features que le v3 ────────────────────────────────────────
    print(f"clustering {METHODE} k={K}…")
    p = clustering.profil()
    sin = clustering.sinistralite()
    cl = clustering.ajuster(p, METHODE, K)
    manq = sorted(set(sin.code_insee) - set(cl.index))
    if manq:
        cl = pd.concat([cl, pd.Series(-1, index=manq, name="cluster_id")])
    taux = clustering.lisser(sin, cl)

    train = clustering.appliquer(
        pd.read_parquet(PROCESSED / "train.parquet"), taux)
    prep = Preparation().fit(train)
    X, y = prep.transform(train), train[CIBLE].to_numpy()

    # ── entraînement DART ───────────────────────────────────────────────
    # rate_drop 0,1 : 10 % des arbres éteints à chaque itération.
    # skip_drop 0,5 : une itération sur deux se fait SANS dropout — c'est le
    #   réglage conseillé par les auteurs, un dropout à toutes les itérations
    #   rendant la convergence très lente.
    dart = dict(params, booster="dart", rate_drop=0.1, skip_drop=0.5,
                sample_type="uniform", normalize_type="tree")
    print(f"\nentraînement DART sur {DEVICE} "
          f"(rate_drop {dart['rate_drop']}, skip_drop {dart['skip_drop']})…")
    t0 = time.time()
    m = XGBClassifier(**dart, tree_method="hist", eval_metric="aucpr",
                      device=DEVICE, n_jobs=-1, random_state=42).fit(X, y)
    print(f"  {time.time() - t0:.0f} s")

    # ── ⚠️ contrôle : les prédictions sont-elles déterministes ? ─────────
    # Si DART appliquait son dropout à l'inférence, deux appels successifs
    # donneraient des scores différents et le modèle serait inutilisable en
    # production. On le vérifie au lieu de le supposer.
    a = m.predict_proba(X[:5000])[:, 1]
    b = m.predict_proba(X[:5000])[:, 1]
    if not np.allclose(a, b):
        raise RuntimeError(
            "les prédictions DART ne sont pas déterministes — le dropout "
            "s'applique à l'inférence, le modèle est inexploitable tel quel"
        )
    print("  ✓ prédictions déterministes (dropout désactivé à l'inférence)")

    # ── évaluation sur la validation intégrale ──────────────────────────
    print("\névaluation sur la validation intégrale…")
    # cf. modele_v3 : sans (commune, date), un fichier de prédictions n'est
    # comparable à aucun autre — la requête n'a pas d'ORDER BY.
    scores, cibles, cles, n, t0 = [], [], [], 0, time.time()
    for bloc in matrices.parcourir("val"):
        bloc = clustering.appliquer(bloc, taux)
        scores.append(m.predict_proba(prep.transform(bloc))[:, 1].astype(np.float32))
        cibles.append(bloc[CIBLE].to_numpy(np.int8))
        cles.append(bloc[["code_insee", "date"]])
        n += len(bloc)
        if n % 10_000_000 < len(bloc):
            print(f"   {n:>12,} lignes   {time.time() - t0:5.0f} s")

    pr, yv = np.concatenate(scores), np.concatenate(cibles)
    ap = average_precision_score(yv, pr)
    pd.concat(cles, ignore_index=True).assign(p_dart=pr, y=yv).to_parquet(
        PROCESSED / "predictions_val_dart.parquet", index=False, compression="zstd")

    base = pd.read_csv(PROCESSED / "baselines.csv").pr_auc.max()
    v3 = pd.read_csv(PROCESSED / "modeles_v3.csv").pr_auc[0]
    print("\n" + "═" * 62)
    print(f"{'':34s} {'PR-AUC':>9s} {'lift':>8s}")
    print(f"{'meilleure baseline':34s} {base:9.4f} {base / yv.mean():7.1f}×")
    print(f"{'XGBoost v3 (gbtree)':34s} {v3:9.4f} {v3 / yv.mean():7.1f}×")
    print(f"{'XGBoost DART (dropout d arbres)':34s} {ap:9.4f} {ap / yv.mean():7.1f}×")
    print("═" * 62)
    print(f"\nécart au gbtree : {100 * (ap / v3 - 1):+.1f} %")

    pd.DataFrame([{"modele": "XGBoost DART", "pr_auc": ap,
                   "lift": ap / yv.mean()}]).to_csv(
        PROCESSED / "modeles_dart.csv", index=False)
    m.save_model(PROCESSED / "modele_dart.json")
    print("\n✅ modeles_dart.csv · predictions_val_dart.parquet")


if __name__ == "__main__":
    main()
