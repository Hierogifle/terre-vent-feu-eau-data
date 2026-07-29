"""Étape 7 — recherche d'hyperparamètres par Optuna.

    python -m tvfed.optimisation --modele xgb --essais 60
    python -m tvfed.optimisation --modele mlp --essais 40

⚠️ DÉCISION DE CONCEPTION — sur quoi Optuna optimise-t-il ?

PAS sur la validation 2020-2022. Elle est déjà réservée à la CALIBRATION.
L'utiliser aussi pour choisir les hyperparamètres reviendrait à s'en servir
deux fois : le modèle finirait par être optimiste sur elle, et la calibration
qui en découle serait faussée.

On découpe donc le train en deux, TEMPORELLEMENT :

    2006-2017  →  ajustement du modèle
    2018-2019  →  évaluation de la configuration

La validation 2020-2022 reste intacte. C'est le même principe que le split
principal, appliqué une couche plus bas.

⚠️ Les scores affichés ici sont bien plus élevés que sur la validation réelle :
le train est échantillonné à 1:10, donc les négatifs y sont 500 fois moins
nombreux qu'en vrai, ce qui gonfle mécaniquement la précision. Ce sont des
scores COMPARATIFS entre configurations, pas des scores absolus.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from .modeles import CIBLE, Preparation
from .paths import PROCESSED

AN_FIT = 2017          # dernière année d'ajustement
AN_EVAL = (2018, 2019)  # années d'évaluation interne


def _decouper(train: pd.DataFrame):
    """Découpe temporelle interne au train. Aucune ligne de val/test ici."""
    annee = pd.to_datetime(train.date).dt.year
    a, b = train[annee <= AN_FIT], train[annee.between(*AN_EVAL)]
    prep = Preparation().fit(a)      # l'imputation s'apprend sur le fit seul
    return (prep.transform(a), a[CIBLE].to_numpy(),
            prep.transform(b), b[CIBLE].to_numpy(), prep)


def objectif_xgb(essai, Xa, ya, Xb, yb):
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    params = {
        "n_estimators":     essai.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate":    essai.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":        essai.suggest_int("max_depth", 3, 10),
        "min_child_weight": essai.suggest_int("min_child_weight", 1, 30),
        "subsample":        essai.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": essai.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma":            essai.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda":       essai.suggest_float("reg_lambda", 1e-3, 20, log=True),
        "reg_alpha":        essai.suggest_float("reg_alpha", 1e-3, 20, log=True),
    }
    m = XGBClassifier(**params, tree_method="hist", eval_metric="aucpr",
                      n_jobs=-1, random_state=42).fit(Xa, ya)
    return average_precision_score(yb, m.predict_proba(Xb)[:, 1])


def objectif_mlp(essai, Xa, ya, Xb, yb):
    """MLP — l'exigence « ANN » de l'énoncé.

    ⚠️ `sklearn.MLPClassifier` n'a pas de dropout : sa régularisation est un
    L2 (`alpha`). L'énoncé mentionne le dropout, qui suppose Keras ou PyTorch
    (~2,5 Go de dépendances). À arbitrer ; `alpha` joue le même rôle ici.

    ⚠️ La standardisation est OBLIGATOIRE pour un réseau (contrairement aux
    arbres). C'est un `.fit()` : il est appris sur la partie ajustement seule.
    """
    from sklearn.metrics import average_precision_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xa)
    Xa_s, Xb_s = sc.transform(Xa), sc.transform(Xb)

    n_couches = essai.suggest_int("n_couches", 1, 4)
    largeur = essai.suggest_categorical("largeur", [32, 64, 128, 256])
    forme = tuple(max(8, largeur // (2 ** i)) for i in range(n_couches))

    m = MLPClassifier(
        hidden_layer_sizes=forme,
        alpha=essai.suggest_float("alpha", 1e-6, 1e-1, log=True),
        learning_rate_init=essai.suggest_float("lr", 1e-4, 1e-2, log=True),
        batch_size=essai.suggest_categorical("batch_size", [256, 512, 1024]),
        activation=essai.suggest_categorical("activation", ["relu", "tanh"]),
        early_stopping=True, n_iter_no_change=8, max_iter=120, random_state=42,
    ).fit(Xa_s, ya)
    return average_precision_score(yb, m.predict_proba(Xb_s)[:, 1])


OBJECTIFS = {"xgb": objectif_xgb, "mlp": objectif_mlp}


def main() -> None:
    import optuna

    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", choices=list(OBJECTIFS), default="xgb")
    ap.add_argument("--essais", type=int, default=60)
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    train = pd.read_parquet(PROCESSED / "train.parquet")
    Xa, ya, Xb, yb, _ = _decouper(train)
    print(f"Optuna — {args.modele}, {args.essais} essais")
    print(f"  ajustement  ≤{AN_FIT} : {len(ya):>7,} lignes, {ya.sum():>6,} positifs")
    print(f"  évaluation  {AN_EVAL[0]}-{AN_EVAL[1]} : {len(yb):>7,} lignes, "
          f"{yb.sum():>6,} positifs\n")

    # étude persistante : relancer la commande reprend là où on s'est arrêté
    PROCESSED.mkdir(parents=True, exist_ok=True)
    etude = optuna.create_study(
        direction="maximize",
        study_name=f"tvfed_{args.modele}",
        storage=f"sqlite:///{(PROCESSED / 'optuna.db').as_posix()}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    t0 = time.time()
    fait = [0]

    def rappel(st, tr):
        fait[0] += 1
        if tr.value is not None and tr.value >= st.best_value:
            print(f"  essai {fait[0]:>3}/{args.essais}  PR-AUC {tr.value:.4f}  ← meilleur")

    etude.optimize(lambda e: OBJECTIFS[args.modele](e, Xa, ya, Xb, yb),
                   n_trials=args.essais, callbacks=[rappel], show_progress_bar=False)

    print(f"\n{'═' * 62}")
    print(f"Meilleure PR-AUC interne : {etude.best_value:.4f}   "
          f"({time.time() - t0:.0f} s, {len(etude.trials)} essais cumulés)")
    print("Hyperparamètres retenus :")
    for k, v in etude.best_params.items():
        print(f"   {k:20s} {v}")

    # ⚠️ json.dump et NON pd.Series.to_json : une Series homogénéise ses types,
    # donc n_estimators=900 ressortait en 900.0 — et XGBoost refuse un flottant
    # là où il attend un entier. Elle tronquait aussi les décimales.
    dest = PROCESSED / f"best_params_{args.modele}.json"
    dest.write_text(json.dumps(etude.best_params, indent=2), encoding="utf-8")
    print(f"\n✅ {dest.name}")
    print("⚠️  Score interne, sur train échantillonné — non comparable aux")
    print("    baselines. La comparaison se fait sur la validation intégrale.")


if __name__ == "__main__":
    main()
