"""Étape 8b — choisir la configuration de clustering.

    python -m tvfed.comparer_clusters

⚠️ LE CHOIX SE FAIT SUR UN DÉCOUPAGE INTERNE AU TRAIN, jamais sur la
validation. Même discipline que pour Optuna : ajustement 2006-2017,
évaluation 2018-2019. Comparer six configurations sur la validation
reviendrait à la consommer six fois, et le score final n'aurait plus rien
d'indépendant.

⚠️ ET IL FAUT LIRE CES ÉCARTS AVEC PRUDENCE. Sur ce même découpage interne,
Optuna annonçait +0,7 % là où la validation réelle a donné +5,2 % — un
facteur 8. La raison : le train échantillonné contient ~8 % de positifs
contre 0,024 % en réalité, et la PR-AUC n'y dépend pas de la même partie du
classement. Ces scores servent à CLASSER des configurations entre elles, pas
à prédire le gain final.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import clustering
from .modeles import CIBLE, Preparation
from .paths import PROCESSED

AN_FIT = 2017
AN_EVAL = (2018, 2019)

# les configurations mises en concurrence
CONFIGS = [
    ("aucun",    None,      0),
    ("kmeans",   "kmeans",  10),
    ("kmeans",   "kmeans",  30),
    ("kmeans",   "kmeans",  60),
    ("kmeans",   "kmeans", 120),
    ("hdbscan",  "hdbscan", 30),
]


def _params() -> dict:
    """Les hyperparamètres retenus par Optuna — mêmes pour toutes les configs.

    Réoptimiser à chaque configuration mélangerait deux effets et rendrait la
    comparaison illisible : on veut mesurer l'apport du clustering, à réglage
    constant.
    """
    p = json.loads((PROCESSED / "best_params_xgb.json").read_text(encoding="utf-8"))
    for k in ("n_estimators", "max_depth", "min_child_weight"):
        p[k] = int(p[k])
    return p


def _evaluer(train: pd.DataFrame, params: dict) -> float:
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    an = pd.to_datetime(train.date).dt.year
    a, b = train[an <= AN_FIT], train[an.between(*AN_EVAL)]
    prep = Preparation().fit(a)
    m = XGBClassifier(**params, tree_method="hist", eval_metric="aucpr",
                      n_jobs=-1, random_state=42).fit(
        prep.transform(a), a[CIBLE].to_numpy())
    return average_precision_score(b[CIBLE].to_numpy(),
                                   m.predict_proba(prep.transform(b))[:, 1])


def main() -> None:
    params = _params()
    base = pd.read_parquet(PROCESSED / "train.parquet")

    print("profil territorial et sinistralité (une seule fois)…")
    p = clustering.profil()
    sin = clustering.sinistralite()
    manquantes = sorted(set(sin.code_insee) - set(p.index))

    resultats = []
    for nom, methode, k in CONFIGS:
        t0 = time.time()
        train = base.copy()

        if methode:
            cl = clustering.ajuster(p, methode, k)
            if manquantes:
                cl = pd.concat([cl, pd.Series(-1, index=manquantes,
                                              name="cluster_id")])
            taux = clustering.lisser(sin, cl)
            train = clustering.appliquer(train, taux)
            n_clu = int(cl.nunique())
        else:
            n_clu = 0

        ap = _evaluer(train, params)
        resultats.append({"config": f"{nom} k={k}" if methode else "aucun",
                          "methode": methode or "—", "k_demande": k,
                          "k_obtenu": n_clu, "pr_auc_interne": ap,
                          "secondes": time.time() - t0})
        print(f"  {resultats[-1]['config']:<16} {n_clu:>4} clusters  "
              f"PR-AUC {ap:.4f}  ({time.time() - t0:.0f} s)")

    R = pd.DataFrame(resultats).sort_values("pr_auc_interne", ascending=False)
    ref = R.loc[R.config == "aucun", "pr_auc_interne"].iloc[0]
    R["gain_pct"] = 100 * (R.pr_auc_interne / ref - 1)
    R.to_csv(PROCESSED / "comparaison_clusters.csv", index=False)

    print(f"\n{'═' * 62}")
    print(R[["config", "k_obtenu", "pr_auc_interne", "gain_pct"]]
          .to_string(index=False))
    print(f"\n→ retenu : {R.iloc[0].config}  ({R.iloc[0].gain_pct:+.2f} % "
          f"sur le découpage interne)")
    print("\n✅ comparaison_clusters.csv")


if __name__ == "__main__":
    main()
