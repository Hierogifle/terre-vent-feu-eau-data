"""Métriques adaptées à un événement rare.

La validation fait 38 M lignes : on ne peut pas la charger en mémoire ligne à
ligne. Mais un prédicteur simple ne produit qu'un petit nombre de scores
DISTINCTS (6 pour les classes EFFIS, ~34 000 pour un taux par commune).

On peut donc travailler sur la forme AGRÉGÉE — (score, effectif, positifs) —
et calculer les métriques exactement, sans jamais matérialiser les 38 M lignes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pr_auc_groupe(scores: np.ndarray, n: np.ndarray, pos: np.ndarray) -> float:
    """Average precision, calculée depuis des données groupées.

    Équivalent exact de `sklearn.average_precision_score` : les ex æquo y sont
    de toute façon traités comme un seul seuil, ce qui est précisément ce que
    représente un groupe ici.

    scores : le score prédit de chaque groupe
    n      : effectif du groupe
    pos    : nombre de positifs du groupe
    """
    ordre = np.argsort(-np.asarray(scores, float))
    n, pos = np.asarray(n, float)[ordre], np.asarray(pos, float)[ordre]

    tp = np.cumsum(pos)          # vrais positifs cumulés, seuil par seuil
    pred = np.cumsum(n)          # nombre de prédictions positives cumulées
    total_pos = tp[-1]

    precision = tp / pred
    rappel = tp / total_pos
    # AP = Σ (R_k − R_{k−1}) · P_k
    return float(np.sum(np.diff(np.r_[0.0, rappel]) * precision))


def brier_groupe(scores: np.ndarray, n: np.ndarray, pos: np.ndarray) -> float:
    """Brier score = erreur quadratique moyenne sur des probabilités.

    Mesure la CALIBRATION (« quand j'annonce 30 %, ça arrive 30 % du temps »),
    distincte de la discrimination que mesure la PR-AUC.
    """
    scores, n, pos = (np.asarray(x, float) for x in (scores, n, pos))
    # somme des (p − y)² : pos lignes à y=1, (n − pos) à y=0
    sse = np.sum(pos * (scores - 1) ** 2 + (n - pos) * scores ** 2)
    return float(sse / n.sum())


def rapport(nom: str, scores, n, pos, taux_base: float) -> dict:
    """Score un prédicteur et le situe par rapport au hasard."""
    ap = pr_auc_groupe(scores, n, pos)
    return {
        "predicteur": nom,
        "pr_auc": ap,
        "brier": brier_groupe(scores, n, pos),
        "lift": ap / taux_base,          # combien de fois mieux que le hasard
    }


def courbe_calibration(scores, n, pos, bins: int = 10) -> pd.DataFrame:
    """Probabilité annoncée vs fréquence observée, par décile de score."""
    df = pd.DataFrame({"score": scores, "n": n, "pos": pos}).sort_values("score")
    df["cum"] = df.n.cumsum()
    df["bin"] = np.minimum((bins * df.cum / df.n.sum()).astype(int), bins - 1)
    g = df.groupby("bin").apply(
        lambda d: pd.Series({
            "annonce": np.average(d.score, weights=d.n),
            "observe": d.pos.sum() / d.n.sum(),
            "effectif": d.n.sum(),
        }),
        include_groups=False,
    )
    return g.reset_index()
