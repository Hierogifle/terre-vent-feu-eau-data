"""Étape 8 — calibration des probabilités.

    python -m tvfed.calibration

Le modèle est entraîné sur un train échantillonné à 1:10 : il y voit 9,12 % de
positifs alors que la réalité en compte 0,0241 %. Ses probabilités sont donc
sur-estimées d'un facteur ~400. Il classe bien, mais il ment sur les niveaux.

Or l'application affiche un score à un utilisateur. Un « 30 % de risque » doit
vouloir dire que ça arrive 30 % du temps — la logique du « 70 % de chance de
pluie ». C'est distinct de la capacité à classer : un modèle peut parfaitement
ordonner les communes et être totalement décalé en niveau.

⚠️ La correction s'apprend sur la validation INTÉGRALE, jamais échantillonnée.
C'est précisément pour ça qu'on l'a gardée entière : le taux de positifs y est
le vrai, donc la fonction de correction apprise absorbe le décalage de prior
automatiquement — sans avoir à appliquer la formule fermée de rééchantillonnage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ajuster(p_val: np.ndarray, y_val: np.ndarray, methode: str = "isotonic"):
    """Apprend la correction sur (probabilité prédite, réalité observée).

    isotonic : monotone, non paramétrique — n'impose aucune forme, mais
               demande beaucoup de données. Ici : 38 M lignes, donc largement.
    platt    : sigmoïde à 2 paramètres — plus robuste sur petits volumes,
               mais suppose une forme logistique de l'erreur.

    Comme la transformation est MONOTONE, elle ne change pas l'ordre des
    scores : la PR-AUC est identique avant et après. Seuls les niveaux bougent.
    """
    if methode == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val, y_val)

    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e10, solver="lbfgs")
    # on travaille en log-odds : c'est l'échelle naturelle de Platt
    z = np.log(np.clip(p_val, 1e-9, 1 - 1e-9) / (1 - np.clip(p_val, 1e-9, 1 - 1e-9)))
    lr.fit(z.reshape(-1, 1), y_val)
    return _PlattWrapper(lr)


class _PlattWrapper:
    def __init__(self, lr):
        self.lr = lr

    def predict(self, p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return self.lr.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def diagnostic(p: np.ndarray, y: np.ndarray, bins: int = 15) -> pd.DataFrame:
    """Courbe de fiabilité : ce qu'on annonce vs ce qui arrive.

    Découpage par QUANTILES et non en tranches égales : sur un événement à
    0,02 %, des tranches régulières mettraient 99 % des lignes dans le premier
    bac et ne montreraient rien.
    """
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    d = pd.DataFrame({"bac": idx, "p": p, "y": y})
    return (d.groupby("bac")
              .agg(annonce=("p", "mean"), observe=("y", "mean"), effectif=("y", "size"))
              .reset_index(drop=True))


def resume(p_avant, p_apres, y) -> pd.DataFrame:
    """Brier et écart de niveau, avant et après correction."""
    from sklearn.metrics import average_precision_score, brier_score_loss

    lignes = []
    for nom, p in [("avant", p_avant), ("après", p_apres)]:
        lignes.append({
            "etat": nom,
            "brier": brier_score_loss(y, p),
            "p_moyen": p.mean(),
            "taux_reel": y.mean(),
            "biais": p.mean() / y.mean(),          # 1,0 = parfaitement recalé
            "pr_auc": average_precision_score(y, p),
        })
    return pd.DataFrame(lignes)
