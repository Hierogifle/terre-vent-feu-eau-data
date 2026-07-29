"""Étape 6 — les premiers modèles. RandomForest et XGBoost, sans réglage.

    python -m tvfed.modeles

Objectif : obtenir un score de RÉFÉRENCE sur les 55 colonnes actuelles, avant
toute feature apprise. C'est lui qui rendra mesurable l'apport du clustering.

⚠️ Entraînement sur le train ÉCHANTILLONNÉ (9,12 % de positifs), évaluation sur
la validation INTÉGRALE (0,0241 %). Les probabilités sorties sont donc
sur-estimées d'un facteur ~500 — c'est attendu, et c'est la calibration qui le
corrigera. La PR-AUC, elle, ne dépend que de l'ORDRE des scores : elle n'est
pas affectée par ce décalage.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import db, matrices, metriques
from .paths import PROCESSED

# ── colonnes à exclure des features ──────────────────────────────────
# 🔴 FUITE : ces deux colonnes SONT la cible déguisée.
#    nb_feux > 0  ⟺  y = 1        surface_m2 > 0  ⟺  y = 1
#    Les laisser donnerait une PR-AUC de 1,00 et un modèle sans valeur.
FUITE = ["nb_feux", "surface_m2"]

# identifiants et métadonnées : pas des features
IDENTIFIANTS = ["code_insee", "date", "split"]

# 96 modalités sans ordre naturel. lat/lon et distance_cote_km encodent déjà
# la géographie de façon continue et plus fine. Écarté du premier modèle.
CATEGORIELLES = ["dep_code"]

CIBLE = "y"


class Preparation:
    """Imputation des manquants — apprise sur le TRAIN, appliquée partout.

    Un imputeur a un `.fit()` : c'est une transformation apprise. La fitter sur
    l'ensemble des données ferait fuiter la distribution de val/test.
    """

    def __init__(self):
        self.medianes_: pd.Series | None = None
        self.colonnes_: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> "Preparation":
        assert set(df["split"].unique()) == {"train"}, (
            f"fuite : Preparation.fit() a vu {set(df['split'].unique())}"
        )
        X = self._features(df)
        self.colonnes_ = list(X.columns)
        self.medianes_ = X.median()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._features(df).reindex(columns=self.colonnes_)
        return X.fillna(self.medianes_).to_numpy(np.float32)

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        aban = set(FUITE + IDENTIFIANTS + CATEGORIELLES + [CIBLE])
        return df[[c for c in df.columns if c not in aban]]


def entrainer(train: pd.DataFrame) -> tuple[dict, Preparation, list[str]]:
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    prep = Preparation().fit(train)
    X, y = prep.transform(train), train[CIBLE].to_numpy()
    print(f"   matrice d'entraînement : {X.shape[0]:,} × {X.shape[1]} features")

    modeles = {}

    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, max_features="sqrt",
        n_jobs=-1, random_state=42,
    ).fit(X, y)
    modeles["RandomForest"] = rf
    print(f"   RandomForest entraîné       {time.time() - t0:6.1f} s")

    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", eval_metric="aucpr",
        n_jobs=-1, random_state=42,
    ).fit(X, y)
    modeles["XGBoost"] = xgb
    print(f"   XGBoost entraîné            {time.time() - t0:6.1f} s")

    return modeles, prep, prep.colonnes_


def evaluer(modeles: dict, prep: Preparation, split: str) -> pd.DataFrame:
    """Parcourt une partition INTÉGRALE par blocs et accumule (score, y).

    On ne garde que les prédictions et la cible — 38 M × 5 octets = 190 Mo,
    au lieu des ~6 Go qu'occuperait la matrice de features complète.
    """
    scores = {n: [] for n in modeles}
    cibles, n_lu, t0 = [], 0, time.time()

    for bloc in matrices.parcourir(split):
        X = prep.transform(bloc)
        for nom, m in modeles.items():
            scores[nom].append(m.predict_proba(X)[:, 1].astype(np.float32))
        cibles.append(bloc[CIBLE].to_numpy(np.int8))
        n_lu += len(bloc)
        if n_lu % 5_000_000 < len(bloc):
            print(f"      {n_lu:>12,} lignes   {time.time() - t0:6.0f} s")

    y = np.concatenate(cibles)
    taux = y.mean()
    print(f"   {split} : {len(y):,} lignes, {y.sum():,} positifs ({100*taux:.4f} %)")

    from sklearn.metrics import average_precision_score

    lignes = []
    for nom in modeles:
        p = np.concatenate(scores[nom])
        ap = average_precision_score(y, p)
        lignes.append({
            "modele": nom, "pr_auc": ap, "lift": ap / taux,
            "p_moyen": float(p.mean()), "p_max": float(p.max()),
        })
    return pd.DataFrame(lignes)


def main() -> None:
    print("Entraînement\n" + "=" * 68)
    train = pd.read_parquet(PROCESSED / "train.parquet")
    modeles, prep, noms = entrainer(train)

    print("\nÉvaluation sur la validation intégrale (streaming)\n" + "=" * 68)
    res = evaluer(modeles, prep, "val")

    base = pd.read_csv(PROCESSED / "baselines.csv")
    ref = base.pr_auc.max()

    print("\n" + "═" * 68)
    print(f"{'':22s} {'PR-AUC':>9s} {'lift':>8s}   verdict")
    print("─" * 68)
    for _, r in base.iterrows():
        print(f"{r.predicteur:22s} {r.pr_auc:9.4f} {r.lift:7.1f}×")
    print("─" * 68)
    for _, r in res.iterrows():
        v = "BAT les baselines" if r.pr_auc > ref else "SOUS la meilleure baseline"
        print(f"{r.modele:22s} {r.pr_auc:9.4f} {r.lift:7.1f}×   {v}")
    print("═" * 68)

    meilleur = res.loc[res.pr_auc.idxmax()]
    print(f"\nMeilleure baseline : {ref:.4f}")
    print(f"Meilleur modèle    : {meilleur.pr_auc:.4f}  "
          f"({meilleur.pr_auc / ref:.2f}× la baseline)")
    print(f"\n⚠️  Probabilité moyenne prédite : {meilleur.p_moyen:.4f} — attendu ~0.0002.")
    print("    Sur-estimation due au downsampling, corrigée à la calibration.")
    print("    La PR-AUC, elle, ne dépend que de l'ordre : elle est valide.")

    if meilleur.pr_auc > 0.80:
        print("\n🔴 PR-AUC > 0,80 sur un événement à 0,02 % → CHERCHER LA FUITE.")

    res.to_csv(PROCESSED / "modeles_v1.csv", index=False)

    # importances — indicatives seulement : biaisées vers les variables
    # continues, et partagées entre features corrélées
    imp = pd.DataFrame({
        "feature": noms,
        "rf": modeles["RandomForest"].feature_importances_,
        "xgb": modeles["XGBoost"].feature_importances_,
    }).sort_values("xgb", ascending=False)
    imp.to_csv(PROCESSED / "importances_v1.csv", index=False)
    print("\nTop 12 features (XGBoost) :")
    for _, r in imp.head(12).iterrows():
        print(f"   {r.feature:28s} xgb={r.xgb:.4f}  rf={r.rf:.4f}")


if __name__ == "__main__":
    main()
