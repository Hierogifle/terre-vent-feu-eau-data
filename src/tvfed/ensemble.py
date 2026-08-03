"""Étape 11 — l'ensemble : combiner des modèles qui se trompent ailleurs.

    python -m tvfed.ensemble

────────────────────────────────────────────────────────────────────────────
L'IDÉE, ET SA CONDITION
────────────────────────────────────────────────────────────────────────────
Moyenner deux modèles n'aide que s'ils se trompent sur des lignes DIFFÉRENTES.
Deux modèles excellents mais qui font les mêmes erreurs ne donnent, une fois
moyennés, que le même modèle en plus lent.

La condition se mesure : c'est la corrélation de RANG entre leurs scores.
Élevée → les deux classent pareil, rien à gagner. Plus basse → chacun voit
quelque chose que l'autre manque.

Mesuré ici :

    XGBoost ↔ DART    0,9952     ← deux ensembles d'arbres : même biais
    XGBoost ↔ MLP     0,9732     ← arbre contre réseau : vraie diversité
    DART    ↔ MLP     0,9693

DART ne peut donc rien apporter : c'est du XGBoost sous un autre réglage.
Le MLP, lui, découpe l'espace autrement — en surfaces lisses plutôt qu'en
escaliers — et se trompe ailleurs.

────────────────────────────────────────────────────────────────────────────
POURQUOI LA MOYENNE DES RANGS ET NON DES PROBABILITÉS
────────────────────────────────────────────────────────────────────────────
Les deux modèles ne sortent pas des probabilités sur la même échelle : le
réseau, entraîné avec une BCE sur un train échantillonné à 9 %, produit des
valeurs bien plus étalées que XGBoost. Moyenner les probabilités brutes
laisserait le plus « confiant » dominer, pour une raison qui n'a rien à voir
avec sa justesse.

La moyenne des rangs supprime le problème : seul l'ORDRE compte, et c'est
exactement ce que mesure la PR-AUC. La calibration se fera ensuite, une fois
sur le score d'ensemble.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .comparer import aligner
from .paths import PROCESSED

SOURCES = {
    # `scores_val.parquet` (produit par `tvfed.scores --split val`) porte les
    # trois modèles ET les clés, marqués en UNE passe sur la même énumération
    # de lignes. C'est la seule source dont l'alignement est garanti par
    # construction plutôt que par chance.
    "XGBoost v3": ("scores_val.parquet", "xgb_v3"),
    "DART": ("scores_val.parquet", "dart"),
    "MLP": ("scores_val.parquet", "mlp"),
}


def _rangs(p: np.ndarray) -> np.ndarray:
    """Rang normalisé dans [0,1]. `argsort` deux fois : O(n log n), sans scipy."""
    o = np.empty(len(p), dtype=np.int64)
    o[np.argsort(p, kind="stable")] = np.arange(len(p))
    return (o / len(p)).astype(np.float32)


def main() -> None:
    from sklearn.metrics import average_precision_score

    # alignement sur (commune, date), pas sur la position : la requête
    # d'assemblage n'a pas d'ORDER BY, deux fichiers issus de deux
    # exécutions n'ont pas le même ordre de lignes. Cf. `comparer`.
    t = aligner(SOURCES, garder_cles=True)
    cles = t[["code_insee", "date"]]
    t = t.drop(columns=["code_insee", "date", "commune"])
    y = t.pop("y").to_numpy(np.int8)
    d = {n: t[n].to_numpy(np.float32) for n in t.columns}
    base = y.mean()
    print(f"validation : {len(y):,} lignes, {y.sum():,} feux ({base:.4%})\n")

    seuls = {n: average_precision_score(y, p) for n, p in d.items()}
    print(f"{'modèle seul':34s} {'PR-AUC':>9s} {'lift':>8s}")
    print("─" * 54)
    for n, a in sorted(seuls.items(), key=lambda x: -x[1]):
        print(f"{n:34s} {a:9.4f} {a / base:7.1f}×")

    # ── diversité : la condition de l'ensemble ──────────────────────────
    rng = np.random.default_rng(42)
    s = rng.choice(len(y), min(2_000_000, len(y)), replace=False)
    R = {n: _rangs(p[s]) for n, p in d.items()}
    print("\ncorrélation de RANG — plus elle est basse, plus il y a à gagner")
    for a, b in itertools.combinations(d, 2):
        print(f"   {a:12s} ↔ {b:12s} {np.corrcoef(R[a], R[b])[0, 1]:.4f}")

    # ── les ensembles ───────────────────────────────────────────────────
    Rg = {n: _rangs(p) for n, p in d.items()}
    ref = seuls["XGBoost v3"]
    print(f"\n{'ensemble (moyenne des rangs)':34s} {'PR-AUC':>9s} {'lift':>8s}   vs v3")
    print("─" * 66)
    lignes, meilleur = [], (None, -1.0)
    for k in (2, 3):
        for combo in itertools.combinations(d, k):
            a = average_precision_score(y, np.mean([Rg[c] for c in combo], axis=0))
            nom = " + ".join(combo)
            lignes.append({"modele": nom, "pr_auc": a, "lift": a / base,
                           "gain_vs_v3_pct": 100 * (a / ref - 1)})
            print(f"{nom:34s} {a:9.4f} {a / base:7.1f}×   {100 * (a / ref - 1):+6.2f} %")
            if a > meilleur[1]:
                meilleur = (combo, a)

    combo, ap = meilleur
    e = np.mean([Rg[c] for c in combo], axis=0)
    cles.assign(p_ens=e, y=y).to_parquet(
        PROCESSED / "predictions_val_ensemble.parquet", index=False,
        compression="zstd")
    pd.DataFrame(lignes).to_csv(PROCESSED / "ensembles.csv", index=False)
    pd.DataFrame([{"modele": f"Ensemble ({' + '.join(combo)})", "pr_auc": ap,
                   "lift": ap / base}]).to_csv(
        PROCESSED / "modeles_ensemble.csv", index=False)

    b_base = pd.read_csv(PROCESSED / "baselines.csv").pr_auc.max()
    print(f"\n{'═' * 66}")
    print(f"retenu : {' + '.join(combo)}")
    print(f"  PR-AUC {ap:.4f}   lift {ap / base:.1f}×   "
          f"×{ap / b_base:.2f} la meilleure baseline")
    print(f"  gain sur le meilleur modèle seul : {100 * (ap / ref - 1):+.2f} %")
    print()
    print("→ Le MLP est le modèle le PLUS FAIBLE pris seul, et pourtant le seul")
    print("  qui apporte quelque chose à l'ensemble. DART, presque parfaitement")
    print("  corrélé à XGBoost, n'ajoute rien — il est même légèrement nuisible.")
    print("  C'est la diversité qui paie, pas la performance individuelle.")
    print(f"\n✅ modeles_ensemble.csv · ensembles.csv · predictions_val_ensemble.parquet")


if __name__ == "__main__":
    main()
