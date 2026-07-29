"""Étape 9 — interprétation du modèle v3 : corrélations et SHAP.

    python -m tvfed.interpretation --lignes 60000

────────────────────────────────────────────────────────────────────────────
POURQUOI LA MATRICE DE CORRÉLATION AVANT SHAP
────────────────────────────────────────────────────────────────────────────
Le v3 vient de montrer que `taux_commune_lisse` a pris 25 points d'importance
pendant que `feux_commune_365j` en perdait 24. Les deux disent la même chose :
« cette commune brûle-t-elle ». Ils sont donc fortement corrélés.

C'est le piège classique de l'interprétation d'un modèle à arbres. Quand deux
features portent la même information, l'arbre en choisit UNE à chaque nœud —
souvent la plus fine — et l'autre paraît inutile. SHAP hérite du problème :
il attribue la contribution à celle qui a été choisie, pas aux deux.

**Lire un graphique SHAP sans la matrice de corrélation sous les yeux, c'est
risquer de conclure qu'une variable ne sert à rien alors qu'elle est
simplement doublée.**

────────────────────────────────────────────────────────────────────────────
SUR QUELLES LIGNES CALCULER SHAP
────────────────────────────────────────────────────────────────────────────
38 M lignes sont hors de question. On en tire deux échantillons, qui ne
répondent pas à la même question :

  ALÉATOIRE     ce que fait le modèle sur une journée ordinaire.
                Représentatif, mais ne contient presque aucun feu
                (0,024 % de positifs).

  SOMMET        les lignes les mieux notées — celles qu'un opérationnel
                regarderait vraiment. C'est là que se joue la valeur du
                modèle, et le seul endroit où la précision n'est pas nulle.

⚠️ Les deux échantillons n'ont PAS le même taux de positifs. Les amplitudes
SHAP ne se comparent donc pas d'un graphique à l'autre — seulement les
classements et les formes.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import clustering, matrices
from .modeles import CIBLE, Preparation
from .paths import PROCESSED

SEUIL_CORR = 0.90        # au-delà, deux features sont considérées redondantes
N_SOMMET = 400_000       # lignes de val parcourues pour retenir le sommet


def _modele_et_prep():
    """Charge le v3 sauvegardé, et la préparation ajustée sur le train.

    ⚠️ Le clustering est RECALCULÉ, pas relu depuis taux_lisses.parquet.
    Ce fichier peut avoir été régénéré entre-temps avec un autre k, et les
    features ne correspondraient plus au modèle sauvegardé — sans qu'aucune
    erreur ne se déclenche, puisque les colonnes porteraient les mêmes noms.
    On repart des mêmes constantes que le v3 ; KMeans est graine fixe, le
    résultat est donc identique.
    """
    import json

    from xgboost import XGBClassifier

    from .modele_v3 import K, METHODE

    fichier = PROCESSED / "modele_v3.json"
    params = json.loads(
        (PROCESSED / "best_params_xgb.json").read_text(encoding="utf-8"))
    for k in ("n_estimators", "max_depth", "min_child_weight"):
        params[k] = int(params[k])

    print(f"clustering {METHODE} k={K} (recalculé)…")
    p = clustering.profil()
    sin = clustering.sinistralite()
    cl = clustering.ajuster(p, METHODE, K)
    manquantes = sorted(set(sin.code_insee) - set(cl.index))
    if manquantes:
        cl = pd.concat([cl, pd.Series(-1, index=manquantes, name="cluster_id")])
    taux = clustering.lisser(sin, cl)

    train = clustering.appliquer(
        pd.read_parquet(PROCESSED / "train.parquet"), taux)
    prep = Preparation().fit(train)

    m = XGBClassifier(**params, tree_method="hist", eval_metric="aucpr",
                      n_jobs=-1, random_state=42)
    if fichier.exists():
        m.load_model(fichier)
        print(f"modèle rechargé depuis {fichier.name}")
    else:
        print("modele_v3.json absent — réentraînement (~30 s)…")
        m.fit(prep.transform(train), train[CIBLE].to_numpy())
        m.save_model(fichier)

    # garde-fou : un modèle entraîné sur un autre jeu de colonnes produirait
    # des explications SHAP silencieusement fausses
    if m.n_features_in_ != len(prep.colonnes_):
        raise ValueError(
            f"le modèle attend {m.n_features_in_} features, la préparation en "
            f"produit {len(prep.colonnes_)} — relancer tvfed.modele_v3"
        )
    return m, prep, train, taux


# ════════════════════════════════════════════════════════════════════════
#  1. corrélations — quelles features disent la même chose ?
# ════════════════════════════════════════════════════════════════════════
def correlations(train: pd.DataFrame, prep: Preparation) -> pd.DataFrame:
    """Matrice de Spearman sur le train.

    Spearman et non Pearson : plusieurs features sont très asymétriques
    (`jours_depuis_dernier_feu`, les comptages de feux, les parts CORINE
    écrasées sur zéro). Une corrélation de rang mesure la co-monotonie sans
    supposer de linéarité, ce qui correspond à ce qu'un arbre exploite.
    """
    X = pd.DataFrame(prep.transform(train), columns=prep.colonnes_)
    return X.corr(method="spearman")


def paires_redondantes(C: pd.DataFrame, seuil: float = SEUIL_CORR) -> pd.DataFrame:
    haut = C.where(np.triu(np.ones(C.shape), k=1).astype(bool)).stack()
    p = haut[haut.abs() >= seuil].sort_values(key=abs, ascending=False)
    return pd.DataFrame({"feature_a": [i[0] for i in p.index],
                         "feature_b": [i[1] for i in p.index],
                         "spearman": p.to_numpy()})


# ════════════════════════════════════════════════════════════════════════
#  2. les deux échantillons de validation
# ════════════════════════════════════════════════════════════════════════
def echantillons(m, prep, taux, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (aléatoire, sommet) — deux vues du même modèle."""
    rng = np.random.default_rng(42)
    alea, hauts, lu, t0 = [], [], 0, time.time()

    for bloc in matrices.parcourir("val"):
        bloc = clustering.appliquer(bloc, taux)
        X = prep.transform(bloc)
        p = m.predict_proba(X)[:, 1]

        # échantillon aléatoire : une fraction constante de chaque bloc
        garde = rng.random(len(bloc)) < (n / 38_068_464)
        if garde.any():
            alea.append(bloc[garde].assign(_p=p[garde]))

        # sommet : on conserve les mieux notées de chaque bloc, puis on
        # retaille à la fin — évite de garder 38 M scores en mémoire
        k = min(len(bloc), max(1, int(N_SOMMET * len(bloc) / 38_068_464)))
        idx = np.argpartition(-p, k - 1)[:k]
        hauts.append(bloc.iloc[idx].assign(_p=p[idx]))

        lu += len(bloc)
        if lu % 10_000_000 < len(bloc):
            print(f"   {lu:>12,} lignes   {time.time() - t0:5.0f} s")

    A = pd.concat(alea, ignore_index=True)
    S = pd.concat(hauts, ignore_index=True).nlargest(n, "_p")
    return A, S


# ════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lignes", type=int, default=60_000,
                    help="taille de chaque échantillon SHAP")
    args = ap.parse_args()

    import shap

    m, prep, train, taux = _modele_et_prep()

    # ── corrélations ────────────────────────────────────────────────────
    print("\nmatrice de corrélation (Spearman, sur le train)…")
    C = correlations(train, prep)
    C.to_parquet(PROCESSED / "correlations_v3.parquet")
    red = paires_redondantes(C)
    red.to_csv(PROCESSED / "paires_redondantes.csv", index=False)
    print(f"  {len(red)} paires au-dessus de |ρ| = {SEUIL_CORR}")
    for r in red.head(12).itertuples():
        print(f"   {r.spearman:+.3f}   {r.feature_a:26s} ↔ {r.feature_b}")

    # ── échantillons ────────────────────────────────────────────────────
    print(f"\nparcours de la validation, {args.lignes:,} lignes par échantillon…")
    A, S = echantillons(m, prep, taux, args.lignes)
    print(f"  aléatoire : {len(A):,} lignes, {A[CIBLE].sum():,} feux "
          f"({A[CIBLE].mean():.4%})")
    print(f"  sommet    : {len(S):,} lignes, {S[CIBLE].sum():,} feux "
          f"({S[CIBLE].mean():.4%})")

    # ── SHAP ────────────────────────────────────────────────────────────
    # TreeSHAP est EXACT sur un modèle à arbres — pas une approximation par
    # échantillonnage comme KernelSHAP. Le seul arbitrage est le nombre de
    # lignes expliquées, pas la qualité de l'explication.
    expl = shap.TreeExplainer(m)
    for nom, D in (("alea", A), ("sommet", S)):
        t0 = time.time()
        X = prep.transform(D)
        v = expl.shap_values(X)
        # selon la version, TreeExplainer rend (n, f) en binaire ou une liste
        # de deux tableaux (classe 0, classe 1) — on veut toujours la classe 1
        if isinstance(v, list):
            v = v[1]
        elif v.ndim == 3:
            v = v[:, :, 1]
        np.save(PROCESSED / f"shap_{nom}.npy", v.astype(np.float32))
        D[prep.colonnes_ + [CIBLE, "_p"]].to_parquet(
            PROCESSED / f"shap_{nom}_X.parquet", index=False)
        print(f"  SHAP {nom:7s} {v.shape}  {time.time() - t0:.0f} s")

    pd.Series(prep.colonnes_).to_csv(PROCESSED / "shap_colonnes.csv",
                                     index=False, header=["feature"])
    print("\n✅ correlations_v3.parquet · paires_redondantes.csv · shap_*.npy")


if __name__ == "__main__":
    main()
