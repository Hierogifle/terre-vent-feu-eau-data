"""Étape 24 — expliquer le modèle qui est RÉELLEMENT DÉPLOYÉ.

    python -m tvfed.explications

────────────────────────────────────────────────────────────────────────────
POURQUOI CE MODULE EXISTE, ALORS QUE `interpretation.py` FAIT DÉJÀ DU SHAP
────────────────────────────────────────────────────────────────────────────
`interpretation.py` explique le **v3, 52 features**. L'application sert le
**modèle C, 41 features**. Les 11 en trop sont exactement l'historique BDIFF
(`feux_commune_*`, `taux_*_lisse`, `ratio_commune_cluster`, `cluster_id`) et
`lat`/`lon`.

Publier le SHAP du v3 à côté d'une carte dessinée par C décrirait un modèle
**qui n'est pas celui qu'on regarde** — et l'erreur serait invisible : les deux
tableaux ont la même tête, les mêmes noms de features, la même allure. C'est
la même famille de piège que le désalignement de lignes corrigé dans
`comparer.py` : rien ne casse, le chiffre est simplement faux.

D'où ce module. Il produit trois choses :

    shap_c.npy / shap_c_X.parquet   TreeSHAP sur les 41 features du modèle C
    fond_dice.parquet               un fond de référence pour LIME et DiCE
    (le tout recopié vers app/donnees par export_app)

────────────────────────────────────────────────────────────────────────────
DEUX ÉCHANTILLONS, ET IL FAUT LES DEUX
────────────────────────────────────────────────────────────────────────────
Un échantillon **aléatoire** décrit le modèle sur le territoire tel qu'il est :
99,98 % de communes-jours sans feu. Un échantillon du **sommet** décrit ce que
le modèle fait quand il s'engage.

Ils ne racontent pas la même histoire, et une version antérieure de l'analyse
d'interaction avait conclu l'inverse de la réalité en ne regardant que le
sommet — sélectionner sur le score introduit un **biais de collision**. On
garde donc les deux, et on dit lequel on lit.

────────────────────────────────────────────────────────────────────────────
POURQUOI UN « FOND » POUR LIME ET DiCE
────────────────────────────────────────────────────────────────────────────
TreeSHAP n'a besoin que du modèle : il lit la structure des arbres. LIME et
DiCE, eux, sont **agnostiques au modèle** et doivent donc échantillonner
autour du point à expliquer — ce qui exige de connaître la distribution des
features. On leur fournit un extrait du train.

⚠️ Du TRAIN, jamais de la validation ni du test. Ce fond décrit la
distribution que le modèle a vue ; le prendre ailleurs ferait entrer des
données d'évaluation dans un artefact servi par l'application.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import clustering, interpretation
from .modele_c import RETIRE
from .modele_v3 import K, METHODE
from .modeles import CIBLE, Preparation
from .paths import PROCESSED

N_FOND = 30_000          # lignes de train servies à LIME et DiCE


def modele_c_et_prep():
    """Charge `modele_c.json` et la préparation ajustée sur le train.

    ⚠️ Le clustering est RECALCULÉ avec les mêmes constantes que `modele_c`,
    et non relu depuis `taux_lisses.parquet` : ce fichier a pu être régénéré
    entre-temps avec un autre k, et les colonnes porteraient les mêmes noms
    pour des valeurs différentes. KMeans est à graine fixe, le résultat est
    reproductible.
    """
    from xgboost import XGBClassifier

    print(f"clustering {METHODE} k={K} (recalculé)…")
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
    garde = [c for c in prep.colonnes_ if c not in RETIRE]

    fichier = PROCESSED / "modele_c.json"
    if not fichier.exists():
        raise FileNotFoundError(
            "modele_c.json absent — lancer d'abord :\n"
            "    python -m tvfed.modele_c --split test")
    m = XGBClassifier()
    m.load_model(fichier)

    # ⚠️ GARDE-FOU. Un modèle chargé sur un autre jeu de colonnes produirait
    # des explications silencieusement fausses — XGBoost ne se plaint que si
    # les NOMS diffèrent, pas si le sens a changé.
    attendu = list(m.get_booster().feature_names)
    if attendu != garde:
        raise ValueError(
            f"le modèle attend {len(attendu)} features, la préparation en "
            f"produit {len(garde)}. Écart : "
            f"{sorted(set(attendu) ^ set(garde))}")
    print(f"modèle C rechargé — {len(garde)} features")
    return m, prep, train, taux, garde


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lignes", type=int, default=60_000,
                    help="taille de chaque échantillon SHAP")
    args = ap.parse_args()

    import shap

    m, prep, train, taux, garde = modele_c_et_prep()

    # ── le fond pour LIME et DiCE ───────────────────────────────────────
    # Un extrait du train, dans l'espace des 41 features du modèle, avec la
    # cible : DiCE en a besoin pour connaître le domaine de chaque variable,
    # LIME pour estimer les quantiles qui servent à discrétiser.
    print(f"\nfond de référence : {N_FOND:,} lignes de train…")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(train), min(N_FOND, len(train)), replace=False)
    sous = train.iloc[idx]
    fond = pd.DataFrame(prep.transform(sous), columns=prep.colonnes_)[garde]
    fond[CIBLE] = sous[CIBLE].to_numpy()
    fond.to_parquet(PROCESSED / "fond_dice.parquet", index=False,
                    compression="zstd")
    print(f"  {len(fond):,} lignes, {int(fond[CIBLE].sum()):,} feux "
          f"({fond[CIBLE].mean():.2%} — le train est sous-échantillonné 1:10)")

    # ── échantillons de validation ──────────────────────────────────────
    print(f"\nparcours de la validation, {args.lignes:,} lignes par échantillon…")
    A, S = interpretation.echantillons(m, prep, taux, args.lignes,
                                       colonnes=garde)
    print(f"  aléatoire : {len(A):,} lignes, {int(A[CIBLE].sum()):,} feux "
          f"({A[CIBLE].mean():.4%})")
    print(f"  sommet    : {len(S):,} lignes, {int(S[CIBLE].sum()):,} feux "
          f"({S[CIBLE].mean():.4%})")

    # ── SHAP ────────────────────────────────────────────────────────────
    # TreeSHAP est EXACT sur un modèle à arbres — ce n'est pas une
    # approximation par échantillonnage comme KernelSHAP. Le seul arbitrage
    # est le nombre de lignes expliquées, jamais la qualité de l'explication.
    # C'est l'argument qui départagera SHAP et LIME dans l'application.
    expl = shap.TreeExplainer(m)
    for nom, D in (("alea", A), ("sommet", S)):
        t0 = time.time()
        X = pd.DataFrame(prep.transform(D), columns=prep.colonnes_)[garde]
        v = expl.shap_values(X)
        # selon la version, TreeExplainer rend (n, f) en binaire ou une liste
        # de deux tableaux (classe 0, classe 1) — on veut toujours la classe 1
        if isinstance(v, list):
            v = v[1]
        elif v.ndim == 3:
            v = v[:, :, 1]
        np.save(PROCESSED / f"shap_c_{nom}.npy", v.astype(np.float32))
        X.assign(**{CIBLE: D[CIBLE].to_numpy(), "_p": D._p.to_numpy()}).to_parquet(
            PROCESSED / f"shap_c_{nom}_X.parquet", index=False)
        print(f"  SHAP {nom:7s} {v.shape}  {time.time() - t0:.0f} s")

    (PROCESSED / "shap_c_colonnes.json").write_text(
        json.dumps(garde, indent=1, ensure_ascii=False), encoding="utf-8")

    # ── ce que le modèle retient, en moyenne ────────────────────────────
    v = np.load(PROCESSED / "shap_c_alea.npy")
    imp = pd.Series(np.abs(v).mean(0), index=garde).sort_values(ascending=False)
    print(f"\nles 10 features qui déplacent le plus le score "
          f"(|SHAP| moyen, échantillon aléatoire) :")
    for f, val in imp.head(10).items():
        print(f"   {f:26s} {val:.4f}")

    print(f"\n✅ shap_c_alea.npy · shap_c_sommet.npy · fond_dice.parquet")


if __name__ == "__main__":
    main()
