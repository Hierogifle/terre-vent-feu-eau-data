"""Modèle v3 — XGBoost + clustering territorial et lissage bayésien.

    python -m tvfed.modele_v3

Même architecture et mêmes hyperparamètres que le v2. La SEULE différence :
quatre colonnes de plus, issues du clustering.

    cluster_id              le groupe territorial de la commune
    taux_cluster_lisse      le risque de fond de ce type de territoire
    taux_commune_lisse      le risque de la commune, rappelé vers son cluster
    ratio_commune_cluster   la commune brûle-t-elle plus ou moins que ses pairs

L'expérience est donc propre : à features météo, végétation et calendrier
identiques, à réglage identique, qu'apporte le fait de savoir à quel type de
territoire on a affaire ?

────────────────────────────────────────────────────────────────────────────
POURQUOI k = 30 ALORS QUE k = 10 A MESURÉ (UN POIL) MIEUX
────────────────────────────────────────────────────────────────────────────
Sur le découpage interne, les six configurations tiennent dans 0,0016 de
PR-AUC — du bruit. Ce découpage évalue sur du train échantillonné à ~8 % de
positifs, alors que le clustering agit sur l'extrême queue du classement :
il est structurellement aveugle à ce qu'on cherche à mesurer.

Le choix se fait donc sur un argument de fond : 80,4 % des communes n'ont
jamais brûlé sur 2006-2019, et pour elles `taux_commune_lisse` vaut
essentiellement le taux de leur cluster. À k = 10 ces 27 938 communes se
répartissent en 10 valeurs ; à k = 30, en 30. Plus de résolution là où le
besoin est. Et à ~1 150 communes par cluster, soit ~5,9 M jours-commune, le
taux de chaque groupe reste estimé sur assez de matière pour être stable.

C'est un arbitrage assumé sur une différence que la mesure ne sait pas
trancher, pas une conclusion tirée des chiffres.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import clustering, matrices
from .modeles import CIBLE, Preparation
from .paths import PROCESSED

METHODE, K = "kmeans", 30
PATIENCE = 30      # itérations sans progrès avant arrêt


def main() -> None:
    params = json.loads(
        (PROCESSED / "best_params_xgb.json").read_text(encoding="utf-8"))
    for k in ("n_estimators", "max_depth", "min_child_weight"):
        params[k] = int(params[k])

    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    # ── les features de clustering ──────────────────────────────────────
    print(f"clustering {METHODE} k={K}…")
    p = clustering.profil()
    sin = clustering.sinistralite()
    cl = clustering.ajuster(p, METHODE, K)
    manquantes = sorted(set(sin.code_insee) - set(cl.index))
    if manquantes:
        cl = pd.concat([cl, pd.Series(-1, index=manquantes, name="cluster_id")])
    taux = clustering.lisser(sin, cl)
    print(f"  {cl.nunique()} clusters, taux lissés sur "
          f"{sin.jours.sum():,} jours-commune de train")

    # ── entraînement ────────────────────────────────────────────────────
    train = clustering.appliquer(
        pd.read_parquet(PROCESSED / "train.parquet"), taux)
    prep = Preparation().fit(train)
    X, y = prep.transform(train), train[CIBLE].to_numpy()
    print(f"  matrice : {X.shape[0]:,} × {X.shape[1]} features "
          f"(+4 vs le v2)")

    # ── passe 1 : combien d'arbres ? ────────────────────────────────────
    # `n_estimators` vient d'Optuna, donc figé en dur. Si les données changent
    # — une année de plus, un autre périmètre — ce nombre peut devenir faux
    # sans que rien ne le signale. L'arrêt précoce le recale automatiquement.
    #
    # ⚠️ Mesuré sur ce jeu : la meilleure itération est la 889ᵉ sur 900, le
    # modèle progresse encore à la fin. Il n'y a donc RIEN à couper aujourd'hui
    # — Optuna avait déjà réglé la sur-spécialisation en divisant le learning
    # rate par quatre. L'arrêt précoce est ici une assurance, pas un gain.
    an = pd.to_datetime(train.date).dt.year
    a, b = train[an <= 2017], train[an.between(2018, 2019)]
    prep_i = Preparation().fit(a)
    sonde = XGBClassifier(**params, tree_method="hist", eval_metric="aucpr",
                          early_stopping_rounds=PATIENCE, n_jobs=-1,
                          random_state=42).fit(
        prep_i.transform(a), a[CIBLE].to_numpy(),
        eval_set=[(prep_i.transform(b), b[CIBLE].to_numpy())], verbose=False)
    n_arbres = sonde.best_iteration + 1
    print(f"  arrêt précoce : {n_arbres} arbres retenus sur "
          f"{params['n_estimators']}")

    # ── passe 2 : entraînement final sur le train COMPLET ───────────────
    t0 = time.time()
    m = XGBClassifier(**{**params, "n_estimators": n_arbres},
                      tree_method="hist", eval_metric="aucpr",
                      n_jobs=-1, random_state=42).fit(X, y)
    print(f"  entraîné en {time.time() - t0:.0f} s")

    # Sauvegardé pour SHAP : sans ça, l'interprétation devrait réentraîner à
    # chaque exécution. Le format JSON de XGBoost est stable entre versions,
    # contrairement à un pickle.
    m.save_model(PROCESSED / "modele_v3.json")

    # ── évaluation sur la validation intégrale ──────────────────────────
    print("\névaluation sur la validation intégrale…")
    scores, cibles, n, t0 = [], [], 0, time.time()
    for bloc in matrices.parcourir("val"):
        bloc = clustering.appliquer(bloc, taux)
        scores.append(m.predict_proba(prep.transform(bloc))[:, 1].astype(np.float32))
        cibles.append(bloc[CIBLE].to_numpy(np.int8))
        n += len(bloc)
        if n % 10_000_000 < len(bloc):
            print(f"   {n:>12,} lignes   {time.time() - t0:5.0f} s")

    pr = np.concatenate(scores)
    yv = np.concatenate(cibles)
    ap = average_precision_score(yv, pr)

    pd.DataFrame({"p_xgb_v3": pr, "y": yv}).to_parquet(
        PROCESSED / "predictions_val_v3.parquet", index=False, compression="zstd")

    # ── importances, pour savoir si les nouvelles colonnes servent ──────
    imp = (pd.Series(m.feature_importances_, index=prep.colonnes_)
           .sort_values(ascending=False))
    imp.rename("importance").to_csv(PROCESSED / "importances_v3.csv")
    neuves = ["cluster_id", "taux_cluster_lisse", "taux_commune_lisse",
              "ratio_commune_cluster"]

    base = pd.read_csv(PROCESSED / "baselines.csv").pr_auc.max()
    v1 = pd.read_csv(PROCESSED / "modeles_v1.csv").set_index("modele").pr_auc["XGBoost"]
    v2 = pd.read_csv(PROCESSED / "modeles_v2.csv").pr_auc[0]

    print("\n" + "═" * 62)
    print(f"{'':32s} {'PR-AUC':>9s} {'lift':>8s}")
    print(f"{'meilleure baseline':32s} {base:9.4f} {base / yv.mean():7.1f}×")
    print(f"{'XGBoost v1 (à la main)':32s} {v1:9.4f} {v1 / yv.mean():7.1f}×")
    print(f"{'XGBoost v2 (Optuna)':32s} {v2:9.4f} {v2 / yv.mean():7.1f}×")
    print(f"{'XGBoost v3 (+ clustering)':32s} {ap:9.4f} {ap / yv.mean():7.1f}×")
    print("═" * 62)
    print(f"\ngain du clustering sur le v2 : {100 * (ap / v2 - 1):+.1f} %")
    print(f"gain total sur la baseline   : ×{ap / base:.2f}")

    print(f"\nPoids des 4 colonnes de clustering : "
          f"{100 * imp[neuves].sum():.1f} % de l'importance totale")
    for c in neuves:
        print(f"   {c:24s} {100 * imp[c]:5.1f} %   (rang "
              f"{list(imp.index).index(c) + 1}/{len(imp)})")

    pd.DataFrame([{"modele": f"XGBoost v3 ({METHODE} k={K})", "pr_auc": ap,
                   "lift": ap / yv.mean()}]).to_csv(
        PROCESSED / "modeles_v3.csv", index=False)
    print(f"\n✅ modeles_v3.csv · predictions_val_v3.parquet · importances_v3.csv")


if __name__ == "__main__":
    main()
