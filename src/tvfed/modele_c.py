"""Étape 18 — le modèle C, physique pur. Le seul déployable en temps réel.

    python -m tvfed.modele_c --split test

────────────────────────────────────────────────────────────────────────────
POURQUOI CE MODÈLE N'EST PAS UN REPLI MAIS UNE NÉCESSITÉ
────────────────────────────────────────────────────────────────────────────
Le modèle A tire **29,3 % de son importance** de l'historique récent des feux
— `feux_commune_7j/30j/90j/365j` et `jours_depuis_dernier_feu`.

Or la BDIFF **ne publie pas l'année en cours** : les données 2026 ne sortiront
qu'au printemps 2027. Pour une prédiction faite aujourd'hui, `feux_commune_7j`
vaudrait donc le nombre de feux d'une semaine de décembre 2025. Pas imprécis :
faux.

C'est un défaut qui ne se voit dans AUCUNE métrique d'entraînement, puisqu'en
validation croisée l'historique est toujours disponible. Il n'apparaît qu'en
pensant au déploiement.

Le modèle C n'utilise que ce qui est connu à l'avance :

    météo         les 9 indices CEMS — prévus par EFFIS à 9 jours
    végétation    CORINE, stable d'un millésime à l'autre
    territoire    relief, superficie, distance à la côte, densité
    calendrier    row-local, calculable pour n'importe quelle date

────────────────────────────────────────────────────────────────────────────
CE QU'IL RETIRE, ET POURQUOI
────────────────────────────────────────────────────────────────────────────
    historique commune (7)   indisponible en temps réel ; et en territoire
                             nouveau il est constant, donc inutile
    lat / lon (2)            elles encodent « le Sud brûle ». Vrai aujourd'hui,
                             et exactement le préjugé à ne pas transporter
                             en 2050
    taux et id du cluster    dérivés de `y`, donc datés. La validation croisée
                             spatiale a montré qu'ils n'apportent que 1,2 %

────────────────────────────────────────────────────────────────────────────
⚠️ CE N'EST PAS UNE SÉLECTION SUR LE TEST
────────────────────────────────────────────────────────────────────────────
A et C ne se départagent PAS sur leur score : ils répondent à deux situations
différentes. A quand l'historique est disponible (analyse rétrospective), C
quand il ne l'est pas (temps réel, territoire nouveau, 2050). Le choix se fait
sur la DISPONIBILITÉ DE LA DONNÉE, pas sur la performance — et c'est ce qui
autorise à mesurer les deux sur le test sans corrompre le protocole.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import clustering, matrices
from .modele_v3 import K, METHODE
from .modeles import CIBLE, Preparation
from .paths import PROCESSED

RETIRE = ["feux_commune_7j", "feux_commune_30j", "feux_commune_90j",
          "feux_commune_365j", "jours_depuis_dernier_feu",
          "taux_commune_lisse", "ratio_commune_cluster",
          "taux_cluster_lisse", "cluster_id", "lat", "lon"]


def main() -> None:
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()

    params = json.loads(
        (PROCESSED / "best_params_xgb.json").read_text(encoding="utf-8"))
    for k in ("n_estimators", "max_depth", "min_child_weight"):
        params[k] = int(params[k])

    # le clustering sert encore à produire les colonnes, qu'on retire ensuite :
    # c'est le prix d'un pipeline commun, et ça garantit que A et C voient
    # exactement la même matrice de départ.
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
    print(f"modèle C : {len(garde)} features "
          f"({len(prep.colonnes_)} − {len(RETIRE)} retirées)")

    Xt = pd.DataFrame(prep.transform(train), columns=prep.colonnes_)[garde]
    t0 = time.time()
    m = XGBClassifier(**params, tree_method="hist", eval_metric="aucpr",
                      n_jobs=-1, random_state=42).fit(Xt, train[CIBLE].to_numpy())
    m.save_model(PROCESSED / "modele_c.json")
    print(f"  entraîné en {time.time() - t0:.0f} s")

    print(f"\névaluation sur {args.split} intégral…")
    # `date` seule ne suffit pas à identifier une ligne : il faut la commune.
    # Sans les deux, ce fichier n'est comparable à aucun autre — la requête
    # d'assemblage n'a pas d'ORDER BY. Cf. `comparer`.
    scores, cibles, cles, n, t0 = [], [], [], 0, time.time()
    for bloc in matrices.parcourir(args.split):
        bloc = clustering.appliquer(bloc, taux)
        X = pd.DataFrame(prep.transform(bloc), columns=prep.colonnes_)[garde]
        scores.append(m.predict_proba(X)[:, 1].astype(np.float32))
        cibles.append(bloc[CIBLE].to_numpy(np.int8))
        cles.append(bloc[["code_insee", "date"]])
        n += len(bloc)
        if n % 10_000_000 < len(bloc):
            print(f"   {n:>12,} lignes   {time.time() - t0:5.0f} s")

    pr, y = np.concatenate(scores), np.concatenate(cibles)
    ap = average_precision_score(y, pr)
    base = y.mean()

    d = pd.concat(cles, ignore_index=True).assign(p_c=pr, y=y)
    dt = d.date.to_numpy().astype("datetime64[D]")
    d.to_parquet(PROCESSED / f"scores_c_{args.split}.parquet", index=False,
                 compression="zstd")

    ref = pd.read_csv(PROCESSED / "resultat_test.csv") if args.split == "test" else None
    print("\n" + "═" * 66)
    print(f"{'':34s} {'PR-AUC':>9s} {'lift':>9s}")
    if ref is not None:
        print(f"{'A · 52 features (historique inclus)':34s} "
              f"{ref.pr_auc[0]:9.4f} {ref.lift[0]:8.1f}×")
    print(f"{'C · physique pur (temps réel)':34s} {ap:9.4f} {ap / base:8.1f}×")
    print("═" * 66)
    if ref is not None:
        print(f"\nécart : {100 * (ap / ref.pr_auc[0] - 1):+.1f} %")
        print("\n→ A reste meilleur quand l'historique est disponible.")
        print("  Mais en production il ne l'est PAS : la BDIFF publie avec plus")
        print("  d'un an de retard. C est donc le modèle réellement déployable,")
        print("  et le seul valide pour 2050.")

    imp = (pd.Series(m.feature_importances_, index=garde)
           .sort_values(ascending=False))
    imp.rename("importance").to_csv(PROCESSED / "importances_c.csv")
    print(f"\nles 8 features les plus utilisées :")
    for f, v in imp.head(8).items():
        print(f"   {f:24s} {100 * v:5.1f} %")

    pd.DataFrame([{"modele": "XGBoost C (physique pur)", "pr_auc": ap,
                   "lift": ap / base, "n_features": len(garde)}]).to_csv(
        PROCESSED / f"modele_c_{args.split}.csv", index=False)
    print(f"\n✅ modele_c.json · scores_c_{args.split}.parquet · importances_c.csv")


if __name__ == "__main__":
    main()
