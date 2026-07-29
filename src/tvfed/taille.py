"""Étape 19 — le second étage : si ça brûle, quelle surface ?

    python -m tvfed.taille

────────────────────────────────────────────────────────────────────────────
UN MODÈLE À DEUX ÉTAGES
────────────────────────────────────────────────────────────────────────────
    étage 1   P(feu)               le modèle actuel
    étage 2   E(surface | feu)     ce module, entraîné sur les feux SEULS

    surface attendue = P(feu) × E(surface | feu)

C'est la structure classique dite « à obstacle » (*hurdle model*) : un premier
modèle décide s'il se passe quelque chose, un second décide de l'ampleur. La
séparer en deux vaut mieux qu'une régression unique sur 253 M lignes dont
99,98 % valent zéro.

────────────────────────────────────────────────────────────────────────────
⚠️ LA DISTRIBUTION INTERDIT DE PRÉDIRE LA SURFACE BRUTE
────────────────────────────────────────────────────────────────────────────
    médiane      0,119 ha   (1 190 m²)
    p90          3,51 ha
    p99          54,2 ha
    maximum   12 552 ha

73,6 % des feux font moins d'un hectare, et une poignée d'incendies fait
l'essentiel de la surface. Une régression sur la valeur brute serait entièrement
pilotée par ces extrêmes : elle minimiserait l'erreur sur les 12 552 ha en
se trompant d'un facteur 100 sur la médiane.

On régresse donc sur **log(1 + surface en ha)**. L'erreur devient
multiplicative : se tromper d'un facteur 2 coûte pareil à 0,1 ha et à 1 000 ha,
ce qui est le bon comportement pour une grandeur qui s'étale sur cinq ordres.

────────────────────────────────────────────────────────────────────────────
⚠️ CE QUE « SURFACE » VEUT DIRE ICI
────────────────────────────────────────────────────────────────────────────
C'est la surface brûlée **dans cette commune**, pas la taille de l'incendie.
Un feu traversant cinq communes est réparti entre elles. À afficher tel quel
dans l'application, sous peine de faire dire au modèle ce qu'il ne dit pas.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import clustering
from .modele_v3 import K, METHODE
from .modeles import CIBLE, Preparation
from .paths import PROCESSED

AN_FIT = 2017
AN_EVAL = (2018, 2019)
SEUIL_GRAND = 5.0        # hectares — au-delà, un feu qui mobilise vraiment


def main() -> None:
    import json

    from sklearn.metrics import (average_precision_score, mean_absolute_error,
                                 r2_score, roc_auc_score)
    from xgboost import XGBClassifier, XGBRegressor

    params = json.loads(
        (PROCESSED / "best_params_xgb.json").read_text(encoding="utf-8"))
    for k in ("n_estimators", "max_depth", "min_child_weight"):
        params[k] = int(params[k])

    p = clustering.profil()
    sin = clustering.sinistralite()
    cl = clustering.ajuster(p, METHODE, K)
    manq = sorted(set(sin.code_insee) - set(cl.index))
    if manq:
        cl = pd.concat([cl, pd.Series(-1, index=manq, name="cluster_id")])
    taux = clustering.lisser(sin, cl)
    train = clustering.appliquer(
        pd.read_parquet(PROCESSED / "train.parquet"), taux)

    # ── les feux seuls ──────────────────────────────────────────────────
    # Le train est échantillonné sur les NÉGATIFS uniquement : tous les feux
    # y sont, aucun n'a été perdu.
    feux = train[train[CIBLE] == 1].copy()
    feux = feux[feux.surface_m2 > 0]

    # ⚠️ LA CIBLE RESTE HORS DU TABLEAU.
    # `Preparation._features` prend TOUTE colonne non explicitement exclue.
    # Une première version ajoutait `ha` et `cible` au DataFrame : le modèle
    # prédisait alors la cible à partir d'elle-même et sortait R² = 0,994,
    # ROC-AUC = 1,0000. Des scores parfaits sur un problème de feu sont un
    # signal d'alarme, jamais une réussite.
    ha = (feux.surface_m2 / 10_000.0).to_numpy()
    cible = np.log1p(ha)
    an = pd.to_datetime(feux.date).dt.year.to_numpy()

    m_a, m_b = an <= AN_FIT, (an >= AN_EVAL[0]) & (an <= AN_EVAL[1])
    a, b = feux[m_a], feux[m_b]
    ha_a, ha_b = ha[m_a], ha[m_b]
    cible_a, cible_b = cible[m_a], cible[m_b]

    print(f"{len(feux):,} feux avec surface")
    print(f"  ajustement ≤{AN_FIT} : {len(a):>6,}")
    print(f"  évaluation {AN_EVAL[0]}-{AN_EVAL[1]} : {len(b):>6,}")
    print(f"\n  médiane {np.median(ha):.3f} ha · p90 {np.quantile(ha, .9):.2f} · "
          f"p99 {np.quantile(ha, .99):.1f} · max {ha.max():,.0f}")
    print(f"  part de la surface totale due aux 1 % les plus gros : "
          f"{100 * np.sort(ha)[-len(ha) // 100:].sum() / ha.sum():.1f} %")

    prep = Preparation().fit(a)
    assert "surface_m2" not in prep.colonnes_ and "nb_feux" not in prep.colonnes_, \
        "la surface ne doit jamais être une feature de sa propre prédiction"
    Xa = prep.transform(a)
    Xb = prep.transform(b)
    print(f"  {len(prep.colonnes_)} features")

    # ── (1) régression sur le log ───────────────────────────────────────
    t0 = time.time()
    reg = XGBRegressor(**params, tree_method="hist", n_jobs=-1,
                       random_state=42).fit(Xa, cible_a)
    pred = reg.predict(Xb)
    reg.save_model(PROCESSED / "modele_taille.json")

    ha_pred = np.expm1(pred).clip(0)
    ha_vrai = ha_b
    # référence : prédire toujours la médiane du train
    ref = np.full(len(b), np.expm1(np.median(cible_a)))

    print(f"\n{'═' * 68}")
    print("ÉTAGE 2 — RÉGRESSION SUR log(1 + surface)")
    print("═" * 68)
    print(f"{'':28s} {'R² (log)':>10s} {'MAE (log)':>11s} {'MAE (ha)':>11s}")
    print(f"{'médiane du train':28s} {0.0:10.3f} "
          f"{mean_absolute_error(cible_b, np.log1p(ref)):11.3f} "
          f"{mean_absolute_error(ha_vrai, ref):11.2f}")
    print(f"{'modèle':28s} {r2_score(cible_b, pred):10.3f} "
          f"{mean_absolute_error(cible_b, pred):11.3f} "
          f"{mean_absolute_error(ha_vrai, ha_pred):11.2f}")
    print(f"  entraîné en {time.time() - t0:.0f} s")

    # ── (2) et la question qui compte vraiment : est-ce un GROS feu ? ────
    # Un R² sur du log est peu parlant. La question opérationnelle est
    # binaire : faut-il envoyer des moyens lourds ?
    ya = (ha_a > SEUIL_GRAND).astype(int)
    yb = (ha_b > SEUIL_GRAND).astype(int)
    t0 = time.time()
    clf = XGBClassifier(**params, tree_method="hist", eval_metric="aucpr",
                        n_jobs=-1, random_state=42).fit(Xa, ya)
    pg = clf.predict_proba(Xb)[:, 1]
    clf.save_model(PROCESSED / "modele_grand_feu.json")

    print(f"\n{'═' * 68}")
    print(f"ÉTAGE 2 bis — « SERA-CE UN GRAND FEU (> {SEUIL_GRAND:.0f} ha) ? »")
    print("═" * 68)
    print(f"  base : {yb.mean():.1%} des feux dépassent {SEUIL_GRAND:.0f} ha "
          f"({yb.sum():,} sur {len(yb):,})")
    ap = average_precision_score(yb, pg)
    print(f"  PR-AUC {ap:.4f}   lift {ap / yb.mean():.2f}×   "
          f"ROC-AUC {roc_auc_score(yb, pg):.4f}")
    print(f"  (la régression sur le log donne, elle, "
          f"PR-AUC {average_precision_score(yb, pred):.4f})")
    print(f"  entraîné en {time.time() - t0:.0f} s")

    o = np.argsort(-pg)
    print(f"\n  parmi les feux classés les plus menaçants :")
    for part in (0.05, 0.10, 0.25):
        k = int(len(yb) * part)
        print(f"    top {part:>4.0%} : {yb[o][:k].sum() / yb.sum():>5.1%} des "
              f"grands feux capturés, précision {yb[o][:k].mean():.1%}")

    pd.DataFrame([{
        "r2_log": r2_score(cible_b, pred),
        "mae_log": mean_absolute_error(cible_b, pred),
        "mae_ha": mean_absolute_error(ha_vrai, ha_pred),
        "pr_auc_grand": ap, "lift_grand": ap / yb.mean(),
        "seuil_ha": SEUIL_GRAND, "n_ajustement": len(a), "n_evaluation": len(b),
    }]).to_csv(PROCESSED / "modele_taille.csv", index=False)
    print(f"\n✅ modele_taille.json · modele_grand_feu.json · modele_taille.csv")


if __name__ == "__main__":
    main()
