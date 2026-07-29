"""Étape 17 — la mémoire du FWI suffit-elle, ou faut-il un modèle récurrent ?

    python -m tvfed.test_decalages

────────────────────────────────────────────────────────────────────────────
LA QUESTION
────────────────────────────────────────────────────────────────────────────
Pourquoi pas un LSTM ? L'argument de fond est que **le système FWI EST déjà
un modèle récurrent** :

    FFMC(t) = f(FFMC(t-1), météo du jour)    mémoire ~2-3 jours
    DMC(t)  = f(DMC(t-1),  météo du jour)    mémoire ~15 jours
    DC(t)   = f(DC(t-1),   météo du jour)    mémoire ~50 jours

Ce sont des équations récursives, avec des poids issus de la physique du
combustible plutôt qu'appris. Un LSTM devrait réapprendre depuis les données
ce que cinquante ans de science du feu ont encodé.

Mais un argument n'est pas une mesure. Le test décisif :

    SI la mémoire du FWI suffit, ALORS ajouter des décalages plus longs ne
    doit rien apporter.

S'ils n'apportent rien, la séquence est bien résumée et un LSTM ne trouverait
rien de plus. S'ils apportent, il reste du signal temporel et la question du
récurrent se repose sérieusement.

────────────────────────────────────────────────────────────────────────────
CE QU'ON AJOUTE
────────────────────────────────────────────────────────────────────────────
Calculé directement depuis `fait_meteo` (8,3 M lignes, une par maille et par
jour) plutôt qu'en refaisant la matrice complète — c'est la même information
pour un centième du coût.

    fwi_j2 · fwi_j3 · fwi_j7 · fwi_j14      valeurs décalées
    fwi_moy_7j · fwi_moy_14j · fwi_moy_30j  moyennes glissantes
    fwi_max_7j · fwi_max_30j                maxima glissants
    fwi_pente_7j                            tendance : ça monte ou ça descend

⚠️ TOUS À DÉCALAGE ≥ 1 JOUR. Une moyenne glissante incluant le jour même
serait une fuite : au matin du 15 août on ne connaît pas encore le FWI du
15 août. Les fenêtres s'arrêtent donc la veille.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import clustering, db
from .modele_v3 import K, METHODE
from .modeles import CIBLE, Preparation
from .paths import PROCESSED

AN_FIT = 2017
AN_EVAL = (2018, 2019)

NOUVELLES = ["fwi_j2", "fwi_j3", "fwi_j7", "fwi_j14",
             "fwi_moy_7j", "fwi_moy_14j", "fwi_moy_30j",
             "fwi_max_7j", "fwi_max_30j", "fwi_pente_7j"]


def decalages() -> pd.DataFrame:
    """Décalages et fenêtres glissantes du FWI, par maille et par jour.

    ⚠️ `shift(1)` AVANT toute fenêtre : la fenêtre « 7 derniers jours » du
    15 août doit couvrir du 8 au 14, jamais le 15. Sinon on donne au modèle
    une information qu'il n'aura pas au moment de prédire.
    """
    print("lecture de fait_meteo…")
    with db.connexion() as c:
        m = pd.read_sql(
            "SELECT cell_id, date, fwi FROM fait_meteo ORDER BY cell_id, date", c)
    print(f"  {len(m):,} lignes, {m.cell_id.nunique():,} mailles")

    g = m.groupby("cell_id").fwi
    v = g.shift(1)                      # ← la veille : base de tout le reste
    out = pd.DataFrame({"cell_id": m.cell_id, "date": m.date})
    for n, d in (("fwi_j2", 2), ("fwi_j3", 3), ("fwi_j7", 7), ("fwi_j14", 14)):
        out[n] = g.shift(d)
    dec = m.assign(v=v).groupby("cell_id").v
    for n, f in (("fwi_moy_7j", 7), ("fwi_moy_14j", 14), ("fwi_moy_30j", 30)):
        out[n] = dec.transform(lambda s, f=f: s.rolling(f, min_periods=1).mean())
    for n, f in (("fwi_max_7j", 7), ("fwi_max_30j", 30)):
        out[n] = dec.transform(lambda s, f=f: s.rolling(f, min_periods=1).max())
    # tendance : le FWI monte-t-il ou redescend-il ?
    out["fwi_pente_7j"] = v - g.shift(8)
    return out


def main() -> None:
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    params = json.loads(
        (PROCESSED / "best_params_xgb.json").read_text(encoding="utf-8"))
    for k in ("n_estimators", "max_depth", "min_child_weight"):
        params[k] = int(params[k])

    # ── la matrice actuelle, plus les nouveaux décalages ────────────────
    p = clustering.profil()
    sin = clustering.sinistralite()
    cl = clustering.ajuster(p, METHODE, K)
    manq = sorted(set(sin.code_insee) - set(cl.index))
    if manq:
        cl = pd.concat([cl, pd.Series(-1, index=manq, name="cluster_id")])
    taux = clustering.lisser(sin, cl)
    train = clustering.appliquer(
        pd.read_parquet(PROCESSED / "train.parquet"), taux)

    with db.connexion() as c:
        cells = pd.read_sql(
            "SELECT code_insee, cell_id FROM ref_commune", c).set_index("code_insee")
    train["cell_id"] = train.code_insee.map(cells.cell_id)

    d = decalages()
    d["date"] = pd.to_datetime(d.date)
    train["date"] = pd.to_datetime(train.date)
    avant = len(train)
    train = train.merge(d, on=["cell_id", "date"], how="left")
    assert len(train) == avant, "la jointure a dupliqué des lignes"
    train = train.drop(columns=["cell_id"])
    print(f"  {len(NOUVELLES)} colonnes ajoutées, "
          f"{train[NOUVELLES].isna().mean().mean():.2%} de manquants "
          f"(début de série)")

    an = pd.to_datetime(train.date).dt.year
    a, b = train[an <= AN_FIT], train[an.between(*AN_EVAL)]

    print(f"\najustement ≤{AN_FIT} : {len(a):>7,} lignes")
    print(f"évaluation {AN_EVAL[0]}-{AN_EVAL[1]} : {len(b):>7,} lignes\n")

    res = []
    for nom, retire in (("52 features (actuel)", NOUVELLES),
                        ("62 features (+ décalages)", [])):
        t0 = time.time()
        prep = Preparation().fit(a)
        garde = [c for c in prep.colonnes_ if c not in retire]
        Xa = pd.DataFrame(prep.transform(a), columns=prep.colonnes_)[garde]
        Xb = pd.DataFrame(prep.transform(b), columns=prep.colonnes_)[garde]
        m = XGBClassifier(**params, tree_method="hist", eval_metric="aucpr",
                          device="cuda", n_jobs=-1, random_state=42).fit(
            Xa, a[CIBLE].to_numpy())
        ap = average_precision_score(b[CIBLE].to_numpy(),
                                     m.predict_proba(Xb)[:, 1])
        imp = pd.Series(m.feature_importances_, index=garde)
        res.append({"jeu": nom, "n_features": len(garde), "pr_auc": ap,
                    "poids_nouvelles": float(imp.reindex(NOUVELLES).sum())})
        print(f"  {nom:26s} {len(garde):>3} features   PR-AUC {ap:.4f}   "
              f"({time.time() - t0:.0f} s)")

    R = pd.DataFrame(res)
    R.to_csv(PROCESSED / "test_decalages.csv", index=False)
    gain = 100 * (R.pr_auc.iloc[1] / R.pr_auc.iloc[0] - 1)

    print(f"\n{'═' * 70}")
    print(f"Gain des 10 décalages supplémentaires : {gain:+.2f} %")
    print(f"Poids qu'ils prennent dans le modèle  : "
          f"{100 * R.poids_nouvelles.iloc[1]:.1f} %")
    print("═" * 70)
    print()
    if abs(gain) < 1.0:
        print("→ NÉGLIGEABLE. La mémoire longue est déjà dans les indices CEMS :")
        print("  DC porte ~50 jours, DMC ~15, FFMC ~3. Rediriger cette mémoire")
        print("  vers des décalages explicites n'ajoute rien, et un LSTM devrait")
        print("  la réapprendre depuis zéro pour arriver au même point.")
        print("  → PAS de modèle récurrent. L'argument est maintenant mesuré.")
    else:
        print("→ Le gain n'est PAS négligeable : il reste du signal temporel que")
        print("  les indices ne capturent pas. La question du récurrent se repose.")


if __name__ == "__main__":
    main()
