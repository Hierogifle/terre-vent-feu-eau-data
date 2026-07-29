"""Étape 12 — un passage unique qui produit des prédictions DATÉES.

    python -m tvfed.scores --split val
    python -m tvfed.scores --split test        # ⚠️ une seule fois, tout gelé

────────────────────────────────────────────────────────────────────────────
POURQUOI CE MODULE EXISTE
────────────────────────────────────────────────────────────────────────────
Les fichiers `predictions_*_v3.parquet` ne contiennent que (score, y). C'était
suffisant pour calculer une PR-AUC — qui ne dépend que de l'ordre — mais pas
pour deux choses dont on a maintenant besoin :

  · CALIBRER HONNÊTEMENT. Ajuster la correction sur la validation puis la
    mesurer sur la même validation est circulaire : elle y paraîtra parfaite
    par construction. Il faut un découpage, et le seul qui ait du sens ici est
    TEMPOREL (2020-2021 pour ajuster, 2022 pour juger) — donc il faut la date.

  · ALIMENTER L'APPLICATION. Afficher une carte suppose de savoir à quelle
    commune et à quel jour se rapporte chaque score.

⚠️ VÉRIFIÉ, ET C'ÉTAIT FAUX : on aurait pu croire que les lignes sortent de
PostgreSQL année par année, la table étant partitionnée. Contrôle fait en
comparant les feux par tranche aux comptages de la base — 2 944 trouvés contre
2 688 attendus pour 2020. L'ordre n'est PAS chronologique. D'où ce passage.

────────────────────────────────────────────────────────────────────────────
LES TROIS MODÈLES EN UN SEUL PARCOURS
────────────────────────────────────────────────────────────────────────────
Le coût dominant n'est pas le modèle mais la lecture PostgreSQL et l'assemblage
des features : ~20 min sur 38 M lignes, contre ~80 s de prédiction XGBoost.
Autant scorer les trois familles pendant qu'on tient les données.
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


def _charger():
    """Les trois modèles, plus la préparation et les taux du clustering."""
    import torch
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    from .reseau import construire

    params = json.loads(
        (PROCESSED / "best_params_xgb.json").read_text(encoding="utf-8"))
    for k in ("n_estimators", "max_depth", "min_child_weight"):
        params[k] = int(params[k])

    print(f"clustering {METHODE} k={K}…")
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

    modeles = {}
    for nom, fichier in (("xgb_v3", "modele_v3.json"), ("dart", "modele_dart.json")):
        f = PROCESSED / fichier
        if f.exists():
            m = XGBClassifier()
            m.load_model(f)
            modeles[nom] = m
        else:
            print(f"  ⚠️ {fichier} absent — {nom} ignoré")

    # le réseau : il faut reconstruire l'architecture avant de charger les poids
    f = PROCESSED / "modele_mlp.pt"
    sc = None
    if f.exists():
        mp = json.loads(
            (PROCESSED / "best_params_mlp.json").read_text(encoding="utf-8"))
        # ⚠️ le MÊME scaler qu'à l'entraînement : ajusté sur l'ajustement
        # interne (≤2017), pas refabriqué ici. Un autre décalerait toutes les
        # entrées du réseau sans qu'aucune erreur ne le signale.
        an = pd.to_datetime(train.date).dt.year
        prep_i = Preparation().fit(train[an <= 2017])
        sc = StandardScaler().fit(prep_i.transform(train[an <= 2017]))

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        r = construire(len(prep.colonnes_), mp["largeur"], mp["n_couches"],
                       mp["dropout"])
        r.load_state_dict(torch.load(f, map_location="cpu"))
        # ⚠️ le réseau doit vivre sur le MÊME appareil que les données, sinon
        # `F.linear` lève « Expected all tensors to be on the same device ».
        # Déplacé une fois ici plutôt qu'à chaque bloc.
        r.to(dev)
        r.eval()          # ← dropout désactivé : à la prédiction, tout le
        modeles["mlp"] = r  #   réseau travaille
    else:
        print("  ⚠️ modele_mlp.pt absent — mlp ignoré")

    print(f"  modèles chargés : {', '.join(modeles)}")
    return modeles, prep, sc, taux


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    args = ap.parse_args()

    if args.split == "test":
        print("=" * 66)
        print("⚠️  ÉVALUATION SUR LE TEST — 2023-2025, jamais touché jusqu'ici.")
        print("    Tout doit être gelé : modèle, hyperparamètres, clustering,")
        print("    calibration, seuils. Si quoi que ce soit est retouché après")
        print("    avoir vu ce résultat, le test cesse d'être un juge.")
        print("=" * 66 + "\n")

    modeles, prep, sc, taux = _charger()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ⚠️ MÉMOIRE. Accumuler 38 M lignes en DataFrames pandas ferait exploser
    # la RAM : `code_insee` en objets Python coûte ~60 octets par valeur, soit
    # 2 à 3 Go pour cette seule colonne, sur une machine qui n'a que ~1 Go de
    # libre. On accumule donc des tableaux numpy à dtype fixe :
    #
    #     code_insee  'S5'      5 octets   →  190 Mo
    #     date        int16     2 octets   →   76 Mo   (jours depuis l'origine)
    #     y           int8      1 octet    →   38 Mo
    #     3 scores    float32   4 octets   →  456 Mo
    #
    # ~800 Mo au lieu de 3-4 Go, pour exactement la même information.
    ORIGINE = np.datetime64("2006-01-01", "D")
    col = {k: [] for k in ["code_insee", "date", "y", *modeles]}
    n, t0 = 0, time.time()

    for bloc in matrices.parcourir(args.split):
        bloc = clustering.appliquer(bloc, taux)
        X = prep.transform(bloc)
        col["code_insee"].append(bloc["code_insee"].to_numpy().astype("S5"))
        col["date"].append(
            (bloc["date"].to_numpy().astype("datetime64[D]") - ORIGINE
             ).astype(np.int16))
        col["y"].append(bloc[CIBLE].to_numpy(np.int8))
        for nom, m in modeles.items():
            if nom == "mlp":
                t = torch.tensor(sc.transform(X), dtype=torch.float32, device=dev)
                with torch.no_grad():
                    col[nom].append(
                        torch.sigmoid(m(t)).squeeze(1).cpu().numpy().astype(np.float32))
            else:
                col[nom].append(m.predict_proba(X)[:, 1].astype(np.float32))
        n += len(bloc)
        if n % 10_000_000 < len(bloc):
            print(f"   {n:>12,} lignes   {time.time() - t0:5.0f} s")

    D = pd.DataFrame({k: np.concatenate(v) for k, v in col.items()})
    del col
    D["code_insee"] = D.code_insee.str.decode("ascii")
    D["date"] = ORIGINE + D.date.to_numpy().astype("timedelta64[D]")
    dest = PROCESSED / f"scores_{args.split}.parquet"
    D.to_parquet(dest, index=False, compression="zstd")

    # ── contrôle : les comptages par année doivent coller à la base ─────
    an = pd.to_datetime(D.date).dt.year
    print(f"\n{len(D):,} lignes, {D.y.sum():,} feux ({D.y.mean():.4%})")
    print(f"\n{'année':>6s} {'lignes':>12s} {'feux':>8s} {'taux':>9s}")
    for a, g in D.groupby(an):
        print(f"{a:>6} {len(g):>12,} {g.y.sum():>8,} {g.y.mean():>8.4%}")

    from sklearn.metrics import average_precision_score
    print(f"\n{'modèle':12s} {'PR-AUC':>9s} {'lift':>8s}")
    for nom in modeles:
        a = average_precision_score(D.y, D[nom])
        print(f"{nom:12s} {a:9.4f} {a / D.y.mean():7.1f}×")

    print(f"\n✅ {dest.name}")


if __name__ == "__main__":
    main()
