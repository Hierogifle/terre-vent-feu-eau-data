"""Étape 10 — le MLP, troisième famille exigée par l'énoncé.

    python -m tvfed.reseau --essais 40        # recherche Optuna
    python -m tvfed.reseau --final            # entraîne et évalue le retenu

────────────────────────────────────────────────────────────────────────────
POURQUOI PYTORCH ET PAS sklearn.MLPClassifier
────────────────────────────────────────────────────────────────────────────
`MLPClassifier` n'a PAS de dropout : sa seule régularisation est un L2
(`alpha`). L'énoncé demande explicitement le dropout, et les deux ne sont pas
le même mécanisme :

    dropout   éteint des NEURONES au hasard    → force la redondance
    L2        pénalise les POIDS trop grands   → force des poids petits

PyTorch donne le dropout réel, et accessoirement le GPU.

────────────────────────────────────────────────────────────────────────────
CE QU'IL FAUT ATTENDRE DE CE MODÈLE
────────────────────────────────────────────────────────────────────────────
Probablement moins bon que XGBoost, pour trois raisons propres au problème :

  · LES SEUILS. Un arbre encode « FWI > 21,3 » en un nœud. Un réseau doit
    l'approcher par une somme de sigmoïdes. Or les seuils EFFIS sont
    structurants ici.
  · LES ZÉROS. 80 % des communes ont un historique à zéro. Un arbre isole ça
    d'un découpage ; un réseau doit l'apprendre dans ses poids.
  · LE DÉSÉQUILIBRE. 9 % de positifs après downsampling, 0,024 % en réalité.

Si c'est le cas, **c'est un résultat, pas un échec** : on aura mesuré l'écart
entre familles au lieu de le supposer.

────────────────────────────────────────────────────────────────────────────
DEUX OBLIGATIONS QUE LES ARBRES N'AVAIENT PAS
────────────────────────────────────────────────────────────────────────────
1. STANDARDISATION. Un arbre se moque des échelles, il compare des seuils.
   Un réseau additionne des produits : laisser `log_population` (~8) à côté
   de `part_maquis` (~0,02) rend la descente de gradient ingérable.
   ⚠️ Le `StandardScaler` a un `.fit()` — il s'apprend sur le TRAIN seul.

2. VALIDATION INTERNE POUR L'ARRÊT. Un réseau sur-apprend sans limite si on
   le laisse tourner. On coupe sur un découpage interne au train
   (2006-2017 / 2018-2019), jamais sur la validation 2020-2022.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import clustering, matrices
from .modeles import CIBLE, Preparation
from .paths import PROCESSED

AN_FIT = 2017
AN_EVAL = (2018, 2019)
GRAINE = 42


def _appareil():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════════
#  le réseau
# ════════════════════════════════════════════════════════════════════════
def construire(n_entrees: int, largeur: int, n_couches: int, dropout: float):
    """Empilement Linéaire → BatchNorm → ReLU → Dropout, en entonnoir.

    ⚠️ L'ORDRE COMPTE. BatchNorm AVANT le dropout : l'inverse ferait calculer
    les statistiques de normalisation sur des activations déjà trouées, donc
    sur une variance artificiellement gonflée, et elles seraient fausses au
    moment de la prédiction où le dropout est désactivé.

    Largeur divisée par deux à chaque couche : le réseau compresse
    progressivement 52 features vers une décision unique.
    """
    import torch.nn as nn

    couches, dedans = [], n_entrees
    for i in range(n_couches):
        dehors = max(16, largeur // (2 ** i))
        couches += [nn.Linear(dedans, dehors),
                    nn.BatchNorm1d(dehors),
                    nn.ReLU(),
                    nn.Dropout(dropout)]     # ← le dropout demandé par l'énoncé
        dedans = dehors
    couches.append(nn.Linear(dedans, 1))     # logit brut, la sigmoïde est dans la perte
    return nn.Sequential(*couches)


def entrainer(Xa, ya, Xb, yb, *, largeur, n_couches, dropout, lr, poids_l2,
              taille_lot, epoques_max=120, patience=10, bavard=False,
              rendre_epoques=False):
    """Entraîne avec arrêt précoce sur (Xb, yb).

    Retourne (modèle, meilleure PR-AUC), et le numéro d'époque retenu si
    `rendre_epoques` — c'est lui qui pilote le réentraînement sur tout le train.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import average_precision_score

    torch.manual_seed(GRAINE)
    dev = _appareil()

    # tout tient en VRAM : 368 k × 52 float32 = 77 Mo
    Xa_t = torch.tensor(Xa, dtype=torch.float32, device=dev)
    ya_t = torch.tensor(ya, dtype=torch.float32, device=dev).unsqueeze(1)
    Xb_t = torch.tensor(Xb, dtype=torch.float32, device=dev)

    m = construire(Xa.shape[1], largeur, n_couches, dropout).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=poids_l2)
    perte = nn.BCEWithLogitsLoss()

    meilleur, meilleur_etat, sans_progres, meilleure_epoque = -1.0, None, 0, 1
    n = len(Xa_t)
    for ep in range(epoques_max):
        m.train()                       # ← ACTIVE le dropout
        ordre = torch.randperm(n, device=dev)
        for d in range(0, n, taille_lot):
            i = ordre[d:d + taille_lot]
            opt.zero_grad(set_to_none=True)
            perte(m(Xa_t[i]), ya_t[i]).backward()
            opt.step()

        m.eval()                        # ← DÉSACTIVE le dropout : à l'évaluation
        with torch.no_grad():           #   tous les neurones travaillent
            p = torch.sigmoid(m(Xb_t)).squeeze(1).cpu().numpy()
        ap = average_precision_score(yb, p)

        if ap > meilleur:
            meilleur, sans_progres, meilleure_epoque = ap, 0, ep + 1
            meilleur_etat = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            sans_progres += 1
            if sans_progres >= patience:
                break
        if bavard and ep % 10 == 0:
            print(f"      époque {ep:>3}  PR-AUC {ap:.4f}  (meilleure {meilleur:.4f})")

    m.load_state_dict(meilleur_etat)
    return (m, meilleur, meilleure_epoque) if rendre_epoques else (m, meilleur)


def entrainer_fixe(X, y, *, epoques, largeur, n_couches, dropout, lr, poids_l2,
                   taille_lot):
    """Entraîne sur TOUT le jeu fourni, pour un nombre d'époques imposé.

    Pas d'arrêt précoce ici — il n'y a rien à surveiller, et c'est voulu :
    le nombre d'époques a déjà été choisi sur le découpage interne. Cette
    passe sert uniquement à faire profiter le réseau des deux années que
    l'arrêt précoce lui avait retirées.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(GRAINE)
    dev = _appareil()
    X_t = torch.tensor(X, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(1)

    m = construire(X.shape[1], largeur, n_couches, dropout).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=poids_l2)
    perte = nn.BCEWithLogitsLoss()

    n = len(X_t)
    for _ in range(epoques):
        m.train()
        ordre = torch.randperm(n, device=dev)
        for d in range(0, n, taille_lot):
            i = ordre[d:d + taille_lot]
            opt.zero_grad(set_to_none=True)
            perte(m(X_t[i]), y_t[i]).backward()
            opt.step()
    return m


# ════════════════════════════════════════════════════════════════════════
#  données
# ════════════════════════════════════════════════════════════════════════
def _jeux():
    """Train enrichi du clustering, découpé, standardisé. Rien de val/test ici."""
    from sklearn.preprocessing import StandardScaler

    from .modele_v3 import K, METHODE

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

    an = pd.to_datetime(train.date).dt.year
    a, b = train[an <= AN_FIT], train[an.between(*AN_EVAL)]
    # ⚠️ le scaler s'ajuste sur la partie AJUSTEMENT seule
    sc = StandardScaler().fit(prep.transform(a))
    return (sc.transform(prep.transform(a)), a[CIBLE].to_numpy(float),
            sc.transform(prep.transform(b)), b[CIBLE].to_numpy(float),
            train, prep, sc, taux)


# ════════════════════════════════════════════════════════════════════════
def recherche(essais: int) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Xa, ya, Xb, yb, *_ = _jeux()
    print(f"appareil : {_appareil()}")
    print(f"  ajustement ≤{AN_FIT} : {len(ya):>7,} lignes, {int(ya.sum()):>6,} positifs")
    print(f"  évaluation {AN_EVAL[0]}-{AN_EVAL[1]} : {len(yb):>7,} lignes, "
          f"{int(yb.sum()):>6,} positifs\n")

    def objectif(e):
        return entrainer(
            Xa, ya, Xb, yb,
            largeur=e.suggest_categorical("largeur", [64, 128, 256, 512]),
            n_couches=e.suggest_int("n_couches", 1, 4),
            dropout=e.suggest_float("dropout", 0.0, 0.6),   # ← le paramètre demandé
            lr=e.suggest_float("lr", 1e-4, 1e-2, log=True),
            poids_l2=e.suggest_float("poids_l2", 1e-7, 1e-2, log=True),
            taille_lot=e.suggest_categorical("taille_lot", [256, 512, 1024, 2048]),
        )[1]

    etude = optuna.create_study(
        direction="maximize", study_name="tvfed_mlp",
        storage=f"sqlite:///{(PROCESSED / 'optuna.db').as_posix()}",
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=GRAINE))

    fait, t0 = [0], time.time()

    def rappel(st, tr):
        fait[0] += 1
        if tr.value is not None and tr.value >= st.best_value:
            print(f"  essai {fait[0]:>3}/{essais}  PR-AUC {tr.value:.4f}  ← meilleur "
                  f"(dropout {tr.params['dropout']:.2f}, "
                  f"{tr.params['n_couches']}×{tr.params['largeur']})")

    etude.optimize(objectif, n_trials=essais, callbacks=[rappel])
    print(f"\n{'═' * 62}")
    print(f"meilleure PR-AUC interne : {etude.best_value:.4f}  "
          f"({time.time() - t0:.0f} s)")
    for k, v in etude.best_params.items():
        print(f"   {k:14s} {v}")
    (PROCESSED / "best_params_mlp.json").write_text(
        json.dumps(etude.best_params, indent=2), encoding="utf-8")
    print("\n✅ best_params_mlp.json")
    return etude.best_params


def final() -> None:
    """Réentraîne sur tout le train et évalue sur la validation intégrale.

    ⚠️ DEUX PIÈGES, tous deux évités ici.

    1. LE SCALER. Le réseau apprend sur des données centrées-réduites par un
       `StandardScaler`. Lui en présenter un AUTRE à la prédiction — même
       ajusté sur des données légitimes — décale toutes les entrées et rend
       les sorties silencieusement fausses. On réutilise donc exactement
       l'objet qui a servi à l'entraînement.

    2. LE VOLUME D'ENTRAÎNEMENT. L'arrêt précoce impose de garder 2018-2019
       hors ajustement, donc le réseau n'apprend que sur ≤2017 — alors que
       XGBoost v3 a vu tout le train. La comparaison serait biaisée contre le
       réseau. On repasse donc sur le train COMPLET, pour le nombre d'époques
       que l'arrêt précoce a désigné comme optimal.
    """
    import torch
    from sklearn.metrics import average_precision_score

    params = json.loads(
        (PROCESSED / "best_params_mlp.json").read_text(encoding="utf-8"))
    print("Hyperparamètres retenus :")
    for k, v in params.items():
        print(f"   {k:14s} {v}")

    Xa, ya, Xb, yb, train, prep, sc, taux = _jeux()

    # passe 1 — combien d'époques ? mesuré sur le découpage interne
    print("\npasse 1 : recherche du nombre d'époques (arrêt précoce)…")
    _, interne, n_epoques = entrainer(Xa, ya, Xb, yb, **params, bavard=True,
                                      rendre_epoques=True)
    print(f"  PR-AUC interne {interne:.4f} à l'époque {n_epoques}")

    # passe 2 — réentraînement sur le train COMPLET, sans arrêt précoce
    print(f"\npasse 2 : réentraînement sur le train complet, {n_epoques} époques…")
    X_tout = sc.transform(prep.transform(train))
    y_tout = train[CIBLE].to_numpy(float)
    m = entrainer_fixe(X_tout, y_tout, epoques=n_epoques, **params)
    print(f"  {len(y_tout):,} lignes, {int(y_tout.sum()):,} positifs")

    dev = _appareil()
    m.eval()

    print("\névaluation sur la validation intégrale…")
    scores, cibles, n, t0 = [], [], 0, time.time()
    for bloc in matrices.parcourir("val"):
        bloc = clustering.appliquer(bloc, taux)
        X = torch.tensor(sc.transform(prep.transform(bloc)),
                         dtype=torch.float32, device=dev)
        with torch.no_grad():
            scores.append(torch.sigmoid(m(X)).squeeze(1).cpu().numpy().astype(np.float32))
        cibles.append(bloc[CIBLE].to_numpy(np.int8))
        n += len(bloc)
        if n % 10_000_000 < len(bloc):
            print(f"   {n:>12,} lignes   {time.time() - t0:5.0f} s")

    p, yv = np.concatenate(scores), np.concatenate(cibles)
    ap = average_precision_score(yv, p)
    pd.DataFrame({"p_mlp": p, "y": yv}).to_parquet(
        PROCESSED / "predictions_val_mlp.parquet", index=False, compression="zstd")

    base = pd.read_csv(PROCESSED / "baselines.csv").pr_auc.max()
    v3 = pd.read_csv(PROCESSED / "modeles_v3.csv").pr_auc[0]
    print("\n" + "═" * 62)
    print(f"{'':28s} {'PR-AUC':>9s} {'lift':>8s}")
    print(f"{'meilleure baseline':28s} {base:9.4f} {base / yv.mean():7.1f}×")
    print(f"{'XGBoost v3':28s} {v3:9.4f} {v3 / yv.mean():7.1f}×")
    print(f"{'MLP (dropout)':28s} {ap:9.4f} {ap / yv.mean():7.1f}×")
    print("═" * 62)
    print(f"\nécart au XGBoost v3 : {100 * (ap / v3 - 1):+.1f} %")

    pd.DataFrame([{"modele": "MLP (PyTorch, dropout)", "pr_auc": ap,
                   "lift": ap / yv.mean()}]).to_csv(
        PROCESSED / "modeles_mlp.csv", index=False)
    torch.save(m.state_dict(), PROCESSED / "modele_mlp.pt")
    print("\n✅ modeles_mlp.csv · predictions_val_mlp.parquet · modele_mlp.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--essais", type=int, default=40)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    if args.final:
        final()
    else:
        recherche(args.essais)


if __name__ == "__main__":
    main()
