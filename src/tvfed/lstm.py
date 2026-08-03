"""Étape 24 — le LSTM : la séquence météo brute vaut-elle mieux que les indices ?

    python -m tvfed.lstm --essais 25      # recherche Optuna
    python -m tvfed.lstm --final          # entraîne le retenu et évalue

────────────────────────────────────────────────────────────────────────────
LA QUESTION, ET POURQUOI ELLE N'EST PAS ENCORE TRANCHÉE
────────────────────────────────────────────────────────────────────────────
Le projet a jusqu'ici argumenté contre le récurrent de façon INDIRECTE :
ajouter dix retards explicites du FWI (j-2, j-3, j-7, j-14, moyennes et maxima
glissants) n'apporte que **+0,02 %**. D'où la conclusion que la séquence est
déjà résumée par les indices CEMS, qui sont eux-mêmes des équations
récursives :

    FFMC(t) = f(FFMC(t-1), météo du jour)    mémoire ~3 jours
    DMC(t)  = f(DMC(t-1),  météo du jour)    mémoire ~15 jours
    DC(t)   = f(DC(t-1),   météo du jour)    mémoire ~50 jours

C'est une bonne preuve. **Ce n'en est pas une définitive.** Un LSTM apprend son
propre résumé de la séquence : il pourrait capter une FORME, un enchaînement,
que des retards figés ne décrivent pas. Tant qu'on ne l'a pas construit, on ne
peut pas l'exclure.

────────────────────────────────────────────────────────────────────────────
L'ARCHITECTURE
────────────────────────────────────────────────────────────────────────────
    séquence   30 jours × 10 indices météo   →  LSTM  →  état final
    statique   végétation, relief, calendrier →  (tel quel)
                            ↓
                    concaténation → couches denses → probabilité

Le LSTM ne voit QUE la météo : c'est là que la séquence a du sens. La
végétation et le relief ne varient pas d'un jour à l'autre, les passer dans un
réseau récurrent n'aurait aucun intérêt et coûterait 30 fois leur poids.

⚠️ LE SEUL VRAI TEST est la comparaison à XGBoost v3 (0,0177) et au MLP
(0,0173) sur la MÊME validation, avec bootstrap apparié. Un LSTM qui fait
0,0175 n'est pas « moins bon » : il est indiscernable, et c'est ce qu'il faut
dire.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import clustering, db, matrices
from .modeles import CIBLE, Preparation
from .paths import PROCESSED

FENETRE = 30           # jours d'historique météo donnés au LSTM
GRAINE = 42
AN_FIT = 2017
AN_EVAL = (2018, 2019)

INDICES = ["fwi", "ffmc", "dmc", "dc", "bui", "isi", "kbdi", "erc"]

# Ce que le LSTM NE voit pas dans sa séquence, et qui lui est donné à part.
# On retire les features météo du jour (elles sont dans la séquence, en
# dernière position) et tout ce qui dérive de `y` — on veut mesurer ce que la
# séquence météo apporte, pas refaire le modèle A.
STATIQUES = [
    "part_foret", "part_feuillus", "part_coniferes", "part_melangees",
    "part_landes", "part_maquis", "part_veg_mutation", "part_veg_clairsemee",
    "part_combustible", "part_agricole", "part_artificialise", "clc_millesime",
    "log_population", "log_densite", "log_superficie", "altitude_moy",
    "amplitude_altitude", "grille_densite", "distance_cote_km",
    "doy", "mois", "jour_semaine", "est_weekend", "est_ferie",
    "est_14_juillet", "est_15_aout", "sin_doy", "cos_doy", "sin_mois",
    "cos_mois",
]


def _appareil():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════════
#  le cube météo : [maille, jour, indice]
# ════════════════════════════════════════════════════════════════════════
def cube_meteo() -> tuple[np.ndarray, dict, np.datetime64]:
    """Toute la météo 2006-2025 dans un tableau dense.

    ⚠️ C'EST CE QUI REND LE LSTM POSSIBLE. Matérialiser les séquences ligne
    par ligne coûterait 38 M × 30 × 10 × 4 octets = **45 Go** pour la seule
    validation. Ici on garde un cube de 1 131 × 7 305 × 8 = 264 Mo, et on
    découpe les fenêtres à la volée par simple indexation.
    """
    with db.connexion() as c:
        m = pd.read_sql(f"""
            SELECT cell_id, date, {', '.join(INDICES)}
            FROM fait_meteo
            WHERE date BETWEEN '2005-12-01' AND '2025-12-31'
            ORDER BY cell_id, date""", c)

    cells = np.sort(m.cell_id.unique())
    idx_cell = {int(c): i for i, c in enumerate(cells)}
    jours = np.sort(m.date.unique())
    origine = jours[0]
    n_j = len(jours)

    cube = np.zeros((len(cells), n_j, len(INDICES)), dtype=np.float32)
    ci = m.cell_id.map(idx_cell).to_numpy()
    ji = (m.date.to_numpy().astype("datetime64[D]")
          - np.datetime64(origine, "D")).astype(int)
    cube[ci, ji] = m[INDICES].to_numpy(np.float32)
    return cube, idx_cell, np.datetime64(origine, "D")


def sequences(codes, dates, cube, idx_cell, origine, cell_de) -> np.ndarray:
    """Les FENETRE derniers jours de météo, pour chaque ligne demandée.

    ⚠️ FENÊTRE À BORNE HAUTE STRICTE : elle s'arrête la VEILLE. Au matin du
    15 août on ne connaît pas encore le FWI du 15 août — l'inclure serait la
    fuite classique, et elle gonflerait le score sans que rien ne le signale.
    """
    ci = np.array([idx_cell[cell_de[c]] for c in codes])
    ji = (np.asarray(dates, dtype="datetime64[D]") - origine).astype(int)
    # décalages -FENETRE .. -1  →  la veille est le dernier élément
    dec = np.arange(-FENETRE, 0)
    return cube[ci[:, None], ji[:, None] + dec[None, :], :]


# ════════════════════════════════════════════════════════════════════════
#  le réseau
# ════════════════════════════════════════════════════════════════════════
def construire(n_stat: int, cache: int, couches: int, dropout: float,
               tete: int):
    import torch
    import torch.nn as nn

    class Reseau(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(len(INDICES), cache, num_layers=couches,
                                batch_first=True,
                                dropout=dropout if couches > 1 else 0.0)
            self.tete = nn.Sequential(
                nn.Linear(cache + n_stat, tete),
                nn.BatchNorm1d(tete), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(tete, tete // 2),
                nn.BatchNorm1d(tete // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(tete // 2, 1))

        def forward(self, seq, stat):
            # on ne garde que l'état du DERNIER pas : c'est le résumé que le
            # LSTM a construit de toute la séquence
            _, (h, _) = self.lstm(seq)
            return self.tete(torch.cat([h[-1], stat], dim=1))

    return Reseau()


def entrainer(Xs_a, Xt_a, ya, Xs_b, Xt_b, yb, *, cache, couches, dropout,
              tete, lr, poids_l2, taille_lot, epoques_max=40, patience=6,
              bavard=False):
    import torch
    import torch.nn as nn
    from sklearn.metrics import average_precision_score

    torch.manual_seed(GRAINE)
    dev = _appareil()
    ta = lambda x, t=torch.float32: torch.tensor(x, dtype=t, device=dev)
    Sa, Ta, Ya = ta(Xs_a), ta(Xt_a), ta(ya).unsqueeze(1)
    Sb, Tb = ta(Xs_b), ta(Xt_b)

    m = construire(Xt_a.shape[1], cache, couches, dropout, tete).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=poids_l2)
    perte = nn.BCEWithLogitsLoss()

    meilleur, etat, sans, ep_best = -1.0, None, 0, 1
    n = len(Sa)
    for ep in range(epoques_max):
        m.train()
        ordre = torch.randperm(n, device=dev)
        for d in range(0, n, taille_lot):
            i = ordre[d:d + taille_lot]
            opt.zero_grad(set_to_none=True)
            perte(m(Sa[i], Ta[i]), Ya[i]).backward()
            opt.step()

        # ⚠️ ÉVALUATION PAR LOTS, MÊME SANS GRADIENT.
        # Un LSTM garde les activations des 30 pas pour toute la séquence :
        # évaluer 51 916 séquences d'un coup demandait 15,85 Gio sur une carte
        # qui en a 8. Le `no_grad` ne suffit pas — c'est le passage avant qui
        # est volumineux, pas la rétropropagation.
        m.eval()
        with torch.no_grad():
            p = np.concatenate([
                torch.sigmoid(m(Sb[i:i + 4096], Tb[i:i + 4096])
                              ).squeeze(1).cpu().numpy()
                for i in range(0, len(Sb), 4096)])
        ap = average_precision_score(yb, p)
        if ap > meilleur:
            meilleur, sans, ep_best = ap, 0, ep + 1
            etat = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            sans += 1
            if sans >= patience:
                break
        if bavard:
            print(f"      époque {ep + 1:>2}  PR-AUC {ap:.4f}  "
                  f"(meilleure {meilleur:.4f})")
    m.load_state_dict(etat)
    return m, meilleur, ep_best


# ════════════════════════════════════════════════════════════════════════
def _preparer():
    """Train enrichi, séquences, et les deux normalisations."""
    from sklearn.preprocessing import StandardScaler

    from .modele_v3 import K, METHODE

    print("clustering et matrice…")
    p = clustering.profil()
    sin = clustering.sinistralite()
    cl = clustering.ajuster(p, METHODE, K)
    manq = sorted(set(sin.code_insee) - set(cl.index))
    if manq:
        cl = pd.concat([cl, pd.Series(-1, index=manq, name="cluster_id")])
    taux = clustering.lisser(sin, cl)
    train = clustering.appliquer(
        pd.read_parquet(PROCESSED / "train.parquet"), taux)
    train["date"] = pd.to_datetime(train.date)

    print("cube météo…")
    t0 = time.time()
    cube, idx_cell, origine = cube_meteo()
    print(f"  {cube.shape} — {cube.nbytes / 1e6:.0f} Mo, {time.time() - t0:.0f} s")

    with db.connexion() as c:
        cell_de = pd.read_sql(
            "SELECT code_insee, cell_id FROM ref_commune", c
        ).set_index("code_insee").cell_id.to_dict()

    an = train.date.dt.year
    a, b = train[an <= AN_FIT], train[an.between(*AN_EVAL)]
    print(f"  ajustement ≤{AN_FIT} : {len(a):>7,} · évaluation "
          f"{AN_EVAL[0]}-{AN_EVAL[1]} : {len(b):>7,}")

    t0 = time.time()
    Sa = sequences(a.code_insee.to_numpy(), a.date.to_numpy(), cube,
                   idx_cell, origine, cell_de)
    Sb = sequences(b.code_insee.to_numpy(), b.date.to_numpy(), cube,
                   idx_cell, origine, cell_de)
    print(f"  séquences {Sa.shape} + {Sb.shape} en {time.time() - t0:.0f} s")

    # ⚠️ les deux normalisations s'apprennent sur l'AJUSTEMENT seul
    sc_seq = StandardScaler().fit(Sa.reshape(-1, len(INDICES)))
    Sa = sc_seq.transform(Sa.reshape(-1, len(INDICES))).reshape(Sa.shape)
    Sb = sc_seq.transform(Sb.reshape(-1, len(INDICES))).reshape(Sb.shape)

    prep = Preparation().fit(a)
    stat = [c for c in STATIQUES if c in prep.colonnes_]
    Ta_df = pd.DataFrame(prep.transform(a), columns=prep.colonnes_)[stat]
    Tb_df = pd.DataFrame(prep.transform(b), columns=prep.colonnes_)[stat]
    sc_stat = StandardScaler().fit(Ta_df)
    return (Sa.astype(np.float32), sc_stat.transform(Ta_df).astype(np.float32),
            a[CIBLE].to_numpy(float),
            Sb.astype(np.float32), sc_stat.transform(Tb_df).astype(np.float32),
            b[CIBLE].to_numpy(float),
            dict(cube=cube, idx_cell=idx_cell, origine=origine,
                 cell_de=cell_de, prep=prep, stat=stat, sc_seq=sc_seq,
                 sc_stat=sc_stat, taux=taux, train=train))


def recherche(essais: int) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Sa, Ta, ya, Sb, Tb, yb, _ = _preparer()
    print(f"\nappareil : {_appareil()}\n")

    def objectif(e):
        return entrainer(
            Sa, Ta, ya, Sb, Tb, yb,
            cache=e.suggest_categorical("cache", [32, 64, 128]),
            couches=e.suggest_int("couches", 1, 2),
            dropout=e.suggest_float("dropout", 0.0, 0.6),
            tete=e.suggest_categorical("tete", [64, 128, 256]),
            lr=e.suggest_float("lr", 3e-4, 1e-2, log=True),
            poids_l2=e.suggest_float("poids_l2", 1e-7, 1e-2, log=True),
            taille_lot=e.suggest_categorical("taille_lot", [512, 1024, 2048]),
        )[1]

    etude = optuna.create_study(
        direction="maximize", study_name="tvfed_lstm",
        storage=f"sqlite:///{(PROCESSED / 'optuna.db').as_posix()}",
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=GRAINE))
    fait, t0 = [0], time.time()

    def rappel(st, tr):
        fait[0] += 1
        if tr.value is not None and tr.value >= st.best_value:
            print(f"  essai {fait[0]:>3}/{essais}  PR-AUC {tr.value:.4f}  "
                  f"← meilleur (cache {tr.params['cache']}, "
                  f"dropout {tr.params['dropout']:.2f})")

    etude.optimize(objectif, n_trials=essais, callbacks=[rappel])
    print(f"\n{'═' * 62}")
    print(f"meilleure PR-AUC interne : {etude.best_value:.4f} "
          f"({time.time() - t0:.0f} s)")
    for k, v in etude.best_params.items():
        print(f"   {k:14s} {v}")
    (PROCESSED / "best_params_lstm.json").write_text(
        json.dumps(etude.best_params, indent=2), encoding="utf-8")
    print("\n⚠️ Score interne, sur train échantillonné — NON comparable aux")
    print("   0,0177 de la validation. Seuls les écarts entre configurations")
    print("   ont un sens ici.")
    return etude.best_params


def evaluer(m, ctx) -> pd.DataFrame:
    """Prédit sur la validation intégrale et écrit un fichier CLÉ.

    ⚠️ `sql/50_matrice.sql` n'a pas d'`ORDER BY` : l'ordre des lignes que
    renvoie PostgreSQL n'est garanti par rien et change d'une exécution à
    l'autre (plan, workers parallèles). Un fichier de prédictions sans
    `(code_insee, date)` est donc INCOMPARABLE à un autre — c'est ce qui a
    invalidé la première comparaison LSTM ↔ XGBoost. On sauvegarde les clés.
    """
    import torch

    print("\névaluation sur la validation intégrale…")
    dev = _appareil()
    m.eval()
    cles, scores, cibles, n, t0 = [], [], [], 0, time.time()
    for bloc in matrices.parcourir("val"):
        bloc = clustering.appliquer(bloc, ctx["taux"])
        bloc["date"] = pd.to_datetime(bloc.date)
        seq = sequences(bloc.code_insee.to_numpy(), bloc.date.to_numpy(),
                        ctx["cube"], ctx["idx_cell"], ctx["origine"],
                        ctx["cell_de"])
        seq = ctx["sc_seq"].transform(
            seq.reshape(-1, len(INDICES))).reshape(seq.shape)
        stat = pd.DataFrame(ctx["prep"].transform(bloc),
                            columns=ctx["prep"].colonnes_)[ctx["stat"]]
        stat = ctx["sc_stat"].transform(stat)
        with torch.no_grad():
            # même découpage qu'à l'entraînement : les blocs de 500 000 lignes
            # de `matrices.parcourir` sont bien trop gros pour un LSTM
            morceaux = []
            for i in range(0, len(seq), 4096):
                s = torch.tensor(seq[i:i + 4096], dtype=torch.float32, device=dev)
                t = torch.tensor(stat[i:i + 4096], dtype=torch.float32, device=dev)
                morceaux.append(torch.sigmoid(m(s, t)).squeeze(1).cpu().numpy())
            scores.append(np.concatenate(morceaux).astype(np.float32))
        cles.append(bloc[["code_insee", "date"]])
        cibles.append(bloc[CIBLE].to_numpy(np.int8))
        n += len(bloc)
        if n % 10_000_000 < len(bloc):
            print(f"   {n:>12,} lignes   {time.time() - t0:5.0f} s")

    d = pd.concat(cles, ignore_index=True)
    d["p_lstm"] = np.concatenate(scores)
    d["y"] = np.concatenate(cibles)
    d.to_parquet(PROCESSED / "predictions_val_lstm.parquet", index=False,
                 compression="zstd")
    return d


def final() -> None:
    import torch
    from sklearn.metrics import average_precision_score

    params = json.loads(
        (PROCESSED / "best_params_lstm.json").read_text(encoding="utf-8"))
    print("Hyperparamètres retenus :")
    for k, v in params.items():
        print(f"   {k:14s} {v}")

    Sa, Ta, ya, Sb, Tb, yb, ctx = _preparer()
    print("\npasse 1 : arrêt précoce sur le découpage interne…")
    _, interne, n_ep = entrainer(Sa, Ta, ya, Sb, Tb, yb, **params, bavard=True)
    print(f"  PR-AUC interne {interne:.4f} à l'époque {n_ep}")

    # passe 2 : tout le train, pour le nombre d'époques retenu
    print(f"\npasse 2 : train complet, {n_ep} époques…")
    S = np.concatenate([Sa, Sb])
    T = np.concatenate([Ta, Tb])
    y = np.concatenate([ya, yb])
    m, _, _ = entrainer(S, T, y, Sb, Tb, yb, **{**params},
                        epoques_max=n_ep, patience=n_ep + 1)
    torch.save(m.state_dict(), PROCESSED / "modele_lstm.pt")

    d = evaluer(m, ctx)
    p, yv = d.p_lstm.to_numpy(), d.y.to_numpy()
    ap = average_precision_score(yv, p)

    v3 = pd.read_csv(PROCESSED / "modeles_v3.csv").pr_auc[0]
    mlp = pd.read_csv(PROCESSED / "modeles_mlp.csv").pr_auc[0]
    print("\n" + "═" * 62)
    print(f"{'':30s} {'PR-AUC':>9s} {'lift':>8s}")
    print(f"{'XGBoost v3':30s} {v3:9.4f} {v3 / yv.mean():7.1f}×")
    print(f"{'MLP (sans séquence)':30s} {mlp:9.4f} {mlp / yv.mean():7.1f}×")
    print(f"{'LSTM (30 jours de météo)':30s} {ap:9.4f} {ap / yv.mean():7.1f}×")
    print("═" * 62)
    print(f"\nvs XGBoost v3 : {100 * (ap / v3 - 1):+.1f} %")
    print(f"vs MLP        : {100 * (ap / mlp - 1):+.1f} %")
    print("\n⚠️ Un écart de quelques pour cent ne veut rien dire sans")
    print("   intervalle de confiance — lancer le bootstrap apparié.")

    pd.DataFrame([{"modele": f"LSTM ({FENETRE} j)", "pr_auc": ap,
                   "lift": ap / yv.mean()}]).to_csv(
        PROCESSED / "modeles_lstm.csv", index=False)
    print(f"\n✅ modeles_lstm.csv · predictions_val_lstm.parquet · modele_lstm.pt")


def reevaluer() -> None:
    """Rejoue l'évaluation depuis `modele_lstm.pt`, sans réentraîner.

    Sert à régénérer un fichier de prédictions CLÉ à partir d'un modèle déjà
    entraîné — les poids sont ceux de l'entraînement d'origine, à l'identique.
    """
    import torch
    from sklearn.metrics import average_precision_score

    params = json.loads(
        (PROCESSED / "best_params_lstm.json").read_text(encoding="utf-8"))
    *_, ctx = _preparer()
    m = construire(len(ctx["stat"]), params["cache"], params["couches"],
                   params["dropout"], params["tete"]).to(_appareil())
    m.load_state_dict(torch.load(PROCESSED / "modele_lstm.pt",
                                 map_location=_appareil()))
    d = evaluer(m, ctx)
    ap = average_precision_score(d.y, d.p_lstm)
    print(f"\nPR-AUC LSTM {ap:.4f}   lift {ap / d.y.mean():.1f}×")
    print("✅ predictions_val_lstm.parquet — désormais avec code_insee + date")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--essais", type=int, default=25)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--reevaluer", action="store_true")
    args = ap.parse_args()
    if args.reevaluer:
        reevaluer()
    else:
        final() if args.final else recherche(args.essais)


if __name__ == "__main__":
    main()
