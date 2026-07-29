"""Étape 15 — le modèle sait-il prédire le feu là où il n'en a jamais vu ?

    python -m tvfed.transfert_spatial

────────────────────────────────────────────────────────────────────────────
LA QUESTION, ET POURQUOI ELLE DÉCIDE DE TOUT POUR 2050
────────────────────────────────────────────────────────────────────────────
Le modèle actuel tire 54 % de son importance de l'historique de la commune.
Il dit, pour l'essentiel, « ça brûlera où ça a déjà brûlé ». C'est excellent
pour alerter demain — 93,8× le hasard sur le test — et **structurellement
inutilisable pour projeter 2050**, où le feu gagnera des zones qui n'ont
jamais brûlé.

On ne va pas le supposer, on va le mesurer : **validation croisée spatiale**.
On retire une région entière, on entraîne sur les douze autres, on teste sur
celle qu'on a retirée. Ses communes sont alors, pour le modèle, exactement ce
que sera le Morbihan en 2050 : un territoire dont il ignore tout l'historique.

────────────────────────────────────────────────────────────────────────────
TROIS MODÈLES, POUR ISOLER CE QUI TRANSFÈRE
────────────────────────────────────────────────────────────────────────────
    A  tout          les 52 features actuelles
    B  territorial   sans l'historique propre à la commune, mais AVEC le taux
                     de son cluster — lequel est estimé sur des communes
                     d'AUTRES régions. C'est la substitution espace-temps.
    C  physique      météo + végétation + relief seulement. Aucune variable
                     dérivée de `y`, ni lat/lon.

`lat` et `lon` sortent de B et C volontairement : ce sont elles qui encodent
« le Sud brûle ». C'est vrai aujourd'hui, et c'est exactement le préjugé qu'il
ne faut pas transporter en 2050.

────────────────────────────────────────────────────────────────────────────
⚠️ LA DISCIPLINE TEMPORELLE EST CONSERVÉE
────────────────────────────────────────────────────────────────────────────
Tout se passe DANS le train : ajustement ≤2017, évaluation 2018-2019. La
validation et le test ne sont pas touchés — le test a déjà été consommé, il
ne servira plus à rien d'autre.

⚠️ ET LES TAUX SONT RECALCULÉS À CHAQUE PLI. Laisser le taux lissé d'une
commune de la région retirée serait la fuite qui viderait l'expérience de son
sens : le modèle connaîtrait justement ce qu'on prétend lui cacher. Les
communes retirées reçoivent donc le taux de leur cluster, estimé sans elles.
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

HIST_COMMUNE = ["feux_commune_7j", "feux_commune_30j", "feux_commune_90j",
                "feux_commune_365j", "jours_depuis_dernier_feu",
                "taux_commune_lisse", "ratio_commune_cluster"]
POSITION = ["lat", "lon"]
CLUSTER = ["taux_cluster_lisse", "cluster_id"]

JEUX = {
    "A · tout": [],
    "B · territorial": HIST_COMMUNE + POSITION,
    "C · physique": HIST_COMMUNE + POSITION + CLUSTER,
}


def regions() -> pd.Series:
    with db.connexion() as c:
        r = pd.read_sql(
            "SELECT code_insee, reg_code FROM ref_commune WHERE in_perimetre", c)
    return r.set_index("code_insee").reg_code


def lisser_sans(sin: pd.DataFrame, clusters: pd.Series,
                exclues: set[str]) -> pd.DataFrame:
    """Taux lissés en IGNORANT totalement les communes exclues.

    Les communes retirées n'ont, du point de vue du modèle, aucun historique.
    Elles reçoivent donc le taux de leur cluster — estimé sur les seules
    communes conservées. C'est exactement le mécanisme qu'on veut tester :
    une commune sans passé hérite du risque de ses semblables ailleurs.
    """
    s = sin[~sin.code_insee.isin(exclues)].merge(
        clusters.rename("cluster_id"), left_on="code_insee", right_index=True)
    p_nat = s.feux.sum() / s.jours.sum()

    k = s.groupby("cluster_id")[["jours", "feux"]].sum()
    p_clu = ((k.feux + clustering.K0_CLUSTER * p_nat)
             / (k.jours + clustering.K0_CLUSTER))

    c = s.groupby("code_insee")[["jours", "feux"]].sum()
    prior_c = clusters.reindex(c.index).map(p_clu)
    p_com = ((c.feux + clustering.K1_COMMUNE * prior_c)
             / (c.jours + clustering.K1_COMMUNE))

    # les exclues : rien d'observé, donc le prior de leur cluster, point.
    ex = pd.Index(sorted(exclues))
    prior_ex = clusters.reindex(ex).map(p_clu)

    out = pd.DataFrame({
        "code_insee": list(c.index) + list(ex),
        "cluster_id": (list(clusters.reindex(c.index)) + list(clusters.reindex(ex))),
        "taux_cluster_lisse": list(prior_c) + list(prior_ex),
        "taux_commune_lisse": list(p_com) + list(prior_ex),
    })
    out["ratio_commune_cluster"] = out.taux_commune_lisse / out.taux_cluster_lisse
    out["an_exclue"] = 0
    return out


def main() -> None:
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    params = json.loads(
        (PROCESSED / "best_params_xgb.json").read_text(encoding="utf-8"))
    for k in ("n_estimators", "max_depth", "min_child_weight"):
        params[k] = int(params[k])

    print("préparation…")
    reg = regions()
    p = clustering.profil()
    sin = clustering.sinistralite()
    cl = clustering.ajuster(p, METHODE, K)
    manq = sorted(set(sin.code_insee) - set(cl.index))
    if manq:
        cl = pd.concat([cl, pd.Series(-1, index=manq, name="cluster_id")])

    base = pd.read_parquet(PROCESSED / "train.parquet")
    # ⚠️ région et année restent HORS du DataFrame. `Preparation._features`
    # prend toute colonne non explicitement exclue : y laisser `reg` ferait
    # planter la médiane sur une chaîne, et y laisser l'année en ferait une
    # feature — exactement ce qu'on a décidé de ne jamais donner au modèle.
    reg_lignes = base.code_insee.map(reg).to_numpy()
    an_lignes = pd.to_datetime(base.date).dt.year.to_numpy()
    ordre = sorted(pd.unique(reg_lignes[pd.notna(reg_lignes)]))
    print(f"  {len(base):,} lignes, {len(ordre)} régions\n")

    lignes, predictions = [], []
    for r in ordre:
        t0 = time.time()
        exclues = set(reg[reg == r].index)
        taux = lisser_sans(sin, cl, exclues)
        d = clustering.appliquer(base.copy(), taux, exclure_annee=False)

        m_a = (reg_lignes != r) & (an_lignes <= AN_FIT)      # 12 régions, ≤2017
        m_b = (reg_lignes == r) & (an_lignes >= AN_EVAL[0]) \
            & (an_lignes <= AN_EVAL[1])                      # la 13e, 2018-2019
        a, b = d[m_a], d[m_b]
        if b[CIBLE].sum() < 30:
            print(f"  région {r} : {int(b[CIBLE].sum())} positifs — ignorée")
            continue

        ligne = {"region": r, "communes": len(exclues), "lignes_test": len(b),
                 "feux_test": int(b[CIBLE].sum()), "taux": b[CIBLE].mean()}
        # les predictions ligne a ligne, pour pouvoir bootstrapper ensuite :
        # sans elles on ne peut mesurer l'incertitude ni DANS une region ni
        # ENTRE regions, et un ecart de 13 % sur 9 regions n'est pas evaluable.
        brut = {"region": np.full(len(b), r), "y": b[CIBLE].to_numpy(np.int8)}
        for nom, retire in JEUX.items():
            prep = Preparation().fit(a)
            garde = [c for c in prep.colonnes_ if c not in retire]
            m = XGBClassifier(**params, tree_method="hist", eval_metric="aucpr",
                              device="cuda", n_jobs=-1,
                              random_state=42).fit(
                pd.DataFrame(prep.transform(a), columns=prep.colonnes_)[garde],
                a[CIBLE].to_numpy())
            pr = m.predict_proba(
                pd.DataFrame(prep.transform(b), columns=prep.colonnes_)[garde])[:, 1]
            ap = average_precision_score(b[CIBLE].to_numpy(), pr)
            ligne[nom] = ap
            ligne[f"lift_{nom[0]}"] = ap / b[CIBLE].mean()
            brut[nom[0]] = pr.astype(np.float32)
        lignes.append(ligne)
        predictions.append(pd.DataFrame(brut))
        print(f"  région {r}  {len(exclues):>5,} communes, "
              f"{ligne['feux_test']:>4,} feux   " +
              "  ".join(f"{n[0]} {ligne[n]:.4f}" for n in JEUX) +
              f"   ({time.time() - t0:.0f} s)")

    R = pd.DataFrame(lignes)
    R.to_csv(PROCESSED / "transfert_spatial.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        PROCESSED / "transfert_spatial_pred.parquet", index=False, compression="zstd")

    print(f"\n{'═' * 74}")
    print("MOYENNE SUR LES RÉGIONS RETIRÉES — chacune inconnue du modèle")
    print(f"{'═' * 74}")
    print(f"{'jeu de features':20s} {'PR-AUC moyen':>14s} {'lift moyen':>12s} "
          f"{'pire région':>13s}")
    for n in JEUX:
        print(f"{n:20s} {R[n].mean():14.4f} {R[f'lift_{n[0]}'].mean():11.1f}× "
              f"{R[n].min():13.4f}")

    a, c = R["A · tout"].mean(), R["C · physique"].mean()
    b = R["B · territorial"].mean()
    print(f"\n{'─' * 74}")
    print("CE QUE ÇA DIT")
    print(f"{'─' * 74}")
    print(f"En territoire inconnu, B vaut {100 * b / a:.0f} % de A "
          f"et C en vaut {100 * c / a:.0f} %.")
    print()
    if b >= a * 0.97:
        print("→ Le modèle territorial TIENT sans l'historique de la commune.")
        print("  Le taux du cluster, estimé sur des communes d'autres régions,")
        print("  suffit à transporter le signal. C'est la substitution")
        print("  espace-temps, et elle fonctionne : elle est utilisable en 2050.")
    else:
        print("→ Retirer l'historique COÛTE, même en territoire inconnu.")
        print("  À arbitrer : un modèle moins bon mais valide en extrapolation")
        print("  reste préférable à un modèle meilleur mais faux hors de sa zone.")
    print()
    print("⚠️ Ces PR-AUC ne se comparent PAS aux 0,0156 du test : on est ici sur")
    print("   le train échantillonné, où les positifs sont ~350 fois plus")
    print("   fréquents. Seuls les ÉCARTS entre A, B et C ont un sens.")
    print(f"\n✅ transfert_spatial.csv")


if __name__ == "__main__":
    main()
