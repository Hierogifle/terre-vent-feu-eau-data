"""Étape 13 — calibrer le v3, et mesurer honnêtement si ça tient.

    python -m tvfed.calibration_v3

────────────────────────────────────────────────────────────────────────────
CE QUE LA CALIBRATION CORRIGE, ET CE QU'ELLE NE CORRIGE PAS
────────────────────────────────────────────────────────────────────────────
Le modèle apprend sur un train échantillonné à 1:10 : il y voit 9,1 % de
positifs quand la réalité en compte 0,024 %. Ses probabilités sont donc
sur-estimées d'un facteur ~400. Il CLASSE bien, il MENT sur les niveaux.

Ça n'a aucun effet sur la PR-AUC, qui ne dépend que de l'ordre. Mais
l'application affiche un chiffre à un humain : « 30 % de risque » doit vouloir
dire que ça arrive 30 % du temps, comme les 70 % de chance de pluie.

────────────────────────────────────────────────────────────────────────────
⚠️ LE PIÈGE DE MÉTHODE, ET COMMENT ON L'ÉVITE
────────────────────────────────────────────────────────────────────────────
Ajuster la correction sur la validation PUIS la mesurer sur la même validation
est circulaire : elle y paraîtra parfaite par construction, et on ne saura rien
de sa tenue sur des données futures.

On découpe donc la validation en deux, TEMPORELLEMENT :

    2020-2021  →  ajustement de la correction
    2022       →  jugement

C'est le même principe que le split principal, appliqué une couche plus bas —
et c'est la seule façon d'estimer ce que vaudra la calibration sur le test.

La correction finale, elle, sera réajustée sur la validation ENTIÈRE : plus de
données, et 2022 n'a plus besoin d'être préservé une fois la méthode choisie.

────────────────────────────────────────────────────────────────────────────
LES DEUX MÉTHODES, ET POURQUOI LE CHOIX N'EST PAS ÉVIDENT
────────────────────────────────────────────────────────────────────────────
PLATT        une sigmoïde à deux paramètres. STRICTEMENT croissante, donc elle
             préserve exactement l'ordre : la PR-AUC ne bouge pas d'un chiffre.

ISOTONIQUE   une fonction en escalier, libre de sa forme. Plus souple, donc
             souvent meilleure sur le Brier — mais elle écrase des millions de
             scores distincts en quelques dizaines de milliers de paliers, et
             l'ordre à l'intérieur d'un palier est DÉFINITIVEMENT PERDU.
             Mesuré sur le v1 : −10 % de PR-AUC.

Le bon choix dépend de l'usage : classer des communes (l'ordre compte) ou
afficher un pourcentage (le niveau compte). L'application fait les deux.

────────────────────────────────────────────────────────────────────────────
⚠️ LA CALIBRATION IGNORE LE TEMPS — ET C'EST MESURÉ COMME ÉTANT LE BON CHOIX
────────────────────────────────────────────────────────────────────────────
Platt est une sigmoïde à deux paramètres, sans aucune entrée temporelle : la
même correction s'applique le 3 janvier et le 15 août. Le biais résiduel par
mois, jugé sur 2022, va de ×0,86 (août) à ×1,70 (décembre) — et dans le pire
sens, puisqu'il SOUS-estime en juillet-août, quand le modèle sert vraiment.

La correction évidente serait de conditionner la calibration au moment de
l'année. Testée dans trois variantes, elle DÉGRADE :

    score seul                      PR-AUC 0,0164   biais 0,86×–1,70×
    score + indicateur juil-août    PR-AUC 0,0130   biais 0,45×–1,97×
    score + 4 harmoniques           PR-AUC 0,0141   biais 0,60×–5,55×

Deux raisons, et la seconde est la vraie :

  1. Le profil saisonnier estimé sur 2020-2021 ne vaut pas pour 2022 — deux
     ans ne suffisent pas à l'estimer, et 2022 fut atypique (Gironde).

  2. **Le modèle connaît déjà la saison.** `doy`, `sin_doy`, `cos_doy` et les
     huit indices CEMS sont dans ses 52 features. Un calibrateur qui rajoute
     la saison la compte DEUX FOIS.

→ Le biais résiduel est donc documenté comme une limite, pas corrigé par
  davantage de paramètres. Le corriger proprement demanderait plus d'années
  de validation — qu'on ne peut pas prendre sans entamer le test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import ajuster, diagnostic, resume
from .paths import PROCESSED

AN_AJUST = (2020, 2021)
AN_JUGE = 2022
MODELE = "xgb_v3"


def _fiabilite(p, y, bins=12) -> pd.DataFrame:
    d = diagnostic(p, y, bins)
    d["ecart"] = d.annonce - d.observe
    return d


def main() -> None:
    from sklearn.metrics import average_precision_score, brier_score_loss

    f = PROCESSED / "scores_val.parquet"
    if not f.exists():
        raise FileNotFoundError(
            "scores_val.parquet absent — lancer d'abord :\n"
            "    python -m tvfed.scores --split val")
    D = pd.read_parquet(f)
    an = pd.to_datetime(D.date).dt.year
    A, J = D[an.isin(AN_AJUST)], D[an == AN_JUGE]

    print(f"ajustement {AN_AJUST[0]}-{AN_AJUST[1]} : {len(A):>12,} lignes, "
          f"{A.y.sum():>5,} feux ({A.y.mean():.4%})")
    print(f"jugement   {AN_JUGE}      : {len(J):>12,} lignes, "
          f"{J.y.sum():>5,} feux ({J.y.mean():.4%})")

    pa, ya = A[MODELE].to_numpy(np.float64), A.y.to_numpy()
    pj, yj = J[MODELE].to_numpy(np.float64), J.y.to_numpy()

    # ── les deux corrections, apprises sur 2020-2021, jugées sur 2022 ────
    resultats = [{
        "methode": "brut", "pr_auc": average_precision_score(yj, pj),
        "brier": brier_score_loss(yj, pj), "p_moyen": pj.mean(),
        "biais": pj.mean() / yj.mean(), "valeurs_distinctes": len(np.unique(pj)),
    }]
    corrigees = {"brut": pj}
    for methode in ("platt", "isotonic"):
        c = ajuster(pa, ya, methode)
        pc = c.predict(pj)
        corrigees[methode] = pc
        resultats.append({
            "methode": methode, "pr_auc": average_precision_score(yj, pc),
            "brier": brier_score_loss(yj, pc), "p_moyen": pc.mean(),
            "biais": pc.mean() / yj.mean(),
            "valeurs_distinctes": len(np.unique(pc)),
        })

    R = pd.DataFrame(resultats)
    R.to_csv(PROCESSED / "calibration_v3.csv", index=False)

    print(f"\n{'═' * 78}")
    print(f"JUGÉ SUR {AN_JUGE} — la correction n'a jamais vu cette année")
    print(f"{'═' * 78}")
    print(f"{'':11s} {'PR-AUC':>8s} {'Brier':>11s} {'p moyen':>10s} "
          f"{'biais':>8s} {'valeurs':>12s}")
    for r in R.itertuples():
        print(f"{r.methode:11s} {r.pr_auc:8.4f} {r.brier:11.3e} "
              f"{r.p_moyen:10.6f} {r.biais:7.1f}× {r.valeurs_distinctes:>12,}")
    print(f"{'réel':11s} {'':8s} {'':11s} {yj.mean():10.6f} {1.0:7.1f}×")

    # ── courbes de fiabilité ────────────────────────────────────────────
    for m, p in corrigees.items():
        _fiabilite(p, yj).assign(methode=m).to_csv(
            PROCESSED / f"fiabilite_{m}.csv", index=False)

    perte = 100 * (1 - R.set_index("methode").pr_auc["isotonic"]
                   / R.set_index("methode").pr_auc["brut"])
    gard = 100 * (1 - R.set_index("methode").pr_auc["platt"]
                  / R.set_index("methode").pr_auc["brut"])
    print(f"\n{'─' * 78}")
    print("LE CHOIX")
    print(f"{'─' * 78}")
    print(f"Platt      : PR-AUC {gard:+.2f} % — l'ordre est préservé, c'est")
    print("             une sigmoïde strictement croissante.")
    print(f"Isotonique : PR-AUC {-perte:+.2f} % — l'escalier écrase des scores")
    print("             distincts en paliers, et l'ordre y est perdu.")
    print()
    print("→ Pour une application qui CLASSE (carte des communes à surveiller),")
    print("  Platt. Pour un affichage de pourcentage isolé, l'isotonique est")
    print("  plus juste en niveau. L'app fait les deux : elle classera sur le")
    print("  score Platt et affichera le niveau calibré.")

    # ── correction finale, réajustée sur TOUTE la validation ────────────
    import pickle
    final = ajuster(D[MODELE].to_numpy(np.float64), D.y.to_numpy(), "platt")
    with open(PROCESSED / "calibrateur_platt.pkl", "wb") as fh:
        pickle.dump(final, fh)
    print(f"\n✅ calibration_v3.csv · fiabilite_*.csv · calibrateur_platt.pkl")
    print("   (le calibrateur final est ajusté sur la validation ENTIÈRE :")
    print("    la méthode étant choisie, 2022 n'a plus à être préservé)")


if __name__ == "__main__":
    main()
