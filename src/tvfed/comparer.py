"""Comparaison appariée de modèles sur la validation, alignée sur les CLÉS.

Pourquoi ce module existe
-------------------------
`sql/50_matrice.sql` ne comporte aucun `ORDER BY` sur la requête externe :
l'ordre des lignes que renvoie PostgreSQL dépend du plan d'exécution et des
workers parallèles, et change d'une exécution à l'autre. Comparer deux
fichiers de prédictions POSITION PAR POSITION est donc faux dès qu'ils
viennent de deux exécutions différentes — et **rien ne le signale** : les
deux fichiers ont la même taille et le même nombre de feux.

C'est exactement ce qui s'est produit avec le LSTM : sa première comparaison
à XGBoost annonçait −97 % alors que l'écart réel est de −52 %. Les fichiers
`predictions_val_v3/dart/mlp.parquet` se trouvaient, eux, partager le même
ordre — par chance, pas par contrat.

D'où la règle : **un fichier de prédictions sans `(code_insee, date)` n'est
comparable à rien.** On aligne sur les clés, et on le vérifie.

Le bootstrap
------------
Rééchantillonner 38 M lignes B fois en appelant `average_precision_score`
coûterait des heures : chaque appel retrie le tableau. On trie **une seule
fois** par modèle ; une réplique n'est alors qu'un jeu de poids entiers le
long de cet ordre figé, et l'AP pondérée se calcule en O(n) par `cumsum`.
Résultat identique à sklearn, environ 100× plus rapide.

On rééchantillonne les **communes**, pas les lignes. Les 1 096 jours d'une
même commune ne sont pas indépendants, et 31 communes partagent en moyenne
la même maille météo : un bootstrap ligne à ligne ignorerait cette
corrélation et produirait des intervalles trop étroits.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from .paths import PROCESSED

# (fichier, colonne de score). Le fichier DOIT porter code_insee et date.
#
# ⚠️ Toutes ces comparaisons ne se valent pas. XGBoost v3, DART et le MLP
# voient l'historique des feux (`feux_commune_*`, `feux_voisins_*`, taux
# lissé par cluster) — dans le modèle A, la BDIFF pesait 29 % des
# importances. Le LSTM, lui, ne reçoit QUE de la météo et des descripteurs de
# territoire : rien qui dérive de `y`.
#
# La seule comparaison loyale pour le LSTM est donc le **modèle C**, bâti sur
# le même jeu d'information (physique pure, sans historique). LSTM vs v3
# mesure surtout le prix de l'information retirée, pas la valeur de la
# séquence temporelle.
SOURCES: dict[str, tuple[str, str]] = {
    "XGBoost v3": ("scores_val.parquet", "xgb_v3"),
    "DART": ("scores_val.parquet", "dart"),
    "MLP": ("scores_val.parquet", "mlp"),
    "XGBoost C": ("scores_c_val.parquet", "p_c"),
    "LSTM": ("predictions_val_lstm.parquet", "p_lstm"),
}

N_REPLIQUES = 200
CLES = ["code_insee", "date"]


def _cle_entiere(d: pd.DataFrame) -> np.ndarray:
    """(code_insee, date) → un entier unique, pour trier sans coût mémoire.

    Compté depuis l'époque Unix : le décalage reste positif pour toute date
    postérieure à 1970, quel que soit le split. Une origine calée sur la
    validation aurait produit des jours négatifs sur le train, et un
    encodage silencieusement faux.
    """
    com = pd.factorize(d.code_insee, sort=True)[0].astype(np.int64)
    jour = d.date.to_numpy().astype("datetime64[D]").astype(np.int64)
    return com * 100_000 + jour       # ~20 000 jours en 2025, marge large


def aligner(sources: dict[str, tuple[str, str]] | None = None,
            bavard: bool = True, garder_cles: bool = False) -> pd.DataFrame:
    """Charge les sources et les aligne sur (code_insee, date).

    Renvoie `commune` (indice entier, unité de rééchantillonnage du
    bootstrap), `y`, et une colonne de score par modèle. Les clés brutes
    sont abandonnées une fois l'alignement fait : 38 M chaînes ne méritent
    pas de rester en mémoire.

    Lève si un fichier n'a pas ses clés, ne couvre pas les mêmes lignes, ou
    porte des cibles différentes après tri.
    """
    import pyarrow.parquet as pq

    sources = SOURCES if sources is None else sources
    cle_ref, com_ref, ref_cles, sortie = None, None, None, {}
    for nom, (fichier, col) in sources.items():
        chemin = PROCESSED / fichier
        if not chemin.exists():
            if bavard:
                print(f"   ⚠️ {nom:12s} absent ({fichier}) — ignoré")
            continue
        manquantes = set(CLES) - set(pq.read_schema(chemin).names)
        if manquantes:
            raise ValueError(
                f"{fichier} n'a pas {manquantes}. La requête d'assemblage "
                f"n'ayant pas d'ORDER BY, un fichier sans clés ne peut être "
                f"comparé à rien — le régénérer.")
        d = pd.read_parquet(chemin, columns=CLES + [col, "y"])
        d["date"] = pd.to_datetime(d.date)
        k = _cle_entiere(d)
        o = np.argsort(k, kind="stable")
        d, cle = d.iloc[o], k[o]
        y = d.y.to_numpy(np.int8)

        if cle_ref is None:
            cle_ref, sortie["y"] = cle, y
            com_ref = pd.factorize(d.code_insee, sort=True)[0].astype(np.int32)
            if garder_cles:
                ref_cles = d[CLES].reset_index(drop=True)
        elif not np.array_equal(cle_ref, cle):
            raise ValueError(f"{nom} ne couvre pas les mêmes (commune, date) "
                             f"que la référence — comparaison impossible.")
        elif not np.array_equal(sortie["y"], y):
            raise ValueError(f"{nom} : cibles incohérentes après alignement.")
        sortie[nom] = d[col].to_numpy(np.float32)
        if bavard:
            print(f"   {nom:12s} {len(d):>12,} lignes  aligné")

    if cle_ref is None:
        raise FileNotFoundError("aucune source de prédictions trouvée")
    sortie = pd.DataFrame({"commune": com_ref, **sortie})
    return pd.concat([ref_cles, sortie], axis=1) if garder_cles else sortie


class ApRapide:
    """AP pondérée le long d'un ordre trié une fois pour toutes.

    `fin` marque le dernier indice de chaque groupe d'ex æquo : c'est ce qui
    rend le résultat identique à `average_precision_score`, qui n'évalue la
    précision qu'aux seuils distincts.
    """

    def __init__(self, p: np.ndarray, y: np.ndarray):
        # int32 partout : 38 M lignes tiennent largement, et les cumuls
        # pondérés plafonnent autour de 40 M — très loin de 2^31.
        o = np.argsort(-p, kind="stable").astype(np.int32)
        self.ordre = o
        self.y = y[o].astype(np.int8)
        self.fin = np.r_[np.nonzero(np.diff(p[o]))[0],
                         len(p) - 1].astype(np.int32)

    def __call__(self, poids: np.ndarray | None = None) -> float:
        w = (np.ones(len(self.y), np.int32) if poids is None
             else poids[self.ordre].astype(np.int32))
        pos = self.y * w
        tp = np.cumsum(pos)[self.fin].astype(np.float64)
        fp = np.cumsum(w - pos)[self.fin].astype(np.float64)
        if tp[-1] == 0:
            return float("nan")
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / tp[-1]
        return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def main() -> None:
    from sklearn.metrics import average_precision_score

    print("Comparaison appariée sur la validation\n" + "=" * 62)
    d = aligner()
    com = d.pop("commune").to_numpy(np.int32)
    noms = [c for c in d.columns if c != "y"]
    y = d.y.to_numpy(np.int8)
    taux = y.mean()
    print(f"\n   {len(d):,} lignes · {y.sum():,} feux · taux {taux:.5%}\n")

    # ── contrôle : l'AP rapide doit reproduire sklearn à l'identique ──
    sous = np.random.default_rng(0).choice(len(d), 2_000_000, replace=False)
    p0 = d[noms[0]].to_numpy()[sous]
    ecart = abs(average_precision_score(y[sous], p0) - ApRapide(p0, y[sous])())
    print(f"   contrôle vs sklearn (2 M lignes) : écart {ecart:.2e}")
    assert ecart < 1e-9, "l'AP pondérée ne reproduit pas sklearn"
    del sous, p0

    # chaque colonne de scores est libérée dès son index construit : à cinq
    # modèles, les garder toutes coûterait 760 Mo pour rien.
    ap = {}
    for n in noms:
        t0 = time.time()
        ap[n] = ApRapide(d.pop(n).to_numpy(), y)
        v, dt = ap[n](), time.time() - t0
        print(f"   {n:12s} PR-AUC {v:.4f}  lift {v / taux:5.1f}×  ({dt:.0f} s)")

    # ── bootstrap par commune ────────────────────────────────────────────
    n_com = com.max() + 1
    rng = np.random.default_rng(42)
    print(f"\n   bootstrap : {N_REPLIQUES} répliques sur {n_com:,} communes")
    tirages = np.zeros((N_REPLIQUES, len(noms)))
    t0 = time.time()
    for b in range(N_REPLIQUES):
        mult = np.bincount(rng.integers(0, n_com, n_com),
                           minlength=n_com).astype(np.int64)
        w = mult[com]
        for j, n in enumerate(noms):
            tirages[b, j] = ap[n](w)
        if (b + 1) % 50 == 0:
            print(f"      {b + 1:>4}/{N_REPLIQUES}   {time.time() - t0:5.0f} s")

    lignes = []
    print("\n" + "=" * 62)
    print(f"{'comparaison':28s} {'écart':>9s} {'IC 95 %':>20s}  verdict")
    print("-" * 62)
    for a, b in itertools.combinations(range(len(noms)), 2):
        rel = 100 * (tirages[:, b] / tirages[:, a] - 1)
        lo, hi = np.percentile(rel, [2.5, 97.5])
        pt = 100 * (ap[noms[b]]() / ap[noms[a]]() - 1)
        sig = "significatif" if lo * hi > 0 else "NON significatif"
        print(f"{noms[b] + ' vs ' + noms[a]:28s} {pt:+8.1f}% "
              f"[{lo:+7.1f}, {hi:+7.1f}]  {sig}")
        lignes.append({"reference": noms[a], "modele": noms[b],
                       "ecart_pct": pt, "ic_bas": lo, "ic_haut": hi,
                       "significatif": lo * hi > 0})
    print("=" * 62)

    dest = PROCESSED / "comparaison_appariee.csv"
    pd.DataFrame(lignes).to_csv(dest, index=False)
    pd.DataFrame({n: [ap[n]()] for n in noms}).to_csv(
        PROCESSED / "pr_auc_val.csv", index=False)
    print(f"\n✅ {dest.name} · pr_auc_val.csv")


if __name__ == "__main__":
    main()
