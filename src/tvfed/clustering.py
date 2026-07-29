"""Étape 8 — clustering territorial et lissage bayésien de la sinistralité.

    python -m tvfed.clustering --methode kmeans --k 30

────────────────────────────────────────────────────────────────────────────
POURQUOI CETTE ÉTAPE
────────────────────────────────────────────────────────────────────────────
Le diagnostic du v1 est sans ambiguïté : 54,6 % de l'importance part de
l'historique de la commune. Le modèle dit surtout « ce qui a brûlé rebrûlera ».

Ce n'est pas une fuite — le jour J on connaît réellement le passé — mais ça
laisse un trou béant : **une commune qui n'a jamais brûlé garde un score bas,
même entourée de communes qui brûlent chaque été.** C'est le problème
classique de *small area estimation* : trop peu d'événements pour estimer un
taux commune par commune.

La parade : regrouper les communes qui se ressemblent, estimer le taux sur le
GROUPE, et faire retomber chaque commune vers le taux de son groupe à
proportion de ce qu'on sait d'elle. Une commune sans historique hérite du
risque de ses semblables au lieu d'hériter de zéro.

────────────────────────────────────────────────────────────────────────────
CE QU'ON REGROUPE — et ce qu'on ne regroupe surtout pas
────────────────────────────────────────────────────────────────────────────
Le clustering porte sur des caractéristiques PHYSIQUES du territoire :
végétation, relief, densité humaine, climatologie du FWI, position.
**Jamais sur `y`.** Un cluster construit sur la sinistralité serait
circulaire : on prédirait le feu avec des groupes définis par le feu.

lat/lon sont inclus mais VOLONTAIREMENT SOUS-PONDÉRÉS (voir POIDS_POSITION).
Sans eux, les clusters seraient éclatés d'un bout à l'autre du pays ; à poids
plein, ils dégénéreraient en pavés géographiques et le clustering ne serait
qu'un découpage administratif déguisé.

────────────────────────────────────────────────────────────────────────────
LES TROIS GARDE-FOUS ANTI-FUITE
────────────────────────────────────────────────────────────────────────────
1. Le profil ne lit que le passé : CORINE 2006 et climatologie FWI 2006-2019
   (voir sql/60_profil_commune.sql).

2. Les taux sont agrégés sur le train COMPLET, jamais sur l'échantillon —
   sinon le prior est faux d'un facteur ×487 (voir sql/61_sinistralite.sql).

3. Pour une ligne de train de l'année Y, les taux EXCLUENT l'année Y.
   Sans ça une ligne de 2012 contribuerait à sa propre feature : c'est la
   fuite classique du target encoding, et elle gonfle le score de train sans
   rien apporter en généralisation. Les lignes de val et de test, elles,
   utilisent les 14 années de train — elles n'y figurent pas.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import db
from .paths import PROCESSED, RACINE

# ── pseudo-effectifs du lissage ─────────────────────────────────────────
# k0 : force de rappel du cluster vers le taux national
# k1 : force de rappel de la commune vers son cluster
#
# Chaque commune a EXACTEMENT le même nombre de jours de train (5 113), la
# grille étant dense et rectangulaire. Le lissage ne corrige donc pas une
# différence d'exposition mais une rareté d'ÉVÉNEMENTS : une commune avec
# 0 feu en 5 113 jours a un taux empirique de 0, ce qui est une estimation
# épouvantable. k1 = 2000 signifie « je ne fais pleinement confiance au
# comptage d'une commune qu'au-delà de ~2 000 jours d'observation », soit
# environ 40 % de rappel vers le cluster.
K0_CLUSTER = 20_000
K1_COMMUNE = 2_000

# lat/lon rapportés à 25 % du poids des autres variables : assez pour que les
# clusters restent géographiquement cohérents, pas assez pour qu'ils ne soient
# QUE géographiques.
POIDS_POSITION = 0.25


# ════════════════════════════════════════════════════════════════════════
#  1. le profil statique
# ════════════════════════════════════════════════════════════════════════
def profil() -> pd.DataFrame:
    """Une ligne par commune : sa carte d'identité territoriale."""
    sql = (RACINE / "sql" / "60_profil_commune.sql").read_text(encoding="utf-8")
    with db.connexion() as conn:
        p = pd.read_sql(sql, conn)
    return p.set_index("code_insee")


def _matrice_clustering(p: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Standardise et pondère. Retourne la matrice prête à clusteriser."""
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in p.columns if c != "dep_code"]
    X = p[cols].astype(float)

    # ⚠️ imputation avant standardisation : 162 communes ont des altitudes
    # corrompues, mises à NULL au chargement (min 9 589 m en Corse…).
    X = X.fillna(X.median())

    Z = StandardScaler().fit_transform(X)
    for i, c in enumerate(cols):                     # sous-pondération position
        if c in ("lat", "lon"):
            Z[:, i] *= POIDS_POSITION
    return Z, cols


def ajuster(p: pd.DataFrame, methode: str, k: int) -> pd.Series:
    """Retourne le cluster de chaque commune.

    Aucun `y` n'entre ici : le clustering peut donc voir toutes les communes,
    train comme val et test. Ce qui devra rester train-only, c'est le TAUX
    attaché à chaque cluster — pas la partition elle-même.
    """
    Z, _ = _matrice_clustering(p)

    if methode == "kmeans":
        from sklearn.cluster import KMeans
        lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Z)

    elif methode == "hdbscan":
        from sklearn.cluster import HDBSCAN
        # min_cluster_size piloté par k : viser ~k groupes sur 34 734 communes
        lab = HDBSCAN(min_cluster_size=max(50, len(Z) // (k * 3)),
                      min_samples=10).fit_predict(Z)
        # HDBSCAN étiquette le bruit -1. On en fait un cluster à part entière :
        # ces communes « atypiques » partagent au moins d'être atypiques, et
        # le lissage leur donnera le taux du groupe des atypiques.
        lab = np.where(lab < 0, lab.max() + 1, lab)

    else:
        raise ValueError(f"méthode inconnue : {methode}")

    return pd.Series(lab, index=p.index, name="cluster_id")


# ════════════════════════════════════════════════════════════════════════
#  2. le lissage bayésien, avec exclusion de l'année
# ════════════════════════════════════════════════════════════════════════
def sinistralite() -> pd.DataFrame:
    """Feux et jours par commune × année, sur le train complet."""
    sql = (RACINE / "sql" / "61_sinistralite.sql").read_text(encoding="utf-8")
    with db.connexion() as conn:
        return pd.read_sql(sql, conn)


def lisser(sin: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    """Taux lissés par (commune, année exclue).

    `an_exclue = 0` est la ligne à utiliser pour val et test : rien n'est
    retranché, les 14 années de train servent.

    Le lissage est hiérarchique — national → cluster → commune :

        p_cluster  = (feux_c + k0·p_national) / (jours_c + k0)
        p_commune  = (feux_i + k1·p_cluster)  / (jours_i + k1)

    Une commune avec beaucoup de feux garde son taux propre ; une commune
    sans historique se voit attribuer celui de son cluster. C'est exactement
    la correction que le v1 ne savait pas faire.
    """
    s = sin.merge(clusters.rename("cluster_id"), left_on="code_insee",
                  right_index=True, how="inner")

    p_national = s.feux.sum() / s.jours.sum()

    # totaux tous ans confondus
    tot_com = s.groupby("code_insee")[["jours", "feux"]].sum()
    tot_clu = s.groupby("cluster_id")[["jours", "feux"]].sum()
    # totaux par année, pour pouvoir les retrancher
    an_com = s.set_index(["code_insee", "an"])[["jours", "feux"]]
    an_clu = s.groupby(["cluster_id", "an"])[["jours", "feux"]].sum()

    # Les années viennent des DONNÉES, pas d'une constante : la borne du train
    # est déjà écrite dans 61_sinistralite.sql, la répéter ici créerait deux
    # sources de vérité qui finiraient par diverger.
    lignes = []
    for an in [0, *sorted(s.an.unique())]:
        c = tot_com.copy()
        k = tot_clu.copy()
        if an:                                   # ⚠️ exclusion de l'année
            c = c.sub(an_com.xs(an, level="an"), fill_value=0)
            k = k.sub(an_clu.xs(an, level="an"), fill_value=0)

            # ⚠️ REMISE À EXPOSITION CONSTANTE — indispensable.
            #
            # Retirer une année fait tomber le dénominateur de 5 113 jours à
            # 4 748. Sans correction, la MÊME commune reçoit un taux ~4,3 %
            # plus élevé sur une ligne de train que sur une ligne de val, et
            # `ratio_commune_cluster` vaut 0,2964 pour toute commune muette de
            # train contre 0,2812 en val : deux plages DISJOINTES. Le modèle
            # apprendrait des seuils qui ne veulent plus rien dire au moment
            # de prédire — un décalage train/service, invisible dans toutes
            # les métriques de train.
            #
            # On extrapole donc les comptages « hors année Y » à l'exposition
            # complète. Le dénominateur redevient identique partout, seul le
            # numérateur porte l'exclusion — ce qui est bien le but.
            c = c.mul(tot_com.jours / c.jours, axis=0)
            k = k.mul(tot_clu.jours / k.jours, axis=0)

        p_clu = (k.feux + K0_CLUSTER * p_national) / (k.jours + K0_CLUSTER)
        prior = clusters.map(p_clu)              # le prior de chaque commune
        p_com = (c.feux + K1_COMMUNE * prior) / (c.jours + K1_COMMUNE)

        lignes.append(pd.DataFrame({
            "code_insee": c.index,
            "an_exclue": an,
            "cluster_id": clusters.reindex(c.index).to_numpy(),
            "taux_cluster_lisse": prior.reindex(c.index).to_numpy(),
            "taux_commune_lisse": p_com.reindex(c.index).to_numpy(),
        }))

    out = pd.concat(lignes, ignore_index=True)
    out["ratio_commune_cluster"] = (out.taux_commune_lisse
                                    / out.taux_cluster_lisse)
    return out


# ════════════════════════════════════════════════════════════════════════
#  3. application aux matrices
# ════════════════════════════════════════════════════════════════════════
def appliquer(df: pd.DataFrame, taux: pd.DataFrame,
              exclure_annee: bool = True) -> pd.DataFrame:
    """Ajoute les colonnes de clustering à une matrice.

    ⚠️ C'est ICI que se joue l'anti-fuite : une ligne de train de 2012 reçoit
    les taux calculés SANS 2012. Une ligne de val ou de test reçoit ceux
    calculés sur les 14 années de train, dont elle ne fait pas partie.

    `exclure_annee=False` désactive ce mécanisme et lit partout la ligne
    `an_exclue = 0`. Réservé à la validation croisée SPATIALE, où le retrait
    porte sur une région entière et non sur une année : les taux y sont déjà
    calculés sans les communes testées, l'exclusion annuelle n'a plus d'objet
    et la table ne contient d'ailleurs qu'un seul millésime.
    """
    an = pd.to_datetime(df["date"]).dt.year
    est_train = df["split"].to_numpy() == "train"
    cle = pd.DataFrame({
        "code_insee": df["code_insee"].to_numpy(),
        "an_exclue": np.where(est_train & exclure_annee, an, 0),
    })
    j = cle.merge(taux, on=["code_insee", "an_exclue"], how="left")

    # Un NaN ici serait une commune absente de la table de taux — donc une
    # ligne à qui le clustering n'apporte rien, silencieusement. L'imputation
    # par la médiane la ferait passer pour une commune moyenne. On échoue.
    if j.taux_commune_lisse.isna().any():
        orphelines = sorted(set(cle.code_insee[j.taux_commune_lisse.isna()]))
        raise ValueError(
            f"{j.taux_commune_lisse.isna().sum():,} lignes sans taux lissé, "
            f"{len(orphelines)} commune(s) : {orphelines[:10]}"
        )

    for c in ("cluster_id", "taux_cluster_lisse", "taux_commune_lisse",
              "ratio_commune_cluster"):
        df[c] = j[c].to_numpy()
    return df


def charger_taux() -> pd.DataFrame:
    f = PROCESSED / "taux_lisses.parquet"
    if not f.exists():
        raise FileNotFoundError(
            f"{f.name} absent — lancer d'abord :\n"
            "    python -m tvfed.clustering --methode kmeans --k 30")
    return pd.read_parquet(f)


# ════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methode", choices=["kmeans", "hdbscan"], default="kmeans")
    ap.add_argument("--k", type=int, default=30)
    args = ap.parse_args()

    print("profil territorial…")
    p = profil()
    print(f"  {len(p):,} communes × {p.shape[1] - 1} variables")

    print(f"clustering {args.methode} k={args.k}…")
    cl = ajuster(p, args.methode, args.k)
    tail = cl.value_counts().sort_values()
    print(f"  {cl.nunique()} clusters  ·  du plus petit ({tail.iloc[0]} communes) "
          f"au plus gros ({tail.iloc[-1]})")

    print("sinistralité sur le train complet…")
    sin = sinistralite()
    print(f"  {sin.jours.sum():,} jours-commune, {sin.feux.sum():,} feux, "
          f"taux {sin.feux.sum() / sin.jours.sum():.6%}")

    # ⚠️ CORINE est en COG 2010 et ne couvre pas 2 communes de la grille
    # (Saint-Lucien 76601, Essarts en Bocage 85212 — créées après).
    # Sans traitement explicite elles disparaîtraient du lissage par une
    # jointure interne, et se retrouveraient à NaN dans la matrice sans
    # qu'aucune erreur ne le signale. On les place dans un cluster -1 :
    # avec k0 = 20 000 pseudo-jours contre ~10 000 jours réels à deux, leur
    # taux de cluster est tiré aux deux tiers vers le taux national, ce qui
    # est exactement le comportement voulu faute de profil.
    manquantes = sorted(set(sin.code_insee) - set(cl.index))
    if manquantes:
        print(f"  ⚠️ {len(manquantes)} commune(s) sans profil CORINE "
              f"({', '.join(manquantes)}) → cluster -1, prior national")
        cl = pd.concat([cl, pd.Series(-1, index=manquantes, name="cluster_id")])

    print("lissage bayésien…")
    taux = lisser(sin, cl)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    taux.to_parquet(PROCESSED / "taux_lisses.parquet", index=False)
    p.assign(cluster_id=cl).to_parquet(PROCESSED / "profil_communes.parquet")
    (PROCESSED / "clustering.json").write_text(json.dumps({
        "methode": args.methode, "k": int(cl.nunique()),
        "k0_cluster": K0_CLUSTER, "k1_commune": K1_COMMUNE,
        "poids_position": POIDS_POSITION,
    }, indent=2), encoding="utf-8")

    # ── contrôle : le lissage a-t-il vraiment rempli le trou ? ──────────
    ref = taux[taux.an_exclue == 0].set_index("code_insee")
    tot = sin.groupby("code_insee").feux.sum()
    muettes = tot[tot == 0].index
    print(f"\n{'═' * 62}")
    print(f"{len(muettes):,} communes n'ont JAMAIS brûlé sur 2006-2019 "
          f"({len(muettes) / len(tot):.1%})")
    print(f"  taux empirique                  : 0 (inutilisable)")
    print(f"  taux lissé, médiane             : "
          f"{ref.loc[muettes, 'taux_commune_lisse'].median():.6%}")
    print(f"  · dont le plus bas              : "
          f"{ref.loc[muettes, 'taux_commune_lisse'].min():.6%}")
    print(f"  · dont le plus haut             : "
          f"{ref.loc[muettes, 'taux_commune_lisse'].max():.6%}")
    print(f"\n→ Le rapport entre la commune muette la plus exposée et la moins")
    print(f"  exposée est de "
          f"×{ref.loc[muettes, 'taux_commune_lisse'].max() / ref.loc[muettes, 'taux_commune_lisse'].min():.0f}. "
          f"C'est de l'information que le v1")
    print(f"  n'avait pas : pour lui, ces {len(muettes):,} communes étaient identiques.")
    print(f"\n✅ taux_lisses.parquet · profil_communes.parquet · clustering.json")


if __name__ == "__main__":
    main()
