"""Étape 20 — en 2050, à quel territoire d'aujourd'hui ressemblera le mien ?

    python -m tvfed.migration

────────────────────────────────────────────────────────────────────────────
L'IDÉE
────────────────────────────────────────────────────────────────────────────
Le clustering range les communes par ressemblance physique, et **trois de ses
vingt variables sont climatiques** : `fwi_moyen`, `fwi_p90`, `jours_fwi_sup_21`.

Si on remplace ces trois-là par leur valeur projetée en 2050 et qu'on demande
au même clustering où ranger la commune, certaines **changent de groupe**.
Une commune du Morbihan dont le climat 2050 rejoint celui du Var actuel
bascule dans le cluster du Var — et hérite du niveau de risque qu'on observe
aujourd'hui chez les communes varoises.

C'est la **substitution espace-temps** : utiliser la variation géographique
d'aujourd'hui pour lire l'évolution temporelle de demain. C'est la méthode
standard des études d'impact, et elle a ici un avantage rare — la population
de comparaison existe vraiment, on ne l'extrapole pas.

────────────────────────────────────────────────────────────────────────────
⚠️ ON PRÉDIT AVEC LE CLUSTERING EXISTANT, ON NE LE RÉAJUSTE PAS
────────────────────────────────────────────────────────────────────────────
Le point qui décide de la validité. Si on réajustait KMeans sur les données
2050, on obtiendrait 30 nouveaux groupes, sans rapport avec les 30 actuels :
« migrer du cluster 6 vers le cluster 28 » ne voudrait plus rien dire.

On garde donc le scaler ET le KMeans ajustés sur le présent, et on appelle
`.predict()` sur le profil projeté. Les clusters conservent leur identité :
c28 reste « maquis littoral », c6 reste « plaine agricole ».

────────────────────────────────────────────────────────────────────────────
⚠️ CORRECTION DE BIAIS PAR RAPPORT MULTIPLICATIF
────────────────────────────────────────────────────────────────────────────
Le FWI d'un modèle climatique porte son propre biais. On n'utilise donc jamais
sa valeur brute, mais le RAPPORT entre futur et historique du même modèle :

    k = FWI_rcm(2041-2055) / FWI_rcm(1986-2005)
    fwi_2050 = fwi_observé(2006-2019) × k

Le biais s'annule dans le rapport. Multiplicatif plutôt qu'additif car le FWI
est positif et très asymétrique — un écart additif pourrait le rendre négatif
dans les zones humides.

Pour le NOMBRE DE JOURS au-dessus de 21,3, un rapport ne s'applique pas
directement : on recompte les jours observés au-dessus du seuil **abaissé**
21,3 / k. Multiplier tous les FWI par k, ou abaisser le seuil d'autant,
donne exactement le même comptage — mais la seconde façon se calcule sur la
distribution réelle plutôt que sur une moyenne.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import clustering, db
from .modele_v3 import K, METHODE
from .paths import PROCESSED, RACINE

PROJ = RACINE / "data" / "projections"
SEUIL_EFFIS = 21.3


def ratios() -> pd.DataFrame:
    """Rapport futur/historique du FWI de saison, par maille météo du projet.

    Les projections sont sur une grille tournée à 0,11°, les observations sur
    une grille régulière à 0,25°. On rattache chaque maille du projet à la
    cellule de projection la plus proche — l'inverse (agréger le fin vers le
    grossier) serait plus juste mais l'écart est négligeable devant
    l'incertitude des scénarios eux-mêmes.
    """
    from scipy.spatial import cKDTree

    c = np.load(PROJ / "cartes_fwi.npz")
    g = np.load(PROJ / "grille.npz")
    lat, lon = g["lat"].ravel(), g["lon"].ravel()

    with db.connexion() as cx:
        mailles = pd.read_sql(
            "SELECT DISTINCT m.cell_id, m.lat, m.lon FROM ref_maille m "
            "JOIN ref_commune r USING (cell_id) WHERE r.in_perimetre", cx)

    arbre = cKDTree(np.column_stack([lat, lon]))
    _, idx = arbre.query(np.column_stack([mailles.lat, mailles.lon]))

    out = mailles[["cell_id"]].copy()
    hist = c["historique"].ravel()[idx]
    for s in ("rcp4_5", "rcp8_5"):
        fut = c[s].ravel()[idx]
        out[f"k_{s}"] = np.where(hist > 0, fut / hist, 1.0)
    return out


def jours_projetes(k: pd.DataFrame) -> pd.DataFrame:
    """Jours par an au-dessus du seuil EFFIS, une fois le climat décalé.

    Multiplier tous les FWI par k revient à abaisser le seuil à 21,3/k. On
    compte donc, sur les observations 2006-2019, les jours dépassant ce seuil
    abaissé — ce qui utilise la vraie distribution et non sa moyenne.
    """
    with db.connexion() as cx:
        m = pd.read_sql(
            "SELECT cell_id, fwi FROM fait_meteo "
            "WHERE date BETWEEN '2006-01-01' AND '2019-12-31'", cx)
    m = m.merge(k, on="cell_id", how="inner")
    out = {"cell_id": [], "jours_rcp4_5": [], "jours_rcp8_5": []}
    for cid, g in m.groupby("cell_id"):
        out["cell_id"].append(cid)
        kk = g.iloc[0]
        for s in ("rcp4_5", "rcp8_5"):
            seuil = SEUIL_EFFIS / max(kk[f"k_{s}"], 1e-6)
            out[f"jours_{s}"].append((g.fwi > seuil).sum() / 14.0)
    return pd.DataFrame(out)


def main() -> None:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    print("profil actuel…")
    p = clustering.profil()
    cols = [c for c in p.columns if c != "dep_code"]
    X = p[cols].astype(float).fillna(p[cols].astype(float).median())

    # le clustering du présent, conservé tel quel
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    for i, c in enumerate(cols):
        if c in ("lat", "lon"):
            Z[:, i] *= clustering.POIDS_POSITION
    km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(Z)
    p["cluster_actuel"] = km.labels_
    print(f"  {len(p):,} communes, {K} clusters")

    print("rapports climatiques par maille…")
    k = ratios()
    j = jours_projetes(k)
    k = k.merge(j, on="cell_id")
    with db.connexion() as cx:
        lien = pd.read_sql(
            "SELECT code_insee, cell_id FROM ref_commune WHERE in_perimetre", cx)
    k = lien.merge(k, on="cell_id").set_index("code_insee")
    # ⚠️ On rapporte le RAPPORT DES MOYENNES, pas la moyenne des rapports.
    # Les deux diffèrent nettement ici : 126 mailles ont un FWI de saison
    # inférieur à 5, et leur rapport individuel explose. Moyenner des rapports
    # donnait 1,43 contre 1,36 pour le bon calcul.
    print(f"  hausse du FWI de saison sur la France : "
          f"RCP4.5 ×{k.k_rcp4_5.mean():.2f} (moy. des rapports) — "
          f"voir migration_resume.csv pour le rapport des moyennes")

    # ── le profil 2050 : SEUL le climat change ──────────────────────────
    resume = []
    for s in ("rcp4_5", "rcp8_5"):
        q = p.copy()
        kk = k[f"k_{s}"].reindex(q.index)
        q["fwi_moyen"] = q.fwi_moyen * kk
        q["fwi_p90"] = q.fwi_p90 * kk
        q["jours_fwi_sup_21"] = k[f"jours_{s}"].reindex(q.index)

        Xq = q[cols].astype(float).fillna(X.median())
        Zq = sc.transform(Xq)          # ← le scaler du PRÉSENT
        for i, c in enumerate(cols):
            if c in ("lat", "lon"):
                Zq[:, i] *= clustering.POIDS_POSITION
        p[f"cluster_{s}"] = km.predict(Zq)   # ← le KMeans du PRÉSENT

        # ⚠️ À QUEL POINT L'AFFECTATION EST-ELLE CRÉDIBLE ?
        # Seules les 3 variables climatiques changent : la végétation reste
        # celle d'aujourd'hui. Une commune se retrouve donc décrite par une
        # COMBINAISON QUI N'EXISTE NULLE PART — bocage normand + climat
        # landais. KMeans l'affecte quand même au centre le plus proche, mais
        # ce centre ne représente pas ce mélange.
        #
        # La distance au centre mesure cet écart. Au-delà du 95e percentile
        # des distances observées AUJOURD'HUI, la commune n'a plus de vrai
        # équivalent : l'affectation devient une extrapolation, pas une
        # analogie. On le compte au lieu de le taire.
        d = np.linalg.norm(Zq - km.cluster_centers_[p[f"cluster_{s}"]], axis=1)
        seuil = np.percentile(
            np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1), 95)
        p[f"distance_{s}"] = d
        p[f"fiable_{s}"] = d <= seuil

        bouge = (p[f"cluster_{s}"] != p.cluster_actuel)
        resume.append({
            "scenario": s, "migrent": int(bouge.sum()), "part": bouge.mean(),
            # rapport des MOYENNES : la moyenne des rapports est dominée par
            # les mailles à faible FWI historique
            "k_rapport_des_moyennes": float(
                (p.fwi_moyen * kk).mean() / p.fwi_moyen.mean()),
            "k_moyenne_des_rapports": float(kk.mean()),
            "jours_avant": float(p.jours_fwi_sup_21.mean()),
            "jours_apres": float(q.jours_fwi_sup_21.mean()),
            "hors_analogie": int((~p[f"fiable_{s}"]).sum()),
        })
        print(f"  {s} : {bouge.sum():,} communes migrent ({bouge.mean():.1%}) · "
              f"{(~p[f'fiable_{s}']).sum():,} hors analogie")

    # ── à quoi ressembleront-elles ? ────────────────────────────────────
    sin = clustering.sinistralite()
    cl = p.cluster_actuel
    taux = clustering.lisser(sin, cl)
    risque = (taux[taux.an_exclue == 0]
              .groupby("cluster_id").taux_cluster_lisse.first())
    p["risque_actuel"] = p.cluster_actuel.map(risque)

    print(f"\n{'═' * 74}")
    print("CE QUE LA MIGRATION CHANGE")
    print("═" * 74)
    for s in ("rcp4_5", "rcp8_5"):
        p[f"risque_{s}"] = p[f"cluster_{s}"].map(risque)
        av, ap = p.risque_actuel.mean(), p[f"risque_{s}"].mean()
        pire = (p[f"risque_{s}"] > p.risque_actuel).sum()
        print(f"\n{s.upper().replace('_', '.')}")
        print(f"  risque de fond moyen : {av:.5%} → {ap:.5%}  "
              f"({100 * (ap / av - 1):+.0f} %)")
        print(f"  communes dont le risque MONTE : {pire:,} ({pire / len(p):.1%})")
        print(f"  jours de danger par an : {resume[0 if s == 'rcp4_5' else 1]['jours_avant']:.1f}"
              f" → {resume[0 if s == 'rcp4_5' else 1]['jours_apres']:.1f}")

    # les migrations les plus parlantes
    s = "rcp8_5"
    m = p[p[f"cluster_{s}"] != p.cluster_actuel].copy()
    m["gain"] = m[f"risque_{s}"] / m.risque_actuel
    top = m.nlargest(8, "gain")
    print(f"\n{'─' * 74}")
    print("LES BASCULEMENTS LES PLUS FORTS (RCP8.5)")
    print(f"{'commune':10s} {'dép':>4s}  {'cluster':>14s}  {'risque quotidien':>26s}")
    with db.connexion() as cx:
        noms = pd.read_sql("SELECT code_insee, nom FROM ref_commune", cx
                           ).set_index("code_insee").nom
    for c, r in top.iterrows():
        marque = "" if r[f"fiable_{s}"] else "  ⚠ hors analogie"
        print(f"{str(noms.get(c, c))[:10]:10s} {r.dep_code:>4s}  "
              f"c{int(r.cluster_actuel):>2d} → c{int(r[f'cluster_{s}']):<2d}      "
              f"{r.risque_actuel:.5%} → {r[f'risque_{s}']:.5%}  ×{r.gain:.0f}{marque}")
    print()
    print("⚠️ CE QUE CES CHIFFRES DISENT, ET CE QU'ILS NE DISENT PAS")
    print("   « à végétation constante, le climat de 2050 rapproche cette")
    print("     commune du profil de risque des Landes »   ← ce qu'ils disent")
    print("   « cette commune aura le risque des Landes »  ← ce qu'ils NE")
    print("     disent PAS. Le maquis ne pousse pas en dix ans, et un bocage")
    print("     humide qui se réchauffe ne devient pas une pinède.")

    p.to_parquet(PROCESSED / "migration_clusters.parquet")
    pd.DataFrame(resume).to_csv(PROCESSED / "migration_resume.csv", index=False)
    print(f"\n✅ migration_clusters.parquet · migration_resume.csv")


if __name__ == "__main__":
    main()
