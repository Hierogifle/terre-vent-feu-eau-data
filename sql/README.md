# Schéma de la base consolidée

Base `tvfed` — PostgreSQL 16 + PostGIS 3.4, conteneurisée (`docker-compose.yml`).

## Démarrage

```bash
docker compose up -d
docker exec tvfed-db psql -U tvfed -d tvfed -f /tmp/00_schema.sql
python -m tvfed.charger_ref       # référentiels
python -m tvfed.charger_faits     # faits observés
python -m tvfed.charger_grille    # grille commune × jour  (~8 min)
python -m tvfed.matrices          # matrices d'entraînement
```

Puis la chaîne de modélisation :

```bash
python -m tvfed.baselines                              # les points de repère
python -m tvfed.modeles                                # v1, RF et XGBoost
python -m tvfed.optimisation --modele xgb --essais 60  # Optuna
python -m tvfed.modele_v2                              # v2, réglé par Optuna
python -m tvfed.clustering --methode kmeans --k 30     # typologie + lissage
python -m tvfed.modele_v3                              # v3, + clustering
```

Connexion : `host=localhost port=5433 dbname=tvfed user=tvfed password=tvfed`
(surchargeable par la variable d'environnement `TVFED_DSN`).

## Organisation

Quatre familles de tables. **La frontière entre elles est la garantie
anti-fuite du projet** :

| Préfixe | Contenu | Peut contenir une statistique apprise ? |
|---|---|---|
| `ref_*` | référentiels stables | non — que du descriptif |
| `fait_*` | faits bruts observés | **non, jamais** |
| `grille` | cible `commune × jour` | non |
| `feat_*` | features de passé strict | non — fenêtres bornées à J-1 |

Ce qui n'entre dans **aucune** table : lissage bayésien, target encoding,
clusters, standardisation. Toute transformation avec un `.fit()` s'apprend sur
le train seul, après le split, et vit dans `data/processed/`.

### Les requêtes de la 60, hors tables

`60_profil_commune.sql` et `61_sinistralite.sql` ne créent rien : ce sont des
**lectures** consommées par `tvfed.clustering`, qui produit les features
apprises hors base. Elles portent chacune leur garde-fou :

| Requête | Ce qu'elle lit | Garde-fou |
|---|---|---|
| `60_profil_commune.sql` | carte d'identité physique de chaque commune | CORINE **millésime 2006** (≤ toutes les dates) et climatologie FWI bornée à **2006-2019** |
| `61_sinistralite.sql` | feux et jours par commune **× année** | bornes train en dur ; le détail annuel permet l'exclusion de l'année de chaque ligne |

Le profil ne lit ni `grille` ni `fait_feu` : un clustering construit sur la
sinistralité serait circulaire. Trois tests le vérifient
(`tests/test_clustering.py`).

## Tables

### Référentiels

| Table | Lignes | Rôle |
|---|---|---|
| `ref_maille` | 2 360 | grille météo CEMS 0,25° (dont 410 en mer) |
| `ref_commune` | 34 734 | référentiel COG 2026 + `cell_id` + `geom` PostGIS |
| `ref_calendrier` | 7 305 | un jour = une ligne, 2006-2025 |
| `ref_passage_cog` | 4 208 | ancien code INSEE → code 2026 |
| `ref_voisinage` | 19 056 340 | toutes les paires de communes ≤ 50 km |

**`ref_commune.cell_id` est la charnière du projet** : c'est cette colonne,
calculée une fois par rattachement du centroïde à la maille la plus proche,
qui permet à chaque commune d'aller chercher sa météo.

### Faits

| Table | Lignes | Source |
|---|---|---|
| `fait_meteo` | 8 261 955 | CEMS — 8 indices feu par maille et par jour |
| `fait_feu` | 52 809 | BDIFF — un feu = une ligne |
| `fait_clc` | 1 083 371 | CORINE — occupation du sol, format long |

La météo est stockée **au grain maille**, jamais dénormalisée vers la commune :
~31 communes partagent la même cellule, ce serait 31 fois la même valeur.

### Grille

`grille` — 253 731 870 lignes, **partitionnée par année** (20 partitions).

| Colonne | Sens |
|---|---|
| `y` | vrai si ≥ 1 feu ce jour-là dans cette commune |
| `nb_feux` | comptage — conservé pour un éventuel GLM Poisson |
| `surface_m2` | surface cumulée |
| `u` | tirage uniforme **déterministe** `hash(code_insee, date)` |

Le partitionnement rend le split temporel exact par construction et permet de
reconstruire une année sans toucher aux autres.

`u` est déterministe et non aléatoire : les échantillons obtenus à différents
ratios sont **emboîtés**, donc l'étude de sensibilité au ratio est propre.

### Vues

| Vue | Rôle |
|---|---|
| `clc_part` | occupation du sol **en part** de la surface communale |
| `grille_split` | grille + colonne `split` (train / val / test) |
| `echantillon` | grille + downsampling des négatifs du train |

`clc_part` convertit les hectares en parts : en absolu, un poste encode
surtout la **taille** de la commune.

## Volumétrie et split

| Partition | Période | Lignes | Positifs | Taux |
|---|---|---|---|---|
| train | 2006-2019 | 177 594 942 | 33 632 | 0,0189 % |
| val | 2020-2022 | 38 068 464 | 9 176 | 0,0241 % |
| test | 2023-2025 | 38 068 464 | 6 322 | 0,0166 % |
| **total** | | **253 731 870** | **49 130** | **0,0194 %** |

⚠️ **val et test ne sont jamais échantillonnés.** C'est la validation
intégrale qui corrige le décalage de prior introduit par le downsampling
(facteur ~×500). Elles ne sont pas non plus matérialisées : on les parcourt
par curseur serveur (`matrices.parcourir('val')`).

## Contrôles

```bash
docker exec tvfed-db psql -U tvfed -d tvfed -f /tmp/80_checks.sql
```

Sept assertions : grille rectangulaire, aucune commune en mer, météo
complète, feux rattachés valides, positifs cohérents, voisinage symétrique,
table de passage saine.

## Limites connues

- **1 378 feux en outre-mer** exclus : hors bbox météo européenne et absents
  de CORINE.
- **30 feux métropolitains** (0,06 %) non rattachables — codes « commune
  inconnue » ou scissions de communes sans cible unique. Comptés, jamais devinés.
- **Dept 64, 2010-2011** : données perdues, documenté par la BDIFF elle-même.
- **~31 communes par maille météo** : elles partagent le même FWI le même
  jour. Le FWI porte le *quand*, les features spatiales portent le *où*.
  Conséquence statistique : les intervalles de confiance naïfs sur les
  coefficients météo seront trop étroits (pseudo-réplication).
- **CORINE « révisée »** produite après son millésime — techniquement une
  fuite temporelle, bénigne (raffinement de mesure d'un paysage statique).
  La colonne `fait_clc.base` conserve la traçabilité.
