# Terre, Vent, Feu, Eau, Data

Prédiction du risque de feu de forêt en France, à la maille **commune × jour**.

Le projet croise cinq sources publiques dans une base PostgreSQL/PostGIS pour
répondre à une question : *quel est le risque de feu de la commune X le jour J ?*

---

## Installation

### Prérequis

- Python **3.12+**
- [`uv`](https://docs.astral.sh/uv/) pour les dépendances
- **Docker Desktop** (la base tourne en conteneur)

### Mise en place

```bash
uv venv
uv pip install -e .
docker compose up -d          # PostgreSQL 16 + PostGIS 3.4 sur le port 5433
```

La chaîne de connexion par défaut est
`host=localhost port=5433 dbname=tvfed user=tvfed password=tvfed`,
surchargeable par la variable d'environnement `TVFED_DSN`.

### Données sources

Elles ne sont pas versionnées (`.gitignore`) — 758 Mo. À déposer dans `data/` :

| Dossier | Source | Contenu |
|---|---|---|
| `data/BDIFF/` | [bdiff.agriculture.gouv.fr](https://bdiff.agriculture.gouv.fr/incendies) | 6 archives ZIP, export par tranches |
| `data/copernicus/raw/cems/` | [EWDS · cems-fire-historical-v1](https://ewds.climate.copernicus.eu/datasets/cems-fire-historical-v1) | 1 NetCDF par année |
| `data/corine/` | SDES — CORINE Land Cover | `clc_etat_com_n3.csv` |
| `data/ville-france/` | référentiel communal | `communes-france-2026.csv` |
| `data/insee-cog/` | [INSEE COG 2026](https://www.insee.fr/fr/information/8740222) | `v_mvt_commune_2026.csv` |

Le téléchargement CEMS se fait par API (`cdsapi`) — voir
`notebook/data-copernicus.ipynb`. Il exige un compte **EWDS** (distinct du CDS).

---

## Construire la base

Les étapes s'enchaînent dans l'ordre et sont toutes **rejouables** :

```bash
docker exec tvfed-db psql -U tvfed -d tvfed -f /tmp/00_schema.sql
python -m tvfed.charger_ref       # référentiels        ~2 min
python -m tvfed.charger_faits     # faits observés      ~2 min
python -m tvfed.charger_grille    # grille 253 M lignes ~8 min
python -m tvfed.matrices          # matrices ML         ~1 min
```

Puis la modélisation :

```bash
python -m tvfed.baselines                          # le score à battre
python -m tvfed.modeles                            # RandomForest + XGBoost
python -m tvfed.optimisation --modele xgb --essais 60
```

Contrôles d'intégrité (7 assertions) :

```bash
docker exec tvfed-db psql -U tvfed -d tvfed -f /tmp/80_checks.sql
pytest
```

---

## Organisation

```
config/perimetre.yaml     périmètre, bornes du split, taux d'échantillonnage
sql/                      schéma et requêtes, numérotés dans l'ordre d'exécution
  ├── 00_schema.sql       DDL + partitions annuelles + index
  ├── 30_grille.sql       produit cartésien commune × jour
  ├── 31_split.sql        LA barrière temporelle
  ├── 50_matrice.sql      assemblage des features
  └── 80_checks.sql       assertions d'intégrité
src/tvfed/
  ├── io/                 un lecteur par source (formats hétérogènes)
  ├── geo.py              rattachement commune → maille météo, voisinage
  ├── charger_*.py        les 3 étapes d'alimentation
  ├── matrices.py         assemblage train / val / test
  ├── baselines.py        les prédicteurs de référence
  ├── modeles.py          RandomForest, XGBoost
  ├── optimisation.py     recherche Optuna
  └── calibration.py      correction des probabilités
notebook/                 audits par source + EDA croisée
figures/                  toutes les figures en PNG, un dossier par notebook
tests/                    27 tests, dont les garde-fous anti-fuite
```

### Les figures

Chaque notebook commence par :

```python
from tvfed.figures import activer
activer("data-all")
```

À partir de là, **chaque `plt.show()` enregistre aussi un PNG** dans
`figures/data-all/`, nommé d'après le titre de la figure et préfixé par son
rang. Sans ce helper, `plt.show()` affiche puis détruit la figure : elle ne
survit que dans les sorties du notebook et disparaît au premier
« Clear outputs ».

| Dossier | Figures |
|---|---|
| `figures/data-copernicus/` | 13 — audit CEMS, spirale, mémoires temporelles, cartes EFFIS |
| `figures/data-all/` | 8 — EDA croisée des cinq sources |
| `figures/data-bdiff/` | 7 — test du dimanche, biais de collecte, saisonnalité |
| `figures/data-corine/` | 4 — occupation du sol, combustible |
| `figures/modele-v1/` | 4 — performance, importances, lecture opérationnelle, calibration |
| `figures/data-ville/` | 3 — variables continues, catégorielles, cartes |

`src/` est la surface de vérité, les notebooks sont la surface de présentation.

---

## Le principe qui structure tout : la barrière du split

Le projet prédit un événement à **0,019 %**. À ce niveau de rareté, une fuite
de données ne produit pas d'erreur : elle produit d'excellentes métriques et un
modèle sans valeur. Trois règles la préviennent.

**1. La base ne stocke que des faits.**

> Une colonne est stockable si sa valeur pour *(commune c, date t)* est
> calculable **le jour t à 8 h**, sans connaître aucun `y` postérieur à t.

Tout ce qui a un `.fit()` — standardisation, clustering, lissage bayésien — vit
hors base et s'apprend après le split.

**2. Une feature datée peut regarder tout le passé ; une statistique non datée
ne peut regarder que le train.**

« Feux des 30 jours précédents » au 3 août 2023 lit juillet 2023, période de
test : ce n'est pas une fuite, car le 3 août à 8 h on connaît juillet. En
revanche « taux moyen de la commune sur toute la période » lit le futur.

**3. Les statistiques dérivées de `y` se calculent sur le train COMPLET.**

Sur le train échantillonné, le taux de positifs est de 9,12 % au lieu de
0,019 % — un facteur 480 qui empoisonnerait tout prior bayésien.

**4. Deux modèles ne se comparent que sur `(code_insee, date)`, jamais sur la
position de la ligne.**

`sql/50_matrice.sql` n'a pas d'`ORDER BY` : l'ordre dans lequel PostgreSQL
renvoie les 38 M lignes dépend du plan d'exécution et des workers parallèles,
et **change d'une exécution à l'autre**. Deux fichiers de prédictions issus de
deux exécutions ont la même taille, le même nombre de feux, et un ordre
différent — les comparer ligne à ligne donne un écart faux sans lever la
moindre erreur.

Le cas s'est produit : la première comparaison LSTM ↔ XGBoost annonçait
**−97 %** au lieu de **−52 %**. Les fichiers `predictions_val_v3/dart/mlp`
partageaient, eux, le même ordre — par chance, pas par contrat.

Ajouter un `ORDER BY` coûterait un tri de 38 M lignes larges à chaque
parcours. On a donc retenu l'autre parade : **tout fichier de prédictions
porte ses clés**, `tvfed.comparer.aligner()` trie et vérifie, et
`tests/test_comparaison.py` refuse un fichier sans clés.

---

## Périmètre et volumétrie

France métropolitaine, **2006-2025**. 2006 est la rupture de collecte BDIFF :
12 à 15 départements couverts avant, 42 à 93 après.

| Partition | Période | Lignes | Positifs | Taux |
|---|---|---|---|---|
| train | 2006-2019 | 177 594 942 | 33 632 | 0,0189 % |
| val | 2020-2022 | 38 068 464 | 9 176 | 0,0241 % |
| test | 2023-2025 | 38 068 464 | 6 322 | 0,0166 % |

**Val et test ne sont jamais échantillonnés** — c'est la validation intégrale
qui corrige le décalage de prior du downsampling. Le train est réduit à
368 826 lignes (100 % des positifs, 10 négatifs par positif).

---

## Limites connues

- **1 378 feux d'outre-mer** exclus : hors couverture météo européenne et
  absents de CORINE.
- **30 feux métropolitains** (0,06 %) non rattachables — codes « commune
  inconnue » ou scissions sans cible unique. Comptés, jamais devinés.
- **Département 64, 2010-2011** : données perdues, documenté par la BDIFF.
- **~31 communes partagent la même maille météo** de 28 km. Le FWI porte le
  *quand*, les features spatiales portent le *où*. Conséquence statistique :
  les intervalles de confiance naïfs sur les coefficients météo sont trop
  étroits (pseudo-réplication).
- **CORINE « révisée »** est produite après son millésime : techniquement une
  fuite temporelle, bénigne (raffinement de mesure d'un paysage statique).
- **Début 2006** : `feux_commune_365j` est sous-estimé, faute d'historique 2005.
- Le référentiel est figé en **COG 2026** : un feu de 2008 dans une commune
  fusionnée depuis est attribué à la commune d'aujourd'hui.

---

## Documentation

- [`sql/README.md`](sql/README.md) — schéma détaillé de la base
- [`docs/series-temporelles.md`](docs/series-temporelles.md) — cours ACF, PACF,
  ADF et SARIMAX, pour savoir quoi penser des résultats
- `notebook/data-*.ipynb` — audit qualité de chaque source
- `notebook/data-all.ipynb` — analyse croisée des cinq sources
- `notebook/series-lstm.ipynb` — l'axe temporel : stationnarité, ordres,
  SARIMAX, tendance sur 53 ans, et pourquoi le LSTM perd contre le modèle C
