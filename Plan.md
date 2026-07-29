# Plan directeur — Terre, Vent, Feu, Eau, Data

> **Statut** : document vivant. À mettre à jour au fur et à mesure.
> **Objectif** : prototype de suivi et de prédiction du risque de feu de forêt en France.
> **Finalité personnelle** : pièce de portfolio / vitrine LinkedIn.

---

## 0. Comment utiliser ce document

- Les sections **1 à 3** sont le **contrat**. Elles ne bougent plus une fois validées.
- La section **6** est ton **fil directeur au quotidien**. Tu la coches.
- La section **5** est ta **liste de pièges**. Relis-la avant chaque phase.
- La section **9** (ce qu'on a écarté) est **aussi importante que le reste** : c'est ce qui te fera passer pour un professionnel en soutenance.

**Règle d'or du projet** : *une question → une figure → une réponse.* Tu ne construis rien sans savoir à quelle question ça répond.

---

## 1. Cadrage — le `PROBLEM.md`

### Produit 1 — Risque structurel *(livrable principal)*

| Élément | Décision |
|---|---|
| **Question** | « Quel est le risque de feu de la commune X le jour J ? » |
| **Unité d'observation** | **`commune × jour`** ⚠️ *conditionné au test du dimanche, ci-dessous* |
| **Cible (target)** | Binaire : y = 1 si ≥1 feu enregistré, sinon 0 |
| **Entraînement** | Grille commune × jour, 2006–2024, **downsampling négatif ~1:10** |
| **Split** | Train ≤2019 · Validation 2020–2022 · Test 2023–2024 |
| **Baseline 1** | Fréquence historique lissée (bayésien empirique) |
| **Baseline 2** | 🎯 Danger rating EFFIS **seul** |
| **Métriques** | PR-AUC (discrimination) + Brier + courbe de calibration |
| **Périmètre v1** | Méditerranéen : PACA + Occitanie + Corse (~5 800 communes) |
| **Sortie modèle** | Probabilité calibrée journalière |
| **Sortie affichage** | **Agrégation hebdo** `1 − Π(1 − pⱼ)` → 6 classes EFFIS |

### Produit 2 — Risque conjoncturel *(bonus)*

Identique, horizon J+7 — nécessite une **prévision** FWI, pas la réanalyse.

### Produit 3 — Curseur climat *(bonus, peu coûteux)*

GLM Poisson + curseur Streamlit `+0 / +1 / +2 / +4 °C` → recalcul de la carte de risque.

---

### 🚨 LA décision ouverte : jour vs semaine

> **Historique honnête** : `commune × semaine` était **un choix par défaut, jamais validé**, hérité de la formulation « une valeur par semaine de l'année **par exemple** » qui portait sur **l'affichage**, pas sur le modèle. Il est devenu faux dès qu'on est passé au FWI journalier.

#### Pourquoi le JOUR est maintenant le défaut

| Argument | Verdict |
|---|---|
| **Le FFMC a une latence de quelques heures/jours** | À la semaine, **l'indice ne veut plus rien dire**. On le détruit en le moyennant. |
| **Le FWI est volatil** | Un seul jour extrême déclenche le feu. La moyenne hebdo **dilue précisément le signal**. |
| **Calendrier (14 juillet, week-ends, fériés)** | Variables **journalières** par nature |
| **Hawkes** | Processus ponctuel en **temps continu** → les bacs de 7 jours le cassent |
| **Volumétrie** | ❌ **Faux argument** (voir calcul ci-dessous) |

#### Le calcul volumétrique honnête *(périmètre méditerranéen, pas la France entière)*

| | **Jour** | Semaine |
|---|---|---|
| Communes | ~5 800 | ~5 800 |
| Pas de temps (2006-2024) | 6 935 | 991 |
| Lignes | ~40 M | ~5,7 M |
| **Positifs** | **~80 000** | ~70 000 |
| Taux de positifs | **0,20 %** | 1,2 % |

> **La ligne qui compte : le nombre de positifs est quasi identique.** Passer à la semaine **ne gagne aucun positif** — ça ne fait que supprimer des négatifs. Or **ce qui compte pour l'apprentissage, c'est le nombre ABSOLU de positifs**, pas le taux. Le taux n'affecte que la métrique et la calibration — deux choses déjà gérées.
>
> *(Le « 12,7 M/an et 0,03 % » du premier échange était calculé sur la France entière. C'était un épouvantail.)*

#### 🎯 La technique qui rend le jour trivial : le downsampling négatif

Garder **100 % des positifs**, échantillonner les négatifs (1:10, 1:50…).

```
80 000 positifs + 800 000 négatifs = 880 000 lignes ≈ 200 Mo
→ entraînement en ~30 s → 50 itérations par jour
```

Méthode standard en prédiction de clic et détection de fraude — **exactement ton régime** (événement rare, gros volume).

**Correction de la probabilité** (formule fermée, si besoin) :

```
p_vrai = p_modèle / (p_modèle + (1 − p_modèle)/r)      # r = taux d'échantillonnage des négatifs
```

⚠️ **Mais tu n'en auras pas besoin** : `CalibratedClassifierCV` fitté sur une **validation NON échantillonnée** corrige le downsampling automatiquement.

#### ❓ Le SEUL vrai argument restant — **à trancher par les données**

> **La date de la BDIFF est-elle fiable au jour près ?**

- Date d'**éclosion** ? de **découverte/alerte** ? de **saisie administrative** ?
- Un feu parti à 23h le 14 est-il enregistré le 14 ou le 15 ?
- La précision est-elle la même sur toutes les décennies ?

| | Bruit de ±1-2 jours sur la date |
|---|---|
| **Semaine** | **Absorbé** — ±2 jours restent dans la même semaine |
| **Jour** | **Fatal** — le FWI du jour J est apparié à un feu du jour J−2. Le signal se dilue, **et rien dans tes métriques ne te le dira.** |

> **Analogie STAPS** : si ton chronomètre a ±0,5 s d'erreur, tu ne peux pas étudier des effets de 0,2 s. Ce n'est **pas** une question de puissance statistique — c'est le **plancher de résolution de l'instrument**. Aucun modèle ne récupère une information que la mesure n'a pas capturée.

#### ✅ LE TEST DU DIMANCHE — bloquant, à faire en phase 2

1. Lire les **métadonnées BDIFF** : quel champ exactement ? Y a-t-il une heure ?
2. **Compter les feux par jour de la semaine.**
   - **Creux le week-end** → ⚠️ ce n'est pas que les forêts brûlent moins le dimanche : **c'est que la saisie se fait en semaine**. Date administrative → **repli sur la semaine**.
   - Distribution plate → ✅ date d'éclosion crédible → **le jour est validé**.
3. **Test complémentaire** : pic anormal le 1er du mois → arrondi administratif → même conclusion.

> Cette figure est de toute façon dans la phase 2. **Tu l'auras gratuitement.**

#### La solution qui te donne les deux — modèle ≠ affichage

```
MODÈLE au JOUR            →  p(feu) pour chaque commune × jour
                                  ↓
AGRÉGATION à l'AFFICHAGE  →  1 − Π(1 − pⱼ)  sur les 7 jours
                             = « proba qu'AU MOINS UN des 7 jours parte en feu »
                                  ↓
STREAMLIT                 →  « risque de la commune X en semaine 32 »
```

**Tu gardes tout** : FWI journalier intact, 14 juillet, week-ends, Hawkes en temps continu — **et** l'affichage hebdomadaire que tu voulais.

---

## 2. Décisions actées et leurs justifications

*(À relire avant la soutenance : chaque « pourquoi » est une réponse au jury.)*

| Décision | Pourquoi |
|---|---|
| **Grille `commune × jour`** | Le FFMC (latence ~1 j) et le FWI sont **volatils** : la moyenne hebdo détruit le signal. La volumétrie n'est pas un obstacle (downsampling). ⚠️ *Sous réserve du test du dimanche.* |
| **Downsampling négatif ~1:10** | 100 % des positifs conservés. 880 k lignes, entraînement en 30 s → 50 itérations/jour. |
| **Modèle ≠ affichage** | Le modèle prédit au jour, l'app agrège à la semaine via `1 − Π(1 − pⱼ)`. Deux couches, pas un compromis. |
| **La BDIFF ne contient que des feux** | → il faut **construire les négatifs** par produit cartésien |
| **PR-AUC, pas accuracy** | Événement à ~0,2 % → un modèle qui prédit toujours 0 fait 99,8 % d'accuracy |
| **Split temporel strict** | Split aléatoire = feux du futur dans l'entraînement = fuite |
| **LightGBM** | Vitesse = nombre d'itérations possibles par jour |
| **GLM Poisson en parallèle** | Seul modèle qui **extrapole** (lien log) → indispensable au curseur climat |
| **CEMS, pas ERA5-Land** | Le FWI officiel existe déjà, **interpolé à midi local** (le problème que j'aurais bricolé). 3 jours économisés. |
| **Les 6 indices, pas juste le FWI** | FFMC (~1 j) · DMC (12 j) · DC (52 j) = **trois mémoires temporelles différentes**. Le FWI seul les écrase. |
| **HDBSCAN, pas DBSCAN** | `eps` global de DBSCAN inadapté : densité de feux variable d'un facteur 100 |
| **SHAP seul** | Une méthode d'interprétabilité bien exploitée > quatre survolées |
| **Streamlit seul** | Un livrable fini et déployé > deux à moitié |
| **Périmètre méditerranéen v1** | Complétude BDIFF + c'est là qu'est le signal |

---

## 3. Architecture en couches

```
COUCHE 1 — GRILLE
   commune × jour, y = 0/1
   (produit cartésien + LEFT JOIN BDIFF + downsampling négatif)
                 ↓
COUCHE 2 — FEATURES  « ils fabriquent des colonnes »
   HDBSCAN (typologie) · Voisinage (feux à 10/20/50 km)
   Lissage bayésien · Calendrier · sin/cos · lags
   FWI · FFMC · DMC · DC · BUI · ISI
                 ↓
COUCHE 3 — MODÈLE
   LightGBM (le socle) · GLM Poisson (curseur climat)
   [bonus : Hawkes]
                 ↓
COUCHE 4 — CALIBRATION
   CalibratedClassifierCV sur validation NON échantillonnée
   → probabilité journalière fiable
                 ↓
COUCHE 5 — AGRÉGATION AFFICHAGE
   1 − Π(1 − pⱼ) sur 7 jours → 6 classes EFFIS
                 ↓
COUCHE 6 — INTERPRÉTATION
   SHAP
```

### 🔑 Le point qui a créé la confusion (à ne pas réoublier)

**HDBSCAN n'est pas un modèle.** Il vit en **couche 2**. Il fabrique une colonne `cluster_id` + une carte. Il ne concurrence rien. **Tu le fais, point.**

**Hawkes et STGNN sont des modèles.** Ils vivent en **couche 3** et concurrencent LightGBM. C'est **là seulement** qu'il y a un arbitrage.

> Les trois encodent la **même intuition** (la contagion spatiale), mais **à des couches différentes** : HDBSCAN et le voisinage l'encodent *en features, à la main* ; Hawkes et STGNN l'encodent *dans le modèle, nativement*. D'où la confusion initiale — qui était légitime.

---

## 4. Données et sources

### 4.1 BDIFF — la cible

- **URL** : https://bdiff.agriculture.gouv.fr/incendies
- **Période** : 1973–2024
- **Granularité** : communale (code INSEE)
- ⚠️ **Limite technique** : CSV par paquets de 30 000 lignes → **boucle de téléchargement par département**
- ⚠️ **Biais de collecte** *(tu le connais déjà — à documenter explicitement)* : couverture fine de la zone méditerranéenne depuis 1973 (Prométhée), lacunaire ailleurs surtout avant 2006. **Un modèle entraîné là-dessus apprend en partie la géographie de la collecte, pas celle du risque.**
- 🔧 **Ton travail** : correction + adaptation aux normes de recueil en vigueur

### 4.2 Référentiel géographique

- **Communes + contours** : `admin-express` (IGN) → centroïde **et** surface
- **Codes** : COG de l'INSEE
- **Alternative** : https://adresse.data.gouv.fr/outils/telechargements
- 🎯 Sortie attendue : `code_insee → (lat, lon, geom, surface_km2)`

### 4.3 Densité de population

- **Source** : INSEE, un CSV, ~10 minutes
- **Rôle** : proxy de l'origine humaine (9 feux sur 10 sont d'origine humaine, la moitié par imprudence)

### 4.4 🎯 Danger météo — CEMS Fire danger indices ⭐ **LA source**

> 🚨 **DEUX COMPTES à ouvrir aujourd'hui** *(stores différents, configs API différentes)* :
> - **CDS** → https://cds.climate.copernicus.eu
> - **EWDS** → https://ewds.climate.copernicus.eu ← **celui-ci d'abord**

**`cems-fire-historical-v1`** — https://ewds.climate.copernicus.eu/datasets/cems-fire-historical-v1

Reconstruction historique complète des conditions météo favorables au départ, à la propagation et à l'entretien des feux. Produit par le CEMS pour l'**EFFIS**, calculé par le modèle **GEFF** sur la réanalyse **ERA5**.

| | |
|---|---|
| **Couverture temporelle** | **1940 → aujourd'hui** |
| **Mise à jour** | **Quotidienne** |
| **Résolution** | 0,25° × 0,25° *(natif : gaussienne réduite N320, ~31 km)* |
| **Fréquence** | Journalière |
| **Format** | **NetCDF** *(prends celui-là, `xarray` l'ouvre direct)* ou GRIB2 |
| **DOI** | `10.24381/cds.0e89c522` + Vitolo et al. (2020), *Scientific Data* |

> ⚠️ **Piège de lecture** : la fiche affiche « Publication date : 2019-09-30 ». **C'est la date de création de la fiche catalogue, PAS la fin des données.** Réflexe : *« Temporal coverage » = les données · « Publication date » = la paperasse.*

**Variables à prendre (système canadien uniquement — laisse le NFDRS US et le McArthur AU)** :

| Variable | Ce que c'est | |
|---|---|---|
| **Fire weather index** | Intensité potentielle du front. Non borné, **50 = extrême** | ⭐ feature n°1 |
| **Danger rating** | FWI en **6 classes EFFIS** harmonisées européennes | ⭐ échelle d'affichage |
| **Drought code** | Couches profondes 10-20 cm. **Latence 52 jours**, max ~800 | 🎯 **le bijou : la mémoire de la sécheresse longue** |
| **Fine fuel moisture code** | Litière fine. Échelle 0-99 (seule bornée). **Ignition ~70** | ⭐ |
| **Duff moisture code** | Couches 5-10 cm. Latence 12 jours | ✅ |
| **Build-up index** | Combustible total disponible | ✅ |
| **Initial spread index** | Propagation. **+13 km/h de vent double sa valeur** | ✅ |
| **Keetch-Byram drought index** | Déficit hydrique du sol, 0-200 | ✅ |
| **Fire daily severity index** | **Transformation exponentielle du FWI** | 🎯 cf. non-linéarité |

> 🎯 **Deux échelles de référence disponibles — ton app peut afficher les deux** :
> - **EFFIS** : 6 classes (très faible → extrême), harmonisées à l'échelle européenne
> - **Météo des forêts** : 4 classes, échelle française départementale
>
> *En soutenance : pourquoi la France a-t-elle choisi 4 niveaux quand l'Europe en utilise 6 ? → culture métier.*

#### ⏱️ Définition temporelle : **1 valeur par jour, à MIDI HEURE LOCALE**

Le FWI est **défini** à 12h locale. Or midi UTC à Brest ≠ midi UTC à Nice. Le CEMS résout ça par un **collage temporel et spatial** de 24 h de simulations : les champs sont découpés en bandes de 3 h puis concaténés pour représenter les conditions autour de midi local.

Ils documentent honnêtement la limite : des **artefacts apparaissent à l'interface entre deux bandes**. D'où une méthode plus récente de l'ECMWF : **moyenne pondérée entre les deux pas de temps les plus proches** → accord bien meilleur avec le cycle diurne réel, artefacts éliminés.

> 🎯 **C'est LA validation du choix CEMS.** C'est exactement le piège « le FWI se calcule à midi mais ERA5-Land ne donne que des max/min journaliers » — **ils l'ont résolu proprement, avec deux méthodes successives et de la littérature derrière.**

#### 🧠 Les 3 mémoires temporelles — **le point clé**

Le dataset est journalier, **mais chaque indice encapsule une fenêtre temporelle différente** :

| Indice | Couche physique | **Latence** | Se souvient de… |
|---|---|---|---|
| **FFMC** | Litière fine, 1-2 cm | **~1 jour** | **hier** |
| **DMC** | Couches lâches, 5-10 cm | **12 jours** | **les 2 dernières semaines** |
| **DC** | Couches profondes, 10-20 cm | **52 jours** | **les 2 derniers mois** |

Les dérivés sont des **combinaisons instantanées** : `ISI = FFMC + vent` · `BUI = DMC + DC` · `FWI = ISI + BUI`.

> **Analogie STAPS, et elle est exacte** : c'est un modèle **impulse-response** de charge d'entraînement. Le FFMC = **fatigue aiguë** (24-48 h). Le DC = **charge chronique** (plusieurs semaines). Le FWI = le **ratio aigu/chronique** qui dit si l'athlète est en zone de risque *aujourd'hui*.
>
> Et comme en STAPS : **les deux comptent, et ils ne disent pas la même chose.** Une forêt humide après 2 mois de sécheresse ≠ une forêt humide après 2 mois de pluie.

**→ Prends les 6, pas juste le FWI.** Laisse LightGBM apprendre quelle échelle de temps compte, et à quelle saison.

#### 🎁 Les seuils ET les couleurs officiels EFFIS

*(Extraits de l'exemple de code du guide utilisateur — les cartes telles qu'affichées sur EFFIS et GWIS.)*

| Classe | Seuil FWI | Couleur |
|---|---|---|
| Très faible | 0 – 5,2 | `#84F07F` |
| Faible | 5,2 – 11,2 | `#FFEB3C` |
| Moyen | 11,2 – 21,3 | `#FFB00C` |
| Élevé | 21,3 – 38 | `#FA4F00` |
| Très élevé | 38 – 50 | `#B40000` |
| Extrême | 50+ | `#280923` |

> **Ta palette Streamlit est faite.** Ta carte aura **exactement** les couleurs de l'EFFIS → reconnaissable au premier coup d'œil, crédibilité immédiate, zéro travail de design.

⚠️ **Incohérence dans leur propre doc** : le FFMC est « 0-101 » dans un paragraphe et « 0-99 » dans le tableau ; le FWI est « 0-100 » ici et « non borné » là. **Fie-toi à `df.describe()`, pas à la doc.**

> **Argument massue (version corrigée)** : *« J'ai croisé la BDIFF avec le produit officiel de danger météo du CEMS/EFFIS — celui qui alimente le système européen d'information sur les feux de forêt. »*
> **Savoir trouver et brancher la bonne source EST une compétence. Réimplémenter un indice qui existe déjà n'en est pas une.**

### 4.5 Météo brute — ERA5-Land ⚠️ en COMPLÉMENT seulement

**`derived-era5-land-daily-statistics`** (sur le CDS) — 0,1° (~9 km), 1950 → aujourd'hui, NetCDF.

Utile **uniquement** pour :
- `2m_temperature` → **le curseur climat en a besoin** (le GLM Poisson tourne sur la température)
- `volumetric_soil_water_layer_1` → humidité du sol, si tu veux compléter

| Variable | ⚠️ Piège |
|---|---|
| `2m_temperature` | **En Kelvin** → `− 273.15` |
| `2m_dewpoint_temperature` | → humidité relative (à combiner avec T et pression) |
| `10m_u_component_of_wind` | ⚠️ **Ce n'est PAS la vitesse**, c'est la composante Est |
| `10m_v_component_of_wind` | ⚠️ idem, composante Nord |
| `volumetric_soil_water_layer_1` | Volume d'eau, couche 0-7 cm |

```python
# Le vent : ERA5 donne 2 composantes vectorielles, pas une vitesse
vitesse_vent = np.sqrt(u10**2 + v10**2)                          # m/s
direction    = (180 + np.degrees(np.arctan2(u10, v10))) % 360    # degrés
```

#### 🚨 Les 3 pièges de ce dataset (raisons pour lesquelles on prend le CEMS)

1. **PAS DE PLUIE.** « Les variables accumulées sont omises (total precipitation, runoff…) ». → il faudrait un **2e dataset** (`reanalysis-era5-land` horaire) et agréger à la main.
2. **Le FWI se calcule à MIDI heure locale.** Ce dataset donne des **moyennes/max/min journaliers**, pas la valeur à 12h. Une option de décalage de fuseau existe, mais ça reste une approximation à documenter.
3. **L'agrégation est calculée à la volée**, pas archivée → **téléchargements lents**, files d'attente.

### 4.6 Calendrier

- `pip install holidays` → `holidays.France(years=range(1973,2025))`
- Vacances scolaires : `data.education.gouv.fr`
- ⚠️ **Le découpage des zones A/B/C a changé plusieurs fois depuis 1973** → piège classique, à documenter

### 4.7 Référence externe — la Météo des forêts

Diffusée par Météo-France depuis 2023 : niveau de danger **par département**, pour **J+1 et J+2**, sur **4 niveaux** (faible/modéré/élevé/très élevé, vert/jaune/orange/rouge). Publiée chaque jour à 17 h.

**Trois enseignements** :
1. **Elle ne prédit pas les feux**, elle prédit **la dangerosité des conditions**. Météo-France est explicite : il n'est pas possible de prévoir les incendies, la plupart étant déclenchés par imprudence. **Fais tienne cette distinction** — elle t'évitera de promettre l'impossible.
2. **Granularité départementale, horizon 2 jours.** Toi : **commune × jour**, affiché à la semaine. → **C'est ton positionnement.**
3. Ses intrants (météo + occupation des sols + zones sensibles) **valident ta liste de features**.

---

## 5. Les 7 pièges mortels

### 🔥 Piège 1 — La fuite de données

> **Définition** : une information qui **ne sera pas disponible au moment réel de la prédiction** se retrouve dans les features d'entraînement.
> **Symptôme** : des scores magnifiques. Ton code tourne, tes métriques sont superbes, tu es content. **Tu es en train de te planter.**
> **Analogie** : tu révises sur des annales avec le corrigé imprimé au dos. 20/20 en révision, 4/20 le jour J. Tu n'as pas appris la matière — **tu as appris à retourner la feuille.**

#### Fuite 1a — temporelle : le futur dans le passé

❌ `feux_voisins_20km` de la semaine 32/2021 **incluant** la semaine 32/2021.
Un gros incendie brûle Fos **et** 4 communes voisines **la même semaine** → la feature vaut 4 *parce que* y = 1. Corrélation 0,95. PR-AUC 0,99. **Mais le 3 août 2026, tu ne sais pas combien de feux les voisins auront cette semaine — c'est le futur.**

✅ Semaines **28 à 31** uniquement — **strictement avant**.

> **Le test** : *« Le jour J, à 8 h du matin, est-ce que je connais cette valeur ? »* Si non → fuite.

#### Fuite 1b — par le split : le test qui contamine le train

❌ `taux_voisin` calculé sur 1973–2024, puis split.
Bormes-les-Mimosas → `taux_voisin = 0.45`, élevé **notamment à cause de 2023**. Ta ligne de 2015 « sait » que le Var va cramer en 2023.

✅ **L'ordre EST la solution** :
```python
train = grille[grille.annee <= 2019]                     # 1. SPLIT D'ABORD
taux_voisin = train.groupby("cluster")["y"].mean()       # 2. APPRENDRE sur train seul → 0.31
grille["taux_voisin"] = grille.cluster.map(taux_voisin)  # 3. APPLIQUER partout
```

#### Le mantra

> *« Je peux **appliquer** au test une statistique apprise ailleurs.
> Je ne peux **jamais calculer** une statistique en regardant le test. »*

#### La question qui trie tout

> **« Pour calculer cette valeur, ai-je besoin de regarder d'AUTRES lignes que celle-ci ? »**

| | Transformation | Regarde d'autres lignes ? | Danger |
|---|---|---|---|
| ✅ | `sin(2π × jour/365)` | Non | Aucun |
| ✅ | `log(surface + 1)` | Non | Aucun |
| ✅ | `est_ferie(date)` | Non | Aucun |
| ⚠️ | `StandardScaler` | Oui (μ, σ) | Fuite |
| ⚠️ | Imputation par la moyenne | Oui | Fuite |
| ⚠️ | HDBSCAN, PCA | Oui | Fuite |
| 🔥 | `taux_voisin`, target encoding | **Oui — moyenne de y !** | **Majeure** |
| 🔥 | Lissage bayésien | **Oui** | **Majeure** |

**Règle mnémotechnique scikit-learn** : *si l'objet a une méthode `.fit()`, il est dangereux.* `fit` = « j'apprends des données » = « je peux apprendre du futur ».
**Ligne la plus rouge** : toute transformation qui utilise `y` est explosive.

#### Rappel — les 3 méthodes sklearn

| Méthode | Ce qu'elle fait |
|---|---|
| `.fit(X)` | **CALCULE** μ et σ sur X et les **mémorise**. Ne transforme rien. |
| `.transform(X)` | **APPLIQUE** les μ/σ déjà mémorisés. Ne recalcule rien. |
| `.fit_transform(X)` | Les deux d'un coup. ⚠️ **C'est ce raccourci qui fait fuiter.** |

```python
X_train, X_test = split(X, 2019)   # 1. SPLITTER D'ABORD. Toujours.
scaler.fit(X_train)                # 2. APPRENDRE sur le train SEUL
X_train = scaler.transform(X_train)  # 3. APPLIQUER
X_test  = scaler.transform(X_test)   # 4. APPLIQUER — mêmes μ/σ, on ne réapprend PAS
```

> *`StandardScaler` = z-score colonne par colonne. Exactement ce que tu fais en STAPS quand tu dis « il est à +2 écarts-types de la moyenne au Wingate ».*

#### Le garde-fou pratique

```python
def build_features(grille, periode_reference):
    """periode_reference : (debut, fin) = SEULE période où on a le droit d'apprendre."""
    ref = grille[grille.annee.between(*periode_reference)]
    taux_voisin = ref.groupby("cluster")["y"].mean()
    return grille.assign(taux_voisin=grille.cluster.map(taux_voisin))

df = build_features(grille, periode_reference=(2006, 2019))   # ← jamais 2024
```

#### 🚨 Le test du bon sens

**Si ta PR-AUC dépasse 0,80 sur un événement à 0,3 % → ne te réjouis pas. CHERCHE LA FUITE. Elle est là.**
Un bon modèle sur ce problème fait **0,20 – 0,35**. C'est la vraie vie.
Vérifie aussi tes SHAP : si une seule feature écrase toutes les autres, elle est probablement une reformulation de la cible.

---

### 🔥 Piège 2 — Les arbres n'extrapolent pas

**Structurel, pas un bug.** Un arbre prédit une **constante** dans chaque boîte. Entraîné sur 20–38 °C, interrogé à 42 °C → il sort **exactement la valeur de 38 °C**.

**Conséquence absurde et concrète** : ton curseur « +4 °C » ne montrera **aucune aggravation dans le Var** (déjà au plafond du range) et une aggravation **dans le Nord** (qui entre dans le range). **L'inverse de la réalité.**

✅ **Parade retenue — GLM Poisson à lien log** :

```
log(λ) = β₀ + β₁·T     ⟺     λ = exp(β₀ + β₁·T)
```

Le modèle est linéaire **dans le log**, donc **exponentiel dans le risque** → **il extrapole**, et il est **structurellement accélérant**. Avec β₁ = 0,15 :

| Scénario | Multiplicateur | |
|---|---|---|
| +1 °C | e^0,15 = **1,16** | +16 % |
| +2 °C | e^0,30 = **1,35** | +35 % |
| +4 °C | e^0,60 = **1,82** | **+82 %** |

> **+4 °C ne fait pas le double de +2 °C : il fait le CARRÉ du multiplicateur** (1,35² = 1,82). C'est exactement la non-linéarité que tu cherchais — **et elle est gratuite**, le lien log te la donne.

Options : ajouter un terme `T²`. ⚠️ Les **splines n'extrapolent pas bien** non plus → pour le curseur, reste sur log-link (+ éventuellement T²).

---

### 🔥 Piège 3 — Le déséquilibre de classes

~0,2 % de positifs à la maille jour sur le périmètre méditerranéen.
- ❌ Jamais d'accuracy
- ✅ PR-AUC + Brier + courbe de calibration
- ✅ **Downsampling négatif ~1:10** (100 % des positifs conservés) → 880 k lignes, 30 s d'entraînement
- ✅ `is_unbalance=True` ou `scale_pos_weight` dans LightGBM
- ⚠️ **Ce qui compte, c'est le nombre ABSOLU de positifs (~80 000), pas le taux.** Le taux n'affecte que la métrique et la calibration.

---

### 🔥 Piège 4 — La calibration

> Un modèle est **calibré** si, parmi toutes les cellules où il annonce 30 %, environ 30 % ont effectivement brûlé.
> *(La logique du « 70 % de chance de pluie ».)*

**C'est distinct de la discrimination** (savoir classer les risqués avant les autres). Un modèle peut très bien discriminer et être **totalement décalibré**.

**Tu affiches un score à l'utilisateur → la calibration n'est pas optionnelle.**
- Mesure : courbe de calibration + **Brier score**
- Correction : `CalibratedClassifierCV` (isotonic ou Platt), **fittée sur le set de validation**
- 🎯 **Bonus** : si la validation n'est **PAS** downsamplée, la calibration **corrige automatiquement** le downsampling. Tu n'as pas à appliquer la formule de correction.

---

### 🔥 Piège 5 — Le biais de collecte BDIFF

Déjà connu, mais **à écrire noir sur blanc** dans le rapport. C'est le genre de lucidité qui sépare un projet d'étudiant d'un projet de professionnel — et c'est un réflexe que ta formation recherche t'a déjà donné : **interroger la validité de la mesure avant d'interpréter le résultat.**

---

### 🔥 Piège 6 — Le plancher de résolution de la mesure

> **Aucun modèle ne récupère une information que la mesure n'a pas capturée.**

Si la date BDIFF est administrative (±1-2 jours), un modèle au jour apparie le FWI du jour J à un feu du jour J−2. **Le signal se dilue — et rien dans tes métriques ne te le dira.** Tu verras juste un modèle « moyen » sans comprendre pourquoi.

C'est le même problème qu'un chronomètre à ±0,5 s avec lequel on prétend mesurer des effets de 0,2 s. Ce n'est **pas** une question de puissance statistique.

→ **Le test du dimanche (phase 2) est bloquant.** Il détermine ta granularité.

---

### 🔥 Piège 7 — L'agrégation temporelle destructrice

Si tu dois replier sur la semaine, **l'agrégation n'est pas la même selon l'indice** :

| Indice | Comportement sur 7 j | ✅ Agrégation |
|---|---|---|
| **FFMC, ISI, FWI** | Très volatils | **`max`** *et* `mean` — *un seul jour extrême suffit, la moyenne le dilue* |
| **DC, DMC, BUI** | Quasi constants (latence 12-52 j) | **`mean`** ou dernière valeur — *le max n'apporte rien* |
| **Danger rating** | Classe discrète | **`max`** — le pire jour |

⚠️ `mean(FWI)` partout serait une **erreur silencieuse** : tu écraserais précisément le jour de canicule qui a déclenché l'incendie.
→ **Produis les deux** pour les volatils (`fwi_max`, `fwi_mean`). Laisse SHAP trancher. Coût : une colonne.

---

## 6. Feuille de route

### Phase 0 — Cadrage ⏱️ 30 min

- [ ] 🚨 **Ouvrir les DEUX comptes Copernicus** *(24 h de validation — en tout premier)*
  - **EWDS** → https://ewds.climate.copernicus.eu *(le FWI — la source principale)*
  - **CDS** → https://cds.climate.copernicus.eu *(la température — pour le curseur climat)*
- [ ] Écrire `PROBLEM.md` (section 1)
- [ ] Créer le repo `terre-vent-feu-eau-data`

> ❗ **Tu ne codes rien avant que `PROBLEM.md` existe.**

---

### Phase 1 — Ingestion ⏱️ 1 jour

- [ ] BDIFF : boucle de téléchargement par département (limite 30 k lignes)
- [ ] Corrections BDIFF + adaptation aux normes de recueil en vigueur
- [ ] `admin-express` (IGN) : contours + centroïdes + surfaces
- [ ] INSEE : densité de population
- [ ] 🎯 **`cems-fire-historical-v1`** (EWDS) → FWI, Danger rating, **Drought code**, FFMC, DMC, BUI, ISI — NetCDF, système canadien
- [ ] ERA5-Land daily stats (CDS) → **uniquement** `2m_temperature` (pour le curseur climat) + `volumetric_soil_water_layer_1`
- [ ] ❌ ~~Calcul du FWI maison~~ → **écarté** (cf. section 9)
- [ ] Calendrier : `holidays` + vacances scolaires
- [ ] Jointure spatiale : centroïde commune → maille 0,25° du CEMS

---

### Phase 2 — 🎯 EDA d'abord ⏱️ 1/2 jour

> **La toute première figure du projet**, avant tout modèle.

- [ ] 🚨 **LE TEST DU DIMANCHE — BLOQUANT** : feux par **jour de la semaine**
  - Creux le week-end → date **administrative** → ⚠️ **repli sur la grille semaine**
  - Distribution plate → date d'**éclosion** → ✅ **la grille jour est validée**
  - [ ] Complément : pic anormal le 1er du mois → arrondi administratif ?
  - [ ] Lire les **métadonnées BDIFF** : quel champ ? Y a-t-il une heure ?
- [ ] **Feux par jour de l'année, 1973–2024** (365 points) → **le test du 14 juillet**
- [ ] **Spaghetti plot cumulé** : x = jour de l'année, y = **cumul** des surfaces brûlées, une courbe par année, dégradé bleu (1973) → rouge (2024)
  - *Pourquoi le cumul : les courbes brutes se croisent et c'est illisible. Le cumul est monotone → les courbes s'empilent et les années récentes se détachent. (Le graphique classique de la banquise arctique.)*
- [ ] **Heatmap année × jour** → la saison qui s'allonge + la ligne verticale du 14 juillet si elle existe
- [ ] **Ridgeline / joyplot par décennie** (`pip install joypy`) → le pic saisonnier qui se décale et s'élargit
  - *C'est la figure que Météo-France utilise pour dire que la saison pourrait durer 1 à 2 mois de plus. Toi, tu la montres **sur données réelles**.*
- [ ] **STL** sur la série agrégée → tendance / saisonnalité / résidu
- [ ] Documenter les biais et les ruptures de collecte

> ⚠️ **Ces figures ne sont pas de la décoration avant les choses sérieuses.** Elles te disent quelles features construire, quel périmètre retenir, où sont les ruptures. Et en soutenance comme sur LinkedIn, **c'est ce que les gens regardent**. Un PR-AUC de 0,28 n'émeut personne. La heatmap 1973–2024, si.

---

### Phase 3 — La grille ⏱️ 1 jour · **le cœur du projet**

> ⚠️ **Ne commence pas avant le test du dimanche (phase 2).** Il détermine si tu bâtis au jour ou à la semaine.

- [ ] PostgreSQL + **PostGIS** *(pas SQLite : il te faut les jointures spatiales, et ça se conteneurise bien)*
- [ ] **Produit cartésien** `communes × jours` (~5 800 × 6 935 ≈ 40 M)
- [ ] **LEFT JOIN** BDIFF → `y = 1` si feu, `0` sinon
- [ ] ✅ **Vérifier** : ~80 000 positifs, taux ≈ 0,20 %
- [ ] **Downsampling négatif ~1:10** → ~880 k lignes pour l'entraînement
  - ⚠️ **Garde une validation NON échantillonnée** (elle sert à la calibration)
- [ ] Table de voisinage, calculée **une fois** :

```sql
CREATE TABLE voisins AS
SELECT a.code_insee AS commune,
       b.code_insee AS voisin,
       ST_Distance(a.geom::geography, b.geom::geography) AS dist_m
FROM communes a
JOIN communes b
  ON ST_DWithin(a.geom::geography, b.geom::geography, 50000)
 AND a.code_insee <> b.code_insee;
```
*(~5 800 communes × ~200 voisins dans 50 km ≈ 1,2 M lignes. Une fois pour toutes.)*

> **C'est ça, le vrai travail.** Pas le HDBSCAN. C'est ça qui impressionnera un recruteur.

---

### Phase 4 — Features ⏱️ 1,5 jour

- [ ] 🎯 **Jointure CEMS** ← **priorité n°1**, c'est là qu'est la performance
  - Centroïde commune → maille 0,25° → **6 indices journaliers** : FWI, FFMC, DMC, DC, BUI, ISI + Danger rating
  - *Le FWI porte le QUAND (temporel). Tes features spatiales portent le OÙ. Division du travail propre — et c'est l'argument qui justifie les 28 km en soutenance.*
- [ ] **Décalage spatial** (⚠️ **strictement `t-1` et avant**) :
```python
# nb de feux chez les voisins < 20 km, sur les 30 jours PRÉCÉDENTS
feux_voisins = (
    voisins.query("dist_m < 20000")
      .merge(grille, left_on="voisin", right_on="code_insee")
      .groupby(["commune", "date"])["nb_feux_30j_avant"].sum()
)
voisins["poids"] = 1 / (voisins["dist_m"] / 1000 + 1)   # version pondérée
```
  → produit `feux_voisins_10km`, `_20km`, `_50km`, `feux_voisins_ponderes`
- [ ] **Lissage bayésien** (⚠️ **fit sur train seul**) :
```python
k = 10  # nb d'années fictives empruntées au voisinage — à calibrer sur la validation
taux_lisse = (nb_feux + k * taux_voisin) / (n_annees + k)
# (0 + 10*0.40) / (30 + 10) = 0.10   au lieu de   0/30 = 0.00
```
  > *C'est ton **effet aléatoire de modèle mixte** en STAPS. Un sujet avec peu de mesures voit son estimation tirée vers la moyenne du groupe. Ici, le « groupe » est géographique.*
- [ ] **HDBSCAN** (⚠️ **fit sur train seul**) — sur le **vecteur de features de risque**, **PAS sur lat/lon seuls** *(sinon tu obtiens juste des paquets de communes proches : aucune information)*
- [ ] **KDE** pour la heatmap *(le bon outil pour la carte de chaleur — pas HDBSCAN)*
- [ ] Cycliques : `sin/cos(2π × jour_julien/365)`
- [ ] 🎯 **Calendaires — au jour, enfin** : `est_ferie`, `est_weekend`, `jour_semaine`, `est_14_juillet`, `est_15_aout`, `vacances_scolaires` (zone A/B/C)
- [ ] Lags : feux dans la commune sur 7/30/90/365 jours
- [ ] Contexte : densité pop, surface, altitude, distance côte

---

### Phase 5 — Baselines ⏱️ 1/2 jour · **non négociable**

- [ ] **Baseline 1 — spatiale** : fréquence historique lissée `commune × jour_de_l_année`
- [ ] 🎯 **Baseline 2 — météo : le Danger rating EFFIS tout seul**
- [ ] **Noter les deux scores.** Ton modèle doit battre **les deux**, sinon il ne sert à rien.

> *Réflexe direct de ta formation scientifique : pas d'effet sans groupe contrôle.*

#### Pourquoi la Baseline 2 est stratégique

**C'est LA question du jury** : *« à quoi sert votre modèle, puisque le FWI existe déjà ? »*
Avec cette baseline, la réponse est chiffrée et préparée :

> *« Le FWI seul atteint X de PR-AUC. Mon modèle atteint Y. Le gain vient de l'historique communal et du voisinage spatial — l'information que le FWI, par construction, ne contient pas : il est purement météorologique et ne sait rien de la végétation locale, de la fréquentation humaine, ni de l'historique du territoire. »*

**C'est ton positionnement en une phrase.** Il découle directement de la nature de la Météo des forêts : elle prédit **la dangerosité des conditions**, pas l'événement. Ton modèle ajoute la **dimension territoriale**.

⚠️ **Et s'il ne bat pas le FWI seul ?** Tu le dis. C'est un résultat légitime et publiable : la météo domine tout le reste — ce qui est cohérent avec le fait que Météo-France ne se base que sur ça.

---

### Phase 6 — Modèles ⏱️ 1,5 jour

- [ ] **LightGBM** — le socle :
```python
import lightgbm as lgb
model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    is_unbalance=True,          # ← le déséquilibre ~0,2 %
    random_state=42,
)
model.fit(X_train_downsampled, y_train_downsampled,
          eval_set=[(X_val, y_val)],            # ⚠️ val NON échantillonnée
          eval_metric="average_precision")      # = PR-AUC
```
- [ ] **GLM Poisson / Binomiale Négative** — interprétable + curseur climat
  - *La **binomiale négative** est la version robuste quand la variance dépasse la moyenne (**surdispersion**) — ce sera ton cas.*
- [ ] **Calibration** : `CalibratedClassifierCV` sur la validation **NON échantillonnée** → corrige le downsampling au passage
- [ ] **Agrégation affichage** : `1 − Π(1 − pⱼ)` sur 7 jours → **6 classes EFFIS** (seuils 5,2 / 11,2 / 21,3 / 38 / 50)
- [ ] Vérifier : **PR-AUC > 0,80 ⇒ chercher la fuite**

> 🚨 **NE FAIS PAS de benchmark LightGBM vs XGBoost vs CatBoost.** L'erreur de débutant la plus coûteuse.

| Levier | Gain réaliste |
|---|---|
| **Features** (IFM, voisinage, lissage, calendrier) | **~80 %** |
| **Qualité des données** (fuites, périmètre, biais) | **~15 %** |
| Choix de l'algo | ~5 % |
| Hyperparamètres | ~5 % |

> *Même logique qu'en recherche : le protocole et la qualité de la mesure déterminent le résultat. Le choix du test statistique ne sauve jamais une mesure bancale.*

---

### Phase 7 — Interprétation ⏱️ 1/2 jour

- [ ] **SHAP uniquement**
- [ ] Vérifier qu'aucune feature n'écrase les autres (→ signal de fuite)
- [ ] Rédiger les insights

---

### Phase 8 — Séries temporelles agrégées ⏱️ 1/2 jour *(onglet tendance)*

> ⚠️ **Les modèles de séries temporelles classiques ne s'appliquent PAS à ta grille creuse.** Une série temporelle = **une** suite de valeurs régulières. Ta grille = ~5 800 séries quasi-vides. **Ils s'appliquent aux séries AGRÉGÉES.**

- [ ] **Prophet** avec le paramètre **`holidays`** → il **quantifie l'effet du 14 juillet** avec un intervalle de confiance. 15 lignes de code. ⭐
- [ ] **SARIMAX** — le « X » permet d'injecter la température en exogène
- [ ] **STL** — déjà fait en EDA

---

### Phase 9 — Application ⏱️ 1,5 jour

- [ ] Streamlit, **Onglet 1** : cartographie + filtres + stats descriptives + les figures de la phase 2
- [ ] Streamlit, **Onglet 2** : prédiction (commune + période → score + classe EFFIS)
- [ ] 🎨 **Palette officielle EFFIS** : `#84F07F` `#FFEB3C` `#FFB00C` `#FA4F00` `#B40000` `#280923`
  - *Ta carte aura exactement les couleurs de l'EFFIS → reconnaissable au premier coup d'œil, crédibilité immédiate, zéro travail de design.*
- [ ] **Curseur climat** `+0/+1/+2/+4 °C` (GLM Poisson)
- [ ] **Déploiement sur Streamlit Cloud** → **lien cliquable dans le post LinkedIn = 10× l'impact**

---

### Phase 10 — Finalisation ⏱️ 1 jour

- [ ] Docker *(en dernier — c'est de l'emballage)*
- [ ] README technique + `requirements.txt`
- [ ] Rapport méthodologique (markdown)
- [ ] Section **limitations** *(cf. section 9)*
- [ ] Slides de soutenance

---

### Phase 11 — Bonus, si et seulement si tout tourne 🔵

- [ ] **Processus de Hawkes** (`pip install tick` — Inria — ou `hawkeslib`)

---

## 7. Le bonus : Hawkes

> **Processus ponctuel auto-excitant** : chaque événement **augmente temporairement la probabilité des événements suivants**, proches dans le temps et l'espace.

```
λᵢ(t) =    μᵢ         +    Σ  g(t − tⱼ) · h(dᵢⱼ)
        ┌──┴──┐            └──────┬──────┘
      risque de fond      excitation : chaque feu passé
      (végétation,        ajoute sa contribution, décroissante
       climat, humains)   dans le temps ET dans l'espace
```

- **μᵢ** = risque de fond de la commune
- **g(t − tⱼ)** = décroissance **temporelle** — un feu d'il y a 2 jours excite fort, d'il y a 6 mois plus du tout `≈ exp(−β·Δt)`
- **h(dᵢⱼ)** = décroissance **spatiale** — un feu à 3 km excite fort, à 80 km non `≈ noyau gaussien`

**Pourquoi c'est LE modèle de ton intuition initiale** — *« une ville au milieu de villes qui prennent toujours feu, factuellement son risque est à 0 %, pourtant en réalité elle est très probablement exposée »* :

**Hawkes formalise ça nativement.** Pas de bricolage de features : **la contagion EST l'équation**. Une commune à 0 feu historique entourée de feux récents aura une λ élevée **par le terme d'excitation**, même avec μᵢ faible.

Et physiquement, un feu **est** auto-excitant : braises portées par le vent → départs secondaires ; même vague de chaleur ; même sécheresse ; même pyromane.

> **Sa généalogie te vend le concept toute seule** : c'est le modèle **ETAS** des répliques sismiques, aussi utilisé en prédiction de criminalité.
> *« J'ai appliqué aux feux le formalisme utilisé pour les répliques sismiques. »* → **Ça, ça s'écoute.**

⚠️ **Franchise** : c'est le morceau le plus exigeant mathématiquement du projet. **2–3 jours.** Ne le lance **que** si LightGBM tourne déjà.

---

## 8. Glossaire

| Terme | Définition |
|---|---|
| **Baseline** | Modèle bête de référence. Sans elle, aucun score n'a de sens. |
| **Binomiale négative** | Régression de comptage robuste à la surdispersion |
| **Brier score** | Erreur quadratique moyenne sur des probabilités. Mesure la calibration. |
| **Calibration** | « Quand j'annonce 30 %, ça arrive 30 % du temps. » Distinct de la discrimination. |
| **DBSCAN** | Clustering par densité, `eps` **global** → inadapté à des densités variables |
| **Discrimination** | Savoir classer les risqués avant les non-risqués (PR-AUC) |
| **ERA5-Land** | Réanalyse météo Copernicus, grille 0,1°, 1950→, sans trou |
| **ETAS** | Modèle des répliques sismiques. Ancêtre de l'usage de Hawkes. |
| **Fuite de données** | Info indisponible le jour J présente à l'entraînement |
| **GBDT** | Gradient Boosting Decision Trees. Arbres **en séquence**, chacun corrigeant les erreurs des précédents. |
| **HDBSCAN** | DBSCAN **hiérarchique**. Plus de `eps`, juste `min_cluster_size`. Gère les densités variables. |
| **IFM / FWI** | Indice Forêt Météo. Agrège T°, humidité, vent, sécheresse. **Calcul récursif.** |
| **KDE** | Kernel Density Estimation. Le bon outil pour une carte de chaleur. |
| **Lien log** | `log(λ) = βX` → réponse **exponentielle**, extrapole, accélérante |
| **PR-AUC** | Aire sous la courbe précision-rappel. **La** métrique en classes déséquilibrées. |
| **Random Forest** | Arbres **indépendants**, moyennés. *(100 préparateurs qui ne se parlent pas.)* |
| **Shrinkage / lissage bayésien** | Tirer une estimation vers la moyenne du groupe quand peu d'observations. = **effet aléatoire** |
| **Small area estimation** | Le nom du problème « commune à 0 feu entourée de brasiers » |
| **Spatial lag** | Features calculées sur les voisins |
| **STGNN** | Spatio-Temporal Graph Neural Network. Communes = nœuds, adjacence = arêtes. |
| **Surdispersion** | Variance > moyenne. Fréquent en comptage. → binomiale négative |
| **z-score** | `(x − μ)/σ`. Ce que fait `StandardScaler`, colonne par colonne. |

---

## 9. Ce qu'on a écarté, et pourquoi

> 📌 **Section stratégique.** En soutenance, savoir dire *« j'ai écarté X pour la raison Y, et voici ce que j'aurais fait avec plus de temps »* vaut mieux que d'avoir tout survolé. **C'est exactement ce qu'on attend d'un professionnel.**

| Écarté | Raison |
|---|---|
| **Trafic routier** | Difficile à obtenir au niveau communal, gain marginal. Déjà capté par densité de population. |
| **SAGE + DiCE + LIME** | Quatre méthodes d'interprétabilité = name-dropping, ça se voit. **SHAP seul, bien exploité.** |
| **FastAPI + React** | Deux fronts = deux livrables à moitié. **Streamlit fini et déployé** > les deux inachevés. |
| **Analogue climatique** | Élégant (« en 2050, Valence aura le climat d'Avignon aujourd'hui » → on **transpose** au lieu d'extrapoler, donc on ne sort jamais du domaine d'entraînement) — **mais c'est un projet dans le projet.** Le curseur GLM coûte 1/2 journée et fait 80 % du boulot pédagogique. |
| **LSTM / ANN** | ANN = pas de mémoire, battu par LightGBM sur tabulaire. LSTM = besoin de séquences denses ; 34 800 séquences à 99,7 % de zéros. Et le vrai driver est **exogène** (la météo), pas dans la séquence. |
| **LSTNN / STGNN** | La **famille est conceptuellement parfaite** (communes = nœuds d'un graphe, la diffusion est native). Mais LSTNN est conçu et validé sur du **trafic routier** : cible continue, signal dense à chaque pas, driver **endogène**. Toi : binaire, creux, driver exogène. **Régime de données incompatible.** *Réactivable en agrégeant à `département × semaine` (96 nœuds, signal dense) — et alors commencer par STGCN ou DCRNN, pas LSTNN.* |
| **Benchmark d'algos** | 2 jours pour 0,5 % d'écart pendant que la vraie perf dort dans un feature non construit |
| **Stations Météo-France** | Des **points**, pas une couverture → interpolation à bricoler. ERA5-Land est déjà une grille. |
| **Calcul du FWI maison** | Le produit officiel CEMS/EFFIS existe (GEFF-ERA5, 0,25°). Le gain 28 → 9 km est **marginal pour une variable synoptique** : la différenciation inter-communale est portée par les features spatiales, pas par la météo. Coût évité : **3 jours**. |

### Phrases prêtes pour le README / la soutenance

> *« Le FWI a été récupéré depuis le produit officiel CEMS/EFFIS (GEFF-ERA5, 0,25°, Vitolo et al. 2020) plutôt que recalculé depuis ERA5-Land (0,1°). Le gain de résolution est marginal pour une variable synoptique, la différenciation inter-communale étant portée par les features spatiales. Un recalcul maison à 0,1° constitue une piste d'amélioration. »*

> *« L'extrapolation hors du domaine d'entraînement est gérée par le lien log du GLM Poisson. Une approche par analogue climatique (transposition spatiale) serait plus robuste et constitue une piste d'amélioration. »*

> *« Un STGNN a été envisagé : la formalisation en graphe correspond exactement à l'hypothèse de contagion spatiale. Il a été écarté pour incompatibilité de régime de données — la littérature STGNN est validée sur des signaux denses et endogènes (trafic routier), alors que les feux constituent un signal creux à driver exogène. »*

> **⚠️ Le résultat négatif est un livrable.** Si un modèle bat LightGBM de 0,3 %, ou se fait battre : **écris pourquoi**. Ton parcours recherche te donne un avantage ici : **tu sais qu'un résultat négatif bien documenté est un résultat.**

---

## 10. Points de contrôle

| # | Checkpoint | Critère de passage |
|---|---|---|
| 1 | `PROBLEM.md` écrit | 5 lignes, sans ambiguïté sur unité / cible / métrique |
| 2 | Comptes **CDS + EWDS** validés | Deux `cdsapi` qui répondent |
| 3 | 🚨 **LE TEST DU DIMANCHE** | **Bloquant.** Distribution plate → jour validé · Creux le week-end → repli semaine |
| 4 | La figure « 365 jours » | Le 14 juillet ressort ou non — **une réponse, dans les deux cas** |
| 5 | Grille construite | Lignes = communes × jours (~40 M). **~80 000 positifs**, taux ≈ 0,20 % |
| 6 | Downsampling | ~880 k lignes en train · **validation NON échantillonnée** conservée |
| 7 | **Les 2 baselines** scorées | Deux nombres noir sur blanc (historique lissé · Danger rating EFFIS) |
| 8 | LightGBM vs baselines | **Il bat les DEUX ?** Si non → features, pas hyperparamètres |
| 9 | Audit anti-fuite | PR-AUC < 0,80 ? SHAP sans feature dominante ? |
| 10 | Calibration | La courbe suit la diagonale |
| 11 | App déployée | Un lien public qui marche |

---

## 11. Ta prochaine action

1. 🚨 **Ouvrir les DEUX comptes Copernicus** — maintenant, il faut 24 h
   - **EWDS** → https://ewds.climate.copernicus.eu *(le FWI, la source principale)*
   - **CDS** → https://cds.climate.copernicus.eu *(la température, pour le curseur climat)*
2. **Écrire `PROBLEM.md`** — 30 min, 5 lignes
3. 🚨 **LE TEST DU DIMANCHE** — 1 h. **Il débloque toute l'architecture.**
```python
bdiff.groupby(bdiff.date.dt.dayofweek).size().plot.bar()
# 0 = lundi … 6 = dimanche
# Creux le week-end ? → date administrative → repli semaine
# Plat ?              → date d'éclosion    → le jour est validé
```
4. **La figure des 365 jours** — 2 h

> Cette figure va soit confirmer ton hypothèse du 14 juillet — et là tu as ton accroche LinkedIn **dès le jour 1** — soit la démentir, et tu auras appris quelque chose de **vrai** sur tes données avant d'avoir écrit une ligne de modèle.
>
> **Une question, une figure, une réponse.** Tu sais déjà faire ça, tu l'as fait en recherche. Le reste n'est que de l'outillage.

---

## 12. Journal des décisions renversées

*(Pourquoi cette section existe : chaque ligne est une hypothèse que j'avais posée par défaut et que **tu** as fait tomber en posant la bonne question. C'est le meilleur indicateur de la santé du projet. Continue.)*

| Ce qui était acté | Ce qui l'a renversé | Leçon |
|---|---|---|
| `commune × semaine` | *« Qui t'a dit que j'étais en semaine ? »* | Un choix par défaut jamais validé, devenu faux quand la source météo a changé. **Demande toujours qui a décidé quoi.** |
| Calculer le FWI soi-même | Tu as trouvé le dataset et lu la fiche | Le produit officiel existait. **Chercher avant de construire.** |
| `total_precipitation` dans ERA5-Land daily | Tu as ouvert la page | Les variables accumulées sont omises. **Lire la doc, pas la mémoire.** |
| « Le dataset s'arrête en 2019 » | Tu as tiqué sur la date | C'était la date de publication de la fiche. **Le réflexe était bon, la ligne était mauvaise.** |
| Volumétrie = 0,03 % de positifs | Recalcul sur le vrai périmètre | 0,20 %, et ~80 000 positifs. **C'était un épouvantail.** |
