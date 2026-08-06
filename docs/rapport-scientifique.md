# Prédire le risque de départ de feu à l'échelle communale en France métropolitaine

**Rapport scientifique**
Romuald Courtois · M1 Data / IA · La Plateforme_, Marseille · 2026

---

## Résumé

Le danger météorologique de feu de forêt est publié en France sur une maille
de 0,25°, soit environ 28 km de côté. À l'intérieur d'une telle maille se
trouvent en moyenne 31 communes, qui reçoivent le même indice qu'elles soient
couvertes de maquis ou entièrement bétonnées. Ce travail cherche à savoir si
l'on peut estimer, à l'échelle du couple commune × jour et à partir des seules
données publiques disponibles en temps réel, la probabilité qu'un départ de
feu soit déclaré.

Une grille dense de 253 731 870 couples commune × jour a été construite sur
2006-2025 en croisant quatre sources publiques. Huit familles de modèles ont
été comparées sur un découpage strictement temporel, au moyen d'un bootstrap
apparié rééchantillonnant les communes plutôt que les lignes.

Le modèle retenu atteint un lift de 63,7 sur une période de test ouverte une
seule fois, contre 42,1 pour la meilleure règle sans apprentissage. Trois
résultats méritent d'être signalés. D'abord, trois des cinq modèles comparés
sont statistiquement indiscernables entre eux, ce qui invalide toute
hiérarchie qu'on aurait pu en tirer. Ensuite, un réseau récurrent optimisé
perd 23,6 % contre un modèle à variables physiques agrégées, à information
égale, et trois analyses indépendantes convergent pour expliquer pourquoi.
Enfin, le modèle mis en service n'est délibérément pas le plus performant :
le meilleur dépend d'une variable qui n'existera pas au moment de la
prédiction.

**Mots-clés** : événement rare, classification spatio-temporelle, indices de
danger météorologique, estimation sur petits domaines, interprétabilité,
validation croisée spatiale.

---

## 1. Introduction

### 1.1 Contexte

Les feux de forêt en France métropolitaine suivent deux saisons distinctes,
et non une seule. Le pic estival, méditerranéen, est le plus connu. Un second
pic apparaît en mars dans le Sud-Ouest et le Massif central, sur des landes et
des herbacées sèches avant la reprise de végétation, souvent à la suite
d'écobuages qui échappent. Sur les 52 809 feux déclarés au fichier BDIFF entre
1973 et 2025 sur le périmètre d'étude, la part survenant hors saison estivale
atteint 52 % dans le Cantal, 44 % en Dordogne et 41 % en Lozère. Un modèle
n'apprenant que l'été manquerait environ un feu sur cinq.

Cette bimodalité impose de traiter la saisonnalité comme une variable
explicative et non comme un filtre appliqué en amont.

### 1.2 L'existant et sa limite

Le *Fire Weather Index* et les sept autres indices du système canadien sont
calculés quotidiennement par le Copernicus Emergency Management Service et
diffusés sur une grille régulière de 0,25°. Ils décrivent l'inflammabilité de
l'air et du combustible : c'est une information météorologique, et elle est de
bonne qualité.

Sa limite est spatiale. La France métropolitaine est couverte par 1 131
cellules pour 34 734 communes, soit environ 31 communes par cellule. Deux
communes partageant une cellule reçoivent un indice identique alors que leur
couvert végétal, leur relief et leur densité de population peuvent tout
opposer. La météo indique quand ; elle n'indique pas où.

L'apport visé n'est donc pas de remplacer ces indices, qui constituent
l'entrée principale du modèle, mais de leur adjoindre une description du
territoire à une résolution que le maillage météorologique n'atteint pas.

### 1.3 Question de recherche

> Peut-on estimer, à l'échelle du couple commune × jour, la probabilité qu'un
> départ de feu soit déclaré, à partir des seules données publiques
> disponibles en temps réel ?

La restriction finale est une contrainte que l'étude s'impose, et elle
détermine le modèle mis en service. Quatre hypothèses en découlent, formulées
de façon à pouvoir être réfutées.

**H1.** La météorologie seule ne suffit pas à localiser le risque.

**H2.** Le territoire porte une information que la météorologie ne contient
pas, et cette information se transfère à des zones non observées.

**H3.** Une architecture séquentielle n'apporte rien de plus que des indices
physiques agrégés.

**H4.** Le danger météorologique augmente de façon détectable sur cinquante
ans d'observation.

H3 est énoncée dans le sens qui permet sa réfutation : si le réseau récurrent
l'avait emporté, il aurait été déployé.

---

## 2. Données

### 2.1 Sources

Quatre jeux de données publics ont été croisés.

Le **CEMS Copernicus** fournit huit indices de danger quotidiens par cellule
de 0,25° sur 1973-2025, soit 21,9 millions de lignes. Outre le FWI, il publie
le *Fine Fuel Moisture Code* (litière de 1 à 2 cm), le *Duff Moisture Code*
(5 à 10 cm), le *Drought Code* (10 à 20 cm), le *Build-Up Index*,
l'*Initial Spread Index*, le *Keetch-Byram Drought Index* et
l'*Energy Release Component*.

La **BDIFF**, tenue par l'IGN, recense les feux déclarés commune par commune.
52 809 d'entre eux tombent dans le périmètre d'étude.

**CORINE Land Cover** décrit l'occupation du sol en 44 postes, dont huit
décrivent la végétation susceptible de brûler, pour 1,08 million de lignes.

Le référentiel **INSEE** fournit la liste des communes et, ce qui s'est révélé
indispensable, le fichier officiel des mouvements de communes.

### 2.2 Unité d'analyse

L'unité retenue est le couple **commune × jour**, et la cible est binaire :
au moins un départ de feu déclaré ce jour-là dans cette commune.

Ce choix a une conséquence qu'il faut énoncer. Un incendie traversant cinq
communes produit cinq observations positives. La quantité modélisée n'est donc
pas le nombre d'incendies mais le nombre de communes-jours touchés, ce qui est
une approximation acceptable pour un problème de ciblage mais interdit toute
lecture en nombre de sinistres.

### 2.3 Appariement et difficulté rencontrée

Le rattachement des feux aux communes s'est heurté aux fusions de communes.
965 feux portent un code INSEE qui n'existe plus dans le référentiel courant.

Un rapprochement par similarité de nom a d'abord été tenté, puis abandonné :
il produisait des faux positifs, la commune de Chirac en Lozère se voyant
rattachée à son homonyme de Charente. Le coût d'une telle erreur est élevé et
diffus, puisqu'elle corrompt simultanément la cible, le voisinage et les
variables d'occupation du sol de communes qui n'ont rien à voir avec le
sinistre.

Le fichier officiel des mouvements de communes de l'INSEE a donc été
téléchargé et appliqué. 935 feux ont été rattachés avec certitude. Les 30
restants, soit 0,68 % du total concerné, ont été **écartés et comptés**. Perdre
une fraction connue vaut mieux qu'en inventer une inconnue.

### 2.4 Limites de la cible

La BDIFF n'est pas exhaustive et le rapport ne le dissimule pas. Sa couverture
varie selon les départements et les périodes, et **64 % des causes déclarées
sont manquantes**. Un contrôle a été mené sur la qualité des dates, en
vérifiant l'absence de déficit le dimanche qui aurait signalé une date de
saisie administrative plutôt qu'une date d'éclosion.

Ces limites portent sur la variable à prédire, et non sur les variables
explicatives. Elles bornent donc ce que l'étude peut prétendre mesurer, et non
seulement sa précision.

---

## 3. Méthode

### 3.1 Construction de la grille

La table centrale contient une ligne par commune et par jour, qu'un feu s'y
soit déclaré ou non, soit 253 731 870 lignes pour 49 130 positifs sur
2006-2025. Elle est stockée dans PostgreSQL 16 avec l'extension PostGIS,
partitionnée par année.

Conserver les jours sans feu multiplie le volume par cinq mille. Ce coût est
assumé pour une raison technique précise : une série creuse rend les fenêtres
glissantes silencieusement fausses. La variable « nombre de feux dans les
trente jours précédents » se calcule en remontant trente lignes ; si les jours
sans feu sont absents, on remonte en réalité plusieurs années, et aucune
exception n'est levée.

Le partitionnement par année n'est pas cosmétique non plus : il fait coïncider
la frontière physique du stockage avec la frontière du découpage temporel.

### 3.2 Protocole d'évaluation

Le découpage est **temporel**, jamais aléatoire : entraînement sur 2006-2019,
validation sur 2020-2022, test sur 2023-2025.

| Partition | Années | Lignes | Feux | Taux |
|---|---|---|---|---|
| entraînement | 2006-2019 | 177 594 942 | 33 632 | 0,0189 % |
| validation | 2020-2022 | 38 068 464 | 9 176 | 0,0241 % |
| test | 2023-2025 | 38 068 464 | 6 322 | 0,0166 % |

Un découpage aléatoire placerait le 14 juillet 2019 dans l'entraînement et le
15 dans le test ; le modèle prédirait un événement observé la veille à
vingt kilomètres, et la métrique serait excellente sans que le modèle ait
aucune valeur.

Les cas ambigus ont été tranchés par une règle explicite : *une variable datée
peut consulter tout le passé, y compris celui de sa propre période
d'évaluation ; une statistique non datée ne peut consulter que
l'entraînement*. Ainsi « feux des trente jours précédents » évalué au 3 août
2023 lit juillet 2023 sans constituer une fuite, puisque cette information est
effectivement disponible au matin du 3 août, tandis qu'un « taux moyen de la
commune sur toute la période » consulterait le futur.

Le jeu de test a été ouvert **une seule fois**, après gel documenté du modèle,
des variables et de la calibration, daté du 28 juillet 2026. La justification
de cette rigidité est que le risque associé à un jeu de test est cumulatif :
chaque consultation en révèle un peu, et les décisions ultérieures en sont
insensiblement informées.

### 3.3 Métrique

À un taux de base de 0,019 %, l'exactitude est inutilisable : un classifieur
constant négatif atteint 99,98 %. Aucune mesure d'exactitude n'est rapportée.

L'aire sous la courbe précision-rappel a été retenue de préférence à l'aire
sous la courbe ROC. Cette dernière vaut 0,50 au hasard quel que soit le
déséquilibre et devient flatteuse lorsque les négatifs sont cinq mille fois
plus nombreux : l'ajout de faux positifs ne déplace presque pas le taux de
faux positifs. La PR-AUC vaut en revanche exactement le taux de base sous
l'hypothèse nulle, ce qui autorise la définition d'un rapport interprétable :

    lift = PR-AUC ÷ taux de base

### 3.4 Sous-échantillonnage

L'entraînement sur 177 millions de lignes étant impraticable, les négatifs de
la partition d'entraînement ont été **échantillonnés à 0,187 %**, tous les
positifs étant conservés, ce qui produit un ratio approximatif de un pour dix.
Aucune donnée n'est ajoutée ni dupliquée. Le tirage est déterministe et donc
reproductible.

Les techniques de sur-échantillonnage par interpolation n'ont pas été
retenues. Les variables décrivant des propriétés physiques du territoire,
interpoler entre deux communes éloignées produirait des combinaisons qui
n'existent pas.

Cette opération déplace le prior : le modèle apprend sur une population à
9,1 % de positifs quand la population réelle en compte 0,019 %, soit un
facteur 487. Trois précautions en découlent. Les partitions de validation et
de test ne sont jamais échantillonnées, ce qui préserve la comparabilité des
scores au monde réel. Les statistiques dérivées de la cible sont calculées sur
l'entraînement complet et non sur l'échantillon, faute de quoi tout lissage
hériterait d'un prior faux d'un facteur 487 sans qu'aucune métrique ne le
signale. Enfin la calibration absorbe le décalage résiduel.

### 3.5 Variables

Le modèle le plus complet compte 52 variables, réparties en cinq familles :
indices météorologiques du jour et de la veille, occupation du sol,
morphologie du territoire (altitude, amplitude altimétrique, superficie,
distance à la côte), démographie, et calendrier avec encodage cyclique du jour
de l'année.

Une étude d'ablation a testé l'ajout de dix décalages météorologiques
supplémentaires, portant le jeu à 62 variables. Le gain mesuré est de
**+0,02 %** et les variables ajoutées ne captent que **3,3 %** de l'importance
totale. Elles n'ont pas été retenues. Ce résultat sera repris en discussion :
il constitue un premier indice que la profondeur temporelle n'apporte rien à
ce problème.

### 3.6 Estimation sur petits domaines

Le diagnostic du premier modèle est sans ambiguïté : 54,6 % de son importance
provient de l'historique de sinistralité de la commune. Il énonce
essentiellement que ce qui a brûlé brûlera de nouveau. Cette proposition est
vraie mais laisse une lacune : une commune n'ayant jamais brûlé conserve un
score faible même lorsqu'elle est entourée de communes qui brûlent chaque été.
73 % des communes françaises, soit 25 297, sont dans ce cas sur la période
d'étude.

Il s'agit du problème classique de l'estimation sur petits domaines : les
événements sont trop rares pour estimer un taux commune par commune.

La réponse retenue procède en deux temps. Les communes sont d'abord regroupées
selon leurs caractéristiques **physiques** (végétation, relief, densité,
climatologie du FWI, position) par un algorithme des k-moyennes à trente
groupes. La cible n'intervient à aucun moment dans cette étape, faute de quoi
la construction serait circulaire. Les coordonnées géographiques sont incluses
mais pondérées à 25 %, dosage qui maintient la cohérence spatiale des groupes
sans les réduire à un découpage administratif. Deux communes dépourvues de
profil exploitable sont affectées à un groupe résiduel recevant le taux
national.

Un lissage bayésien hiérarchique fait ensuite converger le taux de chaque
commune vers celui de son groupe :

    taux_commune = (nb_feux + K₁ × taux_du_groupe) / (nb_jours + K₁)

avec K₁ = 2 000 et, un niveau au-dessus, K₀ = 20 000 pour le rappel du groupe
vers le taux national. La constante K s'interprète comme un nombre de jours
fictifs. Chaque commune disposant de 5 113 jours d'entraînement, le poids
accordé au groupe s'établit à 28,1 %.

Trois garde-fous protègent cette étape de la fuite. Les profils ne consultent
que le passé, soit CORINE 2006 et la climatologie 2006-2019. Les taux sont
agrégés sur l'entraînement complet. Enfin, pour une observation d'entraînement
de l'année Y, les taux **excluent l'année Y**, ce qui neutralise la fuite
classique de l'encodage par la cible.

### 3.7 Modèles comparés

Huit familles ont été évaluées, chacune répondant à une question précise :
trois règles sans apprentissage servant de référence, un gradient boosting
standard, sa version optimisée, sa version enrichie du lissage, une variante à
abandon d'arbres, un perceptron multicouche, un modèle restreint aux variables
physiques, et un réseau récurrent à mémoire longue recevant trente jours de
huit indices.

Les hyperparamètres du gradient boosting ont été recherchés par optimisation
bayésienne (Optuna, échantillonnage TPE, 60 essais). Ceux du réseau récurrent
l'ont été de la même façon sur 25 essais, avec arrêt précoce déclenché à la
vingt-et-unième époque.

### 3.8 Protocole de comparaison

Un écart de PR-AUC entre deux modèles ne constitue pas un résultat tant que
l'incertitude qui l'entoure n'est pas quantifiée. Un **bootstrap apparié** a
donc été mis en œuvre : les deux modèles comparés sont évalués sur exactement
les mêmes lignes rééchantillonnées, deux cents fois, ce qui donne accès à la
distribution de l'écart et non à deux distributions marginales.

Le rééchantillonnage porte sur les **communes** et non sur les lignes. Les
1 096 observations quotidiennes d'une commune ne sont pas indépendantes, et
trente-et-une communes partagent la même cellule météorologique. Un
rééchantillonnage ligne à ligne traiterait 38 millions d'observations comme
autant d'expériences indépendantes et produirait des intervalles de confiance
trop étroits, conduisant à conclure à des différences inexistantes. Il s'agit
d'une pseudo-réplication.

Le coût de calcul a été maîtrisé en triant les scores une seule fois : une
réplique se réduit alors à un vecteur de poids entiers le long d'un ordre figé,
et la précision moyenne pondérée se calcule en une passe par sommes cumulées.
L'équivalence numérique avec l'implémentation de référence a été vérifiée à
10⁻¹² près.

---

## 4. Résultats

### 4.1 Références sans apprentissage

| Prédicteur | PR-AUC | lift |
|---|---|---|
| hasard | 0,000241 | 1,0 |
| historique de la commune | 0,004668 | 19,4 |
| danger EFFIS seul | 0,001220 | 5,1 |
| historique × danger EFFIS | 0,010149 | **42,1** |

La météorologie seule atteint un lift de 5,1. Croisée à l'historique spatial,
elle atteint 42,1. Cet écart constitue une première réponse à **H1** et fixe
le seuil au-delà duquel un modèle appris devient utile.

### 4.2 Progression des modèles

Mesures sur la validation, 38 068 464 lignes et 9 176 feux.

| Modèle | PR-AUC | lift |
|---|---|---|
| Forêt aléatoire | 0,014950 | 62,0 |
| Gradient boosting v1 | 0,016593 | 68,8 |
| v2, hyperparamètres optimisés | 0,017463 | 72,4 |
| v3, avec lissage bayésien | **0,017684** | **73,4** |
| Abandon d'arbres (DART) | 0,017360 | 72,0 |
| Perceptron multicouche | 0,017340 | 71,9 |
| Modèle physique, 41 variables | 0,011174 | 46,4 |
| Réseau récurrent, 30 jours | 0,008542 | 35,4 |

L'apport du lissage bayésien s'élève à +0,83 %. Il est réel mais modeste, et
le rapporter comme tel fait partie du résultat.

Quatre combinaisons d'ensemble ont été évaluées. La meilleure, moyenne de
rangs du gradient boosting et du perceptron, atteint 0,018043 soit +2,03 %.
Elle n'a pas été retenue, pour des raisons exposées en section 5.3.

### 4.3 Comparaison appariée

| Comparaison | Écart | IC 95 % | Significatif |
|---|---|---|---|
| DART contre v3 | −1,8 % | [−5,0 ; +0,8] | **non** |
| MLP contre v3 | −1,9 % | [−7,9 ; +5,2] | **non** |
| Modèle physique contre v3 | −36,8 % | [−43,8 ; −24,0] | oui |
| Réseau récurrent contre v3 | −51,7 % | [−57,2 ; −43,4] | oui |
| Réseau récurrent contre modèle physique | **−23,6 %** | [−33,5 ; −17,3] | oui |

Le résultat le plus instructif de cette section est négatif. Les intervalles
associés à l'abandon d'arbres et au perceptron **contiennent zéro** : ces trois
modèles sont statistiquement indiscernables. Toute hiérarchie établie entre
eux à partir des seules valeurs ponctuelles relèverait du bruit.

Une lecture prudente s'impose : l'absence de significativité n'établit pas
l'équivalence, et une puissance statistique supérieure pourrait révéler un
écart. L'énoncé défendable est que ce protocole ne permet pas de les
départager.

### 4.4 Le réseau récurrent

À information égale, c'est-à-dire comparé au modèle physique dont il partage
l'absence d'historique de sinistralité, le réseau récurrent perd 23,6 %,
intervalle [−33,5 ; −17,3]. L'écart est donc solidement établi.

Ce résultat mérite explication, car le réseau dispose d'un volume
d'information météorologique vingt fois supérieur : trente jours de huit
indices, soit 240 valeurs, contre huit indices du jour et deux décalages pour
son concurrent.

L'explication est physique. Les indices DC, DMC et BUI **sont** des états
récursifs. Le *Drought Code* est une moyenne exponentielle de la météorologie
passée dont la constante de temps vaut 52 jours ; celle du *Duff Moisture
Code* vaut 15 jours. C'est la forme fonctionnelle d'une cellule récurrente, à
ceci près que ses coefficients ont été calibrés par un demi-siècle de
recherche sur le comportement du feu plutôt qu'estimés sur 9 176 exemples
positifs. Le service Copernicus livre donc déjà l'état caché que le réseau
devrait réapprendre.

Trois analyses indépendantes convergent vers cette conclusion.

**L'autocorrélation partielle.** Calculée sur les résidus du cycle annuel,
7 305 jours, elle décroît de 0,697 au premier retard à 0,191 au deuxième et
0,077 au troisième, pour un seuil de significativité de 0,023. La mémoire
utile de la série est de deux à trois jours. Les retards 4 à 8 demeurent
formellement significatifs, mais le retard 8, à 0,034, ne rend compte que de
0,11 % de la variance : la significativité statistique n'implique pas ici
l'utilité prédictive.

**La modélisation ARIMA.** Un modèle autorégressif sans variable exogène
produit une corrélation **négative** de −0,118 entre prévision et observation
sur l'horizon d'évaluation. À 1 096 pas, un processus dont la mémoire utile
vaut trois jours a perdu son point de départ et converge vers sa moyenne ; la
trajectoire quasi constante qu'il produit se trouve légèrement anti-corrélée à
l'observé. L'introduction du FWI en variable exogène réduit l'erreur absolue
moyenne de 37 %.

**L'ablation des décalages.** L'ajout de dix décalages météorologiques
supplémentaires au jeu de variables améliore la PR-AUC de 0,02 % et les
nouvelles variables ne captent que 3,3 % de l'importance.

Trois familles de méthodes, une conclusion identique : la prévisibilité du
départ de feu réside dans l'état météorologique courant et dans le territoire,
non dans la trajectoire temporelle passée.

**Réserve.** Le réseau récurrent n'a pas reçu la variable `danger_effis`, qui
pèse 13,7 % de l'importance du modèle physique. L'écart de 23,6 % constitue
donc un majorant, et la levée de cette asymétrie est le premier travail à
reprendre.

### 4.5 Ablation du regroupement territorial

| Configuration | Groupes obtenus | Gain relatif |
|---|---|---|
| aucun regroupement | — | référence |
| k-moyennes, k = 10 | 11 | +0,216 % |
| k-moyennes, k = 30 | 31 | +0,199 % |
| k-moyennes, k = 60 | 61 | +0,095 % |
| k-moyennes, k = 120 | 121 | +0,078 % |
| HDBSCAN | 5 | +0,022 % |

Le gain décroît avec le nombre de groupes, ce qui est cohérent avec la logique
de l'estimation sur petits domaines : au-delà d'un certain découpage, chaque
groupe redevient trop peu peuplé pour stabiliser l'estimation.

La configuration à dix groupes obtient un gain marginalement supérieur à celle
à trente. L'écart, de 0,017 point de pourcentage, ne saurait être considéré
comme significatif au vu de la variabilité observée ailleurs dans cette étude,
et le choix de trente groupes a été conservé pour la granularité descriptive
qu'il offre à l'application. Ce choix n'est pas justifié par la performance.

HDBSCAN, appliqué avec un paramétrage visant une trentaine de groupes, n'en
produit que cinq et obtient le gain le plus faible. La structure du territoire
français ne présente apparemment pas la séparation en densité que cet
algorithme recherche.

### 4.6 Transfert spatial

Une validation croisée par région a été conduite : chaque région est retirée
de l'entraînement, le modèle est réajusté, puis évalué sur la seule région
exclue. L'opération simule un territoire jamais observé.

Le modèle restreint aux variables physiques l'emporte sur le modèle complet
dans les **neuf régions sur neuf**, avec un gain moyen pondéré de 8,2 % et un
maximum de 137 % en Grand Est.

Ce résultat répond directement à **H2**. Là où l'historique de sinistralité
est le plus pauvre, s'y fier devient un handicap : la variable prend la valeur
zéro, que le modèle interprète comme une absence de risque plutôt que comme
une absence d'information. Les variables de territoire, elles, conservent leur
pouvoir prédictif hors du domaine d'apprentissage.

### 4.7 Calibration

Le score brut surestime la probabilité d'un facteur 144,7, conséquence directe
du sous-échantillonnage.

La régression logistique de Platt ramène ce biais à 1,13 sans coût mesurable
en PR-AUC, et préserve les neuf millions de valeurs distinctes du score. La
régression isotonique atteint une qualité de calibration comparable mais
réduit le score à 136 valeurs distinctes, ce qui détruit le pouvoir de
discrimination à l'intérieur de chaque palier.

Ces calibrations n'ont finalement pas été employées dans l'application
publiée : le calibrateur disponible ayant été ajusté sur un autre modèle et
une autre période, il serait faux d'un facteur voisin de deux. L'application
affiche un **rang** plutôt qu'une probabilité, choix documenté et non omission.

### 4.8 Évaluation finale

Sur la période 2023-2025, ouverte une seule fois :

| Modèle | PR-AUC | lift |
|---|---|---|
| Gradient boosting v3, 52 variables | 0,0156 | 93,8 |
| **Modèle physique, 41 variables** | **0,0106** | **63,7** |

La décomposition par année révèle une variation du simple au double, qui suit
la rareté plutôt que le hasard.

| Année | Feux | lift |
|---|---|---|
| 2023 | 2 591 | 75,6 |
| 2024 | 1 297 | **151,4** |
| 2025 | 2 434 | 87,0 |

L'année la plus calme produit le meilleur lift. Une interprétation cohérente
est qu'une saison peu active concentre les départs dans les configurations les
plus prévisibles, tandis qu'une saison intense en produit également là où le
modèle ne les attend pas.

### 4.9 Rendement opérationnel

Le lift ne se traduit pas directement en décision. La quantité pertinente pour
un service opérationnel est le rappel atteint sous contrainte de budget de
surveillance.

| Budget | Communes-jours suivis | Feux couverts | Rappel |
|---|---|---|---|
| 0,1 % | 38 068 | 924 | 14,6 % |
| 1 % | 380 684 | 2 650 | **41,9 %** |
| 10 % | 3 806 846 | 5 372 | 85,0 % |

En surveillant un centième du territoire-temps, le modèle couvre 42 % des
départs.

Ce cadrage éclaire d'un jour différent l'écart de 37 % de PR-AUC séparant les
deux modèles : il ne vaut que **trois points de rappel** à 1 % de budget, et
devient nul à 10 %. Une part importante de l'écart mesuré est
opérationnellement invisible.

### 4.10 Tendance climatique

| Série | Période | Variation | p | Conclusion |
|---|---|---|---|---|
| FWI moyen annuel | 1973-2025 | +58 % | 4,2 × 10⁻⁵ | significatif |
| FWI moyen juin-septembre | 1973-2025 | +62 % | 1,5 × 10⁻⁴ | significatif |
| jours de danger élevé | 1973-2025 | +197 % | 5,2 × 10⁻⁵ | significatif |
| communes-jours en feu | 2006-2025 | +3 % | 0,89 | **non significatif** |

L'aléa météorologique augmente de manière très significative sur cinquante-deux
années d'observation, ce qui répond affirmativement à **H4**. Le nombre de
départs, mesuré sur vingt années seulement, ne présente aucune tendance
détectable.

Trois lectures restent compatibles avec ces observations : une puissance
statistique insuffisante sur vingt points, une politique de prévention
absorbant pour l'instant la hausse de l'aléa, ou un déplacement du bilan vers
des sinistres plus étendus plutôt que plus nombreux. **Les données mobilisées
ici ne permettent pas de trancher**, et il serait abusif de présenter la
première série sans la seconde.

---

## 5. Interprétation

### 5.1 Ce que le modèle a appris

Les contributions ont été décomposées par la méthode TreeSHAP, exacte sur un
modèle à base d'arbres. Sa correction a été vérifiée : la somme des
contributions augmentée de la valeur de base restitue le logarithme des cotes
du score à 9,5 × 10⁻⁷ près.

Trois mesures d'importance ont été confrontées, et elles ne produisent pas le
même classement. L'importance par gain, sortie par défaut de l'implémentation,
mesure la réduction de perte obtenue à l'entraînement ; elle est biaisée en
faveur des variables qui séparent proprement et **aveugle à la redondance**.
Les contributions moyennes sur échantillon aléatoire décrivent le territoire
tel qu'il est, à 99,97 % exempt de feu. Les contributions au sommet du
classement décrivent le comportement du modèle là où il s'engage.

Deux désaccords éclairent le fonctionnement du modèle. La classe de danger
EFFIS occupe le deuxième rang par gain et le trentième par contribution
moyenne : cette variable est une discrétisation du FWI en six classes, dont
les seuils nets facilitent le découpage, mais l'information sous-jacente est
déjà présente dans le FWI continu auquel la décomposition en attribue le
crédit. La part de maquis occupe le premier rang par gain, le dixième sur
échantillon aléatoire et le deuxième au sommet : elle ne déplace rien sur une
commune-jour ordinaire, où le maquis est absent, et devient déterminante là où
le modèle identifie du risque.

Il en résulte qu'aucune assertion sur « la variable la plus importante » n'a de
sens sans spécification de la mesure employée et de la population considérée.

La lecture d'ensemble confirme **H1** et **H2** conjointement : les variables
de territoire occupent le haut du classement à parité avec les indices
météorologiques. La météorologie situe le moment, le territoire situe le lieu.

Une précaution méthodologique mérite d'être signalée. La sélection des
observations sur le score constitue un conditionnement sur un collisionneur,
le score étant causé conjointement par la météorologie et le territoire. Une
analyse d'interaction menée sur le seul sommet du classement produisait un
signe inversé. Le panneau correspondant n'est donc interprété que pour la
question « lorsque le risque est élevé », jamais pour une relation générale.

### 5.2 Analyse contrefactuelle

La génération de contrefactuels répond à une question distincte : non pas
pourquoi ce score, mais que faudrait-il modifier pour l'infléchir.

Une difficulté d'implémentation a dû être levée. L'outil recherche par défaut
un basculement de la classe prédite au seuil de 0,5. Le score n'étant pas
calibré, ce seuil correspond à un niveau de risque inatteignable et aucune
solution n'était retournée. La frontière a été recentrée sur le décile par une
transformation affine par morceaux strictement croissante, qui laisse le
classement inchangé. La question devient alors « que faudrait-il pour sortir
des 10 % les plus exposés », qui possède un sens opérationnel.

Appliquée à Bormes-les-Mimosas au 12 août 2024, en n'autorisant que la
modification du couvert végétal, la recherche ne retourne **aucune solution**.
L'exposition de cette commune tient à sa position, à son relief et à sa
superficie, et non à des leviers actionnables. L'absence de contrefactuel
constitue ici le résultat.

Un contrefactuel ne saurait au demeurant être lu comme une recommandation.
Rien ne garantit sa réalisabilité, ni le caractère causal de la relation : le
modèle a estimé des corrélations sur 2006-2019, non des mécanismes.

### 5.3 Le choix du modèle mis en service

Le modèle le plus performant n'a pas été déployé. Cette décision demande
justification.

Le gradient boosting v3 tire 29 % de son importance de l'historique de
sinistralité. Or la BDIFF ne publie pas l'année en cours : les sinistres de
2026 seront diffusés au printemps 2027. Une prédiction produite aujourd'hui
verrait donc cette variable renseignée par un décompte arrêté à décembre 2025.
Elle ne serait pas imprécise, elle serait fausse.

En territoire non observé, la même variable prend la valeur zéro, que le
modèle interprète comme une absence de risque là où se manifeste précisément
le risque nouveau. La validation croisée spatiale de la section 4.6 quantifie
ce phénomène. Enfin, pour toute projection à horizon 2050, la variable est
inaccessible par construction.

Ce défaut n'apparaît dans **aucune métrique d'entraînement** : en validation
comme en test, l'historique est disponible. Il ne se manifeste qu'en
considérant les conditions réelles de mise en service.

La même logique a écarté l'ensemble, malgré son gain de 2,03 % : il double la
chaîne de service et rompt la chaîne d'explicabilité, pour un gain qui, à la
lumière de la section 4.9, ne se traduirait par aucune amélioration
opérationnelle observable.

### 5.4 Projections

Les projections à horizon 2100 appliquent un facteur multiplicatif issu des
trajectoires du GIEC à la climatologie observée sur 2006-2019. La forme du
cycle saisonnier provient donc des observations ; seul son niveau est déplacé.

La quantité projetée est l'**aléa météorologique**, et non le nombre de
sinistres. Le couvert végétal, les politiques de prévention et les pratiques
agricoles sont supposés constants, hypothèse dont on sait qu'elle est fausse.

Avant 2045, les trois trajectoires sont indiscernables. En 2030, la
trajectoire la plus sobre dépasse la plus pessimiste sur 59 % des communes.
Ce comportement, qui pourrait passer pour une anomalie de calcul, traduit
l'inertie du système climatique : les trajectoires d'émissions ne divergent
substantiellement qu'après le milieu du siècle. L'application signale
explicitement cette zone.

---

## 6. Discussion

### 6.1 Réponses aux hypothèses

**H1** est confirmée. La météorologie seule atteint un lift de 5,1 ; croisée
au territoire elle atteint 42,1. La décomposition des contributions place les
variables de territoire à parité avec les indices météorologiques.

**H2** est confirmée. Le modèle restreint aux variables physiques l'emporte
dans les neuf régions retirées de l'entraînement, jusqu'à 137 %. Le territoire
transfère ; l'historique ne transfère pas.

**H3** est confirmée, avec la réserve énoncée en 4.4. Trois familles de
méthodes convergent : autocorrélation partielle, modélisation ARIMA, ablation
des décalages.

**H4** est confirmée pour l'aléa et non vérifiée pour les sinistres. La
distinction est essentielle et conditionne toute communication sur ces
résultats.

La réponse à la question principale est positive, sous une réserve explicite :
l'étude estime un **risque relatif** et non une probabilité absolue.

### 6.2 Ce que ces résultats ne permettent pas d'affirmer

Le modèle ne prédit pas la surface brûlée, dont le coefficient de
détermination atteint 0,144, soit une performance inférieure à celle d'une
prédiction constante par la médiane. La surface dépend de ce qui advient
**après** le départ : vent, délai d'intervention, relief.

La distinction entre grand et petit sinistre se prédit également mal, avec un
lift de 2,88. Une aire sous la courbe ROC de 0,766 peut être calculée sur
cette tâche, mais son emploi serait contradictoire avec l'argumentation de la
section 3.3.

Le modèle ne prédit pas davantage le nombre de sinistres futurs, ni leur
localisation en 2050.

### 6.3 Menaces sur la validité

**Validité interne.** Le principal risque est la fuite d'information, d'autant
plus insidieuse qu'à ce taux de base elle ne produit aucune erreur détectable
mais d'excellentes métriques. Trois protections ont été mises en place et
sont testées automatiquement.

**Validité de construit.** La cible mesure des déclarations, non des
occurrences. Le biais de couverture de la BDIFF et les 64 % de causes
manquantes affectent ce que l'étude mesure réellement.

**Validité externe.** Le périmètre est la France métropolitaine ; rien
n'autorise l'extrapolation à d'autres contextes. La validation croisée
spatiale fournit toutefois un argument en faveur d'une certaine robustesse
géographique.

**Validité statistique.** La corrélation spatiale induite par la maille
météorologique a été traitée par le rééchantillonnage au niveau des communes.
Elle n'a pas été traitée dans l'estimation des coefficients eux-mêmes, dont
les intervalles naïfs seraient trop étroits.

### 6.4 Erreurs rencontrées

Sur un événement à 0,02 %, une erreur d'implémentation ne se manifeste jamais
par une exception mais par une valeur plausible. Deux incidents méritent d'être
rapportés pour leur valeur méthodologique.

Le premier verdict porté sur le réseau récurrent annonçait une perte de 97 %,
alors que la valeur exacte est de 52 %. L'écart ne provenait pas du modèle
mais de la procédure de comparaison. La requête d'assemblage ne comportait pas
de clause de tri, l'ordre des 38 millions de lignes retournées dépendant du
plan d'exécution et variant d'une exécution à l'autre. Les fichiers de
prédictions ne portant que le score et la cible, leur comparaison revenait à
les aligner par position. Taille identique, nombre de positifs identique,
ordre différent : aucun contrôle ne pouvait le détecter. Depuis, tout fichier
de prédictions porte ses clés, une fonction d'alignement les vérifie, et un
test refuse tout fichier qui en serait dépourvu.

Le second concerne un modèle secondaire d'estimation de surface, dont le
coefficient de détermination atteignait 0,994 et l'aire sous la courbe ROC
1,0000. La cible figurait parmi les variables explicatives. Un résultat trop
favorable pour être crédible constitue en soi un signal.

Ces deux incidents illustrent que les seules protections efficaces sont les
invariants explicites et les assertions qui échouent bruyamment. Cinquante
tests sont exécutés à chaque intégration.

---

## 7. Travaux futurs

Par ordre de priorité décroissante :

1. **Lever l'asymétrie de la comparaison au réseau récurrent** en lui
   fournissant la variable `danger_effis`, afin de transformer le majorant de
   23,6 % en estimation.
2. **Recalibrer** le modèle effectivement déployé sur sa propre période, ce
   qui permettrait d'afficher une probabilité plutôt qu'un rang.
3. **Implémenter la contagion spatiale**, c'est-à-dire le décompte des
   sinistres survenus dans le voisinage géographique sur une fenêtre glissante.
   La conception et la requête existent, la table de voisinage est chargée,
   mais la variable n'a pas été produite et ne figure dans aucun modèle.
4. **Rechercher une source de cause de départ** moins lacunaire que les 64 %
   de valeurs manquantes actuels, l'origine humaine ou naturelle d'un départ
   étant vraisemblablement le déterminant manquant le plus important.
5. **Instrumenter la dérive** en production, sur la distribution des indices
   d'entrée et sur le rappel à budget constant.

---

## 8. Conclusion

Ce travail établit qu'un risque de départ de feu peut être estimé utilement à
l'échelle communale à partir de données publiques disponibles en temps réel,
avec un lift de 63,7 sur une période de test ouverte une seule fois, contre
42,1 pour la meilleure règle sans apprentissage.

Trois enseignements dépassent le cadre du problème traité.

Le premier est que la comparaison de modèles exige un protocole capable de
conclure à l'indécision. Trois des cinq modèles comparés se sont révélés
indiscernables, résultat qui aurait été présenté comme une hiérarchie sous un
protocole moins exigeant.

Le deuxième est qu'une architecture réputée adaptée à un type de données peut
être supplantée par des variables issues de la connaissance du domaine,
lorsque celles-ci encodent déjà la structure que l'architecture devrait
apprendre. Les indices du système canadien constituent un état récursif
calibré par cinquante ans de recherche.

Le troisième est que le critère de sélection d'un modèle destiné au service ne
se réduit pas à sa performance mesurée. La disponibilité des variables au
moment de la prédiction constitue une contrainte que nulle métrique
d'entraînement ne révèle.

Enfin, l'étude conclut à une augmentation très significative de l'aléa
météorologique sur cinquante-deux années, sans pouvoir conclure à une
augmentation du nombre de départs sur les vingt années observées. Cette
distinction, souvent escamotée, est maintenue ici sans arbitrage.

---

## Références et sources

- **Copernicus Emergency Management Service**, European Forest Fire
  Information System, indices de danger journaliers, grille 0,25°, 1973-2025.
- **IGN / BDIFF**, Base de Données sur les Incendies de Forêts en France.
- **CORINE Land Cover**, Copernicus Land Monitoring Service, millésimes 2006,
  2012 et 2018.
- **INSEE**, Code Officiel Géographique et fichier des mouvements de communes.
- **GIEC**, trajectoires représentatives de concentration RCP 2.6, 4.5 et 8.5.
- Van Wagner, C. E., *Development and structure of the Canadian Forest Fire
  Weather Index System*, Forestry Technical Report 35, 1987.
- Lundberg, S. M. et Lee, S.-I., *A Unified Approach to Interpreting Model
  Predictions*, NeurIPS, 2017.
- Lundberg, S. M. et al., *From local explanations to global understanding
  with explainable AI for trees*, Nature Machine Intelligence, 2020.
- Ribeiro, M. T. et al., *"Why Should I Trust You?" Explaining the Predictions
  of Any Classifier*, KDD, 2016.
- Mothilal, R. K. et al., *Explaining Machine Learning Classifiers through
  Diverse Counterfactual Explanations*, FAT*, 2020.
- Platt, J., *Probabilistic Outputs for Support Vector Machines*, 1999.
- Rashmi, K. V. et Gilad-Bachrach, R., *DART: Dropouts meet Multiple Additive
  Regression Trees*, AISTATS, 2015.

---

## Annexe A — Reproduire l'étude

```bash
docker compose up -d                 # PostgreSQL 16 + PostGIS
python -m tvfed.charger_ref          # référentiels et voisinage
python -m tvfed.charger_faits        # CEMS, BDIFF, CORINE
python -m tvfed.charger_grille       # la grille commune × jour
python -m tvfed.clustering --methode kmeans --k 30
python -m tvfed.matrices             # matrice d'apprentissage
python -m tvfed.modeles              # v1, v2, v3, DART, MLP
python -m tvfed.modele_c             # le modèle déployé
python -m tvfed.lstm                 # le réseau récurrent
python -m tvfed.comparer             # bootstrap apparié
python -m tvfed.series               # ADF, ACF/PACF, SARIMAX
python -m tvfed.explications         # SHAP
python -m tvfed.evaluation_test      # ouverture du jeu de test
pytest tests/ -q                     # 50 tests
```

## Annexe B — Artefacts produits

Chaque résultat chiffré du présent rapport provient d'un fichier versionné
dans `data/processed/`. Les principaux :

| Fichier | Contenu |
|---|---|
| `baselines.csv` | les trois références sans apprentissage |
| `pr_auc_val.csv` | performances sur validation |
| `comparaison_appariee.csv` | écarts appariés et intervalles de confiance |
| `transfert_spatial.csv` | validation croisée par région |
| `calibration_v3.csv` | comparaison des calibrations |
| `resultat_test.csv`, `test_par_annee.csv` | évaluation finale |
| `operationnel_test.csv` | rappel sous contrainte de budget |
| `series_adf.csv`, `series_acf_pacf.csv`, `series_sarimax.csv` | analyse de séries |
| `comparaison_clusters.csv` | ablation du regroupement |
| `test_decalages.csv` | ablation des décalages météorologiques |
| `modele_taille.csv` | modèles secondaires de surface et de grand feu |
| `gel_avant_test.json` | date et configuration du gel |

**Ressources en ligne**

- Application : `terre-vent-feu-eau-data.streamlit.app`
- Présentation : `hierogifle.github.io/terre-vent-feu-eau-data`
- Code source : `github.com/Hierogifle/terre-vent-feu-eau-data`
