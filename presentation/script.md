# Terre, Vent, Feu, Eau, Data — rapport final et aide-mémoire

Document de référence, à consulter avant l'oral ou pendant, en cas de trou.
Il n'est pas fait pour être lu de bout en bout : chaque section est autonome.

Tous les chiffres viennent des fichiers du dépôt, jamais de mémoire. Quand un
chiffre figure ici, il est lisible dans `data/processed/*.csv` ou
`app/donnees/meta.json`.

---

## Trouver vite

| J'ai un trou sur… | Aller à |
|---|---|
| pourquoi la métrique n'est pas l'exactitude | [§2.2](#22-pr-auc-roc-auc-et-lift) |
| le sous-échantillonnage, le facteur 487 | [§2.3](#23-le-sous-échantillonnage-et-le-prior-déplacé) |
| ce qui est une fuite et ce qui n'en est pas | [§2.4](#24-le-split-temporel-et-la-fuite) |
| le clustering, comment et sur quoi | [§2.5](#25-le-clustering-territorial) |
| le lissage bayésien, la formule | [§2.6](#26-le-lissage-bayésien) |
| Platt, isotonique, à quoi ça sert | [§2.7](#27-la-calibration-platt-et-isotonique) |
| le bootstrap apparié, la pseudo-réplication | [§2.8](#28-le-bootstrap-apparié) |
| ADF, ACF, PACF, les axes | [§2.9](#29-stationnarité-acf-et-pacf) |
| SARIMAX et Fourier | [§2.10](#210-sarimax) |
| pourquoi le LSTM perd | [§2.11](#211-le-lstm-et-pourquoi-il-perd) |
| SHAP, LIME, DiCE | [§2.12](#212-shap-lime-dice) |
| les scénarios RCP | [§2.14](#214-les-scénarios-rcp) |
| pourquoi le modèle déployé est le moins bon | [§3](#3-les-choix-et-pourquoi) |
| les chiffres exacts d'un résultat | [§5](#5-tous-les-résultats) |
| une question que je crains | [§6](#6-questions-dexpert) |

---

## 1. Le projet en vingt chiffres

| | |
|---|---|
| Communes de France métropolitaine | 34 734 |
| Lignes de la table centrale | 253 731 870 |
| Feux dans la grille (2006-2025) | 49 130 |
| Feux BDIFF sur le périmètre (1973-2025) | 52 809 |
| Taux de base | 0,019 % |
| Cellules météo CEMS sur la France | 1 131 |
| Communes par cellule météo | ~31 |
| Variables du modèle déployé | 41 |
| Variables du meilleur modèle | 52 |
| Groupes du clustering | 30 (+1 pour 2 communes atypiques) |
| Lift du modèle déployé, sur le test | ×63,7 |
| Lift du meilleur modèle, sur le test | ×93,8 |
| Écart LSTM contre modèle physique | −23,6 % |
| Régions gagnées en validation spatiale | 9 sur 9 |
| Hausse du FWI estival, 1973-2025 | +62 % (p = 1,5 × 10⁻⁴) |
| Hausse du nombre de feux, 2006-2025 | non significative |
| Communes n'ayant jamais brûlé | 25 297 (73 %) |
| Rappel à 1 % de budget de surveillance | 42 % |
| Tests automatisés | 50 |
| Date du gel avant test | 28 juillet 2026 |

---

## 2. Les notions

### 2.1 L'événement rare, et pourquoi tout en découle

Un départ de feu concerne **0,019 %** des couples commune × jour. Une ligne
sur 5 348.

Trois conséquences, et elles commandent tout le reste du projet :

1. **L'exactitude est inutilisable.** Répondre systématiquement « non » donne
   99,98 % de justesse. Aucune métrique d'exactitude n'est présentée.
2. **Une fuite de données ne provoque pas d'erreur.** Elle produit
   d'excellentes métriques et un modèle sans valeur. Rien dans les scores ne
   la signale : c'est le risque principal.
3. **Le déséquilibre décide de la métrique, de l'échantillonnage, de la
   calibration et de la façon de comparer deux modèles.**

### 2.2 PR-AUC, ROC-AUC et lift

| | ROC-AUC | PR-AUC |
|---|---|---|
| Mesure | vrais positifs contre faux positifs | précision contre rappel |
| Valeur au hasard | 0,50 toujours | **le taux de base** |
| À 0,019 % de positifs | flatteuse, 0,95 sans effort | lisible |

La ROC-AUC compare les positifs aux négatifs. Quand les négatifs sont
5 348 fois plus nombreux, ajouter des faux positifs ne bouge presque pas le
taux de faux positifs — donc la courbe reste belle même pour un modèle
médiocre.

La PR-AUC vaut exactement le taux de base quand on répond au hasard. D'où :

```
lift = PR-AUC ÷ taux de base
```

« Combien de fois mieux que tirer au sort. » C'est le seul nombre du projet
qui se dise à voix haute sans être trompeur.

### 2.3 Le sous-échantillonnage et le prior déplacé

**Ce qu'on fait.** On **retire des négatifs du train**. On n'ajoute jamais
rien, on ne duplique jamais rien. `sql/31_split.sql` :

```sql
WHERE split <> 'train'   -- val et test INTÉGRAUX : jamais échantillonnés
   OR y                  -- 100 % des positifs du train
   OR u < 0.00187;       -- 0,187 % des négatifs du train
```

**L'arithmétique.** Taux de base 0,0187 % → environ 5 348 négatifs par
positif. Pour un ratio 1:10, garder 10/5 348 = **0,187 %** des négatifs.

**La conséquence.** Le modèle apprend sur un jeu à 9,1 % de positifs alors
que la réalité est à 0,019 %. Le rapport, **×487**, c'est simplement
9,1 / 0,019. Le classement reste bon, mais le niveau absolu des probabilités
est faux.

**Les trois garde-fous.**

1. Validation et test ne sont **jamais** échantillonnés : les scores restent
   comparables au monde réel.
2. Les statistiques dérivées de la cible se calculent sur le train
   **complet**. Sur l'échantillon, un lissage bayésien vaudrait 9,1 % au lieu
   de 0,019 % : le prior serait empoisonné et rien ne le signalerait.
3. `u` est un tirage **déterministe** : le même échantillon est reproduit à
   chaque exécution.

### 2.4 Le split temporel et la fuite

Train **2006-2019**, validation **2020-2022**, test **2023-2025**.

**Pourquoi jamais aléatoire.** Un tirage au hasard mettrait le 14 juillet 2019
dans le train et le 15 dans le test. Le modèle « prédirait » un feu qu'il a
déjà vu, à 20 km et un jour d'écart. La métrique serait excellente et le
modèle sans valeur.

**La règle qui tranche les cas douteux :**

> Une variable **datée** peut regarder tout le passé, y compris celui de sa
> propre période d'évaluation.
> Une statistique **non datée** ne peut regarder que le train.

**L'exemple qui la rend claire.** « Feux des 30 jours précédents » au 3 août
2023 lit juillet 2023 : ce n'est **pas** une fuite, parce que le 3 août à 8 h
du matin on connaît juillet. En revanche « taux moyen de la commune sur toute
la période » lit le futur : c'en est une.

**Le test.** Ouvert **une seule fois**, après gel complet du modèle, des
variables et de la calibration, le 28 juillet 2026. Aucune décision n'en
découle. La justification : le risque d'un jeu de test est **cumulatif**,
chaque coup d'œil en apprend un peu et les décisions suivantes en sont
insensiblement informées.

### 2.5 Le clustering territorial

**Le problème.** Le modèle v1 tirait 54,6 % de son importance de l'historique
de la commune. Il disait surtout « ce qui a brûlé rebrûlera ». Conséquence :
une commune qui n'a jamais brûlé gardait un score bas, même entourée de
communes qui brûlent chaque été. C'est le problème classique d'**estimation
sur petits domaines** (*small area estimation*) : trop peu d'événements pour
estimer un taux commune par commune.

**Sur quoi porte le clustering.** Des caractéristiques **physiques** :
végétation, relief, densité humaine, climatologie du FWI, position.
**Jamais sur la cible.** Un cluster construit sur la sinistralité serait
circulaire — on prédirait le feu avec des groupes définis par le feu.

**La position est sous-pondérée à 25 %.** Sans latitude et longitude, les
groupes seraient éclatés d'un bout à l'autre du pays. À poids plein, ils
dégénéreraient en pavés géographiques et le clustering ne serait qu'un
découpage administratif déguisé.

**La configuration.** k-means, `k = 30`, `n_init = 10`, `random_state = 42`.
Deux communes n'ont pas de profil exploitable : elles vont dans un cluster
`−1` et reçoivent le prior national. D'où **31 valeurs distinctes** dans les
données pour 30 vrais groupes.

**Les trois garde-fous anti-fuite.**

1. Le profil ne lit que le passé : CORINE 2006, climatologie FWI 2006-2019.
2. Les taux sont agrégés sur le train **complet**, jamais sur l'échantillon.
3. Pour une ligne de train de l'année Y, les taux **excluent l'année Y**.
   Sans ça une ligne de 2012 contribuerait à sa propre variable : c'est la
   fuite classique du *target encoding*. Les lignes de validation et de test
   utilisent les 14 années de train, dont elles ne font pas partie.

### 2.6 Le lissage bayésien

**Le problème.** Une commune sans feu en 5 113 jours a un taux observé de
**0**. Ce n'est pas une estimation, c'est une absence d'information.

**La formule**, hiérarchique national → cluster → commune :

```
taux_commune = (nb_feux + K₁ × taux_du_cluster) / (nb_jours + K₁)
```

avec `K₁ = 2 000` et, un cran au-dessus, `K₀ = 20 000` pour ramener le taux
du cluster vers le taux national.

**Comment lire K.** C'est un nombre de jours fictifs. `K₁ = 2 000` signifie
« je ne fais pleinement confiance au comptage d'une commune qu'au-delà
d'environ 2 000 jours d'observation ». Chaque commune ayant exactement 5 113
jours de train, le poids du groupe vaut 2 000 / (5 113 + 2 000) = **28,1 %**.

**Mesuré sur les données réelles**, pour les 25 297 communes (73 % du pays)
qui n'ont jamais brûlé :

| | |
|---|---|
| `taux_commune_lisse` | de 6,07 × 10⁻⁷ à 9,18 × 10⁻⁴ |
| Combien valent zéro | **0 sur 25 297** |
| Rapport entre la plus basse et la plus haute | **1 512×** |

Le lissage ne leur colle donc pas une valeur par défaut identique : une
commune de garrigue corse sans historique hérite du taux de son groupe
méditerranéen, une commune de Picardie hérite du sien.

**Le gain mesuré** : +0,83 % de PR-AUC. Réel mais modeste, et le dire fait
partie du résultat.

### 2.7 La calibration : Platt et isotonique

**Le problème.** Le modèle sort un **score**, pas une probabilité. À cause du
sous-échantillonnage, ce score est **145 fois trop grand** : il annonce 93 %
là où la réalité est 3 %.

Calibrer, c'est apprendre une fonction qui transforme le score en probabilité.

| | Comment | Effet |
|---|---|---|
| **Platt** | ajuste une sigmoïde, 2 paramètres | lisse, monotone, **garde les 9 millions de valeurs distinctes** |
| **Isotonique** | ajuste une fonction en escalier croissante | très souple, mais **écrase** à 136 valeurs distinctes |

Les deux corrigent aussi bien le biais : ×144,7 → ×1,13 pour Platt. Mais
l'isotonique détruit du pouvoir de discrimination pour rien — si 300 communes
partagent exactement le même score, on ne peut plus les classer entre elles.

**Ni l'un ni l'autre n'est utilisé dans l'application.** Elle affiche un
**rang**, parce que le calibrateur disponible a été ajusté sur un autre modèle
et une autre période : il serait faux d'un facteur voisin de 2. On a préféré
ne pas afficher de probabilité plutôt que d'en afficher une fausse.

### 2.8 Le bootstrap apparié

**Le problème.** v3 fait 0,0177 de PR-AUC, DART 0,0174. Est-ce un vrai écart
ou le hasard de l'échantillon d'évaluation ?

**Apparié** : on tire un échantillon et on évalue **les deux modèles sur
exactement les mêmes lignes**. On répète 200 fois. On obtient 200 écarts,
donc une distribution **de l'écart** — pas deux distributions séparées qu'il
faudrait ensuite comparer.

**On rééchantillonne les communes, pas les lignes.** C'est le point de méthode
le plus important du projet.

Les 1 096 jours d'une même commune ne sont pas indépendants, et 31 communes
partagent la même maille météo — elles voient le même FWI. Le nombre
d'informations réellement indépendantes est bien plus petit que 38 millions.
Tirer ligne à ligne reviendrait à traiter 38 millions d'observations comme
38 millions d'expériences indépendantes : les intervalles seraient **beaucoup
trop étroits** et feraient conclure à des différences inexistantes. Le terme
consacré est la **pseudo-réplication**.

**L'astuce de calcul.** Recalculer la PR-AUC 200 fois sur 38 millions de
lignes coûterait des heures, chaque appel retriant le tableau. On trie **une
fois** ; une réplique n'est alors qu'un jeu de **poids entiers** le long de
cet ordre figé, et la précision moyenne pondérée se calcule en une passe par
sommes cumulées. Vérifié identique à scikit-learn à 1 × 10⁻¹² près.

### 2.9 Stationnarité, ACF et PACF

**ADF — test de Dickey-Fuller augmenté.** Il teste la présence d'une racine
unitaire.

> H₀ : la série **a** une racine unitaire, donc elle **n'est pas**
> stationnaire.
> Rejeter H₀ (p < 0,05) signifie **stationnaire**.

C'est l'inverse de l'intuition, et c'est la confusion la plus fréquente sur
ce test.

**ACF et PACF — les axes.**

- **Ordonnée** : un coefficient de **corrélation**, entre −1 et +1, sans unité.
- **Abscisse** : le **retard, en jours**.
- **Bande bleue** : le seuil de significativité, ±1,96/√n. Une barre qui reste
  dedans est indiscernable de zéro.

**La différence entre les deux.** L'ACF au retard 2 mesure la corrélation
entre aujourd'hui et avant-hier, en incluant tout ce qui transite par hier.
La PACF retire cet effet indirect : elle donne **l'apport propre** du retard 2.

C'est donc la PACF qui répond à « combien de jours de passé faut-il garder ».

**Les valeurs mesurées**, sur les résidus du cycle annuel, 7 305 jours :

| Retard | ACF | PACF | Significatif |
|---|---|---|---|
| 1 jour | 0,697 | **0,697** | oui |
| 2 jours | 0,584 | **0,191** | oui |
| 3 jours | 0,501 | **0,077** | oui |
| 4 jours | 0,435 | 0,041 | oui |
| 7 jours | 0,315 | 0,014 | **non** |
| 8 jours | 0,295 | 0,034 | oui |

Seuil : ±0,023.

**La nuance à donner soi-même.** Les retards 4 à 8 restent *statistiquement*
significatifs. Avec 7 305 points le seuil descend à 0,023 et presque tout le
devient. Mais le retard 8, à 0,034, rend compte de **0,11 %** de la variance.
**Significatif ne veut pas dire utile.** Le retard 7 non significatif entre
deux qui le sont est d'ailleurs le signe qu'on est dans le bruit.

### 2.10 SARIMAX

**La série modélisée** : nombre de communes-jours en feu, par jour. Moyenne
6,73, maximum 89 le 18 juillet 2022.

**Pourquoi pas SARIMA avec s = 365.** Une saisonnalité annuelle sur données
journalières demanderait d'estimer des coefficients à 365 pas de distance sur
5 113 points d'ajustement : instable et très lent. La pratique établie est de
porter la saisonnalité par des **termes de Fourier en variable exogène** —
quelques harmoniques suffisent pour un cycle annuel lisse. Le projet en
utilise 4 paires sinus/cosinus. C'est le « X » de SARIMA**X**.

**Les résultats.** MAE en communes-jours par jour, sur une moyenne de 6,73 :

| Modèle | MAE | r |
|---|---|---|
| SARIMAX(2,0,1) + Fourier + FWI | **4,03** | 0,850 |
| SARIMAX(2,0,1) + Fourier seul | 6,42 | 0,603 |
| ARIMA(2,0,1) sans exogène | 8,34 | **−0,118** |
| Référence : moyenne du jour de l'année | 5,13 | 0,598 |

**La ligne qui compte est la troisième.** Un ARIMA sans variable exogène donne
une corrélation **négative**. À 1 096 pas d'horizon, un modèle autorégressif
dont la mémoire utile vaut trois jours a oublié son point de départ et
converge vers la moyenne ; la ligne plate qu'il produit se trouve légèrement
anti-corrélée à l'observé.

Ajouter le FWI fait tomber l'erreur de 37 %. **La prévisibilité du feu est
dans la météo, pas dans son propre passé.**

Gain sur la référence saisonnière naïve : +21,5 % de MAE. Ce n'est pas un
triomphe, et c'est le point.

### 2.11 Le LSTM, et pourquoi il perd

**Ce qu'il a reçu.** 30 jours × 8 indices météo = 240 valeurs par exemple.
25 essais Optuna, arrêt précoce à l'époque 21. Hyperparamètres retenus :
2 couches, cache 32, tête 256, dropout 0,179, lr 7,8 × 10⁻⁴, lot 2 048.
L'objection « il n'a pas été réglé » ne tient pas.

**Contre qui le comparer.** Pas contre v3 : v3 voit l'historique des feux
(29 % de ses importances), le LSTM n'en voit rien. Les opposer mesurerait le
prix de l'information retirée, pas la valeur de la séquence. **La seule
référence à jeu d'information égal est le modèle C.**

**Le résultat** : **−23,6 %**, intervalle [−33,5 ; −17,3]. Loin de zéro.

**L'explication, physique.** Un LSTM sert quand l'ordre de la séquence porte
une information qu'aucun résumé ne capture. Ici ce résumé existe déjà : les
indices **DC**, **DMC** et **BUI** du système canadien **sont** des états
récursifs. Le *Drought Code* est une moyenne exponentielle de la météo passée
avec une constante de temps de **52 jours**, le *Duff Moisture Code* de
**15 jours**. C'est la forme d'une cellule récurrente, à ceci près que ses
coefficients ont été calibrés par cinquante ans de science du feu plutôt
qu'estimés sur 9 176 exemples positifs.

**Le CEMS livre déjà l'état caché que le LSTM devrait réapprendre.**

**Trois méthodes indépendantes concordent** : la PACF (mémoire utile de deux
à trois jours), l'ARIMA sans exogène (corrélation négative), et les
importances du modèle C (les trois premières variables disent où est le
combustible, pas quand).

**La réserve à donner spontanément.** Le LSTM ne reçoit pas `danger_effis`,
qui pèse 13,7 % dans le modèle C. Les 23,6 % sont donc un **majorant**. C'est
la première chose à reprendre.

### 2.12 SHAP, LIME, DiCE

**Trois outils, trois questions différentes.**

| | Question | Nature |
|---|---|---|
| **SHAP** | pourquoi ce score ? | **exact** sur un modèle d'arbres |
| **LIME** | pourquoi ce score ? | approché, substitut linéaire local |
| **DiCE** | qu'aurait-il fallu changer ? | contrefactuel |

**SHAP.** Décompose un score en contributions par variable, selon les valeurs
de Shapley de la théorie des jeux. Sur un modèle d'arbres, **TreeSHAP est
exact** : il parcourt la structure des arbres en temps polynomial, il
n'échantillonne pas. Vérifié dans le projet — la somme des contributions plus
la valeur de base redonne le logit du score à **9,5 × 10⁻⁷** près.

**Trois mesures d'importance qui ne donnent pas le même classement :**

1. **Le gain d'entraînement** : combien la variable a réduit la perte. Sortie
   par défaut de XGBoost. Biaisée vers les variables qui découpent proprement,
   **aveugle à la redondance**.
2. **SHAP sur échantillon aléatoire** : l'effet sur le territoire tel qu'il
   est, soit 99,97 % de communes-jours sans feu. Lecture pour « en général ».
3. **SHAP au sommet du classement** : l'effet là où le modèle s'engage.
   Lecture pour « quand ça brûle ».

**Deux désaccords à savoir expliquer.**

- `danger_effis` est **2ᵉ par gain** et **30ᵉ par SHAP**. C'est une
  discrétisation du FWI en six classes : XGBoost trouve ces seuils commodes
  pour découper, d'où un gain élevé, mais l'information est déjà dans le FWI
  continu et SHAP en attribue le crédit à ce dernier.
- `part_maquis` est **1ᵉʳ par gain**, **10ᵉ** sur l'échantillon aléatoire et
  **2ᵉ** au sommet. Sur une commune-jour moyenne il n'y a pas de maquis, donc
  la variable ne déplace rien ; là où le modèle voit du risque, elle devient
  déterminante.

**Conséquence pratique** : citer « la variable la plus importante » n'a pas de
sens sans préciser quelle mesure et sur quelle population.

**LIME.** Perturbe le point, interroge le modèle des milliers de fois, ajuste
une régression linéaire pondérée sur ce voisinage. Ce qu'il rend, ce sont les
coefficients d'un **modèle de substitution**, pas la contribution exacte.
Trois conséquences : les variables apparaissent sous forme de **règles**
(`fwi > 12,4`), le résultat est **stochastique**, et il dépend du **fond**
fourni. Sur un modèle d'arbres, il approxime ce que TreeSHAP calcule
exactement : il ne peut pas faire mieux. Il redeviendrait le bon outil sur un
modèle qu'on ne peut pas ouvrir, une API ou un réseau profond.

**DiCE.** Cherche le point le plus proche que le modèle classerait
différemment.

*Le détail d'implémentation qui change tout* : DiCE cherche par défaut à faire
passer la probabilité sous **0,5**. Or le score n'est pas calibré — 0,5
correspond à un risque astronomique et l'outil ne renvoyait **jamais rien**.
On a recentré la frontière sur le **décile**, par une transformation affine
par morceaux strictement croissante qui laisse le classement intact. La
question devient « que faudrait-il pour sortir des 10 % les plus à risque »,
celle qui a un sens opérationnel.

*Le résultat sur Bormes-les-Mimosas*, 12 août 2024, en n'autorisant que la
végétation : **aucun contrefactuel**. Rien, sur ces leviers, ne l'en fait
sortir. Son exposition tient à sa position, à son relief, à sa superficie.
Le risque est **structurel**, et l'absence de solution est ici la réponse.

*La mise en garde* : un contrefactuel n'est pas une recommandation. Rien ne
garantit qu'il soit réalisable — on ne convertit pas 40 % de maquis en terres
agricoles — ni que le lien soit **causal** : le modèle a appris des
corrélations sur 2006-2019, pas des mécanismes.

### 2.13 Le biais de collision

Un **collider** est une variable causée par deux autres. Conditionner dessus
crée une association artificielle entre ces deux causes.

Dans ce projet : sélectionner les lignes sur le **score** du modèle est un
conditionnement sur un collider — le score est causé à la fois par la météo et
par le territoire. En analysant l'interaction météo × territoire sur le seul
sommet du classement, **le signe de l'interaction s'inversait**.

C'est pour cette raison que le panneau « sommet » ne se lit que pour la
question « quand ça brûle », jamais pour « quelle est la relation entre A et
B en général ».

### 2.14 Les scénarios RCP

**RCP** = *Representative Concentration Pathway*. Le chiffre est le **forçage
radiatif en 2100, en W/m²**.

| | Hypothèse |
|---|---|
| **RCP 2.6** | neutralité carbone vers 2070 |
| **RCP 4.5** | émissions plafonnées vers 2040 |
| **RCP 8.5** | aucune politique climatique |

**Ce qu'on projette** : le FWI, seule quantité qui montre un signal et seule
que les modèles climatiques savent fournir.

**Ce qu'on ne projette pas** : le nombre de feux. La végétation, la prévention
et les pratiques agricoles sont supposées **constantes**, ce qui est une
hypothèse, et elle est fausse.

**Ce que « le 2 août 2050 » signifie** : pas une prévision météo, mais un
2 août **ordinaire** sous le climat de 2050. La forme de la saison vient des
observations 2006-2019, seul son niveau est décalé.

**Avant 2045, les trois scénarios sont indiscernables.** Mesuré : en 2030, le
RCP 2.6 dépasse le RCP 8.5 sur **59 %** des communes. Ce n'est pas une
anomalie du code, c'est l'inertie du système climatique — les trajectoires
d'émissions ne divergent réellement qu'après le milieu du siècle.
L'application grise cette zone sur les graphiques.

---

## 3. Les choix, et pourquoi

| Choix | L'alternative | Pourquoi ce choix |
|---|---|---|
| **Grille dense** commune × jour, 253 M lignes | garder la seule liste des 52 809 feux | une série creuse rend les fenêtres glissantes silencieusement fausses : « feux des 30 jours précédents » se calcule en remontant 30 lignes, et sans les jours sans feu on remonte plusieurs années |
| **PostgreSQL + PostGIS**, partitionné par année | tout en Parquet | le partitionnement par année rend le split temporel exact et les requêtes d'évaluation peu coûteuses ; PostGIS pour le rattachement commune → maille et la distance à la côte |
| **Split temporel** | split aléatoire | un split aléatoire met le 14 juillet dans le train et le 15 dans le test |
| **PR-AUC** | ROC-AUC, exactitude, F1 | seule métrique dont la valeur au hasard est le taux de base, donc seule interprétable à 0,019 % |
| **Sous-échantillonnage 1:10 du train seul** | pondération des classes, SMOTE | 177 M lignes sont ingérables ; SMOTE interpolerait des communes-jours qui n'existent pas, sur des variables physiques dont les combinaisons ont un sens |
| **Fichier officiel INSEE** pour les fusions | rapprochement par le nom | le nom produit des faux positifs — « Chirac » en Lozère renvoyait vers la Charente. 935 feux rattachés avec certitude, **30 écartés et comptés**, jamais devinés |
| **Modèle C déployé** malgré ×63,7 contre ×93,8 | déployer v3 | v3 dépend de l'historique des feux, que la BDIFF ne publie pas pour l'année en cours ; en territoire inconnu la variable vaut zéro et se lit « ça ne brûlera pas » ; pour 2050 elle est impossible par construction |
| **Rang affiché**, pas une probabilité | afficher la probabilité calibrée | le calibrateur disponible a été ajusté sur un autre modèle et une autre période, il serait faux d'un facteur ~2. Mieux vaut pas de probabilité qu'une fausse |
| **Bootstrap sur les communes** | bootstrap sur les lignes | pseudo-réplication : les jours d'une commune ne sont pas indépendants, 31 communes partagent une maille météo |
| **Pas d'ensemble** malgré +2,0 % | déployer v3 + MLP | casse la chaîne SHAP et double la chaîne de service pour un gain qui ne se voit pas opérationnellement |
| **Fourier en exogène** | SARIMA d'ordre 365 | 365 pas de distance sur 5 113 points : instable et très lent |
| **Palette EFFIS par défaut**, alternative accessible | EFFIS seule | la palette officielle n'est **pas monotone en luminance** : le jaune « faible » (0,810) est plus clair que le vert « très faible » (0,688), donc les deux premières classes se lisent à l'envers en protanopie et deutéranopie |

---

## 4. L'architecture technique

**Les sources.**

| Source | Contenu | Volume |
|---|---|---|
| CEMS · Copernicus | 8 indices de danger, par jour et par maille de 0,25° | 21,9 M lignes, 1973-2025 |
| BDIFF · IGN | feux déclarés, commune par commune | 52 809 feux |
| CORINE Land Cover | occupation du sol, 44 postes | 1,08 M lignes |
| INSEE | référentiel communal et mouvements de communes | 34 734 communes |

**La chaîne.** Docker → PostgreSQL 16 + PostGIS 3.4 (port 5433) → 20
partitions annuelles → vues de split → matrice d'apprentissage en Parquet →
modèles → artefacts CSV/JSON → `app/donnees/` → Streamlit.

**Les modèles.**

| Modèle | Variables | Ce qu'il teste |
|---|---|---|
| Baselines (3) | 0 | ce que valent des règles sans apprentissage |
| XGBoost v1 | 43 | un gradient boosting standard suffit-il |
| XGBoost v2 | 43 | le problème du v1 est-il un problème de réglage |
| XGBoost v3 | 52 | peut-on donner un risque aux communes sans historique |
| DART | 52 | l'abandon d'arbres améliore-t-il la généralisation |
| MLP | 52 | un réseau dense fait-il mieux que des arbres |
| **XGBoost C** | **41** | **que reste-t-il sans aucun historique de feu** |
| LSTM | 30 j × 8 | la séquence météo porte-t-elle un signal propre |

**Les 41 variables du modèle C** : 11 météo, 12 occupation du sol,
7 territoire, 11 calendrier. Aucune dérivée de l'historique des feux.

**Hyperparamètres XGBoost** (Optuna, TPE, 60 essais) : `n_estimators` 900,
`learning_rate` 0,0117, `max_depth` 10, `min_child_weight` 25, `subsample`
0,695, `colsample_bytree` 0,603, `gamma` 1,03, `reg_lambda` 0,0226,
`reg_alpha` 0,0627.

**Qualité.** 50 tests automatisés en intégration continue à chaque commit,
dont deux dédiés au split et à l'absence de fuite. DVC pour les données
lourdes. Gel documenté dans `data/processed/gel_avant_test.json`.

---

## 5. Tous les résultats

**La grille.**

| Partition | Années | Lignes | Feux | Taux |
|---|---|---|---|---|
| train | 2006-2019 | 177 594 942 | 33 632 | 0,0189 % |
| validation | 2020-2022 | 38 068 464 | 9 176 | 0,0241 % |
| test | 2023-2025 | 38 068 464 | 6 322 | 0,0166 % |
| **total** | 2006-2025 | **253 731 870** | **49 130** | 0,0194 % |

**Les baselines**, sans aucun apprentissage.

| Prédicteur | PR-AUC | lift |
|---|---|---|
| hasard | 0,000241 | ×1,0 |
| historique de la commune | 0,004668 | ×19,4 |
| danger EFFIS seul | 0,001220 | ×5,1 |
| historique × EFFIS | 0,010149 | **×42,1** |

**Les modèles sur la validation** (38 068 464 lignes, 9 176 feux).

| Modèle | PR-AUC | lift | vs v3 | IC 95 % | Significatif |
|---|---|---|---|---|---|
| XGBoost v3 | 0,0177 | ×73,4 | référence | | |
| DART | 0,0174 | ×72,0 | −1,8 % | [−5,0 ; +0,8] | **non** |
| MLP | 0,0173 | ×71,9 | −1,9 % | [−7,9 ; +5,2] | **non** |
| XGBoost C | 0,0112 | ×46,4 | −36,8 % | [−43,8 ; −24,0] | oui |
| LSTM | 0,0085 | ×35,4 | −51,7 % | [−57,2 ; −43,4] | oui |

Ensemble v3 + MLP : 0,0180, ×74,9. Non retenu.
**LSTM contre modèle C**, à information égale : **−23,6 %** [−33,5 ; −17,3].

**Le test, ouvert une fois.**

| Modèle | PR-AUC | lift |
|---|---|---|
| XGBoost v3 | 0,0156 | ×93,8 |
| **XGBoost C** (déployé) | **0,0106** | **×63,7** |

Par année : 2023 ×75,6 (2 591 feux) · 2024 ×151,4 (1 297 feux) · 2025 ×87,0
(2 434 feux). **Le lift suit la rareté** : une année calme concentre les feux
dans les endroits les plus prévisibles.

**Validation croisée spatiale** : le modèle physique gagne **9 régions sur
9**, +8,2 % en moyenne pondérée, jusqu'à **+137 %** dans le Grand Est.

**Rendement opérationnel**, sur le test.

| Budget | Communes-jours suivis | Feux couverts | Part des feux |
|---|---|---|---|
| 0,1 % | 38 068 | 924 | 14,6 % |
| 1 % | 380 684 | 2 650 | **41,9 %** |
| 10 % | 3 806 846 | 5 372 | 85,0 % |

Les 37 % de PR-AUC qui séparent v3 du modèle C ne valent que **3 points de
rappel à 1 % de budget**, et **rien du tout à 10 %**.

**Le climat.**

| Série | Période | Variation | p | Verdict |
|---|---|---|---|---|
| FWI moyen annuel | 1973-2025 | +58 % | 4,2 × 10⁻⁵ | significatif |
| FWI juin-septembre | 1973-2025 | +62 % | 1,5 × 10⁻⁴ | significatif |
| jours de danger élevé | 1973-2025 | +197 % | 5,2 × 10⁻⁵ | significatif |
| **communes-jours en feu** | 2006-2025 | +3 % | 0,89 | **non significatif** |

**Les modèles secondaires.** Surface brûlée : R² 0,144, MAE 3,94 ha — moins
bon que d'annoncer toujours la médiane. Grand feu (seuil 5 ha) : PR-AUC 0,232,
lift 2,88, ROC-AUC 0,766.

**La distance à la côte**, mesurée sur les données observées :

| Distance | Feux par commune | Feux par 1 000 km² |
|---|---|---|
| 0-10 km | 10,26 | 519 |
| 25-50 km | 3,77 | 200 |
| 100-200 km | 0,87 | 55 |
| 200+ km | 0,36 | 25 |

Une commune côtière brûle **28 fois plus** qu'une commune de l'intérieur.
C'est un **proxy géographique** — climat sec et venteux, maquis et pins,
fréquentation estivale — pas un mécanisme.

---

## 6. Questions d'expert

### 6.1 Un data engineer

**Pourquoi PostgreSQL et pas simplement du Parquet ou DuckDB ?**
Pour le partitionnement par année, qui rend le split temporel exact au niveau
du stockage, et pour PostGIS — le rattachement commune → maille météo et la
distance à la côte sont des opérations géométriques. Le Parquet intervient
après, pour la matrice d'apprentissage, là où on relit toujours les mêmes
colonnes.

**253 millions de lignes, quelle taille et quel temps de construction ?**
20 partitions annuelles. `data/processed` pèse 2,6 Go, les artefacts servis à
l'application 112 Mo. La grille se construit en SQL, pas en Python : c'est ce
qui rend l'opération tenable.

**Votre échantillonnage est-il reproductible ?**
Oui, `u` est un tirage déterministe stocké, pas un `random()` évalué à chaque
requête. Le même échantillon ressort à chaque exécution.

**Que se passe-t-il si Copernicus republie des données corrigées ?**
Le chargement est idempotent par période. En revanche la climatologie
2006-2019 et les clusters devraient être recalculés, et le gel avant test
serait invalidé — il faudrait repartir d'un nouveau gel.

**Vous avez eu un incident de qualité de données ?**
Oui, le plus coûteux du projet. `sql/50_matrice.sql` n'a pas d'`ORDER BY` :
l'ordre des 38 millions de lignes que renvoie PostgreSQL dépend du plan
d'exécution et **change d'une exécution à l'autre**. Mes fichiers de
prédictions ne portaient que `(score, cible)` — même taille, même nombre de
feux, ordre différent, aucune exception levée. Le premier verdict du LSTM
annonçait −97 % au lieu de −52 %. Depuis, **tout fichier porte ses clés**
`(code_insee, date)`, une fonction d'alignement vérifie, et un test **refuse**
un fichier sans clés.

**Comment est-ce testé ?**
50 tests en intégration continue à chaque commit, dont un qui reproduit
volontairement le désalignement : on permute les lignes, la PR-AUC tombe de
0,0085 à 0,0002, exactement la ligne du hasard, avec les mêmes valeurs dans le
fichier.

### 6.2 Un data analyst

**Pourquoi le taux de feu diffère-t-il entre train (0,0189 %), validation
(0,0241 %) et test (0,0166 %) ?**
Ce sont des périodes différentes, avec des étés différents. 2022 a été
exceptionnel — le maximum de la série est le 18 juillet 2022 avec 89
communes-jours en feu. C'est aussi pourquoi le lift varie du simple au double
selon l'année du test.

**La BDIFF est-elle exhaustive ?**
Non, et le projet le documente. La couverture n'est pas la même selon les
départements et les périodes, et **64 % des causes déclarées sont
manquantes**. C'est une limite de la **cible**, pas seulement des variables :
elle borne ce qu'on peut prétendre mesurer.

**Comment avez-vous vérifié la qualité des dates BDIFF ?**
Par un « test du dimanche » : si la date enregistrée était une date de saisie
administrative plutôt qu'une date d'éclosion, on verrait un déficit le
week-end. Le contrôle figure dans `figures/data-bdiff/01`.

**Pourquoi deux saisons de feu ?**
Le pic d'août est méditerranéen. Le pic de mars concerne le Sud-Ouest et le
Massif central, sur des landes et herbacées sèches avant la reprise de
végétation, souvent des écobuages qui échappent. En **part** de leurs feux
hors saison estivale : Cantal 52 %, Dordogne 44 %, Lozère 41 %. En **volume**
absolu, la Haute-Corse, la Gironde et l'Hérault dominent, y compris hors été —
confondre les deux lectures est une erreur facile.

**Une commune-jour, c'est un incendie ?**
Non. Un feu traversant cinq communes compte cinq fois. La cible est « cette
commune a-t-elle au moins un départ déclaré ce jour-là », pas « combien
d'incendies ».

**Que valent les communes qui n'ont jamais brûlé ?**
73 % du pays, 25 297 communes. Leur taux lissé va de 6,1 × 10⁻⁷ à
9,2 × 10⁻⁴, **aucune n'est à zéro**, et l'étendue est d'un facteur 1 512.

### 6.3 Un data scientist

**Pourquoi 1:10 et pas 1:1 ou 1:100 ?**
Compromis entre volume d'entraînement tenable et déformation du prior. À 1:1
le prior serait déplacé d'un facteur 5 000 et la calibration deviendrait
l'essentiel du travail ; à 1:100 le train resterait à 3,6 millions de lignes
positives-négatives, coûteux pour un gain marginal. Le choix n'a pas été
optimisé — c'est une limite assumée.

**Pourquoi pas SMOTE ou de la pondération de classe ?**
SMOTE interpole entre exemples pour créer des positifs synthétiques. Ici les
variables sont physiques et leurs combinaisons ont un sens : interpoler entre
une commune corse et une commune picarde produit un territoire qui n'existe
pas. La pondération, elle, ne réduit pas le coût de calcul, qui était le
problème.

**Comment évitez-vous la fuite du target encoding ?**
Pour une ligne de train de l'année Y, les taux **excluent l'année Y**. Sans
ça une ligne de 2012 contribuerait à sa propre variable. Les lignes de
validation et de test utilisent les 14 années de train, dont elles ne font
pas partie.

**Vos variables météo sont fortement redondantes — BUI est une fonction de
DMC et DC. Pourquoi les garder toutes ?**
Parce que les arbres gèrent la redondance sans dommage pour la prédiction, et
que le projet mesure cette redondance plutôt que de l'ignorer : c'est
exactement ce qui explique que `danger_effis` soit 2ᵉ par gain et 30ᵉ par
SHAP. Une paire redondante n'invalide pas le modèle, elle invalide la lecture
naïve de l'importance par gain.

**Pourquoi 200 répliques de bootstrap ?**
Au-delà, la largeur des intervalles ne bouge plus de façon utile et le coût
est linéaire. Les conclusions n'en dépendent pas : les intervalles qui
traversent zéro le traversent largement.

**Vos intervalles traversent zéro pour DART et le MLP. Est-ce un manque de
puissance ?**
C'est possible et je ne peux pas l'exclure. Ce que je peux affirmer, c'est
que **avec ce protocole et ces données je ne peux pas les départager**. Dire
« XGBoost bat le MLP » serait une conclusion tirée du bruit. La lecture la
plus probable est que le signal disponible est capté de façon comparable par
les trois familles, et que la limite est dans les données.

**Pourquoi pas de validation croisée ?**
Parce que le découpage doit rester temporel. Une validation croisée
classique mélangerait les années. Le projet fait en revanche une **validation
croisée spatiale** : on retire une région entière, on entraîne, on teste sur
la région exclue, neuf fois.

**Votre modèle déployé est le moins performant. Ce n'est pas contradictoire ?**
Non, parce que la performance est mesurée dans des conditions qui n'existent
pas en production. v3 a toujours l'historique des feux en validation comme en
test ; il ne l'aurait pas demain matin. Le défaut n'apparaît dans **aucune
métrique d'entraînement** : il ne se voit qu'en se demandant ce que
deviendrait le modèle en service. Et l'écart coûte 3 points de rappel à 1 %
de budget, rien à 10 %.

**Comment savez-vous que le modèle n'a pas simplement appris la géographie ?**
Il n'a ni latitude ni longitude. La carte qu'il produit retrouve pourtant les
Landes, l'arc méditerranéen et la Corse, à partir de la végétation, du relief
et du littoral. Et la validation croisée spatiale montre qu'il **transfère** à
des régions retirées de l'entraînement.

**TreeSHAP est exact, dites-vous. Vous l'avez vérifié ?**
Oui : la somme des contributions plus la valeur de base redonne le logit du
score à 9,5 × 10⁻⁷ près.

### 6.4 Mise en production, MLOps

**À quelle fréquence faut-il réentraîner ?**
Le modèle C ne dépend d'aucune donnée qui bouge en cours d'année. Un
réentraînement annuel, quand CORINE ou le référentiel communal changent,
suffirait. C'est un avantage direct du choix de déploiement.

**Comment détecteriez-vous une dérive ?**
Sur les entrées, en surveillant la distribution des 8 indices CEMS par mois
contre la climatologie 2006-2019. Sur les sorties, en comparant le rappel à
budget fixe d'une année sur l'autre. Ce n'est pas en place — c'est une limite.

**Quelle est la latence d'un calcul quotidien ?**
Le modèle score 34 734 communes en une passe ; le coût est dominé par le
chargement des contours, pas par l'inférence. L'application met environ deux
minutes à démarrer à froid, essentiellement pour charger les polygones.

**Le modèle est-il utilisable par un SDIS demain matin ?**
Pas en l'état. Il donne un **classement national par jour**, ce qui répond à
« où regarder en priorité », mais il n'est pas calibré en probabilité, il ne
tient pas compte des moyens disponibles, et il n'a pas été évalué en
conditions opérationnelles. Ce qu'il fournit est une aide au ciblage.

### 6.5 Questions pièges

**« Votre modèle est fiable à 99,98 %, c'est ça ? »**
Non, et c'est précisément le piège que la rareté tend. Répondre toujours
« non » donne 99,98 %. Le nombre à retenir est le lift : **63,7 fois mieux que
le hasard**.

**« Donc les feux augmentent avec le changement climatique ? »**
Mes données ne le montrent pas. Les conditions **favorables** aux feux
augmentent très significativement — FWI estival +62 %, jours de danger élevé
+197 %. Le nombre de départs, lui, ne montre **aucune tendance significative**
sur les 20 années dont je dispose. Trois lectures restent possibles : faible
puissance statistique sur 20 points, prévention qui absorbe pour l'instant la
hausse de l'aléa, ou déplacement vers des feux plus grands plutôt que plus
nombreux. **Je ne peux pas trancher, et je ne le ferai pas.**

**« Votre carte montre où il brûlera en 2050 ? »**
Elle montre l'**aléa météo** sous le climat de 2050, appliqué à un jour
ordinaire. Pas le nombre de feux, pas la localisation des incendies. La
végétation et la prévention y sont supposées constantes, ce qui est faux.

**« XGBoost bat donc le MLP. »**
Non. Leurs intervalles de confiance se recouvrent : **je ne peux pas les
départager**.

**« Pourquoi ne pas utiliser simplement le FWI, qui existe déjà ? »**
Parce qu'il vaut ×5 le hasard seul, contre ×42 croisé au territoire. Le FWI
est l'entrée principale du modèle, pas son concurrent. Ce que le projet
ajoute, c'est une résolution spatiale : à l'intérieur d'une maille de 28 km on
trouve 31 communes qui reçoivent aujourd'hui le même indice, qu'elles soient
couvertes de maquis ou bétonnées.

**« Les communes voisines d'un feu ont donc un risque nul ? »**
Non. Mesuré sur Appietto (Corse-du-Sud) le 12 août 2024 : ses **40 communes
voisines dans 20 km qui n'ont pas brûlé** ont toutes un score non nul, un rang
médian de 375ᵉ sur 34 734, et **les 40 sont dans le décile le plus à risque**.
Le modèle déployé n'a pourtant aucune variable d'historique : elles remontent
parce qu'elles partagent la maille météo et la végétation.

**« Et la propagation de proche en proche entre communes ? »**
Elle n'est pas dans le projet. La variable est conçue et écrite
(`sql/41_feat_voisinage.sql.reporte`), la table de voisinage est chargée, mais
la requête n'a jamais été exécutée et aucun modèle n'a de colonne
`feux_voisins_*`. C'est la première extension prévue. Ce qui joue ce rôle
aujourd'hui est le clustering, où la proximité est une **ressemblance
physique** et non une adjacence géographique.

**« Si vous aviez trois mois de plus ? »**
Dans l'ordre : donner `danger_effis` au LSTM pour lever l'asymétrie de la
comparaison, refaire une calibration propre sur le modèle déployé et la bonne
période, livrer la feature de voisinage, et chercher une source de cause de
départ moins lacunaire que 64 % de manquants.

**« Quelle erreur vous a coûté le plus cher ? »**
Le désalignement des lignes, parce qu'il ne lève aucune exception et donne un
résultat plausible. J'ai failli conclure que le LSTM perdait de 97 %.

---

## 7. Ce que le projet ne fait pas

| Limite | Chiffre |
|---|---|
| La surface brûlée n'est pas prédictible | R² 0,144, moins bon que la médiane constante |
| « Sera-ce un grand feu » se prédit mal | lift 2,88 contre 63,7 pour les départs |
| Une commune-jour n'est pas un incendie | un feu sur 5 communes compte 5 fois |
| 31 communes partagent une maille météo | les intervalles naïfs sur les coefficients météo seraient trop étroits |
| Le LSTM n'a pas reçu `danger_effis` | les 23,6 % sont un majorant |
| La BDIFF n'est pas homogène | 64 % de causes manquantes, couverture variable |
| Pas de détection de dérive en production | non implémenté |
| 38 communes sans contour cartographique | code changé entre le millésime du fond et le COG 2026 |

Sur la ROC-AUC de 0,766 du modèle « grand feu » : elle est exacte, mais s'en
servir après avoir expliqué pourquoi la ROC-AUC flatte serait se contredire.
Le chiffre honnête sur cette tâche est le lift de 2,88.

---

## 8. Glossaire

| Terme | Définition |
|---|---|
| **FWI** | *Fire Weather Index*, indice synthétique du système canadien |
| **FFMC** | humidité de la litière fine, 1-2 cm, mémoire courte |
| **DMC** | *Duff Moisture Code*, 5-10 cm, constante de temps 15 jours |
| **DC** | *Drought Code*, 10-20 cm, constante de temps 52 jours |
| **BUI** | *Build-Up Index*, combustible disponible, fonction de DMC et DC |
| **ISI** | *Initial Spread Index*, vitesse de propagation |
| **ERC** | *Energy Release Component*, énergie libérable par unité de surface |
| **KBDI** | *Keetch-Byram Drought Index*, sécheresse du sol |
| **PR-AUC** | aire sous la courbe précision-rappel ; vaut le taux de base au hasard |
| **lift** | PR-AUC ÷ taux de base |
| **ACF** | autocorrélation : corrélation entre t et t−k, effets indirects compris |
| **PACF** | autocorrélation **partielle** : apport propre du retard k |
| **ADF** | Dickey-Fuller augmenté ; H₀ = racine unitaire = non stationnaire |
| **SARIMAX** | ARIMA saisonnier avec variables exogènes |
| **DART** | *Dropouts meet Multiple Additive Regression Trees* : boosting où 10 % des arbres sont éteints à chaque itération, une itération sur deux |
| **SHAP** | décomposition exacte d'un score en contributions, valeurs de Shapley |
| **LIME** | substitut linéaire local, approché |
| **DiCE** | génération de contrefactuels |
| **Pseudo-réplication** | traiter des observations corrélées comme indépendantes |
| **Biais de collision** | conditionner sur une variable causée par deux autres, ce qui crée une association artificielle entre elles |
| **Small area estimation** | estimer une quantité sur des domaines à trop peu d'observations |
| **RCP** | *Representative Concentration Pathway*, forçage radiatif en 2100 (W/m²) |

---

## 9. Commandes

Relancer la comparaison de modèles, puis remettre le support et la vitrine
d'aplomb :

```bash
python -m tvfed.comparer && python -m tvfed.vitrine
```

Relancer l'analyse de séries temporelles (ADF, ACF/PACF, SARIMAX) :

```bash
python -m tvfed.series
```

Lancer l'application en local :

```bash
streamlit run app/Carte.py
```

Faire tourner les tests :

```bash
pytest tests/ -q
```

> ⚠️ `python -m tvfed.diaporama` réécrit `presentation/soutenance.pptx`
> **entièrement**. Le fichier de travail a été étendu à la main (clôture et
> annexe de captures) : régénérer effacerait ces diapositives.
