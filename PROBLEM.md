# Le problème, avant les modèles

Document de cadrage. Il fixe ce qu'on cherche, ce qui compte comme une réponse,
et ce qu'on s'interdit — **avant** d'avoir vu les résultats. Il est daté et il
n'a pas été réécrit après coup.

---

## 1. La question

> **Pour une commune française et un jour donné, quel est le risque qu'un feu
> de forêt s'y déclare ?**

Et sa suite, qui décide de l'usage :

> **Où faut-il regarder en premier ?**

La seconde reformulation est celle qui compte. Un service départemental
d'incendie n'a pas les moyens de surveiller 34 734 communes : il en surveille
quelques centaines. La question opérationnelle n'est donc pas « quelle est la
probabilité » mais **« quel classement »**.

## 2. L'unité d'observation

**Une ligne = une commune × un jour.**

| | |
|---|---|
| Périmètre | France métropolitaine, 34 734 communes (COG 2026) |
| Période | 2006-2025, soit 7 305 jours |
| Volume | **253 731 870 lignes** |
| Cible | `y = 1` si ≥ 1 départ de feu ce jour-là dans cette commune |
| Positifs | **49 130**, soit **0,0194 %** |

⚠️ **Une commune-jour n'est pas un incendie.** Un feu traversant cinq communes
produit cinq lignes. « 49 130 feux » signifie 49 130 communes-jours ayant brûlé,
pas 49 130 incendies au sens des pompiers. Cette distinction doit apparaître
dans toute restitution.

## 3. Pourquoi la rareté change tout

À 0,0194 % de positifs, un modèle qui répond « jamais de feu » a **99,98 % de
justesse**. L'accuracy est donc inutilisable, et avec elle tout ce qui en
dérive.

La métrique retenue est la **PR-AUC** (aire sous la courbe précision-rappel),
pour une raison précise : **la PR-AUC d'un modèle aléatoire vaut exactement le
taux de positifs**. Le rapport entre les deux — le **lift** — se lit
directement :

```
lift = PR-AUC / taux de positifs = « combien de fois mieux que le hasard »
```

⚠️ **La PR-AUC n'est pas comparable entre périodes de rareté différente.** Le
test (0,0166 % de positifs) et la validation (0,0241 %) ne peuvent être
comparés que par le lift.

## 4. Ce qui compte comme un succès

Fixé avant la première mesure :

| Niveau | Lift | Lecture |
|---|---|---|
| inutile | < 5× | à peine mieux que le danger météo seul |
| exploitable | 20-50× | surveiller 1 % du territoire capte une part réelle des feux |
| bon | > 50× | |

Et une contrainte non négociable : **le modèle doit être explicable**. Une
alerte qu'on ne sait pas justifier ne sera pas suivie.

## 5. Les trois pièges, identifiés avant de commencer

### 5.1 La fuite temporelle

Une feature calculée sur des données postérieures à la date qu'elle décrit
produit un excellent score et un modèle inutilisable. **Elle ne déclenche
aucune erreur.**

**Règle adoptée**, et elle demande une distinction que peu de projets font :

| Type de feature | Peut lire quoi ? |
|---|---|
| **datée, passé strict** — « feux des 30 derniers jours » au 3 août 2023 | tout le passé, **y compris celui de sa propre période d'évaluation** — le 3 août à 8 h, juillet est connu |
| **statistique non datée** — un taux moyen sur toute la période | **le train uniquement** |

### 5.2 Le décalage de prior

Le train est sous-échantillonné à 1:10 sur les négatifs : le modèle y voit
9,1 % de positifs contre 0,019 % en réalité, **un facteur ×487**. Toute
statistique apprise sur le train échantillonné est donc fausse d'autant.

**Règle adoptée** : les agrégats de `y` se calculent sur le train **complet**
(177,6 M lignes), en SQL, jamais sur l'échantillon.

### 5.3 Le test brûlé

Un jeu de test regardé deux fois n'est plus un test.

**Règle adoptée** : le test 2023-2025 ne sert à **aucun** choix — ni features,
ni hyperparamètres, ni nombre de clusters, ni méthode de calibration, ni
nombre d'arbres. Tous ces choix se font sur un découpage interne au train ou
sur la validation. La configuration est empreintée (SHA-256) dans
`gel_avant_test.json` **avant** de regarder, avec la prédiction attendue.

## 6. Le découpage

```
train  2006-2019   177 594 942 lignes   33 632 feux   échantillonné 1:10
val    2020-2022    38 068 464 lignes    9 176 feux   INTÉGRAL
test   2023-2025    38 068 464 lignes    6 322 feux   INTÉGRAL, une seule fois
```

**Temporel, pas aléatoire.** Un découpage aléatoire mettrait le 14 août 2019
en train et le 15 août 2019 en test : les deux partagent la même sécheresse,
le même vent, souvent le même feu. Le score serait faux et personne ne le
verrait.

Val et test ne sont **jamais** échantillonnés : c'est la validation intégrale
qui porte le vrai taux de base, et donc la calibration.

## 7. Ce qu'on s'interdit

- **deviner un rattachement de commune par le nom** — testé, 3 faux sur 8.
  « Chirac » (Lozère) rapproché de Chirac en Charente. Un feu attribué à la
  mauvaise commune corrompt `y`, la végétation et le voisinage ;
- **mettre l'année en variable** — un arbre ne sait pas extrapoler ; `2050`
  tomberait dans la branche « ≥ 2019 » ;
- **annoncer un écart sans intervalle de confiance** — avec 9 176 positifs,
  tout écart sous ~1,5 % est dans le bruit ;
- **afficher une probabilité qu'on sait fausse** — l'application affiche un
  rang, la calibration disponible étant décalée d'un facteur ~2.

## 8. Ce qu'on ne cherche pas à faire

| | Pourquoi |
|---|---|
| prévoir la météo | le modèle **convertit** une météo en risque ; la prévision vient d'EFFIS, à 9 jours |
| prédire la surface exacte | elle dépend de ce qui se passe **après** le départ — vent, délai d'intervention, relief. Mesuré : R² 0,14, moins bon que d'annoncer la médiane |
| identifier une cause | 64 % des causes sont manquantes en BDIFF |
| couvrir l'outre-mer | hors emprise météo européenne et absent de CORINE — 1 378 feux exclus, comptés |

## 9. Les limites acceptées d'avance

- **~31 communes partagent une maille météo** : même FWI le même jour. Le FWI
  porte le *quand*, les features spatiales portent le *où*. Conséquence
  statistique : les intervalles de confiance naïfs sur les coefficients météo
  sont trop étroits (pseudo-réplication) ;
- **la BDIFF ne publie pas l'année en cours** — conséquence majeure, découverte
  en route : le modèle le plus performant n'est pas déployable ;
- **le modèle suppose stable tout ce qui n'est pas la météo** — prévention,
  pratiques agricoles, déprise rurale. Mesuré sur 20 ans, la dérive résiduelle
  n'est pas significative (p = 0,27), mais elle n'est pas nulle ;
- **dept. 64, 2010-2011** : données perdues, documenté par la BDIFF elle-même.

---

*Rédigé au cadrage, non modifié après les résultats. Les mesures qui ont
contredit les attentes sont conservées telles quelles dans les notebooks.*
