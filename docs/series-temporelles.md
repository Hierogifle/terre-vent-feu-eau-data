# ACF, PACF et SARIMAX — comprendre ce qu'on a mesuré

Un cours court, avec les résultats du projet comme exemples. L'idée n'est pas
de réciter les formules mais de savoir **quoi penser d'un graphique** quand on
le voit.

---

## 1. Le vocabulaire de base

Une série temporelle se décompose en trois morceaux :

```
série  =  tendance  +  saisonnalité  +  reste
          ↑            ↑                ↑
   la direction    ce qui se        ce qui n'est
   sur le long     répète à         expliqué par
   terme           période fixe     aucun des deux
```

Sur nos feux quotidiens en France :

| Composante | Ce qu'on a mesuré |
|---|---|
| tendance | FWI **+45 % sur 53 ans** (p < 0,0001) ; feux **stables** (p = 0,89) |
| saisonnalité | **×14,1** entre juillet et décembre — 34 % de la variance |
| reste | 43 % — l'allumage lui-même, qu'aucune donnée ne contient |

### Stationnarité — le piège du vocabulaire

**Stationnaire ne veut pas dire « sans cycle ».**

Une série est stationnaire si sa moyenne, sa variance et son autocovariance ne
dépendent pas de *quand* on regarde. Une sinusoïde parfaite est stationnaire :
elle revient toujours.

Ce qui casse la stationnarité, c'est une **racine unitaire** — une dérive sans
ancre. Un cours de bourse : il part et ne revient pas. Chaque choc s'accumule
définitivement.

```
STATIONNAIRE           NON STATIONNAIRE (racine unitaire)
    ∿∿∿∿∿∿∿                        ／
  ──────────                    ／
    ∿∿∿∿∿∿∿                  ／
revient toujours          part et ne revient jamais
```

Pourquoi c'est important : la plupart des modèles supposent la stationnarité.
Sur une série non stationnaire, deux séries sans aucun lien peuvent paraître
fortement corrélées — c'est la **régression fallacieuse**.

---

## 2. ADF — le test de Dickey-Fuller augmenté

### Ce qu'il teste

> **H₀ : la série a une racine unitaire** (donc elle n'est PAS stationnaire)

⚠️ **Le sens du test est contre-intuitif, et c'est l'erreur la plus fréquente.**

| p-value | Décision | Conclusion |
|---|---|---|
| **p < 0,05** | on **rejette** H₀ | la série **EST stationnaire** ✅ |
| p ≥ 0,05 | on ne rejette pas | on ne peut pas conclure — *pas* « elle est non stationnaire » |

Retiens : **petit p = bonne nouvelle = stationnaire**.

### Le « augmenté »

Le test de Dickey-Fuller original suppose que le reste est du bruit blanc. Le
nôtre ne l'est pas — il y a de l'autocorrélation. La version **augmentée**
ajoute des retards de la série différenciée pour absorber cette structure.
Le nombre de retards est choisi automatiquement (critère AIC).

### Nos résultats

```
série                            stat ADF         p   retards   conclusion
feux par jour                     -10.116    0.0000        13   STATIONNAIRE
FWI national                       -7.301    0.0000        21   STATIONNAIRE
feux, différenciée 1 jour         -24.561    0.0000        23   STATIONNAIRE
feux par AN (20 points)            -3.850    0.0024         8   STATIONNAIRE
```

**Comment lire.** La statistique ADF est très négative et p est nul : on
rejette franchement H₀. Les séries reviennent toujours vers leur régime.

**Conséquence pratique : `d = 0`.** Le `d` de ARIMA(p,**d**,q) est le nombre de
différenciations nécessaires. Comme la série est déjà stationnaire, on n'en
fait aucune.

⚠️ **Ce que le test ne dit PAS** : rien sur la saisonnalité. Notre cycle ×14,1
est bien là, et l'ADF le déclare stationnaire — parce qu'un cycle revient, il
ne dérive pas. Pour tester une racine unitaire *saisonnière*, il faudrait un
autre test (HEGY, OCSB).

---

## 3. ACF — la fonction d'autocorrélation

### Ce que c'est

La corrélation de la série avec elle-même, décalée de *k* pas.

```
ACF(1)  = corr( aujourd'hui , hier )
ACF(7)  = corr( aujourd'hui , il y a une semaine )
ACF(365)= corr( aujourd'hui , il y a un an )
```

Nos valeurs :

```
retard 1   : 0.801     ← très forte
retard 7   : 0.542
retard 30  : 0.324
retard 182 : -0.123    ← demi-année : été contre hiver
retard 365 : +0.300    ← un an : la saison revient
```

L'alternance −0,12 à six mois / +0,30 à un an est **la signature du cycle
annuel**, lisible directement.

### ⚠️ Le piège qui nous a eus

Sur la **série brute**, l'ACF décroît lentement sur 60 jours et ne montre
qu'une chose : *c'est l'été*. Le cycle saisonnier écrase tout le reste.

C'est exactement ce qui s'est produit au début du projet sur les indices CEMS :
toutes les courbes tombaient à zéro vers 90 jours, et j'en avais conclu à tort
que la mémoire physique des indices valait 90 jours. **90 jours, c'est un quart
de cycle annuel** — je lisais la saison, pas le combustible.

**Il faut retirer le cycle avant de lire l'ACF.** Une fois la saisonnalité
enlevée, les vraies échelles apparaissent : FFMC ~3 jours, DMC ~15, DC ~50.

---

## 4. PACF — l'autocorrélation *partielle*

### La différence avec l'ACF

L'ACF à 3 jours contient l'effet indirect : aujourd'hui ressemble à hier, hier
ressemble à avant-hier, donc aujourd'hui ressemble à avant-hier **par
transitivité**.

La PACF retire cet effet de chaîne :

> **PACF(k) = ce que le jour *k* apporte EN PLUS de tous les jours entre lui
> et aujourd'hui.**

Une analogie : l'ACF, c'est « les enfants de parents grands sont grands, et
leurs petits-enfants aussi ». La PACF demande : *à taille des parents connue*,
la taille des grands-parents apporte-t-elle encore quelque chose ?

### À quoi elle sert

À **choisir l'ordre AR**. C'est sa seule vraie utilité pratique.

| Ce qu'on voit | Ce qu'on choisit |
|---|---|
| PACF coupe net après *p*, ACF décroît doucement | **AR(p)** |
| ACF coupe net après *q*, PACF décroît doucement | **MA(q)** |
| les deux décroissent doucement | **ARMA(p,q)**, à tâtonner |

### Nos résultats

```
seuil de significativité  ±0.0229

retard  1   +0.6972   significatif  ████████████████
retard  2   +0.1906   significatif  ████
retard  3   +0.0768   significatif  █
retard  4   +0.0413   significatif
retard  5   +0.0243   significatif
retard  6   +0.0483   significatif
retard  7   +0.0145
retard  8   +0.0337   significatif
```

**Comment lire.** Chute brutale après le retard 2 : 0,70 → 0,19 → 0,08. Les
retards suivants sont significatifs mais **négligeables** — avec 7 305 points,
le seuil de significativité est minuscule (±0,023) et presque tout le franchit.

→ On retient **AR(2)**. Significatif ne veut pas dire important.

---

## 5. SARIMAX

### Décomposer l'acronyme

```
S      Seasonal      composante saisonnière
AR     AutoRegressive   la valeur dépend des précédentes
I      Integrated       on différencie d fois
MA     Moving Average   la valeur dépend des erreurs précédentes
X      eXogenous        variables extérieures
```

Notation : **SARIMA(p,d,q)(P,D,Q)ₛ** plus les exogènes.

| Terme | Sens | Notre choix | Pourquoi |
|---|---|---|---|
| **p** | ordre AR | **2** | la PACF coupe après 2 |
| **d** | différenciations | **0** | l'ADF dit stationnaire |
| **q** | ordre MA | **1** | par défaut raisonnable |
| **P,D,Q,s** | saison | **aucun** | voir ci-dessous |
| **X** | exogènes | Fourier + FWI | voir ci-dessous |

### ⚠️ Pourquoi aucune composante saisonnière au sens SARIMA

Une saisonnalité annuelle sur données journalières donnerait **s = 365**. Le
modèle devrait estimer des coefficients à 365 pas de distance, sur 5 113 points
d'ajustement. C'est ingérable et instable.

**La pratique établie sur données journalières** est de porter la saisonnalité
par des **termes de Fourier en exogène** : quelques paires sin/cos suffisent à
décrire un cycle annuel lisse.

```
sin(2πk·jour/365,25)  et  cos(2πk·jour/365,25)   pour k = 1..4
```

Quatre harmoniques = 8 colonnes, contre 365 coefficients. C'est le **X** de
SARIMAX qui fait le travail de saisonnalité.

### Nos résultats

Ajusté sur 2006-2019, évalué sur 2020-2022 :

| Modèle | AIC | MAE | RMSE | corrélation |
|---|---|---|---|---|
| **SARIMAX(2,0,1) + Fourier + FWI** | **29 400** | **4,03** | 7,27 | **0,850** |
| SARIMAX(2,0,1) + Fourier seul | 30 146 | 6,42 | 11,30 | 0,603 |
| ARIMA(2,0,1) sans exogène | 30 195 | 8,34 | 13,80 | **−0,118** |
| référence : moyenne du jour de l'année | — | 5,13 | 9,21 | 0,598 |

### Quoi en penser — trois lectures

**1. L'ARIMA seul ne prédit rien (r = −0,118).**

Ce n'est pas un bug, et ça ne veut pas dire qu'il n'y a pas de structure
temporelle. C'est un problème d'**horizon**.

Un AR(2) prédit à partir des deux jours précédents. On lui demande de prévoir
**1 096 jours d'avance** sans jamais lui redonner d'observation. Après quelques
pas la mémoire est épuisée et la prévision converge vers la moyenne — une ligne
plate. Une ligne plate a une corrélation nulle avec n'importe quoi.

> **Règle** : la mémoire autorégressive sert à prévoir **à court terme**. Au-delà
> de quelques pas, seuls la saisonnalité et les exogènes portent de
> l'information.

**2. Le FWI fait tout le travail (MAE 6,42 → 4,03, soit −37 %).**

C'est la mesure directe de la thèse du projet : *la météo dit quand*. Sans elle,
on ne sait que la saison ; avec elle, on sait que **cette semaine-là** il a fait
sec.

**3. Battre la référence de 21,5 % est le seul chiffre qui compte.**

Un MAE de 4,03 ne veut rien dire dans l'absolu. Ce qui compte est de le comparer
à la **prévision naïve** : « la moyenne de ce jour de l'année sur le passé »,
qui donne 5,13. Le modèle fait **21,5 % mieux**.

⚠️ **Toujours comparer un modèle de série temporelle à une référence naïve.**
Sur une série saisonnière, un modèle médiocre peut afficher un beau R² en ne
faisant que reproduire la saison.

### Lire l'AIC

L'AIC arbitre entre qualité d'ajustement et nombre de paramètres. **Plus bas =
meilleur**, et seuls les écarts comptent :

```
Δ AIC < 2      les modèles sont équivalents
Δ AIC 4-7      le plus bas est nettement préféré
Δ AIC > 10     le plus haut est à écarter
```

Nos écarts : 29 400 contre 30 146 → **Δ = 746**. Le modèle avec FWI n'est pas
« un peu mieux », il est d'une autre catégorie.

⚠️ L'AIC se calcule **sur l'ajustement**. Il ne remplace pas une évaluation
hors échantillon — un modèle peut avoir le meilleur AIC et prédire moins bien.
Ici les deux concordent, ce qui est rassurant.

---

## 6. Ce qu'il faut retenir pour la soutenance

**Sur nos données**

| Question | Réponse | Preuve |
|---|---|---|
| Y a-t-il un cycle ? | **oui, massif** | ×14,1 juillet/décembre, 34 % de la variance |
| Y a-t-il une tendance ? | **oui sur le FWI** | +45 % en 53 ans, p < 0,0001 |
| … sur les feux ? | **non** | +3,7/an, p = 0,89 |
| La série est-elle stationnaire ? | **oui** | ADF p < 0,0001 |
| Peut-on prévoir loin sans météo ? | **non** | ARIMA seul : r = −0,118 |

**Sur la méthode**

- *stationnaire* ≠ *sans cycle* — une sinusoïde est stationnaire ;
- ADF : **petit p = stationnaire**, l'inverse de l'intuition ;
- lire une ACF **sans retirer la saison** conduit à des conclusions fausses ;
- la PACF sert à **choisir l'ordre AR**, rien d'autre ;
- sur données journalières, la saisonnalité passe par **Fourier**, pas par
  s = 365 ;
- toujours comparer à une **référence naïve** ;
- avec 7 000 points, *significatif* ne veut pas dire *important*.

**Et pourquoi SARIMAX ne remplace pas le modèle principal**

SARIMAX modélise **une** série. Notre problème principal est un **panneau** de
34 734 séries simultanées, dont 80 % ne contiennent aucun événement. Le modèle
XGBoost répond au **où** ; SARIMAX répond au **combien**. Les deux sont
complémentaires, et le second **valide le premier par une voie indépendante** :
autre cible, autre méthode, autre échelle, même conclusion.

---

*Reproductible par `python -m tvfed.series`. Les résultats chiffrés viennent de
`data/processed/series_adf.csv` et `series_sarimax.csv`, la figure ACF/PACF de
`series_acf_pacf.png`.*
