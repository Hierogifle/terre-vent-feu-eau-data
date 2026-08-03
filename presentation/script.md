# Script de soutenance — Terre, Vent, Feu, Eau, Data

> Ce document accompagne `soutenance.pptx` (31 diapositives). Il contient
> **ce qu'il faut dire**, diapositive par diapositive, plus ce que la
> diapositive ne montre pas et qu'un jury peut demander.
>
> **Durée visée : 20 minutes** de présentation + 10 de questions. Les
> diapositives marquées ⏩ peuvent être sautées si le temps manque.
>
> Les chiffres de ce document et ceux du diaporama viennent des **mêmes
> fichiers** (`data/processed/*.csv`). Si vous relancez un entraînement,
> relancez `python -m tvfed.diaporama` : le diaporama se met à jour seul.

---

## Table des matières

| § | Diapos | Sujet |
|---|---|---|
| [0](#0--avant-de-commencer) | — | Avant de commencer |
| [1](#1--le-problème-diapos-1-à-5) | 1-5 | Le problème |
| [2](#2--le-protocole-diapos-6-à-9) | 6-9 | Le protocole |
| [3](#3--les-modèles-diapos-10-à-20) | 10-20 | Les modèles |
| [4](#4--comprendre-le-modèle-diapos-21-à-23) | 21-23 | Comprendre le modèle |
| [5](#5--le-climat-diapos-24-à-26) | 24-26 | Le climat |
| [6](#6--rigueur-diapos-27-à-31) | 27-31 | Rigueur, application, limites |
| [A](#annexe-a--fiche-de-chaque-modèle) | — | **Fiche de chaque modèle** |
| [B](#annexe-b--questions-attendues-et-réponses) | — | **Questions attendues** |
| [C](#annexe-c--glossaire) | — | Glossaire |

---

## 0 · Avant de commencer

### La phrase qui résume tout

> « J'ai construit un modèle qui note le risque de départ de feu pour chacune
> des 34 734 communes françaises, chaque jour. Il fait **63,7 fois mieux que
> le hasard** sur une période qu'il n'a jamais vue. Mais le résultat dont je
> suis le plus content est **négatif** : j'ai montré que trois de mes six
> modèles sont indiscernables, alors que leurs scores semblaient les
> départager. »

### Les trois choses à ne pas rater

1. **La rareté commande tout** — 0,019 % de positifs. Chaque décision
   méthodologique en découle. Si le jury ne retient qu'une chose, c'est ça.
2. **Le meilleur modèle n'est pas le déployable** — et la raison n'est pas
   dans les métriques, elle est dans la disponibilité de la donnée.
3. **On assume les erreurs** — quatre bugs sérieux trouvés et corrigés, dont
   un qui rendait une comparaison entière fausse. Les montrer est un
   avantage, pas un aveu.

### Ce qu'il ne faut PAS dire

| ❌ Ne pas dire | ✅ Dire |
|---|---|
| « les feux augmentent » | « les conditions favorables augmentent ; le nombre de départs reste stable » |
| « mon modèle prédit les feux de 2050 » | « il donne le risque d'un jour ordinaire sous le climat de 2050 » |
| « XGBoost est meilleur que le MLP » | « les deux sont indiscernables, l'intervalle traverse zéro » |
| « le modèle est précis à 99,98 % » | « l'accuracy n'a aucun sens ici ; on mesure la PR-AUC » |
| « le LSTM n'a pas marché » | « le LSTM perd de 23,6 %, et voici pourquoi c'était prévisible » |

---

## 1 · Le problème (diapos 1 à 5)

### Diapo 1 — Titre

**À l'écran** : le titre, et la carte de France du 12 août 2024.

**À dire** (30 s) :

> « Ce projet répond à une question opérationnelle : *où faut-il envoyer les
> moyens de surveillance demain ?* La carte que vous voyez est une sortie
> réelle du modèle, pour le 12 août 2024. Il retrouve les Landes, l'arc
> méditerranéen, la Corse et le piémont pyrénéen — et je précise tout de
> suite qu'il **n'a jamais vu de carte** : ni la latitude ni la longitude ne
> font partie de ses variables. Il reconstruit cette géographie à partir de
> la végétation, du relief et de la météo. »

> 💡 C'est une bonne accroche : elle prouve immédiatement que le modèle a
> appris quelque chose de physique, pas un code postal.

### Diapo 2 — Section « Le problème »

Transition. Rien à dire, ou une phrase : « Commençons par ce qui rend ce
problème difficile. »

### Diapo 3 — La rareté

**À l'écran** : 0,019 % en très gros.

**À dire** (1 min 30) :

> « Sur les 253 millions de couples commune-jour de la période 2006-2025,
> **0,019 %** comptent au moins un départ de feu. Ça a trois conséquences.
>
> D'abord, **l'accuracy est inutilisable** : un modèle qui répond toujours
> "non" a 99,98 % de justesse et ne sert à rien.
>
> Ensuite, et c'est plus grave, **une fuite de données ne produit pas
> d'erreur**. Elle produit d'excellentes métriques et un modèle sans valeur.
> Personne ne vous prévient.
>
> Enfin, ce déséquilibre commande tout le reste : la métrique, la façon
> d'échantillonner, la calibration, et jusqu'à la manière de comparer deux
> modèles. »

### Diapo 4 — Les quatre sources

**À dire** (1 min 30) :

> « Quatre sources publiques, croisées sur le code INSEE et sur une grille
> météo de 0,25 degré.
>
> Le CEMS de Copernicus donne huit indices de danger par jour et par maille,
> sur 53 ans — 21,9 millions de lignes. La BDIFF de l'IGN donne les feux
> déclarés. CORINE donne l'occupation du sol. L'INSEE donne le référentiel des
> communes.
>
> Le point délicat n'est pas le volume, c'est **les fusions de communes**.
> 965 feux portent un code INSEE qui n'existe plus. J'ai d'abord essayé de les
> rapprocher par le nom : ça donnait des faux positifs — "Chirac", en Lozère,
> renvoyait vers une commune de Charente. J'ai donc téléchargé le fichier
> officiel des mouvements de communes de l'INSEE, et j'ai écarté les 30 cas
> irrécupérables **en les comptant**. Injecter une fausse géolocalisation
> aurait été pire que perdre 0,68 % des feux : ça aurait corrompu la cible, le
> voisinage et les features CORINE de communes innocentes. »

> 💡 C'est le premier moment où vous montrez que vous avez préféré la
> justesse à la complétude. Insistez-y.

### Diapo 5 — La grille dense

**À dire** (1 min) :

> « La table centrale est une grille **commune × jour** : une ligne par
> commune et par jour, qu'il y ait eu un feu ou non. 253 millions de lignes.
>
> On me demande souvent pourquoi ne pas se contenter de la liste des 52 809
> feux. Parce qu'une série creuse rendrait les **fenêtres glissantes
> silencieusement fausses**. "Nombre de feux dans les 30 jours précédents" se
> calcule en remontant 30 lignes ; si les jours sans feu sont absents, on
> remonte en réalité plusieurs années. C'est la raison technique numéro un
> d'avoir matérialisé 253 millions de lignes plutôt que 52 809. »

---

## 2 · Le protocole (diapos 6 à 9)

### Diapo 6 — Section

> « Avant de parler de modèles, je vais parler du protocole. C'est lui qui
> protège le résultat. »

### Diapo 7 — La barrière du split

**À dire** (2 min) :

> « Le découpage est **temporel**, jamais aléatoire.
>
> Le train, 2006 à 2019 : 177 millions de lignes. Tout ce qui a un `.fit()`
> s'ajuste ici et nulle part ailleurs.
>
> La validation, 2020 à 2022 : c'est là que je choisis. Hyperparamètres,
> sélection du modèle, calibration, comparaisons.
>
> Le test, 2023 à 2025 : je l'ai ouvert **une seule fois**, après avoir tout
> gelé. Aucune décision n'en découle.
>
> Pourquoi pas un découpage aléatoire ? Parce qu'il mettrait le 14 juillet
> 2019 dans le train et le 15 dans le test. Le modèle "prédirait" un feu qu'il
> a déjà vu, à vingt kilomètres et un jour d'écart.
>
> Et j'ai une règle qui tranche les cas douteux : **une feature datée peut
> regarder tout le passé, y compris celui de sa propre période d'évaluation ;
> une statistique non datée ne peut regarder que le train.** Concrètement :
> "feux des 30 jours précédents" au 3 août 2023 lit juillet 2023 — ce n'est
> pas une fuite, le 3 août à 8 h du matin on connaît juillet. En revanche
> "taux moyen de la commune sur toute la période" lit le futur : c'en est
> une. »

### Diapo 8 — PR-AUC

**À dire** (1 min 30) :

> « La ROC-AUC est flatteuse et inutile ici : les vrais négatifs écrasent
> tout, et un modèle médiocre affiche 0,95.
>
> La **PR-AUC** a une propriété qui la rend lisible : quand on répond au
> hasard, elle vaut **exactement le taux de base**. Le rapport des deux — que
> j'appelle le **lift** — se lit donc directement : "le modèle est N fois
> meilleur que tirer au sort". C'est un nombre qu'on peut dire à voix haute
> devant un décideur. »

> ⚠️ Si on vous demande « et le F1 ? » : le F1 exige de choisir un seuil, et
> le bon seuil dépend du budget de surveillance, qui n'est pas une donnée du
> problème. La PR-AUC intègre tous les seuils.

### Diapo 9 — Le prior déplacé

**À dire** (1 min 30) :

> « Le train est sous-échantillonné à un positif pour dix négatifs — sans
> quoi l'entraînement serait ingérable sur 177 millions de lignes.
>
> Conséquence : le modèle apprend sur un monde où 9,1 % des jours sont des
> jours de feu, alors que le vrai taux est 0,019 %. Un facteur **487**.
>
> Trois garde-fous. D'abord, **validation et test ne sont jamais
> échantillonnés** : c'est ce qui rend les scores comparables au monde réel.
> Ensuite, **toutes les statistiques dérivées de la cible se calculent sur le
> train complet**, pas sur l'échantillon — sinon un lissage bayésien vaudrait
> 9,1 % au lieu de 0,019 %, et rien dans les métriques ne le signalerait.
> Enfin, la **calibration** absorbe le décalage : la méthode de Platt ramène
> le biais de ×144,7 à ×1,13, sans rien coûter en PR-AUC. »

---

## 3 · Les modèles (diapos 10 à 20)

### Diapo 10 — Section

### Diapo 11 — Les baselines

**À dire** (1 min) :

> « Avant de montrer un chiffre, il faut dire contre quoi on se bat.
>
> Trois prédicteurs sans aucun apprentissage, mesurés sur la même validation.
> L'historique spatial seul vaut déjà **×19**. La météo seule, via le danger
> EFFIS officiel, vaut **×5**. Leur croisement, **×42**.
>
> C'est ça, la barre à battre — pas le hasard. Un modèle qui ferait ×30 serait
> **moins bon qu'une règle de trois**. »

### Diapo 12 ⏩ — Du v1 au v3

**À dire** (1 min 30) :

> « Ma première version tirait 54,6 % de son importance de l'historique de la
> commune. Elle disait surtout : *ce qui a brûlé rebrûlera*.
>
> Ce n'est pas une fuite — le jour J on connaît réellement le passé — mais ça
> laissait un trou : une commune qui n'a jamais brûlé gardait un score bas,
> même entourée de communes qui brûlent chaque été. C'est le problème
> classique de *small area estimation*.
>
> La parade : regrouper les communes qui se ressemblent **physiquement** —
> trente groupes formés sur la végétation, le relief, la densité humaine et la
> climatologie, **sans jamais regarder le feu** — puis faire retomber chaque
> commune vers le taux de son groupe, à proportion de ce qu'on sait d'elle.
>
> Le gain mesuré est de **+1,3 %** de PR-AUC. Réel, mais modeste — et à peine
> significatif au bootstrap. Je le dis parce que c'est un résultat : ce
> clustering, qui m'a demandé le plus de travail, rapporte **quatre fois
> moins** que le simple réglage des hyperparamètres. »

### Diapo 13 — Les six modèles

**À dire** (1 min) — *lire le tableau sans le commenter* :

> « Voici les six modèles sur la même validation : 38 millions de
> communes-jours, 9 176 feux.
>
> Je ne commente volontairement pas ce classement. Un écart de PR-AUC ne veut
> rien dire sans intervalle de confiance — c'est l'objet de la diapositive
> suivante. »

### Diapo 14 — Le graphique en forêt ⭐

**C'est la diapositive la plus importante de la soutenance.**

**À dire** (2 min 30) :

> « J'ai fait un **bootstrap apparié** : 200 répliques, en rééchantillonnant
> les 34 734 **communes** — pas les lignes. C'est important : les 1 096 jours
> d'une même commune ne sont pas indépendants, et 31 communes partagent en
> moyenne la même maille météo. Un bootstrap ligne à ligne donnerait des
> intervalles faussement étroits, et me ferait conclure à tort.
>
> Résultat : **DART et le MLP paraissaient 1,8 % et 1,9 % moins bons que
> XGBoost. Leurs intervalles traversent zéro. Les trois modèles sont
> indiscernables.**
>
> Autrement dit, si j'avais présenté « XGBoost bat le MLP », j'aurais énoncé
> une conclusion **inventée à partir du bruit**. C'est le résultat dont je
> suis le plus content, et c'est un résultat négatif.
>
> Seuls deux écarts survivent : celui du modèle physique, et celui du LSTM. »

> 💡 Si vous n'avez que cinq minutes pour convaincre, montrez cette
> diapositive-là.

**Détail technique, si on demande** : le bootstrap est rendu praticable par
une astuce. Recalculer la PR-AUC 200 fois sur 38 millions de lignes coûterait
des heures, chaque appel retriant le tableau. On trie **une fois** ; une
réplique n'est alors qu'un jeu de **poids entiers** le long de cet ordre figé,
et l'average precision pondérée se calcule en O(n) par somme cumulée. Vérifié
identique à scikit-learn à 1e-12 près, ex æquo et poids compris.

### Diapos 15 et 16 — Le LSTM ⭐

**À dire** (3 min) :

> « "Pour le temps, prends un LSTM" est le réflexe standard, et on me l'a dit
> plusieurs fois. Je l'ai donc construit, optimisé, et mesuré.
>
> **Il perd de 23,6 %**, intervalle de −33,5 à −17,3, loin de zéro.
>
> Première objection possible : il n'a pas été assez réglé. Elle ne tient
> pas : **25 essais Optuna** sur sept hyperparamètres, avec arrêt précoce. Les
> valeurs retenues sont à l'écran.
>
> Deuxième point, plus subtil : **la comparaison loyale n'est pas celle qu'on
> croit**. XGBoost v3 voit l'historique des feux — 29 % de ses importances. Le
> LSTM n'en voit rien. Les opposer mesurerait le prix de l'information
> retirée, pas la valeur de la séquence. La seule référence à jeu
> d'information égal est le modèle physique. Et là encore, le LSTM perd — alors
> qu'il voit **30 jours × 8 indices, soit 240 valeurs**, contre onze features
> météo pour le modèle physique. Vingt fois plus d'historique.
>
> *(passer à la diapo 16)*
>
> Pourquoi ? L'explication est **physique, pas informatique**.
>
> Un LSTM sert quand l'ordre de la séquence porte une information qu'aucun
> résumé ne capture. Ici, ce résumé existe déjà. Les indices DC, DMC et BUI du
> système canadien **sont** des états récursifs : le *Drought Code* est
> littéralement une moyenne exponentielle de la météo passée, avec une
> constante de temps de 52 jours. Le *Duff Moisture Code*, de 15 jours. C'est
> exactement la forme d'une cellule récurrente — sauf que ses coefficients ont
> été calibrés par cinquante ans de science du feu, plutôt qu'estimés sur
> 9 176 exemples positifs.
>
> **Le CEMS livre déjà l'état caché que le LSTM devrait réapprendre.**
>
> Trois observations indépendantes le confirment : la PACF montre une
> autocorrélation épuisée en deux à trois jours ; un ARIMA sans variable
> exogène est inutilisable, avec une corrélation **négative** de −0,118 ; et
> les trois premières features du modèle physique décrivent le combustible,
> pas le passé.
>
> Ce problème n'est pas une prévision de série temporelle. C'est une
> **classification spatio-temporelle d'événement rare**, sur 34 734 séries
> parallèles pilotées par un exogène déjà résumé par la physique du domaine. »

> ⚠️ **Réserve à donner spontanément** : « Une asymétrie subsiste — le LSTM ne
> reçoit pas `danger_effis`, qui pèse 13,7 % dans le modèle physique. Les
> 23,6 % sont donc un **majorant**. C'est la première chose que je referais. »
> Dire cela avant qu'on vous le demande vaut beaucoup.

### Diapo 17 — v3 contre C ⭐

**À dire** (2 min) :

> « Voici la décision la plus contre-intuitive du projet.
>
> Sur le test, XGBoost v3 fait **×93,8**. Le modèle physique, **×63,7**. v3
> est nettement meilleur. Et pourtant c'est le modèle physique qui tourne dans
> l'application.
>
> Trois raisons, dont **aucune n'est visible dans une métrique
> d'entraînement**.
>
> Un : la donnée n'existe pas en temps réel. v3 tire 29 % de son importance de
> l'historique des feux, or la BDIFF ne publie pas l'année en cours — les feux
> de 2026 sortiront au printemps 2027. Pour une prédiction faite aujourd'hui,
> `feux_commune_7j` vaudrait le décompte d'une semaine de décembre 2025. Pas
> imprécis : **faux**.
>
> Deux : en territoire inconnu, cette variable vaut zéro partout. Et le modèle
> lit ce zéro comme "ça n'a jamais brûlé, donc ça ne brûlera pas" —
> précisément là où le risque nouveau apparaît.
>
> Trois : pour 2050, elle est impossible par construction. On ne connaîtra
> jamais les feux de 2049.
>
> **Le choix se fait sur la disponibilité de la donnée, pas sur la
> performance.** Et c'est ce qui autorise à mesurer les deux modèles sur le
> test sans corrompre le protocole : ils répondent à deux situations
> différentes, pas à la même question. »

### Diapo 18 — La validation croisée spatiale

**À dire** (1 min 30) :

> « Ce n'est pas qu'un raisonnement, je l'ai mesuré. Protocole : je retire une
> région entière de l'entraînement, j'entraîne, et je teste **sur la région
> exclue**. Neuf fois, une par région. C'est la simulation d'un territoire
> jamais vu.
>
> Le modèle physique gagne dans les **neuf régions sur neuf**, sans exception.
> +8,2 % en moyenne pondérée, et jusqu'à **+137 % dans le Grand Est** — c'est-
> à-dire là où l'historique est le plus pauvre.
>
> Là où le passé manque, s'y fier est un **handicap**. C'est l'argument
> décisif pour 2050 : le climat va déplacer le risque vers des communes qui
> n'ont pas de passé. »

### Diapo 19 ⏩ — La calibration

**À dire** (1 min) :

> « Le score brut est **145 fois trop grand** : c'est le sous-échantillonnage
> qui remonte à la surface.
>
> Platt le corrige à ×1,13 sans rien coûter en PR-AUC — le classement est
> intact, seule l'échelle bouge. L'isotonique calibre aussi bien mais écrase
> le score sur 136 valeurs distinctes au lieu de 9 millions : on perd du
> pouvoir de discrimination pour rien.
>
> Dans l'application, j'affiche finalement un **rang**, pas une probabilité :
> le calibrateur disponible a été ajusté sur un autre modèle et une autre
> période, il serait faux d'un facteur 2. J'ai préféré ne pas afficher une
> probabilité plutôt que d'en afficher une fausse. »

### Diapo 20 — L'évaluation test

**À dire** (1 min 30) :

> « J'ai ouvert le test une seule fois, après gel complet.
>
> Le lift varie du simple au double d'une année à l'autre — et ce n'est pas du
> bruit, **il suit la rareté**. 2024 est l'année la plus calme, avec 1 297
> feux, et donne le **meilleur** lift : ×151. 2023, la plus active, le plus
> faible : ×76.
>
> C'est contre-intuitif et instructif : **une année calme concentre les feux
> dans les endroits les plus prévisibles**. Quand tout brûle, y compris là où
> ce n'est pas censé arriver, le modèle est pris en défaut. »

> 💡 Ce point montre que vous avez regardé la variabilité, pas seulement la
> moyenne. Les jurys y sont sensibles.

---

## 4 · Comprendre le modèle (diapos 21 à 23)

### Diapo 21 — Section

### Diapo 22 — SHAP ⭐

**À dire** (2 min) :

> « "Qu'est-ce qui fait partir un feu ?" La question paraît simple. Elle a
> **trois réponses différentes**, et chacune est juste — mais pas à la même
> question.
>
> À gauche, l'**importance par gain** : combien chaque variable a réduit la
> perte pendant l'entraînement. C'est ce que renvoie XGBoost par défaut.
>
> Au milieu, **SHAP sur un échantillon aléatoire** : combien elle déplace le
> score sur le territoire tel qu'il est — 99,97 % de communes-jours sans feu.
>
> À droite, **SHAP sur le sommet du classement** : combien elle déplace le
> score là où le modèle s'engage.
>
> Deux désaccords méritent d'être expliqués.
>
> `danger_effis` est **2ᵉ par gain et 30ᵉ par SHAP**. Ce n'est pas une erreur :
> c'est une discrétisation du FWI en six classes. XGBoost adore ces seuils
> nets pour découper, d'où un gain élevé — mais l'information est déjà dans le
> FWI continu, et SHAP lui en attribue le crédit. **L'importance par gain ne
> sait pas gérer la redondance.**
>
> `part_maquis` est **1ᵉʳ par gain, 10ᵉ sur l'échantillon aléatoire, 2ᵉ au
> sommet**. Là encore tout est cohérent : le maquis ne change rien sur une
> commune-jour moyenne — il n'y en a pas — mais il devient déterminant là où
> le modèle voit du risque.
>
> La morale : **ne jamais citer "la feature numéro un" sans dire de quelle
> mesure il s'agit et sur quelle population.** »

**Si on demande pourquoi SHAP plutôt qu'autre chose** : sur un modèle à base
d'arbres, TreeSHAP est **exact** — ce n'est pas une approximation par
échantillonnage comme KernelSHAP. Je l'ai vérifié : la somme des contributions
plus la valeur de base redonne le logit du score à 1e-6 près.

⚠️ **Piège méthodologique à mentionner** : sélectionner sur le score introduit
un **biais de collision**. J'en ai fait les frais — une première analyse
d'interaction, menée sur le seul échantillon du sommet, donnait un signe
**opposé** à la réalité. C'est pourquoi les deux panneaux SHAP sont affichés
côte à côte, et jamais l'un sans l'autre.

### Diapo 23 — DiCE

**À dire** (1 min 30) :

> « SHAP et LIME répondent à *pourquoi ce score*. DiCE répond à *qu'aurait-il
> fallu changer* — et c'est la seule question dont la réponse soit
> actionnable.
>
> Un exemple mesuré : Bormes-les-Mimosas, 99,9ᵉ percentile du 12 août 2024.
> J'autorise à modifier la végétation seule — maquis, forêt, part combustible,
> part agricole. **Résultat : aucun contrefactuel.** Rien, sur ces leviers, ne
> fait sortir la commune du décile à risque. Son exposition tient à sa
> position, à son relief, à sa superficie. Le risque est **structurel** — et
> c'est un résultat, pas un échec.
>
> Un détail d'implémentation a tout changé. DiCE cherche par défaut à faire
> passer la probabilité sous 0,5. Or le score n'est pas calibré : 0,5
> correspond à un risque astronomique, et l'outil ne renvoyait jamais rien.
> J'ai recentré la frontière sur le **décile**, par une transformation
> strictement croissante qui laisse le classement intact. La question devient
> "que faudrait-il pour sortir des 10 % les plus à risque" — celle qui a un
> sens opérationnel.
>
> Enfin, **un contrefactuel n'est pas une recommandation**. Rien ne garantit
> qu'il soit réalisable — on ne convertit pas 40 % de maquis en terres
> agricoles — ni que le lien soit causal : le modèle a appris des
> corrélations, pas des mécanismes. »

**Si on demande pourquoi LIME** : sur un modèle d'arbres, LIME approxime par
une régression linéaire locale ce que TreeSHAP calcule exactement. Je le montre
pour la comparaison, et parce que savoir **pourquoi on ne s'en sert pas** vaut
mieux que l'ignorer. Sur un modèle qu'on ne peut pas ouvrir — une API, un
réseau profond — LIME redeviendrait le bon outil.

---

## 5 · Le climat (diapos 24 à 26)

### Diapo 24 — Section

### Diapo 25 — La tendance ⭐

**C'est la diapositive où l'honnêteté compte le plus.**

**À dire** (2 min) :

> « Le danger météo monte de façon incontestable. Sur 53 ans de mesures, le
> FWI moyen annuel gagne **+58 %**, avec un p de 4×10⁻⁵. La moyenne
> juin-septembre, **+62 %** — c'est l'été qui se réchauffe et s'assèche le
> plus. Les jours de danger élevé, **+197 %**.
>
> **Et pourtant le nombre de feux ne monte pas** : +3 % sur 19 ans, avec un p
> de 0,89. Rigoureusement rien.
>
> Il serait malhonnête de ne montrer que le premier graphique. Trois lectures
> cohabitent.
>
> Un : **la puissance statistique**. Les feux ne sont observés que depuis
> 2006 — dix-neuf points annuels très bruités ne peuvent pas détecter une
> tendance modérée. L'absence de preuve n'est pas une preuve d'absence.
>
> Deux : **la prévention fonctionne**. Le nombre de départs dépend autant des
> moyens de lutte et du débroussaillement que du climat. Un aléa qui monte à
> sinistralité constante est le résultat **attendu** d'une politique efficace.
>
> Trois : **ce que je projette, c'est l'aléa, pas le bilan.**
>
> Donc la phrase juste est : *les conditions favorables aux feux augmentent
> très significativement, et le nombre de départs reste stable, ce qui est
> cohérent avec une prévention qui absorbe pour l'instant la hausse de
> l'aléa.* »

> ⚠️ **Cette diapositive contient une leçon de méthode** : j'ai d'abord
> affiché +45 % dans l'application, chiffre écrit en dur dans une chaîne de
> caractères. Il était faux. Depuis, toutes les tendances sont **calculées et
> exportées** ; l'application les lit. Si on vous demande d'où vient +58 %,
> répondez : « d'une régression linéaire sur les moyennes annuelles de
> 21,9 millions de lignes, recalculée à chaque export ».

### Diapo 26 — Les projections

**À dire** (1 min 30) :

> « Trois scénarios du GIEC — RCP 2.6, 4.5 et 8.5 — appliqués à l'aléa météo.
>
> Ce que je projette : le **FWI**, seule quantité qui montre un signal et que
> les modèles climatiques savent fournir.
>
> Ce que je ne projette pas : le nombre de feux. La végétation, la prévention
> et les pratiques agricoles sont supposées constantes. C'est une hypothèse,
> et elle est fausse — mais elle est explicite.
>
> Et il faut être précis sur ce que "le 2 août 2050" veut dire. Ce n'est
> **pas** une prévision météo : personne ne connaît le temps qu'il fera ce
> jour-là. C'est un 2 août **ordinaire** sous le climat de 2050. La forme de
> la saison vient des observations 2006-2019 ; seul son niveau est décalé par
> le réchauffement projeté. L'application affiche cet avertissement à chaque
> date postérieure à 2025. »

---

## 6 · Rigueur (diapos 27 à 31)

### Diapo 27 — Section

> « Je voudrais terminer par les erreurs que j'ai commises. »

### Diapo 28 — Le bug d'alignement ⭐

**À dire** (2 min) :

> « Le premier verdict que j'avais pour le LSTM annonçait **−97 %**. Le vrai
> est −52 %. L'écart ne venait pas du modèle, mais de la façon dont je
> comparais deux fichiers.
>
> La requête qui assemble la matrice n'a pas d'`ORDER BY`. L'ordre dans lequel
> PostgreSQL renvoie les 38 millions de lignes dépend du plan d'exécution et
> des workers parallèles — il **change d'une exécution à l'autre**. Mes
> fichiers de prédictions ne portaient que le score et la cible : les comparer
> revenait à les aligner **par position**.
>
> Deux fichiers issus de deux exécutions ont la même taille, le même nombre de
> feux, et un ordre différent. **Rien ne signale l'erreur.**
>
> Je l'ai reproduit volontairement — c'est le graphique à l'écran. On permute
> les lignes, la PR-AUC tombe de 0,0085 à 0,0002 : exactement la ligne du
> hasard, avec les mêmes valeurs dans le fichier.
>
> La parade : tout fichier de prédictions porte désormais ses clés, une
> fonction d'alignement vérifie qu'elles correspondent, et un test **refuse**
> un fichier sans clés. »

> 💡 Ne présentez jamais cela comme un aveu. C'est une démonstration de
> maturité : vous avez trouvé un bug que rien ne signalait, vous en avez
> mesuré l'effet, et vous avez posé un garde-fou.

### Diapo 29 — Les autres erreurs

**À dire** (1 min 30) — *ne pas lire tout le tableau, en choisir deux* :

> « Trois autres erreurs sérieuses, trouvées et corrigées.
>
> J'avais laissé `ha` et la cible dans les features du modèle de surface
> brûlée : R² de 0,994 et ROC-AUC de 1,0000. Un score trop beau pour être
> vrai — c'est ce qui m'a alerté.
>
> J'ai mené une analyse d'interaction sur le seul échantillon du sommet : le
> signe de l'interaction s'inversait. C'est un **biais de collision** —
> sélectionner sur le score conditionne le résultat.
>
> Et j'avais un dentelé de période quatre ans dans les séries annuelles : le
> 15 août tombe au jour 227 ou 228 selon les années bissextiles, et la
> climatologie non lissée sautait de 50 % d'un jour à l'autre.
>
> Le point commun : sur un événement à 0,02 %, **une erreur ne se manifeste
> jamais par une exception. Elle se manifeste par un chiffre plausible.** Les
> seules défenses sont les invariants explicites et les assertions qui
> échouent bruyamment. Cinquante tests tournent en intégration continue à
> chaque commit. »

### Diapo 30 — L'application

**Démonstration en direct** (2 à 3 min). Ordre conseillé :

1. **La carte** — choisir le 12 août 2024, montrer l'arc méditerranéen.
2. **Basculer sur une date future** — 2050, RCP 8.5 — et lire l'avertissement
   à voix haute.
3. **Le mode rétrospectif** — montrer qu'il compare les deux modèles, puis
   **essayer une date de 2021** pour montrer qu'il **refuse** et explique
   pourquoi. C'est le moment le plus fort de la démo.
4. **La page Pourquoi** — chercher une commune, montrer la décomposition SHAP.

> 💡 Le refus sur 2021 est ce qui distingue une démo d'étudiant d'une démo
> d'ingénieur. Ne le sautez pas.

### Diapo 31 — Les limites

**À dire** (1 min 30) :

> « Je termine par ce que le projet ne fait pas.
>
> **La surface brûlée n'est pas prédictible** : R² de 0,14, moins bon que
> d'annoncer toujours la médiane. Elle dépend de ce qui se passe *après* le
> départ : vent, délai d'intervention, relief. En revanche, "sera-ce un grand
> feu de plus de 5 hectares ?" se prédit à 0,77 de ROC-AUC.
>
> **Une commune-jour n'est pas un incendie** : un feu traversant cinq communes
> compte cinq fois.
>
> **31 communes partagent une maille météo de 28 kilomètres** : elles ont le
> même FWI le même jour. Le FWI porte le *quand*, la végétation porte le *où*.
> Conséquence statistique, que je dois signaler : les intervalles naïfs sur
> les coefficients météo seraient trop étroits.
>
> Et **le LSTM n'a pas reçu `danger_effis`**. Les 23,6 % sont un majorant tant
> que cette asymétrie n'est pas levée. C'est la première chose que je
> referais. »

---

## Annexe A · Fiche de chaque modèle

> C'est le tableau à connaître par cœur. Un jury demande souvent : *« pour ce
> modèle-là, quel était l'objectif, la cible, et qu'avez-vous mesuré ? »*

### Baselines (aucun apprentissage)

| | |
|---|---|
| **Objectif** | établir la barre à battre |
| **Cible** | `y` binaire — au moins un départ de feu ce jour-là dans cette commune |
| **Prédicteurs** | ① hasard ② taux historique de la commune ③ classe de danger EFFIS ④ le produit des deux |
| **Mesuré sur** | validation 2020-2022, 38 M lignes, non échantillonnée |
| **Résultat** | ×1 · ×19,4 · ×5,1 · ×42,1 |
| **Conclusion** | l'historique spatial est déjà un prédicteur puissant ; tout modèle doit battre ×42 |

### Modèle v1 — RandomForest et XGBoost, 43 features

| | |
|---|---|
| **Objectif** | première mesure honnête, et diagnostic |
| **Cible** | `y` binaire |
| **Features** | météo du jour et décalages, végétation CORINE, territoire, calendrier, historique des feux |
| **Entraîné sur** | train 2006-2019 sous-échantillonné 1:10 → 368 826 lignes |
| **Mesuré** | PR-AUC, lift, courbe de fiabilité, courbe d'apprentissage |
| **Résultat** | RandomForest **0,0150 · ×62,0** · XGBoost **0,0166 · ×68,8** |
| **Conclusion** | XGBoost devance nettement RandomForest. Surtout : **54,6 % de l'importance vient de l'historique de la commune** — le modèle est à moitié un modèle de persistance. C'est ce diagnostic qui déclenche le v3. |

### Modèle v2 — v1 optimisé par Optuna

| | |
|---|---|
| **Objectif** | savoir ce que le réglage rapporte, séparément des features |
| **Mesuré** | 60 essais Optuna, échantillonnage TPE, sur la validation |
| **Résultat** | PR-AUC **0,0175 · ×72,4** — soit **+5,2 %** sur le v1 |
| **Conclusion** | le réglage rapporte réellement, et davantage que le clustering qui suivra |

### Modèle v3 — clustering territorial et lissage bayésien, 52 features

| | |
|---|---|
| **Objectif** | donner un risque aux communes sans historique |
| **Méthode** | 30 clusters par k-means sur profil physique (**jamais sur `y`**), puis lissage bayésien à deux niveaux : commune → cluster → national |
| **Garde-fou** | les taux sont ajustés en *leave-one-year-out*, avec **remise à exposition constante** — sans quoi l'année exclue biaise le dénominateur |
| **Résultat** | PR-AUC **0,0177 · ×73,4** — soit **+1,3 %** sur le v2 |
| **Conclusion** | gain réel mais **modeste**, et à peine significatif au bootstrap apparié. Le dire est un résultat : le clustering territorial rapporte quatre fois moins que le simple réglage des hyperparamètres. |

### DART — v3 avec abandon d'arbres

| | |
|---|---|
| **Objectif** | tester une régularisation différente (dropout sur les arbres) |
| **Résultat** | PR-AUC 0,0174 · **−1,8 %, IC [−5,0 ; +0,8]** |
| **Conclusion** | **indiscernable de v3.** L'intervalle traverse zéro. |

### MLP — réseau dense, 52 features

| | |
|---|---|
| **Objectif** | satisfaire l'exigence « réseau de neurones » de l'énoncé, et tester une famille de modèle différente |
| **Architecture** | 3 couches denses, BatchNorm, dropout, early stopping ; entraîné sur GPU |
| **Résultat** | PR-AUC 0,0173 · **−1,9 %, IC [−7,9 ; +5,2]** |
| **Conclusion** | **indiscernable de v3.** Un gradient boosting et un réseau dense font le même travail sur des features tabulaires. |

### Modèle C — physique pure, 41 features ⭐ *le déployé*

| | |
|---|---|
| **Objectif** | être **déployable** : ne dépendre d'aucune donnée indisponible à l'avance |
| **Retiré** | les 7 features d'historique de feux, `lat`, `lon`, et les 3 taux dérivés de `y` |
| **Pourquoi `lat`/`lon`** | elles encodent « le Sud brûle ». Vrai aujourd'hui, et exactement le préjugé à ne pas transporter en 2050 |
| **Résultat validation** | PR-AUC 0,0112 · ×46,4 · **−36,8 % vs v3** |
| **Résultat test** | PR-AUC 0,0106 · **×63,7** |
| **Validation croisée spatiale** | **gagne 9 régions sur 9**, +8,2 % pondéré, jusqu'à +137 % |
| **Conclusion** | moins bon là où l'historique existe, **meilleur là où il manque** — c'est-à-dire en temps réel et en 2050 |

### LSTM — séquence de 30 jours

| | |
|---|---|
| **Objectif** | tester si une représentation temporelle **apprise** bat les décalages faits à la main |
| **Entrée** | 30 jours × 8 indices CEMS = 240 valeurs, plus 30 descripteurs de territoire et calendrier |
| **N'a PAS** | l'historique des feux, ni `danger_effis` |
| **Réglage** | 25 essais Optuna, 7 hyperparamètres, arrêt précoce (époque 21) |
| **Résultat** | PR-AUC 0,0085 · ×35,4 |
| **Comparaison loyale** | contre le modèle C : **−23,6 %, IC [−33,5 ; −17,3]** — significatif |
| **Conclusion** | les indices CEMS **sont déjà** des états récursifs ; le LSTM devrait réapprendre ce qu'on lui donne en entrée |

### Ensemble — moyenne de rangs v3 + MLP

| | |
|---|---|
| **Résultat** | PR-AUC 0,0180 · ×74,9 · **+2,0 % vs v3** |
| **Conclusion** | gain réel mais faible, au prix de deux modèles en production. Non retenu. |

### Modèles secondaires

| Modèle | Cible | Résultat | Conclusion |
|---|---|---|---|
| Régression de surface | hectares brûlés | **R² 0,14** | **échec assumé** — pire que la médiane |
| Classification « grand feu » | surface > 5 ha | ROC-AUC 0,77 | utilisable |
| SARIMAX national | nombre de communes-jours en feu | MAE 4,03 · **−21,5 % vs référence saisonnière** | la météo porte la prévisibilité |

---

## Annexe B · Questions attendues, et réponses

### Sur la méthode

**« Pourquoi ne pas avoir utilisé l'accuracy ? »**
> À 0,019 % de positifs, répondre toujours « non » donne 99,98 %. L'accuracy
> ne distingue pas un bon modèle d'un modèle vide.

**« Pourquoi pas de validation croisée classique ? »**
> Parce que le problème est temporel. Un *k-fold* aléatoire mettrait des jours
> voisins de part et d'autre du découpage : le modèle « prédirait » un feu
> qu'il a vu à 20 km et un jour d'écart. J'ai en revanche fait une validation
> croisée **spatiale**, par région, qui répond à une vraie question : que vaut
> le modèle sur un territoire inconnu ?

**« Comment savez-vous qu'il n'y a pas de fuite ? »**
> Trois règles explicites, et des tests qui les vérifient : la base ne stocke
> que des faits, les statistiques dérivées de `y` s'ajustent sur le train
> seul, et une feature datée ne regarde que le passé strict — borne haute la
> veille. Un test vérifie aussi que la PR-AUC reste sous 0,80 : au-delà, sur
> un événement à 0,019 %, il faut chercher la fuite.

**« Votre modèle est-il calibré ? »**
> Non, et c'est délibéré. Le score brut est 145 fois trop grand à cause du
> sous-échantillonnage. Platt corrige à ×1,13, mais ce calibrateur a été
> ajusté sur un autre modèle et une autre période — il serait faux d'un
> facteur 2. J'affiche donc un **rang**, qui reste juste.

### Sur les modèles

**« Pourquoi XGBoost et pas un réseau de neurones ? »**
> J'ai fait les deux. Le MLP fait 0,0173 contre 0,0177 : **indiscernable**,
> l'intervalle traverse zéro. Sur des features tabulaires, les deux familles
> font le même travail. J'ai gardé le gradient boosting parce qu'il s'entraîne
> en 30 secondes contre plusieurs minutes, et qu'il donne des explications
> SHAP exactes.

**« Votre LSTM était-il assez entraîné ? »**
> 25 essais Optuna sur sept hyperparamètres, avec arrêt précoce. Et sa
> défaite est explicable *a priori* : les indices DC, DMC et BUI sont déjà des
> moyennes exponentielles de la météo passée. Je lui demandais de réapprendre
> ce que je lui donnais en entrée.

**« Pourquoi ne déployez-vous pas le meilleur modèle ? »**
> Parce que le meilleur a besoin d'une donnée qui n'existe pas au moment de
> prédire. La BDIFF ne publie pas l'année en cours. Le choix se fait sur la
> disponibilité de la donnée, pas sur le score — et la validation croisée
> spatiale confirme que le modèle physique gagne dès qu'on sort du territoire
> connu.

### Sur les données

**« 0,68 % des feux écartés, n'est-ce pas beaucoup ? »**
> C'est 965 feux sur 142 787, et l'alternative était pire. Rapprocher par le
> nom donnait des faux positifs entre départements. Un feu attribué à la
> mauvaise commune corrompt la cible, le voisinage et les features CORINE de
> cette commune. J'ai préféré compter ce que je perds.

**« Vos données couvrent-elles tout le territoire ? »**
> Métropole seulement. L'outre-mer est hors de la couverture météo européenne
> et absent de CORINE : 1 378 feux exclus, comptés et documentés. Et la BDIFF
> n'est pas homogène avant 2006 — d'où la période retenue.

### Sur les projections

**« Prédisez-vous les feux de 2050 ? »**
> Non. Je projette l'**aléa météo** sous trois scénarios du GIEC. Le nombre de
> feux dépend aussi de la prévention et des pratiques agricoles, que je
> suppose constantes — hypothèse explicite et certainement fausse.

**« Pourquoi le nombre de feux n'augmente-t-il pas ? »**
> Trois explications compatibles : la puissance statistique est faible sur 19
> points annuels ; la prévention absorbe la hausse de l'aléa ; et l'aléa n'est
> pas le bilan. Je ne tranche pas — les données ne le permettent pas.

### Questions pièges

**« Votre modèle est-il utile ? »**
> À ×63,7, surveiller 1 % du territoire capture environ 39 % des départs. La
> comparaison honnête n'est pas au hasard mais au danger EFFIS officiel, qui
> vaut ×5 seul. Le modèle apporte donc un facteur 12 sur la pratique actuelle.

**« Qu'est-ce qui vous a le plus surpris ? »**
> Que 2024, l'année la plus calme, donne le **meilleur** lift. Une année calme
> concentre les feux dans les endroits prévisibles ; quand tout brûle, le
> modèle est pris en défaut.

**« Qu'auriez-vous fait avec plus de temps ? »**
> Trois choses, dans l'ordre : donner `danger_effis` au LSTM pour lever
> l'asymétrie ; remplir `vacances_scolaires`, aujourd'hui à `false` partout ;
> et tester un rattachement des feux nocturnes à la veille, puisque le FWI est
> défini à midi solaire et qu'un feu déclaré à 3 h du matin est piloté par les
> conditions de la veille.

---

## Annexe C · Glossaire

| Terme | Définition courte |
|---|---|
| **FWI** | *Fire Weather Index* — indice canadien de danger météo, standard européen via EFFIS |
| **DC / DMC / BUI** | composantes du FWI : moyennes exponentielles de la météo passée, constantes de temps 52 j / 15 j / dérivée |
| **PR-AUC** | aire sous la courbe précision-rappel ; vaut le taux de base au hasard |
| **lift** | PR-AUC ÷ taux de base — « combien de fois mieux que le hasard » |
| **bootstrap apparié** | rééchantillonnage où les deux modèles voient les **mêmes** lignes à chaque réplique, ce qui annule la variance commune |
| **biais de collision** | biais introduit en conditionnant sur une variable influencée par les deux qu'on étudie |
| **TreeSHAP** | calcul **exact** des valeurs de Shapley sur un modèle à base d'arbres |
| **contrefactuel** | point proche que le modèle classerait différemment |
| **RCP** | *Representative Concentration Pathway* — trajectoire d'émissions du GIEC ; le chiffre est le forçage radiatif en 2100, en W/m² |
| **lissage bayésien** | estimation d'un taux rare en le rappelant vers celui d'un groupe plus large |
| **ADF** | test de Dickey-Fuller augmenté ; H₀ = série **non** stationnaire |
| **PACF** | autocorrélation partielle — donne l'ordre AR d'un modèle |

---

## Rappel des commandes

```bash
python -m tvfed.comparer      # bootstrap apparié, produit comparaison_appariee.csv
python -m tvfed.explications  # SHAP sur le modèle C
python -m tvfed.export_app    # met à jour app/donnees
python -m tvfed.diaporama     # régénère soutenance.pptx
pytest tests/ -q              # 50 tests
```
