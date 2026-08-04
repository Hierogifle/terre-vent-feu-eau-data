# Script de soutenance · Terre, Vent, Feu, Eau, Data

Support : `presentation/soutenance.pptx`, 37 diapositives.
Durée visée : 25 minutes, plus 10 à 15 minutes de questions.

Ce document donne, pour chaque diapositive, ce qu'il y a à dire, ce qu'il faut
éviter de dire, et les détails à garder en réserve pour les questions. Le
texte entre guillemets est un appui, pas une récitation.

Le diaporama suit un plan scientifique : introduction, question de recherche,
méthodologie, résultats, interprétation. Les résultats n'arrivent qu'une fois
la question et le protocole posés, ce qui permet de répondre à « comment
savez-vous que c'est vrai » avant qu'on ne le demande.

Les chiffres de ce document et ceux du diaporama viennent des mêmes fichiers
`data/processed/*.csv`. Après un réentraînement, `python -m tvfed.diaporama`
remet le support à jour tout seul.

---

## Table des matières

- [Avant de commencer](#avant-de-commencer)
- [01 · Introduction](#01--introduction-diapos-2-à-4)
- [02 · Question de recherche](#02--question-de-recherche-diapos-5-à-7)
- [03 · Méthodologie](#03--méthodologie-diapos-8-à-15)
- [04 · Résultats](#04--résultats-diapos-16-à-29)
- [05 · Interprétation](#05--interprétation-diapos-30-à-37)
- [Annexe A · Fiche de chaque modèle](#annexe-a--fiche-de-chaque-modèle)
- [Annexe B · Questions attendues](#annexe-b--questions-attendues)
- [Annexe C · Glossaire](#annexe-c--glossaire)
- [Annexe D · Commandes](#annexe-d--commandes)

---

## Avant de commencer

### La phrase qui résume le travail

> « J'ai construit un modèle qui estime, pour chacune des 34 734 communes de
> France métropolitaine et pour chaque jour, la probabilité qu'un départ de
> feu y soit déclaré. Il tourne sur 253 millions de lignes, il fait 64 fois
> mieux que le hasard sur une période qu'il n'a jamais vue, et le modèle que
> j'ai déployé n'est délibérément pas le plus performant que j'aie obtenu. »

La dernière proposition est celle qui ouvre la discussion. Laissez-la faire.

### Les trois points à ne pas manquer

1. La comparaison de modèles avec intervalles de confiance, diapo 20. Trois
   des cinq modèles comparés sont indiscernables entre eux, et le dire est un
   résultat.
2. Le choix de déployer le modèle le moins performant, diapo 24. La décision
   se prend sur la disponibilité de la donnée, pas sur la métrique.
3. Le LSTM qui perd, avec une explication physique, diapos 21 à 23. Trois
   familles de méthodes concordent.

### Ce qu'il faut éviter de dire

| Ne pas dire | Dire à la place |
|---|---|
| « les feux augmentent » | « les conditions favorables aux feux augmentent ; le nombre de départs reste stable sur nos 20 années » |
| « le modèle prédit les feux de 2050 » | « le modèle applique le climat de 2050 à un jour ordinaire ; ce n'est pas une prévision météo » |
| « 99 % de précision » | « le lift : 64 fois mieux que le hasard » |
| « XGBoost bat le MLP » | « leurs intervalles se recouvrent, je ne peux pas les départager » |
| « mon modèle est fiable à X % » | « il classe bien, il ne quantifie pas ; c'est pourquoi l'application affiche un rang » |

---

## 01 · Introduction (diapos 2 à 4)

### Diapo 2 · Section

Transition, deux secondes. Ne rien ajouter au titre.

### Diapo 3 · Le phénomène

**À dire** (1 min 30) :

> « Voici les 52 809 feux déclarés sur le périmètre, cumulés par jour de
> l'année sur 53 ans. On voit le pic d'août, celui qu'on attend : l'arc
> méditerranéen, la Corse, un été sec.
>
> Le second pic, en mars, est moins connu. Il ne concerne ni les mêmes
> régions ni la même végétation : le Sud-Ouest et le Massif central, avec des
> landes et des herbacées sèches avant la reprise, souvent des écobuages qui
> échappent.
>
> Conséquence directe pour la modélisation : un modèle qui n'apprendrait que
> l'été raterait environ un feu sur cinq. La saisonnalité doit entrer comme
> variable, pas comme filtre. »

**Si on demande quels départements** : Cantal, 52 % de ses feux hors saison
estivale ; Dordogne, 44 % ; Lozère, 41 %. Ces parts sont recalculées dans
l'application, page *Les données*, à partir des données et non citées de
mémoire.

**Précaution** : ce sont des parts, pas des volumes. En valeur absolue, la
Haute-Corse, la Gironde et l'Hérault dominent, y compris hors été. Confondre
les deux lectures est une erreur facile à commettre, et je l'ai commise une
fois en écrivant l'application.

### Diapo 4 · Ce qui existe déjà, et ce qui manque

**À dire** (1 min 30) :

> « Le danger météo est déjà un service public : EFFIS au niveau européen,
> Météo-France en France. Il ne s'agit pas de le remplacer, mais de partir de
> là où il s'arrête.
>
> Ce système mesure l'inflammabilité de l'air et du combustible, sur une
> maille de 0,25°, soit environ 28 km. À l'intérieur d'une même maille on
> trouve en moyenne 31 communes, qui reçoivent aujourd'hui le même indice de
> danger, qu'elles soient couvertes de maquis ou entièrement bétonnées.
>
> C'est cet écart que le projet cherche à combler. La météo dit quand, elle
> ne dit pas où. »

**Ne pas dire** que le FWI serait insuffisant ou dépassé. Il est l'entrée
principale du modèle, et les baselines montrent qu'il porte un signal réel.
L'apport est une résolution spatiale, pas une correction.

---

## 02 · Question de recherche (diapos 5 à 7)

### Diapo 5 · Section

Transition.

### Diapo 6 · La question

**À dire** (2 min) :

> « La question est la suivante : peut-on estimer, à l'échelle du couple
> commune × jour, la probabilité qu'un départ de feu soit déclaré, à partir
> des seules données publiques disponibles en temps réel ?
>
> La fin de la phrase est une contrainte que je me suis imposée, et elle a
> décidé du modèle déployé. J'y reviens en détail.
>
> J'ai décliné cette question en quatre hypothèses, chacune formulée pour
> pouvoir être réfutée. »

Puis lire le tableau, en insistant sur H3 :

> « H3 dit qu'une architecture séquentielle n'apporte rien de plus que des
> indices physiques agrégés. Je l'ai formulée dans ce sens, mais je l'ai
> testée dans les deux : si le LSTM avait gagné, je l'aurais déployé. Le
> résultat n'était pas décidé à l'avance. »

**Point de méthode à revendiquer** : chaque hypothèse est associée à une
mesure précise, décidée avant de regarder le résultat. C'est ce qui distingue
une expérience d'une exploration.

### Diapo 7 · Ce qui rend la question difficile

**À dire** (1 min 30) :

> « Un départ de feu concerne 0,019 % des couples commune × jour. Trois
> conséquences.
>
> Répondre systématiquement non donne 99,98 % de justesse, et ne sert à rien.
> L'exactitude est inutilisable ici, et je n'en montrerai aucune.
>
> Deuxièmement, une fuite de données ne provoque pas d'erreur : elle produit
> d'excellentes métriques et un modèle sans valeur. Rien, dans les scores, ne
> la signale. C'est le risque principal de ce type de problème.
>
> Troisièmement, ce déséquilibre commande tout le reste : la métrique,
> l'échantillonnage, la calibration, et jusqu'à la façon de comparer deux
> modèles entre eux. »

Cette diapositive justifie par avance toute la méthodologie. La poser
correctement évite d'avoir à se justifier cinq fois ensuite.

---

## 03 · Méthodologie (diapos 8 à 15)

### Diapo 8 · Section

Transition.

### Diapo 9 · Quatre sources publiques

**À dire** (1 min 30) :

> « Quatre sources publiques : les indices de danger du CEMS Copernicus, les
> feux déclarés de la BDIFF, l'occupation du sol CORINE, et le référentiel
> INSEE des communes.
>
> Le point délicat n'est pas le volume, ce sont les fusions de communes. 965
> feux portent un code INSEE qui n'existe plus. J'ai d'abord essayé de les
> rapprocher par le nom, et cela produisait des faux positifs : Chirac en
> Lozère renvoyait vers la Charente.
>
> J'ai donc téléchargé le fichier officiel des mouvements de communes de
> l'INSEE. Les 30 cas restants sont écartés et comptés, jamais devinés. »

**Pourquoi ce choix, si on demande** : une heuristique par le nom aurait
« résolu » tous les cas, et corrompu au passage la cible, le voisinage et les
variables CORINE de communes qui n'avaient rien à voir. Perdre 0,68 % des
feux coûte moins cher que d'en inventer.

### Diapo 10 · La table centrale

**À dire** (1 min 30) :

> « La table centrale a une ligne par commune et par jour, qu'il y ait eu un
> feu ou non. 253 millions de lignes pour 49 130 feux.
>
> On peut se demander pourquoi ne pas garder la seule liste des feux. La
> raison est technique et précise : une série creuse rend les fenêtres
> glissantes silencieusement fausses. « Feux des 30 jours précédents » se
> calcule en remontant 30 lignes ; si les jours sans feu sont absents, on
> remonte en réalité plusieurs années, et rien ne lève d'erreur. »

**En réserve** : PostgreSQL 16 avec PostGIS, 20 partitions annuelles. Le
partitionnement par année suit le découpage des splits, ce qui rend les
requêtes d'évaluation peu coûteuses.

### Diapo 11 · La barrière du split

**À dire** (2 min) :

> « Le découpage est temporel, jamais aléatoire. Train 2006-2019, validation
> 2020-2022, test 2023-2025.
>
> Un découpage aléatoire mettrait le 14 juillet 2019 dans le train et le 15
> dans le test. Le modèle prédirait un feu qu'il a déjà vu, à 20 km et un
> jour d'écart. La métrique serait excellente et le modèle sans valeur.
>
> J'ai eu besoin d'une règle pour trancher les cas douteux, parce qu'ils sont
> nombreux. La voici : une variable datée peut regarder tout le passé, y
> compris celui de sa propre période d'évaluation. Une statistique non datée
> ne peut regarder que le train. »

**L'exemple qui rend la règle claire** : « feux des 30 jours précédents » au
3 août 2023 lit juillet 2023, et ce n'est pas une fuite, parce que le 3 août
à 8 h du matin on connaît juillet. En revanche, « taux moyen de la commune
sur toute la période » lit le futur, et c'en est une.

**Sur le test** : ouvert une seule fois, après gel complet du modèle, des
variables et de la calibration. Si on demande pourquoi une telle rigidité :
le risque d'un jeu de test est cumulatif, chaque coup d'œil en apprend un
peu, et les décisions suivantes en sont insensiblement informées.

### Diapo 12 · Pourquoi PR-AUC

**À dire** (1 min 30) :

> « À 0,019 % de positifs, le choix de la métrique décide de ce qu'on croit
> avoir réussi.
>
> La ROC-AUC vaut 0,50 au hasard quel que soit le déséquilibre, et sur un
> problème aussi déséquilibré elle est flatteuse : un modèle médiocre affiche
> 0,95 sans difficulté.
>
> La PR-AUC vaut exactement le taux de base au hasard. Le rapport entre les
> deux donne le lift : combien de fois mieux que tirer au sort. C'est un
> nombre qui se dit à voix haute. »

**Question fréquente, pourquoi pas l'exactitude** : répondre par le chiffre,
99,98 % de justesse en répondant toujours non.

### Diapo 13 · Le piège du sous-échantillonnage

**À dire** (2 min) :

> « Le train est réduit à un positif pour dix négatifs, sans quoi
> l'entraînement serait ingérable. Cela crée un piège que je veux exposer,
> parce qu'il est invisible.
>
> Le prior appris vaut 9,1 %, le prior réel 0,019 % : un facteur 487.
>
> Trois conséquences. La validation et le test ne sont jamais échantillonnés,
> ce qui rend les scores comparables au monde réel. Les statistiques dérivées
> de la cible se calculent sur le train complet, jamais sur l'échantillon :
> sur l'échantillon, un lissage bayésien vaudrait 9,1 % au lieu de 0,019 %,
> le prior serait empoisonné, et rien dans les métriques ne le signalerait.
> Enfin la calibration absorbe le décalage : Platt ramène le biais de 144,7 à
> 1,13, sans rien coûter en PR-AUC. »

Insister sur le mot invisible : le modèle fonctionne, les métriques sont
bonnes, seul le niveau absolu des probabilités est faux.

### Diapo 14 · Comment comparer deux modèles

**À dire** (2 min) :

> « Un écart de PR-AUC entre deux modèles n'est pas un résultat tant qu'on
> ignore le bruit qui l'entoure. J'ai donc mis en place un bootstrap apparié.
>
> Apparié veut dire que les deux modèles sont évalués sur exactement les
> mêmes lignes, rééchantillonnées ensemble, 200 fois. On mesure la
> distribution de l'écart, et non deux distributions séparées.
>
> Et je rééchantillonne les communes, pas les lignes. C'est le point de
> méthode le plus important de cette partie. Les 1 096 jours d'une même
> commune ne sont pas indépendants, et 31 communes partagent la même maille
> météo. Tirer ligne à ligne reviendrait à traiter 38 millions
> d'observations comme 38 millions d'expériences indépendantes : les
> intervalles seraient beaucoup trop étroits, et je conclurais à des
> différences qui n'existent pas. Le terme consacré est la
> pseudo-réplication. »

**Détail technique, en réserve** : recalculer la PR-AUC 200 fois sur 38
millions de lignes coûterait des heures, chaque appel retriant le tableau. On
trie une fois ; une réplique n'est alors qu'un jeu de poids entiers le long
de cet ordre figé, et la précision moyenne pondérée se calcule en une passe
par sommes cumulées. Vérifié identique à scikit-learn à 1e-12 près.

### Diapo 15 · Les modèles mis en concurrence

**À dire** (1 min 30) :

> « Chaque modèle teste une idée précise. Ce n'est pas une collection
> d'essais.
>
> Les trois baselines mesurent ce que valent des règles sans apprentissage.
> Le v1 demande si un gradient boosting standard suffit. Le v3 ajoute une
> réponse au problème des communes sans historique. DART et le MLP testent
> deux familles d'architectures alternatives. Le modèle C demande ce qui
> reste quand on retire tout historique de feu. Et le LSTM demande si la
> séquence météo porte un signal en propre.
>
> Le modèle C joue un double rôle : c'est celui qui satisfait la contrainte
> de déploiement, et c'est aussi la seule référence honnête face au LSTM,
> puisqu'aucun des deux ne voit l'historique des feux. »

---

## 04 · Résultats (diapos 16 à 29)

### Diapo 16 · Section

Transition. Annoncer que tout ce qui suit est mesuré, et que les
interprétations viendront après.

### Diapo 17 · La barre à battre

**À dire** (1 min 30) :

> « Sans référence, un lift de 63 ne veut rien dire. J'ai donc construit trois
> prédicteurs sans aucun apprentissage.
>
> L'historique spatial seul, c'est-à-dire ce qui a brûlé rebrûlera, vaut déjà
> 19 fois le hasard. Le danger EFFIS seul vaut 5 fois. Leur croisement vaut
> 42 fois.
>
> C'est la barre à battre, et non le hasard. Un modèle à 30 fois paraîtrait
> excellent, et serait pourtant moins bon qu'une règle de trois. »

**Ce que cela répond** : la première moitié de H1. La météo seule plafonne à
5, le croisement avec le territoire monte à 42.

### Diapo 18 · Donner un risque aux communes sans historique

**À dire** (1 min 30, à accélérer si le temps manque) :

> « Le v1 tirait 54,6 % de son importance de l'historique de la commune. Il
> disait surtout que ce qui a brûlé rebrûlera, ce qui est vrai mais peu
> utile : une commune qui n'a jamais brûlé gardait un score bas, même
> entourée de communes qui brûlent chaque été.
>
> La réponse est un lissage bayésien. Je regroupe les communes qui se
> ressemblent physiquement, en 30 groupes formés sans jamais regarder le feu,
> puis je fais retomber chaque commune vers le taux de son groupe, à
> proportion de ce qu'on sait d'elle.
>
> Le gain est de 0,83 % de PR-AUC. Réel mais modeste, et le dire fait partie
> du résultat. »

**Point de méthode** : les groupes sont formés sans regarder la cible, la
sinistralité n'entre qu'ensuite. C'est ce qui empêche cette étape d'être une
fuite. C'est le problème classique d'estimation sur petits domaines.

### Diapo 19 · Les modèles, sur la même validation

**À dire** (1 min) :

> « Voici les cinq modèles retenus pour la comparaison, sur la même
> validation : 38 millions de communes-jours, 9 176 feux.
>
> Je ne vais pas commenter ce classement maintenant, parce que le commenter
> sans intervalle de confiance serait précisément l'erreur que je veux
> éviter. C'est l'objet de la diapositive suivante. »

Résister à la tentation de commenter. La retenue ici prépare la diapo 20.

### Diapo 20 · Ces écarts survivent-ils au bruit ?

C'est la diapositive centrale de la soutenance.

**À dire** (2 min 30) :

> « J'applique le bootstrap apparié décrit tout à l'heure : 200 répliques, en
> rééchantillonnant les 34 734 communes.
>
> DART et le MLP paraissaient 1,8 % et 1,9 % moins bons que XGBoost v3. Leurs
> intervalles traversent zéro. Les trois modèles sont indiscernables.
>
> Autrement dit, si j'avais présenté « XGBoost bat le MLP », j'aurais énoncé
> une conclusion tirée du bruit. Le résultat le plus instructif de cette
> partie est négatif.
>
> Deux écarts seulement survivent : celui du modèle physique et celui du
> LSTM. Ce sont les deux dont je vais parler. »

**Ce que cette diapositive démontre vraiment** : que le protocole sait dire
« je ne sais pas ». Un protocole qui conclut toujours quelque chose ne
conclut rien. C'est la formulation à employer si on vous demande pourquoi
vous y tenez.

### Diapo 21 · H3, la séquence météo

**À dire** (2 min) :

> « Face à une série temporelle, le réflexe est de prendre un LSTM. Je l'ai
> construit, optimisé et mesuré.
>
> 25 essais Optuna, arrêt précoce à l'époque 21. L'objection « il n'a pas été
> réglé » ne tient pas.
>
> Résultat : 23,6 % en dessous du modèle C, avec un intervalle de −33,5 à
> −17,3, loin de zéro.
>
> Un mot sur la comparaison, parce qu'elle n'est pas celle qu'on croit.
> XGBoost v3 voit l'historique des feux, qui pèse 29 % de ses importances ;
> le LSTM n'en voit rien. Les opposer mesurerait le prix de l'information
> retirée, pas la valeur de la séquence. La seule référence à jeu
> d'information égal est le modèle C. »

**Le chiffre qui frappe** : le LSTM voit 30 jours × 8 indices, soit 240
valeurs. Le modèle C voit les 8 indices du jour plus deux décalages. Vingt
fois plus d'historique météo, et il perd.

**Honnêteté à afficher spontanément** : le premier verdict que j'avais obtenu
annonçait −97 %. Il était faux, pour une raison décrite à la diapo 34.

### Diapo 22 · Pourquoi il perd

**À dire** (2 min) :

> « L'explication est physique, et c'est ce qui rend le résultat intéressant.
>
> Un LSTM sert quand l'ordre de la séquence porte une information qu'aucun
> résumé ne capture. Ici, ce résumé existe déjà. Les indices DC, DMC et BUI
> du système canadien sont des états récursifs : le Drought Code est une
> moyenne exponentielle de la météo passée, de constante de temps 52 jours,
> le Duff Moisture Code 15 jours. C'est la forme d'une cellule récurrente, à
> ceci près que ses coefficients ont été calibrés par cinquante ans de
> science du feu plutôt qu'estimés sur 9 176 exemples positifs.
>
> Le CEMS livre déjà l'état caché que le LSTM devrait réapprendre.
>
> Trois observations indépendantes convergent. La PACF des résidus tombe de
> 0,70 au premier retard à 0,08 au troisième : la mémoire utile de la série
> est de deux à trois jours. Un ARIMA sans variable exogène est inutilisable,
> avec une corrélation négative. Et les trois premières variables du modèle C
> sont la part de maquis, le danger EFFIS et l'ERC : le signal dit surtout où
> se trouve le combustible. »

**La nuance à donner sans attendre la question** : les retards 4 à 8 de la
PACF restent statistiquement significatifs. Avec 7 305 jours, le seuil
descend à 0,023 et presque tout le devient. Mais le retard 8, à 0,034, rend
compte de 0,11 % de la variance. Significatif ne veut pas dire utile, et
c'est une distinction qu'un jury attend.

**Réserve à donner spontanément** : le LSTM ne reçoit pas `danger_effis`, qui
pèse 13,7 % dans le modèle C. L'écart de 23,6 % est donc un majorant. C'est
la première chose que je reprendrais.

### Diapo 23 · Ce que le passé de la série prédit

**À dire** (1 min 30) :

> « J'ai fait un détour par les méthodes classiques, qui sert de
> contre-épreuve.
>
> Le test de Dickey-Fuller augmenté d'abord. Une précision, parce que le sens
> du test est contre-intuitif : l'hypothèse nulle est que la série a une
> racine unitaire, donc qu'elle n'est pas stationnaire. Rejeter H₀ signifie
> stationnaire.
>
> Ensuite SARIMAX. La saisonnalité annuelle passe par des termes de Fourier
> en variable exogène, plutôt que par une composante SARIMA d'ordre 365 qui
> serait instable et très lente.
>
> La ligne qui compte est la dernière. Un ARIMA sans variable exogène donne
> une corrélation négative : à 1 096 pas d'horizon, un modèle autorégressif
> dont la mémoire utile vaut trois jours a oublié son point de départ et
> converge vers la moyenne. Ajouter le FWI fait tomber l'erreur de 37 %. La
> prévisibilité du feu est dans la météo, pas dans son propre passé. »

**Pourquoi ce détour n'est pas décoratif** : il montre que la conclusion sur
le LSTM ne repose pas sur une seule expérience, mais sur trois familles de
méthodes qui disent la même chose.

### Diapo 24 · Le meilleur modèle n'est pas celui qu'on déploie

**À dire** (2 min 30) :

> « C'est la décision la plus contre-intuitive du projet.
>
> Sur le test, XGBoost v3 fait 93,8 fois le hasard, le modèle C 63,7. J'ai
> déployé le second.
>
> Trois raisons. D'abord, la donnée n'existe pas en temps réel : v3 tire 29 %
> de son importance de l'historique des feux, et la BDIFF ne publie pas
> l'année en cours. Les feux de 2026 sortiront au printemps 2027. Si je
> lançais le modèle ce matin, la variable feux de la commune sur 7 jours
> vaudrait le décompte d'une semaine de décembre 2025. Ce n'est pas
> imprécis, c'est faux.
>
> Ensuite, en territoire inconnu cette variable vaut zéro, et le modèle lit
> ce zéro comme « ça n'a jamais brûlé, donc ça ne brûlera pas », précisément
> là où un risque nouveau apparaît.
>
> Enfin, pour 2050 elle est impossible par construction : on ne connaîtra
> jamais les feux de 2049.
>
> Le choix se fait donc sur la disponibilité de la donnée, pas sur la
> performance. Et ce défaut n'apparaît dans aucune métrique d'entraînement :
> en validation comme en test, l'historique est toujours là. »

**Si on objecte qu'il est dommage de perdre 30 points de lift** : enchaîner
sur la diapo 28, qui montre que cet écart ne vaut que 3 points de rappel à
1 % de budget de surveillance, et rien du tout à 10 %.

### Diapo 25 · H2, retirer une région entière

**À dire** (1 min 30) :

> « Pour vérifier que ce raisonnement tient, j'ai fait une validation croisée
> spatiale : je retire une région entière du train, j'entraîne, et je teste
> sur la région exclue. Neuf fois, une par région.
>
> Le modèle physique gagne les neuf fois, sans exception : plus 8,2 % en
> moyenne pondérée, et jusqu'à plus 137 % dans le Grand Est.
>
> Là où l'historique est le plus pauvre, s'y fier devient un handicap. C'est
> l'argument décisif pour les projections, puisque le climat déplacera le
> risque vers des communes qui n'ont pas de passé. »

C'est la réponse la plus directe à H2 : le territoire porte une information
qui se transfère, l'historique non.

### Diapo 26 · La calibration

**À dire** (1 min 30, à accélérer si nécessaire) :

> « Le score brut est 145 fois trop grand, conséquence directe du
> sous-échantillonnage.
>
> Platt corrige sans rien coûter : le classement reste intact, seule
> l'échelle bouge. L'isotonique calibre aussi bien mais écrase le score, avec
> 136 valeurs distinctes au lieu de 9 millions ; on perd du pouvoir de
> discrimination sans rien gagner.
>
> Une précision sur l'application : elle affiche un rang, pas une
> probabilité. Le calibrateur dont je dispose a été ajusté sur un autre
> modèle et une autre période, il serait faux d'un facteur voisin de 2. J'ai
> préféré ne pas afficher de probabilité plutôt que d'en afficher une
> fausse. »

C'est un choix documenté, pas un oubli. Le formuler ainsi.

### Diapo 27 · L'évaluation finale

**À dire** (1 min 30) :

> « Le test a été ouvert une seule fois, après gel complet.
>
> Le lift moyen est de 63,7, mais il varie du simple au double selon l'année,
> et cette variation n'est pas du bruit : elle suit la rareté. 2024 est
> l'année la plus calme, avec 1 297 feux, et donne le meilleur lift, 151.
> 2023, la plus active, donne le plus faible, 76.
>
> L'explication est qu'une année calme concentre les feux dans les endroits
> les plus prévisibles. Quand tout brûle, y compris là où ce n'est pas
> attendu, le modèle est pris en défaut. »

Ce que cette diapositive montre méthodologiquement : on a regardé la
variabilité, et pas seulement la moyenne.

### Diapo 28 · Ce que le modèle change concrètement

**À dire** (1 min 30) :

> « Un lift ne se traduit pas directement en décision. La question utile est :
> si on peut surveiller 1 % du territoire, combien de départs couvre-t-on ?
>
> La réponse est 42 %. En surveillant 1 % des communes-jours, on couvre 42 %
> des départs.
>
> Et il y a une lecture moins flatteuse, que je préfère donner moi-même : les
> 37 % de PR-AUC qui séparent v3 du modèle C ne valent que 3 points de rappel
> à 1 % de budget, et plus rien du tout à 10 %. Une grande partie de l'écart
> que je mesure est opérationnellement invisible. »

C'est la diapositive qui relie les métriques à une décision, et qui achève de
justifier le choix de la diapo 24.

### Diapo 29 · H4, le danger monte, les feux non

**À dire** (2 min) :

> « Quatre séries, sur deux fenêtres d'observation différentes.
>
> Le FWI moyen annuel augmente de 58 % sur 53 ans. Le FWI estival de 62 %. Le
> nombre de jours de danger élevé de 197 %. Ces trois tendances sont très
> significatives.
>
> Le nombre de communes-jours en feu, lui, ne montre aucune tendance
> significative sur les 20 années dont je dispose.
>
> La formulation exacte est donc : les conditions favorables aux feux
> augmentent très significativement, et le nombre de départs reste stable. Je
> ne dirai pas que les feux augmentent, parce que mes données ne le montrent
> pas. »

**Les trois lectures possibles, à donner dans cet ordre** :

1. La puissance statistique est faible : 20 points d'observation seulement,
   contre 53 pour les séries météo.
2. La prévention absorbe pour l'instant la hausse de l'aléa, ce qui est
   cohérent avec les politiques de débroussaillement et de surveillance.
3. On mesure l'aléa, pas le bilan : un aléa plus élevé peut se traduire par
   des feux plus grands plutôt que plus nombreux.

Ne pas trancher. Dire qu'on ne peut pas trancher avec ces données est la
réponse juste.

---

## 05 · Interprétation (diapos 30 à 37)

### Diapo 30 · Section

Transition. Annoncer qu'on passe de ce qui est mesuré à ce que cela veut
dire.

### Diapo 31 · Qu'est-ce qui fait partir un feu ?

**À dire** (2 min 30) :

> « Trois façons de poser la question, trois réponses, chacune juste mais pas
> à la même question.
>
> L'importance par gain mesure combien chaque variable a réduit la perte
> pendant l'entraînement. SHAP sur un échantillon aléatoire mesure combien
> elle déplace le score sur le territoire tel qu'il est, c'est-à-dire à
> 99,97 % des communes-jours sans feu. SHAP au sommet du classement mesure
> combien elle le déplace là où le modèle s'engage.
>
> Deux désaccords valent d'être expliqués. Le danger EFFIS est deuxième par
> gain et trentième par SHAP : c'est une discrétisation du FWI en six
> classes, XGBoost trouve ces seuils commodes pour découper, mais
> l'information est déjà dans le FWI continu et SHAP lui en attribue le
> crédit. L'importance par gain ne sait pas traiter la redondance.
>
> La part de maquis est première par gain, dixième sur l'échantillon
> aléatoire et deuxième au sommet : elle ne change rien sur une commune-jour
> moyenne, où il n'y a pas de maquis, et devient déterminante là où le modèle
> voit du risque.
>
> Conséquence pratique : citer la variable la plus importante n'a pas de sens
> sans préciser quelle mesure et sur quelle population. »

**Ce que cela répond** : H1 et H2 réunies. Les variables de territoire
occupent le haut du classement à égalité avec la météo. La météo situe le
moment, le territoire situe le lieu.

**Si on demande quelle garantie** : TreeSHAP est exact sur un modèle
d'arbres, ce n'est pas une approximation par échantillonnage. Vérifié dans le
projet, la somme des contributions plus la valeur de base redonne le logit du
score à 9,5e-07 près.

**Attention au biais de collision** : sélectionner sur le score conditionne
le résultat, et j'ai vu le signe d'une interaction s'inverser à cause de
cela. Le panneau « sommet » ne se lit que pour la question « quand ça
brûle ».

### Diapo 32 · La seule question actionnable

**À dire** (2 min) :

> « SHAP et LIME répondent à pourquoi ce score. DiCE répond à qu'aurait-il
> fallu changer, et c'est la seule des trois dont la réponse se traduise en
> décision.
>
> Un exemple mesuré : Bormes-les-Mimosas, 99,9ᵉ percentile du 12 août 2024.
> J'autorise à modifier la végétation seule, maquis, forêt, part combustible,
> part agricole.
>
> Résultat : aucun contrefactuel. Rien, sur ces leviers, ne fait sortir la
> commune du décile à risque. Son exposition tient à sa position, à son
> relief, à sa superficie. Le risque est structurel, et l'absence de solution
> est ici la réponse. »

**Le détail d'implémentation qui change tout** : DiCE cherche par défaut à
faire passer la probabilité sous 0,5. Or le score n'est pas calibré, 0,5
correspond à un risque astronomique, et l'outil ne renvoyait jamais rien.
J'ai recentré la frontière sur le décile par une transformation strictement
croissante, qui laisse le classement intact. La question devient « que
faudrait-il pour sortir des 10 % les plus à risque », celle qui a un sens
opérationnel.

**La mise en garde à donner** : un contrefactuel n'est pas une
recommandation. Rien ne garantit qu'il soit réalisable, on ne convertit pas
40 % de maquis en terres agricoles, ni que le lien soit causal, le modèle
ayant appris des corrélations et non des mécanismes.

**Sur LIME, si la question vient** : sur un modèle d'arbres il approxime ce
que TreeSHAP calcule exactement, donc il ne peut pas faire mieux. Il figure
dans l'application parce qu'il est très répandu et qu'il vaut mieux savoir
pourquoi on ne l'a pas retenu. Sur un modèle qu'on ne peut pas ouvrir, une
API ou un réseau profond, il redeviendrait le bon outil.

### Diapo 33 · Projeter jusqu'en 2100

**À dire** (2 min) :

> « Trois scénarios GIEC, appliqués à l'aléa météo. Je veux être précis sur
> ce que cela veut dire et sur ce que cela ne veut pas dire.
>
> Ce que je projette, c'est le FWI, seule quantité qui montre un signal et
> seule que les modèles climatiques savent fournir.
>
> Ce que je ne projette pas, c'est le nombre de feux. La végétation, la
> prévention et les pratiques agricoles sont supposées constantes, ce qui est
> une hypothèse, et elle est fausse.
>
> Quand l'application affiche le 2 août 2050, cela ne désigne pas une
> prévision météo pour ce jour-là, mais un 2 août ordinaire sous le climat de
> 2050. La forme de la saison vient des observations, seul son niveau est
> décalé.
>
> Dernier point, que je préfère donner moi-même : avant 2045, les trois
> scénarios sont indiscernables. En 2030, le RCP 2.6 dépasse le RCP 8.5 sur
> 59 % des communes. Ce n'est pas une anomalie de mon code, c'est l'inertie
> du système climatique : les trajectoires d'émissions ne divergent
> réellement qu'après le milieu du siècle. »

L'application affiche cet avertissement à chaque date postérieure à 2025 et
grise la zone antérieure à 2045 sur les graphiques de projection.

### Diapo 34 · Les erreurs commises

**À dire** (2 min 30) :

> « Sur un événement à 0,02 %, une erreur ne se manifeste jamais par une
> exception. Elle se manifeste par un chiffre plausible. Voici les
> principales.
>
> Mon premier verdict sur le LSTM annonçait moins 97 %. Le vrai est moins 52.
> L'écart ne venait pas du modèle, mais de la façon de comparer deux
> fichiers. La requête d'assemblage n'a pas d'ORDER BY : l'ordre des 38
> millions de lignes que renvoie PostgreSQL dépend du plan d'exécution et
> change d'une exécution à l'autre. Mes fichiers ne portaient que le score et
> la cible : même taille, même nombre de feux, ordre différent, et aucune
> erreur levée.
>
> La parade est simple : tout fichier porte désormais ses clés, une fonction
> d'alignement vérifie, et un test refuse un fichier sans clés. »

Puis parcourir rapidement le tableau des autres erreurs.

**La démonstration, si le temps le permet** : le bug est reproduit
volontairement dans le notebook et dans l'application. On permute les lignes,
la PR-AUC tombe de 0,0085 à 0,0002, exactement la ligne du hasard, avec
strictement les mêmes valeurs dans le fichier.

**La formulation à employer** : les seules défenses sont les invariants
explicites et les assertions qui échouent bruyamment. 50 tests tournent en
intégration continue à chaque commit.

### Diapo 35 · Ce que le projet ne fait pas

**À dire** (2 min) :

> « Les limites que je connais valent mieux que celles qu'on me
> découvrirait.
>
> La surface brûlée n'est pas prédictible : R² de 0,14, moins bon que
> d'annoncer toujours la médiane. Elle dépend de ce qui se passe après le
> départ, vent, délai d'intervention, relief.
>
> Sera-ce un grand feu se prédit mal aussi : lift de 2,9 seulement, contre
> 63,7 pour les départs. On peut lire une ROC-AUC de 0,77 sur cette tâche,
> c'est exact, mais m'en servir après avoir expliqué pourquoi la ROC-AUC
> flatte serait me contredire.
>
> Une commune-jour n'est pas un incendie : un feu traversant cinq communes
> compte cinq fois.
>
> 31 communes partagent une maille météo de 28 km, avec une conséquence
> statistique directe : les intervalles naïfs sur les coefficients météo
> seraient trop étroits.
>
> Et le LSTM n'a pas reçu danger_effis. L'écart de 23,6 % reste un majorant
> tant que cette asymétrie n'est pas levée. C'est la première chose à
> refaire. »

### Diapo 36 · L'application

**À dire** (3 min, démonstration comprise) :

Faire la démonstration en direct : la carte d'abord, puis le basculement
rétrospectif sur une date d'août, puis la page *Pourquoi un feu part*.

> « L'application est déployée publiquement. La carte affiche les 34 696
> communes en aplat, de 1973 à 2100, avec trois scénarios GIEC.
>
> Un point que je veux souligner : le mode rétrospectif refuse d'afficher le
> modèle v3 sur 20 des 23 années qu'il pourrait techniquement couvrir. Le
> train, parce qu'il a appris ces lignes. La validation, parce qu'elle a
> servi à choisir. L'avenir, parce qu'il n'y a pas d'historique. Et il sait
> dire laquelle des trois raisons s'applique à la date demandée. »

**Si la connexion échoue** : les captures des diapos 1, 20, 25 et 29 viennent
de l'application et suffisent à montrer l'essentiel.

### Diapo 37 · Réponses aux quatre hypothèses

**À dire** (1 min 30) :

Reprendre le tableau ligne par ligne sans le paraphraser longuement, puis
conclure :

> « La réponse à la question principale est positive, avec une réserve
> explicite : j'estime un risque relatif, pas une probabilité absolue. C'est
> pour cette raison que l'application affiche un rang.
>
> S'il ne devait rester qu'une chose : le modèle classe bien, il ne quantifie
> pas. »

Terminer sur la réserve plutôt que sur la performance.

---

## Annexe A · Fiche de chaque modèle

### Baselines (aucun apprentissage)

| Prédicteur | PR-AUC | lift |
|---|---|---|
| hasard | 0,000241 | ×1,0 |
| historique de la commune | 0,004668 | ×19,4 |
| danger EFFIS seul | 0,001220 | ×5,1 |
| historique × EFFIS | 0,010149 | ×42,1 |

Aucun paramètre ajusté. Ce sont les références qui donnent son sens au reste.

### Modèle v1 · XGBoost, 43 variables

Premier modèle complet : météo du jour, occupation du sol, démographie,
historique brut de la commune. Sert de point de départ et révèle le problème
principal, à savoir que 54,6 % de l'importance vient de l'historique.

### Modèle v2 · v1 optimisé

Recherche d'hyperparamètres par Optuna, échantillonnage TPE, 60 essais. Gain
réel mais faible. Confirme que le problème du v1 n'est pas un problème de
réglage.

### Modèle v3 · lissage bayésien, 52 variables

Ajoute une typologie territoriale en 30 groupes, formée sans regarder la
cible, et un lissage bayésien qui fait retomber chaque commune vers le taux
de son groupe. Meilleur modèle du projet en PR-AUC : 0,0156 sur le test,
lift 93,8. Non déployé, pour les raisons de la diapo 24.

### DART · v3 avec abandon d'arbres

Variante de XGBoost où des arbres sont ignorés à chaque itération, ce qui
limite en principe le surapprentissage. Écart avec v3 : −1,8 %, intervalle
[−5,0 ; +0,8]. Indiscernable.

### MLP · réseau dense, 52 variables

Réseau dense sur variables normalisées. Écart avec v3 : −1,9 %, intervalle
[−7,9 ; +5,2]. Indiscernable également.

### Modèle C · physique pure, 41 variables, le modèle déployé

Aucune variable dérivée de l'historique des feux. Uniquement météo,
occupation du sol, relief, densité, littoral, calendrier.

Test 2023-2025 : PR-AUC 0,0106, lift 63,7.

Déployé parce qu'il est le seul à ne dépendre d'aucune donnée indisponible en
temps réel, et parce qu'il gagne les 9 régions en validation croisée
spatiale.

### LSTM · séquence de 30 jours

Entrée : 30 jours × 8 indices météo, soit 240 valeurs par exemple. 25 essais
Optuna, arrêt précoce à l'époque 21.

Écart avec le modèle C, à information égale : −23,6 %, intervalle
[−33,5 ; −17,3]. L'écart est réel et significatif.

Réserve connue : ne reçoit pas `danger_effis`, donc le chiffre est un
majorant.

### Ensemble · moyenne de rangs v3 + MLP

PR-AUC 0,0180, lift 74,9 sur la validation. Meilleur que chacun de ses
composants, au prix de deux modèles à faire tourner en production. Non
retenu : le gain ne justifie pas le doublement de la chaîne de service.

### Modèles secondaires

- Surface brûlée, régression : R² 0,14, MAE 3,94 ha. Non exploitable.
- Grand feu, classification au seuil de 5 ha : PR-AUC 0,232, lift 2,88,
  ROC-AUC 0,766. Faible.

---

## Annexe B · Questions attendues

### Sur la méthode

**Pourquoi ne pas avoir utilisé l'exactitude ?**
Parce qu'à 0,019 % de positifs, répondre toujours non donne 99,98 %. La
métrique doit être la PR-AUC, et le lift en est la lecture parlante.

**Pourquoi 200 répliques de bootstrap et pas 1 000 ?**
Au-delà de 200, la largeur des intervalles ne bouge plus de façon utile,
alors que le coût est linéaire. Les conclusions ne dépendent pas de ce
choix : les intervalles qui traversent zéro le traversent largement.

**Pourquoi rééchantillonner les communes et pas les lignes ?**
Parce que les 1 096 jours d'une même commune ne sont pas indépendants, et que
31 communes partagent la même maille météo. Un bootstrap ligne à ligne
traiterait 38 millions d'observations comme 38 millions d'expériences
indépendantes, et donnerait des intervalles faussement étroits.

**Comment savez-vous qu'il n'y a pas de fuite ?**
Par une règle explicite appliquée à chaque variable : une variable datée peut
regarder tout le passé, une statistique non datée ne peut regarder que le
train. Et par des tests : 50 tournent à chaque commit, dont deux dédiés au
split et à l'absence de fuite.

**Avez-vous regardé le test plusieurs fois ?**
Une seule, après gel complet. Toutes les comparaisons de modèles, tous les
réglages et la calibration ont été faits sur la validation.

### Sur les modèles

**Pourquoi pas de deep learning plus élaboré ?**
J'ai testé un LSTM, qui perd de 23,6 % à information égale, et j'explique
pourquoi : les indices du système canadien sont déjà des états récursifs, le
CEMS livre l'état caché que le réseau devrait réapprendre. Un modèle plus
gros ne changerait pas cette structure.

**Pourquoi déployer le modèle le moins performant ?**
Parce que la variable qui fait la différence n'existe pas en temps réel, vaut
zéro en territoire inconnu, et est impossible par construction pour 2050. Le
choix se fait sur la disponibilité, pas sur la métrique. Et l'écart vaut 3
points de rappel à 1 % de budget, rien à 10 %.

**Comment expliquez-vous que DART et le MLP soient équivalents à XGBoost ?**
Je ne l'explique pas, je le constate : leurs intervalles traversent zéro. La
lecture la plus probable est que le signal disponible est capté de façon
comparable par les trois familles, et que la limite est dans les données, pas
dans l'architecture.

### Sur les données

**965 feux avec un code INSEE disparu, comment les avez-vous traités ?**
Par le fichier officiel des mouvements de communes de l'INSEE. 935 sont
rattachés avec certitude, 30 sont écartés et comptés. Aucun n'a été deviné
par le nom, parce que le nom produit des faux positifs.

**Pourquoi 253 millions de lignes plutôt que la liste des feux ?**
Parce qu'une série creuse rend les fenêtres glissantes silencieusement
fausses : remonter 30 lignes ne remonte pas 30 jours si les jours sans feu
sont absents.

**La BDIFF est-elle exhaustive ?**
Non, et le projet le documente. La couverture varie selon les départements et
les périodes, et 64 % des causes déclarées sont manquantes. C'est une limite
de la cible, pas seulement des variables.

### Sur les projections

**Comment projetez-vous jusqu'en 2100 ?**
En appliquant un facteur multiplicatif issu des scénarios GIEC à la
climatologie observée sur 2006-2019. La forme de la saison vient des
observations, seul son niveau est décalé.

**Pourquoi le RCP 2.6 apparaît-il parfois au-dessus du RCP 8.5 ?**
Parce qu'avant 2045 les scénarios sont indiscernables : en 2030, le 2.6
dépasse le 8.5 sur 59 % des communes. C'est l'inertie du système climatique,
les trajectoires ne divergent qu'après le milieu du siècle. L'application
grise cette zone.

**Peut-on en déduire le nombre de feux en 2050 ?**
Non. Je projette l'aléa météo, pas le bilan. La végétation, la prévention et
les pratiques agricoles sont supposées constantes, ce qui est faux.

### Questions plus difficiles

**Votre modèle est-il utilisable par un SDIS demain matin ?**
Pas en l'état. Il donne un classement national par jour, ce qui répond à « où
regarder en priorité », mais il n'est pas calibré en probabilité, il ne tient
pas compte des moyens disponibles, et il n'a pas été évalué en conditions
opérationnelles. Ce qu'il fournit est une aide au ciblage.

**Qu'est-ce qui vous dit que le modèle n'a pas simplement appris la
géographie ?**
Il n'a ni latitude ni longitude. La carte qu'il produit retrouve pourtant les
Landes, l'arc méditerranéen et la Corse, à partir de la végétation, du relief
et du littoral. Et la validation croisée spatiale montre qu'il transfère à
des régions retirées de l'entraînement.

**Si vous aviez trois mois de plus ?**
Dans l'ordre : donner `danger_effis` au LSTM pour lever l'asymétrie de la
comparaison, refaire une calibration propre sur le modèle déployé et sur la
bonne période, et chercher une source de cause de départ moins lacunaire que
les 64 % de valeurs manquantes actuelles.

**Quelle est l'erreur qui vous a coûté le plus cher ?**
Le désalignement des lignes, parce qu'il ne lève aucune exception et donne un
résultat plausible. J'ai failli conclure que le LSTM perdait de 97 %.

---

## Annexe C · Glossaire

| Terme | Définition courte |
|---|---|
| FWI | Fire Weather Index, indice synthétique du système canadien |
| DC / DMC / BUI | codes de sécheresse à mémoire longue (52 j), moyenne (15 j), et combustible disponible |
| PR-AUC | aire sous la courbe précision-rappel ; vaut le taux de base au hasard |
| lift | PR-AUC divisée par le taux de base : combien de fois mieux que le hasard |
| ACF / PACF | autocorrélation, et autocorrélation partielle (apport propre d'un retard) |
| ADF | test de Dickey-Fuller augmenté ; H₀ = racine unitaire = non stationnaire |
| SARIMAX | ARIMA saisonnier avec variables exogènes |
| SHAP | décomposition exacte d'un score en contributions par variable |
| LIME | substitut linéaire local, approché |
| DiCE | génération de contrefactuels : que faudrait-il changer |
| RCP | Representative Concentration Pathway, scénario d'émissions du GIEC |
| Pseudo-réplication | traiter des observations corrélées comme indépendantes |
| Biais de collision | conditionner sur une variable causée par deux autres, ce qui crée une association artificielle |

---

## Annexe D · Commandes

Régénérer le diaporama après un changement de résultats :

```bash
python -m tvfed.comparer && python -m tvfed.diaporama
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
