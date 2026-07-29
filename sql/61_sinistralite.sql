-- Sinistralité par commune et par année, sur le TRAIN uniquement.
--
-- C'est la matière première du lissage bayésien. Deux points de méthode :
--
-- ⚠️ 1. AGRÉGÉ SUR LE TRAIN COMPLET (177,6 M lignes), pas sur le train
--    échantillonné. Sur l'échantillon 1:10 le taux vaut 9,1 % contre
--    0,0187 % en réalité — un facteur ×487. Un lissage calé sur 9,1 %
--    donnerait un prior absurde, et RIEN dans les métriques ne le
--    signalerait : le modèle serait simplement moins bon sans qu'on sache
--    pourquoi. C'est le piège n°1 du cadrage.
--
-- ⚠️ 2. DÉCOUPÉ PAR ANNÉE, pour permettre l'exclusion année par année.
--    Utiliser la sinistralité 2006-2019 d'une commune comme feature d'une
--    ligne de 2012 laisserait cette ligne contribuer à sa propre feature :
--    c'est la fuite classique du target encoding. Le détail annuel permet
--    de retrancher l'année de la ligne (voir clustering.py).
--
-- Les bornes sont écrites en dur : filtrer sur la vue `grille_split`
-- empêcherait PostgreSQL d'élaguer les partitions et ferait scanner
-- les 253,7 M lignes au lieu de 177,6 M.

SELECT code_insee,
       extract(year from date)::int AS an,
       count(*)                     AS jours,
       count(*) FILTER (WHERE y)    AS feux
FROM grille
WHERE date BETWEEN '2006-01-01' AND '2019-12-31'
GROUP BY 1, 2;
