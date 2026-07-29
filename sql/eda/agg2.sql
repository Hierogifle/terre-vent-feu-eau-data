-- ═══════════════════════════════════════════════════════════════════════
-- Agrégats croisant la météo et la grille — les plus coûteux (253 M lignes).
-- Calculés une fois, stockés, relus par le notebook.
-- ═══════════════════════════════════════════════════════════════════════

-- 1. Taux de feu par classe EFFIS  ← c'est la BASELINE 2 de Plan.md :
--    « le Danger rating EFFIS tout seul ». Si le FWI sépare bien les classes,
--    on le voit ici, et on a le score de référence à battre.
DROP TABLE IF EXISTS eda_effis;
CREATE TABLE eda_effis AS
SELECT
    CASE WHEN m.fwi <  5.2 THEN 1 WHEN m.fwi < 11.2 THEN 2
         WHEN m.fwi < 21.3 THEN 3 WHEN m.fwi < 38.0 THEN 4
         WHEN m.fwi < 50.0 THEN 5 ELSE 6 END          AS classe_effis,
    count(*)                                          AS jours_communes,
    count(*) FILTER (WHERE g.y)                       AS feux,
    sum(g.surface_m2) / 10000                         AS ha
FROM grille g
JOIN ref_commune c ON c.code_insee = g.code_insee
JOIN fait_meteo  m ON m.cell_id = c.cell_id AND m.date = g.date
GROUP BY 1;

-- 2. FWI au moment du feu, selon la végétation dominante de la commune.
--    Question : le maquis brûle-t-il à un FWI plus BAS que la forêt ?
--    Si oui, c'est l'interaction que le FWI seul ne peut pas capter — et
--    l'argument qui justifie d'avoir végétation ET météo dans le modèle.
DROP TABLE IF EXISTS eda_fwi_vegetation;
CREATE TABLE eda_fwi_vegetation AS
SELECT
    CASE
        WHEN clc.part_maquis     > 0.15 THEN 'maquis / garrigue'
        WHEN clc.part_coniferes  > 0.20 THEN 'conifères'
        WHEN clc.part_feuillus   > 0.30 THEN 'feuillus'
        WHEN clc.part_agricole   > 0.60 THEN 'agricole'
        WHEN clc.part_artificialise > 0.30 THEN 'urbain'
        ELSE 'mixte'
    END                                               AS vegetation,
    m.fwi,
    f.surface_m2 / 10000                              AS ha,
    EXTRACT(doy FROM f.date_alerte)::int              AS doy
FROM fait_feu f
JOIN ref_commune c ON c.code_insee = f.code_insee
JOIN fait_meteo  m ON m.cell_id = c.cell_id AND m.date = f.date_alerte
LEFT JOIN clc_part clc ON clc.code_insee = f.code_insee AND clc.millesime = 2018
WHERE f.code_insee IS NOT NULL;

-- 3. Exposition : combien de jours-communes dans chaque classe de végétation ?
--    Indispensable pour normaliser — sans ça on mesure ce qui est abondant
--    en France, pas ce qui brûle.
DROP TABLE IF EXISTS eda_expo_vegetation;
CREATE TABLE eda_expo_vegetation AS
SELECT
    CASE
        WHEN clc.part_maquis     > 0.15 THEN 'maquis / garrigue'
        WHEN clc.part_coniferes  > 0.20 THEN 'conifères'
        WHEN clc.part_feuillus   > 0.30 THEN 'feuillus'
        WHEN clc.part_agricole   > 0.60 THEN 'agricole'
        WHEN clc.part_artificialise > 0.30 THEN 'urbain'
        ELSE 'mixte'
    END                                               AS vegetation,
    count(*)                                          AS communes,
    sum(c.superficie_km2)                             AS km2
FROM ref_commune c
LEFT JOIN clc_part clc ON clc.code_insee = c.code_insee AND clc.millesime = 2018
GROUP BY 1;
