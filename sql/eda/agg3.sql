-- ═══════════════════════════════════════════════════════════════════════
-- LE croisement qui compte : végétation × classe EFFIS.
--
-- Le FWI brut au moment du feu est TROMPEUR : le maquis est méditerranéen,
-- donc son FWI ambiant est structurellement élevé. Comparer « FWI moyen au
-- moment du feu » entre végétations mesure surtout la géographie.
--
-- La bonne question : À FWI ÉGAL, quelle végétation brûle le plus ?
-- Si l'écart persiste à classe EFFIS constante, alors la végétation apporte
-- une information que la météo seule ne contient pas — ce qui est
-- exactement l'argument du projet.
-- ═══════════════════════════════════════════════════════════════════════

DROP TABLE IF EXISTS eda_veg_effis;
CREATE TABLE eda_veg_effis AS
SELECT
    CASE
        WHEN clc.part_maquis     > 0.15 THEN 'maquis / garrigue'
        WHEN clc.part_coniferes  > 0.20 THEN 'conifères'
        WHEN clc.part_feuillus   > 0.30 THEN 'feuillus'
        WHEN clc.part_agricole   > 0.60 THEN 'agricole'
        WHEN clc.part_artificialise > 0.30 THEN 'urbain'
        ELSE 'mixte'
    END                                               AS vegetation,
    CASE WHEN m.fwi <  5.2 THEN 1 WHEN m.fwi < 11.2 THEN 2
         WHEN m.fwi < 21.3 THEN 3 WHEN m.fwi < 38.0 THEN 4
         WHEN m.fwi < 50.0 THEN 5 ELSE 6 END          AS classe_effis,
    count(*)                                          AS jours_communes,
    count(*) FILTER (WHERE g.y)                       AS feux,
    sum(g.surface_m2) / 10000                         AS ha
FROM grille g
JOIN ref_commune c ON c.code_insee = g.code_insee
JOIN fait_meteo  m ON m.cell_id = c.cell_id AND m.date = g.date
LEFT JOIN clc_part clc ON clc.code_insee = g.code_insee AND clc.millesime = 2018
GROUP BY 1, 2;

-- Altitude : nombre de feux et hectares par bande, avec l'exposition
-- (nombre de communes) pour pouvoir normaliser.
DROP TABLE IF EXISTS eda_altitude;
CREATE TABLE eda_altitude AS
WITH bande AS (
    SELECT c.code_insee,
           width_bucket(c.altitude_moy, 0, 2000, 20) AS b,
           c.superficie_km2
    FROM ref_commune c WHERE c.altitude_moy IS NOT NULL
)
SELECT b.b                                            AS bande,
       (b.b - 1) * 100                                AS alt_min,
       count(DISTINCT b.code_insee)                   AS communes,
       sum(b.superficie_km2)                          AS km2,
       COALESCE(sum(f.n), 0)                          AS feux,
       COALESCE(sum(f.ha), 0)                         AS ha
FROM bande b
LEFT JOIN (SELECT code_insee, count(*) n, sum(COALESCE(surface_m2,0))/10000 ha
           FROM fait_feu WHERE code_insee IS NOT NULL GROUP BY 1) f
       ON f.code_insee = b.code_insee
GROUP BY 1, 2 ORDER BY 1;

-- Saisonnalité par végétation : le second pic de mars est-il agricole ?
DROP TABLE IF EXISTS eda_saison_veg;
CREATE TABLE eda_saison_veg AS
SELECT
    CASE
        WHEN clc.part_maquis     > 0.15 THEN 'maquis / garrigue'
        WHEN clc.part_coniferes  > 0.20 THEN 'conifères'
        WHEN clc.part_feuillus   > 0.30 THEN 'feuillus'
        WHEN clc.part_agricole   > 0.60 THEN 'agricole'
        WHEN clc.part_artificialise > 0.30 THEN 'urbain'
        ELSE 'mixte'
    END                                               AS vegetation,
    EXTRACT(doy FROM f.date_alerte)::int              AS doy,
    count(*)                                          AS feux,
    sum(COALESCE(f.surface_m2, 0)) / 10000            AS ha
FROM fait_feu f
LEFT JOIN clc_part clc ON clc.code_insee = f.code_insee AND clc.millesime = 2018
WHERE f.code_insee IS NOT NULL
GROUP BY 1, 2;
