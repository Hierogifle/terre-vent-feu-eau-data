DROP TABLE IF EXISTS eda_commune;
CREATE TABLE eda_commune AS
SELECT c.code_insee, c.nom, c.dep_code, c.reg_code, c.lat, c.lon,
       c.population, c.densite, c.superficie_km2, c.grille_densite,
       c.altitude_moy, c.altitude_max - c.altitude_min AS relief, c.distance_cote_km,
       clc.part_foret, clc.part_maquis, clc.part_combustible, clc.part_agricole,
       clc.part_artificialise, clc.part_coniferes, clc.part_feuillus,
       COALESCE(f.n, 0) AS nb_feux,
       COALESCE(f.ha, 0) AS ha_brules
FROM ref_commune c
LEFT JOIN (SELECT code_insee, count(*) n, sum(COALESCE(surface_m2,0))/10000 ha
           FROM fait_feu WHERE code_insee IS NOT NULL GROUP BY 1) f
       ON f.code_insee = c.code_insee
LEFT JOIN clc_part clc ON clc.code_insee = c.code_insee AND clc.millesime = 2018;
SELECT count(*) AS communes, count(*) FILTER (WHERE nb_feux>0) AS avec_feu,
       round(sum(ha_brules)) AS ha_total FROM eda_commune;
