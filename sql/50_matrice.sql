-- ═══════════════════════════════════════════════════════════════════════
-- LA MATRICE D'ENTRAÎNEMENT — une ligne = une commune, un jour.
--
-- C'est ici que les 5 sources se rencontrent enfin :
--   BDIFF      → y, nb_feux, et les features d'historique
--   CEMS       → les 8 indices feu du jour, via ref_commune.cell_id
--   CORINE     → la végétation, au millésime le plus récent PAS DANS LE FUTUR
--   communes   → population, altitude, relief, distance à la côte
--   calendrier → saisonnalité et jours fériés
--
-- ⚠️ Ce qui N'EST PAS ici : lissage bayésien, clusters HDBSCAN, standardisation.
-- Ces transformations ont un .fit() et doivent être apprises sur le train
-- SEUL, après le split, en Python. Les mettre ici serait une fuite.
--
-- Paramétrée par :date_min / :date_max, et NON par un nom de split.
-- ⚠️ C'est délibéré : filtrer sur une colonne `split` calculée par jointure
-- empêche PostgreSQL d'élaguer les partitions et force un balayage des
-- 253 M lignes. Avec des bornes de date littérales, seules les partitions
-- annuelles concernées sont lues.
-- ═══════════════════════════════════════════════════════════════════════

SELECT
    -- ── clés et cibles ──
    g.code_insee,
    g.date,
    g.y::int                                        AS y,
    g.nb_feux,
    g.surface_m2,

    -- ── A. météo du jour, via la maille de la commune ──
    m.fwi, m.ffmc, m.dmc, m.dc, m.bui, m.isi, m.kbdi, m.erc,
    -- classe EFFIS officielle, dérivée des seuils publiés
    CASE WHEN m.fwi <  5.2 THEN 1 WHEN m.fwi < 11.2 THEN 2
         WHEN m.fwi < 21.3 THEN 3 WHEN m.fwi < 38.0 THEN 4
         WHEN m.fwi < 50.0 THEN 5 ELSE 6 END        AS danger_effis,

    -- ── A'. météo décalée : la veille et les extrêmes récents ──
    mv.fwi                                          AS fwi_j1,
    mv.ffmc                                         AS ffmc_j1,

    -- ── B. végétation, millésime ≤ année (anti-fuite temporelle) ──
    clc.millesime                                   AS clc_millesime,
    clc.part_foret, clc.part_feuillus, clc.part_coniferes, clc.part_melangees,
    clc.part_landes, clc.part_maquis, clc.part_veg_mutation,
    clc.part_veg_clairsemee, clc.part_combustible,
    clc.part_agricole, clc.part_artificialise,

    -- ── C. territoire (statique) ──
    ln(COALESCE(c.population, 0) + 1)               AS log_population,
    ln(COALESCE(c.densite, 0) + 1)                  AS log_densite,
    ln(COALESCE(c.superficie_km2, 0) + 1)           AS log_superficie,
    c.altitude_moy,
    c.altitude_max - c.altitude_min                 AS amplitude_altitude,
    c.grille_densite,
    c.distance_cote_km,
    c.lat, c.lon,
    c.dep_code,

    -- ── D. calendrier (row-local, aucun risque) ──
    cal.doy, cal.mois, cal.jour_semaine,
    cal.est_weekend::int, cal.est_ferie::int,
    cal.est_14_juillet::int, cal.est_15_aout::int,
    cal.sin_doy, cal.cos_doy, cal.sin_mois, cal.cos_mois,

    -- ── E. historique, passé STRICT (0 si aucun feu antérieur) ──
    COALESCE(fl.feux_commune_7j,   0)               AS feux_commune_7j,
    COALESCE(fl.feux_commune_30j,  0)               AS feux_commune_30j,
    COALESCE(fl.feux_commune_90j,  0)               AS feux_commune_90j,
    COALESCE(fl.feux_commune_365j, 0)               AS feux_commune_365j,
    COALESCE(fl.jours_depuis_dernier_feu, 9999)     AS jours_depuis_dernier_feu,

    -- ⚠️ Les features de VOISINAGE (feux à 10/20/50 km sur 30 j) ont été
    -- retirées volontairement. Décision de cadrage : la dimension spatiale
    -- sera traitée d'UNE seule façon, par clustering appris après le split
    -- (voir sql/41_feat_voisinage.sql.reporte pour l'implémentation gardée
    -- de côté). Elles ne fuyaient pas — la fenêtre était bornée à J-1, et
    -- tests/test_no_leakage.py le vérifiait — mais mélanger deux encodages
    -- de la même intuition rendait l'interprétation SHAP ambiguë.

    :split_nom                                      AS split
FROM grille g
JOIN ref_commune    c   ON c.code_insee = g.code_insee
JOIN ref_calendrier cal ON cal.date     = g.date
JOIN fait_meteo     m   ON m.cell_id    = c.cell_id AND m.date = g.date
LEFT JOIN fait_meteo mv ON mv.cell_id   = c.cell_id AND mv.date = g.date - 1
LEFT JOIN feat_lags     fl ON fl.code_insee = g.code_insee AND fl.date = g.date
-- LATERAL : le millésime CORINE le plus récent qui ne soit pas dans le futur
LEFT JOIN LATERAL (
    SELECT * FROM clc_part p
    WHERE p.code_insee = g.code_insee
      AND p.millesime <= cal.annee
    ORDER BY p.millesime DESC
    LIMIT 1
) clc ON true
-- bornes littérales : c'est ce qui déclenche l'élagage des partitions
WHERE g.date >= :date_min AND g.date <= :date_max
  -- downsampling : 100 % des positifs, et une fraction déterministe des
  -- négatifs. `:frac` vaut 1.0 pour val et test, qui restent INTÉGRAUX.
  AND (g.y OR g.u < :frac);
