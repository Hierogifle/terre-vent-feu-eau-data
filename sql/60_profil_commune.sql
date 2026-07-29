-- Profil STATIQUE de chaque commune, pour le clustering territorial.
--
-- Une ligne par commune, aucune dimension temporelle : c'est une carte
-- d'identité du territoire, pas une série.
--
-- ⚠️ DEUX RÈGLES ANTI-FUITE, toutes deux appliquées ici.
--
-- 1. MILLÉSIME CORINE = 2006, le plus ancien de la période modélisée.
--    Le profil sert à classer des communes pour TOUTES les années 2006-2025.
--    Prendre CLC 2018 ferait décrire une commune de 2006 par son occupation
--    du sol de 2018 — techniquement du futur. CLC 2006 est ≤ à toutes les
--    dates du jeu, donc sûr par construction.
--    (Les features `part_*` de la matrice, elles, suivent bien le millésime
--     de chaque ligne : c'est le LATERAL JOIN de 50_matrice.sql.)
--
-- 2. CLIMATOLOGIE FWI CALCULÉE SUR 2006-2019 SEULEMENT — la période train.
--    Une moyenne sur 2006-2025 donnerait à une commune un profil qui dépend
--    de la météo de 2023, inconnue au moment de prédire 2023. Le FWI n'est
--    pas `y`, mais une statistique NON DATÉE ne peut regarder que le train.
--
-- Aucun agrégat de `y` ici : le profil ne sait pas où le feu est tombé.
-- C'est volontaire — la sinistralité entre plus tard, par le lissage.

SELECT
  c.code_insee,
  c.dep_code,

  -- ── végétation, millésime 2006 ────────────────────────────────────
  clc.part_feuillus,
  clc.part_coniferes,
  clc.part_melangees,
  clc.part_landes,
  clc.part_maquis,
  clc.part_veg_clairsemee,
  clc.part_combustible,
  clc.part_agricole,
  clc.part_artificialise,

  -- ── relief et forme du territoire ─────────────────────────────────
  c.altitude_moy,
  (c.altitude_max - c.altitude_min)      AS amplitude_altitude,
  ln(c.superficie_km2 + 1)               AS log_superficie,
  c.distance_cote_km,

  -- ── présence humaine ──────────────────────────────────────────────
  ln(COALESCE(c.densite, 0) + 1)         AS log_densite,
  c.grille_densite,

  -- ── position ──────────────────────────────────────────────────────
  -- lat/lon sont INCLUS volontairement, mais pondérés faiblement à
  -- l'étape de standardisation : sans eux le clustering ignorerait
  -- totalement la contiguïté et produirait des « clusters » éclatés
  -- d'un bout à l'autre du pays.
  c.lat,
  c.lon,

  -- ── climatologie du feu, période TRAIN uniquement ─────────────────
  m.fwi_moyen,
  m.fwi_p90,
  m.jours_fwi_sup_21    -- seuil EFFIS « danger élevé », moyenne annuelle

FROM ref_commune c

-- occupation du sol 2006
JOIN clc_part clc
  ON clc.code_insee = c.code_insee AND clc.millesime = 2006

-- climatologie de la maille météo de la commune, 2006-2019
JOIN (
  SELECT cell_id,
         avg(fwi)                                            AS fwi_moyen,
         percentile_cont(0.9) WITHIN GROUP (ORDER BY fwi)    AS fwi_p90,
         count(*) FILTER (WHERE fwi > 21.3) / 14.0           AS jours_fwi_sup_21
  FROM fait_meteo
  WHERE date BETWEEN '2006-01-01' AND '2019-12-31'   -- ⚠️ train seul
  GROUP BY cell_id
) m ON m.cell_id = c.cell_id

WHERE c.in_perimetre
ORDER BY c.code_insee;
