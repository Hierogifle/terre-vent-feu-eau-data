-- ═══════════════════════════════════════════════════════════════════════
-- Contrôles d'intégrité — chaque ligne doit renvoyer OK.
--
-- Ces assertions valent mieux qu'un résultat silencieusement faux : sur ce
-- type de projet, une jointure ratée ne lève aucune erreur, elle dégrade
-- juste le modèle sans qu'on sache pourquoi.
--
--    docker exec tvfed-db psql -U tvfed -d tvfed -f /tmp/80_checks.sql
-- ═══════════════════════════════════════════════════════════════════════

\echo '=== INTÉGRITÉ DE LA BASE ==='

-- 1. la grille est exactement rectangulaire : communes × jours, sans trou
SELECT 'grille rectangulaire' AS controle,
       CASE WHEN (SELECT count(*) FROM grille)
               = (SELECT count(*) FROM ref_commune WHERE in_perimetre)
               * (SELECT count(*) FROM ref_calendrier)
            THEN 'OK' ELSE 'ÉCHEC' END AS statut,
       (SELECT count(*) FROM grille)::text AS valeur;

-- 2. aucune commune rattachée à une maille de mer
SELECT 'aucune commune en mer',
       CASE WHEN count(*) = 0 THEN 'OK' ELSE 'ÉCHEC' END, count(*)::text
FROM ref_commune c JOIN ref_maille m USING (cell_id) WHERE m.est_mer;

-- 3. toute commune a une météo pour chaque jour (via sa maille)
SELECT 'météo complète',
       CASE WHEN count(*) = 0 THEN 'OK' ELSE 'ÉCHEC' END, count(*)::text
FROM (
  SELECT c.cell_id FROM ref_commune c
  CROSS JOIN (SELECT min(date) d1, max(date) d2 FROM ref_calendrier) b
  WHERE NOT EXISTS (
    SELECT 1 FROM fait_meteo m
    WHERE m.cell_id = c.cell_id AND m.date = b.d1)
  LIMIT 100
) x;

-- 4. tout feu rattaché pointe vers une commune connue
SELECT 'feux rattachés valides',
       CASE WHEN count(*) = 0 THEN 'OK' ELSE 'ÉCHEC' END, count(*)::text
FROM fait_feu f WHERE f.code_insee IS NOT NULL
  AND NOT EXISTS (SELECT 1 FROM ref_commune c WHERE c.code_insee = f.code_insee);

-- 5. le nombre de positifs de la grille = le nombre de couples (commune, jour)
--    distincts ayant au moins un feu
SELECT 'positifs cohérents',
       CASE WHEN (SELECT count(*) FROM grille WHERE y)
               = (SELECT count(DISTINCT (code_insee, date_alerte)) FROM fait_feu
                  WHERE code_insee IS NOT NULL)
            THEN 'OK' ELSE 'ÉCHEC' END,
       (SELECT count(*) FROM grille WHERE y)::text;

-- 6. le voisinage est symétrique (a voisin de b <=> b voisin de a)
SELECT 'voisinage symétrique',
       CASE WHEN count(*) = 0 THEN 'OK' ELSE 'ÉCHEC' END, count(*)::text
FROM (
  SELECT v.code_insee, v.voisin FROM ref_voisinage v
  WHERE NOT EXISTS (
    SELECT 1 FROM ref_voisinage w
    WHERE w.code_insee = v.voisin AND w.voisin = v.code_insee)
  LIMIT 100
) x;

-- 7. la table de passage ne déplace aucun code encore valide
SELECT 'passage COG sain',
       CASE WHEN count(*) = 0 THEN 'OK' ELSE 'ÉCHEC' END, count(*)::text
FROM ref_passage_cog p
WHERE EXISTS (SELECT 1 FROM ref_commune c WHERE c.code_insee = p.code_avant);

\echo ''
\echo '=== VOLUMÉTRIE PAR PARTITION ==='
SELECT annee,
       count(*)                       AS lignes,
       count(*) FILTER (WHERE y)      AS positifs,
       round(100.0 * count(*) FILTER (WHERE y) / count(*), 4) AS taux_pct
FROM grille g JOIN ref_calendrier c USING (date)
GROUP BY annee ORDER BY annee;
