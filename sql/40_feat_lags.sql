-- ═══════════════════════════════════════════════════════════════════════
-- Features d'historique — PASSÉ STRICT.
--
-- Toutes les fenêtres s'arrêtent à J-1 : le jour J lui-même n'entre jamais
-- dans le décompte. Sans cette borne, `feux_30j` contiendrait le feu qu'on
-- cherche à prédire — la fuite la plus classique du domaine.
--
-- ⚠️ Écriture par EXPANSION, pas par fenêtre glissante sur la grille.
-- Un `SUM(...) OVER (RANGE 30 DAY PRECEDING)` sur 253 M lignes serait
-- correct mais très coûteux. Les feux sont rares (52 809) : on part d'eux
-- et on projette chacun sur les jours qu'il influence. ~45 k jours-feux
-- × 365 = 16 M lignes intermédiaires au lieu de 253 M balayées.
-- ═══════════════════════════════════════════════════════════════════════

DROP TABLE IF EXISTS feat_lags;

CREATE TABLE feat_lags AS
WITH jours_feu AS (
    -- un feu = une ligne BDIFF ; plusieurs feux le même jour dans la même
    -- commune sont agrégés ici
    SELECT code_insee, date_alerte, count(*) AS n, sum(COALESCE(surface_m2, 0)) AS surf
    FROM fait_feu
    WHERE code_insee IS NOT NULL
    GROUP BY 1, 2
),
projete AS (
    -- chaque jour-feu influence les 365 jours SUIVANTS (décalage ≥ 1)
    SELECT jf.code_insee,
           jf.date_alerte + d AS date,
           jf.n, jf.surf, d AS decalage
    FROM jours_feu jf
    CROSS JOIN generate_series(1, 365) AS d
)
SELECT
    code_insee,
    date,
    sum(n)    FILTER (WHERE decalage <=   7)::int  AS feux_commune_7j,
    sum(n)    FILTER (WHERE decalage <=  30)::int  AS feux_commune_30j,
    sum(n)    FILTER (WHERE decalage <=  90)::int  AS feux_commune_90j,
    sum(n)::int                                    AS feux_commune_365j,
    sum(surf) FILTER (WHERE decalage <=  30)       AS surface_commune_30j,
    min(decalage)::int                             AS jours_depuis_dernier_feu
FROM projete
GROUP BY 1, 2;

ALTER TABLE feat_lags ADD PRIMARY KEY (code_insee, date);
