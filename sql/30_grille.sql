-- ═══════════════════════════════════════════════════════════════════════
-- La grille commune × jour — le cœur du projet.
--
-- Produit cartésien de TOUTES les communes du périmètre par TOUS les jours,
-- puis LEFT JOIN des feux : y = 1 s'il y a eu au moins un feu, 0 sinon.
--
-- Pourquoi une grille DENSE et pas seulement les positifs : les fenêtres
-- glissantes (« feux des 30 jours précédents ») ont besoin d'une série
-- continue. Sur une série creuse, les jours absents sortent silencieusement
-- de la fenêtre et le résultat est FAUX sans que rien ne le signale.
--
-- Appelé année par année depuis charger_grille.py : en une seule transaction
-- sur 253 M lignes, le WAL PostgreSQL exploserait et une erreur ferait tout
-- recommencer.
-- ═══════════════════════════════════════════════════════════════════════

INSERT INTO grille (code_insee, date, y, nb_feux, surface_m2, u)
SELECT
    c.code_insee,
    d.date,
    COALESCE(f.n, 0) > 0                       AS y,
    COALESCE(f.n, 0)                           AS nb_feux,
    COALESCE(f.surface, 0)                     AS surface_m2,
    -- Tirage DÉTERMINISTE, pas aléatoire : les échantillons obtenus à
    -- différents ratios sont emboîtés, donc l'étude de sensibilité au ratio
    -- est propre et reproductible sans stocker de graine.
    (abs(hashtextextended(c.code_insee || '|' || d.date::text, 42)) % 1000000)::double precision
        / 1000000.0                            AS u
FROM ref_commune c
CROSS JOIN (
    SELECT date FROM ref_calendrier WHERE annee = :annee
) d
LEFT JOIN (
    SELECT code_insee, date_alerte, count(*) AS n, sum(COALESCE(surface_m2, 0)) AS surface
    FROM fait_feu
    WHERE code_insee IS NOT NULL
      AND date_alerte >= make_date(:annee, 1, 1)
      AND date_alerte <  make_date(:annee + 1, 1, 1)
    GROUP BY 1, 2
) f ON f.code_insee = c.code_insee AND f.date_alerte = d.date
WHERE c.in_perimetre;
