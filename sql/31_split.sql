-- ═══════════════════════════════════════════════════════════════════════
-- LA BARRIÈRE — le split temporel.
--
-- Tout ce qui suit dans le pipeline doit respecter une règle :
--
--    Une feature DATÉE peut regarder tout le passé, y compris celui de sa
--    propre période d'évaluation. Une statistique NON DATÉE ne peut regarder
--    que le train.
--
-- Exemple : « feux des voisins dans les 30 jours précédents » pour le
-- 3 août 2023 lit juillet 2023 (période de test) — ce n'est PAS une fuite,
-- car le 3 août à 8 h on connaît juillet. En revanche « taux moyen de feux
-- de la commune sur toute la période » lit le futur : fuite.
--
-- La vue `echantillon` matérialise le downsampling : 100 % des positifs,
-- 10 négatifs par positif, et val/test INTÉGRAUX (la calibration en dépend).
-- ═══════════════════════════════════════════════════════════════════════

DROP VIEW IF EXISTS echantillon;
DROP VIEW IF EXISTS grille_split;

CREATE VIEW grille_split AS
SELECT g.*,
       c.annee,
       CASE
           WHEN c.annee <= 2019 THEN 'train'
           WHEN c.annee <= 2022 THEN 'val'
           ELSE                      'test'
       END AS split
FROM grille g
JOIN ref_calendrier c USING (date);

-- Ratio 1:10 sur un taux de positifs de ~0,0187 % dans le train
-- → il faut garder ~0,187 % des négatifs.
CREATE VIEW echantillon AS
SELECT *
FROM grille_split
WHERE split <> 'train'     -- val et test INTÉGRAUX : jamais échantillonnés
   OR y                    -- 100 % des positifs du train
   OR u < 0.00187;         -- négatifs du train, tirage déterministe
