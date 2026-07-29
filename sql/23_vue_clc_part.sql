-- ═══════════════════════════════════════════════════════════════════════
-- Occupation du sol en PART de la surface communale, jamais en hectares.
--
-- En hectares absolus, un poste encode surtout la TAILLE de la commune :
-- une grande commune a beaucoup de tout. Le ratio est row-local, donc sûr,
-- et c'est lui qui porte l'information « de quoi cette commune est faite ».
-- ═══════════════════════════════════════════════════════════════════════

DROP VIEW IF EXISTS clc_part;

CREATE VIEW clc_part AS
SELECT
    code_insee,
    millesime,
    sum(surface_ha)                                              AS surface_tot_ha,

    -- forêts : les trois postes 311/312/313
    -- ⚠️ COALESCE partout : un poste absent d'une commune n'est pas une donnée
    -- manquante, c'est un zéro. Sans lui, `sum(...) FILTER (...)` renvoie NULL
    -- et 92 % des `part_maquis` remontaient en NaN au lieu de 0.
    COALESCE(sum(surface_ha) FILTER (WHERE poste IN ('CLC_311','CLC_312','CLC_313')), 0)
        / nullif(sum(surface_ha), 0)                             AS part_foret,
    COALESCE(sum(surface_ha) FILTER (WHERE poste = 'CLC_311'), 0)
        / nullif(sum(surface_ha), 0)                             AS part_feuillus,
    COALESCE(sum(surface_ha) FILTER (WHERE poste = 'CLC_312'), 0)
        / nullif(sum(surface_ha), 0)                             AS part_coniferes,
    COALESCE(sum(surface_ha) FILTER (WHERE poste = 'CLC_313'), 0)
        / nullif(sum(surface_ha), 0)                             AS part_melangees,

    -- végétation basse : landes, et surtout le maquis méditerranéen
    COALESCE(sum(surface_ha) FILTER (WHERE poste = 'CLC_322'), 0)
        / nullif(sum(surface_ha), 0)                             AS part_landes,
    COALESCE(sum(surface_ha) FILTER (WHERE poste = 'CLC_323'), 0)
        / nullif(sum(surface_ha), 0)                             AS part_maquis,
    COALESCE(sum(surface_ha) FILTER (WHERE poste = 'CLC_324'), 0)
        / nullif(sum(surface_ha), 0)                             AS part_veg_mutation,
    COALESCE(sum(surface_ha) FILTER (WHERE poste = 'CLC_333'), 0)
        / nullif(sum(surface_ha), 0)                             AS part_veg_clairsemee,

    -- combustible total = tous les postes 3xx
    COALESCE(sum(surface_ha) FILTER (WHERE poste LIKE 'CLC_3%'), 0)
        / nullif(sum(surface_ha), 0)                             AS part_combustible,

    -- contexte
    COALESCE(sum(surface_ha) FILTER (WHERE poste LIKE 'CLC_2%'), 0)
        / nullif(sum(surface_ha), 0)                             AS part_agricole,
    COALESCE(sum(surface_ha) FILTER (WHERE poste LIKE 'CLC_1%'), 0)
        / nullif(sum(surface_ha), 0)                             AS part_artificialise
FROM fait_clc
GROUP BY code_insee, millesime;
