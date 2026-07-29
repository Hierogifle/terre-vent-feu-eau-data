-- ═══════════════════════════════════════════════════════════════════════
-- Terre, Vent, Feu, Eau, Data — schéma de la base consolidée
--
-- Trois familles de tables, et la frontière entre elles est la garantie
-- anti-fuite du projet :
--   ref_*    référentiels stables      (commune, maille, calendrier, voisinage)
--   fait_*   faits bruts observés      (feu, météo, occupation du sol)
--   grille   cible commune × jour      (y, nb_feux, split)
--   feat_*   features de passé STRICT  (lags temporels et spatiaux)
--
-- ⚠️ Ce qui n'entre JAMAIS ici : toute colonne issue d'un .fit() ou agrégeant
--    y (lissage bayésien, target encoding, HDBSCAN, StandardScaler). Ces
--    artefacts vivent dans data/processed/learned/, calculés APRÈS le split.
-- ═══════════════════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS postgis;

DROP TABLE IF EXISTS feat_lags CASCADE;
DROP TABLE IF EXISTS grille CASCADE;
DROP TABLE IF EXISTS fait_clc CASCADE;
DROP TABLE IF EXISTS fait_feu CASCADE;
DROP TABLE IF EXISTS fait_meteo CASCADE;
DROP TABLE IF EXISTS ref_voisinage CASCADE;
DROP TABLE IF EXISTS ref_passage_cog CASCADE;
DROP TABLE IF EXISTS ref_calendrier CASCADE;
DROP TABLE IF EXISTS ref_commune CASCADE;
DROP TABLE IF EXISTS ref_maille CASCADE;

-- ─────────────────────────────────────────────────────────────────────
-- RÉFÉRENTIELS
-- ─────────────────────────────────────────────────────────────────────

-- Maille météo CEMS 0,25°. 2 360 cellules dont 410 en mer (NaN permanents).
CREATE TABLE ref_maille (
    cell_id  integer PRIMARY KEY,
    ilat     smallint NOT NULL,
    ilon     smallint NOT NULL,
    lat      double precision NOT NULL,
    lon      double precision NOT NULL,
    est_mer  boolean NOT NULL
);

-- Référentiel communal FIXE en COG 2026. Les codes historiques de la BDIFF
-- y sont remappés (voir ref_passage_cog) : on modélise les entités
-- d'aujourd'hui, rétroactivement.
CREATE TABLE ref_commune (
    code_insee       char(5) PRIMARY KEY,
    nom              text NOT NULL,
    dep_code         varchar(3) NOT NULL,   -- ⚠️ TEXTE : '2A', '2B', '01'
    reg_code         varchar(2),
    lat              double precision NOT NULL,
    lon              double precision NOT NULL,
    -- rempli après le COPY (voir charger_ref.py) : le point dérive toujours
    -- de lat/lon, il n'y a donc jamais deux sources de vérité
    geom             geography(Point, 4326),
    superficie_km2   double precision,      -- 0,37 % manquants
    population       integer,
    densite          double precision,
    altitude_moy     smallint,
    altitude_min     smallint,
    altitude_max     smallint,
    grille_densite   smallint,              -- typologie INSEE 7 niveaux, ordinale
    distance_cote_km double precision,
    cell_id          integer NOT NULL REFERENCES ref_maille(cell_id),
    in_perimetre     boolean NOT NULL DEFAULT true
);
CREATE INDEX ix_commune_geom ON ref_commune USING GIST (geom);
CREATE INDEX ix_commune_cell ON ref_commune (cell_id);
CREATE INDEX ix_commune_dep  ON ref_commune (dep_code);

-- Calendrier : tout est row-local, donc aucun risque de fuite.
CREATE TABLE ref_calendrier (
    date            date PRIMARY KEY,
    annee           smallint NOT NULL,
    mois            smallint NOT NULL,
    doy             smallint NOT NULL,
    jour_semaine    smallint NOT NULL,      -- 0 = lundi
    est_weekend     boolean NOT NULL,
    est_ferie       boolean NOT NULL,
    nom_ferie       text,
    est_14_juillet  boolean NOT NULL,
    est_15_aout     boolean NOT NULL,
    vacances        boolean NOT NULL DEFAULT false,
    sin_doy         real NOT NULL,
    cos_doy         real NOT NULL,
    sin_mois        real NOT NULL,
    cos_mois        real NOT NULL
);
CREATE INDEX ix_cal_annee ON ref_calendrier (annee);

-- Table de passage COG : ancien code -> code 2026, résolu transitivement
-- depuis le fichier officiel INSEE v_mvt_commune_2026.csv.
CREATE TABLE ref_passage_cog (
    code_avant char(5) PRIMARY KEY,
    code_apres char(5) NOT NULL REFERENCES ref_commune(code_insee),
    source     text NOT NULL DEFAULT 'INSEE v_mvt_commune_2026'
);

-- Voisinage spatial, calculé une fois par scipy.cKDTree (~19 M paires à 50 km).
CREATE TABLE ref_voisinage (
    code_insee char(5) NOT NULL,
    voisin     char(5) NOT NULL,
    dist_m     real NOT NULL,
    PRIMARY KEY (code_insee, voisin)
);
-- ⚠️ index sur `voisin` : c'est le SENS de jointure des features de voisinage
-- (on part des feux du voisin pour les diffuser vers la commune cible).
CREATE INDEX ix_vois_voisin ON ref_voisinage (voisin);

-- ─────────────────────────────────────────────────────────────────────
-- FAITS
-- ─────────────────────────────────────────────────────────────────────

-- Indices feu CEMS au grain MAILLE, jamais dénormalisés vers la commune :
-- ~31 communes partagent la même cellule, ce serait 31x la même valeur.
CREATE TABLE fait_meteo (
    cell_id integer NOT NULL REFERENCES ref_maille(cell_id),
    date    date NOT NULL,
    fwi     real, ffmc real, dmc real, dc  real,
    bui     real, isi  real, kbdi real, erc real,
    PRIMARY KEY (cell_id, date)
);
CREATE INDEX ix_meteo_date ON fait_meteo (date);

-- Un feu = une ligne. ⚠️ Un incendie traversant 5 communes produit 5 lignes :
-- c'est correct pour une grille commune × jour, mais « nombre de feux » n'est
-- donc pas « nombre d'incendies » au sens des pompiers.
CREATE TABLE fait_feu (
    feu_id            bigserial PRIMARY KEY,
    code_insee_source char(5) NOT NULL,     -- tel quel dans la BDIFF
    code_insee        char(5),              -- remappé COG 2026, NULL si irrécupérable
    dep_code          varchar(3) NOT NULL,
    ts_alerte         timestamp NOT NULL,   -- l'heure est réelle à 99,8 %
    date_alerte       date NOT NULL,
    heure_alerte      smallint,
    surface_m2        double precision,
    surface_foret_m2  double precision,
    surface_maquis_m2 double precision,
    nature            text,                 -- 64 % manquant
    type_peuplement   text,                 -- 45 % manquant
    src_zip           text NOT NULL,
    doublon_suspect   boolean NOT NULL DEFAULT false
);
CREATE INDEX ix_feu_commune_date ON fait_feu (code_insee, date_alerte);
CREATE INDEX ix_feu_date         ON fait_feu (date_alerte);

-- CORINE en format LONG plutôt que 44 colonnes : ajouter un poste ne change
-- pas le schéma, et les vues dérivées restent lisibles.
CREATE TABLE fait_clc (
    code_insee char(5) NOT NULL,
    millesime  smallint NOT NULL,
    base       text NOT NULL,               -- 'CLC 2018' — traçabilité de la version
    poste      varchar(8) NOT NULL,
    surface_ha double precision NOT NULL,
    PRIMARY KEY (code_insee, millesime, poste)
);

-- ─────────────────────────────────────────────────────────────────────
-- GRILLE — la cible, partitionnée par année
--
-- Le partitionnement n'est pas cosmétique : il rend le split temporel exact
-- par construction, permet de reconstruire une année sans toucher aux autres,
-- et évite de scanner 253 M lignes quand on n'en veut qu'une.
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE grille (
    code_insee char(5) NOT NULL,
    date       date NOT NULL,
    y          boolean NOT NULL DEFAULT false,
    nb_feux    smallint NOT NULL DEFAULT 0,   -- gardé pour le GLM Poisson
    surface_m2 double precision NOT NULL DEFAULT 0,
    u          double precision NOT NULL,     -- uniforme déterministe pour l'échantillonnage
    PRIMARY KEY (code_insee, date)
) PARTITION BY RANGE (date);
