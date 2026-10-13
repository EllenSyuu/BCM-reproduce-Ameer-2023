/* =====================================================================
   Missing-data queries for faithful reproduction of Paper 1
   Purpose: find sex-specific population, SDG/M49 regions, and the exact
            raw enrolment indicators needed before Appendix A.2.1 cleaning.
   Database: UISProd / schema dbasisdess
   Run each section separately in SSMS and export only the result sets
   explicitly marked EXPORT.
   ===================================================================== */

USE UISProd;
SET NOCOUNT ON;

/* ---------------------------------------------------------------------
   1. POPULATION: inspect candidate internal UIS population tables.
   The paper requires country x year x single age x sex.
   --------------------------------------------------------------------- */
SELECT TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION, COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'dbasisdess'
  AND TABLE_NAME IN ('POPULATION', 'POPULATION_ALT', 'POPULATION_PRE_WPP2024',
                     'POPULATION_UISTEST', 'EDU_POPULATION')
ORDER BY TABLE_NAME, ORDINAL_POSITION;

/* Inspect the table that contains country, age, sex, and population fields. */
SELECT TOP 100 *
FROM dbasisdess.POPULATION;

/* Candidate population volume check. */
SELECT
    COUNT(*) AS n_rows,
    COUNT(DISTINCT CO_CODE) AS n_countries,
    MIN(EMCO_YEAR) AS min_year,
    MAX(EMCO_YEAR) AS max_year,
    MIN(EAG_AGE) AS min_age,
    MAX(EAG_AGE) AS max_age
FROM dbasisdess.POPULATION
WHERE EMCO_YEAR BETWEEN 2000 AND 2022;

/* ---------------------------------------------------------------------
   2. COUNTRY -> SDG/M49 REGION: inspect available group/reference tables.
   --------------------------------------------------------------------- */
SELECT TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION, COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE (TABLE_SCHEMA = 'dbasisdess' AND TABLE_NAME IN
       ('COUNTRY_COUNTRY_GROUP', 'COUNTRY_GROUP_REF',
        'COUNTRY_GROUP_REF_ALIAS', 'UNSD_CO_CODE_vs_UIS_CO_CODE'))
   OR (TABLE_SCHEMA = 'dbo' AND TABLE_NAME LIKE '%REGION%')
ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION;

SELECT TOP 100 *
FROM dbasisdess.COUNTRY_COUNTRY_GROUP;

SELECT TOP 100 *
FROM dbasisdess.COUNTRY_GROUP_REF;

/* ---------------------------------------------------------------------
   3. ENROLMENT: exact raw rows with age, sex, grade, indicator, and value.
   Do not filter on English descriptions until the indicator inventory has
   been reviewed; UIS descriptions are multilingual and may be aggregates.
   --------------------------------------------------------------------- */
SELECT
    o.CO_CODE,
    o.EMCO_YEAR AS year,
    TRY_CONVERT(int, age.EC_AS_CONT) AS age,
    cl.EC_SX_ID AS sex,
    cl.EC_GR_ID AS grade,
    o.EMC_ID,
    cl.EMC_DESS_CONT AS indicator_desc,
    o.EM_FIG AS value,
    o.MQ_ID AS qualifier,
    o.MG_ID AS magnitude,
    o.ST_ID,
    o.ABE_ID,
    o.EC_TD_ID
FROM dbasisdess.EDU_METER97_OBS o
JOIN dbasisdess.EDU_METER_CL97 cl
  ON cl.EMC_ID = o.EMC_ID
 AND cl.EC_TD_ID = o.EC_TD_ID
JOIN dbasisdess.EC_AGE_START age
  ON age.EC_AS_ID = cl.EC_AS_ID
WHERE o.EMCO_YEAR BETWEEN 2000 AND 2022
  AND TRY_CONVERT(int, age.EC_AS_CONT) BETWEEN 0 AND 98
ORDER BY o.CO_CODE, o.EMCO_YEAR, age.EC_AS_CONT, cl.EC_SX_ID, o.EMC_ID;

/* Indicator inventory: use this to identify the exact pupil/enrolment
   EMC_IDs before exporting the final restricted extract. */
SELECT
    o.EMC_ID,
    cl.EMC_DESS_CONT AS indicator_desc,
    cl.EC_SX_ID AS sex,
    cl.EC_GR_ID AS grade,
    COUNT(*) AS n_rows,
    MIN(o.EMCO_YEAR) AS min_year,
    MAX(o.EMCO_YEAR) AS max_year,
    SUM(CASE WHEN o.EM_FIG < 0 THEN 1 ELSE 0 END) AS n_negative_values
FROM dbasisdess.EDU_METER97_OBS o
JOIN dbasisdess.EDU_METER_CL97 cl
  ON cl.EMC_ID = o.EMC_ID
 AND cl.EC_TD_ID = o.EC_TD_ID
JOIN dbasisdess.EC_AGE_START age
  ON age.EC_AS_ID = cl.EC_AS_ID
WHERE o.EMCO_YEAR BETWEEN 2000 AND 2022
  AND TRY_CONVERT(int, age.EC_AS_CONT) BETWEEN 0 AND 98
GROUP BY o.EMC_ID, cl.EMC_DESS_CONT, cl.EC_SX_ID, cl.EC_GR_ID
ORDER BY n_negative_values DESC, n_rows DESC;

/* ---------------------------------------------------------------------
   4. APPENDIX A.2.1 diagnostics: retain negative values and expose all
   flags needed to reproduce the three-step exclusion in Python.
   --------------------------------------------------------------------- */
SELECT
    COUNT(*) AS n_rows,
    SUM(CASE WHEN o.EM_FIG < -0.1 THEN 1 ELSE 0 END) AS below_minus_0_1,
    SUM(CASE WHEN o.EM_FIG < 0 THEN 1 ELSE 0 END) AS negative_values,
    SUM(CASE WHEN o.EM_FIG IS NULL THEN 1 ELSE 0 END) AS missing_values,
    MIN(o.EM_FIG) AS min_value,
    MAX(o.EM_FIG) AS max_value
FROM dbasisdess.EDU_METER97_OBS o
JOIN dbasisdess.EDU_METER_CL97 cl
  ON cl.EMC_ID = o.EMC_ID
 AND cl.EC_TD_ID = o.EC_TD_ID
JOIN dbasisdess.EC_AGE_START age
  ON age.EC_AS_ID = cl.EC_AS_ID
WHERE o.EMCO_YEAR BETWEEN 2000 AND 2022
  AND TRY_CONVERT(int, age.EC_AS_CONT) BETWEEN 0 AND 98;