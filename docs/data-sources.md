# Data and provenance catalogue

This catalogue is written for a public reader. It describes source classes, access boundaries, and model compatibility without publishing restricted files, private contacts, credentials, or internal extracts.

## Source coverage and model role

| Source class | Model role | Public status |
|---|---|---|
| **Single-age population denominators** | Denominator for administrative and survey rates | Public references and acquisition code; local files excluded. |
| **Administrative enrolment counts** | Main likelihood component | Restricted input; not included or publishable here. |
| **Household survey microdata** | Survey likelihood and cross-checks | Access and licensing vary; only permitted local copies may be used. |
| **Census/IPUMS extracts** | Survey/census observations | Registration and extract terms apply; do not redistribute. |
| **Published aggregate survey tables** | Supplemental evidence | May be incompatible with a single-age likelihood. |
| **Official published indicators** | Validation reference | Validation target, not a substitute for raw model inputs. |

## Population denominators

The implementation supports public single-age population sources such as UN World Population Prospects, Eurostat demographic tables for relevant European coverage, and Singapore's official population table where applicable. The exact revision, download date, transformation, and country assignment must be recorded in local run metadata. A newer public revision is a documented deviation from the paper's source vintage, not an exact match.

## Survey and census sources

The paper uses multiple survey and census families. Their use is conditional on permission, paper-vintage alignment, variable availability, and the ability to calculate uncertainty compatible with the likelihood. Harmonisers target a common observation contract containing country, year, age or education-level support, sex, an observation or rate, and an uncertainty field. Weights, clusters, waves, and sample identifiers remain local.

## Restricted administrative input

The paper's central administrative component requires raw enrolment counts by country, year, single age, sex, and education level, before publication-side cleaning that would remove or cap invalid observations. Those inputs are not included here. The public project therefore claims a reproducible method and pipeline, not a complete public-data reproduction of the administrative likelihood.

## Compatibility checklist

Before admitting a source to the model, document whether it has single-age or education-level support; whether sex, country, and year definitions match; whether numerator and denominator are compatible; whether weights, clusters, and uncertainty are available; whether the source vintage matches the paper; and whether the licence permits the intended use.

