# Bayesian Reproduction of SDG 4.1.4 Out-of-School Rates

An independent, non-official reproduction of the Bayesian cohort model described in the published SDG 4.1.4 out-of-school-rate paper.

This repository makes the work auditable: the model translation, data contracts, harmonisation logic, validation plan, and known deviations are documented separately. It is not the original UNESCO/UIS/GEMR implementation and is not endorsed by UNESCO, UIS, GEMR, or the paper authors.

## Current status — 13 October 2026

| Workstream | Status | Evidence / boundary |
|---|---|---|
| Project scaffold and environment | Complete | Repository structure and dependencies are documented. |
| Population denominators | Implemented and checked | Public single-age sources are wired into the acquisition path. |
| Administrative enrolment input | Blocked | The raw age-by-sex enrolment extract required by the paper is restricted and excluded. |
| Survey/census harmonisation | Partially implemented | Source-specific harmonisers and a common observation contract exist; access and coverage vary. |
| Cohort model translation | Implemented on synthetic data | A Stan equation reference and a NumPyro/JAX execution path are maintained. |
| Full real-data reproduction | Not complete | It requires the restricted administrative input and final source-vintage decisions. |
| Validation and paper comparison | Planned / partial | The validation design exists; no final real-data equivalence claim is made. |

**Bottom line:** the computational pipeline is substantially developed, but the full empirical reproduction is not complete.

## What this project demonstrates

- Translation of a multi-equation cohort model into executable probabilistic code.
- Explicit mapping from paper quantities to country-year-age-sex data structures.
- Data-source triage based on coverage, provenance, access conditions, and model compatibility.
- Harmonisation of administrative-style and survey observations into a common model contract.
- Uncertainty handling using source-appropriate survey design information where available.
- Synthetic end-to-end checks for exclusion rules, spline bases, parameter constraints, and sampling.
- Transparent documentation of deviations instead of presenting a partial reproduction as an exact one.

## Data-protection boundary

The public repository contains code, method notes, synthetic checks, and high-level data documentation. It must not contain restricted microdata, raw administrative extracts, credentials, cookies, tokens, personal information, unpublished outputs, or copyrighted reference PDFs without redistribution permission. Authorised local users may place permitted inputs in the ignored data directories and run the documented pipeline without publishing those files.

## Repository map

```text
A. Data/                         Local data area; raw/processed data are ignored.
B. Documentation/                Public method notes and project status.
C. Reference Papers/             Local reference material; not for redistribution.
D. Code/                         Acquisition, harmonisation, model assembly, fitting.
PUBLICATION_NOTES.md             Public-release and privacy checklist.
requirements.txt                 Python dependencies.
```

## Reproduction workflow

1. Acquire only data for which you have permission and retain it locally.
2. Build public population denominators.
3. Convert permitted administrative or survey inputs to the documented observation contract.
4. Apply exclusions and uncertainty calculations.
5. Assemble model input by sex population and region.
6. Fit the NumPyro/JAX model, or inspect the Stan reference implementation.
7. Save diagnostics and derived summaries locally; publish only approved aggregates or synthetic examples.

Do not infer a successful full reproduction from a successful synthetic smoke test.

## Known limitations and deviations

- Public population revisions may differ from the paper's source vintage.
- Survey sources may be registration-controlled, licensed, or available only as aggregates.
- Aggregate education-level observations cannot automatically substitute for single-age observations.
- Exact spline conventions, exclusion decisions, and regional classifications require explicit validation.
- A full fit and paper-level comparison remain conditional on authorised administrative inputs.

## Update standard

Every substantive update should state what changed, which input or assumption it affects, how it was checked, and whether the result is synthetic, public-data-only, or restricted-data-only. See `PUBLICATION_NOTES.md` before pushing to a public repository.
