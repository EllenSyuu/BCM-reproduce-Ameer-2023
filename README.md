# GEM Bayesian Model Reproduction — Out-of-School Rate (SDG 4.1.4)

> **Important:** This is an independent, non-official reproduction attempt. It
> is not the original implementation and is not endorsed by UNESCO, UIS, GEMR,
> or the paper authors. Internal and restricted data are intentionally excluded
> from this project and must not be uploaded to a public repository. See
> `PUBLICATION_NOTES.md` before publishing.

Reproduction of the UNESCO Institute for Statistics / Global Education Monitoring Report
Bayesian cohort model for SDG indicator 4.1.4 (out-of-school rates), as specified in
`C. Reference Papers/Paper 1 - OOSC Cohort Model (4.1.4).pdf`.

A second workstream reproducing the ABCR completion-rate model (SDG 4.1.2,
`C. Reference Papers/Paper 2 - School Completion Rate ABCR (4.1.2).pdf`) will follow
the same structure once this one is running end-to-end.

## Status

Phase 0 (scaffold) done and verified on synthetic data. Phase 1 (public data) done —
WPP, Eurostat, and SingStat population data all fetched and verified live. SingStat
table M810801 provides Singapore resident population by single year of age and sex.
Phases 2+ blocked on real data (UIS
admin data request, DHS/MICS/IPUMS registration — both on the user's side). No
real-world fit yet.

## Project structure

```
A. Data/
  A.1. Raw Sources/
    1. Population Denominators/       -- WPP, Eurostat, SingStat (verified single-age)
    2. Administrative Data/           -- UIS internal single-age enrolment data.
    3. Survey Data/                   -- MICS (received, verified), DHS (pending
                                          project approval), IPUMS (census CSV extract
                                          received 2026-09-28)
    4. Validation Reference (Not Model Input)/ -- UIS's own official modelled output
                                          (ROFST.MOD, for validating our results later)
                                          and public API indicators for validation
  A.2. Processed/                     -- harmonized model-ready data (empty until
                                          Phase 2/3 unblocks)
B. Documentation/                     -- data source catalog, UIS request checklist,
                                          received-data audit, model implementation notes
C. Reference Papers/                  -- the two source papers (4.1.4 and 4.1.2)
D. Code/
  D.1. Paper 1 - OOSC Cohort Model (4.1.4)/
    Python/                            -- the model actually run in this environment
    Stan/                              -- canonical equation-level Stan implementation
  D.2. Paper 2 - ABCR Completion Rate Model (4.1.2)/ -- not started yet
README.md
requirements.txt
.venv/                                -- local Python environment
```

## Reproduction plan

### Phase 0 — Scaffold (done, verified on synthetic data)
- Project structure, model implementation, spline basis helpers, public-data fetch
  stubs, admin-data harmonization logic.
- Toolchain: Python (not R). NumPyro/JAX drives the model (not `cmdstanpy`/CmdStan —
  see Phase 4). See `requirements.txt`. A local `.venv/` holds the environment.
- Everything below marked "verified" has actually been run and checked against
  real output (live API calls, synthetic-data self-tests, or an end-to-end NUTS
  fit), not just written and assumed correct — see conversation/session history
  for the specific bugs this caught (an off-by-one in the entry-year grid, a
  logic error in the admin-data exclusion rules, a missing User-Agent header
  on the SingStat API).

### Phase 1 — Public data acquisition
- Population denominators: UN World Population Prospects (WPP), Eurostat, SingStat.
- Survey data: DHS, MICS, IPUMS-International census extracts, plus the ~16
  country-specific household surveys listed in the paper's Appendix A.1 (EPH, ILCS, EH,
  PNAD, CASEN, CFPS, ECV, ENEMDU, EHPM, ENIGH, HIES, EPHC, ENAHO, RLMS-HSE, HBS, ECH).
- Each of these needs separate registration/access terms — see
  `B. Documentation/1. Data Sources Catalog.md`.

### Phase 2 — Administrative data (BLOCKED on UIS internal access)
- Age-by-age, sex-disaggregated enrolment counts from UIS, by country-year.
- This is NOT the same as UIS.Stat's public bulk download, which only publishes
  pre-aggregated indicators (already netted against population, negatives dropped).
  We specifically need the *raw* enrolment counts and the *raw* population denominators
  used to compute out-of-school rates, so that invalid (negative) observations survive
  into the dataset — that's the paper's core contribution (Section 2.1) and the model
  cannot reproduce it if we start from pre-cleaned public indicators.
- See `B. Documentation/2. Internal UIS Data Request.md` for the exact request to make.

### Phase 3 — Data harmonization
- `D. Code/D.1. Paper 1 - OOSC Cohort Model (4.1.4)/Python/harmonize_admin_data.py` — the three-step enrolment-count exclusion process
  (Appendix A.2.1) and rate computation. **Verified on synthetic data**, including the
  step-1/step-2 exclusion logic actually firing (not just a vacuous pass — see
  the file's self-test for the specific data-shape bug this caught: enrolment
  data can carry multiple education-level rows per age, one per on-time/delayed
  population, which the exclusion rules must check per-age not per-row).
- `harmonize_ipums_census.py` — IPUMS-I census samples -> weighted single-age x sex
  out-of-school rates with delete-one-household Jackknife SEs, 1M-record cap per
  sample (Appendix A.1/A.2.2). Output: `A. Data/A.2. Processed/ipums_census_oosc_observations.csv`.
- `harmonize_dhs.py` — DHS household-member (PR) files -> same observation format,
  PSU-clustered Jackknife SEs, wave = DHS phase. Output:
  `A. Data/A.2. Processed/dhs_oosc_observations.csv`. Files fetched with
  `download_dhs.py` (reuses a logged-in browser cookie).
- `harmonize_mics.py` — MICS2-6 household-member (hl.sav) files -> same observation
  format, cluster (HH1) Jackknife SEs, wave = MICS round; national surveys only.
  Output: `A. Data/A.2. Processed/mics_oosc_observations.csv`.
- `harmonize_wide.py` — the 14 available WIDE country surveys ("Other") -> 348
  observations, paper-cited years only. WIDE files are published aggregates, so
  these are by *education level*, not single age, with binomial SEs (no design
  effect; empty where WIDE gives no sample count). They need a level-aggregated
  likelihood term before the model can use them. **Not used in the model for now**
  (single-age WIDE data unobtainable; see Data Sources Catalog). Output:
  `A. Data/A.2. Processed/wide_oosc_observations.csv`.
- Full pipeline (run in this order, from `D. Code/.../Python/`):
  1. `build_population.py` — WPP 2024 by sex, Eurostat (SDG Europe), SingStat (SGP).
  2. `prepare_admin_observations.py` — admin rates by single age x sex, Appendix
     A.2.1 exclusions, -0.1 cutoff (supersedes `prepare_total_model_input.py`).
  3. `harmonize_ipums_census.py`, `harmonize_dhs.py`, `harmonize_mics.py` — surveys.
  4. `assemble_model_input.py --sex MF|M|F` — relative age, SDG regions,
     series/waves, probit + Delta-method inputs.
  5. `run_full_model.py --tag MF|M|F` — paper MCMC settings; writes estimates,
     parameters and diagnostics to `A. Data/A.2. Processed/model_output/<tag>/`.
- Admin enrolment: SQL step 6 (2000-2012) + step 6b (2013-2022). 2000-2002
  are still thin (24-43 countries); see `B. Documentation/0. Decision Log.docx`.
  (relative age, country/series/wave indices, probit + Delta-method SEs) with clustered-Jackknife sampling
  variance (Appendix A.2.2) — not yet implemented, blocked on having real DHS/MICS
  microdata to build/test against.
- Age convention: age 1 = country's minimum entry age; ages run through late teens.

### Phase 4 — Model implementation
- `D. Code/D.1. Paper 1 - OOSC Cohort Model (4.1.4)/Stan/oosc_cohort_model.stan` —
  canonical implementation of Equations (1)-(37) (cohort process model + dual
  administrative/survey likelihood), kept as the equation-level reference.
- `D. Code/D.1. Paper 1 - OOSC Cohort Model (4.1.4)/Python/oosc_model_numpyro.py` —
  the model actually run in this environment. Same model, ported to NumPyro/JAX
  because CmdStan needs a C++ compiler and installing one (RTools) failed under
  this session's sandboxing — see `B. Documentation/4. Model Notes.md` section 8
  for the full Stan<->NumPyro correspondence and translation notes. **Verified
  working**: compiles, samples via NUTS, produces valid kappa in [0,1] on
  synthetic data (2026-09-11 smoke test).
- Needs country-specific B-spline basis matrices (annual knots for mu, 5-year knots for
  Theta/alpha) precomputed in Python and passed in as data — see the sibling
  `Python/spline_basis.py`
  (verified: partition-of-unity + entry-year-grid boundary checks pass).

### Phase 5 — Fitting
- 4 chains, run separately for male/female/total populations (paper: 4000 iter, 2000
  burn-in, thinned to 1000/chain — `run_mcmc()` in `oosc_model_numpyro.py` takes the
  same warmup/samples/chains knobs). At full scale (192 countries) this is a
  multi-hour-to-multi-day job even on NumPyro; plan for cloud/HPC compute (ideally
  with a GPU, which JAX will use automatically if available), not a laptop.

### Phase 6 — Validation & products
- Out-of-sample validation (20% leave-out per country): MSE/MAE/aMSE/aMAE, coverage.
- Reduced-model comparisons (Simple, NoBias, MuRW, MuK5) per Table 2.
- Sensitivity analysis on knot spacing and tau priors per Table 3.
- Aggregation to education level and SDG region, reproducing Figure 7 (out-of-school
  population by region).

## Known gaps vs. the paper (to resolve as we go)
- Exact spline basis family/implementation (B-spline vs P-spline parameterization,
  knot placement at boundaries) is under-specified in the paper text — we will document
  our choices in `B. Documentation/4. Model Notes.md` as we make them.
- The country-level exclusion rules for incomplete enrolment counts (Appendix A.2.1,
  three-step process) require applying judgment per country; we'll log exceptions.
