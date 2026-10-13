# Independent re-implementation of the UIS/GEM out-of-school rate model (SDG 4.1.4)

**Status: method demonstration. Not official estimates.** The numbers here come from an independent
re-implementation of the Bayesian cohort model in *Dharamshi, Barakat, Antoninis, Montoya (2022), "A Bayesian
Cohort Model for Estimating SDG Indicator 4.1.4"*. They differ from the official UIS modelled series and must not be
cited as such.

## What was done

| | |
|---|---|
| Model | Same equations as the paper (Appendix B): probit-space cohort model, dual likelihood (administrative data in real space, surveys in probit space), survey bias, shared variance hierarchy |
| Software | NumPyro/NUTS (the paper used Stan); `D. Code/.../Python/oosc_model_numpyro.py`, Stan file kept as reference |
| Population fitted | Both sexes (MF), 213 countries, 2000-2022 |
| Data | Administrative enrolment (UIS internal extract) 37,060 observations; surveys 6,028 (DHS 2,073, MICS 2,281, IPUMS census 1,674); population from WPP 2024, Eurostat, SingStat |
| Sampling | 4 chains, 2,000 warm-up + 2,000 draws, thinned to 1,000, target acceptance 0.99; about 9 hours on one machine |
| Convergence | No divergences. Global scale parameters R-hat <= 1.006. Worst R-hat 1.022 (spline coefficients); 251 of 122,953 diagnostics above 1.01; minimum effective sample size 418. Strictly short of the 1.01 target |

## Results

* `figures/fig1_cohort_panels.png` - estimates and observations for four school-entry cohorts in four countries. The
  model reproduces the paper's qualitative patterns, e.g. the U-shaped cohort curve for Burkina Faso and near-zero rates
  for Sweden.
* `figures/fig2_regional_stack.png` - out-of-school population by SDG region (paper Fig. 7 analogue).
  `table_out_of_school_millions_by_region.csv` has the values (global 433M in 2000, 300M in 2022).
* `figures/fig3_vs_official.png` - our posterior medians against the official UIS modelled series, by education level.

### Agreement with the official UIS modelled series (14,209 country-year-levels)

Ten countries are excluded from the benchmark because their administrative and survey data conflict so strongly
that the official series is not a fair reference: COD, CYP, HTI, PAK, AFG, CHN, JOR, MKD, UGA, RUS
(`excluded_from_benchmark.csv`). They remain in the fitted model.

| | Correlation | Mean abs. error | Median bias | Within 5 points |
|---|---|---|---|---|
| Primary age | 0.96 | 2.4 pts | +0.5 | 87% |
| Lower-secondary age | 0.95 | 2.7 pts | +0.3 | 86% |
| Upper-secondary age | 0.93 | 4.5 pts | +0.7 | 75% |
| All levels | 0.95 | 3.2 pts | +0.5 | 83% |

Global counts for the included countries are within 2% of the official series in 2000 but about 10-20% higher in
later years. India alone explains about 45% of the 2010 gap (its administrative upper-secondary rate is 55% against
about 21% from surveys, and the model follows the administrative data).

### Out-of-sample validation (paper Table 2 analogue)

A random 20% of each country's observations is held out and the country is refit on the rest. Global parameters are
held at the full-run values (see limitations). 8,614 held-out observations.

| Variant | MSE | aMSE | MAE | aMAE | Cov. 80% | Cov. 90% | Cov. 95% |
|---|---|---|---|---|---|---|---|
| **Model (ours)** | 0.0071 | 0.0066 | 0.0500 | 0.0462 | 0.87 | 0.92 | 0.95 |
| NoBias (ours) | 0.0082 | 0.0077 | 0.0548 | 0.0510 | 0.85 | 0.91 | 0.94 |
| Simple (ours) | 0.0152 | 0.0149 | 0.0754 | 0.0716 | 0.83 | 0.91 | 0.95 |
| *Paper: Model* | *0.0051* | *0.0046* | *0.0435* | *0.0389* | *0.88* | *0.93* | *0.95* |
| *Paper: NoBias* | *0.0057* | *0.0052* | *0.0459* | *0.0413* | *0.88* | *0.93* | *0.95* |
| *Paper: Simple* | *0.0072* | *0.0069* | *0.0533* | *0.0487* | *0.75* | *0.80* | *0.82* |

The ordering Model < NoBias < Simple matches the paper and uncertainty is calibrated, but our errors are about 40%
higher than the paper's. The 'Simple' benchmark is not comparable (its specification is not given in the paper).

## Deviations and limitations - please read

1. **Data are not identical to the paper's.** The paper's administrative cleaning (Appendix A.2.1) is only partly
   specified; we implement steps 1-2 and treat zero school-age enrolment as missing. We cover 212 administrative
   countries against the paper's 187. Secondary-level administrative rates look incomplete in several countries
   (India, Pakistan, Uganda, ...), probably missing vocational or open-school enrolment.
2. **WIDE ("Other" surveys, 1.5% of the paper's data) enters approximately.** Only level-aggregate rates are available,
   not single-age rates with clustered-Jackknife errors. They were added by importance-reweighting the saved draws with a
   level-aggregated likelihood (`reweight_wide.py`). Effect is under 1 point on average.
3. **Country patches are not a full refit.** Ten countries whose data changed (ALB, CHN, COD, COG, MDG, MHL, MKD, SYC,
   TCD, ZAF) were refit alone with the global parameters fixed at their posterior medians (`refit_countries.py`;
   validated against the full run on ARG and TCD to 0.1 points). Global uncertainty is not propagated.
4. **Validation shortcuts.** Held-out data also informed the global parameters, and NoBias cannot re-balance its variance
   terms. The reduced models MuRW and MuK5 and the sensitivity runs of Table 3 were not reproduced.
5. **Unspecified modelling choices** (spline degree and boundary knots, the pre-2000 extension of the random walk, the
   year-2000 anchor) follow `B. Documentation/4. Model Notes.md`; they could not be checked against UIS code.
6. **Only the both-sexes population was fitted;** the paper also fits males and females separately.

## Issues found in the source paper

* The prior on the shared error scale is stated as half-normal(0, 0.01^2), but the reported error sizes (Fig. 8, median 0.0725)
  imply values four to six prior standard deviations away; our posterior is 0.05. Probably a typo or a very tight prior
  overridden by the data.
* The model uses sqrt(sigma^2 + omega^2) as the administrative error, while Fig. 8 and the 0.0725 median use the sum
  sigma + omega; the two differ by about 40% on our fit (0.060 against 0.084).
* Section 3.5 says coverage is evaluated on left-out data; Section 4.4 and Table 3 report it "for the entire sample".
* Table 2/3 differences are 1-2% with no standard errors, at the noise floor (RMSE 0.071 against a median data error of
  0.0725). The text also says the model is "in-line or better on MSE" (MuK5 is better) and that the sensitivity
  analysis validates the 5-year knot interval (the 8-year interval scores slightly better).
* The probit-scale error of a survey observation grows as its rate falls (SD about 1.5 below 5%, 2.7 at 1%), so surveys
  with low out-of-school rates carry almost no information. Not discussed.
* "Coverage results in Figure 2" in Section 4.4 should read Table 2.

None of these suggests fabricated results; they are specification, reporting and validation weaknesses.

## Files

| Path | Content |
|---|---|
| `figures/` | Figures 1-3 |
| `table_validation_holdout.csv` | Validation table above |
| `table_out_of_school_millions_by_region.csv` | Regional and global counts |
| `excluded_from_benchmark.csv` | The ten countries left out of the official-series comparison |
| `A. Data/A.2. Processed/model_output/MF/estimates_final.csv` | Country x year x age posterior median and 90% interval |
| `.../MF/estimates.csv`, `estimates_patched.csv`, `estimates_with_wide.csv` | Intermediate versions |
| `D. Code/.../Python/` | `run_full_model.py`, `reweight_wide.py`, `refit_countries.py`, `holdout_validation.py` |
