# Model implementation notes — `src/stan/oosc_cohort_model.stan`

Judgment calls made translating the paper's equations into a runnable Stan program.
Flag these for review against UIS's actual implementation if/when a reference
extract becomes available (see `2. Internal UIS Data Request.md`).

## 1. Shared entry-year grid for mu and Theta
Eq. (3): Φ⁻¹(κ_c,y,a) = μ_{c,y−a+1} + Θ_{c,y,a}. Both terms are functions of the
cohort's **entry year** (y−a+1), not of the observation year y. The paper defines
μ_c,y "generally" (Eq. 4) but only ever uses it evaluated at entry year in Eq. 3, and
Θ_c,y,a's inner spline is explicitly indexed by entry year (Eq. 10, "the spline
coefficients correspond to y−a+1"). We therefore evaluate both on one shared
entry-year grid, `N_EY` wide enough to cover the earliest entry year implied by
`min(year) - (max(age)-1)` across all observations (i.e. back to roughly 1981-1990
depending on max observed age ~19 and data starting 2000).

This means mu's "annual knot" spline (Eq. 4) is fit over a longer span than the
official 2000-2022 reporting window — it has to be, since a 19-year-old observed in
2000 entered school around 1982. The paper doesn't explicitly discuss this; it's our
inference from the model structure. Worth confirming against UIS's actual code.

## 2. Year-2000 prior on `mu_star`
CORRECTED (2026-09-14): this note previously claimed the prior wasn't specified in
Paper 1's Appendix B and was borrowed from Paper 2 by analogy -- that was wrong,
caught while cross-checking the model files against the paper's full equation list.
Paper 1's own Appendix B **does** specify this exactly, as Eq. 27:
`mu*_{c,0} ~ N(0, 2^2)`, "where mu*_{c,0} is the baseline out-of-school rate for the
year 2000." The implementation now applies this prior to the annual B-spline
coefficient whose Greville abscissa is exactly 2000 and constructs the random walk
outwards in both directions, preserving its forward drift and innovation scale.
Input assembly fails explicitly if there is not exactly one such control point.
This is the coefficient-level interpretation of the paper's anchor and still
requires confirmation against UIS code if it becomes available. It does not put the
prior directly on the evaluated spline value `Mu[c, year == 2000]`, since cubic
B-spline basis functions overlap.

## 3. Alpha 2D smoothing as independent conditional priors
Eqs. (11)-(13) are written as three separate sampling statements (row-wise smoothing,
column-wise smoothing, column-sum soft constraint). This isn't a single coherent GMRF
density — it's three separate penalty terms stacked, which is a common (if slightly
informal) way applied Bayesian papers specify 2D P-spline smoothers. We implemented
literally as three separate `~` statements per the paper, rather than deriving a
proper joint sparse-precision GMRF. Should behave similarly in practice but may have
different exact posterior geometry / effective degrees of freedom than UIS's actual
implementation if they built a formal joint GMRF.

If early test fits show poor mixing or non-identifiability in `alpha`, first thing to
try: add a weak anchor on `alpha[c][1,]` (the youngest post-entry age row), since
Eq. 11's row-smoothing only starts at g=2 and Eq. 13's column-sum constraint may not
fully pin down the g=1 row on its own.

## 4. Spline basis construction (see `src/python/spline_basis.py`)
The paper says "B-spline" / "P-spline" with annual knots (mu) and 5-year-interval
knots (alpha) but doesn't give the spline degree or boundary-knot handling. We use
cubic B-splines (`scipy.interpolate.BSpline`, degree 3, clamped boundary knots) with
interior knots at the stated spacing — implemented in `spline_basis.py`,
matching the convention of R's `splines2::bSpline(..., intercept = TRUE)` so the two
would agree if ever cross-checked. This is a reasonable default but not verified
against the paper's actual choice — flag as a source of potential divergence in the
sensitivity-analysis-equivalent step (paper Section 4.5 varies knot spacing 3/5/8
years; we should replicate that check once fitting works).

## 5. Regional grouping
For this reproduction, use one consistent regional classification throughout the
administrative data, survey data, validation outputs, and model input: the UIS
country-group membership rows labelled `SDG_*` in `uis_country_group_reference.csv`
and `uis_country_group_membership.csv`. This is the closest available match to the
SDG regions shown in the paper's Figure 7 and avoids mixing UIS, GPE, income, and
other country-group definitions.

The final processed input should contain one `region` label and one integer
`region_id` for every country. The same mapping must be used for `omega_r`,
`sigma_gamma_r`, `region_of_country`, and all regional post-processing. This is a
reproduction convention; if UIS later confirms a different grouping for the
random effects, the mapping can be swapped and treated as a sensitivity check.

## 6. Age-misreporting / negative-observation handling
Handled entirely on the data-prep side (Python), not in Stan: per Appendix A.2, extreme
administrative observations below -0.1 are excluded before the model ever sees them;
everything else (including negative values down to -0.1) is passed to the Stan
`A_obs` array as-is. No `<lower=0>` constraint exists on the administrative
likelihood's mean because `A_obs` itself is allowed to go negative — only `kappa`
(via `Phi()`) is bounded to [0,1]. This is the paper's central trick (Sec. 3.1) and
is preserved in the Stan code as written.

## 7. Not yet implemented
- Extracting `kappa` / aggregating to education level and SDG region (Section
  4.1-4.2 products) as a post-processing step over posterior samples.
- Reduced-model variants (Simple, NoBias, MuRW, MuK5) for the Table 2 validation
  comparison — each is a small structural variant of the model; plan to branch
  off `oosc_model_numpyro.py` once the full-scale base model runs.

## 8. Stan vs. NumPyro implementation
`oosc_cohort_model.stan` and `oosc_model_numpyro.py` implement the
identical model (same equation numbers referenced in both). We run NumPyro, not
Stan, because CmdStan requires a C++ compiler and installing one (RTools on
Windows) failed in this dev environment (the installer couldn't be launched
under the session's sandboxing; see git/session history if picking this back
up) — NumPyro (JAX-backed) needs no separate compiler toolchain and was
confirmed working via `pip install jax numpyro` alone. Keep the Stan file as
the canonical equation reference / for anyone who does have a working CmdStan
setup (e.g. Linux, Mac, or cloud compute) to cross-check against; treat
`oosc_model_numpyro.py` as the actual source of truth for what's been run.

Translation notes where the port required judgment, not just syntax:
- **Negative half-normal** (`gamma`, Eq. 6): Stan's `vector<upper=0>[C] gamma;
  gamma ~ normal(0, sigma);` is represented by a positive standard half-normal
  latent scaled by the regional standard deviation, then negated. This is the
  same distribution and avoids the centered hierarchical parameterization.
- **`mu_star` random walk with drift** (Eq. 4-5): both implementations anchor
  the coefficient at control year 2000 and generate coefficients outwards.
  Both implementations draw standard-normal innovations and scale them by
  `sigma_mu`. Coefficients before 2000 use the reverse recurrence with drift
  `-gamma`, while
  coefficients after 2000 use the paper's forward recurrence with drift
  `gamma`. This is a non-centered representation of the same Gaussian
  random-walk density along each adjacent pair.
- **Survey bias** uses a non-centered standard-normal parameterization in both
  implementations (`beta_raw[c,s] = sigma_beta[s] * beta_z[c,s]`). This is
  distributionally identical to Eq. 3.3.1 and helps sampling when country/series
  bias scales are weakly informed.
- **Shared observation-error hierarchy** (`sigma_adm`, `omega_r`, and `eta_w`)
  is likewise represented as `sigma_sigma` times standard half-normal latent
  variables in both implementations. These non-centered forms preserve the
  half-normal priors in Section 3.3.2 while avoiding a funnel when the shared
  scale is weakly informed.
- **`alpha` 2D P-spline smoothing** (Eq. 11-13): this is the one place the
  translation isn't mechanical. In Stan, `alpha` gets an implicit flat/improper
  prior (declared with no bounds, no top-level `~` statement targeting it
  directly as a "sample"), and the three `~ normal(...)` statements over row
  differences, column differences, and column sums are just additive
  `target +=` log-density penalties -- the standard Stan idiom for a GMRF/CAR-
  style smoother. NumPyro's exact equivalent is `dist.ImproperUniform` (a real,
  named primitive for this case, also used in NumPyro's own CAR examples) for
  the base measure, plus `numpyro.factor(...)` for each penalty term. This is
  what's implemented in `oosc_model_numpyro.py` -- verified it samples without
  divergence pathology in the smoke test (1 divergence out of 400 draws on
  synthetic data, i.e., unremarkable).

Verified (2026-09-11): `python oosc_model_numpyro.py` compiles, runs NUTS on
synthetic data (2 chains, 200 warmup + 200 samples), no NaNs, kappa stays in
[0,1], r_hat ~1.0 for the large majority of parameters. `sigma_mu` showed poor
mixing (r_hat 1.40) in that short run -- expected, since the synthetic test
data isn't actually generated from the model itself, just a plausible-looking
pattern; revisit once fitting on real data or data simulated from the model's
own prior.
- WIDE/“Other” observations remain excluded: the available records are level
  aggregates and do not have the single-age clustered-Jackknife uncertainty
  required by Eq. 2. Outputs must be labelled partial reproductions unless an
  appropriate level-aggregated likelihood is developed and validated.
- Full-fit diagnostics now include every latent scalar and every reported
  country-age-year rate. After moving the random-walk innovations and the
  hierarchical half-normal effects to non-centered representations, a
  four-chain, 14-country MF pilot covering all seven SDG regions passed the
  configured diagnostics (1,000 warmup and 1,000 retained draws per chain,
  no thinning, target acceptance 0.99; max R-hat 1.0073, min n_eff 885.4,
  zero divergences, no invalid diagnostics).
- A controlled rerun of the original eight-country subset still did not pass
  (max R-hat 1.0187, min n_eff 268.8, zero divergences). Its worst-mixing
  parameter remains the MICS2 wave-error scale `eta_w[5]`: that subset has
  only 24 MICS2 observations from COD and TCD, and COD has just one
  administrative observation. This points to weak identification in that
  deliberately small subset; do not use its estimates as final results.
- The passing 14-country MF pilot is evidence for sampler readiness on a
  regionally stratified subset, not proof that the full MF/M/F fits will
  converge. Run corresponding M and F pilots before full-country fits.
- **Runtime/memory optimization (2026-10-06):** the sampler now evaluates the
  cohort spline only at administrative and survey observation cells instead
  of materializing the full country × age × entry-year `kappa_probit` array
  at every NUTS step. `mu_star` coefficients are retained (much smaller than
  the full surface); reporting rates are reconstructed after sampling in
  bounded chunks, and rate diagnostics are calculated one country at a time.
  A synthetic coefficient-level comparison confirms that reconstructed rates
  match the former dense calculation; the age-specific cohort spline row must
  be paired with its corresponding age when contracting the spline basis.
  A one-country Dominican Republic smoke fit (258 administrative and 96
  survey observations; one chain, 30 warmup and 30 retained draws) completed
  in 15.1 seconds with finite parameters, bounded reconstructed rates, and
  zero divergences. This checks execution and output reconstruction only;
  the draw count is too small and there is only one chain, so it is not a
  convergence test. A second one-country Bangladesh smoke fit (206
  administrative and 96 survey observations) also completed and produced
  finite, bounded reported rates. With 30 warmup and 30 draws at target
  acceptance 0.90 it had 5 divergences; a follow-up with 100 warmup and 50
  draws at the configured 0.99 target had none. These short, single-chain
  checks cannot establish convergence or rule out divergences in longer runs.
  A longer 14-country benchmark was stopped after about 41 minutes without
  returning diagnostics; runtime savings and convergence of the optimized
  implementation therefore remain unverified. This is an
  algebraically equivalent likelihood/output transformation, not a change to
  the statistical model; do not start full-country runs based on these smoke
  checks alone.
- Exact reproduction is still conditional on confirming the year-2000
  coefficient interpretation and cubic spline/boundary convention with UIS
  implementation details. WIDE/“Other” also remains excluded because
  compatible single-age observations and clustered-Jackknife uncertainty are
  unavailable; label outputs as partial reproductions.

## 9. WIDE/"Other" added by importance reweighting (2026-10-09)
`Python/reweight_wide.py` adds the 14 available WIDE surveys to the finished MF fit
without new MCMC: for each WIDE country the saved posterior draws are reweighted by a
level-aggregated likelihood (population-weighted average of the single-age kappa over the
level's ages; design-effect-inflated binomial SE; "Other"-wave eta set per draw to the
median survey-wave eta; country bias integrated out with sigma_beta = mean survey-series
scale) and resampled. Countries are conditionally independent given the global
parameters, so the other countries and hyper-parameters are left untouched. This is an
approximation to Eq. 2, not the paper's single-age/clustered-Jackknife likelihood.
Result: effective sample size 200-990 of 1,000 draws per country; estimates move by
under 1 percentage point on average. WIDE is weak evidence under the model's own error
terms. Outputs: `model_output/MF/estimates_with_wide*.csv`, `wide_reweighting_report*.csv`.

## 10. Zero-enrolment fix, country-only refits, benchmark exclusions (2026-10-09)
- `prepare_admin_observations.py` now treats zero school-age enrolment (MF == 0, or rate >= 0.999;
  F == 0 with MF > 0 drops F and M only) as missing: 91 cells (25 MF), mostly ZAF, PLW, MHL, MKD, TCD,
  SYC, CHN, MDG, COG, COD, TGO, ALB. COD is left with no administrative data, as in the paper.
  The extract has only MF and F (M = MF - F), so the zeros cannot be completed by matching.
  Previous file kept as `admin_oosc_observations_pre_zerofix.csv`; removed cells in
  `admin_removed_zero_enrolment.csv`. `model_input_*.npz` and the saved posterior were NOT regenerated.
- `refit_countries.py` refits single countries with the global parameters fixed at their posterior
  medians (about 15 s per country). Validated on ARG and TCD with original data: mean absolute change in
  the median 0.1 percentage points, interval widths within 2%. Used to patch the 10 affected countries
  -> `estimates_patched.csv`; `estimates_final.csv` = patched + WIDE-reweighted (Section 9).
- Automatic Appendix A.2.1 step 3 (routine primary-enrolment ceiling) drops only 0.3% of cells and does
  not change agreement with the official series, so it is not applied.
- Administrative secondary-level rates are far above survey rates in several countries (PAK, UGA, MKD,
  RUS, CHN, JOR, HTI), consistent with incomplete secondary counts (e.g. vocational). CYP (admin only)
  disagrees with the official series by 28 points. AFG goes the other way (admin far below surveys).
  These 10 (COD, CYP, HTI, PAK, AFG, CHN, JOR, MKD, UGA, RUS) are listed in
  `excluded_from_benchmark.csv` and left out of the comparison with the official series; they remain
  in the fit.
