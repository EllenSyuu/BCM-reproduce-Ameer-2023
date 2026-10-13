// Bayesian cohort model for SDG 4.1.4 out-of-school rates.
// Reproduction of Dharamshi, Barakat, Antoninis & Montoya (UIS/GEMR, 2022),
// "A Bayesian Cohort Model for Estimating SDG Indicator 4.1.4: Out-of-School Rates".
// Equation numbers in comments refer to the paper's Section 3 / Appendix B.
//
// Run separately per sex population (male / female / total), per the paper (Section 3.4).
//
// Indexing conventions (see README / docs/model_notes.md for full rationale):
//  - age index a = 1 is the country's minimum school-entry age; a runs 1..A_max.
//  - "entry year" (ey) = calendar year y minus (age - 1); both mu and Theta are
//    evaluated on a shared entry-year grid of size N_EY, wide enough to cover the
//    earliest entry year implied by the oldest observed age in the data.
//  - B1 is the spline basis for mu (annual knots, Eq. 4); B5 is the spline basis
//    for alpha / Theta (5-year knots, Eq. 10).

data {
  int<lower=1> C;                 // number of countries
  int<lower=1> R;                 // number of regions
  int<lower=1> A_max;             // max relative age index
  int<lower=1> N_EY;              // entry-year grid size
  int<lower=1> N_K1;              // number of mu spline knots
  int<lower=2, upper=N_K1 - 1> MU_ANCHOR; // 1-based: control coefficient at calendar year 2000
  int<lower=1> N_K5;              // number of alpha spline knots
  int<lower=1> N_SERIES;          // number of survey series (DHS, MICS, Census, Other, ...)
  int<lower=1> N_WAVE;            // number of survey waves (DHS4, DHS5, MICS3, ...)

  matrix[N_EY, N_K1] B1;          // mu spline basis matrix
  matrix[N_EY, N_K5] B5;          // alpha spline basis matrix

  array[C] int<lower=1, upper=R> region_of_country;

  // --- administrative observations (Eq. 1) ---
  int<lower=0> N_A;
  array[N_A] real A_obs;                            // observed rate, may be negative (Sec. 2.1)
  array[N_A] int<lower=1, upper=C> A_country;
  array[N_A] int<lower=1, upper=R> A_region;
  array[N_A] int<lower=1, upper=A_max> A_age;
  array[N_A] int<lower=1, upper=N_EY> A_ey_idx;      // grid index of (year - age + 1)

  // --- survey observations (Eq. 2) ---
  int<lower=0> N_S;
  array[N_S] real K_obs;                             // probit(S_j), precomputed in R
  array[N_S] real<lower=0> f_obs;                    // std-normal pdf at K_obs (Delta method)
  array[N_S] real<lower=0> nu;                       // sampling SE, real space, clustered Jackknife
  array[N_S] int<lower=1, upper=C> S_country;
  array[N_S] int<lower=1, upper=A_max> S_age;
  array[N_S] int<lower=1, upper=N_EY> S_ey_idx;
  array[N_S] int<lower=1, upper=N_SERIES> S_series;
  array[N_S] int<lower=1, upper=N_WAVE> S_wave;
}

parameters {
  // --- baseline entry-age process, mu (Eq. 4-9) ---
  vector[C] mu_star_2000;
  vector<lower=0>[C] gamma_z;             // non-centered regional half-normal effect
  vector<lower=0>[R] sigma_gamma_r;       // Eq. 7
  real<lower=0> sigma_mu;                 // Eq. 9
  matrix[C, N_K1 - 1] mu_eps_z;           // non-centered random-walk innovations

  // --- cohort change process, alpha / Theta (Eq. 10-15) ---
  array[C] matrix[A_max - 1, N_K5] alpha;
  real<lower=0> tau_alpha1;               // Eq. 14, column (age) smoothing precision
  real<lower=0> tau_alpha2;               // Eq. 15, row (spline/time) smoothing precision

  // --- survey bias (Sec. 3.3.1) ---
  matrix[C, N_SERIES] beta_z;
  vector<lower=0>[N_SERIES] sigma_beta;

  // --- variance components (Sec. 3.3.2) ---
  real<lower=0> sigma_sigma;              // shared hyper-scale, Eq. 34-37
  real<lower=0> sigma_adm_z;              // non-centered half-normal effects
  vector<lower=0>[R] omega_r_z;
  vector<lower=0>[N_WAVE] eta_w_z;
}

transformed parameters {
  matrix[C, N_EY] Mu;                      // Eq. 4, evaluated on the entry-year grid
  array[C] matrix[A_max, N_EY] kappa_probit;  // Eq. 3: Phi^-1(kappa_c,y,a)
  matrix[C, N_SERIES] beta_raw;
  matrix[C, N_K1] mu_star;
  vector<upper=0>[C] gamma;
  matrix[C, N_K1 - 1] mu_eps;
  real<lower=0> sigma_adm;
  vector<lower=0>[R] omega_r;
  vector<lower=0>[N_WAVE] eta_w;

  gamma = -sigma_gamma_r[region_of_country] .* gamma_z;
  mu_eps = sigma_mu * mu_eps_z;
  sigma_adm = sigma_sigma * sigma_adm_z;
  omega_r = sigma_sigma * omega_r_z;
  eta_w = sigma_sigma * eta_w_z;
  for (c in 1:C) {
    mu_star[c, MU_ANCHOR] = mu_star_2000[c];
    for (k in 1:(MU_ANCHOR - 1)) {
      mu_star[c, k] = mu_star[c, k + 1] - gamma[c] - mu_eps[c, k];
    }
    for (k in (MU_ANCHOR + 1):N_K1) {
      mu_star[c, k] = mu_star[c, k - 1] + gamma[c] + mu_eps[c, k - 1];
    }
  }
  Mu = mu_star * B1';
  beta_raw = beta_z .* rep_matrix(sigma_beta', C);

  for (c in 1:C) {
    matrix[A_max - 1, N_K5] cum_alpha;
    for (k in 1:N_K5) {
      cum_alpha[, k] = cumulative_sum(alpha[c][, k]);   // outer sum in Eq. 10
    }
    kappa_probit[c][1, ] = Mu[c, ];                       // entry age: Theta = 0
    kappa_probit[c][2:A_max, ] = rep_matrix(Mu[c, ], A_max - 1) + cum_alpha * B5';
  }
}

model {
  // --- mu: smoothed random walk with drift ---
  sigma_gamma_r ~ normal(0, 0.02);
  gamma_z ~ std_normal();
  sigma_mu ~ normal(0, 0.1);
  to_vector(mu_eps_z) ~ std_normal();
  mu_star_2000 ~ normal(0, 2);             // Eq. 27: annual spline control point at 2000

  // --- alpha: 2D P-spline smoothing over age (g) and cohort-entry spline (k) ---
  tau_alpha1 ~ normal(0, 1);
  tau_alpha2 ~ normal(0, 1);
  for (c in 1:C) {
    for (g in 2:(A_max - 1)) {
      alpha[c][g, ] ~ normal(alpha[c][g - 1, ], inv_sqrt(tau_alpha1));   // Eq. 11
    }
    for (k in 2:N_K5) {
      alpha[c][, k] ~ normal(alpha[c][, k - 1], inv_sqrt(tau_alpha2));   // Eq. 12
    }
    for (k in 1:N_K5) {
      sum(alpha[c][, k]) ~ normal(0, 2);                                  // Eq. 13
    }
  }

  // --- survey bias, hierarchical by series, shared across countries ---
  sigma_beta ~ normal(0, 0.1);
  to_vector(beta_z) ~ std_normal();

  // --- variance components ---
  sigma_sigma ~ normal(0, 0.01);
  sigma_adm_z ~ std_normal();
  omega_r_z ~ std_normal();
  eta_w_z ~ std_normal();

  // --- likelihood: administrative (Eq. 1), real space, allows negative observations ---
  {
    vector[N_A] mean_A;
    vector[N_A] sd_A;
    for (i in 1:N_A) {
      mean_A[i] = Phi(kappa_probit[A_country[i]][A_age[i], A_ey_idx[i]]);
      sd_A[i] = sqrt(square(sigma_adm) + square(omega_r[A_region[i]]));
    }
    A_obs ~ normal(mean_A, sd_A);
  }

  // --- likelihood: survey (Eq. 2), probit space ---
  {
    vector[N_S] mean_S;
    vector[N_S] sd_S;
    for (j in 1:N_S) {
      mean_S[j] = kappa_probit[S_country[j]][S_age[j], S_ey_idx[j]]
                  + beta_raw[S_country[j], S_series[j]];
      sd_S[j] = sqrt((square(nu[j]) + square(eta_w[S_wave[j]])) / square(f_obs[j]));
    }
    K_obs ~ normal(mean_S, sd_S);
  }
}

// NOTE: no generated quantities block yet. Extracting the full kappa_probit /
// Phi(kappa_probit) surface (C x A_max x N_EY) for every posterior draw is
// expensive at full scale (192 countries) — add it once the model is validated
// on a handful of countries, and consider thinning or extracting only the
// country-year-age combinations actually needed for reporting.
