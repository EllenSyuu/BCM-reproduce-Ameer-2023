"""Add the WIDE ("Other") surveys to a finished MF fit by importance reweighting
the saved posterior draws -- no new MCMC.

Why this is valid: given the global hyper-parameters in a draw (sigma_*, eta_w,
sigma_beta, ...), countries are conditionally independent. For each WIDE country
we therefore reweight that country's own draws by the likelihood of its WIDE
observations and resample; other countries and global parameters are untouched.

WIDE is only available as published education-LEVEL out-of-school rates, so the
single-age likelihood of Eq. 2 is replaced by a level-aggregated version
(documented deviation from the paper):

  probit(S_j) ~ N( probit( sum_a w_a kappa_{c,y,a} ) + beta_c , (nu_j^2 + eta^2) / f(probit(S_j))^2 )

  * the sum runs over the single ages of the level in that survey year, weighted
    by WPP population (MF);
  * nu_j = design-effect-inflated binomial SE (deff estimated from the DHS/MICS
    jackknife SEs); WIDE rows without a sample size get the median nu of the
    others (the clustered Jackknife is unavailable for aggregates);
  * eta ("Other" wave) has no data to be estimated from, so it is set per draw to
    the median of that draw's eta_w across survey waves (x --eta-scale);
  * beta_c (country bias for the Other series) is integrated out analytically,
    beta_c ~ N(0, sigma_beta^2) with sigma_beta = mean over the draw's survey series.

Inputs : model_output/MF/checkpoints/posterior_draws.npz, model_input/model_input_MF.npz
Outputs: model_output/MF/estimates_with_wide.csv, wide_reweighting_report.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtr, ndtri
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "A. Data" / "A.2. Processed"
STRUCT = (ROOT / "A. Data" / "A.1. Raw Sources" / "2. Administrative Data" / "2. UIS API - School System Structure"
          / "uis_entry_age_duration_2000_2022.csv")
OUT = PROC / "model_output" / "MF"

ap = argparse.ArgumentParser()
ap.add_argument("--eta-scale", type=float, default=1.0, help="multiplier on the plug-in Other-wave eta")
ap.add_argument("--suffix", default="", help="suffix for output file names")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

LEVELS = {"primary": ("entry_primary", "dur_primary"), "lower_secondary": ("entry_lowsec", "dur_lowsec"),
          "upper_secondary": ("entry_upsec", "dur_upsec")}


def reported_rates(mu_star, alpha, B1, B5, ey0, years, a_max):
    """Probability-scale rates (draws, a_max, years) for one country."""
    ai = np.arange(a_max)
    ey = years[None, :] - ai[:, None] - ey0
    base = np.einsum("sk,ayk->say", mu_star, B1[ey], optimize=True)
    out = np.empty(base.shape, dtype=np.float64)
    out[:, 0] = ndtr(base[:, 0])
    cum = np.cumsum(alpha, axis=1)
    out[:, 1:] = ndtr(base[:, 1:] + np.einsum("sak,ayk->say", cum, B5[ey][1:], optimize=True))
    return out


def main() -> None:
    rng = np.random.default_rng(args.seed)
    index = json.loads((PROC / "model_input" / "model_index_MF.json").read_text())
    z = np.load(PROC / "model_input" / "model_input_MF.npz")
    B1, B5 = z["B1"], z["B5"]
    a_max = int(z["A_max"])
    ey0 = index["ey_grid"][0]
    countries = index["countries"]
    cid = {c: i for i, c in enumerate(countries)}
    min_entry = dict(zip(countries, index["min_entry"]))

    d = np.load(OUT / "checkpoints" / "posterior_draws.npz")
    flat = lambda k: d[k].reshape((-1,) + d[k].shape[2:])
    mu_star, alpha = flat("sample__mu_star"), flat("sample__alpha")
    eta_w, sigma_beta = flat("sample__eta_w"), flat("sample__sigma_beta")
    S = mu_star.shape[0]
    eta_draw = np.median(eta_w, axis=1) * args.eta_scale  # (S,)
    sb_draw = sigma_beta.mean(axis=1)  # (S,)

    # --- WIDE observations (MF, level aggregates) ---
    w = pd.read_csv(PROC / "wide_oosc_observations.csv")
    w = w[(w.sex == "MF") & w.iso3.isin(cid) & (w.oos_rate > 0) & (w.oos_rate < 1)].copy()
    st = pd.read_csv(STRUCT)
    pop = pd.read_csv(PROC / "population_single_age_sex.csv")
    pop = pop[pop.sex == "MF"].set_index(["iso3", "year", "age"]).population

    # design effect from jackknife vs binomial SE of DHS/MICS single-age rates
    deff = []
    for f in ("dhs_oosc_observations.csv", "mics_oosc_observations.csv"):
        s = pd.read_csv(PROC / f)
        s = s[(s.sex == "MF") & (s.n_persons >= 300) & s.oos_rate.between(0.05, 0.95) & (s.se_jackknife > 0)]
        deff.append(s.se_jackknife / np.sqrt(s.oos_rate * (1 - s.oos_rate) / s.n_persons))
    sqrt_deff = float(pd.concat(deff).median())
    nu = sqrt_deff * np.sqrt(w.oos_rate * (1 - w.oos_rate) / w.n_persons)
    nu_fill = float(nu.median())
    w["nu"] = nu.fillna(nu_fill)
    print(f"sqrt(design effect) = {sqrt_deff:.2f}; WIDE nu median = {nu_fill:.4f}; "
          f"{int(nu.isna().sum())}/{len(w)} rows lack n and get that median")

    # --- per observation: pop-weighted level-average of kappa for each draw ---
    rows, kap = [], []
    years_all = np.arange(2000, 2023)
    for _, r in w.iterrows():
        c = cid[r.iso3]
        sy = st[(st.iso3 == r.iso3) & (st.year == r.year)]
        if sy.empty:
            sy = st[st.iso3 == r.iso3].sort_values("year").tail(1)
        a_col, d_col = LEVELS[r.level]
        lo, n = int(sy[a_col].iloc[0]), int(sy[d_col].iloc[0])
        ages = [a for a in range(lo, lo + n) if (r.iso3, r.year, a) in pop.index and a - min_entry[r.iso3] in range(a_max)]
        if not ages or r.year not in years_all:
            continue
        wt = np.array([pop[(r.iso3, r.year, a)] for a in ages], float)
        wt /= wt.sum()
        ai = np.array(ages) - min_entry[r.iso3]
        ey = r.year - ai - ey0
        base = np.einsum("sk,ak->sa", mu_star[:, c], B1[ey], optimize=True)
        cum = np.cumsum(alpha[:, c], axis=1)  # (S, a_max-1, K5)
        chg = np.zeros_like(base)
        for j, a_i in enumerate(ai):
            if a_i > 0:
                chg[:, j] = cum[:, a_i - 1] @ B5[ey[j]]
        rows.append(r)
        kap.append(ndtr(base + chg) @ wt)
    obs = pd.DataFrame(rows).reset_index(drop=True)
    kap = np.array(kap).T  # (S, n_obs)
    zS = ndtri(obs.oos_rate.to_numpy())
    f = norm.pdf(zS)
    print(f"{len(obs)} WIDE MF observations over {obs.iso3.nunique()} countries")

    # --- per-country log-weights with beta_c integrated out ---
    report, new_idx = [], {}
    mean_p = ndtri(np.clip(kap, 1e-6, 1 - 1e-6))
    for iso3, g in obs.groupby("iso3"):
        ii = g.index.to_numpy()
        logw = np.empty(S)
        for s in range(S):
            sd2 = (g.nu.to_numpy() ** 2 + eta_draw[s] ** 2) / f[ii] ** 2
            cov = np.diag(sd2) + sb_draw[s] ** 2
            r = zS[ii] - mean_p[s, ii]
            sign, ld = np.linalg.slogdet(cov)
            logw[s] = -0.5 * (r @ np.linalg.solve(cov, r) + ld)
        p = np.exp(logw - logsumexp(logw))
        ess = 1 / np.sum(p ** 2)
        # systematic resampling
        u = (rng.random() + np.arange(S)) / S
        idx = np.minimum(np.searchsorted(np.cumsum(p), u), S - 1)
        new_idx[iso3] = idx
        before = np.average(kap[:, ii], axis=0)
        after = np.average(kap[idx][:, ii], axis=0)
        report.append(dict(iso3=iso3, n_obs=len(ii), ess=round(ess, 1), max_weight=round(p.max(), 3),
                           wide_mean=g.oos_rate.mean(), fit_before=before.mean(), fit_after=after.mean(),
                           abs_gap_before=np.abs(before - g.oos_rate.to_numpy()).mean(),
                           abs_gap_after=np.abs(after - g.oos_rate.to_numpy()).mean()))
    rep = pd.DataFrame(report)
    print(rep.round(3).to_string(index=False))
    rep.to_csv(OUT / f"wide_reweighting_report{args.suffix}.csv", index=False)

    # --- rewrite estimates for reweighted countries ---
    est = pd.read_csv(OUT / "estimates.csv")
    est = est.set_index(["iso3", "year", "age"])
    for iso3, idx in new_idx.items():
        c = cid[iso3]
        rates = reported_rates(mu_star[idx, c], alpha[idx, c], B1, B5, ey0, years_all, a_max)
        q = np.quantile(rates, [0.05, 0.5, 0.95], axis=0)
        for a_i in range(a_max):
            age = min_entry[iso3] + a_i
            for yi, y in enumerate(years_all):
                key = (iso3, int(y), age)
                if key in est.index:
                    est.loc[key, ["oos_q05", "oos_median", "oos_q95"]] = q[:, a_i, yi]
    est.reset_index().to_csv(OUT / f"estimates_with_wide{args.suffix}.csv", index=False)
    print("wrote", OUT / f"estimates_with_wide{args.suffix}.csv")


if __name__ == "__main__":
    main()
