"""Refit individual countries with the global hyper-parameters held fixed.

Use case: a data fix changes a few countries' observations (e.g. spurious
zero-enrolment records removed) and a full ~9 h refit is not wanted. Given the
global parameters, countries are conditionally independent in the model, so a
country's posterior can be recomputed on its own with sigma_adm, omega_r,
eta_w, sigma_beta, sigma_mu, sigma_gamma_r, tau_alpha1/2 fixed at their
posterior medians from the finished MF run. Everything else (gamma, mu*, alpha,
survey bias) is sampled exactly as in oosc_model_numpyro.model.

Approximation: global uncertainty is not propagated, so intervals are slightly
narrower than a full refit. Validate with --validate (refits a country on its
ORIGINAL data and compares with the full-run estimates).

Usage:
  refit_countries.py --countries COD ZAF --drop admin_removed_zero_enrolment.csv --suffix _patched
  refit_countries.py --countries ARG TCD --validate
Outputs (model_output/MF/): estimates<suffix>.csv, refit_report<suffix>.csv
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--countries", nargs="+", required=True)
ap.add_argument("--drop", default=None, help="CSV (in A.2. Processed) of removed admin cells: iso3,year,age,sex")
ap.add_argument("--validate", action="store_true", help="refit on original data and compare to full run")
ap.add_argument("--suffix", default="_patched")
ap.add_argument("--warmup", type=int, default=1000)
ap.add_argument("--samples", type=int, default=1000)
ap.add_argument("--chains", type=int, default=4)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={args.chains}")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402
from jax.scipy.special import ndtr as Phi  # noqa: E402
from numpyro.diagnostics import summary  # noqa: E402
from numpyro.distributions import constraints  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

numpyro.set_host_device_count(args.chains)
ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "A. Data" / "A.2. Processed"
OUT = PROC / "model_output" / "MF"


def country_model(d, fx):
    """Single-country version of oosc_model_numpyro.model with fixed globals."""
    A_max, K1, K5 = d["A_max"], d["B1"].shape[1], d["B5"].shape[1]
    B1, B5 = jnp.asarray(d["B1"]), jnp.asarray(d["B5"])
    gamma_z = numpyro.sample("gamma_z", dist.HalfNormal(1.0))
    gamma = -fx["sigma_gamma"] * gamma_z
    mu2000 = numpyro.sample("mu_star_2000", dist.Normal(0, 2))
    eps = fx["sigma_mu"] * numpyro.sample("mu_eps_z", dist.Normal(0, 1).expand([K1 - 1]))
    a = d["mu_anchor_idx"]
    before = mu2000 + jnp.cumsum(-gamma - eps[:a][::-1])[::-1]
    after = mu2000 + jnp.cumsum(gamma + eps[a:])
    mu_star = numpyro.deterministic("mu_star", jnp.concatenate([before, mu2000[None], after]))
    alpha = numpyro.sample("alpha", dist.ImproperUniform(constraints.real, (), (A_max - 1, K5)))
    numpyro.factor("row", dist.Normal(0, 1 / jnp.sqrt(fx["tau1"])).log_prob(alpha[1:] - alpha[:-1]).sum())
    numpyro.factor("col", dist.Normal(0, 1 / jnp.sqrt(fx["tau2"])).log_prob(alpha[:, 1:] - alpha[:, :-1]).sum())
    numpyro.factor("sum", dist.Normal(0, 2.0).log_prob(alpha.sum(axis=0)).sum())
    cum = jnp.cumsum(alpha, axis=0)

    def kappa(age, ey):
        base = jnp.sum(mu_star * B1[ey], axis=-1)
        chg = jnp.sum(cum[jnp.maximum(age - 1, 0)] * B5[ey], axis=-1)
        return base + jnp.where(age > 0, chg, 0.0)

    beta = fx["sigma_beta"] * numpyro.sample("beta_z", dist.Normal(0, 1).expand([len(fx["sigma_beta"])]))
    if len(d["A_obs"]):
        numpyro.sample("A_obs", dist.Normal(Phi(kappa(d["A_age"], d["A_ey"])), fx["sd_A"]), obs=jnp.asarray(d["A_obs"]))
    if len(d["K_obs"]):
        sd = jnp.sqrt((d["nu"] ** 2 + fx["eta_w"][d["S_wave"]] ** 2) / d["f"] ** 2)
        numpyro.sample("K_obs", dist.Normal(kappa(d["S_age"], d["S_ey"]) + beta[d["S_series"]], sd), obs=jnp.asarray(d["K_obs"]))


def reported_rates(mu_star, alpha, B1, B5, ey0, years, a_max):
    from scipy.special import ndtr
    ai = np.arange(a_max)
    ey = years[None, :] - ai[:, None] - ey0
    base = np.einsum("sk,ayk->say", mu_star, B1[ey], optimize=True)
    out = np.empty(base.shape)
    out[:, 0] = ndtr(base[:, 0])
    out[:, 1:] = ndtr(base[:, 1:] + np.einsum("sak,ayk->say", np.cumsum(alpha, axis=1), B5[ey][1:], optimize=True))
    return out


def main():
    index = json.loads((PROC / "model_input" / "model_index_MF.json").read_text())
    z = np.load(PROC / "model_input" / "model_input_MF.npz")
    obs = pd.read_csv(PROC / "model_input" / "model_observations_MF.csv")
    adm_rows = obs[obs.kind == "admin"].reset_index(drop=True)
    assert len(adm_rows) == len(z["A_obs"]), "admin row order mismatch"
    sur_rows = obs[obs.kind == "survey"].reset_index(drop=True)
    assert len(sur_rows) == len(z["K_obs"])
    countries = index["countries"]
    cid = {c: i for i, c in enumerate(countries)}
    regions = index["regions"]
    ey0 = index["ey_grid"][0]
    a_max = int(z["A_max"])

    dr = np.load(OUT / "checkpoints" / "posterior_draws.npz")
    med = lambda k: np.median(dr[k].reshape((-1,) + dr[k].shape[2:]), axis=0)
    g = dict(sigma_adm=float(med("sample__sigma_adm")), omega=med("sample__omega_r"), eta_w=med("sample__eta_w"),
             sigma_beta=med("sample__sigma_beta"), sigma_mu=float(med("sample__sigma_mu")),
             sigma_gamma=med("sample__sigma_gamma_r"), tau1=float(med("sample__tau_alpha1")), tau2=float(med("sample__tau_alpha2")))

    drop = None
    if args.drop and not args.validate:
        dd = pd.read_csv(PROC / args.drop)
        dd = dd[dd.sex == "MF"][["iso3", "year", "age"]].assign(_drop=1)
        j = adm_rows.merge(dd, on=["iso3", "year", "age"], how="left")
        drop = j._drop.fillna(0).to_numpy().astype(bool)
        print(f"{int(drop.sum())} admin rows dropped across all countries")

    est = pd.read_csv(OUT / "estimates.csv").set_index(["iso3", "year", "age"])
    full = est.copy()
    years = np.arange(2000, 2023)
    report = []
    for iso in args.countries:
        c = cid[iso]
        r = int(z["region_of_country"][c])
        am = (z["A_country"] == c) & (~drop if drop is not None else True)
        sm = z["S_country"] == c
        d = dict(A_max=a_max, B1=z["B1"], B5=z["B5"], mu_anchor_idx=int(z["mu_anchor_idx"]),
                 A_obs=z["A_obs"][am], A_age=z["A_age"][am], A_ey=z["A_ey_idx"][am],
                 K_obs=z["K_obs"][sm], f=z["f_obs"][sm], nu=z["nu"][sm], S_age=z["S_age"][sm], S_ey=z["S_ey_idx"][sm],
                 S_series=z["S_series"][sm], S_wave=z["S_wave"][sm])
        fx = dict(sigma_gamma=g["sigma_gamma"][r], sigma_mu=g["sigma_mu"], tau1=g["tau1"], tau2=g["tau2"],
                  sigma_beta=jnp.asarray(g["sigma_beta"]), eta_w=jnp.asarray(g["eta_w"]),
                  sd_A=float(np.sqrt(g["sigma_adm"] ** 2 + g["omega"][r] ** 2)))
        t = time.time()
        mcmc = MCMC(NUTS(country_model, target_accept_prob=0.99), num_warmup=args.warmup, num_samples=args.samples,
                    num_chains=args.chains, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(args.seed + c), d, fx)
        s = mcmc.get_samples(group_by_chain=True)
        div = int(np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum())
        flat = {k: np.asarray(v).reshape((-1,) + v.shape[2:]) for k, v in s.items()}
        sm_ = summary({"mu_star": s["mu_star"], "alpha": s["alpha"]}, prob=0.9)
        rhat = max(float(np.nanmax(sm_[k]["r_hat"])) for k in sm_)
        neff = min(float(np.nanmin(sm_[k]["n_eff"])) for k in sm_)
        rates = reported_rates(flat["mu_star"], flat["alpha"], z["B1"], z["B5"], ey0, years, a_max)
        q = np.quantile(rates, [0.05, 0.5, 0.95], axis=0)
        me = index["min_entry"][c]
        for ai in range(a_max):
            for yi, y in enumerate(years):
                k = (iso, int(y), me + ai)
                if k in est.index:
                    est.loc[k, ["oos_q05", "oos_median", "oos_q95"]] = q[:, ai, yi]
        old = full.loc[iso]
        new = est.loc[iso]
        report.append(dict(iso3=iso, admin_obs=int(am.sum()), survey_obs=int(sm.sum()), minutes=round((time.time() - t) / 60, 1),
                           max_rhat=round(rhat, 3), min_neff=round(neff), divergences=div,
                           mean_abs_change=float((new.oos_median - old.oos_median).abs().mean()),
                           mean_interval_width_old=float((old.oos_q95 - old.oos_q05).mean()),
                           mean_interval_width_new=float((new.oos_q95 - new.oos_q05).mean())))
        print(report[-1], flush=True)
    rep = pd.DataFrame(report)
    rep.to_csv(OUT / f"refit_report{args.suffix}.csv", index=False)
    if not args.validate:
        est.reset_index().to_csv(OUT / f"estimates{args.suffix}.csv", index=False)
        print("wrote", OUT / f"estimates{args.suffix}.csv")


if __name__ == "__main__":
    main()
