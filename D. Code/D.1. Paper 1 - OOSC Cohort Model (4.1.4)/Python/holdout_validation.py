"""Out-of-sample validation in the style of paper Section 3.5 / Table 2.

For each country, 20% of its observations (admin + survey, random) are held out,
the country is refit on the rest with the global hyper-parameters fixed at the
full-run posterior medians (see refit_countries.py), and the held-out
observations are compared with the model's predictions:

  admin  : prediction Phi(kappa)            vs A
  survey : prediction Phi(kappa + beta)     vs S    (bias-adjusted, as in the paper)
  aMSE / aMAE : admin errors use max(0, A) (negative A floored at 0)
  coverage     : share of held-out observations inside the central 80/90/95%
                 posterior-PREDICTIVE interval (observation noise included)

Variants:  model   - the full model
           nobias  - survey bias beta_{c,s} set to 0
           simple  - per-country linear probit model in year and age, all data
                     in the probit space with negative admin set to 0.0032%
                     (-4 in probit), as described for 'Simple' in Section 4.4

Caveat (documented deviation): the global parameters were estimated on ALL data,
including the held-out points, and are not re-estimated per variant. With ~100k
observations the leakage is small, but 'nobias' is handicapped because eta_w
cannot absorb the missing bias term as it would in a full refit.

Output: model_output/MF/holdout_<variant>.csv (one row per held-out observation)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["model", "nobias", "simple"], default="model")
ap.add_argument("--countries", nargs="*", default=None)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--samples", type=int, default=500)
ap.add_argument("--chains", type=int, default=4)
ap.add_argument("--seed", type=int, default=11)
args = ap.parse_args()
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={args.chains}")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402
from jax.scipy.special import ndtr as Phi_j  # noqa: E402
from numpyro.distributions import constraints  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402
from scipy.special import ndtr, ndtri  # noqa: E402
from scipy.stats import norm  # noqa: E402

numpyro.set_host_device_count(args.chains)
ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "A. Data" / "A.2. Processed"
OUT = PROC / "model_output" / "MF"


def cohort_model(d, fx, use_bias):
    A_max, K1, K5 = d["A_max"], d["B1"].shape[1], d["B5"].shape[1]
    B1, B5 = jnp.asarray(d["B1"]), jnp.asarray(d["B5"])
    gamma = -fx["sigma_gamma"] * numpyro.sample("gamma_z", dist.HalfNormal(1.0))
    mu2000 = numpyro.sample("mu_star_2000", dist.Normal(0, 2))
    eps = fx["sigma_mu"] * numpyro.sample("mu_eps_z", dist.Normal(0, 1).expand([K1 - 1]))
    a = d["mu_anchor_idx"]
    mu_star = numpyro.deterministic("mu_star", jnp.concatenate(
        [mu2000 + jnp.cumsum(-gamma - eps[:a][::-1])[::-1], mu2000[None], mu2000 + jnp.cumsum(gamma + eps[a:])]))
    alpha = numpyro.sample("alpha", dist.ImproperUniform(constraints.real, (), (A_max - 1, K5)))
    numpyro.factor("row", dist.Normal(0, 1 / jnp.sqrt(fx["tau1"])).log_prob(alpha[1:] - alpha[:-1]).sum())
    numpyro.factor("col", dist.Normal(0, 1 / jnp.sqrt(fx["tau2"])).log_prob(alpha[:, 1:] - alpha[:, :-1]).sum())
    numpyro.factor("sum", dist.Normal(0, 2.0).log_prob(alpha.sum(axis=0)).sum())
    cum = jnp.cumsum(alpha, axis=0)

    def kappa(age, ey):
        base = jnp.sum(mu_star * B1[ey], axis=-1)
        return base + jnp.where(age > 0, jnp.sum(cum[jnp.maximum(age - 1, 0)] * B5[ey], axis=-1), 0.0)

    beta = fx["sigma_beta"] * numpyro.sample("beta_z", dist.Normal(0, 1).expand([len(fx["sigma_beta"])]))
    if not use_bias:
        beta = beta * 0.0
    if len(d["A_obs"]):
        numpyro.sample("A_obs", dist.Normal(Phi_j(kappa(d["A_age"], d["A_ey"])), fx["sd_A"]), obs=jnp.asarray(d["A_obs"]))
    if len(d["K_obs"]):
        sd = jnp.sqrt((d["nu"] ** 2 + fx["eta_w"][d["S_wave"]] ** 2) / d["f"] ** 2)
        numpyro.sample("K_obs", dist.Normal(kappa(d["S_age"], d["S_ey"]) + beta[d["S_series"]], sd), obs=jnp.asarray(d["K_obs"]))


def simple_model(year, age, z):
    b0 = numpyro.sample("b0", dist.Normal(0, 3))
    b1 = numpyro.sample("b1", dist.Normal(0, 3))
    b2 = numpyro.sample("b2", dist.Normal(0, 3))
    s = numpyro.sample("s", dist.HalfNormal(1.0))
    numpyro.sample("z", dist.Normal(b0 + b1 * year + b2 * age, s), obs=jnp.asarray(z))


def np_kappa(mu_star, cum, B1, B5, age, ey):
    base = mu_star @ B1[ey].T  # (S, n)
    chg = np.einsum("snk,nk->sn", cum[:, np.maximum(age - 1, 0), :], B5[ey])
    return base + np.where(age[None, :] > 0, chg, 0.0)


def main():
    rng_root = np.random.SeedSequence(args.seed)
    index = json.loads((PROC / "model_input" / "model_index_MF.json").read_text())
    z = np.load(PROC / "model_input" / "model_input_MF.npz")
    countries = index["countries"]
    cid = {c: i for i, c in enumerate(countries)}
    a_max = int(z["A_max"])
    dr = np.load(OUT / "checkpoints" / "posterior_draws.npz")
    med = lambda k: np.median(dr[k].reshape((-1,) + dr[k].shape[2:]), axis=0)
    g = dict(sigma_adm=float(med("sample__sigma_adm")), omega=med("sample__omega_r"), eta_w=med("sample__eta_w"),
             sigma_beta=med("sample__sigma_beta"), sigma_mu=float(med("sample__sigma_mu")),
             sigma_gamma=med("sample__sigma_gamma_r"), tau1=float(med("sample__tau_alpha1")), tau2=float(med("sample__tau_alpha2")))
    todo = args.countries or countries
    rows = []
    for iso in todo:
        c = cid[iso]
        r = int(z["region_of_country"][c])
        am, sm = np.flatnonzero(z["A_country"] == c), np.flatnonzero(z["S_country"] == c)
        n = len(am) + len(sm)
        if n < 5:
            continue
        rng = np.random.default_rng([args.seed, c])
        hold = rng.permutation(n)[: max(1, int(round(0.2 * n)))]
        is_hold = np.zeros(n, bool)
        is_hold[hold] = True
        ha, hs = is_hold[: len(am)], is_hold[len(am):]
        t = time.time()
        sd_A = float(np.sqrt(g["sigma_adm"] ** 2 + g["omega"][r] ** 2))
        if args.variant == "simple":
            def zs(idx_a, idx_s):
                A = np.clip(z["A_obs"][idx_a], 3.2e-5, 1 - 3.2e-5)
                yr = np.concatenate([z["A_ey_idx"][idx_a] + z["A_age"][idx_a], z["S_ey_idx"][idx_s] + z["S_age"][idx_s]])
                ag = np.concatenate([z["A_age"][idx_a], z["S_age"][idx_s]]).astype(float)
                return yr.astype(float) - 20.0, ag - 5.0, np.concatenate([ndtri(A), z["K_obs"][idx_s]])
            tr, te = zs(am[~ha], sm[~hs]), zs(am[ha], sm[hs])
            mc = MCMC(NUTS(simple_model), num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains, progress_bar=False)
            mc.run(jax.random.PRNGKey(args.seed + c), *tr)
            sp = {k: np.asarray(v) for k, v in mc.get_samples().items()}
            mean = sp["b0"][:, None] + sp["b1"][:, None] * te[0][None] + sp["b2"][:, None] * te[1][None]
            zdraw = mean + sp["s"][:, None] * np.random.default_rng(1).standard_normal(mean.shape)
            real_pred = ndtr(mean)
            obs_real = np.concatenate([np.clip(z["A_obs"][am[ha]], -1, 2), ndtr(z["K_obs"][sm[hs]])])
            adm_flag = np.r_[np.ones(ha.sum(), bool), np.zeros(hs.sum(), bool)]
            # coverage on the probit scale of the transformed observation
            lo = {p: np.quantile(zdraw, [(1 - p) / 2, 1 - (1 - p) / 2], axis=0) for p in (0.8, 0.9, 0.95)}
            zobs = np.concatenate([ndtri(np.clip(z["A_obs"][am[ha]], 3.2e-5, 1 - 3.2e-5)), z["K_obs"][sm[hs]]])
            cover = {p: (zobs >= lo[p][0]) & (zobs <= lo[p][1]) for p in lo}
            pred_med = np.median(real_pred, axis=0)
        else:
            def sub(idx_a, idx_s):
                return dict(A_max=a_max, B1=z["B1"], B5=z["B5"], mu_anchor_idx=int(z["mu_anchor_idx"]),
                            A_obs=z["A_obs"][idx_a], A_age=z["A_age"][idx_a], A_ey=z["A_ey_idx"][idx_a],
                            K_obs=z["K_obs"][idx_s], f=z["f_obs"][idx_s], nu=z["nu"][idx_s], S_age=z["S_age"][idx_s],
                            S_ey=z["S_ey_idx"][idx_s], S_series=z["S_series"][idx_s], S_wave=z["S_wave"][idx_s])
            fx = dict(sigma_gamma=g["sigma_gamma"][r], sigma_mu=g["sigma_mu"], tau1=g["tau1"], tau2=g["tau2"],
                      sigma_beta=jnp.asarray(g["sigma_beta"]), eta_w=jnp.asarray(g["eta_w"]), sd_A=sd_A)
            mc = MCMC(NUTS(cohort_model, target_accept_prob=0.95), num_warmup=args.warmup, num_samples=args.samples,
                      num_chains=args.chains, progress_bar=False)
            mc.run(jax.random.PRNGKey(args.seed + c), sub(am[~ha], sm[~hs]), fx, args.variant == "model")
            sp = {k: np.asarray(v) for k, v in mc.get_samples().items()}
            cum = np.cumsum(sp["alpha"], axis=1)
            rng2 = np.random.default_rng(2)
            ka = np_kappa(sp["mu_star"], cum, z["B1"], z["B5"], z["A_age"][am[ha]], z["A_ey_idx"][am[ha]]) if ha.any() else np.zeros((len(sp["mu_star"]), 0))
            ks = np_kappa(sp["mu_star"], cum, z["B1"], z["B5"], z["S_age"][sm[hs]], z["S_ey_idx"][sm[hs]]) if hs.any() else np.zeros((len(sp["mu_star"]), 0))
            if args.variant == "model" and hs.any():
                beta = sp["beta_z"] * g["sigma_beta"][None, :]
                ks = ks + beta[:, z["S_series"][sm[hs]]]
            real_pred = np.concatenate([ndtr(ka), ndtr(ks)], axis=1)
            sd_s = np.sqrt((z["nu"][sm[hs]] ** 2 + g["eta_w"][z["S_wave"][sm[hs]]] ** 2) / z["f_obs"][sm[hs]] ** 2)
            pa = ndtr(ka) + sd_A * rng2.standard_normal(ka.shape)
            ps = ks + sd_s[None] * rng2.standard_normal(ks.shape)
            obs_real = np.concatenate([z["A_obs"][am[ha]], ndtr(z["K_obs"][sm[hs]])])
            adm_flag = np.r_[np.ones(ha.sum(), bool), np.zeros(hs.sum(), bool)]
            cover = {}
            for p in (0.8, 0.9, 0.95):
                q = [(1 - p) / 2, 1 - (1 - p) / 2]
                ca = (z["A_obs"][am[ha]] >= np.quantile(pa, q[0], axis=0)) & (z["A_obs"][am[ha]] <= np.quantile(pa, q[1], axis=0)) if ha.any() else np.zeros(0, bool)
                cs = (z["K_obs"][sm[hs]] >= np.quantile(ps, q[0], axis=0)) & (z["K_obs"][sm[hs]] <= np.quantile(ps, q[1], axis=0)) if hs.any() else np.zeros(0, bool)
                cover[p] = np.concatenate([ca, cs])
            pred_med = np.median(real_pred, axis=0)
        e = pred_med - obs_real
        e_adj = np.where(adm_flag, pred_med - np.maximum(0, obs_real), e)
        for i in range(len(e)):
            rows.append(dict(iso3=iso, admin=bool(adm_flag[i]), obs=obs_real[i], pred=pred_med[i], err=e[i], err_adj=e_adj[i],
                             c80=bool(cover[0.8][i]), c90=bool(cover[0.9][i]), c95=bool(cover[0.95][i])))
        print(f"{iso} {args.variant}: held {len(e)}/{n} in {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    suffix = "" if args.countries is None else "_subset"
    df.to_csv(OUT / f"holdout_{args.variant}{suffix}.csv", index=False)
    print(f"{args.variant}: n={len(df)} MSE={np.mean(df.err ** 2):.4f} aMSE={np.mean(df.err_adj ** 2):.4f} "
          f"MAE={df.err.abs().mean():.4f} aMAE={df.err_adj.abs().mean():.4f} "
          f"cov80={df.c80.mean():.2f} cov90={df.c90.mean():.2f} cov95={df.c95.mean():.2f}")


if __name__ == "__main__":
    main()
