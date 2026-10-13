"""Fit the full out-of-school cohort model (admin + survey) for one sex
population and write estimates (paper Section 3.4 / 4).

Paper settings (Section 3.4): 4 chains x 4,000 iterations, first 2,000 burn-in,
thinned to 1,000 draws in total per population -> --warmup 2000 --samples 2000
--chains 4 --thinning 8 (the defaults here).

Input : A. Data/A.2. Processed/model_input/model_input_<tag>.npz (+ index json),
        from assemble_model_input.py
Output: A. Data/A.2. Processed/model_output/<tag>/
          progress.txt      live sampling progress, including per-chain bars
          checkpoints/      restartable sampler states and posterior chunks
          posterior_draws.npz all retained posterior draws and divergence flags
          estimates.csv      out-of-school rate by country x year x age:
                             median and 90% interval (Fig. 5)
          parameters.csv     variance components and survey bias scales
          diagnostics.csv    r_hat / n_eff for every scalar and vector site
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="MF", help="model input tag, e.g. MF, M, F or MF_BFA-SWE")
ap.add_argument("--warmup", type=int, default=2000)
ap.add_argument("--samples", type=int, default=2000)
ap.add_argument("--chains", type=int, default=4)
ap.add_argument("--thinning", type=int, default=8)
ap.add_argument("--target-accept-prob", type=float, default=0.99)
ap.add_argument("--seed", type=int, default=2022)
ap.add_argument("--checkpoint-every", type=int, default=80,
                help="Sampling iterations between durable checkpoints (must be divisible by thinning).")
args = ap.parse_args()

# must be set before JAX starts: one CPU "device" per chain so chains run in parallel
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={args.chains}")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpyro.diagnostics import summary  # noqa: E402
from scipy.special import ndtr as scipy_ndtr  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from oosc_model_numpyro import ModelData, run_mcmc_checkpointed  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
IN_DIR = ROOT / "A. Data" / "A.2. Processed" / "model_input"
OUT_ROOT = ROOT / "A. Data" / "A.2. Processed" / "model_output"


def load(tag: str) -> tuple[ModelData, dict]:
    z = np.load(IN_DIR / f"model_input_{tag}.npz")
    fields = {k: (int(z[k]) if z[k].ndim == 0 else z[k]) for k in z.files}
    index = json.loads((IN_DIR / f"model_index_{tag}.json").read_text())
    return ModelData(**fields), index


def checkpoint_fingerprint(tag: str) -> str:
    digest = hashlib.sha256()
    for path in (
        IN_DIR / f"model_input_{tag}.npz",
        IN_DIR / f"model_index_{tag}.json",
        Path(__file__),
        Path(__file__).with_name("oosc_model_numpyro.py"),
    ):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def country_reported_rates(
    mu_star: np.ndarray,
    alpha: np.ndarray,
    data: ModelData,
    index: dict,
    chunk_size: int = 64,
) -> np.ndarray:
    """Calculate reported rates in bounded chunks; inputs may include chain/draw axes."""
    mu_star = np.asarray(mu_star)
    alpha = np.asarray(alpha)
    leading_shape = mu_star.shape[:-1]
    flat_mu = mu_star.reshape((-1, mu_star.shape[-1]))
    flat_alpha = alpha.reshape((-1,) + alpha.shape[-2:])
    ey_grid = np.array(index["ey_grid"])
    reported_years = np.arange(2000, 2023)
    age_indices = np.arange(data.A_max)
    reported_ey = reported_years[None, :] - age_indices[:, None] - ey_grid[0]
    if np.any((reported_ey < 0) | (reported_ey >= len(ey_grid))):
        raise ValueError("Reported country-age-year cells exceed the entry-year grid.")

    B1_reported = data.B1[reported_ey]
    B5_reported = data.B5[reported_ey]
    rates = np.empty(
        (len(flat_mu), data.A_max, len(reported_years)),
        dtype=np.float32,
    )
    for start in range(0, len(flat_mu), chunk_size):
        stop = min(start + chunk_size, len(flat_mu))
        baseline = np.einsum(
            "sk,ayk->say", flat_mu[start:stop], B1_reported, optimize=True
        )
        rates[start:stop, 0] = scipy_ndtr(baseline[:, 0])
        if data.A_max > 1:
            cumulative_alpha = np.cumsum(flat_alpha[start:stop], axis=1)
            change = np.einsum(
                "sak,ayk->say",
                cumulative_alpha,
                B5_reported[1:],
                optimize=True,
            )
            rates[start:stop, 1:] = scipy_ndtr(baseline[:, 1:] + change)
    return rates.reshape(leading_shape + rates.shape[-2:])


def main() -> None:
    data, index = load(args.tag)
    out = OUT_ROOT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.txt"
    print(f"[{args.tag}] C={data.C} R={data.R} A_max={data.A_max} N_EY={data.N_EY} "
          f"admin={len(data.A_obs):,} survey={len(data.K_obs):,}  devices={jax.device_count()}", flush=True)

    t0 = time.time()
    grouped, extra_fields = run_mcmc_checkpointed(
        data,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        seed=args.seed,
        thinning=args.thinning,
        target_accept_prob=args.target_accept_prob,
        progress_path=progress_path,
        checkpoint_dir=out / "checkpoints",
        checkpoint_every=args.checkpoint_every,
        checkpoint_id=checkpoint_fingerprint(args.tag),
    )
    print(f"sampling/checkpoint load took {(time.time() - t0) / 60:.1f} min", flush=True)

    diag = []
    for name, v in grouped.items():
        s = summary({name: np.asarray(v)}, prob=0.9)[name]
        r_hat = np.asarray(s["r_hat"])
        n_eff = np.asarray(s["n_eff"])
        finite_site_r_hat = r_hat[np.isfinite(r_hat)]
        finite_site_n_eff = n_eff[np.isfinite(n_eff)]
        site_r_hat_max = float(np.max(finite_site_r_hat)) if finite_site_r_hat.size else float("nan")
        site_n_eff_min = float(np.min(finite_site_n_eff)) if finite_site_n_eff.size else float("nan")
        indices = list(np.ndindex(r_hat.shape)) if r_hat.shape else [()]
        for idx in indices:
            diag.append({
                "site": name,
                "index": ",".join(map(str, idx)),
                "r_hat": float(r_hat[idx]),
                "n_eff": float(n_eff[idx]),
                "site_r_hat_max": site_r_hat_max,
                "site_n_eff_min": site_n_eff_min,
            })
    reported_years = np.arange(2000, 2023)
    estimate_rows = []
    for c, iso3 in enumerate(index["countries"]):
        progress_path.write_text(
            f"Status: Postprocessing estimates {c + 1}/{len(index['countries'])} ({iso3})\n"
            f"Updated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n",
            encoding="utf-8",
        )
        prob_chains = country_reported_rates(
            grouped["mu_star"][:, :, c, :],
            grouped["alpha"][:, :, c, :, :],
            data,
            index,
        )
        prob_draws = prob_chains.reshape((-1, data.A_max, len(reported_years)))
        s = summary({"oos_rate": prob_chains}, prob=0.9)["oos_rate"]
        r_hat = np.asarray(s["r_hat"])
        n_eff = np.asarray(s["n_eff"])
        finite_site_r_hat = r_hat[np.isfinite(r_hat)]
        finite_site_n_eff = n_eff[np.isfinite(n_eff)]
        site_r_hat_max = float(np.max(finite_site_r_hat)) if finite_site_r_hat.size else float("nan")
        site_n_eff_min = float(np.min(finite_site_n_eff)) if finite_site_n_eff.size else float("nan")
        quantiles = np.quantile(prob_draws, [0.05, 0.5, 0.95], axis=0)
        entry_age = index["min_entry"][c]
        for a, year_idx in np.ndindex(r_hat.shape):
            diag.append({
                "site": "oos_rate",
                "index": f"{iso3},age_rel={a + 1},year={reported_years[year_idx]}",
                "r_hat": float(r_hat[a, year_idx]),
                "n_eff": float(n_eff[a, year_idx]),
                "site_r_hat_max": site_r_hat_max,
                "site_n_eff_min": site_n_eff_min,
            })
            estimate_rows.append((
                iso3,
                reported_years[year_idx],
                entry_age + a,
                a + 1,
                quantiles[1, a, year_idx],
                quantiles[0, a, year_idx],
                quantiles[2, a, year_idx],
            ))
    diag_df = pd.DataFrame(diag)
    diag_df.to_csv(out / "diagnostics.csv", index=False)

    samples = {
        name: value.reshape((-1,) + value.shape[2:])
        for name, value in grouped.items()
    }
    params = []
    for name, labels in (("sigma_adm", None), ("sigma_sigma", None), ("sigma_mu", None),
                         ("omega_r", index["regions"]), ("eta_w", index["waves"]),
                         ("sigma_beta", index["series"])):
        v = np.asarray(samples[name])
        v = v[:, None] if v.ndim == 1 else v
        for j in range(v.shape[1]):
            params.append({"parameter": name, "label": labels[j] if labels else "",
                           "median": np.median(v[:, j]), "q05": np.quantile(v[:, j], 0.05),
                           "q95": np.quantile(v[:, j], 0.95)})
    pd.DataFrame(params).to_csv(out / "parameters.csv", index=False)

    est = pd.DataFrame(
        estimate_rows,
        columns=["iso3", "year", "age", "age_rel", "oos_median", "oos_q05", "oos_q95"],
    )
    est.to_csv(out / "estimates.csv", index=False)
    finite_r_hat = diag_df["r_hat"].replace([np.inf, -np.inf], np.nan).dropna()
    finite_n_eff = diag_df["n_eff"].replace([np.inf, -np.inf], np.nan).dropna()
    invalid_diagnostics = int((~np.isfinite(diag_df[["r_hat", "n_eff"]])).to_numpy().sum())
    max_r_hat = float(finite_r_hat.max()) if len(finite_r_hat) else None
    min_n_eff = float(finite_n_eff.min()) if len(finite_n_eff) else None
    divergence_count = int(np.asarray(extra_fields["diverging"]).sum())
    diagnostic_summary = {
        "max_r_hat": max_r_hat,
        "min_n_eff": min_n_eff,
        "r_hat_over_1_01": int((finite_r_hat > 1.01).sum()),
        "n_eff_under_400": int((finite_n_eff < 400).sum()),
        "invalid_diagnostics": invalid_diagnostics,
        "divergences": divergence_count,
    }
    convergence_passed = (
        max_r_hat is not None
        and max_r_hat <= 1.01
        and min_n_eff is not None
        and min_n_eff >= 400
        and invalid_diagnostics == 0
        and divergence_count == 0
    )
    diagnostic_summary["convergence_passed"] = convergence_passed
    (out / "run_info.json").write_text(json.dumps(
        vars(args) | {
            "minutes": (time.time() - t0) / 60,
            "posterior_draws": "posterior_draws.npz",
            "checkpoint_directory": "checkpoints",
            "reproduction_scope": index.get(
                "reproduction_scope",
                "partial: WIDE/Other excluded (single-age jackknife uncertainty unavailable)",
            ),
            "diagnostics": diagnostic_summary,
            "convergence_passed": convergence_passed,
        },
        indent=1,
    ))
    progress_path.write_text(
        f"Status: Complete\nUpdated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
        f"Sampling iterations: {args.samples:,}/{args.samples:,} per chain\n"
        f"Estimates: {len(est):,}\n",
        encoding="utf-8",
    )

    if not convergence_passed:
        print("WARNING: convergence thresholds not met; do not treat estimates as final.")
    print(f"max r_hat={max_r_hat}; min n_eff={min_n_eff}; divergences={divergence_count}; "
          f"wrote {len(est):,} estimates and {len(diag_df):,} scalar diagnostics -> {out}")


if __name__ == "__main__":
    main()
