"""Preliminary administrative-only fit for Paper 1 (SDG 4.1.4).

This module deliberately uses the existing NumPyro model with an empty survey
likelihood. It is a pipeline/model smoke test, not the final paper reproduction.
The input must contain country, iso3, year, age, oosc_rate and optionally region.
If no region column is supplied, all countries are placed in one temporary
region and a warning is printed; use the UIS SDG mapping before interpreting
regional variance parameters.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import sys

HERE = Path(__file__).resolve().parent
PYTHON_DIR = HERE.parent / "Python"
sys.path.insert(0, str(PYTHON_DIR))

from oosc_model_numpyro import ModelData, run_mcmc  # noqa: E402
from spline_basis import build_model_bases, ey_index, mu_anchor_index  # noqa: E402


def load_admin_input(path: Path, entry_age: int = 6) -> ModelData:
    df = pd.read_csv(path)
    required = {"country", "year", "age", "oosc_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["country", "year", "age", "oosc_rate"]).copy()
    df = df[np.isfinite(df["oosc_rate"])].copy()
    df = df[(df["age"] >= entry_age) & (df["oosc_rate"] >= -0.1)].copy()
    if df.empty:
        raise ValueError("No usable administrative observations remain after filtering.")

    countries = sorted(df["country"].astype(str).unique())
    country_id = {x: i for i, x in enumerate(countries)}
    df["country_id"] = df["country"].astype(str).map(country_id)
    if "region" not in df.columns:
        warnings.warn("No region column supplied: using one temporary region for a smoke test.")
        df["region"] = "TEMP_ALL"
    regions = sorted(df["region"].fillna("UNKNOWN").astype(str).unique())
    region_id = {x: i for i, x in enumerate(regions)}
    df["region_id"] = df["region"].fillna("UNKNOWN").astype(str).map(region_id)
    df["age_index"] = df["age"].astype(int) - entry_age
    a_max = int(df["age_index"].max()) + 1
    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    bases = build_model_bases(
        min_year=min(min_year, 2000), max_year=max(max_year, 2022), max_age=a_max
    )
    anchor_idx = mu_anchor_index(bases.ey_grid)
    ey_idx = ey_index(df["year"].to_numpy(), df["age"].to_numpy(), bases.ey_grid) - 1

    # Empty survey arrays are intentional. N_SERIES/N_WAVE are one-element
    # placeholders because the shared model still declares survey parameters.
    empty_i = np.empty(0, dtype=int)
    empty_f = np.empty(0, dtype=float)
    return ModelData(
        C=len(countries), R=len(regions), A_max=a_max, N_EY=len(bases.ey_grid),
        mu_anchor_idx=anchor_idx,
        B1=bases.B1, B5=bases.B5,
        region_of_country=np.array([region_id[df.loc[df.country == c, "region"].iloc[0]] for c in countries]),
        A_obs=df["oosc_rate"].to_numpy(float),
        A_country=df["country_id"].to_numpy(int),
        A_region=df["region_id"].to_numpy(int),
        A_age=df["age_index"].to_numpy(int), A_ey_idx=ey_idx,
        K_obs=empty_f, f_obs=empty_f, nu=empty_f,
        S_country=empty_i, S_age=empty_i, S_ey_idx=empty_i,
        S_series=empty_i, S_wave=empty_i, N_SERIES=1, N_WAVE=1,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("A. Data/A.2. Processed/paper1_total_model_input_candidate.csv"))
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--chains", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    data = load_admin_input(args.input)
    print(f"ADMIN-ONLY FIT: countries={data.C}, regions={data.R}, observations={len(data.A_obs)}")
    print("SURVEY OBSERVATIONS: 0 (intentional; DHS/IPUMS/MICS not included)")
    mcmc = run_mcmc(data, num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains, seed=args.seed)
    mcmc.print_summary(exclude_deterministic=False)
    out = Path("A. Data/A.2. Processed/admin_only_mcmc_summary.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("Paper 1 administrative-only preliminary fit\n")
        f.write(f"countries={data.C}, regions={data.R}, admin_observations={len(data.A_obs)}, survey_observations=0\n")
        f.write(str(mcmc.print_summary(exclude_deterministic=False)) + "\n")
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
