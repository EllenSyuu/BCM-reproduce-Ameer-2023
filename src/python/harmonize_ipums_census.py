"""Harmonize IPUMS-International census microdata into survey out-of-school rate
observations (the "Census" source of Table 1, Appendix A.1 / A.2.2).

Per Appendix A.2.2: school-attendance responses are aggregated with the survey
weights, and sampling errors (nu_j in Eq. 2) come from a clustered delete-one
Jackknife (DHS 2012 sampling manual). Per Appendix A.1, each IPUMS sample is
capped at 1,000,000 person records -- we reproduce that by keeping a
deterministic pseudo-random subset of whole households.

Choices the paper leaves open (documented, change here if needed):
- Out-of-school = SCHOOL in {2,3,4} (no / formerly / never attended); in school
  = SCHOOL 1. SCHOOL 0 (NIU) and 9 (unknown) are dropped from numerator and
  denominator.
- Jackknife cluster = household (SERIAL). IPUMS census samples are drawn by
  dwelling/household and carry no PSU, so the household is the sampling unit.
- Year window 1999-2019 (paper's overall data span). IPUMS v7.3 vs later
  releases cannot be told apart from the microdata; see the inventory output.
- Cells with a rate of exactly 0 or 1 are dropped: probit(S_j) in Eq. 2 is
  infinite there, and in this extract they are census coding artefacts (e.g.
  whole compulsory-age cohorts coded as attending), not measured rates.
- Ages 5-19 in calendar age; conversion to relative age (a = 1 at the
  country's minimum entry age) happens at model-input assembly, same as admin.

Input : A. Data/A.1. Raw Sources/3. Survey Data/3. IPUMS International (v7.3 census)/ipumsi_00003.csv.gz
Output: A. Data/A.2. Processed/ipums_census_oosc_observations.csv
        A. Data/A.2. Processed/ipums_sample_inventory.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "A. Data"
RAW = DATA / "A.1. Raw Sources" / "3. Survey Data" / "3. IPUMS International (v7.3 census)" / "ipumsi_00003.csv.gz"
COUNTRY_MAP = (DATA / "A.1. Raw Sources" / "2. Administrative Data"
               / "1. UIS Internal Database - Paper 1 Enrolment" / "uis_country_code_mapping.csv")
OUT_DIR = DATA / "A.2. Processed"
INVENTORY = OUT_DIR / "ipums_sample_inventory.csv"
OUT = OUT_DIR / "ipums_census_oosc_observations.csv"

YEAR_MIN, YEAR_MAX = 1999, 2019
AGE_MIN, AGE_MAX = 5, 19
MAX_RECORDS = 1_000_000  # Appendix A.1 cap per IPUMS sample
IN_SCHOOL, OUT_OF_SCHOOL = {1}, {2, 3, 4}
CHUNK = 5_000_000
SEED = 20220801


def household_uniform(serial: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Deterministic U(0,1) per household (splitmix64 hash of SAMPLE, SERIAL)."""
    x = serial.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    x ^= sample.astype(np.uint64) + np.uint64(SEED)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def jackknife_ratio(y: np.ndarray, w: np.ndarray, cluster: np.ndarray, k: int) -> tuple[float, float, int]:
    """Weighted rate sum(y)/sum(w) with delete-one-cluster Jackknife SE.

    var = (k-1)/k * sum_i (r_(-i) - r)^2 over all k clusters in the sample;
    clusters with no members in this cell have r_(-i) = r and contribute 0.
    """
    df = pd.DataFrame({"c": cluster, "y": y, "w": w}).groupby("c", sort=False).sum()
    Y, W = df["y"].sum(), df["w"].sum()
    r = Y / W
    rest = W - df["w"].to_numpy()
    if len(df) < 2 or (rest <= 0).any():
        return r, np.nan, len(df)
    r_del = (Y - df["y"].to_numpy()) / rest
    return r, float(np.sqrt((k - 1) / k * np.sum((r_del - r) ** 2))), len(df)


def read_chunks(usecols):
    return pd.read_csv(RAW, usecols=usecols, chunksize=CHUNK, compression="gzip",
                       dtype={"PERWT": "float64"}, low_memory=False)


def build_inventory() -> pd.DataFrame:
    """Pass 1: per-sample person counts and SCHOOL coverage at ages 5-19."""
    parts = []
    for i, ch in enumerate(read_chunks(["COUNTRY", "YEAR", "SAMPLE", "AGE", "SCHOOL"])):
        age_ok = ch["AGE"].between(AGE_MIN, AGE_MAX)
        g = ch.assign(
            persons=1,
            school_age=age_ok.astype(int),
            school_valid=(age_ok & ch["SCHOOL"].isin(IN_SCHOOL | OUT_OF_SCHOOL)).astype(int),
        ).groupby(["SAMPLE", "COUNTRY", "YEAR"])[["persons", "school_age", "school_valid"]].sum()
        parts.append(g)
        print(f"  pass 1 chunk {i + 1}: {len(ch):,} rows", flush=True)
    inv = pd.concat(parts).groupby(level=[0, 1, 2]).sum().reset_index()
    inv["school_valid_share"] = inv["school_valid"] / inv["school_age"].where(inv["school_age"] > 0)
    inv["in_year_window"] = inv["YEAR"].between(YEAR_MIN, YEAR_MAX)
    inv["usable"] = inv["in_year_window"] & (inv["school_valid_share"] > 0.5)
    inv["keep_fraction"] = np.minimum(1.0, MAX_RECORDS / inv["persons"])
    return inv


def process_sample(df: pd.DataFrame, n_households: int, meta: dict) -> list[dict]:
    rows = []
    oos = df["SCHOOL"].isin(OUT_OF_SCHOOL).to_numpy(float)
    for sex_label, mask in (("M", df["SEX"] == 1), ("F", df["SEX"] == 2), ("MF", df["SEX"].isin([1, 2]))):
        sub = df[mask]
        o = oos[mask.to_numpy()]
        for age, idx in sub.groupby("AGE").indices.items():
            s = sub.iloc[idx]
            w = s["PERWT"].to_numpy()
            r, se, n_cl = jackknife_ratio(o[idx] * w, w, s["SERIAL"].to_numpy(), n_households)
            rows.append({**meta, "age": int(age), "sex": sex_label, "n_persons": len(s),
                         "n_households": n_cl, "oos_rate": r, "se_jackknife": se})
    return rows


def build_observations(inv: pd.DataFrame, iso: dict) -> pd.DataFrame:
    """Pass 2: household-capped subsample, weighted rates, Jackknife SEs."""
    use = inv[inv["usable"]].set_index("SAMPLE")
    frac = use["keep_fraction"].to_dict()
    cols = ["COUNTRY", "YEAR", "SAMPLE", "SERIAL", "PERWT", "AGE", "SEX", "SCHOOL"]
    buf: dict[int, list[pd.DataFrame]] = {}
    hh: dict[int, set] = {}
    out: list[dict] = []

    def flush(s):
        df = pd.concat(buf.pop(s))
        n_hh = len(hh.pop(s))
        c = int(use.at[s, "COUNTRY"])
        meta = {"iso3": iso.get(c, ""), "ipums_country": c, "ipums_sample": s,
                "year": int(use.at[s, "YEAR"]), "series": "Census", "wave": "Census",
                "source": "IPUMS-I"}
        out.extend(process_sample(df, n_hh, meta))
        print(f"  sample {s} ({meta['iso3']} {meta['year']}): {len(df):,} age-{AGE_MIN}-{AGE_MAX} records, "
              f"{n_hh:,} households", flush=True)

    for i, ch in enumerate(read_chunks(cols)):
        ch = ch[ch["SAMPLE"].isin(frac.keys())]
        if len(ch):
            f = ch["SAMPLE"].map(frac).to_numpy()
            ch = ch[household_uniform(ch["SERIAL"].to_numpy(), ch["SAMPLE"].to_numpy()) < f]
            for s, g in ch.groupby("SAMPLE"):
                hh.setdefault(s, set()).update(g["SERIAL"].unique().tolist())
                g = g[g["AGE"].between(AGE_MIN, AGE_MAX) & g["SCHOOL"].isin(IN_SCHOOL | OUT_OF_SCHOOL)]
                buf.setdefault(s, []).append(g)
            # file is sorted by SAMPLE: everything but the chunk's last sample is complete
            last = ch["SAMPLE"].iloc[-1]
            for s in [s for s in buf if s != last]:
                flush(s)
        print(f"  pass 2 chunk {i + 1}", flush=True)
    for s in list(buf):
        flush(s)
    return pd.DataFrame(out)


def load_iso_map() -> dict[int, str]:
    m = pd.read_csv(COUNTRY_MAP, header=None, dtype=str, encoding="utf-8-sig")
    m = m[pd.to_numeric(m[4], errors="coerce").notna()]
    return {int(k): v for k, v in zip(m[4].astype(int), m[3])}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    iso = load_iso_map()
    if INVENTORY.exists() and "--rebuild" not in sys.argv:
        inv = pd.read_csv(INVENTORY)
    else:
        print("Pass 1: sample inventory")
        inv = build_inventory()
        inv.insert(0, "iso3", inv["COUNTRY"].map(iso))
        inv.to_csv(INVENTORY, index=False)
    print(f"{len(inv)} samples in extract, {int(inv['usable'].sum())} usable "
          f"({YEAR_MIN}-{YEAR_MAX}, SCHOOL present)")
    if "--inventory-only" in sys.argv:
        return
    print("Pass 2: observations")
    obs = build_observations(inv, iso)
    degenerate = (obs["oos_rate"] <= 0) | (obs["oos_rate"] >= 1)
    print(f"Dropping {int(degenerate.sum())} cells with rate exactly 0 or 1 (probit undefined)")
    obs = obs[~degenerate]
    obs.to_csv(OUT, index=False)
    print(f"Wrote {len(obs):,} observations -> {OUT}")


if __name__ == "__main__":
    main()
