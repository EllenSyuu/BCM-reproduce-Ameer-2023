"""Build administrative out-of-school rate observations from the UIS internal
single-age enrolment extract (paper Section 2 / Appendix A.2.1).

Replaces prepare_total_model_input.py, which summed every single-grade row on
top of the "All grades" totals (double counting) and covered MF only.

Steps:
1. Keep FULL-AND PART-TIME pupils, public + private, one total per level:
   pre-primary, primary (All grades), secondary general (lower+upper, All
   grades), secondary vocational (lower+upper, else lower + upper), and
   post-secondary non-tertiary. Early childhood development (ISCED 01) is not
   counted -- the SDG 4.1.4 definition counts pre-primary, primary, secondary
   and higher.
2. Sex: the extract has MF and F only; M = MF - F level by level.
3. Exclusions (Appendix A.2.1), using UIS official entrance ages/durations:
   step 1 -- the age's main level count is missing (primary below the
   lower-secondary entry age, secondary general from it upward; applied to all
   ages, not only the official range) -> excluded; step 2 -- first
   lower-secondary age and the primary count is missing -> excluded.
   Step 3 (country-specific raised thresholds) needs per-country judgment and
   is not applied (tested 2026-10-09: an automatic version drops only 0.3% of
   cells and does not change agreement with the official series). Additionally
   zero school-age enrolment is treated as missing (drop_zero_enrolment). Age-band totals stored under one age (value, then empty
   ages, then values again) and open-ended buckets ("<10", "25+") are treated
   as missing before these rules. Secondary general is only reported as lower+upper combined,
   so the rules distinguish primary vs secondary only.
4. Rate = 1 - total enrolment / population (build_population.py output);
   observations below -0.1 dropped, other negatives kept (Section 2.1).

Output: A. Data/A.2. Processed/admin_oosc_observations.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "A. Data"
UIS = DATA / "A.1. Raw Sources" / "2. Administrative Data" / "1. UIS Internal Database - Paper 1 Enrolment"
ENROL = UIS / "uis_internal_enrolment_single_age_2000_2022.csv"  # SQL step 6: effectively 2000-2012
ENROL_RECENT = UIS / "uis_internal_enrolment_single_age_2013_2022.csv"  # SQL step 6b, if exported
STRUCT = (DATA / "A.1. Raw Sources" / "2. Administrative Data" / "2. UIS API - School System Structure"
          / "uis_entry_age_duration_2000_2022.csv")
POP = DATA / "A.2. Processed" / "population_single_age_sex.csv"
OUT = DATA / "A.2. Processed" / "admin_oosc_observations.csv"

NEGATIVE_CUTOFF = -0.1
AGES = (3, 25)
COLS = ["co_code", "year", "age", "sex", "grade", "emc", "desc", "value", "qual", "mag", "st", "abe", "td"]
LEVEL_PATTERNS = {
    "pre": r"FULL-AND PART-TIME-PRE-PRIMARY-(?:MF|F)-All prog\.-Age",
    "prim": r"FULL-AND PART-TIME-PRIMARY-(?:MF|F)-All prog\.-All grades",
    "sec_gen": r"FULL-AND PART-TIME-SECONDARY-Lower and Upper-(?:MF|F)-General -All grades",
    "sec_voc": r"FULL-AND PART-TIME-SECONDARY-Lower and Upper-(?:MF|F)-Vocational -All grades",
    "sec_voc_low": r"FULL-AND PART-TIME-SECONDARY-Lower-(?:MF|F)-Vocational -All grades",
    "sec_voc_up": r"FULL-AND PART-TIME-SECONDARY-Upper-(?:MF|F)-Vocational -All grades",
    "postsec": r"FULL-AND PART-TIME-Post-SECONDARY-(?:MF|F)-Vocational -",
}


def drop_band_totals(d: pd.DataFrame) -> pd.DataFrame:
    """Blank out age-band totals stored under a single age.

    Some countries report a level for an age band under its first age and leave
    the following ages empty (e.g. Benin 2011 secondary: 13 = 366k, 14-15 NULL,
    16 = 207k). A value followed by a missing age with reported ages again later
    is treated as a band total and set missing, so step 1 excludes it.
    """
    d = d.sort_values(["co_code", "year", "sex", "level", "age"])
    g = d.groupby(["co_code", "year", "sex", "level"])["value"]
    next_missing = g.shift(-1).isna() & (d.groupby(["co_code", "year", "sex", "level"])["age"].shift(-1) == d["age"] + 1)
    later_reported = g.transform(lambda s: s[::-1].notna().cumsum()[::-1].shift(-1, fill_value=0) > 0)
    band = d["value"].notna() & next_missing & later_reported
    d.loc[band, "value"] = np.nan
    return d


def load_levels() -> pd.DataFrame:
    """One row per country-year-age-sex, one column per level (NaN = missing)."""
    d = pd.read_csv(ENROL, header=None, names=COLS, dtype=str, keep_default_na=False)
    if ENROL_RECENT.exists():  # step 6b file supersedes the old extract from 2013 on
        recent = pd.read_csv(ENROL_RECENT, header=None, names=COLS, dtype=str, keep_default_na=False)
        recent = recent[recent["co_code"].str.isdigit()]  # drops a header row if one was exported
        d = pd.concat([d[d["year"].astype(int) <= 2012], recent], ignore_index=True)
    d = d[d["desc"].str.contains("FULL-AND PART-TIME") & d["sex"].isin(["1", "3"])]
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    d = d[d["age"].between(*AGES)]
    d["level"] = None
    for level, pat in LEVEL_PATTERNS.items():
        d.loc[d["desc"].str.contains(pat, regex=True), "level"] = level
    d = d[d["level"].notna()]
    # open-ended buckets ("Age <10", "Age 25+") are not single ages
    d = d[d["desc"].str.extract(r"Age ([^-]*)-?$")[0].str.fullmatch(r"\d+").fillna(False)]
    d["value"] = pd.to_numeric(d["value"].replace("NULL", np.nan), errors="coerce")
    d["sex"] = d["sex"].map({"1": "MF", "3": "F"})
    d = drop_band_totals(d)
    wide = d.pivot_table(index=["co_code", "year", "age", "sex"], columns="level", values="value",
                         aggfunc="first", dropna=False).reset_index()
    for c in LEVEL_PATTERNS:
        if c not in wide:
            wide[c] = np.nan
    # vocational: combined lower+upper total, else the sum of lower and upper
    voc_parts = wide[["sec_voc_low", "sec_voc_up"]].sum(axis=1, min_count=1)
    wide["sec_voc"] = wide["sec_voc"].fillna(voc_parts)
    wide = wide.drop(columns=["sec_voc_low", "sec_voc_up"])
    wide["year"] = wide["year"].astype(int)
    wide["age"] = wide["age"].astype(int)
    return wide


def add_male(wide: pd.DataFrame) -> pd.DataFrame:
    levels = ["pre", "prim", "sec_gen", "sec_voc", "postsec"]
    key = ["co_code", "year", "age"]
    mf = wide[wide["sex"] == "MF"].set_index(key)[levels]
    f = wide[wide["sex"] == "F"].set_index(key)[levels]
    both = mf.index.intersection(f.index)
    m = (mf.loc[both] - f.loc[both]).reset_index().assign(sex="M")  # NaN if either side missing
    return pd.concat([wide, m], ignore_index=True)


def apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """Appendix A.2.1 steps 1-2 on the official level age ranges."""
    # Step 1, applied to every age: below the lower-secondary entry age the
    # primary count must be reported, from it upward the secondary one --
    # otherwise a missing main level would be summed as 0 enrolment.
    s_lo = df["entry_lowsec"]
    step1 = ((df["age"] < s_lo) & df["prim"].isna()) | ((df["age"] >= s_lo) & df["sec_gen"].isna())
    step2 = (df["age"] == s_lo) & df["prim"].isna()
    df["exclusion"] = np.select([step1, step2], ["step1_own_level_missing", "step2_prior_level_missing"], "")
    return df


def drop_zero_enrolment(df: pd.DataFrame) -> pd.DataFrame:
    """Treat zero school-age enrolment as missing, not as 100% out of school.

    A reported 0 for a school-age cell (e.g. COD 2014/2018, ZAF) is a missing
    value coded as zero. The extract holds only MF and F (M = MF - F), so it can
    not be completed by matching: MF == 0 (or a rate of ~1) invalidates all three
    series for that country-year-age, and F == 0 with MF > 0 invalidates F and M
    (girls' enrolment missing) while MF stays usable. Only applied from the
    year's official primary entry age to the end of upper secondary.
    """
    key = ["iso3", "year", "age"]
    in_range = (df["age"] >= df["entry_primary"]) & (df["age"] <= df["entry_upsec"] + df["dur_upsec"] - 1)
    piv = df.pivot_table(index=key, columns="sex", values="enrolment", aggfunc="first")
    rate = df.pivot_table(index=key, columns="sex", values="oos_rate", aggfunc="first")
    bad_all = ((piv.get("MF") == 0) | (rate.get("MF") >= 0.999)).rename("bad_all")
    bad_fm = ((piv.get("F") == 0) & (piv.get("MF") > 0)).rename("bad_fm")
    flags = pd.concat([bad_all, bad_fm], axis=1).fillna(False).reset_index()
    df = df.merge(flags, on=key, how="left")
    hit = in_range & (df["exclusion"] == "") & (
        df["bad_all"].fillna(False) | (df["bad_fm"].fillna(False) & df["sex"].isin(["F", "M"])))
    df.loc[hit, "exclusion"] = "zero_enrolment_as_missing"
    return df.drop(columns=["bad_all", "bad_fm"])


def main() -> None:
    wide = add_male(load_levels())
    m = pd.read_csv(UIS / "uis_country_code_mapping.csv", header=None, dtype=str, encoding="utf-8-sig")
    iso = dict(zip(m[0], m[3]))
    wide["iso3"] = wide["co_code"].map(iso)
    struct = pd.read_csv(STRUCT)
    df = wide.merge(struct, on=["iso3", "year"], how="inner")
    df = apply_exclusions(df)

    levels = ["pre", "prim", "sec_gen", "sec_voc", "postsec"]
    df["enrolment"] = df[levels].sum(axis=1, min_count=1)
    df = df[df["enrolment"].notna()]
    pop = pd.read_csv(POP)
    df = df.merge(pop, on=["iso3", "year", "age", "sex"], how="inner")
    df["oos_rate"] = 1 - df["enrolment"] / df["population"]
    df = drop_zero_enrolment(df)

    n_all = len(df)
    kept = df[(df["exclusion"] == "") & (df["oos_rate"] >= NEGATIVE_CUTOFF)]
    kept = kept.assign(source="Admin")[["iso3", "year", "age", "sex", "enrolment", "population", "oos_rate",
                                        "entry_primary", "source"]]
    kept.to_csv(OUT, index=False)
    print(f"cells with enrolment + population: {n_all:,}")
    print(df["exclusion"].replace("", "kept").value_counts().to_string())
    print(f"below {NEGATIVE_CUTOFF}: {int(((df['exclusion'] == '') & (df['oos_rate'] < NEGATIVE_CUTOFF)).sum()):,}")
    print(f"Wrote {len(kept):,} observations ({kept['iso3'].nunique()} countries, "
          f"{int((kept['oos_rate'] < 0).sum()):,} negative) -> {OUT}")
    print(kept.groupby("sex").size().to_string())


if __name__ == "__main__":
    main()
