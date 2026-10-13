"""Harmonize administrative enrolment counts into out-of-school rate observations.

Implements the three-step enrolment-count exclusion process from Appendix A.2.1
of the out-of-school rate paper, then computes rates and applies the Section 2.1
/ Appendix A.2.1 negative-observation cutoff (drop anything below -0.1, keep
everything else -- including negative values -- for the Stan model to use).

Data model (important): UIS enrolment data can report *multiple* education-level
rows for the *same* country-year-age-sex cell. A 12-year-old might appear both
as a "lower_secondary" enrolment count (on-time) and a "primary" enrolment count
(delayed, still repeating/finishing primary) in the same year. Out-of-school rate
computation sums across all reported levels for a given age (Appendix A.2.1
preamble: "these enrolment counts must be aggregated across education level for
each country-age-sex-year combination"). The exclusion rules below operate on
that same per-age-cell, multi-level-row structure -- NOT on a single row per age.

Input shape expected (one row per country-year-age-sex-level cell; a given
country-year-age-sex may have more than one row, one per level reported):
    country, year, age, sex, education_level, enrolment_count

This module is data-source-agnostic: it doesn't know how to *get* UIS
administrative counts (that's the blocked step, see
B. Documentation/2. Internal UIS Data Request.md)
-- it only knows what to do with them once you have a dataframe in the shape
above. Tested here against synthetic data so the exclusion logic itself is
verified ahead of getting real data.
"""
from __future__ import annotations

import pandas as pd

NEGATIVE_CUTOFF = -0.1  # paper Section 2.1 / Appendix A.2.1: drop rates below this


def apply_exclusion_rules(
    enrolment: pd.DataFrame, level_order: list[str], expected_level_by_age: dict[int, str]
) -> pd.DataFrame:
    """Apply the three-step enrolment-count exclusion process (Appendix A.2.1).

    Parameters
    ----------
    enrolment : DataFrame with columns
        ['country', 'year', 'age', 'sex', 'education_level', 'enrolment_count']
        Multiple rows may share the same (country, year, age, sex) -- one per
        education level reported at that age (on-time + delayed).
    level_order : ordered list of education levels, youngest-first, e.g.
        ['primary', 'lower_secondary', 'upper_secondary'].
    expected_level_by_age : maps age -> the "on-time" education level for a
        child of that age (e.g. {6: 'primary', ..., 11: 'primary',
        12: 'lower_secondary', ...}).

    Returns
    -------
    DataFrame with one row per (country, year, age, sex), the boolean
    `excluded` flag and a `reason` string, ahead of rate computation.
    Exclusion is decided at the cell level (country-year-age-sex), consistent
    with the paper: a missing own-level or prior-level count makes the whole
    age's aggregated total unreliable, not just one row of it.
    """
    df = enrolment.copy()
    cells = (
        df[["country", "year", "age", "sex"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    cells["excluded"] = False
    cells["reason"] = ""

    level_rank = {lvl: i for i, lvl in enumerate(level_order)}

    def has_level_count(country, year, age, sex, level) -> bool:
        row = df[
            (df["country"] == country)
            & (df["year"] == year)
            & (df["age"] == age)
            & (df["sex"] == sex)
            & (df["education_level"] == level)
        ]
        return bool(len(row)) and row["enrolment_count"].notna().any()

    for i, cell in cells.iterrows():
        country, year, age, sex = cell["country"], cell["year"], cell["age"], cell["sex"]
        own_level = expected_level_by_age[age]

        # Step 1: own expected-level enrolment count missing for this cell.
        if not has_level_count(country, year, age, sex, own_level):
            cells.loc[i, "excluded"] = True
            cells.loc[i, "reason"] = "step1_own_level_missing"
            continue

        # Step 2: age is the first expected age of its level, but the prior
        # level's count (delayed students still in the prior level, reported
        # at this same age) is missing.
        own_rank = level_rank[own_level]
        first_age_of_own_level = min(a for a, lvl in expected_level_by_age.items() if lvl == own_level)
        if own_rank > 0 and age == first_age_of_own_level:
            prior_level = level_order[own_rank - 1]
            if not has_level_count(country, year, age, sex, prior_level):
                cells.loc[i, "excluded"] = True
                cells.loc[i, "reason"] = "step2_prior_level_missing"

    # Step 3 (consistently large delayed population -> raise exclusion age
    # threshold for subsequent levels) requires a country-specific judgment
    # call the paper doesn't fully mechanize ("if enrolment counts for age 15
    # routinely include material levels of primary enrolment..."). Left as a
    # manual review step -- see B. Documentation/4. Model Notes.md item 6 /
    # README known gaps.
    return cells


def compute_rates(
    enrolment: pd.DataFrame,
    population: pd.DataFrame,
    level_order: list[str],
    expected_level_by_age: dict[int, str],
) -> pd.DataFrame:
    """Apply exclusion rules, sum enrolment across all reported levels per
    age, and compute out-of-school rate = 1 - total_enrolment / population."""
    cells = apply_exclusion_rules(enrolment, level_order, expected_level_by_age)
    kept_cells = cells.loc[~cells["excluded"], ["country", "year", "age", "sex"]]

    totals = (
        enrolment.merge(kept_cells, on=["country", "year", "age", "sex"], how="inner")
        .groupby(["country", "year", "age", "sex"], as_index=False)["enrolment_count"]
        .sum()
        .rename(columns={"enrolment_count": "total_enrolment"})
    )

    merged = totals.merge(population, on=["country", "year", "age", "sex"], how="inner")
    merged["out_of_school_rate"] = 1 - merged["total_enrolment"] / merged["population"]

    n_dropped_no_pop = len(totals) - len(merged)
    if n_dropped_no_pop:
        print(f"Warning: {n_dropped_no_pop} country-year-age-sex cells had enrolment "
              f"but no matching population estimate; dropped.")

    return merged


def apply_negative_cutoff(rates: pd.DataFrame, rate_col: str = "out_of_school_rate") -> pd.DataFrame:
    """Drop extreme negative observations (< -0.1); keep everything else,
    including remaining negatives, per Section 2.1's core modelling decision."""
    n_before = len(rates)
    out = rates.loc[rates[rate_col] >= NEGATIVE_CUTOFF].copy()
    n_dropped = n_before - len(out)
    print(f"Dropped {n_dropped}/{n_before} observations below the {NEGATIVE_CUTOFF} cutoff "
          f"({100 * n_dropped / max(n_before, 1):.1f}%).")
    return out


# --- self-test on synthetic data --------------------------------------------
LEVEL_ORDER = ["primary", "lower_secondary"]
EXPECTED_LEVEL_BY_AGE = {a: "primary" for a in range(6, 12)} | {a: "lower_secondary" for a in range(12, 14)}


def _make_synthetic_enrolment() -> pd.DataFrame:
    rows = []

    # Country A: clean. Each age has its own-level row; ages 12-13 (delayed)
    # also carry a small residual "primary" row, present every year.
    for year in [2019, 2020]:
        for age in range(6, 14):
            own = EXPECTED_LEVEL_BY_AGE[age]
            rows.append(dict(country="A", year=year, age=age, sex="F",
                              education_level=own, enrolment_count=1000 - age * 20))
            if age in (12, 13):
                rows.append(dict(country="A", year=year, age=age, sex="F",
                                  education_level="primary", enrolment_count=30))

    # Country B, 2020: primary *rows exist but with null counts* (the realistic
    # "missing" case -- UIS records the cell but the value is absent) for
    # every primary age, so step 1 should genuinely fire on all of them.
    # Lower_secondary ages (12-13) have their own-level count but no
    # delayed-primary row at all -- age 12 (first lower_secondary age) should
    # therefore also fall to step 2 (prior-level row absent); age 13 is not a
    # "first age of level" so step 2 doesn't apply to it.
    for age in range(6, 14):
        own = EXPECTED_LEVEL_BY_AGE[age]
        if own == "primary":
            rows.append(dict(country="B", year=2020, age=age, sex="F",
                              education_level="primary", enrolment_count=None))
        else:
            rows.append(dict(country="B", year=2020, age=age, sex="F",
                              education_level=own, enrolment_count=900 - age * 15))

    # Country B, 2019: normal reporting, including a delayed-primary row at
    # age 12 -- except that delayed-primary row is missing (null), which
    # should trigger step 2 specifically at age 12.
    for age in range(6, 14):
        own = EXPECTED_LEVEL_BY_AGE[age]
        rows.append(dict(country="B", year=2019, age=age, sex="F",
                          education_level=own, enrolment_count=950 - age * 15))
    # age 13 has a delayed-primary row present (so step2 shouldn't fire there)
    rows.append(dict(country="B", year=2019, age=13, sex="F",
                      education_level="primary", enrolment_count=20))
    # age 12's delayed-primary row is deliberately absent -> step 2 should fire

    return pd.DataFrame(rows)


def _make_synthetic_population() -> pd.DataFrame:
    rows = []
    for country in ["A", "B"]:
        for year in [2019, 2020]:
            for age in range(6, 14):
                rows.append(dict(country=country, year=year, age=age, sex="F",
                                  population=1050 - age * 10))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    enrolment = _make_synthetic_enrolment()
    population = _make_synthetic_population()

    cells = apply_exclusion_rules(enrolment, LEVEL_ORDER, EXPECTED_LEVEL_BY_AGE)
    print("Excluded cells:")
    print(cells.loc[cells["excluded"]].sort_values(["country", "year", "age"]).to_string(index=False))

    rates = compute_rates(enrolment, population, LEVEL_ORDER, EXPECTED_LEVEL_BY_AGE)
    print("\nComputed rates:")
    print(rates.sort_values(["country", "year", "age"]).to_string(index=False))

    final = apply_negative_cutoff(rates)
    print(f"\nFinal observation count: {len(final)}")

    # --- sanity checks ---
    # NOTE: every check below first asserts the selection is non-empty --
    # `.all()` / `.any()` on an empty Series is vacuously True/False and will
    # silently "pass" even if the row never existed in `cells` at all (this
    # bit an earlier version of this test: B/2020 primary ages had no row at
    # all rather than a null-valued row, so the step-1 assertion passed
    # without step 1 ever having fired).
    def _select(country, year, age):
        sel = cells[(cells["country"] == country) & (cells["year"] == year) & (cells["age"] == age)]
        assert len(sel) > 0, f"expected a cell for {country}/{year}/age {age}, found none"
        return sel

    for age in range(6, 12):
        assert _select("B", 2020, age)["excluded"].all(), \
            f"step 1 should exclude B/2020/age {age} (own-level primary missing)"

    assert _select("B", 2019, 12)["excluded"].all(), \
        "step 2 should exclude B/2019/age 12 (delayed-primary row missing at that age)"
    assert _select("B", 2020, 12)["excluded"].all(), \
        "step 2 should exclude B/2020/age 12 (no delayed-primary row reported at all)"
    assert not _select("B", 2020, 13)["excluded"].any(), \
        "age 13 should NOT be excluded -- not a first-age-of-level, has its own-level count"
    assert not _select("B", 2019, 13)["excluded"].any(), \
        "age 13 should NOT be excluded -- its delayed-primary row is present"

    a_cells = cells[cells["country"] == "A"]
    assert len(a_cells) == 16, f"expected 16 cells for country A, found {len(a_cells)}"
    assert not a_cells["excluded"].any(), "country A is clean data and should have no exclusions"

    print("\nAll self-checks passed.")
