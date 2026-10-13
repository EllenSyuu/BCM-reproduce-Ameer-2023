"""Harmonize the WIDE country-survey extracts (the paper's "Other" source,
Appendix A.1 items a-p) into out-of-school rate observations.

IMPORTANT limitation: the WIDE files we have are published aggregates, not
microdata. They give out-of-school rates only by education level
(eduout_prim / eduout_lowsec / eduout_upsec) with a sample count (*_no), and no
single-age breakdown and no standard errors. The paper's model uses single-age
observations (Eq. 2), so these rows cannot enter the likelihood as-is: using
them needs either the microdata or a level-aggregated likelihood term (mapping
each level to its official age range). They are kept here in a model-ready
long format so that step can be added later.

Choices (documented, change here if needed):
- Only the survey years the paper cites (References / Appendix A.1) are kept.
- National rows only: category "Total" (-> MF) and "Sex" (-> F / M). Argentina
  (EPH, urban-only survey) uses its Location=Urban rows instead.
- SE = binomial sqrt(p(1-p)/n) from the WIDE sample count. This ignores the
  survey design effect (no clusters in the aggregates), so it understates the
  paper's clustered-Jackknife nu_j. Left empty where WIDE gives no count.
- Rates of exactly 0 or 1 are dropped (probit undefined in Eq. 2), as for the
  other sources.
- Papua New Guinea (HIES) and Colombia (ECV) are unavailable -- see
  docs/data-sources.md.

Input : A. Data/A.1. Raw Sources/3. Survey Data/4. WIDE/<country>/wide_*.csv
Output: A. Data/A.2. Processed/wide_oosc_observations.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "A. Data"
RAW = DATA / "A.1. Raw Sources" / "3. Survey Data" / "4. WIDE"
OUT = DATA / "A.2. Processed" / "wide_oosc_observations.csv"

# paper's cited years per country-survey (Appendix A.1 / References)
PAPER_YEARS = {
    ("ARG", "EPH"): (2004, 2012), ("ARM", "ILCS"): (2018, 2018), ("BOL", "EH"): (2019, 2019),
    ("BRA", "PNAD"): (2007, 2015), ("CHL", "CASEN"): (2000, 2015), ("CHN", "CFPS"): (2010, 2014),
    ("ECU", "ENEMDU"): (2018, 2018), ("SLV", "EHPM"): (2019, 2019), ("MEX", "ENIGH"): (2018, 2018),
    ("PRY", "EPHC"): (2019, 2019), ("PER", "ENAHO"): (2019, 2019), ("RUS", "RLMS-HSE"): (2000, 2019),
    ("TZA", "HBS"): (2017, 2017), ("URY", "ECH"): (2019, 2019),
}
LEVELS = {"prim": "primary", "lowsec": "lower_secondary", "upsec": "upper_secondary"}


def main() -> None:
    frames = []
    for f in sorted(RAW.glob("*/wide_*.csv")):
        frames.append(pd.read_csv(f, low_memory=False))
    d = pd.concat(frames, ignore_index=True)

    # Argentina's EPH covers urban areas only, so WIDE files its rates under
    # Location=Urban instead of Total/Sex -- use those as its national rows
    arg_urban = (d["iso_code"] == "ARG") & (d["location"] == "Urban")
    d.loc[arg_urban & (d["category"] == "Location"), "category"] = "Total"
    d.loc[arg_urban & (d["category"] == "Location & Sex"), "category"] = "Sex"
    d = d[d["category"].isin(["Total", "Sex"])].copy()
    d["sex"] = np.where(d["category"] == "Total", "MF", d["sex"].map({"Female": "F", "Male": "M"}))
    d = d[d["sex"].notna()]
    span = d.apply(lambda r: PAPER_YEARS.get((r["iso_code"], r["survey"])), axis=1)
    keep = [s is not None and s[0] <= y <= s[1] for s, y in zip(span, d["year"])]
    d = d[keep]

    rows = []
    for key, level in LEVELS.items():
        part = d[["iso_code", "country", "survey", "year", "sex", f"eduout_{key}_m", f"eduout_{key}_no"]]
        part = part.rename(columns={"iso_code": "iso3", f"eduout_{key}_m": "oos_rate",
                                    f"eduout_{key}_no": "n_persons"})
        rows.append(part.assign(level=level))
    obs = pd.concat(rows).dropna(subset=["oos_rate"])
    obs = obs[(obs["oos_rate"] > 0) & (obs["oos_rate"] < 1)]
    # some WIDE extracts (ARG, ECU, MEX, PER, SLV, URY) leave the sample count
    # blank -- keep the rate, SE stays empty
    n = obs["n_persons"].where(obs["n_persons"] > 0)
    obs["se_binomial"] = np.sqrt(obs["oos_rate"] * (1 - obs["oos_rate"]) / n)
    obs = obs.assign(series="Other", wave="Other", source="WIDE", observation_unit="education level")
    obs = obs.sort_values(["iso3", "year", "level", "sex"])
    obs.to_csv(OUT, index=False)

    print(f"Wrote {len(obs):,} level-level observations -> {OUT}")
    print(obs.groupby(["iso3", "survey"])["year"].agg(["min", "max", "nunique"]).to_string())
    missing = sorted(set(PAPER_YEARS) - set(zip(obs["iso3"], obs["survey"])))
    if missing:
        print(f"No rows in the paper's years for: {missing}")


if __name__ == "__main__":
    main()
