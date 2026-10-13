"""Assemble the full model input (admin + survey) for one sex population
(paper Section 3 / Appendix B; run separately for M, F, MF per Section 3.4).

- Relative age a = 1 at the country's minimum official primary entry age over
  the data years (Section 3: "we begin counting at the minimum entry age
  across all years"); ages kept up to the latest end of upper secondary.
- Survey observations (Eq. 2): K = probit(S), f = std-normal pdf at K,
  nu = clustered-Jackknife SE in real space. Series = DHS / MICS / Census,
  wave = DHS4-8, MICS2-6, Census (Census aggregated, as in Fig. 8).
- Regions: the 7 UIS SDG regions of Fig. 7.
- Entry-year grid always spans the 2000-2022 estimate window.
- WIDE ("Other") is not included: its available aggregates lack single-age
  observations and clustered-Jackknife standard errors. Fits are partial
  reproductions until a compatible likelihood is available.

Output: A. Data/A.2. Processed/model_input/model_input_<sex>.npz (arrays for
        ModelData) + model_observations_<sex>.csv (one row per observation)
        + model_index_<sex>.json (country/region/series/wave labels, grid)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from spline_basis import build_model_bases, ey_index, mu_anchor_index  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "A. Data"
PROC = DATA / "A.2. Processed"
UIS = DATA / "A.1. Raw Sources" / "2. Administrative Data" / "1. UIS Internal Database - Paper 1 Enrolment"
STRUCT = (DATA / "A.1. Raw Sources" / "2. Administrative Data" / "2. UIS API - School System Structure"
          / "uis_entry_age_duration_2000_2022.csv")
OUT_DIR = PROC / "model_input"

SURVEYS = {"DHS": "dhs_oosc_observations.csv", "MICS": "mics_oosc_observations.csv",
           "Census": "ipums_census_oosc_observations.csv"}
SDG_REGIONS = {  # the 7 regions of the paper's Figure 7
    "SDG_SSA": "Sub-Saharan Africa", "SDG_NAFR_WEST_AS": "Northern Africa and Western Asia",
    "SDG_CEN_SAS": "Central and Southern Asia", "SDG_EAS_SEAS": "Eastern and South-eastern Asia",
    "SDG_LAC": "Latin America and the Caribbean", "SDG_OC": "Oceania",
    "SDG_EUR_NAM": "Europe and Northern America",
}


def region_map() -> dict[str, str]:
    ref = pd.read_csv(UIS / "uis_country_group_reference.csv", header=None, dtype=str, encoding="utf-8-sig")
    ref[12] = ref[12].str.strip()
    links = pd.read_csv(UIS / "uis_country_group_links.csv", header=None, dtype=str)
    m = pd.read_csv(UIS / "uis_country_code_mapping.csv", header=None, dtype=str, encoding="utf-8-sig")
    iso = dict(zip(m[0], m[3]))
    out = {}
    for abr, name in SDG_REGIONS.items():
        code = ref.loc[ref[12] == abr, 0].iloc[0]
        for co in links.loc[links[0] == code, 1]:
            if co in iso:
                out[iso[co]] = name
    return out


def age_ranges() -> pd.DataFrame:
    s = pd.read_csv(STRUCT)
    s["end_age"] = s["entry_upsec"] + s["dur_upsec"] - 1
    g = s.groupby("iso3").agg(min_entry=("entry_primary", "min"), end_age=("end_age", "max"))
    return g.dropna().astype(int).reset_index()


def load_observations(sex: str) -> pd.DataFrame:
    admin = pd.read_csv(PROC / "admin_oosc_observations.csv")
    admin = admin[admin["sex"] == sex].assign(kind="admin", series="Admin", wave="Admin")
    parts = [admin[["iso3", "year", "age", "oos_rate", "kind", "series", "wave", "source"]]]
    for series, fname in SURVEYS.items():
        s = pd.read_csv(PROC / fname)
        s = s[(s["sex"] == sex) & s["se_jackknife"].notna() & (s["se_jackknife"] > 0)]
        s = s.assign(kind="survey", series=series)
        parts.append(s[["iso3", "year", "age", "oos_rate", "se_jackknife", "kind", "series", "wave", "source"]])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sex", choices=["MF", "M", "F"], default="MF")
    ap.add_argument("--countries", nargs="*", help="ISO3 subset (for test fits)")
    args = ap.parse_args()

    obs = load_observations(args.sex)
    obs = obs.merge(age_ranges(), on="iso3", how="inner")
    obs = obs[obs["age"].between(obs["min_entry"], obs["end_age"])]
    obs["region"] = obs["iso3"].map(region_map())
    dropped = sorted(obs.loc[obs["region"].isna(), "iso3"].unique())
    obs = obs[obs["region"].notna()]
    if args.countries:
        obs = obs[obs["iso3"].isin(args.countries)]
    obs["age_rel"] = obs["age"] - obs["min_entry"] + 1  # 1-indexed, a=1 at min entry age

    countries = sorted(obs["iso3"].unique())
    regions = sorted(obs["region"].unique())
    c_id = {c: i for i, c in enumerate(countries)}
    r_id = {r: i for i, r in enumerate(regions)}
    obs["c"] = obs["iso3"].map(c_id)
    obs["r"] = obs["region"].map(r_id)

    a_max = int(obs["age_rel"].max())
    # grid covers the paper's 2000-2022 estimate window even where data stop earlier
    bases = build_model_bases(min(int(obs["year"].min()), 2000), max(int(obs["year"].max()), 2022), a_max)
    mu_anchor_idx = mu_anchor_index(bases.ey_grid)
    obs["ey"] = ey_index(obs["year"].to_numpy(), obs["age_rel"].to_numpy(), bases.ey_grid) - 1

    adm = obs[obs["kind"] == "admin"]
    sur = obs[obs["kind"] == "survey"].copy()
    series = sorted(sur["series"].unique())
    waves = sorted(sur["wave"].unique())
    sur["K"] = norm.ppf(sur["oos_rate"])
    sur["f"] = norm.pdf(sur["K"])

    region_of_country = obs.drop_duplicates("iso3").set_index("iso3").loc[countries, "r"].to_numpy()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.sex + ("_" + "-".join(args.countries) if args.countries else "")
    np.savez_compressed(
        OUT_DIR / f"model_input_{tag}.npz",
        C=len(countries), R=len(regions), A_max=a_max, N_EY=len(bases.ey_grid),
        mu_anchor_idx=mu_anchor_idx,
        B1=bases.B1, B5=bases.B5, region_of_country=region_of_country,
        A_obs=adm["oos_rate"].to_numpy(float), A_country=adm["c"].to_numpy(int),
        A_region=adm["r"].to_numpy(int), A_age=adm["age_rel"].to_numpy(int) - 1, A_ey_idx=adm["ey"].to_numpy(int),
        K_obs=sur["K"].to_numpy(float), f_obs=sur["f"].to_numpy(float), nu=sur["se_jackknife"].to_numpy(float),
        S_country=sur["c"].to_numpy(int), S_age=sur["age_rel"].to_numpy(int) - 1, S_ey_idx=sur["ey"].to_numpy(int),
        S_series=sur["series"].map({s: i for i, s in enumerate(series)}).to_numpy(int),
        S_wave=sur["wave"].map({w: i for i, w in enumerate(waves)}).to_numpy(int),
        N_SERIES=len(series), N_WAVE=len(waves),
    )
    obs.to_csv(OUT_DIR / f"model_observations_{tag}.csv", index=False)
    index = {"sex": args.sex, "countries": countries, "regions": regions, "series": series, "waves": waves,
             "ey_grid": bases.ey_grid.tolist(), "A_max": a_max,
             "mu_anchor_year": 2000, "mu_anchor_idx": mu_anchor_idx,
             "reproduction_scope": "partial: WIDE/Other excluded (single-age jackknife uncertainty unavailable)",
             "min_entry": obs.drop_duplicates("iso3").set_index("iso3").loc[countries, "min_entry"].astype(int).tolist(),
             "dropped_no_region": dropped}
    (OUT_DIR / f"model_index_{tag}.json").write_text(json.dumps(index, indent=1))

    print(f"[{tag}] countries={len(countries)} regions={len(regions)} A_max={a_max} "
          f"entry years {bases.ey_grid.min()}-{bases.ey_grid.max()}")
    print(f"  admin obs={len(adm):,} ({int((adm['oos_rate'] < 0).sum()):,} negative)  survey obs={len(sur):,}")
    print("  " + sur.groupby("series").size().to_string().replace("\n", "  "))
    if dropped:
        print(f"  dropped (no SDG region): {dropped}")


if __name__ == "__main__":
    main()
