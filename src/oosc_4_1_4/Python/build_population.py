"""Build the single-age x sex population denominators for the administrative
out-of-school rates (paper Section 2 / Appendix A.1).

Source per country, as in the paper:
- Eurostat (demo_pjan, 1 January) for European countries (UIS SDG: Europe
  group) where Eurostat has the country-year-age-sex cell;
- SingStat (M810801, residents, end June) for Singapore;
- WPP (2024 revision, 1 July, by sex) for everything else and as fallback.

Output: A. Data/A.2. Processed/population_single_age_sex.csv
        columns iso3, year, age, sex (MF/M/F), population (persons), source
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "A. Data"
POP = DATA / "A.1. Raw Sources" / "1. Population Denominators"
WPP = POP / "1. WPP (2024 revision, 1950-2023 estimates)" / "WPP2024_PopulationBySingleAgeSex_Medium_1950-2023.csv.gz"
EUROSTAT = POP / "2. Eurostat (1960-2025)" / "eurostat_population.csv"
SINGSTAT = POP / "3. SingStat (M810801, single-age residents, 1980-2025)" / "M810801-table.csv"
UIS = DATA / "A.1. Raw Sources" / "2. Administrative Data" / "1. UIS Internal Database - Paper 1 Enrolment"
OUT = DATA / "A.2. Processed" / "population_single_age_sex.csv"

YEARS = (1999, 2022)
AGES = (3, 25)


def uis_country_table() -> pd.DataFrame:
    m = pd.read_csv(UIS / "uis_country_code_mapping.csv", header=None, dtype=str, encoding="utf-8-sig")
    return m.rename(columns={0: "co_code", 3: "iso3", 4: "iso_num", 5: "iso2"})[["co_code", "iso3", "iso_num", "iso2"]]


def sdg_europe_iso3() -> set[str]:
    ref = pd.read_csv(UIS / "uis_country_group_reference.csv", header=None, dtype=str, encoding="utf-8-sig")
    code = ref.loc[ref[12].str.strip() == "SDG_EUR", 0].iloc[0]
    links = pd.read_csv(UIS / "uis_country_group_links.csv", header=None, dtype=str)
    co = set(links.loc[links[0] == code, 1])
    m = uis_country_table()
    return set(m.loc[m["co_code"].isin(co), "iso3"])


def load_wpp() -> pd.DataFrame:
    d = pd.read_csv(WPP, usecols=["ISO3_code", "LocTypeName", "Time", "AgeGrpStart", "PopMale", "PopFemale", "PopTotal"],
                    dtype={"ISO3_code": str, "LocTypeName": str}, low_memory=False)
    d = d[(d["LocTypeName"] == "Country/Area") & d["Time"].between(*YEARS) & d["AgeGrpStart"].between(*AGES)]
    d = d.rename(columns={"ISO3_code": "iso3", "Time": "year", "AgeGrpStart": "age",
                          "PopTotal": "MF", "PopMale": "M", "PopFemale": "F"})
    long = d.melt(id_vars=["iso3", "year", "age"], value_vars=["MF", "M", "F"], var_name="sex", value_name="population")
    long["population"] *= 1000  # thousands -> persons
    return long.assign(source="WPP2024")


def load_eurostat(europe: set[str]) -> pd.DataFrame:
    d = pd.read_csv(EUROSTAT)
    d = d.rename(columns={"geo\\TIME_PERIOD": "geo"})
    d = d[d["age"].str.fullmatch(r"Y\d+") & d["sex"].isin(["T", "M", "F"])]
    iso2_to_3 = dict(zip(uis_country_table()["iso2"], uis_country_table()["iso3"]))
    iso2_to_3.update({"EL": "GRC", "UK": "GBR", "XK": "XKX"})  # Eurostat's own codes
    d["iso3"] = d["geo"].map(iso2_to_3)
    d = d[d["iso3"].isin(europe)]
    years = [c for c in d.columns if re.fullmatch(r"\d{4}", str(c)) and YEARS[0] <= int(c) <= YEARS[1]]
    long = d.melt(id_vars=["iso3", "age", "sex"], value_vars=years, var_name="year", value_name="population")
    long = long.dropna(subset=["population"])
    long["age"] = long["age"].str[1:].astype(int)
    long["year"] = long["year"].astype(int)
    long["sex"] = long["sex"].map({"T": "MF", "M": "M", "F": "F"})
    long = long[long["age"].between(*AGES)]
    return long.assign(source="Eurostat")[["iso3", "year", "age", "sex", "population", "source"]]


def load_singstat() -> pd.DataFrame:
    raw = pd.read_csv(SINGSTAT, header=None, dtype=str, skip_blank_lines=False)
    header_row = raw.index[raw[0].fillna("").str.strip() == "Data Series"][0]
    years = raw.iloc[header_row, 1:].str.strip()
    section = {"Total Residents": "MF", "Total Male Residents": "M", "Total Female Residents": "F"}
    rows, current = [], None
    for _, r in raw.iloc[header_row + 1:].iterrows():
        label = str(r[0]).strip()
        if label in section:
            current = section[label]
        elif label.startswith("Total"):
            current = None  # ethnic-group sections: not needed
        elif current and (m := re.fullmatch(r"(\d+) Years?", label)):
            for y, v in zip(years, r[1:]):
                if y and str(y).isdigit() and pd.notna(v) and str(v).strip() not in ("", "na", "-"):
                    rows.append({"iso3": "SGP", "year": int(y), "age": int(m.group(1)), "sex": current,
                                 "population": float(v)})
    d = pd.DataFrame(rows)
    d = d[d["year"].between(*YEARS) & d["age"].between(*AGES)]
    return d.assign(source="SingStat")


def main() -> None:
    wpp = load_wpp()
    eu = load_eurostat(sdg_europe_iso3())
    sg = load_singstat()
    key = ["iso3", "year", "age", "sex"]
    override = pd.concat([eu, sg]).drop_duplicates(key)
    wpp = wpp.merge(override[key], on=key, how="left", indicator=True)
    wpp = wpp[wpp["_merge"] == "left_only"].drop(columns="_merge")
    pop = pd.concat([override, wpp], ignore_index=True).sort_values(key)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pop.to_csv(OUT, index=False)
    print(f"Wrote {len(pop):,} rows -> {OUT}")
    print(pop.groupby("source")["iso3"].nunique().rename("countries").to_string())


if __name__ == "__main__":
    main()
