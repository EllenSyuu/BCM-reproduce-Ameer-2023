"""Harmonize DHS household-member recode (PR) files into survey out-of-school
rate observations (the "DHS" source of Table 1, Appendix A.2.2).

Same method as harmonize_ipums_census.py: weighted single-age x sex rates of
non-attendance during the current school year, sampling SE (nu_j in Eq. 2) from
a delete-one-cluster Jackknife -- here the cluster is the DHS PSU, as in the DHS
sampling manual the paper cites.

Choices the paper leaves open (documented, change here if needed):
- Out of school = hv121 == 0; in school = hv121 in {1, 2} (currently attending /
  attended at some time during the current school year). 8/9 dropped.
- De facto population (hv103 == 1), the DHS convention for education tables;
  avoids counting absent de jure members who are visitors elsewhere.
- Weight = hv005 / 1e6. PSU = hv021 when present, else hv001.
- Survey year and country from the DHS API (via the surv_id in the download URL
  list); wave = DHS phase from the file-name version digit (DHS4..DHS7).
- Cells with a rate of exactly 0 or 1 are dropped (probit undefined in Eq. 2),
  same rule as the IPUMS script.

Input : A. Data/A.1. Raw Sources/3. Survey Data/2. DHS (...)/??PR*FL.zip
Output: A. Data/A.2. Processed/dhs_oosc_observations.csv
        A. Data/A.2. Processed/dhs_file_inventory.csv
"""
from __future__ import annotations

import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from harmonize_ipums_census import AGE_MAX, AGE_MIN, jackknife_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "A. Data"
RAW = DATA / "A.1. Raw Sources" / "3. Survey Data" / "2. DHS (2000-2018, pending project approval)"
META = RAW / "dhs_survey_metadata.csv"  # file -> country/year, built once from the API
URL_LIST = Path.home() / "Downloads" / "dhs_PR_2000_2018_urls.txt"
OUT_DIR = DATA / "A.2. Processed"
OUT = OUT_DIR / "dhs_oosc_observations.csv"
INVENTORY = OUT_DIR / "dhs_file_inventory.csv"

VARS = ["hv001", "hv005", "hv021", "hv103", "hv104", "hv105", "hv121"]
API = "https://api.dhsprogram.com/rest/dhs/{}?f=json&perpage=5000"


def api(endpoint: str) -> list[dict]:
    req = urllib.request.Request(API.format(endpoint), headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=120))["Data"]


def build_metadata() -> pd.DataFrame:
    """Map each PR file to its survey (country ISO3, year) via surv_id in the URL list."""
    surveys = {str(s["SurveyNum"]): s for s in api("surveys")}
    iso = {c["DHS_CountryCode"]: c["ISO3_CountryCode"] for c in api("countries")}
    rows = []
    for u in URL_LIST.read_text().split():
        m = re.search(r"Filename=(\w+)\.zip.*surv_id=(\d+)", u)
        if m and m.group(2) in surveys:
            s = surveys[m.group(2)]
            rows.append({"file": m.group(1), "surv_id": int(m.group(2)), "dhs_country": s["DHS_CountryCode"],
                         "iso3": iso.get(s["DHS_CountryCode"], ""), "country": s["CountryName"],
                         "year": int(s["SurveyYear"]), "survey_id": s["SurveyId"]})
    return pd.DataFrame(rows)


def open_members(path: Path) -> tuple[bytes, str]:
    """Return (.DAT bytes, .DCT text); some DHS zips wrap a second zip."""
    z = zipfile.ZipFile(path)
    names = {n.upper(): n for n in z.namelist()}
    if not any(n.endswith(".DAT") for n in names):
        inner = next(n for n in names if n.endswith(".ZIP"))
        z = zipfile.ZipFile(io.BytesIO(z.read(names[inner])))
        names = {n.upper(): n for n in z.namelist()}
    dat = next(v for k, v in names.items() if k.endswith(".DAT"))
    dct = next(v for k, v in names.items() if k.endswith(".DCT"))
    return z.read(dat), z.read(dct).decode("latin1")


def read_pr(path: Path) -> pd.DataFrame:
    dat, dct = open_members(path)
    spec = {}
    for m in re.finditer(r"^\s*\w+\s+(\w+)\s+1:\s*(\d+)-(\d+)", dct, re.M):
        if m.group(1).lower() in VARS:
            spec[m.group(1).lower()] = (int(m.group(2)) - 1, int(m.group(3)))
    if not spec:  # older "infile" dictionaries list widths (%12s, %2f) in order instead
        pos = 0
        for m in re.finditer(r"^\s*\w+\s+(\w+)\s+%(\d+)[a-z]", dct, re.M):
            width = int(m.group(2))
            if m.group(1).lower() in VARS:
                spec[m.group(1).lower()] = (pos, pos + width)
            pos += width
    missing = {"hv001", "hv005", "hv104", "hv105", "hv121"} - spec.keys()
    if missing:
        raise KeyError(f"missing {sorted(missing)}")
    names = list(spec)
    df = pd.read_fwf(io.BytesIO(dat), colspecs=[spec[n] for n in names], names=names, header=None)
    return df.apply(pd.to_numeric, errors="coerce")


def process_file(df: pd.DataFrame, meta: dict) -> tuple[list[dict], dict]:
    psu = df["hv021"] if "hv021" in df and df["hv021"].notna().all() else df["hv001"]
    df = df.assign(psu=psu.to_numpy(), w=df["hv005"] / 1e6)
    k = df["psu"].nunique()  # all PSUs in the survey, for the Jackknife factor
    if "hv103" in df:
        df = df[df["hv103"] == 1]
    df = df[df["hv105"].between(AGE_MIN, AGE_MAX) & df["hv121"].isin([0, 1, 2])]
    oos = (df["hv121"] == 0).to_numpy(float)
    rows = []
    for sex_label, mask in (("M", df["hv104"] == 1), ("F", df["hv104"] == 2), ("MF", df["hv104"].isin([1, 2]))):
        sub, o = df[mask], oos[mask.to_numpy()]
        for age, idx in sub.groupby("hv105").indices.items():
            s = sub.iloc[idx]
            w = s["w"].to_numpy()
            r, se, n_cl = jackknife_ratio(o[idx] * w, w, s["psu"].to_numpy(), k)
            rows.append({**meta, "age": int(age), "sex": sex_label, "n_persons": len(s),
                         "n_clusters": n_cl, "oos_rate": r, "se_jackknife": se})
    return rows, {"n_psu": k, "n_school_age_valid": len(df)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not META.exists():
        build_metadata().to_csv(META, index=False)
    meta = pd.read_csv(META).set_index("file")
    out, inv = [], []
    for path in sorted(RAW.glob("*PR*FL.zip")):
        name = path.stem.upper()
        info = {"file": name}
        if name not in meta.index:
            inv.append({**info, "status": "no survey metadata"})
            continue
        m = meta.loc[name]
        obs_meta = {"iso3": m["iso3"], "country": m["country"], "survey_id": m["survey_id"],
                    "dhs_file": name, "year": int(m["year"]), "series": "DHS",
                    "wave": f"DHS{name[4]}", "source": "DHS"}
        try:
            rows, extra = process_file(read_pr(path), obs_meta)
        except Exception as e:  # e.g. hv121 not collected in this survey
            inv.append({**info, **obs_meta, "status": f"skipped: {e}"})
            print(f"  {name}: skipped ({e})", flush=True)
            continue
        out.extend(rows)
        inv.append({**info, **obs_meta, **extra, "n_cells": len(rows), "status": "ok"})
        print(f"  {name} ({m['iso3']} {int(m['year'])}): {len(rows)} cells", flush=True)

    obs = pd.DataFrame(out)
    degenerate = (obs["oos_rate"] <= 0) | (obs["oos_rate"] >= 1)
    print(f"Dropping {int(degenerate.sum())} cells with rate exactly 0 or 1 (probit undefined)")
    obs[~degenerate].to_csv(OUT, index=False)
    pd.DataFrame(inv).to_csv(INVENTORY, index=False)
    print(f"Wrote {int((~degenerate).sum()):,} observations -> {OUT}")


if __name__ == "__main__":
    main()
