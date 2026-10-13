"""Harmonize MICS household-member (hl.sav) files into survey out-of-school rate
observations (the "MICS" source of Table 1, Appendix A.2.2).

Same method as harmonize_ipums_census.py / harmonize_dhs.py: weighted
single-age x sex rates of non-attendance during the current school year, with a
delete-one-cluster Jackknife SE (cluster = MICS cluster number, HH1).

MICS variable names change across rounds (age is HL4 in MICS2, HL5 in MICS3,
HL6 later; attendance is ED17/ED18, ED4, ED5 or ED9), so each round uses its
standard questionnaire names, with English/French/Spanish SPSS labels as a
fallback for surveys that renamed variables.

Choices the paper leaves open (documented, change here if needed):
- In school = any current-school-year attendance question answered 1 (yes).
  Out of school = "ever attended school" answered 2 (these children skip the
  attendance question) or the attendance question answered 2. Else dropped.
- Subnational MICS (folder name has a parenthesis, e.g. "KEN(Nyanza Province)")
  are excluded: they are not national rates for the country.
- Weight = hhweight (household weight, applied to members as in MICS tables).
- Year = survey year in the archive folder name; wave = MICS round (MICS2..6).
- Cells with a rate of exactly 0 or 1 are dropped (probit undefined in Eq. 2).

Input : A. Data/A.1. Raw Sources/3. Survey Data/1. MICS (1999-2019)/MICS_Datasets.zip
Output: A. Data/A.2. Processed/mics_oosc_observations.csv
        A. Data/A.2. Processed/mics_file_inventory.csv
"""
from __future__ import annotations

import io
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import pyreadstat

sys.path.insert(0, str(Path(__file__).parent))
from harmonize_ipums_census import AGE_MAX, AGE_MIN, jackknife_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "A. Data"
RAW = DATA / "A.1. Raw Sources" / "3. Survey Data" / "1. MICS (1999-2019)" / "MICS_Datasets.zip"
OUT_DIR = DATA / "A.2. Processed"
OUT = OUT_DIR / "mics_oosc_observations.csv"
INVENTORY = OUT_DIR / "mics_file_inventory.csv"

SUBNATIONAL_HINT = re.compile(r"(?i)provinc|region|selected")

# Standard MICS questionnaire names per round (first existing one wins); the
# attendance entry lists alternative sets, e.g. MICS2 split the question in two.
ROUND_NAMES = {
    "2": {"age": ["HL4"], "sex": ["HL3"], "cluster": ["HI1", "HH1"], "ever": ["ED15"],
          "attend": [["ED17", "ED18"], ["ED17"], ["ED18"]]},
    "3": {"age": ["HL5"], "sex": ["HL4"], "cluster": ["HH1", "HI1"], "ever": ["ED2"], "attend": [["ED4"]]},
    "4": {"age": ["HL6"], "sex": ["HL4"], "cluster": ["HH1"], "ever": ["ED3"], "attend": [["ED5"]]},
    "5": {"age": ["HL6"], "sex": ["HL4"], "cluster": ["HH1"], "ever": ["ED3"], "attend": [["ED5"]]},
    "6": {"age": ["HL6"], "sex": ["HL4"], "cluster": ["HH1"], "ever": ["ED4"], "attend": [["ED9"]]},
}
# label fallbacks (English / French / Spanish) for surveys that renamed variables
LABELS = {
    "age": re.compile(r"(?i)^(age|âge|edad|idade)\b"),
    "sex": re.compile(r"(?i)^(sex|sexe|sexo)\b"),
    "cluster": re.compile(r"(?i)cluster|grappe|conglomerado|segmento|UPM"),
    "ever": re.compile(r"(?i)ever attended|déjà fréquenté|jamais fréquenté|alguna vez asisti"),
    "weight": re.compile(r"(?i)household (sample )?weight|poids.*ménage|ponderaci.n.*hogar|peso.*hogar"),
}


def pick_hl(names: list[str]) -> str | None:
    """The household-member file (hl.sav, hlAL.sav, ...); prefer the national one."""
    hl = [n for n in names if re.search(r"(?i)(^|/)([a-z]{2})?hl([a-z]{2})?\.sav$", n)]
    national = [n for n in hl if not SUBNATIONAL_HINT.search(n)]
    return (national or hl or [None])[0]


def find_vars(cols: list[str], labels: list[str | None], rnd: str) -> dict:
    lab = {c: (l or "").strip() for c, l in zip(cols, labels)}
    up = {c.upper(): c for c in cols}
    names = ROUND_NAMES[rnd]

    def by_name(key):
        return next((up[n] for n in names[key] if n in up), None)

    def by_label(key):
        return next((c for c in cols if LABELS[key].search(lab[c])), None)

    v = {k: by_name(k) or by_label(k) for k in ("age", "sex", "cluster", "ever")}
    v["weight"] = up.get("HHWEIGHT") or up.get("HHWGT") or up.get("HLWEIGHT") or by_label("weight")
    v["attend"] = next(([up[n] for n in group] for group in names["attend"] if all(n in up for n in group)), [])
    return v


def open_survey(zip_bytes: bytes) -> tuple[zipfile.ZipFile, str]:
    """Find the hl file, descending into nested zips (national one first)."""
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    hl = pick_hl(z.namelist())
    if hl:
        return z, hl
    inner = [n for n in z.namelist() if n.lower().endswith(".zip")]
    inner.sort(key=lambda n: bool(SUBNATIONAL_HINT.search(n)))
    for n in inner:
        try:
            return open_survey(z.read(n))
        except FileNotFoundError:
            continue
    raise FileNotFoundError("no hl.sav")


def read_hl(zip_bytes: bytes, rnd: str) -> tuple[pd.DataFrame, dict, str]:
    inner, hl = open_survey(zip_bytes)
    with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as f:
        f.write(inner.read(hl))
        tmp = f.name
    try:
        try:
            _, meta = pyreadstat.read_sav(tmp, metadataonly=True)
            enc = None
        except pyreadstat.ReadstatError:  # a few old files have mis-declared text encodings
            enc = "latin1"
            _, meta = pyreadstat.read_sav(tmp, metadataonly=True, encoding=enc)
        v = find_vars(meta.column_names, meta.column_labels, rnd)
        missing = [k for k in ("weight", "cluster", "sex", "age") if not v[k]] + ([] if v["attend"] else ["attend"])
        if missing:
            raise KeyError(f"missing {missing}")
        use = [v[k] for k in ("weight", "cluster", "sex", "age", "ever") if v[k]] + v["attend"]
        df, _ = pyreadstat.read_sav(tmp, usecols=use, encoding=enc)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return df, v, hl


def process(df: pd.DataFrame, v: dict, meta: dict) -> tuple[list[dict], dict]:
    k = df[v["cluster"]].nunique()
    att = df[v["attend"]]
    in_school = (att == 1).any(axis=1)
    out = ~in_school & ((att == 2).any(axis=1) | ((df[v["ever"]] == 2) if v["ever"] else False))
    df = df.assign(oos=out.astype(float), valid=in_school | out)
    df = df[df["valid"] & df[v["age"]].between(AGE_MIN, AGE_MAX) & (df[v["weight"]] > 0)]
    rows = []
    sex = df[v["sex"]]
    for sex_label, mask in (("M", sex == 1), ("F", sex == 2), ("MF", sex.isin([1, 2]))):
        sub = df[mask]
        for age, idx in sub.groupby(v["age"]).indices.items():
            s = sub.iloc[idx]
            w = s[v["weight"]].to_numpy(float)
            r, se, n_cl = jackknife_ratio(s["oos"].to_numpy() * w, w, s[v["cluster"]].to_numpy(), k)
            rows.append({**meta, "age": int(age), "sex": sex_label, "n_persons": len(s),
                         "n_clusters": n_cl, "oos_rate": r, "se_jackknife": se})
    return rows, {"n_clusters_total": k, "n_school_age_valid": len(df)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(RAW)
    out, inv = [], []
    for name in sorted(archive.namelist()):
        folder = name.split("/")[1]
        m = re.match(r"([A-Z]{3})(\(.*\))?_(\d{4})_MICS(\d)", folder)
        info = {"folder": folder}
        if not m:
            inv.append({**info, "status": "unparsed folder name"})
            continue
        meta = {"iso3": m.group(1), "year": int(m.group(3)), "series": "MICS",
                "wave": f"MICS{m.group(4)}", "source": "MICS", "mics_folder": folder}
        if m.group(2):
            inv.append({**info, **meta, "status": "skipped: subnational"})
            continue
        try:
            df, v, hl = read_hl(archive.read(name), m.group(4))
            rows, extra = process(df, v, meta)
        except Exception as e:  # education module missing, unexpected layout, ...
            inv.append({**info, **meta, "status": f"skipped: {e}"})
            print(f"  {folder}: skipped ({e})", flush=True)
            continue
        out.extend(rows)
        inv.append({**info, **meta, **extra, "hl_file": hl, "attend_vars": "+".join(v["attend"]),
                    "ever_var": v["ever"] or "", "n_cells": len(rows), "status": "ok"})
        print(f"  {folder}: {len(rows)} cells (attend={'+'.join(v['attend'])}, ever={v['ever']})", flush=True)

    obs = pd.DataFrame(out)
    degenerate = (obs["oos_rate"] <= 0) | (obs["oos_rate"] >= 1)
    print(f"Dropping {int(degenerate.sum())} cells with rate exactly 0 or 1 (probit undefined)")
    obs[~degenerate].to_csv(OUT, index=False)
    pd.DataFrame(inv).to_csv(INVENTORY, index=False)
    print(f"Wrote {int((~degenerate).sum()):,} observations -> {OUT}")


if __name__ == "__main__":
    main()
