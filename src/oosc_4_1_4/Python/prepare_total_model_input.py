"""Prepare the first Paper 1 TOTAL/MF model-input candidate.

This is deliberately conservative: it keeps only MF pupil rows that describe
full- and part-time enrolment, all programmes, and excludes ECD duplicates.
Negative values are retained for Appendix A.2.1 cleaning.
"""
from collections import defaultdict
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "A. Data"
INTERNAL = DATA / "A.1. Raw Sources" / "2. Administrative Data" / "3. UIS Internal Database - Paper 1 Enrolment"
POP = DATA / "A.1. Raw Sources" / "1. Population Denominators" / "1. WPP (2024 revision, 1950-2023 estimates)" / "wpp_population.csv"
ENROL = INTERNAL / "uis_internal_enrolment_single_age_2000_2022.csv"
COUNTRY = INTERNAL / "uis_country_code_mapping.csv"
OUT = DATA / "A.2. Processed" / "paper1_total_model_input_candidate.csv"
REPORT = DATA / "A.2. Processed" / "paper1_total_model_input_candidate_report.txt"

OUT.parent.mkdir(parents=True, exist_ok=True)

country = {}
with COUNTRY.open(encoding="utf-8-sig", newline="") as f:
    for r in csv.reader(f):
        if len(r) >= 6 and r[0].strip().isdigit():
            country[int(r[0])] = (r[1], r[3], r[5])

enrol = defaultdict(float)
enrol_rows = 0
negative = 0
with ENROL.open(encoding="utf-8-sig", newline="") as f:
  for r in csv.reader(f):
      if len(r) < 13:
          continue
      try:
          co, year, age = int(r[0]), int(r[1]), int(r[2])
          value = float(r[7]) if r[7].strip().upper() not in ("", "NULL") else None
      except ValueError:
          continue
      desc = r[6].upper()
      if not (2000 <= year <= 2022 and 6 <= age <= 19 and value is not None):
          continue
      if "PUPILS" not in desc or "-MF-" not in desc:
          continue
      if "FULL-AND PART-TIME" not in desc or "ALL PROG" not in desc:
          continue
      if "EARLY CHILDHOOD DEV" in desc:
          continue
      enrol[(co, year, age)] += value
      enrol_rows += 1
      negative += value < 0

population = {}
with POP.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            year, age = int(r["year"]), int(r["age"])
            if 2000 <= year <= 2022 and 6 <= age <= 19:
                population[(r["iso3"], year, age)] = float(r["population_thousands"]) * 1000
        except (KeyError, ValueError):
            continue

matched = 0
missing_pop = 0
with OUT.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["co_code", "country", "iso3", "year", "age", "sex", "enrolment_count", "population", "oosc_rate"])
    for (co, year, age), n in sorted(enrol.items()):
        name, iso3, _ = country.get(co, ("", "", ""))
        pop = population.get((iso3, year, age))
        if pop is None:
            missing_pop += 1
            continue
        matched += 1
        w.writerow([co, name, iso3, year, age, "MF", n, pop, 1 - n / pop if pop else ""])

REPORT.write_text(
    f"enrolment_rows_selected={enrol_rows}\n"
    f"negative_enrolment_values={negative}\n"
    f"country_age_year_cells={len(enrol)}\n"
    f"matched_population_cells={matched}\n"
    f"missing_population_cells={missing_pop}\n"
    "population_source=WPP; population_thousands converted to persons\n"
    "region_mapping=not assigned; GPE groups are not substituted for SDG/M49 regions\n",
    encoding="utf-8",
)
print(OUT)
print(REPORT)
