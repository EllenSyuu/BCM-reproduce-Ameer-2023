"""Fetch public population-denominator data for the out-of-school cohort model.

Covers the three sources named in the paper (Appendix A.1): WPP for most
countries, Eurostat for European countries, SingStat for Singapore. All three
are public / no registration required, so this is the unblocked half of data
acquisition -- run this before chasing DHS/MICS/IPUMS access or the UIS
administrative-data request (B. Documentation/2. Internal UIS Data Request.md).

Output: single-age population counts by country/year/sex, written under
A. Data/A.1. Raw Sources/1. Population Denominators/{1. WPP.../2. Eurostat.../3. SingStat...}/

Run from the project root (paths below are relative to it, not to this file).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

POP_DIR = Path("A. Data") / "A.1. Raw Sources" / "1. Population Denominators"
WPP_DIR = POP_DIR / "1. WPP (2024 revision, 1950-2023 estimates)"
EUROSTAT_DIR = POP_DIR / "2. Eurostat (1960-2025)"
SINGSTAT_DIR = POP_DIR / "3. SingStat (1957-2025, 5-year age bands only)"
for _d in (WPP_DIR, EUROSTAT_DIR, SINGSTAT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Some public statistics APIs (confirmed: SingStat) 403 requests without a
# browser-like User-Agent, apparently as basic bot-filtering rather than any
# real access restriction -- verified against the live API on 2026-09-11.
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}


def fetch_wpp(wpp_url: str | None = None, dest: Path = WPP_DIR / "wpp_population.csv") -> Path:
    """Download UN World Population Prospects single-age population estimates.

    WPP publishes a bulk CSV of single-age population by country/year/sex on
    population.un.org. The exact download path changes with each WPP revision
    (the paper uses the 2019 revision); check
    https://population.un.org/wpp/Download/Standard/CSV/ for the current
    "Population by 1-year age groups" file and pass its URL in explicitly.
    """
    if not wpp_url:
        raise ValueError(
            "Pass wpp_url= the current WPP single-age population CSV link from "
            "https://population.un.org/wpp/Download/Standard/CSV/ before running.\n"
            "The paper used the 2019 revision; a full reproduction should use that "
            "vintage specifically (not the latest WPP revision) to match published "
            "figures, unless deliberately re-basing to newer population estimates."
        )
    resp = requests.get(wpp_url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
    print(f"WPP data written to {dest}")
    return dest


def load_wpp_from_xlsx(
    xlsx_path: Path, dest: Path = WPP_DIR / "wpp_population.csv"
) -> Path:
    """Convert a manually-downloaded WPP "single age" bulk Excel file into our
    standard long-format population CSV.

    UN Population Division bulk downloads (population.un.org/wpp/Download) are
    served through a browser flow, not a stable anonymous-fetch URL (the UN
    Data Portal API equivalent, indicator 47, needs an API key we don't have).
    So the practical path is: download File POP/01-1 "Total population (both
    sexes combined) by single age..." manually from
    https://population.un.org/wpp/Download/Standard/CSV/, and point this
    function at wherever it landed (e.g. the Downloads folder) instead of
    fetching it ourselves.

    CONFIRMED (2026-09-14) against WPP2024_POP_F01_1_POPULATION_SINGLE_AGE_
    BOTH_SEXES.xlsx: the "Estimates" sheet is real long/wide hybrid data --
    one row per (location, year), one column per single age 0..99 plus a
    "100+" column (112 columns total: 11 metadata + 101 age columns). Values
    are in thousands. Verified against a real row (Singapore, 2020, ages
    6-19: sensible values in the tens of thousands, matching Singapore's
    actual population).

    NOTE on vintage: this loads whatever revision the file says it is (WPP2024
    in the case above), not necessarily the WPP2019 revision the paper used --
    the UN revises historical estimates with each release, so this won't be
    bit-exact against the paper's published figures. Flagged in
    B. Documentation/4. Model Notes.md as a known, accepted deviation rather
    than silently treated as equivalent.

    openpyxl's read_only worksheet.dimensions / max_column / max_row can
    under-report the true extent for large files with a stale dimension tag
    in the XML (confirmed on this file: reported max_column=12, max_row=18
    when the real data goes to column 112 and well over 200,000 rows) --
    always pass explicit max_col/max_row when iterating (generously
    oversized -- iteration still stops at the real last row), don't trust
    ws.max_column/ws.max_row.

    Filters to Type == "Country/Area" (drops World/Region/Subregion/Income
    Group/Development Group/SDG region aggregate rows and the "Estimates"
    sheet's own "Label/Separator" spacer rows) and to the "Estimates" sheet
    only (historical 1950-2023; the other sheets are future projection
    scenarios we don't need).
    """
    import csv

    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["Estimates"]

    header = None
    for row in ws.iter_rows(min_row=17, max_row=17, max_col=112, values_only=True):
        header = row
    age_labels = header[11:112]

    n_written = 0
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iso3", "country", "year", "age", "population_thousands"])
        for row in ws.iter_rows(min_row=18, max_row=2_000_000, max_col=112, values_only=True):
            if row[8] != "Country/Area":
                continue
            iso3 = row[5]
            if not iso3:
                continue
            country = row[2]
            year = row[10]
            for age_label, value in zip(age_labels, row[11:112]):
                if value is None:
                    continue
                writer.writerow([iso3, country, year, age_label, value])
                n_written += 1

    print(f"WPP data written to {dest} ({n_written} (country, year, age) records)")
    return dest


def fetch_eurostat(dest: Path = EUROSTAT_DIR / "eurostat_population.csv") -> Path:
    """Download Eurostat single-age population stock.

    Table: demo_pjan (population on 1 January by age, sex, and country). Uses
    the `eurostat` pip package if available (handles the SDMX/TSV parsing
    cleanly); falls back to a raw API request (JSON-stat) otherwise.
    """
    try:
        import eurostat  # type: ignore

        df = eurostat.get_data_df("demo_pjan")
        df.to_csv(dest, index=False)
    except ImportError:
        print(
            "Install the `eurostat` package (pip install eurostat) for a clean "
            "pull of table demo_pjan; falling back to raw API request."
        )
        resp = requests.get(
            "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/demo_pjan",
            params={"format": "JSON"},
            timeout=120,
        )
        resp.raise_for_status()
        dest.write_text(resp.text, encoding="utf-8")
    print(f"Eurostat data written to {dest}")
    return dest


def fetch_singstat(
    table_id: str = "M810011", dest: Path = SINGSTAT_DIR / "singstat_population.csv"
) -> Path:
    """Download and parse a SingStat Table Builder population table into real CSV rows.

    Default table M810011 ("Singapore Residents By Age Group, Ethnic Group And
    Sex") is CONFIRMED (2026-09-14) to only break age down into 5-year bands
    (0-4, 5-9, 10-14, ...), not single years -- not sufficient on its own for
    the model's single-age requirement. Since Singapore is only 1 of 192
    countries in the paper, this is a low-priority gap; revisit with a
    different `table_id` if a single-year-age SingStat table is found (check
    the "Interact with Our Data" tool under Population > Elderly, Youth and
    Sex Profile on singstat.gov.sg -- a human can search table titles there
    far faster than blind API guessing).

    Writes real tabular CSV (year, series, row_label, value), not raw JSON --
    an earlier version of this function wrote the raw API response text with
    a .csv extension, which opens as garbage/empty-looking in Excel.

    Paginates via `offset`: the API silently caps each response at 5000
    (row, year) data points regardless of the `limit` parameter's echoed
    value, splitting mid-row when a table has more than that (confirmed
    2026-09-14 -- M810011 alone needed 2 pages). Fetches pages until one
    comes back with fewer than `page_size` points, i.e. the last page.
    """
    page_size = 5000
    all_records = []
    offset = 0
    while True:
        resp = requests.get(
            f"https://tablebuilder.singstat.gov.sg/api/table/tabledata/{table_id}",
            headers=BROWSER_HEADERS,
            params={"offset": offset},
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()

        page_records = [
            {
                "seriesNo": row["seriesNo"],
                "rowText": row["rowText"],
                "year": col["key"],
                "value": col["value"],
            }
            for row in payload["Data"]["row"]
            for col in row["columns"]
        ]
        all_records.extend(page_records)
        if len(page_records) < page_size:
            break
        offset += page_size

    df = pd.DataFrame.from_records(all_records)
    df.to_csv(dest, index=False)
    print(f"SingStat table {table_id} written to {dest} ({len(df)} rows across "
          f"{offset // page_size + 1} page(s), {df['rowText'].nunique()} distinct row categories)")
    return dest


if __name__ == "__main__":
    # Driver -- fetch_wpp requires an explicit URL (see docstring), so it's
    # commented out by default rather than failing the whole run.
    # fetch_wpp(wpp_url="REPLACE_WITH_CURRENT_WPP_2019_REVISION_CSV_URL")
    fetch_eurostat()
    fetch_singstat()
