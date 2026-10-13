"""Download DHS files from a dhsprogram.com URL list using your own browser login.

DHS download links only work inside a logged-in session, so this script reuses
the session cookie from your browser instead of a download manager.

One-time setup (per browser session, no install needed):
  1. Log in at https://dhsprogram.com/data/dataset_admin/login_main.cfm in Edge.
  2. Press F12 -> Network tab -> reload the page -> click the first request ->
     Headers -> Request Headers -> copy the whole value of "Cookie".
  3. Paste it as a single line into %USERPROFILE%\\.dhs_cookie.txt
     (kept outside the project so it is never shared or committed).

Run:
  .venv\\Scripts\\python.exe download_dhs.py [url_list.txt]
Files already downloaded are skipped, so it can be re-run after a session expires.
"""
from __future__ import annotations

import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "A. Data" / "A.1. Raw Sources" / "3. Survey Data" / "2. DHS (2000-2018, pending project approval)"
URL_LIST = Path.home() / "Downloads" / "dhs_PR_2000_2018_urls.txt"
COOKIE = next((p for p in (Path.home() / ".dhs_cookie.txt", Path.home() / "Downloads" / ".dhs_cookie.txt")
               if p.exists()), Path.home() / ".dhs_cookie.txt")


def main() -> None:
    url_list = Path(sys.argv[1]) if len(sys.argv) > 1 else URL_LIST
    if not COOKIE.exists():
        sys.exit(f"Missing {COOKIE} -- see setup steps at the top of this script.")
    cookie = COOKIE.read_text(encoding="utf-8").strip()
    urls = [u.strip() for u in url_list.read_text().splitlines() if "Filename=" in u]
    DEST.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    for i, url in enumerate(urls, 1):
        name = re.search(r"Filename=([^&]+)", url).group(1)
        target = DEST / name
        if target.exists() and target.stat().st_size > 0:
            done += 1
            continue
        req = urllib.request.Request(url, headers={"Cookie": cookie, "User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
        except Exception as e:  # network hiccup: report and move on
            print(f"[{i}/{len(urls)}] {name}: ERROR {e}")
            failed += 1
            continue
        if not data.startswith(b"PK"):  # got the login page instead of a zip
            sys.exit(f"[{i}/{len(urls)}] {name}: not a zip -- session cookie expired or wrong. "
                     f"Refresh {COOKIE} and re-run (finished files are kept).")
        target.write_bytes(data)
        done += 1
        print(f"[{i}/{len(urls)}] {name}: {len(data) / 1e6:.1f} MB")
        time.sleep(1)  # be polite to the DHS server
    print(f"Done: {done}/{len(urls)} present, {failed} errors -> {DEST}")


if __name__ == "__main__":
    main()
