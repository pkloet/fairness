"""
scrape.py  –  Scrape all rowing regatta data from time-team.nl and save as JSON.

Output structure:
    data/
        {regatta}/
            {year}_sat.json    # list of races, each race is a list of 8 slots (None if lane absent)
            {year}_sun.json

Usage:
    python scrape.py                        # scrape all regattas, all years
    python scrape.py --regatta arb          # one regatta, all years
    python scrape.py --regatta arb --year 2024   # one regatta, one year
    python scrape.py --since 2018           # all regattas from 2018 onwards
"""

import argparse
import json
import os
import time
import requests
from bs4 import BeautifulSoup

# ─── Config ──────────────────────────────────────────────────────────────────

REGATTAS = [
    "bvr",
    "voorjaarsregatta",
    "hollandia",
    "raceroei",
    "arb",
    "westelijke",
    "hollandbeker",
    "nsrf",
    "diyr",
]

FIRST_YEAR = 2010          # earliest year to try
import datetime
CURRENT_YEAR = datetime.date.today().year

REQUEST_DELAY = 0.3        # seconds between HTTP requests (be polite to the server)
REQUEST_TIMEOUT = 15       # seconds before giving up on a request

OUTPUT_DIR = "data"

# ─── URL helpers ─────────────────────────────────────────────────────────────

def regatta_index_url(regatta, year):
    if year <= 2017:
        return f"https://regatta.time-team.nl/{regatta}/{year}/results/heats.php"
    else:
        return f"https://regatta.time-team.nl/{regatta}/{year}/results/races.php"


def race_url(regatta, year, href):
    return f"https://regatta.time-team.nl/{regatta}/{year}/results/{href}"


# ─── HTTP helper ─────────────────────────────────────────────────────────────

def fetch(url):
    """GET a URL and return the response, or None on failure."""
    try:
        time.sleep(REQUEST_DELAY)
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp
        print(f"  [HTTP {resp.status_code}] {url}")
        return None
    except requests.RequestException as e:
        print(f"  [ERROR] {url} — {e}")
        return None


# ─── URL extraction ───────────────────────────────────────────────────────────

def extract_race_urls(index_url):
    """
    Parse the regatta index page and return (urls_sat, urls_sun).
    Each is a list of relative hrefs (e.g. 'h001.php' or 'r<uuid>.php').
    The first non-empty table of heat links = Saturday,
    the second = Sunday.
    """
    resp = fetch(index_url)
    if resp is None:
        return [], []

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table", class_="timeteam")

    urls_sat = []
    urls_sun = []

    for table in tables:
        urls = []
        for row in table.find_all("tr")[1:]:
            a = row.find("a")
            if a and "href" in a.attrs:
                link_text = str(a)
                if any(kw in link_text for kw in ["heat", "Heat", "voorwedstrijd"]):
                    urls.append(a["href"])
        if urls:
            if not urls_sat:
                urls_sat = urls
            elif not urls_sun:
                urls_sun = urls
                break  # we have both days, stop

    return urls_sat, urls_sun


# ─── Race extraction helpers ──────────────────────────────────────────────────

def has_exact_class(tag, tag_type, class_name):
    return tag.name == tag_type and tag.get("class") == [class_name]


def find_col_index(table, names):
    """Return the column index whose header matches any name in `names`, or -1."""
    header_cells = table.find("tr").find_all(["th", "td"])
    for i, cell in enumerate(header_cells):
        if any(name in cell.get_text(strip=True) for name in names):
            return i
    return -1


def sort_with_gaps(lanes, times, total_length=8):
    """
    Map lane numbers (1-based) to times, filling a fixed-length list with
    None wherever a lane is absent.
    """
    result = [None] * total_length
    for lane, t in zip(lanes, times):
        if 1 <= lane <= total_length:
            result[lane - 1] = t
    return result


def time_to_seconds(time_str):
    """Convert 'M:SS.cc' or 'M:SS,cc' to a float number of seconds."""
    time_str = time_str.replace(",", ".")
    minutes_part, rest = time_str.split(":")
    seconds_part, fraction_part = rest.split(".")
    return int(minutes_part) * 60 + int(seconds_part) + int(fraction_part) / 100


# ─── Race extraction ──────────────────────────────────────────────────────────

def extract_race(url):
    """
    Scrape a single race result page and return a list of length 8:
        [time_lane1, time_lane2, ..., time_lane8]
    Missing lanes are None. Returns None if the page is unusable.
    """
    resp = fetch(url)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    race_table = soup.find(lambda tag: has_exact_class(tag, "table", "timeteam"))
    if race_table is None:
        print(f"  [SKIP] No race table found: {url}")
        return None

    # Every other row (skip header rows)
    race_rows = race_table.find_all(
        lambda tag: tag.name == "tr" and tag.has_attr("class")
    )[::2]

    lane_index = find_col_index(race_table, ["baan", "lane"])
    if lane_index == -1:
        print(f"  [SKIP] No lane column found: {url}")
        return None

    lanes = []
    times = []

    for row in race_rows:
        cols = row.find_all("td")
        if not cols:
            continue

        # Lane
        lane_text = cols[lane_index].get_text(strip=True)
        if len(lane_text) < 2:
            continue
        try:
            lane = int(lane_text[1])
        except ValueError:
            continue

        # Time — second-to-last right-aligned column
        time_cols = row.find_all("td", style="text-align: right;")
        if len(time_cols) < 2 or times is None:
            times = None
            break

        time_str = time_cols[-2].get_text(strip=True)
        if len(time_str) not in {7, 8}:
            times = None
            break

        try:
            t = time_to_seconds(time_str)
        except Exception:
            times = None
            break

        # Sanity check: 5–12 minutes
        if not (300 < t < 720):
            times = None
            break

        lanes.append(lane)
        times.append(t)

    if times is None or not lanes:
        print(f"  [SKIP] Bad times: {url}")
        return None

    return sort_with_gaps(lanes, times)


# ─── Regatta extraction ───────────────────────────────────────────────────────

def extract_regatta(regatta, year):
    """
    Scrape all Saturday and Sunday heats for one regatta+year.
    Returns (races_sat, races_sun) — each a list of 8-element lists.
    """
    index_url = regatta_index_url(regatta, year)
    print(f"  Index: {index_url}")

    urls_sat, urls_sun = extract_race_urls(index_url)
    print(f"  Found {len(urls_sat)} Saturday, {len(urls_sun)} Sunday races")

    def scrape_day(urls):
        results = []
        for href in urls:
            url = race_url(regatta, year, href)
            result = extract_race(url)
            if result:
                results.append(result)
        return results

    return scrape_day(urls_sat), scrape_day(urls_sun)


# ─── JSON persistence ─────────────────────────────────────────────────────────

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Saved {len(data)} races → {path}")


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(regattas, years, force=False):
    """
    Scrape all combinations of regattas × years.
    Skips a year if the output files already exist (unless force=True).
    """
    total_saved = 0

    for regatta in regattas:
        for year in years:
            sat_path = os.path.join(OUTPUT_DIR, regatta, f"{year}_sat.json")
            sun_path = os.path.join(OUTPUT_DIR, regatta, f"{year}_sun.json")

            if not force and os.path.exists(sat_path) and os.path.exists(sun_path):
                print(f"[SKIP] {regatta}/{year} — already scraped (use --force to re-scrape)")
                continue

            print(f"\n[{regatta}/{year}]")

            # First check the index page exists (avoids scraping dead years)
            index_url = regatta_index_url(regatta, year)
            resp = fetch(index_url)
            if resp is None:
                print(f"  → Regatta/year not found, skipping")
                continue

            sat, sun = extract_regatta(regatta, year)

            if sat or sun:   # save even if one day is empty
                save_json(sat, sat_path)
                save_json(sun, sun_path)
                total_saved += 1
            else:
                print(f"  → No usable race data found")

    print(f"\nDone. Saved data for {total_saved} regatta-year combinations.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape rowing regatta data from time-team.nl")
    parser.add_argument("--regatta", nargs="+", choices=REGATTAS,
                        help="Regatta code(s) to scrape (default: all)")
    parser.add_argument("--year", type=int,
                        help="Single year to scrape")
    parser.add_argument("--since", type=int, default=FIRST_YEAR,
                        help=f"Scrape from this year onwards (default: {FIRST_YEAR})")
    parser.add_argument("--force", action="store_true",
                        help="Re-scrape even if output files already exist")
    args = parser.parse_args()

    regattas = args.regatta if args.regatta else REGATTAS
    years    = [args.year] if args.year else range(args.since, CURRENT_YEAR + 1)

    run(regattas, list(years), force=args.force)
