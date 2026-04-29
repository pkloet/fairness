"""
parse_schedule.py  –  Scrape the time-team.nl regatta calendar and save
                      data/schedule.json with upcoming race dates for all
                      tracked regattas.

Run by the Monday GitHub Action. Output is read by:
  - scrape.py  (to decide whether to run the full scrape today)
  - index.html (to show Live / Upcoming / Historic labels)

Output format (data/schedule.json):
{
  "generated": "2026-04-28",
  "regattas": {
    "arb": {
      "name": "ARB Bosbaanwedstrijden",
      "dates": ["2026-05-30", "2026-05-31"]
    },
    "westelijke": {
      "name": "Westelijke Regatta",
      "dates": ["2026-05-16", "2026-05-17"]
    },
    ...
  }
}
"""

import json
import os
import re
import datetime
import requests
from bs4 import BeautifulSoup

# ─── Config ───────────────────────────────────────────────────────────────────

CALENDAR_URL = "https://time-team.nl/en/info/regattas"
OUTPUT_PATH  = "data/schedule.json"

# Map keywords in regatta names (lowercase) → our internal regatta codes.
# Order matters: more specific patterns first.
REGATTA_NAME_MAP = {
    "arb bosbaanwedstrijden":   "arb",
    "arb":                      "arb",
    "westelijke":               "westelijke",
    "hollandia":                "hollandia",
    "raceroei":                 "raceroei",
    "holland beker":            "hollandbeker",
    "hollandbeker":             "hollandbeker",
    "voorjaarsregatta":         "voorjaarsregatta",
    "bvr":                      "bvr",
    "nsrf":                     "nsrf",
    "diyr":                     "diyr",
}

MONTH_MAP = {
    "january": 1,  "february": 2,  "march": 3,     "april": 4,
    "may": 5,      "june": 6,      "july": 7,       "august": 8,
    "september": 9,"october": 10,  "november": 11,  "december": 12,
}

REQUEST_TIMEOUT = 15

# ─── Helpers ──────────────────────────────────────────────────────────────────

def match_regatta_code(name: str) -> str | None:
    """Return the internal code for a regatta name, or None if not tracked."""
    lower = name.lower()
    for keyword, code in REGATTA_NAME_MAP.items():
        if keyword in lower:
            return code
    return None


def parse_date_range(date_str: str, year: int) -> list[str]:
    """
    Parse strings like:
        "30 and 31 May"      → ["2026-05-30", "2026-05-31"]
        "1 to 3 May"         → ["2026-05-01", "2026-05-02", "2026-05-03"]
        "20 June"            → ["2026-06-20"]
        "29 May"             → ["2026-05-29"]
        "22 to 24 May"       → ["2026-05-22", "2026-05-23", "2026-05-24"]

    Returns a list of ISO date strings. Returns [] if parsing fails.
    """
    date_str = date_str.strip().lower()

    # Extract month
    month = None
    for name, num in MONTH_MAP.items():
        if name in date_str:
            month = num
            break
    if month is None:
        return []

    # Extract all day numbers
    days = [int(d) for d in re.findall(r'\b(\d{1,2})\b', date_str)]
    if not days:
        return []

    # If two days found and "to" or "and" in string, expand the range
    if len(days) == 2 and ("to" in date_str or "and" in date_str):
        start, end = days[0], days[1]
        # Handle month boundary (e.g. "31 May to 1 June") — already split by month
        if start <= end:
            days = list(range(start, end + 1))

    dates = []
    for day in days:
        try:
            dates.append(datetime.date(year, month, day).isoformat())
        except ValueError:
            pass  # invalid date, skip

    return dates


def fetch_calendar() -> str | None:
    """Fetch the time-team.nl regatta calendar page."""
    try:
        resp = requests.get(CALENDAR_URL, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.text
        print(f"[HTTP {resp.status_code}] {CALENDAR_URL}")
        return None
    except requests.RequestException as e:
        print(f"[ERROR] Could not fetch calendar: {e}")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_schedule() -> dict:
    """
    Scrape the calendar and return a schedule dict.
    """
    html = fetch_calendar()
    if html is None:
        return {}

    soup = BeautifulSoup(html, "html.parser")
    today = datetime.date.today()
    year  = today.year

    schedule = {}

    # Each regatta on the page is an <a> or block containing an <h2> (name)
    # and a text node with the date. We look for all h2 tags and their
    # surrounding context.
    for h2 in soup.find_all("h2"):
        name = h2.get_text(strip=True)
        if not name:
            continue

        code = match_regatta_code(name)
        if code is None:
            continue  # not one of our tracked regattas

        # The date string is usually in the same <a> tag or the next sibling text
        parent = h2.parent
        parent_text = parent.get_text(separator=" ", strip=True)

        # Try to extract a date phrase — look for month names
        date_phrase = None
        for month in MONTH_MAP:
            if month in parent_text.lower():
                # Extract the surrounding words (up to 8 words around the month)
                words = parent_text.split()
                for i, w in enumerate(words):
                    if month in w.lower():
                        start = max(0, i - 4)
                        end   = min(len(words), i + 2)
                        date_phrase = " ".join(words[start:end])
                        break
                break

        if date_phrase is None:
            continue

        dates = parse_date_range(date_phrase, year)

        # If all dates are in the past, try next year
        if dates and all(d < today.isoformat() for d in dates):
            dates = parse_date_range(date_phrase, year + 1)

        if not dates:
            print(f"  [WARN] Could not parse dates for '{name}': '{date_phrase}'")
            continue

        # Keep the earliest upcoming occurrence per code
        # (a regatta might appear twice if it spans a year boundary)
        if code not in schedule or dates[0] < schedule[code]["dates"][0]:
            schedule[code] = {
                "name":  name,
                "dates": dates,
            }
        print(f"  {code:20s} {name:40s} {dates}")

    return schedule


def save_schedule(schedule: dict):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output = {
        "generated": datetime.date.today().isoformat(),
        "regattas":  schedule,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved schedule for {len(schedule)} regattas → {OUTPUT_PATH}")


def is_race_day(schedule_path: str = OUTPUT_PATH) -> bool:
    """
    Called by scrape.py to check if today is a race day for any tracked regatta.
    Returns True if today falls within any regatta's dates.
    """
    if not os.path.exists(schedule_path):
        print("[WARN] No schedule.json found — assuming no race today.")
        return False

    with open(schedule_path) as f:
        data = json.load(f)

    today = datetime.date.today().isoformat()
    for code, info in data.get("regattas", {}).items():
        if today in info.get("dates", []):
            print(f"  Race day detected: {info['name']} ({code})")
            return True
    return False


if __name__ == "__main__":
    print(f"Fetching regatta calendar from {CALENDAR_URL} …\n")
    schedule = parse_schedule()
    if schedule:
        save_schedule(schedule)
    else:
        print("No tracked regattas found in calendar.")
