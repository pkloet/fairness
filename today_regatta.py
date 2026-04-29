"""
Prints the regatta code(s) racing today, one per line.
Used by the GitHub Action to target only the live regatta.
"""
import json, datetime, sys, os

schedule_path = "data/schedule.json"
if not os.path.exists(schedule_path):
    sys.exit(0)

with open(schedule_path) as f:
    data = json.load(f)

today = datetime.date.today().isoformat()
for code, info in data.get("regattas", {}).items():
    if today in info.get("dates", []):
        print(code)
