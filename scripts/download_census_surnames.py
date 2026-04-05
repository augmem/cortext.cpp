#!/usr/bin/env python3
"""
Fetch surname priors from the public US Census surname API and write a simple
CSV suitable for build_name_priors.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path


def fetch(limit: int) -> list[dict[str, str]]:
    url = (
        "https://api.census.gov/data/2010/surname?"
        + urllib.parse.urlencode({"get": "NAME,COUNT,RANK", "RANK": f"1:{limit}"})
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        rows = json.load(response)
    header = rows[0]
    return [dict(zip(header, row, strict=False)) for row in rows[1:]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Census surnames as CSV.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum rank to fetch from the Census API.",
    )
    parser.add_argument(
        "--out",
        default="data/raw/names/census_surnames.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    rows = fetch(args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["surname", "count", "rank"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "surname": row["NAME"].title(),
                    "count": row["COUNT"],
                    "rank": row["RANK"],
                }
            )
    print(f"[OK] wrote {len(rows)} census surnames to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
