#!/usr/bin/env python3
"""
Build lightweight given-name and surname priors for label-classifier training.

Supports:
- built-in smoke-test seed lexicons
- SSA yob*.txt directories
- multilingual Kaggle-style name CSVs
- generic CSV/TXT name lists
- Census-style surname CSVs
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SEED_GIVEN = [
    "Gabriel",
    "Olivia",
    "Emma",
    "Amelia",
    "Charlotte",
    "Sophia",
    "Isabella",
    "Ava",
    "Mia",
    "Luna",
    "Harper",
    "Evelyn",
    "Noah",
    "Liam",
    "Oliver",
    "Elijah",
    "Mateo",
    "Lucas",
    "Levi",
    "Ezra",
    "Asher",
    "James",
    "Henry",
    "Leo",
    "Michael",
    "Benjamin",
    "William",
    "Theodore",
    "Jack",
    "Sarah",
    "Emily",
    "Henry",
    "Alice",
    "Grace",
    "Chloe",
    "Zoe",
    "Madison",
    "Ella",
    "Scarlett",
    "Victoria",
    "Nora",
    "Layla",
    "Hannah",
    "Elizabeth",
    "Daniel",
    "Samuel",
    "Owen",
    "John",
    "Luke",
    "David",
    "Joseph",
    "Julian",
    "Wyatt",
    "Sebastian",
    "Anthony",
    "Isaac",
    "Andrew",
    "Joshua",
    "Christopher",
    "Nathan",
    "Thomas",
    "Avery",
    "Abigail",
    "Riley",
    "Aria",
    "Camila",
    "Penelope",
    "Gianna",
    "Sofia",
    "Mila",
    "Aurora",
    "Hazel",
]

SEED_SURNAMES = [
    "Willen",
    "Smith",
    "Johnson",
    "Brown",
    "Nguyen",
    "Patel",
    "Garcia",
    "Kim",
    "Lopez",
    "Taylor",
    "Davis",
    "Martinez",
    "Anderson",
    "Thomas",
    "Jackson",
    "White",
    "Harris",
    "Martin",
    "Thompson",
    "Moore",
    "Allen",
    "Young",
    "King",
    "Wright",
    "Scott",
    "Torres",
    "Hill",
    "Flores",
    "Green",
    "Adams",
    "Nelson",
    "Baker",
    "Hall",
    "Rivera",
    "Campbell",
    "Mitchell",
    "Carter",
    "Roberts",
    "Gomez",
    "Phillips",
    "Evans",
    "Turner",
    "Diaz",
    "Parker",
    "Cruz",
    "Edwards",
    "Collins",
    "Reyes",
    "Stewart",
    "Morris",
    "Morales",
    "Murphy",
]


def normalize_key(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def add_name(counter: dict[str, float], name: str, weight: float) -> None:
    key = normalize_key(name)
    if key:
        counter[key] += float(weight)


def ingest_generic_file(path: Path, target: dict[str, float], name_field: str | None = None, weight_field: str | None = None) -> None:
    if path.suffix.lower() in {".txt", ".lst"}:
        for line in path.read_text(encoding="utf-8").splitlines():
            add_name(target, line, 1.0)
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            detected_name = ""
            if name_field and row.get(name_field):
                detected_name = row[name_field]
            else:
                for candidate in ("name", "given_name", "surname", "last_name", "first_name"):
                    if row.get(candidate):
                        detected_name = row[candidate]
                        break
            if not detected_name:
                continue
            weight = 1.0
            if weight_field and row.get(weight_field):
                try:
                    weight = float(row[weight_field])
                except ValueError:
                    weight = 1.0
            elif not weight_field:
                for candidate_weight in ("count", "COUNT", "frequency", "freq", "weight", "Weight"):
                    if row.get(candidate_weight):
                        try:
                            weight = float(row[candidate_weight])
                            break
                        except ValueError:
                            continue
            add_name(target, detected_name, weight)


def ingest_ssa_dir(path: Path, target: dict[str, float]) -> None:
    for child in sorted(path.glob("yob*.txt")):
        with child.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(",")
                if len(parts) != 3:
                    continue
                name, _, count = parts
                try:
                    weight = float(count)
                except ValueError:
                    weight = 1.0
                add_name(target, name, weight)


def normalize_counter(counter: dict[str, float], top_k: int | None) -> dict[str, float]:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None and top_k > 0:
        items = items[:top_k]
    if not items:
        return {}
    max_value = max(value for _, value in items) or 1.0
    return {key: round(value / max_value, 6) for key, value in items}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build name priors for label-classifier training.")
    parser.add_argument(
        "--ssa-dir",
        default=None,
        help="Optional SSA yob*.txt directory for given-name priors.",
    )
    parser.add_argument(
        "--given-source",
        action="append",
        default=[],
        help="Optional generic CSV/TXT source for given names.",
    )
    parser.add_argument(
        "--surname-source",
        action="append",
        default=[],
        help="Optional generic CSV/TXT source for surnames.",
    )
    parser.add_argument(
        "--kaggle-forenames",
        action="append",
        default=[],
        help="Kaggle-style multilingual forenames.csv source (forename,gender,country,count).",
    )
    parser.add_argument(
        "--kaggle-surnames",
        action="append",
        default=[],
        help="Kaggle-style multilingual surnames.csv source (surname,gender,country,count).",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Disable the built-in seed lexicons.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50000,
        help="Optional top-k cutoff per lexicon after weighting. Use 0 to keep all names.",
    )
    parser.add_argument(
        "--out",
        default="data/label_classifier/name_priors.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    given: dict[str, float] = defaultdict(float)
    surname: dict[str, float] = defaultdict(float)

    if not args.no_seed:
        for entry in SEED_GIVEN:
            add_name(given, entry, 1.0)
        for entry in SEED_SURNAMES:
            add_name(surname, entry, 1.0)

    if args.ssa_dir:
        ingest_ssa_dir(Path(args.ssa_dir), given)

    for source in args.kaggle_forenames:
        ingest_generic_file(Path(source), given, name_field="forename", weight_field="count")
    for source in args.kaggle_surnames:
        ingest_generic_file(Path(source), surname, name_field="surname", weight_field="count")
    for source in args.given_source:
        ingest_generic_file(Path(source), given)
    for source in args.surname_source:
        ingest_generic_file(Path(source), surname)

    payload = {
        "given": normalize_counter(given, args.top_k),
        "surname": normalize_counter(surname, args.top_k),
        "metadata": {
            "no_seed": args.no_seed,
            "ssa_dir": args.ssa_dir,
            "kaggle_forenames": args.kaggle_forenames,
            "kaggle_surnames": args.kaggle_surnames,
            "given_sources": args.given_source,
            "surname_sources": args.surname_source,
            "top_k": args.top_k,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        f"[OK] wrote {len(payload['given'])} given names and {len(payload['surname'])} surnames to {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
