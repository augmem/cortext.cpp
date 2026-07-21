#!/usr/bin/env python3
"""Extract a one-based successful consolidation schedule from a replay profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    events = profile.get("consolidation_events")
    if not isinstance(events, list):
        raise ValueError("profile consolidation_events must be an array")
    schedule: list[int] = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("consolidation event must be an object")
        value = event.get("one_based_event_index")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("consolidation event lacks a positive one-based index")
        schedule.append(value)
    if schedule != sorted(set(schedule)):
        raise ValueError("consolidation schedule must be strictly increasing")
    if len(schedule) != profile.get("consolidation_runs"):
        raise ValueError("consolidation run count does not match schedule")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(schedule, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
