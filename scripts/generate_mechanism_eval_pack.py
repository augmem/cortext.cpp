#!/usr/bin/env python3
"""
Generate a deterministic study-only synthetic pack for mechanism ablations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def scenario(conversation_id: str, family: str, turns: list[tuple[str, str]], answer_key: dict) -> tuple[list, dict]:
    content = [{"agent": agent, "message": message} for agent, message in turns]
    record = [conversation_id, {"content": content, "scenario_family": family}]
    key = {"conversation_id": conversation_id, "scenario_family": family}
    key.update(answer_key)
    return record, key


def build_pack() -> tuple[list[list], list[dict]]:
    records: list[list] = []
    keys: list[dict] = []

    cases = [
        scenario(
            "mech_delayed_correction_001",
            "delayed_correction",
            [
                ("agent_1", "My pharmacy changed my blood pressure pill from lisinopril to losartan last week."),
                ("agent_2", "I will keep that change in mind."),
                ("agent_1", "Ignore the old pill if it appears in older notes. What am I taking now?"),
            ],
            {
                "query_id": "current_medication",
                "expected_current": "losartan",
                "expected_historical": "lisinopril",
                "expected_belief": "losartan",
                "expected_abstain": False,
                "severity": "high",
                "provenance_class": "direct_user_update",
                "temporal_mode": "current",
            },
        ),
        scenario(
            "mech_mixed_provenance_001",
            "mixed_provenance",
            [
                ("agent_1", "A coworker guessed my new address might still be Austin."),
                ("agent_2", "That sounds uncertain."),
                ("agent_1", "For the record, I moved to Chicago in March. Where do I live now?"),
            ],
            {
                "query_id": "current_address",
                "expected_current": "Chicago",
                "expected_historical": "Austin",
                "expected_belief": "Chicago",
                "expected_abstain": False,
                "severity": "high",
                "provenance_class": "mixed",
                "temporal_mode": "current",
            },
        ),
        scenario(
            "mech_conflicting_low_conf_001",
            "conflicting_low_confidence",
            [
                ("agent_1", "Someone in chat said my caregiver is probably Emily, but they were guessing."),
                ("agent_2", "I will treat that as uncertain."),
                ("agent_1", "My caregiver is Sarah. Who is my caregiver now?"),
            ],
            {
                "query_id": "caregiver_now",
                "expected_current": "Sarah",
                "expected_historical": "Emily",
                "expected_belief": "Sarah",
                "expected_abstain": False,
                "severity": "critical",
                "provenance_class": "low_conflict",
                "temporal_mode": "current",
            },
        ),
        scenario(
            "mech_routine_exception_001",
            "routine_exception",
            [
                ("agent_1", "I usually take my walk at 7 a.m. every weekday."),
                ("agent_2", "That sounds like a stable routine."),
                ("agent_1", "Tomorrow I have a dentist appointment, so the walk moves to 9 a.m. What time is tomorrow's walk?"),
            ],
            {
                "query_id": "tomorrow_walk",
                "expected_current": "9 a.m.",
                "expected_historical": "7 a.m.",
                "expected_belief": "9 a.m.",
                "expected_abstain": False,
                "severity": "medium",
                "provenance_class": "direct_user_update",
                "temporal_mode": "current",
            },
        ),
        scenario(
            "mech_current_vs_historical_001",
            "current_vs_historical",
            [
                ("agent_1", "I lived in Austin for ten years."),
                ("agent_2", "Noted."),
                ("agent_1", "I moved to Chicago last month."),
                ("agent_2", "I will treat Chicago as current."),
                ("agent_1", "Where did I live before Chicago?"),
            ],
            {
                "query_id": "previous_home",
                "expected_current": "Chicago",
                "expected_historical": "Austin",
                "expected_belief": "Chicago",
                "expected_abstain": False,
                "severity": "medium",
                "provenance_class": "direct_user_update",
                "temporal_mode": "historical",
            },
        ),
        scenario(
            "mech_resumable_routine_001",
            "resumable_routine",
            [
                ("agent_1", "My morning routine is coffee, insulin, then breakfast."),
                ("agent_2", "Coffee, insulin, breakfast."),
                ("agent_1", "Yesterday I stopped after coffee because of a phone call."),
                ("agent_2", "So insulin and breakfast were still pending."),
                ("agent_1", "If I resume the routine after coffee, what comes next?"),
            ],
            {
                "query_id": "routine_resume",
                "expected_current": "insulin",
                "expected_historical": "coffee",
                "expected_belief": "insulin",
                "expected_abstain": False,
                "severity": "high",
                "provenance_class": "direct_user_update",
                "temporal_mode": "current",
            },
        ),
    ]
    for record, key in cases:
        records.append(record)
        keys.append(key)
    return records, keys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic mechanism evaluation pack for Cortext ablations."
    )
    parser.add_argument(
        "--out-dir",
        default="data/mechanism_eval",
        help="Output directory for the generated pack.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, keys = build_pack()
    data_path = out_dir / "valid.jsonl"
    answer_key_path = out_dir / "valid.answer_key.jsonl"
    with data_path.open("w", encoding="utf-8") as data_file:
        for record in records:
            data_file.write(json.dumps(record, ensure_ascii=True) + "\n")
    with answer_key_path.open("w", encoding="utf-8") as key_file:
        for key in keys:
            key_file.write(json.dumps(key, ensure_ascii=True) + "\n")
    print(f"[OK] wrote {len(records)} conversations to {data_path}")
    print(f"[OK] wrote answer key to {answer_key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
