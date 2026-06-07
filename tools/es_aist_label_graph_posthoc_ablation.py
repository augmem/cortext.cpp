#!/usr/bin/env python3
"""Posthoc modality/kind ablations for ES-AIST label graph experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


STOP_TOKENS = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "on",
    "with",
    "for",
    "and",
    "or",
    "by",
    "about",
    "memory",
    "audio",
    "video",
    "text",
    "evidence",
    "scene",
    "containing",
    "involving",
    "someone",
    "something",
}

CORE_KINDS = {
    "action",
    "artifact",
    "entity_animal",
    "entity_event",
    "entity_location",
    "entity_person",
    "entity_plant",
    "event",
    "event_action",
    "name_or_entity",
    "object",
    "perception_action",
    "scene_event",
}

STRICT_KINDS = {
    "entity_animal",
    "entity_location",
    "entity_person",
    "event",
    "event_action",
    "object",
    "perception_action",
    "scene_event",
}


def tokens(text: str) -> set[str]:
    out: set[str] = set()
    current: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        elif current:
            token = "".join(current)
            if len(token) > 1 and token not in STOP_TOKENS:
                out.add(token)
            current.clear()
    if current:
        token = "".join(current)
        if len(token) > 1 and token not in STOP_TOKENS:
            out.add(token)
    return out


def load_rows(path: Path) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str, str], list[dict]]]:
    signals: dict[tuple[str, str], dict] = {}
    by_signal_view: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["scenario"], row["signal_id"])
            signals[key] = {
                "scenario": row["scenario"],
                "signal_id": row["signal_id"],
                "modality": row["modality"],
                "offline_group": row["offline_group"],
            }
            by_signal_view[(row["scenario"], row["signal_id"], row["view"])].append(row)
    for rows in by_signal_view.values():
        rows.sort(key=lambda r: int(r["rank"]))
    return signals, by_signal_view


def filtered_labels(rows: list[dict], top_k: int, kind_mode: str) -> list[dict]:
    if kind_mode == "all":
        allowed = None
    elif kind_mode == "core":
        allowed = CORE_KINDS
    elif kind_mode == "strict":
        allowed = STRICT_KINDS
    else:
        raise ValueError(kind_mode)
    out = []
    for row in rows:
        if allowed is not None and row["kind"] not in allowed:
            continue
        out.append(row)
        if len(out) >= top_k:
            break
    return out


def overlap(left: list[dict], right: list[dict]) -> tuple[bool, str]:
    for a in left:
        at = tokens(a["label"])
        for b in right:
            common = at & tokens(b["label"])
            if common:
                return True, f'{a["kind"]}:{a["label"]}<->{b["kind"]}:{b["label"]}'
    return False, ""


def evaluate(input_csv: Path) -> dict:
    signals, by_signal_view = load_rows(input_csv)
    views = sorted({key[2] for key in by_signal_view})
    modality_modes = {
        "all_modalities": {"audio", "image", "text"},
        "no_audio": {"image", "text"},
        "image_text_only": {"image", "text"},
        "audio_image_only": {"audio", "image"},
        "text_only": {"text"},
        "image_only": {"image"},
        "audio_only": {"audio"},
    }
    kind_modes = ["all", "core", "strict"]
    top_ks = [10, 25, 50]
    rows = []
    for view in views:
        for top_k in top_ks:
            for kind_mode in kind_modes:
                for modality_mode, allowed_modalities in modality_modes.items():
                    possible_same = 0
                    possible_cross = 0
                    same = 0
                    cross = 0
                    edge_examples = []
                    by_scenario: dict[str, list[dict]] = defaultdict(list)
                    for signal in signals.values():
                        if signal["modality"] in allowed_modalities:
                            by_scenario[signal["scenario"]].append(signal)
                    for scenario_signals in by_scenario.values():
                        for i, left_signal in enumerate(scenario_signals):
                            for right_signal in scenario_signals[i + 1 :]:
                                same_group = (
                                    left_signal["offline_group"]
                                    == right_signal["offline_group"]
                                )
                                if same_group:
                                    possible_same += 1
                                else:
                                    possible_cross += 1
                                left = filtered_labels(
                                    by_signal_view[
                                        (
                                            left_signal["scenario"],
                                            left_signal["signal_id"],
                                            view,
                                        )
                                    ],
                                    top_k,
                                    kind_mode,
                                )
                                right = filtered_labels(
                                    by_signal_view[
                                        (
                                            right_signal["scenario"],
                                            right_signal["signal_id"],
                                            view,
                                        )
                                    ],
                                    top_k,
                                    kind_mode,
                                )
                                hit, detail = overlap(left, right)
                                if not hit:
                                    continue
                                if same_group:
                                    same += 1
                                else:
                                    cross += 1
                                if len(edge_examples) < 8:
                                    edge_examples.append(
                                        {
                                            "left": left_signal["signal_id"],
                                            "right": right_signal["signal_id"],
                                            "same_group": same_group,
                                            "detail": detail,
                                        }
                                    )
                    scored = same + cross
                    rows.append(
                        {
                            "view": view,
                            "top_k": top_k,
                            "kind_mode": kind_mode,
                            "modality_mode": modality_mode,
                            "possible_same_group_pairs": possible_same,
                            "possible_cross_group_pairs": possible_cross,
                            "same_group_edges": same,
                            "cross_group_edges": cross,
                            "edge_count": scored,
                            "precision": same / scored if scored else 1.0,
                            "same_group_recall": (
                                same / possible_same if possible_same else 0.0
                            ),
                            "cross_group_rate": (
                                cross / possible_cross if possible_cross else 0.0
                            ),
                            "examples": edge_examples,
                        }
                    )
    best = sorted(
        rows,
        key=lambda r: (
            r["cross_group_rate"],
            -r["same_group_recall"],
            -r["precision"],
            r["top_k"],
        ),
    )[:12]
    return {"rows": rows, "best_low_cross": best}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(Path(args.input_csv))
    with (output_dir / "es_aist_label_graph_posthoc_ablation.json").open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    with (output_dir / "es_aist_label_graph_posthoc_ablation.csv").open("w", newline="") as f:
        fieldnames = [
            "view",
            "top_k",
            "kind_mode",
            "modality_mode",
            "possible_same_group_pairs",
            "possible_cross_group_pairs",
            "same_group_edges",
            "cross_group_edges",
            "edge_count",
            "precision",
            "same_group_recall",
            "cross_group_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({key: row[key] for key in fieldnames})
    print(json.dumps({"best_low_cross": result["best_low_cross"][:8]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
