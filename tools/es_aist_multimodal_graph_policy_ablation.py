#!/usr/bin/env python3
"""Evaluate multimodal graph policies over ES-AIST label evidence.

This is benchmark-only. It treats ES-AIST labels as signal evidence and tests
whether audio can be included through temporal co-occurrence without allowing
audio lexical-label noise to create durable graph edges.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
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
    "1",
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

SIGNAL_ORDERS = {
    "wikimedia_dog_multimodal": ["dog_image", "dog_text", "dog_audio"],
    "wikimedia_car_crash_multimodal": ["car_image", "car_text", "crash_audio"],
    "wikimedia_dog_then_car": [
        "dog_image",
        "dog_text",
        "dog_audio",
        "car_image",
        "car_text",
        "crash_audio",
    ],
    "wikimedia_audio_image_interleave": [
        "dog_audio",
        "dog_image",
        "crash_audio",
        "car_image",
    ],
    "wikimedia_cat_multimodal": ["cat_image", "cat_text", "cat_audio"],
    "wikimedia_train_multimodal": ["train_image", "train_text", "train_audio"],
    "wikimedia_bell_multimodal": ["bell_image", "bell_text", "bell_audio"],
    "wikimedia_cat_train_bell_sequence": [
        "cat_image",
        "cat_text",
        "cat_audio",
        "train_image",
        "train_text",
        "train_audio",
        "bell_image",
        "bell_text",
        "bell_audio",
    ],
    "wikimedia_extended_audio_image_interleave": [
        "cat_audio",
        "cat_image",
        "train_audio",
        "train_image",
        "bell_audio",
        "bell_image",
    ],
}


@dataclass
class Signal:
    scenario: str
    signal_id: str
    modality: str
    offline_group: str
    step: int
    labels: list[dict] = field(default_factory=list)


class DSU:
    def __init__(self, nodes: list[str]):
        self.parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


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


def load_signals(path: Path, view: str, top_k: int, core_only: bool) -> dict[str, list[Signal]]:
    rows_by_signal: dict[tuple[str, str], list[dict]] = defaultdict(list)
    signal_meta: dict[tuple[str, str], dict] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["view"] != view:
                continue
            if int(row["rank"]) > top_k:
                continue
            if core_only and row["kind"] not in CORE_KINDS:
                continue
            key = (row["scenario"], row["signal_id"])
            signal_meta[key] = row
            rows_by_signal[key].append(row)

    by_scenario: dict[str, list[Signal]] = defaultdict(list)
    for (scenario, signal_id), labels in rows_by_signal.items():
        if scenario not in SIGNAL_ORDERS:
            continue
        meta = signal_meta[(scenario, signal_id)]
        try:
            step = SIGNAL_ORDERS[scenario].index(signal_id)
        except ValueError:
            continue
        by_scenario[scenario].append(
            Signal(
                scenario=scenario,
                signal_id=signal_id,
                modality=meta["modality"],
                offline_group=meta["offline_group"],
                step=step,
                labels=sorted(labels, key=lambda r: int(r["rank"])),
            )
        )
    for signals in by_scenario.values():
        signals.sort(key=lambda s: s.step)
    return by_scenario


def label_overlap(a: Signal, b: Signal) -> bool:
    for left in a.labels:
        lt = tokens(left["label"])
        for right in b.labels:
            if lt & tokens(right["label"]):
                return True
    return False


def components(nodes: list[Signal], links: list[tuple[str, str]]) -> list[list[Signal]]:
    by_id = {node.signal_id: node for node in nodes}
    dsu = DSU(list(by_id))
    for a, b in links:
        dsu.union(a, b)
    groups: dict[str, list[Signal]] = defaultdict(list)
    for node in nodes:
        groups[dsu.find(node.signal_id)].append(node)
    return list(groups.values())


def unit_stats(units: list[list[Signal]]) -> dict:
    durable_units = [unit for unit in units if len(unit) > 1]
    mixed = [
        unit
        for unit in durable_units
        if len({signal.offline_group for signal in unit}) > 1
    ]
    multimodal = [
        unit
        for unit in durable_units
        if len({signal.modality for signal in unit}) > 1
    ]
    audio_in_units = [
        signal
        for unit in durable_units
        for signal in unit
        if signal.modality == "audio"
    ]
    all_audio = [signal for unit in units for signal in unit if signal.modality == "audio"]
    return {
        "unit_count": len(units),
        "durable_unit_count": len(durable_units),
        "mixed_durable_units": len(mixed),
        "durable_precision": (
            (len(durable_units) - len(mixed)) / len(durable_units)
            if durable_units
            else 1.0
        ),
        "multimodal_durable_units": len(multimodal),
        "audio_durable_coverage": (
            len(audio_in_units) / len(all_audio) if all_audio else 0.0
        ),
        "unit_examples": [
            {
                "signals": [signal.signal_id for signal in unit],
                "modalities": sorted({signal.modality for signal in unit}),
                "groups": sorted({signal.offline_group for signal in unit}),
            }
            for unit in durable_units[:8]
        ],
    }


def build_text_image_units(signals: list[Signal]) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    eligible = [s for s in signals if s.modality in {"text", "image"}]
    for i, left in enumerate(eligible):
        for right in eligible[i + 1 :]:
            if label_overlap(left, right):
                links.append((left.signal_id, right.signal_id))
    return links


def audio_label_links(signals: list[Signal]) -> list[tuple[str, str]]:
    links = build_text_image_units(signals)
    for i, left in enumerate(signals):
        for right in signals[i + 1 :]:
            if left.modality != "audio" and right.modality != "audio":
                continue
            if label_overlap(left, right):
                links.append((left.signal_id, right.signal_id))
    return links


def unit_roots_for_text_image(
    signals: list[Signal], include_visual_singletons: bool = False
) -> tuple[dict[str, list[Signal]], list[tuple[str, str]]]:
    base_links = build_text_image_units(signals)
    non_audio = [s for s in signals if s.modality != "audio"]
    base_units = components(non_audio, base_links)
    root_by_signal: dict[str, list[Signal]] = {}
    for unit in base_units:
        if len(unit) <= 1 and not (
            include_visual_singletons and unit[0].modality == "image"
        ):
            continue
        for signal in unit:
            root_by_signal[signal.signal_id] = unit
    return root_by_signal, base_links


def cooccurrence_links(
    signals: list[Signal],
    radius: int,
    mode: str,
    include_visual_singletons: bool = False,
) -> tuple[list[tuple[str, str]], list[dict]]:
    unit_by_signal, links = unit_roots_for_text_image(
        signals, include_visual_singletons
    )
    ambiguous: list[dict] = []
    for audio in [signal for signal in signals if signal.modality == "audio"]:
        nearby = [
            signal
            for signal in signals
            if signal.modality != "audio"
            and abs(signal.step - audio.step) <= radius
            and signal.signal_id in unit_by_signal
        ]
        if mode == "previous":
            nearby = [signal for signal in nearby if signal.step < audio.step]
        elif mode == "next":
            nearby = [signal for signal in nearby if signal.step > audio.step]
        elif mode != "both":
            raise ValueError(mode)

        candidate_units: dict[tuple[str, ...], list[Signal]] = {}
        for signal in nearby:
            unit = unit_by_signal[signal.signal_id]
            key = tuple(sorted(s.signal_id for s in unit))
            candidate_units[key] = unit

        if len(candidate_units) == 1:
            representative = next(iter(candidate_units.values()))[0]
            links.append((audio.signal_id, representative.signal_id))
        elif len(candidate_units) > 1:
            ambiguous.append(
                {
                    "audio_signal_id": audio.signal_id,
                    "candidate_units": [list(key) for key in candidate_units],
                    "offline_group": audio.offline_group,
                }
            )
    return links, ambiguous


def temporal_audio_label_links(signals: list[Signal], radius: int) -> tuple[list[tuple[str, str]], list[dict]]:
    links = build_text_image_units(signals)
    for audio in [signal for signal in signals if signal.modality == "audio"]:
        for signal in signals:
            if signal.modality == "audio":
                continue
            if abs(signal.step - audio.step) > radius:
                continue
            if label_overlap(audio, signal):
                links.append((audio.signal_id, signal.signal_id))
    return links, []


def ordered_audio_cooccurrence_links(signals: list[Signal]) -> tuple[list[tuple[str, str]], list[dict]]:
    links = build_text_image_units(signals)
    unit_by_signal, _ = unit_roots_for_text_image(signals, True)
    by_step = {signal.step: signal for signal in signals}

    def unit_has_text(signal: Signal | None) -> bool:
        if signal is None or signal.signal_id not in unit_by_signal:
            return False
        return any(s.modality == "text" for s in unit_by_signal[signal.signal_id])

    def representative(signal: Signal) -> str:
        if signal.signal_id in unit_by_signal:
            return unit_by_signal[signal.signal_id][0].signal_id
        return signal.signal_id

    for audio in [signal for signal in signals if signal.modality == "audio"]:
        prev_signal = by_step.get(audio.step - 1)
        next_signal = by_step.get(audio.step + 1)
        target: Signal | None = None
        if prev_signal is not None and unit_has_text(prev_signal):
            target = prev_signal
        elif next_signal is not None and next_signal.modality in {"image", "text"}:
            target = next_signal
        elif prev_signal is not None and prev_signal.modality in {"image", "text"}:
            target = prev_signal

        if target is not None:
            links.append((audio.signal_id, representative(target)))
    return links, []


def evaluate(path: Path, view: str, top_k: int, core_only: bool) -> dict:
    by_scenario = load_signals(path, view, top_k, core_only)
    variants = [
        ("direct_label_all_modalities", lambda s: (audio_label_links(s), [])),
        ("text_image_only", lambda s: (build_text_image_units(s), [])),
        ("audio_cooccur_both_r1", lambda s: cooccurrence_links(s, 1, "both")),
        ("audio_cooccur_both_r2", lambda s: cooccurrence_links(s, 2, "both")),
        ("audio_cooccur_previous_r1", lambda s: cooccurrence_links(s, 1, "previous")),
        ("audio_cooccur_next_r1", lambda s: cooccurrence_links(s, 1, "next")),
        (
            "audio_cooccur_both_r1_visual_singletons",
            lambda s: cooccurrence_links(s, 1, "both", True),
        ),
        (
            "audio_cooccur_previous_r1_visual_singletons",
            lambda s: cooccurrence_links(s, 1, "previous", True),
        ),
        (
            "audio_cooccur_next_r1_visual_singletons",
            lambda s: cooccurrence_links(s, 1, "next", True),
        ),
        ("audio_label_temporal_r1", lambda s: temporal_audio_label_links(s, 1)),
        ("audio_label_temporal_r2", lambda s: temporal_audio_label_links(s, 2)),
        ("audio_ordered_cooccurrence_r1", ordered_audio_cooccurrence_links),
    ]
    rows = []
    cases = []
    for variant, builder in variants:
        totals = defaultdict(float)
        scenario_count = 0
        ambiguous_count = 0
        for scenario, signals in by_scenario.items():
            links, ambiguous = builder(signals)
            units = components(signals, links)
            stats = unit_stats(units)
            scenario_count += 1
            ambiguous_count += len(ambiguous)
            case = {
                "scenario": scenario,
                "variant": variant,
                "links": links,
                "ambiguous_audio": ambiguous,
                **stats,
            }
            cases.append(case)
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    totals[key] += float(value)
        rows.append(
            {
                "variant": variant,
                "scenario_count": scenario_count,
                "ambiguous_audio_count": ambiguous_count,
                "mean_durable_precision": (
                    totals["durable_precision"] / scenario_count
                    if scenario_count
                    else 0.0
                ),
                "mixed_durable_units": int(totals["mixed_durable_units"]),
                "durable_unit_count": int(totals["durable_unit_count"]),
                "multimodal_durable_units": int(totals["multimodal_durable_units"]),
                "mean_audio_durable_coverage": (
                    totals["audio_durable_coverage"] / scenario_count
                    if scenario_count
                    else 0.0
                ),
            }
        )
    return {
        "config": {"view": view, "top_k": top_k, "core_only": core_only},
        "summary": rows,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view", default="entity768")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--all-kinds", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(
        Path(args.input_csv),
        view=args.view,
        top_k=args.top_k,
        core_only=not args.all_kinds,
    )
    with (output_dir / "es_aist_multimodal_graph_policy_ablation.json").open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    with (output_dir / "es_aist_multimodal_graph_policy_ablation.csv").open(
        "w", newline=""
    ) as f:
        fieldnames = [
            "variant",
            "scenario_count",
            "durable_unit_count",
            "mixed_durable_units",
            "mean_durable_precision",
            "multimodal_durable_units",
            "mean_audio_durable_coverage",
            "ambiguous_audio_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["summary"]:
            writer.writerow(row)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
