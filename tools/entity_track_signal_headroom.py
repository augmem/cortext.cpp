#!/usr/bin/env python3
"""Offline entity-track signal headroom sweep for anchor replay artifacts.

This script is benchmark-only. It starts from the ES-AIST contextual anchor
candidate feature export and injects a controlled synthetic entity-track signal
into the existing candidate pool. Labels are used only to simulate signal
quality and evaluate headroom; they are not runtime features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Candidate:
    case_index: int
    candidate_index: int
    label: int
    features: dict[str, float]


@dataclass
class Case:
    case_index: int
    case_id: str
    label_class: str
    is_reference: bool
    candidates: list[Candidate]


def stable_unit(*parts: object) -> float:
    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value / float(2**64 - 1)


def stage1_attention_score(features: dict[str, float]) -> float:
    return (
        0.14 * features["semantic_cos"]
        + 0.27 * features["entity_cos"]
        + 0.17 * features["full_cos"]
        + 0.09 * features["wm_entity_support"]
        + 0.12 * features["stm_entity_support"]
        + 0.06 * features["wm_full_support"]
        + 0.09 * features["stm_full_support"]
        + 0.07 * features["ltm_entity_attention"]
        + 0.04 * features["ltm_full_attention"]
        + 0.05 * features["context_margin_proxy"]
        + 0.05 * features["recency_score"]
        + 0.03 * features["same_source_flag"]
        + 0.02 * features["wm_entropy_inverse"]
        + 0.02 * features["stm_entropy_inverse"]
        + 0.02 * features["wm_current_fit"]
    )


def load_cases(artifact_dir: Path) -> list[Case]:
    case_meta: dict[int, tuple[str, str, bool]] = {}
    cases_csv = artifact_dir / "es_aist_contextual_anchor_cases.csv"
    with cases_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["policy"] != "semantic_key_signal":
                continue
            case_index = len(case_meta)
            case_meta[case_index] = (
                row["case_id"],
                row["label_class"],
                row["is_reference"] == "1",
            )

    grouped: dict[int, list[Candidate]] = {index: [] for index in case_meta}
    feature_csv = artifact_dir / "es_aist_contextual_anchor_candidate_features.csv"
    with feature_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_index = int(row["case_index"])
            candidate_index = int(row["candidate_index"])
            label = int(row["label"])
            features = {
                key: float(value)
                for key, value in row.items()
                if key not in {"case_index", "candidate_index", "label"}
            }
            grouped[case_index].append(
                Candidate(case_index, candidate_index, label, features)
            )

    cases: list[Case] = []
    for case_index in sorted(case_meta):
        case_id, label_class, is_reference = case_meta[case_index]
        candidates = sorted(grouped.get(case_index, []), key=lambda c: c.candidate_index)
        cases.append(Case(case_index, case_id, label_class, is_reference, candidates))
    return cases


def top_rank(scores: list[float], index: int) -> int:
    target = scores[index]
    better = sum(1 for score in scores if score > target)
    return better + 1


def threshold_at_fpr(control_scores: list[float], fpr: float) -> float:
    if not control_scores:
        return math.inf
    ordered = sorted(control_scores, reverse=True)
    if fpr <= 0.0:
        return ordered[0] + 1.0e-9
    allowed = max(1, math.floor(fpr * len(ordered)))
    return ordered[min(allowed - 1, len(ordered) - 1)]


def summarize(cases: list[Case], scores_by_case: list[list[float]]) -> dict[str, float | int]:
    refs = 0
    controls = 0
    top1 = 0
    top3 = 0
    top5 = 0
    reference_case_scores: list[float] = []
    control_case_scores: list[float] = []
    for case, scores in zip(cases, scores_by_case):
        if case.is_reference:
            refs += 1
            target_index = next(
                (i for i, candidate in enumerate(case.candidates) if candidate.label == 1),
                -1,
            )
            if target_index < 0:
                reference_case_scores.append(0.0)
                continue
            rank = top_rank(scores, target_index)
            top1 += rank <= 1
            top3 += rank <= 3
            top5 += rank <= 5
            reference_case_scores.append(scores[target_index])
        else:
            controls += 1
            control_case_scores.append(max(scores) if scores else 0.0)

    zero_threshold = threshold_at_fpr(control_case_scores, 0.0)
    five_threshold = threshold_at_fpr(control_case_scores, 0.05)
    zero_recovery = sum(score >= zero_threshold for score in reference_case_scores)
    five_recovery = sum(score >= five_threshold for score in reference_case_scores)
    control_abstain_zero = sum(score < zero_threshold for score in control_case_scores)
    control_abstain_five = sum(score < five_threshold for score in control_case_scores)
    return {
        "reference_count": refs,
        "control_count": controls,
        "target_top1": top1 / refs if refs else 0.0,
        "target_top3": top3 / refs if refs else 0.0,
        "target_top5": top5 / refs if refs else 0.0,
        "zero_fpr_recovery": zero_recovery,
        "five_pct_fpr_recovery": five_recovery,
        "zero_fpr_recovery_rate": zero_recovery / refs if refs else 0.0,
        "five_pct_fpr_recovery_rate": five_recovery / refs if refs else 0.0,
        "zero_fpr_control_abstain_rate": control_abstain_zero / controls
        if controls
        else 0.0,
        "five_pct_control_abstain_rate": control_abstain_five / controls
        if controls
        else 0.0,
        "zero_fpr_threshold": zero_threshold,
        "five_pct_threshold": five_threshold,
    }


def score_grid(
    cases: list[Case],
    strengths: Iterable[float],
    recalls: Iterable[float],
    false_rates: Iterable[float],
    seed: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    base_scores_by_case = [
        [stage1_attention_score(candidate.features) for candidate in case.candidates]
        for case in cases
    ]
    baseline = summarize(cases, base_scores_by_case)
    rows.append(
        {
            "signal_strength": 0.0,
            "target_track_recall": 0.0,
            "false_track_rate": 0.0,
            "mode": "baseline_wm_stm_ltm_attention",
            **baseline,
        }
    )

    for strength in strengths:
        for recall in recalls:
            for false_rate in false_rates:
                scored_cases: list[list[float]] = []
                for case, base_scores in zip(cases, base_scores_by_case):
                    scores = list(base_scores)
                    for idx, candidate in enumerate(case.candidates):
                        if candidate.label == 1:
                            if (
                                case.is_reference
                                and stable_unit(seed, "target", case.case_id, idx)
                                < recall
                            ):
                                scores[idx] += strength
                        else:
                            if (
                                stable_unit(seed, "false", case.case_id, idx)
                                < false_rate
                            ):
                                scores[idx] += strength
                    scored_cases.append(scores)
                rows.append(
                    {
                        "signal_strength": strength,
                        "target_track_recall": recall,
                        "false_track_rate": false_rate,
                        "mode": "noisy_entity_track_signal",
                        **summarize(cases, scored_cases),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        default="build/es_aist_contextual_anchor_shadow_sequence_mlp",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir) if args.output_dir else artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(artifact_dir)
    strengths = [0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00]
    recalls = [0.50, 0.70, 0.85, 0.95, 1.00]
    false_rates = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
    rows = score_grid(cases, strengths, recalls, false_rates, args.seed)

    csv_path = output_dir / "entity_track_signal_quality_sweep.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    viable = [
        row
        for row in rows
        if row["mode"] != "baseline_wm_stm_ltm_attention"
        and row["zero_fpr_recovery"] > 0
        and row["false_track_rate"] <= 0.02
    ]
    noisy_rows = [
        row for row in rows if row["mode"] == "noisy_entity_track_signal"
    ]

    def breakpoints(metric: str, targets: Iterable[float]) -> dict[str, object]:
        out: dict[str, object] = {}
        for target in targets:
            hits = [
                row
                for row in noisy_rows
                if float(row[f"{metric}_recovery_rate"]) >= target
            ]
            out[str(target)] = (
                min(
                    hits,
                    key=lambda row: (
                        float(row["false_track_rate"]),
                        float(row["signal_strength"]),
                        -float(row["target_track_recall"]),
                    ),
                )
                if hits
                else None
            )
        return out

    def best_by_recall_at_zero_false() -> list[dict[str, object]]:
        out = []
        recalls_seen = sorted(
            {float(row["target_track_recall"]) for row in noisy_rows}
        )
        for recall in recalls_seen:
            subset = [
                row
                for row in noisy_rows
                if float(row["target_track_recall"]) == recall
                and float(row["false_track_rate"]) == 0.0
            ]
            if not subset:
                continue
            out.append(
                max(
                    subset,
                    key=lambda row: (
                        int(row["zero_fpr_recovery"]),
                        int(row["five_pct_fpr_recovery"]),
                        float(row["target_top3"]),
                    ),
                )
            )
        return out

    def best_by_false_rate() -> list[dict[str, object]]:
        out = []
        false_rates_seen = sorted({float(row["false_track_rate"]) for row in noisy_rows})
        for false_rate in false_rates_seen:
            subset = [
                row
                for row in noisy_rows
                if float(row["false_track_rate"]) == false_rate
            ]
            out.append(
                max(
                    subset,
                    key=lambda row: (
                        int(row["zero_fpr_recovery"]),
                        int(row["five_pct_fpr_recovery"]),
                        float(row["target_top3"]),
                    ),
                )
            )
        return out

    best = max(
        rows,
        key=lambda row: (
            row["zero_fpr_recovery"],
            row["five_pct_fpr_recovery"],
            row["target_top3"],
            -row["false_track_rate"],
        ),
    )
    summary = {
        "mode": "entity_track_signal_quality_sweep",
        "source_artifact": str(artifact_dir),
        "runtime_effect": "none",
        "production_retrieval_changed": False,
        "labels_used_for_runtime_features": False,
        "case_count": len(cases),
        "reference_count": sum(case.is_reference for case in cases),
        "control_count": sum(not case.is_reference for case in cases),
        "grid_rows": len(rows),
        "baseline": rows[0],
        "best": best,
        "zero_fpr_recovery_breakpoints": breakpoints(
            "zero_fpr", [0.25, 0.50, 0.80, 0.95]
        ),
        "five_pct_recovery_breakpoints": breakpoints(
            "five_pct_fpr", [0.25, 0.50, 0.80, 0.95]
        ),
        "best_by_target_recall_when_false_track_zero": best_by_recall_at_zero_false(),
        "best_by_false_track_rate": best_by_false_rate(),
        "minimal_viable_low_false_track": min(
            viable,
            key=lambda row: (
                row["signal_strength"],
                -row["target_track_recall"],
                row["false_track_rate"],
            ),
        )
        if viable
        else None,
        "interpretation": (
            "Offline headroom sweep. A usable upstream entity/track signal must "
            "boost true targets strongly enough while keeping false track "
            "matches very rare; labels are used only to simulate signal quality."
        ),
    }
    json_path = output_dir / "entity_track_signal_quality_sweep.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
