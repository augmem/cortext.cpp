#!/usr/bin/env python3
"""Audit whether existing ES-AIST candidate scores can satisfy signal gates.

This is benchmark-only. It treats the existing ES-AIST + WM/STM/LTM candidate
feature export as a possible proposal source and asks whether any score view can
produce high target recall while keeping false proposals under the
precision-first anchor-signal contract gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Candidate:
    case_index: int
    candidate_index: int
    label: int
    features: dict[str, float]


@dataclass(frozen=True)
class Case:
    case_index: int
    case_id: str
    label_class: str
    is_reference: bool
    candidates: tuple[Candidate, ...]


ScoreFn = Callable[[dict[str, float]], float]


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
        for row in csv.DictReader(handle):
            case_index = int(row["case_index"])
            candidate_index = int(row["candidate_index"])
            label = int(row["label"])
            features = {
                key: float(value)
                for key, value in row.items()
                if key not in {"case_index", "candidate_index", "label"}
            }
            grouped.setdefault(case_index, []).append(
                Candidate(case_index, candidate_index, label, features)
            )

    cases: list[Case] = []
    for case_index in sorted(case_meta):
        case_id, label_class, is_reference = case_meta[case_index]
        candidates = tuple(
            sorted(grouped.get(case_index, []), key=lambda c: c.candidate_index)
        )
        cases.append(Case(case_index, case_id, label_class, is_reference, candidates))
    return cases


def score_views() -> dict[str, ScoreFn]:
    return {
        "semantic_cos": lambda f: f["semantic_cos"],
        "entity_cos": lambda f: f["entity_cos"],
        "full_cos": lambda f: f["full_cos"],
        "entity_margin": lambda f: f["entity_margin"],
        "full_margin": lambda f: f["full_margin"],
        "max_entity_full": lambda f: max(f["entity_cos"], f["full_cos"]),
        "entity_wm_stm_mean": lambda f: (
            f["entity_cos"] + f["wm_entity_support"] + f["stm_entity_support"]
        )
        / 3.0,
        "full_wm_stm_mean": lambda f: (
            f["full_cos"] + f["wm_full_support"] + f["stm_full_support"]
        )
        / 3.0,
        "context_only_mean": lambda f: (
            f["wm_entity_support"]
            + f["stm_entity_support"]
            + f["wm_full_support"]
            + f["stm_full_support"]
            + f["wm_current_fit"]
            + f["stm_current_fit"]
        )
        / 6.0,
        "stage1_attention": stage1_attention_score,
    }


def threshold_for_fpr(negative_scores: list[float], fpr: float) -> float:
    if not negative_scores:
        return math.inf
    ordered = sorted(negative_scores, reverse=True)
    if fpr <= 0.0:
        return ordered[0] + 1.0e-9
    allowed = math.floor(len(ordered) * fpr)
    if allowed <= 0:
        return ordered[0] + 1.0e-9
    return ordered[min(allowed - 1, len(ordered) - 1)]


def target_rank(scores: list[float], target_index: int) -> int:
    return 1 + sum(score > scores[target_index] for score in scores)


def audit_view(
    cases: list[Case],
    view_name: str,
    score_fn: ScoreFn,
    fpr_targets: list[float],
) -> tuple[list[dict[str, float | int | str | bool]], list[dict[str, str | int | float]]]:
    scored: list[tuple[Case, list[float]]] = []
    negatives: list[float] = []
    for case in cases:
        scores = [score_fn(candidate.features) for candidate in case.candidates]
        scored.append((case, scores))
        for candidate, score in zip(case.candidates, scores):
            if candidate.label != 1:
                negatives.append(score)

    rows: list[dict[str, float | int | str | bool]] = []
    failures: list[dict[str, str | int | float]] = []
    for fpr in fpr_targets:
        threshold = threshold_for_fpr(negatives, fpr)
        refs = controls = target_present = 0
        target_selected = target_top1 = target_top3 = target_top5 = 0
        reference_false_cases = control_false_cases = 0
        false_candidate_selected = 0
        negative_count = 0
        selected_candidate_count = 0
        for case, scores in scored:
            selected = [score >= threshold for score in scores]
            selected_candidate_count += sum(selected)
            false_selected = False
            for candidate, is_selected in zip(case.candidates, selected):
                if candidate.label != 1:
                    negative_count += 1
                    if is_selected:
                        false_candidate_selected += 1
                        false_selected = True
            if case.is_reference:
                refs += 1
                reference_false_cases += int(false_selected)
                target_index = next(
                    (idx for idx, candidate in enumerate(case.candidates) if candidate.label == 1),
                    -1,
                )
                if target_index >= 0:
                    target_present += 1
                    rank = target_rank(scores, target_index)
                    target_top1 += rank <= 1
                    target_top3 += rank <= 3
                    target_top5 += rank <= 5
                    target_selected += selected[target_index]
                    if (
                        fpr == 0.005
                        and len(failures) < 40
                        and not selected[target_index]
                    ):
                        best_index = max(range(len(scores)), key=lambda i: scores[i])
                        failures.append(
                            {
                                "view": view_name,
                                "fpr_target": fpr,
                                "case_id": case.case_id,
                                "label_class": case.label_class,
                                "failure_type": "target_not_selected",
                                "target_score": scores[target_index],
                                "best_score": scores[best_index],
                                "best_candidate_index": best_index,
                                "threshold": threshold,
                            }
                        )
            else:
                controls += 1
                control_false_cases += int(false_selected)
                if fpr == 0.005 and false_selected and len(failures) < 80:
                    best_index = max(range(len(scores)), key=lambda i: scores[i])
                    failures.append(
                        {
                            "view": view_name,
                            "fpr_target": fpr,
                            "case_id": case.case_id,
                            "label_class": case.label_class,
                            "failure_type": "control_false_proposal",
                            "target_score": 0.0,
                            "best_score": scores[best_index],
                            "best_candidate_index": best_index,
                            "threshold": threshold,
                        }
                    )

        false_candidate_rate = (
            false_candidate_selected / negative_count if negative_count else 0.0
        )
        rows.append(
            {
                "view": view_name,
                "fpr_target": fpr,
                "threshold": threshold,
                "reference_count": refs,
                "control_count": controls,
                "target_present": target_present,
                "target_selected": target_selected,
                "target_recall": target_selected / target_present if target_present else 0.0,
                "target_top1": target_top1 / target_present if target_present else 0.0,
                "target_top3": target_top3 / target_present if target_present else 0.0,
                "target_top5": target_top5 / target_present if target_present else 0.0,
                "false_candidate_selected": false_candidate_selected,
                "negative_candidate_count": negative_count,
                "false_candidate_rate": false_candidate_rate,
                "reference_false_case_rate": reference_false_cases / refs if refs else 0.0,
                "control_false_case_rate": control_false_cases / controls
                if controls
                else 0.0,
                "selected_candidate_count": selected_candidate_count,
                "passes_required_false_gate": false_candidate_rate < 0.005,
                "passes_target_false_gate": false_candidate_rate <= 0.001,
            }
        )
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        default="build/es_aist_contextual_anchor_shadow_sequence_mlp",
    )
    parser.add_argument(
        "--output-dir",
        default="build/es_aist_signal_contract_threshold_audit",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(artifact_dir)
    fpr_targets = [0.0, 0.001, 0.005, 0.01, 0.05]
    rows: list[dict[str, float | int | str | bool]] = []
    failures: list[dict[str, str | int | float]] = []
    for name, score_fn in score_views().items():
        view_rows, view_failures = audit_view(cases, name, score_fn, fpr_targets)
        rows.extend(view_rows)
        failures.extend(view_failures)

    csv_path = output_dir / "es_aist_signal_contract_threshold_audit.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    failure_path = output_dir / "es_aist_signal_contract_failure_examples.csv"
    with failure_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(failures[0].keys()))
        writer.writeheader()
        writer.writerows(failures)

    required_rows = [row for row in rows if row["fpr_target"] == 0.005]
    target_rows = [row for row in rows if row["fpr_target"] == 0.001]
    summary = {
        "mode": "es_aist_signal_contract_threshold_audit",
        "runtime_effect": "none",
        "production_retrieval_changed": False,
        "artifact_dir": str(artifact_dir),
        "case_count": len(cases),
        "reference_count": sum(1 for case in cases if case.is_reference),
        "control_count": sum(1 for case in cases if not case.is_reference),
        "views": sorted(score_views().keys()),
        "best_required_gate_row": max(
            required_rows,
            key=lambda row: (
                row["passes_required_false_gate"],
                row["target_recall"],
                -row["false_candidate_rate"],
            ),
        ),
        "best_target_gate_row": max(
            target_rows,
            key=lambda row: (
                row["passes_target_false_gate"],
                row["target_recall"],
                -row["false_candidate_rate"],
            ),
        ),
        "required_false_gate_target": 0.005,
        "strict_target_false_gate_target": 0.001,
        "interpretation": (
            "This audits whether current ES-AIST/WM/STM/LTM scalar candidate "
            "scores can be used directly as high-precision anchor-signal "
            "proposals. It does not evaluate a new model artifact."
        ),
    }
    json_path = output_dir / "es_aist_signal_contract_threshold_results.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
