#!/usr/bin/env python3
"""Summarize completed long-horizon mechanism sweep arm judgments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_ARMS = [
    "control",
    "emotion_mood_threshold_cascade",
    "neuromodulator_effect_scales",
    "daily_consolidation",
    "graph_expansion",
    "stm_ltm_graph_label_handoff",
    "synaptic_tag_ttl",
]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def sample_stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    center = mean(values)
    assert center is not None
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def judge_rows_path(judge_path: Path) -> Path:
    return judge_path.with_name(judge_path.name + ".rows.jsonl")


def metric_by_repetition(judge_path: Path, metric: str) -> list[float]:
    rows_path = judge_rows_path(judge_path)
    if not rows_path.exists():
        return []
    by_rep: dict[int, list[float]] = {}
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rep = row.get("repetition")
        value = row.get("systems", {}).get("cortext_native", {}).get(metric)
        if rep is None or value is None:
            continue
        by_rep.setdefault(int(rep), []).append(float(value))
    return [
        rep_mean
        for rep in sorted(by_rep)
        if (rep_mean := mean(by_rep[rep])) is not None
    ]


def matched_deltas(arm_values: list[float], control_values: list[float]) -> list[float]:
    count = min(len(arm_values), len(control_values))
    return [arm_values[index] - control_values[index] for index in range(count)]


def value_delta(values: list[float], first: int, second: int) -> float | None:
    if len(values) <= max(first, second):
        return None
    return values[first] - values[second]


def paired_delta_interpretation(
    arm: str,
    mean_delta: float | None,
    paired_deltas: list[float],
    signal_floor: float,
) -> str | None:
    if arm == "control":
        return "measured_control"
    if mean_delta is None or not paired_deltas:
        return None
    if mean_delta > 0:
        return "removal_positive"
    if all(delta <= -signal_floor for delta in paired_deltas):
        return "stable_removal_harm"
    after_rep1 = paired_deltas[1:]
    if after_rep1 and all(abs(delta) < signal_floor for delta in after_rep1):
        return "mean_loss_control_rep1_sensitive"
    if after_rep1 and any(delta >= 0 for delta in after_rep1):
        return "mixed_paired_deltas_control_sensitive"
    if mean_delta < 0:
        return "negative_mean_unresolved"
    return "null"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(root: Path, arms: list[str]) -> dict[str, Any]:
    judge_repetitions = int(os.environ.get("JUDGE_REPETITIONS", "3"))
    signal_floor = 0.15
    control_path = root / "control" / "judge.json"
    control_data = load_json(control_path)
    control_quality = control_data.get("quality", {}).get("cortext_native", {})
    control_tokens = control_data.get("tokens", {})
    control_token_by_system = control_tokens.get("mean_context_tokens_by_system", {})
    control_suff_by_rep = metric_by_repetition(control_path, "sufficiency")
    control = {
        "artifact": str(control_path),
        "wins": control_quality.get("wins"),
        "judged": control_data.get("judged"),
        "mean_sufficiency": control_quality.get("mean_sufficiency"),
        "mean_noise": control_quality.get("mean_noise"),
        "mean_context_tokens": control_token_by_system.get(
            "cortext_native",
            control_tokens.get("mean_cortext_context_tokens"),
        ),
        "mean_sufficiency_by_repetition": control_suff_by_rep,
        "mean_sufficiency_rep_sd": sample_stdev(control_suff_by_rep),
    }
    summary: dict[str, Any] = {
        "schema": "cortext_long_horizon_mechanism_sweep_summary_v2",
        "root": str(root),
        "protocol": {
            "screen": "18k-message context-blowout deferred-family removal",
            "input_dir": os.environ.get("INPUT_DIR", "/shared/Memory/Julie"),
            "max_messages": int(os.environ.get("MAX_MESSAGES", "18000")),
            "media_limit": int(os.environ.get("MEDIA_LIMIT", "128")),
            "warmup_events": int(os.environ.get("WARMUP_EVENTS", "0")),
            "probe_stride": int(os.environ.get("PROBE_STRIDE", "600")),
            "active_history_token_budget": int(
                os.environ.get("ACTIVE_HISTORY_TOKEN_BUDGET", "49152")
            ),
            "judge_repetitions": judge_repetitions,
            "judge_model": os.environ.get("JUDGE_MODEL", "qwen-omni-judge-32k"),
            "judge_context_window_tokens": int(
                os.environ.get("JUDGE_CONTEXT_WINDOW_TOKENS", "32768")
            ),
            "control_reference": control,
            "stability_evidence_fields": [
                "protocol.control_reference.mean_sufficiency_by_repetition",
                "arms.*.mean_sufficiency_by_repetition",
                "arms.*.delta_sufficiency_vs_control_by_repetition",
                "arms.*.delta_sufficiency_vs_control_rep_sd",
                "arms.*.paired_delta_interpretation",
            ],
            "control_rep1_sensitivity": {
                "control_mean_sufficiency_by_repetition": control_suff_by_rep or None,
                "rep1_minus_rep2": value_delta(control_suff_by_rep, 0, 1),
                "rep1_minus_rep3": value_delta(control_suff_by_rep, 0, 2),
                "arms_with_nonnegative_paired_delta_after_rep1": [],
                "arms_with_no_signal_sized_loss_after_rep1": [],
                "interpretation": (
                    "Control repetition 1 was higher than later control repetitions; "
                    "mean-vs-mean negative deltas should be read next to paired "
                    "per-repetition deltas before being treated as stable harm."
                ),
            },
            "verdict_rules": {
                "sufficiency_delta_signal_floor": signal_floor,
                "long_horizon_deferred_family": [
                    "emotion_mood_threshold_cascade",
                    "neuromodulator_effect_scales",
                    "daily_consolidation",
                    "graph_expansion",
                    "stm_ltm_graph_label_handoff",
                    "synaptic_tag_ttl",
                ],
            },
        },
        "arms": {},
    }
    control_suff = control.get("mean_sufficiency")
    for arm in arms:
        judge_path = root / arm / "judge.json"
        if not judge_path.exists():
            summary["arms"][arm] = {"error": f"missing {judge_path}"}
            continue
        data = load_json(judge_path)
        quality = data.get("quality", {}).get("cortext_native", {})
        token_by_system = data.get("tokens", {}).get("mean_context_tokens_by_system", {})
        mean_tokens = token_by_system.get(
            "cortext_native",
            data.get("tokens", {}).get("mean_cortext_context_tokens"),
        )
        suff = quality.get("mean_sufficiency")
        delta = None if suff is None or control_suff is None else suff - control_suff
        suff_by_rep = metric_by_repetition(judge_path, "sufficiency")
        delta_by_rep = matched_deltas(suff_by_rep, control_suff_by_rep)
        after_rep1 = delta_by_rep[1:]
        nonnegative_after_rep1 = bool(after_rep1) and any(
            value >= 0 for value in after_rep1
        )
        no_signal_sized_loss_after_rep1 = bool(after_rep1) and all(
            value > -signal_floor for value in after_rep1
        )
        if arm != "control" and nonnegative_after_rep1:
            summary["protocol"]["control_rep1_sensitivity"][
                "arms_with_nonnegative_paired_delta_after_rep1"
            ].append(arm)
        if arm != "control" and no_signal_sized_loss_after_rep1:
            summary["protocol"]["control_rep1_sensitivity"][
                "arms_with_no_signal_sized_loss_after_rep1"
            ].append(arm)
        if arm == "control" and delta == 0:
            reading = "measured_control"
        elif delta is None:
            reading = "missing"
        elif abs(delta) < signal_floor:
            reading = "null"
        elif delta < 0:
            reading = "removal_hurts"
        else:
            reading = "removal_positive"
        summary["arms"][arm] = {
            "artifact": str(judge_path),
            "judged": data.get("judged"),
            "probe_count": data.get("probe_count"),
            "wins": quality.get("wins"),
            "mean_sufficiency": suff,
            "mean_noise": quality.get("mean_noise"),
            "mean_context_tokens": mean_tokens,
            "delta_sufficiency_vs_control": delta,
            "mean_sufficiency_by_repetition": suff_by_rep or None,
            "mean_sufficiency_rep_sd": sample_stdev(suff_by_rep),
            "delta_sufficiency_vs_control_by_repetition": delta_by_rep or None,
            "delta_sufficiency_vs_control_rep_sd": sample_stdev(delta_by_rep),
            "paired_delta_after_rep1_signal_count": sum(
                1 for value in after_rep1 if abs(value) >= signal_floor
            ),
            "paired_delta_after_rep1_nonnegative_count": sum(
                1 for value in after_rep1 if value >= 0
            ),
            "paired_delta_interpretation": paired_delta_interpretation(
                arm,
                delta,
                delta_by_rep,
                signal_floor,
            ),
            "reading_by_sufficiency_rule": reading,
            "judgment_complete": data.get("judgment_complete"),
            "missing_judgments": data.get("missing_judgments"),
            "fairness_checks": data.get("fairness_checks"),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    args = parser.parse_args()
    arms = [arm for arm in args.arms.split(",") if arm]
    output = summarize(args.root, arms)
    out_path = args.root / "mechanism_sweep_summary.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
