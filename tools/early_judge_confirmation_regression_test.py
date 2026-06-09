#!/usr/bin/env python3
"""Regression checks for streamed early-judge confirmation accounting."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile


TOOLS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


judge = load_module("judge_julie_live_run_test_target", TOOLS_DIR / "judge_julie_live_run.py")
watcher = load_module(
    "watch_julie_probe_stream_judge_test_target",
    TOOLS_DIR / "watch_julie_probe_stream_judge.py",
)
report = load_module(
    "julie_release_protocol_report_test_target",
    TOOLS_DIR / "julie_release_protocol_report.py",
)
ablations = load_module(
    "run_julie_release_ablations_test_target",
    TOOLS_DIR / "run_julie_release_ablations.py",
)


def row(event_index: int, repetition: int, cortext: tuple[float, float, float], rag: tuple[float, float, float]) -> dict:
    cr, cs, cn = cortext
    rr, rs, rn = rag
    return {
        "event_index": event_index,
        "repetition": repetition,
        "systems": {
            "cortext_native": {
                "relevance": cr,
                "sufficiency": cs,
                "noise": cn,
            },
            "traditional_chat_rag": {
                "relevance": rr,
                "sufficiency": rs,
                "noise": rn,
            },
        },
    }


BAD_ROW = row(1, 0, (1.0, 1.0, 5.0), (5.0, 5.0, 0.0))
GOOD_ROW = row(2, 0, (5.0, 5.0, 0.0), (0.0, 0.0, 5.0))


def write_rows(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(item, separators=(",", ":")) + "\n" for item in rows))


def test_unrecoverable_bound_waits_until_recovery_is_impossible() -> None:
    early = judge.unrecoverable_quality_stop([BAD_ROW], 60, -0.5)
    assert early is None, early

    late = judge.unrecoverable_quality_stop([BAD_ROW] * 59, 60, -0.5)
    assert late is not None
    assert late["reason"] == "quality_delta_floor_unrecoverable"
    assert late["rows_judged"] == 59
    assert late["expected_rows"] == 60
    assert late["optimistic_final_quality_delta_vs_rag"] < -0.5


def test_prior_segments_are_counted_before_stopping_delta_confirmation() -> None:
    protected = judge.unrecoverable_quality_stop(
        [BAD_ROW] * 9,
        10,
        -0.5,
        prior_quality_delta_sum=150.0,
        prior_judgment_count=10,
    )
    assert protected is None, protected

    failing = judge.unrecoverable_quality_stop(
        [BAD_ROW] * 9,
        10,
        -0.5,
        prior_quality_delta_sum=-150.0,
        prior_judgment_count=10,
    )
    assert failing is not None
    assert failing["segment_rows_judged"] == 9
    assert failing["segment_expected_rows"] == 10
    assert failing["prior_rows"] == 10
    assert failing["expected_rows"] == 20


def test_prior_segment_stats_deduplicate_rows_like_cumulative_aggregate() -> None:
    with tempfile.TemporaryDirectory(prefix="cortext_early_judge_regression_") as tmp:
        root = pathlib.Path(tmp)
        seg_a = root / "seg_a.json"
        seg_b = root / "seg_b.json"
        seg_a.write_text("{}\n")
        seg_b.write_text("{}\n")
        write_rows(
            seg_a.with_name(seg_a.name + ".rows.jsonl"),
            [
                GOOD_ROW,
                row(3, 0, (4.0, 4.0, 1.0), (3.0, 3.0, 1.0)),
            ],
        )
        write_rows(
            seg_b.with_name(seg_b.name + ".rows.jsonl"),
            [
                GOOD_ROW,
                BAD_ROW,
            ],
        )

        delta_sum, count = watcher.prior_quality_delta_stats([seg_a, seg_b])
        assert count == 3
        assert delta_sum == 15.0 + 2.0 - 13.0


def test_release_report_surfaces_early_stop_metadata() -> None:
    early_stop = {
        "reason": "quality_delta_floor_unrecoverable",
        "rows_judged": 19,
        "expected_rows": 20,
    }
    status = {
        "schema": "cortext_julie_release_benchmark_status_v1",
        "status": "early_judge_failed",
        "early_judge_exit_code": 2,
        "probe_stream": {"rows": 20, "required_rows_after_benchmark": 10},
        "early_judge_latest": {
            "schema": "julie_probe_stream_early_judge_latest_v1",
            "judge_provider": "ollama",
            "judge_model": "gemma4:12b-it-qat",
            "judge_repetitions": 1,
            "confirm_fail_repetitions": 3,
            "fixed_milestones": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "periodic_stride": 1,
            "quality_gate_min_milestone": 8,
            "quality_trend_gate_min_milestone": 4,
            "quality_trend_window": 2,
            "quality_gate_requires_rag_pressure": True,
            "quality_gate_min_history_budget_ratio": 1.0,
            "latest": {
                "milestone": 20,
                "fail_fast_status": "fail",
                "quality_gate_active": True,
                "early_stop": early_stop,
            },
        },
    }
    manifest = {
        "schema": "julie_probe_stream_early_judge_manifest_v1",
        "sha256": "abc123",
        "blind_packets": True,
        "judge_packet_item_limit": 256,
        "completed": [
            {
                "milestone": 20,
                "fail_fast_status": "fail",
                "early_stop": early_stop,
            }
        ],
    }
    summary = report.early_judge_summary(status, manifest)
    assert summary["latest_early_stop"] == early_stop
    assert summary["completed_early_stop_count"] == 1
    gate = [
        item
        for item in report.early_judge_checks(summary)
        if item["name"] == "early_judge_gate_passed_for_final_release"
    ][0]
    assert gate["status"] == "fail"
    assert "quality_delta_floor_unrecoverable" in gate["detail"]


def test_release_ablation_auto_mode_is_strict_when_final_pass_required() -> None:
    assert ablations.strict_early_judge_required("on", False, False) is True
    assert ablations.strict_early_judge_required("auto", False, True) is False
    assert ablations.strict_early_judge_required("auto", True, True) is True
    assert ablations.strict_early_judge_required("off", True, False) is False


def main() -> int:
    test_unrecoverable_bound_waits_until_recovery_is_impossible()
    test_prior_segments_are_counted_before_stopping_delta_confirmation()
    test_prior_segment_stats_deduplicate_rows_like_cumulative_aggregate()
    test_release_report_surfaces_early_stop_metadata()
    test_release_ablation_auto_mode_is_strict_when_final_pass_required()
    print("early judge confirmation regression checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
