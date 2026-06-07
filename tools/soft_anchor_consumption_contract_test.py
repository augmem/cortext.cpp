#!/usr/bin/env python3
"""Replay-test Soft Anchor consumption contracts.

The test consumes already-formed Soft Anchor candidate artifacts. It does not run
retrieval, does not create anchors from retrieved memories, and does not use
gold labels for runtime selection. Labels are used only after selection to score
whether a surfaced hint would have helped or harmed the replay case.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SUPPORTED_INPUT = "ingress_adaptive_anchor_candidates.csv"


def as_bool(value: str | int | bool | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def estimate_context_chars(row: dict[str, str], multiplier: float = 1.0) -> int:
    text = row.get("last_seen_text", "")
    return int(round(multiplier * (48 + min(len(text), 160))))


@dataclass(frozen=True)
class ConsumptionPolicy:
    name: str
    family: str
    description: str
    max_links: int = 0
    allowed_labels: tuple[str, ...] = ("tentative", "ambiguous", "durable")
    allowed_tiers: tuple[str, ...] = ("wm", "stm", "ltm")
    min_score: float = 0.0
    min_strength: float = 0.0
    min_posterior: float = 0.0
    min_margin: float | None = None
    max_entropy: float = 1.0
    max_generic: float = 1.0
    require_ambiguous: bool = False
    ask_instead_of_hint: bool = False
    prompt_char_multiplier: float = 1.0
    visible_only: bool = False
    supported: bool = True
    unsupported_reason: str = ""


@dataclass
class CaseGroup:
    formation_policy: str
    case_id: str
    rows: list[dict[str, str]] = field(default_factory=list)

    @property
    def meta(self) -> dict[str, str]:
        return self.rows[0]


def policy_catalog() -> list[ConsumptionPolicy]:
    """Return the planned consumption contracts.

    Unsupported entries are intentionally included so the output records which
    designs need additional traces before they can be judged.
    """

    return [
        ConsumptionPolicy(
            "C0_no_consumption",
            "baseline",
            "Form anchors but surface nothing.",
        ),
        ConsumptionPolicy(
            "C1_silent_safety",
            "silent_safety",
            "Use anchor uncertainty only as a future safety suppressor.",
        ),
        ConsumptionPolicy(
            "C2_possible_top1",
            "possible_continuity",
            "Surface one possible continuity hint.",
            max_links=1,
            min_score=0.54,
            min_strength=0.05,
            min_posterior=0.04,
            max_entropy=0.86,
            max_generic=0.82,
        ),
        ConsumptionPolicy(
            "C2_possible_top1_loose",
            "possible_continuity",
            "Surface one weak possible continuity hint.",
            max_links=1,
            min_score=0.46,
            min_strength=0.025,
            min_posterior=0.02,
            max_entropy=0.95,
            max_generic=0.90,
        ),
        ConsumptionPolicy(
            "C3_anchor_neighborhood_top1",
            "anchor_neighborhood",
            "Surface one anchor plus compact neighborhood evidence.",
            max_links=1,
            min_score=0.54,
            min_strength=0.05,
            min_posterior=0.04,
            max_entropy=0.86,
            max_generic=0.82,
            prompt_char_multiplier=2.25,
        ),
        ConsumptionPolicy(
            "C4_ambiguous_top3",
            "ambiguous_set",
            "Surface up to three candidates only when uncertainty is visible.",
            max_links=3,
            min_score=0.46,
            min_strength=0.025,
            min_posterior=0.02,
            max_entropy=0.95,
            max_generic=0.90,
            require_ambiguous=True,
        ),
        ConsumptionPolicy(
            "C5_clarify_on_ambiguous",
            "clarification",
            "Ask a clarification question when top candidates are ambiguous.",
            max_links=2,
            min_score=0.46,
            min_strength=0.025,
            min_posterior=0.02,
            max_entropy=0.95,
            max_generic=0.90,
            require_ambiguous=True,
            ask_instead_of_hint=True,
            prompt_char_multiplier=0.40,
        ),
        ConsumptionPolicy(
            "C6_llm_self_select_top3",
            "llm_self_selection",
            "Send structured top-k uncertain candidates to the LLM.",
            max_links=3,
            min_score=0.46,
            min_strength=0.025,
            min_posterior=0.02,
            max_entropy=0.95,
            max_generic=0.90,
            prompt_char_multiplier=1.35,
        ),
        ConsumptionPolicy(
            "C7_ui_only_top3",
            "ui_only",
            "Show top-k continuity chips to the human, not the LLM prompt.",
            max_links=3,
            min_score=0.46,
            min_strength=0.025,
            min_posterior=0.02,
            max_entropy=0.95,
            max_generic=0.90,
            visible_only=True,
        ),
        ConsumptionPolicy(
            "C8_retrieval_annotation",
            "retrieval_annotation",
            "Annotate returned retrieval memories with pre-formed anchors.",
            supported=False,
            unsupported_reason="requires retrieval-result trace with memory ids",
        ),
        ConsumptionPolicy(
            "C9_anchor_context_packing",
            "context_packing",
            "Group or deduplicate final context by formed anchor.",
            supported=False,
            unsupported_reason="requires context-pack trace",
        ),
        ConsumptionPolicy(
            "C10_retrieval_neighborhood_expansion",
            "neighborhood_expansion",
            "Expand ordinary retrieval hits through same-anchor neighbors.",
            supported=False,
            unsupported_reason="requires retrieval-result trace with memory ids",
        ),
        ConsumptionPolicy(
            "C11_current_turn_recall",
            "current_turn_recall",
            "Include the best evidence memory from the current formed anchor.",
            max_links=2,
            min_score=0.50,
            min_strength=0.04,
            min_posterior=0.02,
            max_entropy=0.92,
            max_generic=0.88,
        ),
        ConsumptionPolicy(
            "C12_durable_only_fact",
            "durable_fact",
            "Allow durable links only; intended for fact formation, not hints.",
            max_links=2,
            allowed_labels=("durable",),
            min_score=0.70,
            min_strength=0.80,
            min_posterior=0.10,
            max_entropy=0.70,
        ),
        ConsumptionPolicy(
            "C13_decayed_reminder",
            "decayed_reminder",
            "Surface only decayed links as older possible context.",
            max_links=2,
            allowed_labels=("decayed",),
            min_score=0.35,
            min_strength=0.02,
            prompt_char_multiplier=0.75,
        ),
        ConsumptionPolicy(
            "C14_group_anchor",
            "group_anchor",
            "Surface group/entity-set anchors.",
            supported=False,
            unsupported_reason="requires group-anchor composition metadata",
        ),
        ConsumptionPolicy(
            "C15_cross_modal_evidence",
            "cross_modal",
            "Surface modality-neutral cross-modal anchor evidence.",
            supported=False,
            unsupported_reason="requires multimodal evidence rows",
        ),
        ConsumptionPolicy(
            "C16_correction_aware",
            "correction_aware",
            "Consume user corrections to demote or reject links.",
            supported=False,
            unsupported_reason="requires correction events",
        ),
        ConsumptionPolicy(
            "C17_ask_or_show_hybrid",
            "ask_or_show",
            "Show a hint when moderately confident, ask when ambiguous.",
            max_links=2,
            min_score=0.50,
            min_strength=0.04,
            min_posterior=0.02,
            max_entropy=0.92,
            max_generic=0.88,
        ),
    ]


def is_ambiguous_case(rows: list[dict[str, str]]) -> bool:
    if len(rows) > 1:
        return True
    meta = rows[0]
    return as_float(meta, "entropy") >= 0.70 or as_float(meta, "margin") < 0.12


def select_rows(
    case: CaseGroup, policy: ConsumptionPolicy
) -> tuple[str, list[dict[str, str]]]:
    if not policy.supported or policy.max_links <= 0:
        return ("none", [])
    if policy.require_ambiguous and not is_ambiguous_case(case.rows):
        return ("none", [])

    meta = case.meta
    if as_float(meta, "generic_score") > policy.max_generic:
        return ("none", [])
    if as_float(meta, "entropy") > policy.max_entropy:
        return ("none", [])
    if policy.min_margin is not None and as_float(meta, "margin") < policy.min_margin:
        return ("none", [])

    selected: list[dict[str, str]] = []
    for row in sorted(case.rows, key=lambda item: as_int(item, "candidate_rank")):
        if row.get("anchor_label") not in policy.allowed_labels:
            continue
        if row.get("tier") not in policy.allowed_tiers:
            continue
        if as_float(row, "score") < policy.min_score:
            continue
        if as_float(row, "anchor_strength") < policy.min_strength:
            continue
        if as_float(row, "attention") < policy.min_posterior:
            continue
        selected.append(row)
        if len(selected) >= policy.max_links:
            break

    if not selected:
        return ("none", [])
    if policy.ask_instead_of_hint:
        return ("clarify", selected)
    if policy.name == "C17_ask_or_show_hybrid" and is_ambiguous_case(case.rows):
        return ("clarify", selected)
    return ("surface", selected)


def read_cases(path: Path, formation_policies: set[str]) -> list[CaseGroup]:
    grouped: dict[tuple[str, str], CaseGroup] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            formation_policy = row["policy"]
            if formation_policies and formation_policy not in formation_policies:
                continue
            key = (formation_policy, row["case_id"])
            grouped.setdefault(
                key, CaseGroup(formation_policy=formation_policy, case_id=row["case_id"])
            ).rows.append(row)
    return list(grouped.values())


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    refs = [row for row in rows if row["is_reference"]]
    controls = [row for row in rows if not row["is_reference"]]
    useful = sum(1 for row in rows if row["useful_context"])
    harmful = sum(1 for row in rows if row["harmful_context"])
    evidence_useful = sum(1 for row in rows if row["evidence_packet_useful"])
    evidence_harmful = sum(1 for row in rows if row["evidence_packet_harmful"])
    no_anchor = sum(1 for row in controls if row["surfaced_any"])
    wrong_only = sum(1 for row in refs if row["surfaced_wrong_without_target"])
    stale_only = sum(1 for row in refs if row["surfaced_stale_without_target"])
    target = sum(1 for row in refs if row["surfaced_target"])
    target_top1 = sum(1 for row in refs if row["surfaced_target_top1"])
    clarified = sum(1 for row in rows if row["action"] == "clarify")
    surfaces = sum(int(row["surfaced_link_count"]) for row in rows)
    prompt_chars = [float(row["prompt_context_chars"]) for row in rows]
    visible_chars = [float(row["visible_context_chars"]) for row in rows]
    no_anchor_rate = no_anchor / len(controls) if controls else 0.0
    wrong_only_rate = wrong_only / len(refs) if refs else 0.0
    useful_harmful = useful / harmful if harmful else float(useful)
    evidence_ratio = (
        evidence_useful / evidence_harmful
        if evidence_harmful
        else float(evidence_useful)
    )
    strict_replay_pass = (
        useful > 0
        and useful_harmful >= 5.0
        and no_anchor_rate < 0.02
        and wrong_only_rate < 0.01
        and p95(prompt_chars) <= 600
    )
    evidence_packet_pass = (
        evidence_useful > 0
        and evidence_ratio >= 5.0
        and wrong_only_rate < 0.01
        and p95(prompt_chars) <= 600
    )
    return {
        "reference_count": len(refs),
        "control_count": len(controls),
        "target_surfaced_count": target,
        "target_surfaced_rate": target / len(refs) if refs else 0.0,
        "target_top1_count": target_top1,
        "target_top1_rate": target_top1 / len(refs) if refs else 0.0,
        "no_anchor_surfaced_count": no_anchor,
        "no_anchor_surfaced_rate": no_anchor_rate,
        "wrong_active_without_target_count": wrong_only,
        "wrong_active_without_target_rate": wrong_only_rate,
        "stale_without_target_count": stale_only,
        "stale_without_target_rate": stale_only / len(refs) if refs else 0.0,
        "useful_context_count": useful,
        "harmful_context_count": harmful,
        "useful_to_harmful_ratio": useful_harmful,
        "evidence_packet_useful_count": evidence_useful,
        "evidence_packet_harmful_count": evidence_harmful,
        "evidence_packet_useful_to_harmful_ratio": evidence_ratio,
        "clarification_count": clarified,
        "surfaced_link_count": surfaces,
        "mean_surfaced_links": surfaces / len(rows) if rows else 0.0,
        "mean_prompt_context_chars": sum(prompt_chars) / len(prompt_chars)
        if prompt_chars
        else 0.0,
        "p95_prompt_context_chars": p95(prompt_chars),
        "mean_visible_context_chars": sum(visible_chars) / len(visible_chars)
        if visible_chars
        else 0.0,
        "p95_visible_context_chars": p95(visible_chars),
        "zero_harm_recovery": target if harmful == 0 else 0,
        "strict_replay_pass": strict_replay_pass,
        "evidence_packet_pass": evidence_packet_pass,
        "promising_for_manual_review": useful > 0
        and evidence_ratio >= 5.0
        and wrong_only_rate < 0.02,
    }


def run_policy(cases: Iterable[CaseGroup], policy: ConsumptionPolicy) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for case in cases:
        action, selected = select_rows(case, policy)
        is_reference = as_bool(case.meta["is_reference"])
        surfaced_target = any(as_bool(row["is_target"]) for row in selected)
        surfaced_wrong = any(
            as_bool(row["is_wrong_active"]) and not as_bool(row["is_target"])
            for row in selected
        )
        surfaced_stale = any(
            as_bool(row["is_stale"]) and not as_bool(row["is_target"])
            for row in selected
        )
        no_anchor_surfaced = (not is_reference) and bool(selected)
        wrong_without_target = is_reference and surfaced_wrong and not surfaced_target
        stale_without_target = is_reference and surfaced_stale and not surfaced_target
        useful = is_reference and surfaced_target
        harmful = no_anchor_surfaced or wrong_without_target or stale_without_target
        evidence_packet_useful = useful
        # Evidence packets are uncertain candidate context, not asserted facts.
        # A no-anchor packet is noise/load for manual review, but it is not
        # automatically a harmful commitment unless the packet contains only a
        # wrong/stale referent for a reference case.
        evidence_packet_harmful = wrong_without_target or stale_without_target
        prompt_chars = 0
        visible_chars = 0
        for row in selected:
            chars = estimate_context_chars(row, policy.prompt_char_multiplier)
            if policy.visible_only:
                visible_chars += chars
            else:
                prompt_chars += chars
                visible_chars += chars
        top = selected[0] if selected else {}
        output.append(
            {
                "consumer_policy": policy.name,
                "consumer_family": policy.family,
                "formation_policy": case.formation_policy,
                "case_id": case.case_id,
                "dataset": case.meta.get("dataset", ""),
                "conversation_id": case.meta.get("conversation_id", ""),
                "label_class": case.meta.get("label_class", ""),
                "is_reference": is_reference,
                "current_ingress_step": as_int(case.meta, "current_ingress_step"),
                "current_signal_text": case.meta.get("current_signal_text", ""),
                "action": action,
                "surfaced_link_count": len(selected),
                "prompt_context_chars": prompt_chars,
                "visible_context_chars": visible_chars,
                "top_anchor_id": top.get("anchor_id", ""),
                "top_anchor_label": top.get("anchor_label", "none"),
                "top_tier": top.get("tier", "none"),
                "top_score": as_float(top, "score", -1.0) if top else -1.0,
                "top_posterior": as_float(top, "attention") if top else 0.0,
                "top_strength": as_float(top, "anchor_strength") if top else 0.0,
                "margin": as_float(case.meta, "margin"),
                "entropy": as_float(case.meta, "entropy"),
                "generic_score": as_float(case.meta, "generic_score"),
                "surfaced_any": bool(selected),
                "surfaced_target": surfaced_target,
                "surfaced_target_top1": bool(selected)
                and as_bool(selected[0]["is_target"]),
                "surfaced_wrong_active": surfaced_wrong,
                "surfaced_stale": surfaced_stale,
                "surfaced_wrong_without_target": wrong_without_target,
                "surfaced_stale_without_target": stale_without_target,
                "no_anchor_surfaced": no_anchor_surfaced,
                "useful_context": useful,
                "harmful_context": harmful,
                "evidence_packet_useful": evidence_packet_useful,
                "evidence_packet_harmful": evidence_packet_harmful,
                "runtime_policy_uses_labels": False,
            }
        )
    return output


def write_cases(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "consumer_policy",
        "consumer_family",
        "formation_policy",
        "case_id",
        "dataset",
        "conversation_id",
        "label_class",
        "is_reference",
        "current_ingress_step",
        "current_signal_text",
        "action",
        "surfaced_link_count",
        "prompt_context_chars",
        "visible_context_chars",
        "top_anchor_id",
        "top_anchor_label",
        "top_tier",
        "top_score",
        "top_posterior",
        "top_strength",
        "margin",
        "entropy",
        "generic_score",
        "surfaced_any",
        "surfaced_target",
        "surfaced_target_top1",
        "surfaced_wrong_active",
        "surfaced_stale",
        "surfaced_wrong_without_target",
        "surfaced_stale_without_target",
        "no_anchor_surfaced",
        "useful_context",
        "harmful_context",
        "evidence_packet_useful",
        "evidence_packet_harmful",
        "runtime_policy_uses_labels",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summaries: list[dict[str, object]]) -> None:
    fields = [
        "consumer_policy",
        "consumer_family",
        "formation_policy",
        "supported",
        "unsupported_reason",
        "reference_count",
        "control_count",
        "target_surfaced_count",
        "target_surfaced_rate",
        "target_top1_count",
        "target_top1_rate",
        "no_anchor_surfaced_count",
        "no_anchor_surfaced_rate",
        "wrong_active_without_target_count",
        "wrong_active_without_target_rate",
        "stale_without_target_count",
        "stale_without_target_rate",
        "useful_context_count",
        "harmful_context_count",
        "useful_to_harmful_ratio",
        "evidence_packet_useful_count",
        "evidence_packet_harmful_count",
        "evidence_packet_useful_to_harmful_ratio",
        "clarification_count",
        "surfaced_link_count",
        "mean_surfaced_links",
        "mean_prompt_context_chars",
        "p95_prompt_context_chars",
        "mean_visible_context_chars",
        "p95_visible_context_chars",
        "zero_harm_recovery",
        "strict_replay_pass",
        "evidence_packet_pass",
        "promising_for_manual_review",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def write_failures(path: Path, rows: list[dict[str, object]], limit: int) -> None:
    failures = []
    for row in rows:
        failure = ""
        if row["is_reference"] and not row["surfaced_target"]:
            failure = "target_not_surfaced"
        elif row["surfaced_wrong_without_target"]:
            failure = "wrong_active_without_target"
        elif row["surfaced_stale_without_target"]:
            failure = "stale_without_target"
        elif row["no_anchor_surfaced"]:
            failure = "no_anchor_surfaced"
        if not failure:
            continue
        item = dict(row)
        item["failure_type"] = failure
        failures.append(item)
        if len(failures) >= limit:
            break
    fields = [
        "consumer_policy",
        "formation_policy",
        "case_id",
        "label_class",
        "failure_type",
        "current_ingress_step",
        "current_signal_text",
        "action",
        "surfaced_link_count",
        "top_anchor_id",
        "top_anchor_label",
        "top_tier",
        "top_score",
        "top_posterior",
        "top_strength",
        "margin",
        "entropy",
        "generic_score",
        "prompt_context_chars",
        "visible_context_chars",
        "evidence_packet_useful",
        "evidence_packet_harmful",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(failures)


def default_formation_policies(path: Path) -> list[str]:
    available = set()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            available.add(row["policy"])
    preferred = [
        "adaptive_soft_anchor_v1",
        "adaptive_soft_anchor_v1_update_strict",
        "adaptive_soft_anchor_v1_update_very_strict",
        "adaptive_soft_anchor_v1_high_focus",
        "adaptive_soft_anchor_v1_high_sensitivity",
        "adaptive_soft_anchor_v1_no_generic",
        "adaptive_soft_anchor_v1_f25_s75_t50",
        "adaptive_soft_anchor_v1_f75_s50_t50",
    ]
    return [policy for policy in preferred if policy in available]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("build/soft_anchor_consumption_full_ablation_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/soft_anchor_consumption_contract_test"),
    )
    parser.add_argument(
        "--formation-policy",
        action="append",
        default=[],
        help="Formation policy to test. May be repeated. Defaults to curated policies.",
    )
    parser.add_argument(
        "--all-formation-policies",
        action="store_true",
        help="Evaluate all formation policies found in the candidate artifact.",
    )
    parser.add_argument("--failure-limit", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input_dir / SUPPORTED_INPUT
    if not input_path.exists():
        raise FileNotFoundError(f"missing input artifact: {input_path}")

    if args.all_formation_policies:
        formation_policies: set[str] = set()
    elif args.formation_policy:
        formation_policies = set(args.formation_policy)
    else:
        formation_policies = set(default_formation_policies(input_path))

    cases = read_cases(input_path, formation_policies)
    if not cases:
        raise RuntimeError("no cases matched the requested formation policies")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policies = policy_catalog()
    all_case_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    unsupported: list[dict[str, str]] = []

    by_formation: dict[str, list[CaseGroup]] = defaultdict(list)
    for case in cases:
        by_formation[case.formation_policy].append(case)

    for policy in policies:
        if not policy.supported:
            unsupported.append(
                {
                    "consumer_policy": policy.name,
                    "consumer_family": policy.family,
                    "unsupported_reason": policy.unsupported_reason,
                }
            )
            for formation_policy in sorted(by_formation):
                summaries.append(
                    {
                        "consumer_policy": policy.name,
                        "consumer_family": policy.family,
                        "formation_policy": formation_policy,
                        "supported": False,
                        "unsupported_reason": policy.unsupported_reason,
                        **summarize_rows([]),
                    }
                )
            continue
        for formation_policy, formation_cases in sorted(by_formation.items()):
            rows = run_policy(formation_cases, policy)
            all_case_rows.extend(rows)
            summaries.append(
                {
                    "consumer_policy": policy.name,
                    "consumer_family": policy.family,
                    "formation_policy": formation_policy,
                    "supported": True,
                    "unsupported_reason": "",
                    **summarize_rows(rows),
                }
            )

    supported_summaries = [row for row in summaries if row["supported"]]
    best_by_useful = max(
        supported_summaries,
        key=lambda row: (
            row["strict_replay_pass"],
            row["useful_to_harmful_ratio"],
            row["target_surfaced_count"],
            -row["harmful_context_count"],
        ),
    )
    best_recall = max(
        supported_summaries,
        key=lambda row: (row["target_surfaced_count"], -row["harmful_context_count"]),
    )
    best_evidence_packet = max(
        supported_summaries,
        key=lambda row: (
            row["evidence_packet_pass"],
            row["evidence_packet_useful_to_harmful_ratio"],
            row["target_surfaced_count"],
            -row["evidence_packet_harmful_count"],
            -row["no_anchor_surfaced_count"],
        ),
    )
    strict_passes = [row for row in supported_summaries if row["strict_replay_pass"]]
    evidence_passes = [
        row for row in supported_summaries if row["evidence_packet_pass"]
    ]
    promising = [
        row for row in supported_summaries if row["promising_for_manual_review"]
    ]

    write_summary(args.output_dir / "soft_anchor_consumption_contract_summary.csv", summaries)
    write_cases(args.output_dir / "soft_anchor_consumption_contract_cases.csv", all_case_rows)
    write_failures(
        args.output_dir / "soft_anchor_consumption_contract_failures.csv",
        all_case_rows,
        args.failure_limit,
    )
    with (args.output_dir / "soft_anchor_consumption_unsupported.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "consumer_policy",
                "consumer_family",
                "unsupported_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(unsupported)
    result = {
        "input_artifact": str(input_path),
        "output_dir": str(args.output_dir),
        "formation_policy_count": len(by_formation),
        "case_count": len(cases),
        "consumer_policy_count": len(policies),
        "supported_consumer_policy_count": sum(1 for policy in policies if policy.supported),
        "unsupported_consumer_policy_count": len(unsupported),
        "runtime_policy_uses_labels": False,
        "uses_retrieval_candidates": False,
        "strict_replay_pass_count": len(strict_passes),
        "evidence_packet_pass_count": len(evidence_passes),
        "promising_for_manual_review_count": len(promising),
        "best_strict_or_ratio_variant": best_by_useful,
        "best_evidence_packet_variant": best_evidence_packet,
        "best_recall_variant": best_recall,
        "strict_replay_passes": strict_passes,
        "evidence_packet_passes": evidence_passes,
        "promising_for_manual_review": promising,
        "unsupported": unsupported,
    }
    with (args.output_dir / "soft_anchor_consumption_contract_results.json").open(
        "w"
    ) as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
