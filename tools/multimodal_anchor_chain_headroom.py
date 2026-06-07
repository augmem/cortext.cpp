#!/usr/bin/env python3
"""Synthetic multimodal anchor-chain headroom audit.

This benchmark-only script tests the structure implied by mixed references such
as Jared -> image -> dog -> "he had it". It does not change production
retrieval and does not claim real model evidence. It asks what kind of upstream
state is required before an engine policy can anchor modality-agnostic incoming
episodes by entity/object/relation continuity.
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


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    kind: str
    tracks: tuple[str, ...]
    last_seen_step: int
    is_target: bool
    is_wrong_active: bool = False
    is_stale: bool = False
    is_remote: bool = False


@dataclass(frozen=True)
class ChainCase:
    case_id: str
    family: str
    modality: str
    current_text: str
    current_step: int
    is_reference: bool
    target_kind: str
    required_tracks: tuple[str, ...]
    candidates: tuple[Candidate, ...]


def stable_unit(*parts: object) -> float:
    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def clamp(value: float, lo: float = -10.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, value))


def make_cases(episodes: int) -> list[ChainCase]:
    cases: list[ChainCase] = []
    for i in range(episodes):
        prefix = f"ep{i:04d}"
        jared = f"{prefix}:person:jared"
        alex = f"{prefix}:person:alex"
        maya = f"{prefix}:person:maya"
        dog = f"{prefix}:object:jared_dog"
        alex_dog = f"{prefix}:object:alex_dog"
        pool_image = f"{prefix}:media:pool_image"
        doc_image = f"{prefix}:media:doc_image"
        work_doc = f"{prefix}:object:work_doc"
        source_offset = i % 3

        def candidate(
            cid: str,
            kind: str,
            tracks: Iterable[str],
            last_seen: int,
            target: bool,
            wrong: bool = False,
            stale: bool = False,
            remote: bool = False,
        ) -> Candidate:
            return Candidate(
                candidate_id=cid,
                kind=kind,
                tracks=tuple(tracks),
                last_seen_step=last_seen,
                is_target=target,
                is_wrong_active=wrong,
                is_stale=stale,
                is_remote=remote,
            )

        person_candidates = (
            candidate("jared", "person", (jared,), 1 + source_offset, True),
            candidate("alex_recent", "person", (alex,), 3 + source_offset, False, True),
            candidate("maya_remote", "person", (maya,), -8, False, remote=True),
        )
        cases.append(
            ChainCase(
                case_id=f"{prefix}:he_phone",
                family="person_pronoun",
                modality="text",
                current_text="Great, what did he say?",
                current_step=4 + source_offset,
                is_reference=True,
                target_kind="person",
                required_tracks=(jared,),
                candidates=person_candidates,
            )
        )

        media_candidates = (
            candidate("pool_image", "media", (pool_image, dog), 5 + source_offset, True),
            candidate("doc_image_recent", "media", (doc_image, work_doc), 6 + source_offset, False, True),
            candidate("old_pool_image", "media", (pool_image, dog), -3, False, stale=True),
        )
        cases.append(
            ChainCase(
                case_id=f"{prefix}:send_it",
                family="media_object_reference",
                modality="text",
                current_text="Can you send it to me?",
                current_step=7 + source_offset,
                is_reference=True,
                target_kind="media",
                required_tracks=(pool_image,),
                candidates=media_candidates,
            )
        )

        object_candidates = (
            candidate("jared_dog", "object", (dog,), 5 + source_offset, True),
            candidate("alex_dog_recent", "object", (alex_dog,), 6 + source_offset, False, True),
            candidate("work_doc", "object", (work_doc,), 6 + source_offset, False),
        )
        cases.append(
            ChainCase(
                case_id=f"{prefix}:how_long_had_it",
                family="owner_object_relation",
                modality="text",
                current_text="How long has he had it?",
                current_step=8 + source_offset,
                is_reference=True,
                target_kind="relation",
                required_tracks=(jared, dog),
                candidates=(
                    candidate("jared_owns_dog", "relation", (jared, dog), 5 + source_offset, True),
                    candidate(
                        "alex_owns_dog_recent",
                        "relation",
                        (alex, dog),
                        6 + source_offset,
                        False,
                        True,
                    ),
                    candidate(
                        "jared_owns_work_doc",
                        "relation",
                        (jared, work_doc),
                        6 + source_offset,
                        False,
                    ),
                    candidate(
                        "stale_jared_owns_dog",
                        "relation",
                        (jared, dog),
                        -5,
                        False,
                        stale=True,
                    ),
                ),
            )
        )

        cases.append(
            ChainCase(
                case_id=f"{prefix}:wrong_active_work_doc",
                family="wrong_active_person",
                modality="text",
                current_text="He sent the work document.",
                current_step=7 + source_offset,
                is_reference=True,
                target_kind="person",
                required_tracks=(alex,),
                candidates=(
                    candidate("alex", "person", (alex,), 6 + source_offset, True),
                    candidate("jared_stale", "person", (jared,), 1 + source_offset, False, stale=True),
                    candidate("maya_remote", "person", (maya,), -8, False, remote=True),
                ),
            )
        )

        cases.append(
            ChainCase(
                case_id=f"{prefix}:image_from_audio",
                family="cross_modal_person_to_media",
                modality="image",
                current_text="<image> blurry dog photo sent by Jared",
                current_step=6 + source_offset,
                is_reference=True,
                target_kind="media",
                required_tracks=(pool_image, dog, jared),
                candidates=(
                    candidate("pool_image_from_jared", "media", (pool_image, dog, jared), 6 + source_offset, True),
                    candidate("doc_image_from_alex", "media", (doc_image, work_doc, alex), 6 + source_offset, False, True),
                    candidate("remote_dog_image", "media", (alex_dog,), -8, False, remote=True),
                ),
            )
        )

        cases.append(
            ChainCase(
                case_id=f"{prefix}:topic_shift",
                family="no_anchor_topic_shift",
                modality="text",
                current_text="The dishwasher broke this morning.",
                current_step=9 + source_offset,
                is_reference=False,
                target_kind="none",
                required_tracks=(),
                candidates=(
                    candidate("jared_active", "person", (jared,), 5 + source_offset, False, True),
                    candidate("pool_image_active", "media", (pool_image, dog), 6 + source_offset, False, True),
                    candidate("work_doc_active", "object", (work_doc,), 6 + source_offset, False),
                ),
            )
        )
    return cases


def base_score(case: ChainCase, candidate: Candidate) -> float:
    age = max(0, case.current_step - candidate.last_seen_step)
    recency = 1.0 / (1.0 + age)
    kind_match = 0.24 if candidate.kind == case.target_kind else 0.0
    active_bias = 0.14 if candidate.is_wrong_active else 0.0
    stale_penalty = -0.06 if candidate.is_stale else 0.0
    remote_penalty = -0.12 if candidate.is_remote else 0.0
    noise = 0.035 * (stable_unit("base", case.case_id, candidate.candidate_id) - 0.5)
    return recency + kind_match + active_bias + stale_penalty + remote_penalty + noise


def candidate_signal_match(
    case: ChainCase,
    candidate: Candidate,
    policy: str,
    recall: float,
    false_rate: float,
    relation_recall: float,
    relation_false_rate: float,
    seed: int,
) -> bool:
    if policy == "recency_only":
        return False
    required = set(case.required_tracks)
    tracks = set(candidate.tracks)
    exact = case.is_reference and candidate.is_target
    if policy == "entity_only":
        supports_kind = candidate.kind == "person"
    elif policy == "entity_object":
        supports_kind = candidate.kind in {"person", "object", "media"}
    else:
        supports_kind = candidate.kind in {"person", "object", "media", "relation"}

    if exact and supports_kind:
        if candidate.kind == "relation":
            if policy != "entity_relation":
                return False
            return stable_unit(seed, "relation_target", case.case_id) < relation_recall
        return stable_unit(seed, "target", case.case_id, candidate.candidate_id) < recall

    if not exact and supports_kind:
        if candidate.kind == "relation":
            if policy != "entity_relation":
                return False
            return stable_unit(seed, "relation_false", case.case_id, candidate.candidate_id) < relation_false_rate
        if tracks and required and tracks.intersection(required):
            return stable_unit(seed, "partial_false", case.case_id, candidate.candidate_id) < false_rate
        return stable_unit(seed, "false", case.case_id, candidate.candidate_id) < false_rate
    return False


def scores_for_case(
    case: ChainCase,
    policy: str,
    strength: float,
    recall: float,
    false_rate: float,
    relation_recall: float,
    relation_false_rate: float,
    seed: int,
) -> list[float]:
    out = []
    for candidate in case.candidates:
        score = base_score(case, candidate)
        if candidate_signal_match(
            case,
            candidate,
            policy,
            recall,
            false_rate,
            relation_recall,
            relation_false_rate,
            seed,
        ):
            score += strength
        out.append(clamp(score))
    return out


def rank(scores: list[float], index: int) -> int:
    return 1 + sum(score > scores[index] for score in scores)


def threshold(control_scores: list[float], fpr: float) -> float:
    if not control_scores:
        return math.inf
    ordered = sorted(control_scores, reverse=True)
    if fpr <= 0:
        return ordered[0] + 1.0e-9
    allowed = max(1, math.floor(len(ordered) * fpr))
    return ordered[min(allowed - 1, len(ordered) - 1)]


def summarize(cases: list[ChainCase], scores_by_case: list[list[float]]) -> dict[str, float | int]:
    refs = 0
    controls = 0
    top1 = top3 = safe = wrong = stale = 0
    ref_scores: list[float] = []
    ref_best_scores: list[float] = []
    ref_best_is_target: list[bool] = []
    control_scores: list[float] = []
    family_counts: dict[str, int] = {}
    family_top1: dict[str, int] = {}
    for case, scores in zip(cases, scores_by_case):
        if case.is_reference:
            refs += 1
            family_counts[case.family] = family_counts.get(case.family, 0) + 1
            target_index = next((i for i, c in enumerate(case.candidates) if c.is_target), -1)
            if target_index < 0:
                ref_scores.append(0.0)
                continue
            target_rank = rank(scores, target_index)
            top1 += target_rank <= 1
            top3 += target_rank <= 3
            family_top1[case.family] = family_top1.get(case.family, 0) + int(target_rank <= 1)
            best_index = max(range(len(scores)), key=lambda i: scores[i])
            best = case.candidates[best_index]
            safe += best.is_target
            wrong += best.is_wrong_active
            stale += best.is_stale
            ref_scores.append(scores[target_index])
            ref_best_scores.append(scores[best_index])
            ref_best_is_target.append(best.is_target)
        else:
            controls += 1
            control_scores.append(max(scores) if scores else 0.0)
    zero_t = threshold(control_scores, 0.0)
    five_t = threshold(control_scores, 0.05)
    zero_safe = sum(
        is_target and score >= zero_t
        for is_target, score in zip(ref_best_is_target, ref_best_scores)
    )
    five_safe = sum(
        is_target and score >= five_t
        for is_target, score in zip(ref_best_is_target, ref_best_scores)
    )
    return {
        "reference_count": refs,
        "control_count": controls,
        "target_top1": top1 / refs if refs else 0.0,
        "target_top3": top3 / refs if refs else 0.0,
        "safe_target_selected": safe / refs if refs else 0.0,
        "selected_wrong_active": wrong / refs if refs else 0.0,
        "selected_stale": stale / refs if refs else 0.0,
        "zero_fpr_recovery": sum(score >= zero_t for score in ref_scores),
        "five_pct_fpr_recovery": sum(score >= five_t for score in ref_scores),
        "zero_fpr_recovery_rate": sum(score >= zero_t for score in ref_scores) / refs if refs else 0.0,
        "five_pct_fpr_recovery_rate": sum(score >= five_t for score in ref_scores) / refs if refs else 0.0,
        "zero_fpr_safe_recovery": zero_safe,
        "five_pct_safe_recovery": five_safe,
        "zero_fpr_safe_recovery_rate": zero_safe / refs if refs else 0.0,
        "five_pct_safe_recovery_rate": five_safe / refs if refs else 0.0,
        "no_anchor_abstain_zero_fpr": sum(score < zero_t for score in control_scores) / controls if controls else 0.0,
        "no_anchor_abstain_five_pct": sum(score < five_t for score in control_scores) / controls if controls else 0.0,
        "relation_top1": family_top1.get("owner_object_relation", 0) / family_counts.get("owner_object_relation", 1),
        "media_top1": (
            family_top1.get("media_object_reference", 0)
            + family_top1.get("cross_modal_person_to_media", 0)
        )
        / max(
            1,
            family_counts.get("media_object_reference", 0)
            + family_counts.get("cross_modal_person_to_media", 0),
        ),
    }


def run_grid(cases: list[ChainCase], seed: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    policies = ["recency_only", "entity_only", "entity_object", "entity_relation"]
    strengths = [0.0, 0.20, 0.40, 0.60, 0.80]
    false_rates = [0.0, 0.005, 0.01, 0.02]
    for policy in policies:
        for strength in strengths:
            for false_rate in false_rates:
                relation_false = false_rate if policy == "entity_relation" else 0.0
                scores = [
                    scores_for_case(
                        case,
                        policy,
                        strength,
                        recall=1.0,
                        false_rate=false_rate,
                        relation_recall=1.0,
                        relation_false_rate=relation_false,
                        seed=seed,
                    )
                    for case in cases
                ]
                rows.append(
                    {
                        "policy": policy,
                        "signal_strength": strength,
                        "track_recall": 1.0,
                        "false_track_rate": false_rate,
                        "relation_recall": 1.0,
                        "relation_false_rate": relation_false,
                        **summarize(cases, scores),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output-dir", default="build/multimodal_anchor_chain_headroom")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = make_cases(args.episodes)
    rows = run_grid(cases, args.seed)
    csv_path = output_dir / "multimodal_anchor_chain_headroom.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_by_policy = {}
    for policy in sorted({row["policy"] for row in rows}):
        subset = [row for row in rows if row["policy"] == policy]
        best_by_policy[policy] = max(
            subset,
            key=lambda row: (
                row["zero_fpr_recovery"],
                row["five_pct_fpr_recovery"],
                row["safe_target_selected"],
                -row["selected_wrong_active"],
                -row["false_track_rate"],
            ),
        )
    precision_rows = [
        row
        for row in rows
        if row["policy"] == "entity_relation"
        and row["signal_strength"] == 0.6
        and row["track_recall"] == 1.0
    ]
    summary = {
        "mode": "multimodal_anchor_chain_headroom",
        "runtime_effect": "none",
        "production_retrieval_changed": False,
        "labels_used_for_runtime_features": False,
        "episodes": args.episodes,
        "case_count": len(cases),
        "families": sorted({case.family for case in cases}),
        "best_by_policy": best_by_policy,
        "entity_relation_false_track_sensitivity": precision_rows,
        "interpretation": (
            "Synthetic headroom for modality-agnostic chain anchoring. Relation "
            "candidates model cases like 'he had it', where the engine needs "
            "person, object/media, and ownership/event edges, not just a single "
            "entity key."
        ),
    }
    json_path = output_dir / "multimodal_anchor_chain_headroom.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    case_path = output_dir / "multimodal_anchor_chain_cases.csv"
    with case_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "family",
                "modality",
                "current_text",
                "is_reference",
                "target_kind",
                "required_tracks",
                "candidate_count",
            ]
        )
        for case in cases:
            writer.writerow(
                [
                    case.case_id,
                    case.family,
                    case.modality,
                    case.current_text,
                    int(case.is_reference),
                    case.target_kind,
                    " ".join(case.required_tracks),
                    len(case.candidates),
                ]
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
