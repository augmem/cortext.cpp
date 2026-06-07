#!/usr/bin/env python3
"""Evaluate the anchor signal contract on a synthetic multimodal chain surface.

This is a benchmark-only contract verifier. It does not change production
retrieval and does not claim model evidence. It provides an executable gate for
future entity/object/media/relation proposal artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ExpectedRelation:
    relation_type: str
    source: str
    target: str


@dataclass(frozen=True)
class SignalCase:
    case_id: str
    family: str
    modality: str
    text: str
    expected_entities: tuple[str, ...]
    expected_objects: tuple[str, ...]
    expected_relations: tuple[ExpectedRelation, ...]


def stable_unit(*parts: object) -> float:
    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def make_cases(episodes: int) -> list[SignalCase]:
    cases: list[SignalCase] = []
    for i in range(episodes):
        prefix = f"ep{i:04d}"
        jared = f"{prefix}:person:jared"
        alex = f"{prefix}:person:alex"
        dog = f"{prefix}:object:jared_dog"
        image = f"{prefix}:media:dog_image"
        work_doc = f"{prefix}:object:work_doc"
        cases.extend(
            [
                SignalCase(
                    case_id=f"{prefix}:he_phone",
                    family="person_reference",
                    modality="text",
                    text="Great, what did he say?",
                    expected_entities=(jared,),
                    expected_objects=(),
                    expected_relations=(),
                ),
                SignalCase(
                    case_id=f"{prefix}:image_handoff",
                    family="cross_modal_media_handoff",
                    modality="image",
                    text="<image> dog photo sent by Jared",
                    expected_entities=(jared,),
                    expected_objects=(dog, image),
                    expected_relations=(
                        ExpectedRelation("sender", jared, image),
                        ExpectedRelation("depicts", image, dog),
                    ),
                ),
                SignalCase(
                    case_id=f"{prefix}:send_it",
                    family="object_reference",
                    modality="text",
                    text="Can you send it to me?",
                    expected_entities=(),
                    expected_objects=(image,),
                    expected_relations=(),
                ),
                SignalCase(
                    case_id=f"{prefix}:he_had_it",
                    family="person_object_relation",
                    modality="text",
                    text="How long has he had it?",
                    expected_entities=(jared,),
                    expected_objects=(dog, image),
                    expected_relations=(ExpectedRelation("owner", jared, dog),),
                ),
                SignalCase(
                    case_id=f"{prefix}:wrong_active_doc",
                    family="wrong_active_reference",
                    modality="text",
                    text="He sent the work document.",
                    expected_entities=(alex,),
                    expected_objects=(work_doc,),
                    expected_relations=(ExpectedRelation("sender", alex, work_doc),),
                ),
                SignalCase(
                    case_id=f"{prefix}:topic_shift",
                    family="no_anchor_topic_shift",
                    modality="text",
                    text="The dishwasher broke this morning.",
                    expected_entities=(),
                    expected_objects=(),
                    expected_relations=(),
                ),
            ]
        )
    return cases


def entity_proposal(track: str) -> dict[str, Any]:
    return {
        "proposal_id": track,
        "kind": "person" if ":person:" in track else "object",
        "key": [],
        "confidence": 1.0,
        "modality_evidence": ["text"],
        "eval_track": track,
    }


def object_proposal(track: str) -> dict[str, Any]:
    kind = "image" if ":media:" in track else "physical_object"
    return {
        "proposal_id": track,
        "kind": kind,
        "key": [],
        "confidence": 1.0,
        "modality_evidence": ["image" if kind == "image" else "text"],
        "eval_track": track,
    }


def relation_proposal(rel: ExpectedRelation) -> dict[str, Any]:
    return {
        "relation_type": rel.relation_type,
        "source_proposal_id": rel.source,
        "target_proposal_id": rel.target,
        "confidence": 1.0,
        "evidence_key": [],
        "modality_evidence": ["text"],
        "eval_relation": {
            "relation_type": rel.relation_type,
            "source": rel.source,
            "target": rel.target,
        },
    }


def false_tracks_for(case: SignalCase) -> tuple[str, str, ExpectedRelation]:
    prefix = case.case_id.split(":", 1)[0]
    false_person = f"{prefix}:person:false_active"
    false_object = f"{prefix}:object:false_active"
    false_relation = ExpectedRelation("owner", false_person, false_object)
    return false_person, false_object, false_relation


def proposal_for_case(
    case: SignalCase, mode: str, false_rate: float, seed: int
) -> dict[str, Any]:
    include_entity = mode in {
        "entity_only",
        "entity_object",
        "entity_object_relation",
        "entity_object_relation_noisy",
    }
    include_object = mode in {
        "entity_object",
        "entity_object_relation",
        "entity_object_relation_noisy",
    }
    include_relation = mode in {
        "entity_object_relation",
        "entity_object_relation_noisy",
    }
    false_person, false_object, false_relation = false_tracks_for(case)
    entity_items = [entity_proposal(track) for track in case.expected_entities] if include_entity else []
    object_items = [object_proposal(track) for track in case.expected_objects] if include_object else []
    relation_items = [relation_proposal(rel) for rel in case.expected_relations] if include_relation else []

    if mode.endswith("_noisy"):
        if stable_unit(seed, "entity_false", case.case_id) < false_rate:
            entity_items.append(entity_proposal(false_person))
        if stable_unit(seed, "object_false", case.case_id) < false_rate:
            object_items.append(object_proposal(false_object))
        if stable_unit(seed, "relation_false", case.case_id) < false_rate:
            relation_items.append(relation_proposal(false_relation))

    return {
        "case_id": case.case_id,
        "mode": mode,
        "semantic_key": [],
        "entity_key": [],
        "object_key": [],
        "event_key": [],
        "salience": 0.0,
        "boundary_score": 0.0,
        "novelty": 0.0,
        "entity_proposals": entity_items,
        "object_media_proposals": object_items,
        "relation_proposals": relation_items,
    }


def relation_tuple(value: dict[str, Any]) -> tuple[str, str, str]:
    if "eval_relation" in value:
        rel = value["eval_relation"]
        return (rel["relation_type"], rel["source"], rel["target"])
    return (
        value.get("relation_type", ""),
        value.get("source_proposal_id", ""),
        value.get("target_proposal_id", ""),
    )


def load_cases_jsonl(path: Path) -> list[SignalCase]:
    cases: list[SignalCase] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            relations = tuple(
                ExpectedRelation(
                    relation_type=rel["relation_type"],
                    source=rel["source"],
                    target=rel["target"],
                )
                for rel in row.get("expected_relations", [])
            )
            cases.append(
                SignalCase(
                    case_id=row["case_id"],
                    family=row.get("family", "external"),
                    modality=row.get("modality", "unknown"),
                    text=row.get("text", ""),
                    expected_entities=tuple(row.get("expected_entities", [])),
                    expected_objects=tuple(row.get("expected_objects", [])),
                    expected_relations=relations,
                )
            )
    return cases


def load_proposals(path: Path) -> dict[str, dict[str, Any]]:
    proposals: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            proposals[row["case_id"]] = row
    return proposals


def evaluate(cases: list[SignalCase], proposals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "entity_expected": 0,
        "entity_tp": 0,
        "entity_fp": 0,
        "object_expected": 0,
        "object_tp": 0,
        "object_fp": 0,
        "relation_expected": 0,
        "relation_tp": 0,
        "relation_fp": 0,
    }
    family_rows: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []
    false_audit: list[dict[str, Any]] = []
    for case in cases:
        row = proposals.get(case.case_id, {"case_id": case.case_id})
        expected_entities = set(case.expected_entities)
        expected_objects = set(case.expected_objects)
        expected_relations = {
            (rel.relation_type, rel.source, rel.target)
            for rel in case.expected_relations
        }
        got_entities = {
            item.get("eval_track") or item.get("proposal_id", "")
            for item in row.get("entity_proposals", [])
        }
        got_objects = {
            item.get("eval_track") or item.get("proposal_id", "")
            for item in row.get("object_media_proposals", [])
        }
        got_relations = {
            relation_tuple(item) for item in row.get("relation_proposals", [])
        }
        counts["entity_expected"] += len(expected_entities)
        counts["entity_tp"] += len(expected_entities & got_entities)
        counts["entity_fp"] += len(got_entities - expected_entities)
        counts["object_expected"] += len(expected_objects)
        counts["object_tp"] += len(expected_objects & got_objects)
        counts["object_fp"] += len(got_objects - expected_objects)
        counts["relation_expected"] += len(expected_relations)
        counts["relation_tp"] += len(expected_relations & got_relations)
        counts["relation_fp"] += len(got_relations - expected_relations)

        fam = family_rows.setdefault(
            case.family,
            {"cases": 0, "complete": 0, "entity_ok": 0, "object_ok": 0, "relation_ok": 0},
        )
        fam["cases"] += 1
        entity_ok = expected_entities.issubset(got_entities)
        object_ok = expected_objects.issubset(got_objects)
        relation_ok = expected_relations.issubset(got_relations)
        fam["entity_ok"] += int(entity_ok)
        fam["object_ok"] += int(object_ok)
        fam["relation_ok"] += int(relation_ok)
        complete = entity_ok and object_ok and relation_ok
        fam["complete"] += int(complete)
        if not complete and len(failures) < 25:
            failures.append(
                {
                    "case_id": case.case_id,
                    "family": case.family,
                    "missing_entities": sorted(expected_entities - got_entities),
                    "missing_objects": sorted(expected_objects - got_objects),
                    "missing_relations": sorted(
                        ["|".join(rel) for rel in expected_relations - got_relations]
                    ),
                }
            )
        for kind, items in (
            ("entity", got_entities - expected_entities),
            ("object", got_objects - expected_objects),
            ("relation", {"|".join(rel) for rel in got_relations - expected_relations}),
        ):
            for item in sorted(items):
                false_audit.append({"case_id": case.case_id, "family": case.family, "kind": kind, "value": item})

    case_count = len(cases)
    entity_recall = counts["entity_tp"] / counts["entity_expected"] if counts["entity_expected"] else 1.0
    object_recall = counts["object_tp"] / counts["object_expected"] if counts["object_expected"] else 1.0
    relation_recall = counts["relation_tp"] / counts["relation_expected"] if counts["relation_expected"] else 1.0
    entity_false_rate = counts["entity_fp"] / case_count if case_count else 0.0
    object_false_rate = counts["object_fp"] / case_count if case_count else 0.0
    relation_false_rate = counts["relation_fp"] / case_count if case_count else 0.0
    required_gates = {
        "entity_false_rate_pass": entity_false_rate < 0.005,
        "object_false_rate_pass": object_false_rate < 0.005,
        "relation_false_rate_pass": relation_false_rate < 0.005,
        "entity_recall_pass": entity_recall >= 0.95,
        "object_recall_pass": object_recall >= 0.95,
        "relation_recall_pass": relation_recall >= 0.90,
    }
    target_gates = {
        "entity_false_rate_target_pass": entity_false_rate <= 0.001,
        "object_false_rate_target_pass": object_false_rate <= 0.001,
        "relation_false_rate_target_pass": relation_false_rate <= 0.001,
        "entity_recall_target_pass": entity_recall >= 0.95,
        "object_recall_target_pass": object_recall >= 0.95,
        "relation_recall_target_pass": relation_recall >= 0.90,
    }
    return {
        "case_count": case_count,
        "counts": counts,
        "entity_recall": entity_recall,
        "object_recall": object_recall,
        "relation_recall": relation_recall,
        "entity_false_rate_per_case": entity_false_rate,
        "object_false_rate_per_case": object_false_rate,
        "relation_false_rate_per_case": relation_false_rate,
        "family_metrics": {
            family: {
                key: (value / metrics["cases"] if key != "cases" else value)
                for key, value in metrics.items()
            }
            for family, metrics in sorted(family_rows.items())
        },
        "required_gates": required_gates,
        "target_gates": target_gates,
        "passed_required_gates": all(required_gates.values()),
        "passed_target_gates": all(target_gates.values()),
        "passed": all(required_gates.values()),
        "failures": failures,
        "false_audit": false_audit,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output-dir", default="build/anchor_signal_contract_eval")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--cases-jsonl", default=None)
    parser.add_argument("--proposals-jsonl", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases_jsonl(Path(args.cases_jsonl)) if args.cases_jsonl else make_cases(args.episodes)
    case_rows = [
        {
            "case_id": case.case_id,
            "family": case.family,
            "modality": case.modality,
            "text": case.text,
            "expected_entities": list(case.expected_entities),
            "expected_objects": list(case.expected_objects),
            "expected_relations": [
                {
                    "relation_type": rel.relation_type,
                    "source": rel.source,
                    "target": rel.target,
                }
                for rel in case.expected_relations
            ],
        }
        for case in cases
    ]
    write_jsonl(output_dir / "anchor_signal_contract_eval_cases.jsonl", case_rows)

    modes = [
        ("empty", 0.0),
        ("entity_only", 0.0),
        ("entity_object", 0.0),
        ("entity_object_relation", 0.0),
        ("entity_object_relation_noisy", 0.005),
        ("entity_object_relation_noisy", 0.01),
    ]
    results: dict[str, Any] = {
        "mode": "anchor_signal_contract_eval",
        "runtime_effect": "none",
        "production_retrieval_changed": False,
        "interpretation": (
            "Built-in proposal modes are headroom/scaffold checks, not model "
            "evidence. External proposal files can be evaluated with "
            "--cases-jsonl and --proposals-jsonl."
        ),
        "gate_thresholds": {
            "required_false_proposal_rate_lt": 0.005,
            "target_false_proposal_rate_lte": 0.001,
            "entity_recall_gte": 0.95,
            "object_recall_gte": 0.95,
            "relation_recall_gte": 0.90,
        },
        "episodes": args.episodes,
        "case_count": len(cases),
        "evaluations": {},
    }
    false_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for mode, false_rate in modes:
        name = mode if false_rate == 0.0 else f"{mode}_{false_rate:g}"
        rows = [proposal_for_case(case, mode, false_rate, args.seed) for case in cases]
        proposal_path = output_dir / f"anchor_signal_proposals_{name}.jsonl"
        write_jsonl(proposal_path, rows)
        eval_result = evaluate(cases, {row["case_id"]: row for row in rows})
        eval_result["proposal_path"] = str(proposal_path)
        eval_result["false_rate"] = false_rate
        results["evaluations"][name] = {
            key: value
            for key, value in eval_result.items()
            if key not in {"false_audit", "failures"}
        }
        for row in eval_result["false_audit"]:
            false_rows.append({"evaluation": name, **row})
        for row in eval_result["failures"]:
            failure_rows.append({"evaluation": name, **row})

    if args.proposals_jsonl:
        proposal_path = Path(args.proposals_jsonl)
        eval_result = evaluate(cases, load_proposals(proposal_path))
        eval_result["proposal_path"] = str(proposal_path)
        results["evaluations"]["external"] = {
            key: value
            for key, value in eval_result.items()
            if key not in {"false_audit", "failures"}
        }
        for row in eval_result["false_audit"]:
            false_rows.append({"evaluation": "external", **row})
        for row in eval_result["failures"]:
            failure_rows.append({"evaluation": "external", **row})

    with (output_dir / "anchor_signal_contract_results.json").open("w") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    with (output_dir / "anchor_signal_contract_summary.csv").open("w", newline="") as handle:
        fieldnames = [
            "evaluation",
            "passed",
            "passed_required_gates",
            "passed_target_gates",
            "entity_recall",
            "object_recall",
            "relation_recall",
            "entity_false_rate_per_case",
            "object_false_rate_per_case",
            "relation_false_rate_per_case",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, row in results["evaluations"].items():
            writer.writerow({field: row.get(field) if field != "evaluation" else name for field in fieldnames})
    with (output_dir / "anchor_signal_false_proposal_audit.csv").open("w", newline="") as handle:
        fieldnames = ["evaluation", "case_id", "family", "kind", "value"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(false_rows)
    with (output_dir / "anchor_signal_failure_examples.csv").open("w", newline="") as handle:
        fieldnames = [
            "evaluation",
            "case_id",
            "family",
            "missing_entities",
            "missing_objects",
            "missing_relations",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failure_rows)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
