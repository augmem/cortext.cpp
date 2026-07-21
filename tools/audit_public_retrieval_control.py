#!/usr/bin/env python3
"""Validate and sanitize the exact public GraphRetrieve control artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


class DuplicateKeyError(ValueError):
    pass


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=reject_duplicate_keys)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranked_ids(
    value: Any, *, query_ordinal: int, field: str, limit: int
) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"query {query_ordinal} has empty {field}")
    result: list[int] = []
    previous_score = math.inf
    for expected_rank, row in enumerate(value[:limit]):
        if not isinstance(row, dict):
            raise ValueError(f"query {query_ordinal} {field} row is not an object")
        memory_id = row.get("memory_id")
        rank = row.get("rank")
        score = row.get("score")
        if not isinstance(memory_id, int) or memory_id <= 0:
            raise ValueError(f"query {query_ordinal} {field} memory id is invalid")
        if rank != expected_rank:
            raise ValueError(f"query {query_ordinal} {field} ranks are not contiguous")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(f"query {query_ordinal} {field} score is invalid")
        if float(score) > previous_score + 1.0e-12:
            raise ValueError(f"query {query_ordinal} {field} scores are not ranked")
        previous_score = float(score)
        result.append(memory_id)
    if len(result) != len(set(result)):
        raise ValueError(f"query {query_ordinal} {field} repeats a memory id")
    return result


def audit(profile: dict[str, Any], profile_sha256: str) -> dict[str, Any]:
    control = profile.get("public_retrieval_control")
    if not isinstance(control, dict):
        raise ValueError("profile lacks public_retrieval_control object")
    if control.get("schema") != "cortext_current_public_retrieval_control_v1":
        raise ValueError("unexpected public retrieval control schema")
    query_count = control.get("query_count")
    result_k = control.get("result_k")
    queries = control.get("queries")
    if not isinstance(query_count, int) or query_count <= 0:
        raise ValueError("query_count must be positive")
    if not isinstance(result_k, int) or result_k <= 0:
        raise ValueError("result_k must be positive")
    if not isinstance(queries, list) or len(queries) != query_count:
        raise ValueError("query population does not match query_count")
    semantics = control.get("eligibility_semantics")
    required_semantics = {
        "timestamp_exclusion",
        "supersession_exclusion",
        "current_surface_selection",
        "family_collapse",
        "deterministic_tie_order",
    }
    if not isinstance(semantics, dict) or any(
        semantics.get(key) is not True for key in required_semantics
    ):
        raise ValueError("public retrieval eligibility semantics are incomplete")
    if control.get("identity_kind") != "memory-id":
        raise ValueError("public retrieval control is not keyed by memory id")
    if control.get("production_cutover") is not False:
        raise ValueError("benchmark control must not claim production cutover")
    coverage_contract = control.get("coverage_contract")
    if not isinstance(coverage_contract, dict):
        raise ValueError("public retrieval coverage contract is missing")
    minimum_source_ids = coverage_contract.get("minimum_opaque_source_ids")
    required_modalities_raw = coverage_contract.get("required_modalities")
    modality_agnostic_proof = coverage_contract.get("modality_agnostic_proof")
    if not isinstance(minimum_source_ids, int) or minimum_source_ids <= 0:
        raise ValueError("minimum opaque source count is invalid")
    if (
        not isinstance(required_modalities_raw, list)
        or not required_modalities_raw
        or any(
            not isinstance(modality, str) or not modality
            for modality in required_modalities_raw
        )
    ):
        raise ValueError("required modality coverage is invalid")
    required_modalities = set(required_modalities_raw)
    if not isinstance(modality_agnostic_proof, str) or not modality_agnostic_proof:
        raise ValueError("modality-agnostic proof locator is missing")

    first_memory_events: dict[int, tuple[int, int]] = {}
    surface_mutations = control.get("surface_mutations", [])
    if not isinstance(surface_mutations, list):
        raise ValueError("public retrieval surface mutations are invalid")
    for mutation in surface_mutations:
        if not isinstance(mutation, dict) or mutation.get("action") != "upsert":
            continue
        memory_id = mutation.get("memory_id")
        event_index = mutation.get("event_index")
        event_timestamp = mutation.get("event_timestamp")
        if (
            not isinstance(memory_id, int)
            or memory_id <= 0
            or not isinstance(event_index, int)
            or event_index < 0
            or not isinstance(event_timestamp, int)
            or event_timestamp < 0
        ):
            raise ValueError("public retrieval surface upsert identity is invalid")
        prior = first_memory_events.get(memory_id)
        if prior is None or (event_index, event_timestamp) < prior:
            first_memory_events[memory_id] = (event_index, event_timestamp)

    prior_event = -1
    prior_timestamp = -1
    source_digests: set[str] = set()
    modalities: set[str] = set()
    exact_hits = 0
    exact_expected = 0
    top1_hits = 0
    candidate_queries = 0
    semantic_coverage_total = 0.0
    deterministic_ties = True
    query_fingerprints: list[dict[str, Any]] = []
    top1_miss_fingerprints: list[dict[str, Any]] = []
    for ordinal, query in enumerate(queries):
        if not isinstance(query, dict) or query.get("query_ordinal") != ordinal:
            raise ValueError("query ordinals are not contiguous")
        event_index = query.get("event_index")
        timestamp = query.get("query_timestamp")
        if not isinstance(event_index, int) or event_index <= prior_event:
            raise ValueError("query event indices are not strictly increasing")
        if not isinstance(timestamp, int) or timestamp <= prior_timestamp:
            raise ValueError("query timestamps are not strictly increasing")
        prior_event = event_index
        prior_timestamp = timestamp
        source_digest = query.get("source_id_blake3")
        modality = query.get("modality")
        if not isinstance(source_digest, str) or len(source_digest) != 64:
            raise ValueError("query source digest is invalid")
        if not isinstance(modality, str) or not modality:
            raise ValueError("query modality is invalid")
        source_digests.add(source_digest)
        modalities.add(modality)
        query_embedding = query.get("query_embedding")
        if (
            not isinstance(query_embedding, list)
            or len(query_embedding) != 256
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in query_embedding
            )
        ):
            raise ValueError("query embedding is not a finite 256-vector")
        control_ids = ranked_ids(
            query.get("control_seed_ranked"),
            query_ordinal=ordinal,
            field="control_seed_ranked",
            limit=result_k,
        )
        candidate_ids: list[int] | None = None
        top1_exact_match: bool | None = None
        if query.get("candidate_available") is True:
            candidate_ids = ranked_ids(
                query.get("candidate_seed_ranked"),
                query_ordinal=ordinal,
                field="candidate_seed_ranked",
                limit=result_k,
            )
            candidate_queries += 1
            exact_hits += len(set(control_ids) & set(candidate_ids))
            exact_expected += len(control_ids)
            top1_exact_match = control_ids[0] == candidate_ids[0]
            top1_hits += int(top1_exact_match)
            coverage = query.get("semantic_coverage")
            if not isinstance(coverage, (int, float)) or not math.isfinite(
                float(coverage)
            ):
                raise ValueError("candidate semantic coverage is invalid")
            semantic_coverage_total += float(coverage)
            deterministic_ties = deterministic_ties and (
                query.get("deterministic_tie_order") is True
            )
        first_memory_event = first_memory_events.get(control_ids[0])
        control_top1_age_events = (
            event_index - first_memory_event[0]
            if first_memory_event is not None
            else None
        )
        control_top1_age_ms = (
            timestamp - first_memory_event[1]
            if first_memory_event is not None
            else None
        )
        fingerprint = {
                "query_ordinal": ordinal,
                "event_index": event_index,
                "query_timestamp": timestamp,
                "source_id_blake3": source_digest,
                "modality": modality,
                "query_embedding_sha256": sha256(query_embedding),
                "control_seed_rank_sha256": sha256(control_ids),
                "control_graph_rank_sha256": sha256(
                    query.get("control_graph_ranked")
                ),
                "control_public_rank_sha256": sha256(
                    query.get("control_public_ranked")
                ),
                "candidate_seed_rank_sha256": (
                    sha256(candidate_ids) if candidate_ids is not None else None
                ),
                "top1_exact_match": top1_exact_match,
                "control_top1_age_events": control_top1_age_events,
                "control_top1_age_ms": control_top1_age_ms,
            }
        query_fingerprints.append(fingerprint)
        if top1_exact_match is False:
            top1_miss_fingerprints.append(
                {
                    "query_ordinal": ordinal,
                    "query_progress_quartile": min(3, ordinal * 4 // query_count),
                    "source_id_blake3": source_digest,
                    "modality": modality,
                    "control_top1_age_events": control_top1_age_events,
                    "control_top1_age_ms": control_top1_age_ms,
                }
            )

    candidate_complete = candidate_queries == query_count
    recall = (
        exact_hits / exact_expected
        if candidate_complete and exact_expected > 0
        else None
    )
    top1 = top1_hits / query_count if candidate_complete else None
    top1_miss_count = query_count - top1_hits if candidate_complete else None
    semantic = (
        semantic_coverage_total / query_count if candidate_complete else None
    )
    quality_passed = bool(
        candidate_complete
        and recall is not None
        and recall >= 0.998
        and top1 is not None
        and top1 >= 0.998
        and top1_miss_count is not None
        and top1_miss_count <= 1
        and semantic is not None
        and semantic >= 0.95
        and deterministic_ties
        and len(source_digests) >= minimum_source_ids
        and required_modalities.issubset(modalities)
    )
    return {
        "schema": "cortext_current_public_retrieval_control_audit_v1",
        "status": (
            "candidate-quality-passed"
            if quality_passed
            else (
                "exact-public-control-validated-candidate-pending"
                if not candidate_complete
                else "candidate-quality-failed"
            )
        ),
        "passed": True,
        "control_population_passed": True,
        "candidate_complete": candidate_complete,
        "candidate_quality_passed": quality_passed,
        "query_count": query_count,
        "result_k": result_k,
        "retrieval_active_event_count": control.get(
            "retrieval_active_event_count"
        ),
        "event_index_first": queries[0]["event_index"],
        "event_index_last": queries[-1]["event_index"],
        "source_digest_count": len(source_digests),
        "modalities": sorted(modalities),
        "coverage_contract": {
            "minimum_opaque_source_ids": minimum_source_ids,
            "required_modalities": sorted(required_modalities),
            "modality_agnostic_proof": modality_agnostic_proof,
        },
        "exact_identity_recall_at_k": recall,
        "exact_top1": top1,
        "exact_top1_hits": top1_hits,
        "exact_top1_miss_count": top1_miss_count,
        "exact_top1_minimum": max (0.998, (query_count - 1) / query_count),
        "semantic_coverage": semantic,
        "deterministic_tie_order": deterministic_ties if candidate_complete else None,
        "private_profile_sha256": profile_sha256,
        "query_fingerprints": query_fingerprints,
        "top1_miss_fingerprints": top1_miss_fingerprints,
        "production_cutover": False,
        "unresolved_limits": (
            []
            if quality_passed
            else [
                "bounded candidate ranks or the declared source/modality coverage contract remains pending"
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(load_json(args.profile), file_sha256(args.profile))
    except (DuplicateKeyError, json.JSONDecodeError, OSError, ValueError) as error:
        result = {
            "schema": "cortext_current_public_retrieval_control_audit_v1",
            "status": "invalid",
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
            "production_cutover": False,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    sys.exit(main())
