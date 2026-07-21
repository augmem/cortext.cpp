#!/usr/bin/env python3
"""Build the content-addressed paper traceability inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_paper_traceability import load_json, merge_state, state_inventory


BASE_COMMIT = "1bf3f1c5f6470bb8c4d458a0389c19c5123f9b81"
SECTION = {
    "path": "docs/paper/sections/11_optimization.qmd",
    "heading": "Experiment and algorithm traceability inventory",
}
VARIANTS = [
    "reciprocal-route-only",
    "three-sparse-graphs",
    "construction-budget-4096",
    "construction-budget-8192-reciprocal-64",
    "two-graphs-construction-16384-retrieval-1280-reciprocal-64",
]
ALGORITHM_GROUPS = [
    {
        "id": "consolidation-envelope-and-detector",
        "paths": [
            "src/operations/accumulator_reset.cpp",
            "src/operations/consolidation.cpp",
            "src/operations/consolidation_shallow.cpp",
            "src/operations/consolidation_throughput_state_internal.hpp",
        ],
    },
    {
        "id": "shared-retention-and-rollback",
        "paths": [
            "src/operations/accumulator.cpp",
            "src/operations/signal_record_rollback_internal.hpp",
            "src/operations/write_gate.cpp",
            "src/operations/working_memory.cpp",
            "src/signal_processor.cpp",
        ],
    },
    {
        "id": "lazy-rif-active-epoch",
        "paths": [
            "src/operations/competition.cpp",
            "src/operations/execution_cache_sidecar_internal.hpp",
            "src/operations/memory_strength.cpp",
            "src/operations/rif_active_epoch_cache_internal.hpp",
            "src/operations/rif_state_internal.hpp",
            "src/store.cpp",
            "src/store/schema.cpp",
        ],
    },
    {
        "id": "bounded-graph-retrieval",
        "paths": [
            "src/operations/sparse_retrieval_knobs_internal.hpp",
            "src/operations/bounded_activation_shadow_internal.hpp",
            "src/operations/family_embedding_features_internal.hpp",
            "src/operations/graph_retrieval.cpp",
            "src/operations/historical_surface_search_cache_internal.hpp",
            "src/operations/retrieval_trace_state.cpp",
            "src/operations/retrieval_trace_state.hpp",
            "src/operations/sparse_retrieval_route_internal.cpp",
            "src/operations/sparse_retrieval_route_internal.hpp",
            "src/operations/sparse_retrieval_route_sqlite_internal.cpp",
            "src/operations/sparse_retrieval_route_sqlite_internal.hpp",
        ],
    },
    {
        "id": "emotional-propagation-cache",
        "paths": [
            "src/experimental_env.hpp",
            "src/operations/emotion.cpp",
            "src/operations/emotion_cascade.cpp",
            "src/operations/emotional_metadata_cache_internal.hpp",
        ],
    },
    {
        "id": "memory-storage-and-graph-maintenance",
        "paths": [
            "src/operations/active_signal_embedding_ring_internal.hpp",
            "src/operations/association_fanout_cache_internal.hpp",
            "src/operations/detect_memory_usage.cpp",
            "src/operations/eviction_policy.hpp",
            "src/operations/memory_storage.cpp",
            "src/operations/predictive.cpp",
            "src/operations/reconsolidation.cpp",
            "src/operations/stability.cpp",
        ],
    },
    {
        "id": "runtime-composition",
        "paths": [
            "src/cortext.cpp",
            "src/cortext_pipeline_internal.hpp",
        ],
    },
    {
        "id": "bounded-work-instrumentation",
        "paths": [
            "src/operations/consolidation_epoch_profile_internal.hpp",
            "src/store/commit_profile_internal.hpp",
            "src/store/mutation_audit_internal.hpp",
        ],
    },
    {
        "id": "experiment-harness-and-audit",
        "paths": [
            "examples/benchmark/chat_replay_live_run.cpp",
            "tools/audit_public_retrieval_control.py",
            "tools/audit_storage_cost_profile.py",
            "tools/benchmark_row_addressed_route_sqlite.py",
            "tools/build_sqlite_hnsw_knob_ablation.py",
            "tools/build_signal_centroid_ring_ablation.py",
            "tools/extract_claude_session_packets.py",
            "tools/extract_consolidation_schedule.py",
            "tools/build_recenter_activation_overlap.py",
            "tools/build_write_gate_incidence_attribution.py",
        ],
    },
]


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def first_decision(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("decision", "status", "verdict", "state"):
            candidate = value.get(key)
            if isinstance(candidate, (str, bool, int, float)):
                return str(candidate)
    return "recorded-measurement"


def state_scope(key: str) -> str:
    if key == "rif_epoch_consolidation_reset_candidate":
        return (
            "15,695-packet Natural replay at F=0.45, S=T=0.5; 27 "
            "structural F/S/T points; nine production-shaped reset points; "
            "B=round(64+64F+32S+32T), with neutral B=128 and 129 used only "
            "as the logical B+1 population boundary; text/audio/image labels "
            "and two opaque source ids"
        )
    if key == "hnsw_fixed_6c_experiment":
        return (
            "rejected private fixed-6C envelope; 27 structural F/S/T points "
            "and 931 assertions; one additional 15,695-packet Natural "
            "production-default pre-screen at F=0.45, S=T=0.5 with 512 public "
            "top-16 queries; exact nine-point corpus matrix not started after "
            "fail-fast rejection"
        )
    if key == "signal_centroid_ring_cutover":
        return (
            "migration 30; 15,695-packet Natural and 2,016-message Durable "
            "replays at F=0.45, S=T=0.5; 27 structural F/S/T points; nine "
            "2,000-packet same-knob runs with 512 public queries and four "
            "opaque source ids per run; separate text/audio/image label proof"
        )
    if key == "sqlite_hnsw_production_cutover":
        return (
            "production-default 8C-to-9C SQLite HNSW route; 15,695-packet "
            "Natural replay, 2,016-message Durable replay, 30,380-packet "
            "maturity extension, 27 structural F/S/T points, and nine "
            "4,000-packet quality points with 512 controls each"
        )
    if key == "emotional_cascade_fixed_work_candidate":
        return (
            "rejected knob-derived priority-prefix candidate; 27 F/S/T "
            "structural points; exact B and logical B+1 shared-member seams; "
            "15,695-packet fixed-time Natural midpoint against a same-binary "
            "control and identical 118-event consolidation schedule; 512 "
            "public top-16 queries over four opaque sources; nine-point "
            "production matrix intentionally not run after midpoint fail-fast"
        )
    if key == "bounded_route_hybrid_observation":
        return (
            "34,456 normalized 256-dimensional embeddings; 512 deterministic "
            "top-16 queries; four opaque sources; text, audio, and image "
            "post-encoding labels; two graphs; 16,384 construction and 1,280 "
            "retrieval comparisons per graph; 64 reciprocal proposals"
        )
    if key == "bounded_sparse_routing_observation":
        return (
            "35,496 normalized 256-dimensional embeddings; 512 age-stratified "
            "top-16 queries; F=S=T=0.5; observation-only private switch"
        )
    if key == "packed_route_sqlite_representation_experiment":
        return (
            "34,456 normalized 256-dimensional embeddings; four SQLite blobs; "
            "1,024 anchors; 512 deterministic top-16 route queries; separate "
            "1,024-embedding fresh-prefix restart"
        )
    if key == "row_addressed_route_sqlite_experiment":
        return (
            "200-event fresh and 34,256-row copied-late SQLite seals; 3,544 "
            "reciprocal route-row updates; 1,024 anchors; 512 fixed-route queries"
        )
    if key == "integrated_hnsw_sparse_route_experiment":
        return (
            "2,500 deterministic Natural packets; four opaque source ids; "
            "text, audio, and image post-encoding labels; 128 public top-16 "
            "queries; two HNSW graphs; 512-candidate sealed-plus-delta route"
        )
    if key == "retrieval_quality_control_correction":
        return (
            "correction from 34,456 complete embedding ids to 3,172 current "
            "long-term memory identities; 512 top-16 queries; ten 128-memory-node "
            "incremental feasibility epochs; 25/50/100-percent restart probes"
        )
    if key == "plateau_profiler_smoke_proof" or key == "work_counter_activity_repair":
        return "100- and 200-event Natural and Durable instrumentation smoke profiles"
    if "shadow" in key:
        return "paired fresh and copied-late 200-event Natural and Durable windows"
    return (
        "owner-authorized 15,695-packet Natural replay and focused fresh/copied-late "
        "windows where recorded; F=0.45, S=0.5, T=0.5"
    )


def state_proof_level(key: str) -> tuple[str, str]:
    if key == "hnsw_fixed_6c_experiment":
        return "benchmark-only", "measured"
    if key == "signal_centroid_ring_cutover":
        return "source-health", "measured"
    if key == "sqlite_hnsw_production_cutover":
        return "production-path", "measured"
    if key == "emotional_cascade_fixed_work_candidate":
        return "benchmark-only", "measured"
    if key == "bounded_route_hybrid_observation":
        return "benchmark-only", "modeled"
    if key == "bounded_sparse_routing_observation":
        return "benchmark-only", "observation-only"
    if key == "packed_route_sqlite_representation_experiment":
        return "benchmark-only", "benchmark-only"
    if key == "row_addressed_route_sqlite_experiment":
        return "benchmark-only", "benchmark-only"
    if key == "retrieval_quality_control_correction":
        return "benchmark-only", "measured"
    if key == "integrated_hnsw_sparse_route_experiment":
        return "source-health", "measured"
    return "source-health", "measured"


def build(state: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for inventory_id in sorted(state_inventory(state)):
        category, identity = inventory_id.split(":", 1)
        if category == "state":
            value = state[identity]
            proof_level, evidence_kind = state_proof_level(identity)
            limits = []
            if identity == "emotional_cascade_fixed_work_candidate":
                limits.extend(
                    [
                        "midpoint p99 and public identity-quality gates failed",
                        "nine-point knob ablation was not run after the fail-fast rejection",
                        "Durable full-horizon behavior is not proven for this rejected candidate",
                        "production cutover is not proven",
                    ]
                )
            if identity == "hnsw_fixed_6c_experiment":
                limits.extend(
                    [
                        "the additional production-default pre-screen failed the hard gates",
                        "zero of nine exact corpus-matrix points were run after fail-fast rejection",
                        "the production fixed-5C default was not changed",
                        "whole-engine boundedness and production cutover are not proven",
                    ]
                )
            if identity == "rif_epoch_consolidation_reset_candidate":
                limits.extend(
                    [
                        "the named RIF publication hotspot is flat but the whole-engine Natural plateau failed",
                        "mandatory Natural retrieval quality failed",
                        "Durable verification and bounded restart remain open",
                        "production cutover is not proven",
                    ]
                )
            if identity == "signal_centroid_ring_cutover":
                limits.extend(
                    [
                        "whole-engine Natural and Durable plateaus are not proven",
                        "emotional traversal and whole-engine restart remain unbounded",
                        "production cutover is not proven",
                    ]
                )
            if isinstance(value, dict) and value.get("production_cutover") is False:
                if "production cutover is not proven" not in limits:
                    limits.append("production cutover is not proven")
            if not limits:
                limits.append("the paper claim is limited to the recorded proof scope")
            record = {
                "inventory_id": inventory_id,
                "decision": first_decision(value),
                "corpus_parameter_scope": state_scope(identity),
                "evidence_fingerprints": [canonical_sha(value)],
                "unresolved_limits": limits,
                "proof_level": proof_level,
                "evidence_kind": evidence_kind,
            }
        else:
            record = {
                "inventory_id": inventory_id,
                "decision": "rejected",
                "corpus_parameter_scope": (
                    "focused fresh/copied-late candidate window under the recorded "
                    "15,695-packet Natural profile contract"
                ),
                "evidence_fingerprints": [canonical_sha(identity)],
                "unresolved_limits": ["candidate was removed after its failed gate"],
                "proof_level": "source-health",
                "evidence_kind": "measured",
            }
        records.append(record)

    variant_decisions = {
        "reciprocal-route-only": "rejected-recall-at-16-0.979858",
        "three-sparse-graphs": "rejected-no-recall-gain-at-equal-per-graph-work",
        "construction-budget-4096": "rejected-recall-at-16-0.995972",
        "construction-budget-8192-reciprocal-64": (
            "rejected-recall-at-16-0.997803"
        ),
        "two-graphs-construction-16384-retrieval-1280-reciprocal-64": (
            "selected-benchmark-design-production-proof-pending"
        ),
    }
    feasibility = state.get("bounded_route_hybrid_observation", {})
    for variant in VARIANTS:
        records.append(
            {
                "inventory_id": f"bounded-route-variant:{variant}",
                "decision": variant_decisions[variant],
                "corpus_parameter_scope": (
                    "34,456 normalized 256-dimensional embeddings and 512 "
                    "deterministic top-16 queries"
                ),
                "evidence_fingerprints": [canonical_sha(feasibility), canonical_sha(variant)],
                "unresolved_limits": [
                    "benchmark routing evidence is not production-path latency proof"
                ],
                "proof_level": "benchmark-only",
                "evidence_kind": "modeled",
            }
        )
    for group in ALGORITHM_GROUPS:
        records.append(
            {
                "inventory_id": f"algorithm:{group['id']}",
                "decision": "task-owned-source-change-recorded",
                "corpus_parameter_scope": (
                    "branch diff from the content-addressed base; experiment scope "
                    "is recorded by the associated state and candidate entries"
                ),
                "evidence_fingerprints": [canonical_sha(group)],
                "unresolved_limits": [
                    "production-wide boundedness is not inferred from source health"
                ],
                "proof_level": "source-health",
                "evidence_kind": "measured",
            }
        )
    for record in records:
        record["source_section"] = SECTION
    return {
        "schema": "cortext_paper_traceability_v1",
        "base_commit": BASE_COMMIT,
        "evaluated_bounded_route_variants": VARIANTS,
        "algorithm_groups": ALGORITHM_GROUPS,
        "record_defaults": {
            "source_section": SECTION,
        },
        "ownership_model": {
            "natural_durable_shared_algorithm": True,
            "durable_only_flush_barrier": True,
            "source_claim_fingerprints": [
                "Natural and Durable use one ingestion algorithm",
                "post-commit flush/checkpoint barrier",
            ],
            "manuscript_claim_fingerprints": [
                "Natural and Durable use one ingestion algorithm",
                "post-commit flush/checkpoint barrier",
            ],
        },
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--state-overlay", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    state = load_json(args.state)
    if args.state_overlay is not None:
        state = merge_state(state, load_json(args.state_overlay))
    manifest = build(state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
