#!/usr/bin/env python3
"""Fail-closed audit for experiment and algorithm paper traceability."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


STATE_INVENTORY_PATTERN = re.compile(
    r"(candidate|proof|experiment|evaluation|observation|repair|result|rerank)",
    re.IGNORECASE,
)
ALGORITHM_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hh", ".hpp"}
REQUIRED_RECORD_FIELDS = {
    "inventory_id",
    "decision",
    "corpus_parameter_scope",
    "evidence_fingerprints",
    "unresolved_limits",
    "proof_level",
    "evidence_kind",
    "source_section",
    "source_claim_fingerprints",
    "manuscript_claim_fingerprints",
}
NON_PRODUCTION_EVIDENCE = {
    "benchmark-only",
    "modeled",
    "observation-only",
    "synthetic",
}


class DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_reject_duplicate_keys)


def merge_state(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Apply a task-local current-state overlay to the historical ledger."""
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        raise ValueError("paper traceability state and overlay must be objects")
    return {**base, **overlay}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def state_inventory(state: dict[str, Any]) -> set[str]:
    inventory: set[str] = set()
    for key, value in state.items():
        if not isinstance(value, dict):
            continue
        serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if (
            STATE_INVENTORY_PATTERN.search(key)
            or '"decision"' in serialized
            or '"proof_artifact"' in serialized
            or '"experiment_result"' in serialized
        ):
            inventory.add(f"state:{key}")
    for candidate in state.get("rejected_candidate_ids", []):
        if isinstance(candidate, str) and candidate:
            inventory.add(f"rejected:{candidate}")
    return inventory


def _git_lines(repository: Path, arguments: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def changed_algorithm_paths(repository: Path, base_commit: str) -> set[str]:
    committed = _git_lines(
        repository, ["diff", "--name-only", f"{base_commit}...HEAD"]
    )
    working = _git_lines(repository, ["diff", "--name-only"])
    untracked = _git_lines(repository, ["ls-files", "--others", "--exclude-standard"])
    result: set[str] = set()
    for relative in committed + working + untracked:
        path = Path(relative)
        is_engine_algorithm = (
            path.parts
            and path.parts[0] in {"src", "include"}
            and path.suffix.lower() in ALGORITHM_SUFFIXES
        )
        is_benchmark_implementation = (
            len(path.parts) >= 3
            and path.parts[:2] == ("examples", "benchmark")
            and path.suffix.lower() in ALGORITHM_SUFFIXES
        )
        is_experiment_tool = (
            len(path.parts) == 2
            and path.parts[0] == "tools"
            and path.suffix == ".py"
            and not path.stem.endswith("_test")
            and path.name not in {
                "audit_paper_traceability.py",
                "build_paper_traceability_manifest.py",
            }
        )
        if is_engine_algorithm or is_benchmark_implementation or is_experiment_tool:
            result.add(path.as_posix())
    return result


def section_contains_heading(text: str, heading: str) -> bool:
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        normalized = re.sub(r"\s*\{[^}]*\}\s*$", "", line.lstrip("#").strip())
        if normalized == heading:
            return True
    return False


def _all_present(text: str, fingerprints: Iterable[str]) -> bool:
    return all(isinstance(value, str) and value and value in text for value in fingerprints)


def audit(
    state: dict[str, Any],
    manifest: dict[str, Any],
    sections_root: Path,
    manuscript_path: Path,
    repository: Path,
) -> dict[str, Any]:
    uncovered_inventory: list[str] = []
    missing_source_claims: list[str] = []
    generated_claim_mismatches: list[str] = []
    proof_level_violations: list[str] = []
    ownership_violations: list[str] = []

    if manifest.get("schema") != "cortext_paper_traceability_v1":
        uncovered_inventory.append("manifest:schema")

    records = manifest.get("records")
    if not isinstance(records, list):
        records = []
        uncovered_inventory.append("manifest:records")

    record_by_id: dict[str, dict[str, Any]] = {}
    duplicate_record_ids: set[str] = set()
    record_defaults = manifest.get("record_defaults", {})
    if not isinstance(record_defaults, dict):
        uncovered_inventory.append("manifest:record_defaults")
        record_defaults = {}
    for index, raw_record in enumerate(records):
        record = raw_record
        if not isinstance(record, dict):
            uncovered_inventory.append(f"record:{index}:not-object")
            continue
        record = {**record_defaults, **record}
        inventory_id = record.get("inventory_id")
        if isinstance(inventory_id, str) and inventory_id:
            marker = f"TRACE[{inventory_id}]"
            record.setdefault("source_claim_fingerprints", [marker])
            record.setdefault("manuscript_claim_fingerprints", [marker])
        missing = sorted(REQUIRED_RECORD_FIELDS - set(record))
        if missing:
            uncovered_inventory.extend(
                f"record:{index}:missing:{field}" for field in missing
            )
        inventory_id = record.get("inventory_id")
        if not isinstance(inventory_id, str) or not inventory_id:
            uncovered_inventory.append(f"record:{index}:invalid-inventory-id")
            continue
        if inventory_id in record_by_id:
            duplicate_record_ids.add(inventory_id)
        record_by_id[inventory_id] = record
    uncovered_inventory.extend(
        f"duplicate:{inventory_id}" for inventory_id in sorted(duplicate_record_ids)
    )

    expected_inventory = state_inventory(state)
    variants = manifest.get("evaluated_bounded_route_variants", [])
    if not isinstance(variants, list) or not variants:
        uncovered_inventory.append("manifest:evaluated_bounded_route_variants")
        variants = []
    for variant in variants:
        if isinstance(variant, str) and variant:
            expected_inventory.add(f"bounded-route-variant:{variant}")
        else:
            uncovered_inventory.append("manifest:invalid-bounded-route-variant")

    groups = manifest.get("algorithm_groups", [])
    if not isinstance(groups, list) or not groups:
        ownership_violations.append("manifest:algorithm_groups")
        groups = []
    owned_paths: dict[str, list[str]] = {}
    for group in groups:
        if not isinstance(group, dict):
            ownership_violations.append("algorithm-group:not-object")
            continue
        group_id = group.get("id")
        paths = group.get("paths")
        if not isinstance(group_id, str) or not group_id:
            ownership_violations.append("algorithm-group:invalid-id")
            continue
        expected_inventory.add(f"algorithm:{group_id}")
        if not isinstance(paths, list) or not paths:
            ownership_violations.append(f"algorithm:{group_id}:missing-paths")
            continue
        for path in paths:
            if not isinstance(path, str) or not path:
                ownership_violations.append(f"algorithm:{group_id}:invalid-path")
                continue
            owned_paths.setdefault(path, []).append(group_id)

    base_commit = manifest.get("base_commit")
    if not isinstance(base_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", base_commit):
        ownership_violations.append("manifest:base_commit")
        changed_paths: set[str] = set()
    else:
        try:
            changed_paths = changed_algorithm_paths(repository, base_commit)
        except (OSError, subprocess.CalledProcessError) as error:
            ownership_violations.append(f"git-diff:{type(error).__name__}")
            changed_paths = set()
    for path in sorted(changed_paths):
        owners = owned_paths.get(path, [])
        if len(owners) != 1:
            ownership_violations.append(
                f"algorithm-path:{path}:owner-count:{len(owners)}"
            )
    for path, owners in sorted(owned_paths.items()):
        if path not in changed_paths:
            ownership_violations.append(f"algorithm-path:{path}:not-in-task-diff")
        if len(owners) != 1:
            ownership_violations.append(
                f"algorithm-path:{path}:owner-count:{len(owners)}"
            )

    for inventory_id in sorted(expected_inventory):
        if inventory_id not in record_by_id:
            uncovered_inventory.append(inventory_id)
    for inventory_id in sorted(set(record_by_id) - expected_inventory):
        uncovered_inventory.append(f"unexpected:{inventory_id}")

    manuscript = manuscript_path.read_text(encoding="utf-8")
    section_cache: dict[Path, str] = {}
    for inventory_id, record in sorted(record_by_id.items()):
        evidence = record.get("evidence_fingerprints")
        limits = record.get("unresolved_limits")
        if not isinstance(evidence, list) or not evidence:
            missing_source_claims.append(f"{inventory_id}:evidence-fingerprints")
        if not isinstance(limits, list):
            missing_source_claims.append(f"{inventory_id}:unresolved-limits")

        proof_level = record.get("proof_level")
        evidence_kind = record.get("evidence_kind")
        if proof_level == "production-path" and evidence_kind in NON_PRODUCTION_EVIDENCE:
            proof_level_violations.append(
                f"{inventory_id}:{evidence_kind}-labeled-production-path"
            )
        if proof_level not in {
            "source-health",
            "benchmark-only",
            "modeled",
            "production-path",
        }:
            proof_level_violations.append(f"{inventory_id}:unknown:{proof_level}")

        section = record.get("source_section")
        if not isinstance(section, dict):
            missing_source_claims.append(f"{inventory_id}:source-section")
            continue
        relative_path = section.get("path")
        heading = section.get("heading")
        if not isinstance(relative_path, str) or not isinstance(heading, str):
            missing_source_claims.append(f"{inventory_id}:source-location")
            continue
        section_path = repository / relative_path
        try:
            section_path.relative_to(sections_root)
        except ValueError:
            missing_source_claims.append(f"{inventory_id}:outside-sections")
            continue
        if not section_path.is_file():
            missing_source_claims.append(f"{inventory_id}:missing-section")
            continue
        text = section_cache.setdefault(
            section_path, section_path.read_text(encoding="utf-8")
        )
        if not section_contains_heading(text, heading):
            missing_source_claims.append(f"{inventory_id}:missing-heading:{heading}")
        source_fingerprints = record.get("source_claim_fingerprints")
        if not isinstance(source_fingerprints, list) or not _all_present(
            text, source_fingerprints
        ):
            missing_source_claims.append(f"{inventory_id}:claim-fingerprint")
        manuscript_fingerprints = record.get("manuscript_claim_fingerprints")
        if not isinstance(manuscript_fingerprints, list) or not _all_present(
            manuscript, manuscript_fingerprints
        ):
            generated_claim_mismatches.append(
                f"{inventory_id}:manuscript-fingerprint"
            )

    ownership = manifest.get("ownership_model", {})
    if not isinstance(ownership, dict):
        ownership_violations.append("ownership-model:not-object")
    else:
        if ownership.get("natural_durable_shared_algorithm") is not True:
            ownership_violations.append("ownership-model:shared-algorithm")
        if ownership.get("durable_only_flush_barrier") is not True:
            ownership_violations.append("ownership-model:durable-barrier")
        source_fingerprints = ownership.get("source_claim_fingerprints", [])
        manuscript_fingerprints = ownership.get("manuscript_claim_fingerprints", [])
        section_text = "\n".join(section_cache.values())
        if not _all_present(section_text, source_fingerprints):
            ownership_violations.append("ownership-model:source-claim")
        if not _all_present(manuscript, manuscript_fingerprints):
            ownership_violations.append("ownership-model:manuscript-claim")

    result = {
        "schema": "cortext_paper_traceability_audit_v1",
        "passed": not any(
            (
                uncovered_inventory,
                missing_source_claims,
                generated_claim_mismatches,
                proof_level_violations,
                ownership_violations,
            )
        ),
        "inventory_count": len(expected_inventory),
        "record_count": len(records),
        "state_sha256": sha256_text(
            json.dumps(state, sort_keys=True, separators=(",", ":"))
        ),
        "manifest_sha256": sha256_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        ),
        "manuscript_sha256": sha256_text(manuscript),
        "uncovered_inventory": sorted(set(uncovered_inventory)),
        "missing_source_claims": sorted(set(missing_source_claims)),
        "generated_claim_mismatches": sorted(set(generated_claim_mismatches)),
        "proof_level_violations": sorted(set(proof_level_violations)),
        "ownership_violations": sorted(set(ownership_violations)),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--state-overlay", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sections", required=True, type=Path)
    parser.add_argument("--manuscript", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    repository = Path.cwd().resolve()
    try:
        state = load_json(args.state)
        if args.state_overlay is not None:
            state = merge_state(state, load_json(args.state_overlay))
        manifest = load_json(args.manifest)
        result = audit(
            state,
            manifest,
            args.sections.resolve(),
            args.manuscript.resolve(),
            repository,
        )
    except (DuplicateKeyError, json.JSONDecodeError, OSError, ValueError) as error:
        result = {
            "schema": "cortext_paper_traceability_audit_v1",
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
            "uncovered_inventory": ["audit-input-invalid"],
            "missing_source_claims": [],
            "generated_claim_mismatches": [],
            "proof_level_violations": [],
            "ownership_violations": [],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    sys.exit(main())
