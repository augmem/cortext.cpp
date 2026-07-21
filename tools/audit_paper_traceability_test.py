#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import audit_paper_traceability as traceability
import build_paper_traceability_manifest as manifest_builder


class PaperTraceabilityAuditTests(unittest.TestCase):
    def test_current_state_overlay_replaces_stale_control_fields(self):
        merged = traceability.merge_state(
            {"phase": "implement-hotspot", "completion": False},
            {
                "phase": "production-cutover-review",
                "sqlite_hnsw_production_cutover": {
                    "decision": "retained-production-default"
                },
            },
        )
        self.assertEqual(merged["phase"], "production-cutover-review")
        self.assertIn(
            "state:sqlite_hnsw_production_cutover",
            traceability.state_inventory(merged),
        )

    def test_knob_derived_candidates_preserve_exact_ablation_scope(self):
        rif_scope = manifest_builder.state_scope(
            "rif_epoch_consolidation_reset_candidate"
        )
        self.assertIn("27 structural F/S/T points", rif_scope)
        self.assertIn("nine production-shaped reset points", rif_scope)
        self.assertIn("neutral B=128", rif_scope)
        self.assertIn("129 used only", rif_scope)

        hnsw_scope = manifest_builder.state_scope("hnsw_fixed_6c_experiment")
        self.assertIn("27 structural F/S/T points", hnsw_scope)
        self.assertIn("production-default pre-screen", hnsw_scope)
        self.assertIn("exact nine-point corpus matrix not started", hnsw_scope)
        self.assertEqual(
            manifest_builder.state_proof_level("hnsw_fixed_6c_experiment"),
            ("benchmark-only", "measured"),
        )

    def test_state_inventory_includes_candidates_repairs_and_rejections(self):
        state = {
            "plain": {"value": 1},
            "candidate_a": {"status": "rejected"},
            "measured": {"decision": "accepted"},
            "repair_b": {"artifact": "proof.json"},
            "rejected_candidate_ids": ["old-route"],
        }
        self.assertEqual(
            traceability.state_inventory(state),
            {
                "state:candidate_a",
                "state:measured",
                "state:repair_b",
                "rejected:old-route",
            },
        )

    def test_duplicate_json_keys_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaises(traceability.DuplicateKeyError):
                traceability.load_json(path)

    def test_heading_ignores_quarto_identifier(self):
        text = "# Title {#title}\n\n## Exact heading\n"
        self.assertTrue(traceability.section_contains_heading(text, "Title"))
        self.assertTrue(
            traceability.section_contains_heading(text, "Exact heading")
        )
        self.assertFalse(traceability.section_contains_heading(text, "Other"))

    def test_modeled_evidence_cannot_be_production_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sections = root / "docs" / "paper" / "sections"
            sections.mkdir(parents=True)
            section = sections / "test.qmd"
            section.write_text("# Test\n\nTRACE[state:candidate]\n", encoding="utf-8")
            manuscript = root / "index.md"
            manuscript.write_text("TRACE[state:candidate]\nshared path\nbarrier\n")
            state = {"candidate": {"decision": "accepted"}}
            manifest = {
                "schema": "cortext_paper_traceability_v1",
                "base_commit": "0" * 40,
                "evaluated_bounded_route_variants": ["route"],
                "algorithm_groups": [{"id": "group", "paths": ["src/a.cpp"]}],
                "ownership_model": {
                    "natural_durable_shared_algorithm": True,
                    "durable_only_flush_barrier": True,
                    "source_claim_fingerprints": ["TRACE[state:candidate]"],
                    "manuscript_claim_fingerprints": ["shared path", "barrier"],
                },
                "records": [
                    {
                        "inventory_id": "state:candidate",
                        "decision": "accepted",
                        "corpus_parameter_scope": "fixed",
                        "evidence_fingerprints": ["a" * 64],
                        "unresolved_limits": ["live proof pending"],
                        "proof_level": "production-path",
                        "evidence_kind": "modeled",
                        "source_section": {
                            "path": "docs/paper/sections/test.qmd",
                            "heading": "Test",
                        },
                        "source_claim_fingerprints": ["TRACE[state:candidate]"],
                        "manuscript_claim_fingerprints": ["TRACE[state:candidate]"],
                    }
                ],
            }
            original = traceability.changed_algorithm_paths
            traceability.changed_algorithm_paths = lambda *_: {"src/a.cpp"}
            try:
                result = traceability.audit(
                    state, manifest, sections, manuscript, root
                )
            finally:
                traceability.changed_algorithm_paths = original
            self.assertFalse(result["passed"])
            self.assertEqual(
                result["proof_level_violations"],
                ["state:candidate:modeled-labeled-production-path"],
            )


if __name__ == "__main__":
    unittest.main()
