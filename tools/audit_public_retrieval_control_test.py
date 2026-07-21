import copy
import unittest

from audit_public_retrieval_control import audit


def ranked(ids):
    return [
        {"memory_id": identity, "embedding_id": identity + 100, "score": 1 - rank / 100, "rank": rank}
        for rank, identity in enumerate(ids)
    ]


def fixture(candidate=False):
    queries = []
    for ordinal in range(2):
        query = {
            "query_ordinal": ordinal,
            "event_index": 10 + ordinal * 10,
            "query_timestamp": 1000 + ordinal * 1000,
            "source_id_blake3": f"{ordinal + 1:064x}",
            "modality": "text",
            "query_embedding": [0.0] * 256,
            "control_seed_ranked": ranked([1, 2]),
            "control_graph_ranked": ranked([1, 2]),
            "control_public_ranked": ranked([1, 2]),
            "candidate_available": candidate,
        }
        if candidate:
            query.update(
                {
                    "candidate_seed_ranked": ranked([1, 2]),
                    "semantic_coverage": 1.0,
                    "deterministic_tie_order": True,
                }
            )
        queries.append(query)
    return {
        "public_retrieval_control": {
            "schema": "cortext_current_public_retrieval_control_v1",
            "identity_kind": "memory-id",
            "query_count": 2,
            "result_k": 2,
            "retrieval_active_event_count": 10,
            "eligibility_semantics": {
                "timestamp_exclusion": True,
                "supersession_exclusion": True,
                "current_surface_selection": True,
                "family_collapse": True,
                "deterministic_tie_order": True,
            },
            "coverage_contract": {
                "minimum_opaque_source_ids": 4,
                "required_modalities": ["text"],
                "modality_agnostic_proof": "focused active-route label-invariance test",
            },
            "production_cutover": False,
            "surface_mutations": [
                {
                    "action": "upsert",
                    "memory_id": 1,
                    "event_index": 2,
                    "event_timestamp": 200,
                },
                {
                    "action": "upsert",
                    "memory_id": 2,
                    "event_index": 3,
                    "event_timestamp": 300,
                },
            ],
            "queries": queries,
        }
    }


class PublicRetrievalControlAuditTest(unittest.TestCase):
    def test_control_only_is_valid_but_not_candidate_proof(self):
        result = audit(fixture(), "0" * 64)
        self.assertTrue(result["passed"])
        self.assertTrue(result["control_population_passed"])
        self.assertFalse(result["candidate_complete"])
        self.assertFalse(result["candidate_quality_passed"])

    def test_duplicate_or_unsorted_identity_fails(self):
        value = fixture()
        value["public_retrieval_control"]["queries"][0]["control_seed_ranked"] = ranked([1, 1])
        with self.assertRaisesRegex(ValueError, "repeats"):
            audit(value, "0" * 64)

    def test_missing_runtime_semantic_fails(self):
        value = fixture()
        del value["public_retrieval_control"]["eligibility_semantics"]["family_collapse"]
        with self.assertRaisesRegex(ValueError, "semantics"):
            audit(value, "0" * 64)

    def test_candidate_without_declared_source_coverage_does_not_pass(self):
        result = audit(fixture(candidate=True), "0" * 64)
        self.assertTrue(result["candidate_complete"])
        self.assertEqual(result["exact_top1"], 1.0)
        self.assertFalse(result["candidate_quality_passed"])

    def test_candidate_passes_declared_system_coverage_contract(self):
        value = fixture(candidate=True)
        value["public_retrieval_control"]["coverage_contract"][
            "minimum_opaque_source_ids"
        ] = 2
        result = audit(value, "0" * 64)
        self.assertTrue(result["candidate_quality_passed"])
        self.assertEqual(result["modalities"], ["text"])
        self.assertEqual(result["top1_miss_fingerprints"], [])
        self.assertTrue(result["query_fingerprints"][0]["top1_exact_match"])
        self.assertEqual(
            result["query_fingerprints"][0]["control_top1_age_events"], 8
        )

    def test_single_top1_miss_has_sanitized_cluster_dimensions(self):
        value = fixture(candidate=True)
        value["public_retrieval_control"]["coverage_contract"][
            "minimum_opaque_source_ids"
        ] = 2
        value["public_retrieval_control"]["queries"][0][
            "candidate_seed_ranked"
        ] = ranked([2, 1])
        result = audit(value, "0" * 64)
        self.assertEqual(result["exact_top1_miss_count"], 1)
        self.assertEqual(
            result["top1_miss_fingerprints"],
            [
                {
                    "query_ordinal": 0,
                    "query_progress_quartile": 0,
                    "source_id_blake3": f"{1:064x}",
                    "modality": "text",
                    "control_top1_age_events": 8,
                    "control_top1_age_ms": 800,
                }
            ],
        )

    def test_missing_coverage_contract_fails(self):
        value = fixture(candidate=True)
        del value["public_retrieval_control"]["coverage_contract"]
        with self.assertRaisesRegex(ValueError, "coverage contract"):
            audit(value, "0" * 64)


if __name__ == "__main__":
    unittest.main()
