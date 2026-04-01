#include <catch2/catch_test_macros.hpp>

#include "../examples/chat/chunk_diagnostics.hpp"

TEST_CASE("Chunk probe reason classifier follows streaming debug priority",
          "[chat][chunks]") {
  using chat::ClassifyChunkProbeReason;
  using chat::ChunkProbeReasonInputs;

  SECTION("interrupt wins when new memories were added") {
    REQUIRE(ClassifyChunkProbeReason({
        .should_interrupt = true,
        .new_memory_count = 2,
        .at_boundary = true,
        .boundary_score_pass = true,
        .interrupt_gate_has_candidates = true,
        .interrupt_gate_rel_pass = true,
        .interrupt_gate_novelty_mu_pass = true,
        .interrupt_gate_dup_pass = true,
        .interrupt_gate_boundary_mu_pass = true,
    }) == "interrupt_triggered");
  }

  SECTION("suppressed interrupt wins before gate failure reasons") {
    REQUIRE(ClassifyChunkProbeReason({
        .should_interrupt = true,
        .new_memory_count = 0,
        .at_boundary = true,
        .boundary_score_pass = true,
        .interrupt_gate_has_candidates = true,
        .interrupt_gate_rel_pass = true,
        .interrupt_gate_novelty_mu_pass = true,
        .interrupt_gate_dup_pass = true,
        .interrupt_gate_boundary_mu_pass = true,
    }) == "interrupt_suppressed_no_new_memories");
  }

  SECTION("boundary checks happen before gate diagnostics") {
    REQUIRE(ClassifyChunkProbeReason({
        .should_interrupt = false,
        .new_memory_count = 0,
        .at_boundary = false,
        .boundary_score_pass = false,
        .interrupt_gate_has_candidates = false,
    }) == "not_at_boundary");

    REQUIRE(ClassifyChunkProbeReason({
        .should_interrupt = false,
        .new_memory_count = 0,
        .at_boundary = true,
        .boundary_score_pass = false,
        .interrupt_gate_has_candidates = true,
    }) == "boundary_score_below_threshold");
  }

  SECTION("gate failures map to the expected reasons") {
    REQUIRE(ClassifyChunkProbeReason({
        .should_interrupt = false,
        .new_memory_count = 0,
        .at_boundary = true,
        .boundary_score_pass = true,
        .interrupt_gate_has_candidates = false,
    }) == "no_candidates");

    REQUIRE(ClassifyChunkProbeReason({
        .should_interrupt = false,
        .new_memory_count = 0,
        .at_boundary = true,
        .boundary_score_pass = true,
        .interrupt_gate_has_candidates = true,
        .interrupt_gate_blocked_no_store = true,
    }) == "blocked_no_store");

    REQUIRE(ClassifyChunkProbeReason({
        .should_interrupt = false,
        .new_memory_count = 0,
        .at_boundary = true,
        .boundary_score_pass = true,
        .interrupt_gate_has_candidates = true,
        .interrupt_gate_rel_pass = false,
    }) == "relevance_below_threshold");

    REQUIRE(ClassifyChunkProbeReason({
        .should_interrupt = false,
        .new_memory_count = 0,
        .at_boundary = true,
        .boundary_score_pass = true,
        .interrupt_gate_has_candidates = true,
        .interrupt_gate_rel_pass = true,
        .interrupt_gate_novelty_mu_pass = false,
    }) == "novelty_and_mu_failed");

    REQUIRE(ClassifyChunkProbeReason({
        .should_interrupt = false,
        .new_memory_count = 0,
        .at_boundary = true,
        .boundary_score_pass = true,
        .interrupt_gate_has_candidates = true,
        .interrupt_gate_rel_pass = true,
        .interrupt_gate_novelty_mu_pass = true,
        .interrupt_gate_dup_pass = false,
    }) == "duplicate_overlap");

    REQUIRE(ClassifyChunkProbeReason({
        .should_interrupt = false,
        .new_memory_count = 0,
        .at_boundary = true,
        .boundary_score_pass = true,
        .interrupt_gate_has_candidates = true,
        .interrupt_gate_rel_pass = true,
        .interrupt_gate_novelty_mu_pass = true,
        .interrupt_gate_dup_pass = true,
        .interrupt_gate_boundary_mu_pass = false,
    }) == "boundary_mu_failed");

    REQUIRE(ClassifyChunkProbeReason({
        .should_interrupt = false,
        .new_memory_count = 0,
        .at_boundary = true,
        .boundary_score_pass = true,
        .interrupt_gate_has_candidates = true,
        .interrupt_gate_rel_pass = true,
        .interrupt_gate_novelty_mu_pass = true,
        .interrupt_gate_dup_pass = true,
        .interrupt_gate_boundary_mu_pass = true,
    }) == "gate_denied");
  }
}
