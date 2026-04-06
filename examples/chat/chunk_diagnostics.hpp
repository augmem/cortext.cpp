#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace chat {

struct ChunkProbeReasonInputs {
  bool should_interrupt = false;
  int new_memory_count = 0;
  bool interrupt_ignored_restart_cap = false;
  bool at_boundary = false;
  bool boundary_score_pass = false;
  bool interrupt_gate_has_candidates = false;
  bool interrupt_gate_blocked_no_store = false;
  bool interrupt_gate_rel_pass = false;
  bool interrupt_gate_novelty_mu_pass = false;
  bool interrupt_gate_dup_pass = false;
  bool interrupt_gate_boundary_mu_pass = false;
};

inline std::string ClassifyChunkProbeReason(const ChunkProbeReasonInputs& inputs) {
  if (inputs.should_interrupt && inputs.interrupt_ignored_restart_cap) {
    return "interrupt_ignored_restart_cap";
  }
  if (inputs.should_interrupt && inputs.new_memory_count > 0) {
    return "interrupt_triggered";
  }
  if (inputs.should_interrupt && inputs.new_memory_count == 0) {
    return "interrupt_suppressed_no_new_memories";
  }
  if (!inputs.at_boundary) {
    return "not_at_boundary";
  }
  if (!inputs.boundary_score_pass) {
    return "boundary_score_below_threshold";
  }
  if (!inputs.interrupt_gate_has_candidates) {
    return "no_candidates";
  }
  if (inputs.interrupt_gate_blocked_no_store) {
    return "blocked_no_store";
  }
  if (!inputs.interrupt_gate_rel_pass) {
    return "relevance_below_threshold";
  }
  if (!inputs.interrupt_gate_novelty_mu_pass) {
    return "novelty_and_mu_failed";
  }
  if (!inputs.interrupt_gate_dup_pass) {
    return "duplicate_overlap";
  }
  if (!inputs.interrupt_gate_boundary_mu_pass) {
    return "boundary_mu_failed";
  }
  return "gate_denied";
}

struct ChunkRetrievedMemory {
  long long memory_id = 0;
  std::string source_id;
  std::string preview;
  double relevance = 0.0;
  double composite_score = 0.0;
};

struct ChunkProbeEvent {
  std::uint64_t event_id = 0;
  std::uint64_t started_at_ms = 0;
  std::uint64_t completed_at_ms = 0;
  bool in_progress = false;
  bool had_error = false;
  std::string error_message;

  std::string chunk_text;
  std::size_t chunk_char_count = 0;
  int restart_index = 0;
  std::size_t tokens_so_far = 0;

  double encode_ms = 0.0;
  double process_ms = 0.0;
  double hydrate_ms = 0.0;
  double total_ms = 0.0;

  bool at_boundary = false;
  std::optional<double> boundary_score;
  double boundary_threshold = 0.0;
  bool boundary_score_pass = false;

  bool should_interrupt = false;
  int new_memory_count = 0;
  bool interrupt_ignored_restart_cap = false;
  std::size_t raw_retrieved_count = 0;
  std::string reason;

  bool interrupt_gate_has_candidates = false;
  bool interrupt_gate_blocked_no_store = false;
  bool interrupt_gate_rel_pass = false;
  bool interrupt_gate_novelty_pass = false;
  bool interrupt_gate_mu_pass = false;
  bool interrupt_gate_novelty_mu_pass = false;
  bool interrupt_gate_dup_pass = false;
  bool interrupt_gate_boundary_mu_pass = false;
  double interrupt_gate_rel_star = 0.0;
  double interrupt_gate_retrieval_thresh = 0.0;
  double interrupt_gate_boundary_mult_eff = 0.0;
  double interrupt_gate_affect_drive = 0.0;

  std::vector<ChunkRetrievedMemory> retrieved_memories;
};

struct ChunkDiagnosticsState {
  mutable std::mutex mu;
  std::optional<ChunkProbeEvent> active_probe;
  std::deque<ChunkProbeEvent> recent_probes;
  std::uint64_t next_event_id = 1;
  std::size_t total_probes = 0;
  std::size_t interrupts = 0;
  std::size_t suppressed = 0;
  std::size_t errors = 0;
};

}  // namespace chat
