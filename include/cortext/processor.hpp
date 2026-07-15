#pragma once

#include "cortext/consolidation_state.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/processor/operation.hpp"
#include "cortext/processor/processor_context.hpp"
#include "cortext/store/object_store.hpp"
#include "cortext/store/store.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cortext
{

struct Signal;
class OperationContext;
class Encoder;
class Clock;

/// @brief Orchestrates the signal processing.
///
/// The SignalProcessor hosts a composable set of operations that evaluate
/// signals and store them as memories within episodic transactions.
class SignalProcessor
{
public:
  /// @brief Configuration for the Processor's behavior.
  struct Config
  {
    double focus = 0.5;
    double sensitivity = 0.5;
    double stability = 0.5;
    bool affect_interrupt = true;
    bool affect_retrieval = true;
    bool reinforcement_enabled = true;
    bool procedural_enabled = true;
    bool sequential_edges_enabled = true;

    Encoder *encoder = nullptr; // Required for embedding-based operations
    std::shared_ptr<Clock> clock;
  };

  /// @brief Constructs a SignalProcessor with a defined set of operations.
  /// @param config The initial knob settings.
  /// @param store A shared pointer to the underlying database store.
  /// @param root_operation The root of the operation tree to execute.
  SignalProcessor (const Config &config, std::shared_ptr<Store> store,
                   std::unique_ptr<IOperation> root_operation,
                   std::shared_ptr<ObjectStore> object_store = nullptr);

  ~SignalProcessor ();

  // Disable copy and move semantics.
  SignalProcessor (const SignalProcessor &) = delete;
  SignalProcessor &operator= (const SignalProcessor &) = delete;
  SignalProcessor (SignalProcessor &&) = delete;
  SignalProcessor &operator= (SignalProcessor &&) = delete;

  /// @brief Output for a single Process() call.
  struct Output
  {
    // Memory-related
    std::vector<long long> candidate_memory_ids; // retrieved candidates (ids)
    std::vector<long long>
        used_memory_ids; // subset marked used in this processing step

    // Gate decisions
    bool interrupt_allowed = false;  // Algorithm 27
    bool at_boundary = false;        // Algorithm 12
    bool write_decision = false;     // Algorithm 7+8: score > (T - hysteresis)
    bool interrupt_aborted = false;  // Interrupt-triggered accumulator abort

    // Storage output (MemoryStorage operation)
    std::optional<long long> stored_embedding_id;  // Set if stored to memory
    std::optional<long long> stored_memory_id;     // Set if stored to memory
    std::optional<long long> stored_signal_id;     // Set if stored to signal

    // Key thresholds and stabilizers
    double threshold_T_dynamic = 0.0;  // Alg 8
    double threshold_hysteresis = 0.0; // Alg 8
    double effective_focus = 0.0;      // Alg 10 stabilizer
    double coherence = 0.0;            // Structural coherence

    // Emotion projections (Alg 4)
    double emotion_intensity = 0.0;
    double valence = 0.5;
    double arousal = 0.0;

    // MNI diagnostics (Alg 27)
    double mni_jaccard = 0.0;
    double mni_best_mu = 0.0;
    double mni_dup_thresh = 0.0;
    double mni_tau_jaccard_eff = 0.0;
    double mni_tau_mu_eff = 0.0;
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
    std::optional<double> boundary_score;
    /// @brief Boundary trigger label when at_boundary (e.g. explicit_turn).
    std::optional<std::string> boundary_type;
    ConsolidationState consolidation_state = ConsolidationState::None;

    // Composite score and serial-position application (if provided)
    std::optional<double> composite_score;
    std::optional<double> serial_position_multiplier;

    // Metrics (Algorithm 7 inputs) in normalized domains
    std::unordered_map<operations::Metric, double> metrics;

    // Per-operation timings (ms) for this signal
    std::unordered_map<std::string, double> operation_ms;

    // Soft Anchor diagnostics. Formation runs at ingress; retrieval/chat
    // consumption is not changed by these fields.
    bool soft_anchor_enabled = false;
    int soft_anchor_state_count = 0;
    int soft_anchor_link_count = 0;
    int soft_anchor_create_count = 0;
    int soft_anchor_update_count = 0;
    int soft_anchor_none_count = 0;
    double soft_anchor_last_update_us = 0.0;
    double soft_anchor_mean_update_us = 0.0;

  };

  /// @brief Processes a single signal by executing the instruction set.
  /// @param signal The signal to process.
  Output Process (const Signal &signal);

  /// @brief Commits buffered write instructions and starts a new episode.
  void Flush ();

private:
  void StartNewEpisode (Transaction *tx, uint64_t start_ts);
  void FinalizeEpisode (Transaction *tx, const OperationContext *op_context);

  // State persistence helpers (called within Process transaction)
  // v2 schema: Unified state persistence
  void PersistState (Transaction &tx);           // Unified STATE table
  void PersistWorkingMemory (Transaction &tx, bool force = false,
                             OperationContext *op_context = nullptr);

  Config config_;
  std::shared_ptr<Clock> clock_;
  std::shared_ptr<Store> store_;
  std::shared_ptr<ObjectStore> object_store_;
  std::unique_ptr<IOperation> root_operation_;

  std::unique_ptr<ProcessorContext> context_;
};

} // namespace cortext
