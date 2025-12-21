#pragma once

#include "cortext/operations/metrics.hpp"
#include "cortext/processor/operation.hpp"
#include "cortext/processor/processor_context.hpp"
#include "cortext/store/store.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace cortext
{

struct Signal;
class OperationContext;
class Encoder;

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

    // LLM components (extractor/summarizer may be null)
    Extractor *extractor = nullptr;
    Summarizer *summarizer = nullptr;
    Encoder *encoder = nullptr; // Required for embedding-based operations
  };

  /// @brief Constructs a SignalProcessor with a defined set of operations.
  /// @param config The initial knob settings.
  /// @param store A shared pointer to the underlying database store.
  /// @param root_operation The root of the operation tree to execute.
  SignalProcessor (const Config &config, std::shared_ptr<Store> store,
                   std::unique_ptr<IOperation> root_operation);

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

    // Storage output (MemoryStorage operation)
    std::optional<long long> stored_embedding_id;  // Set if stored to memory

    // Key thresholds and stabilizers
    double threshold_T_dynamic = 0.0;  // Alg 8
    double threshold_hysteresis = 0.0; // Alg 8
    double effective_focus = 0.0;      // Alg 10 stabilizer

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

    // Composite score and serial-position application (if provided)
    std::optional<double> composite_score;
    std::optional<double> serial_position_multiplier;

    // Metrics (Algorithm 7 inputs) in normalized domains
    std::unordered_map<operations::Metric, double> metrics;
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
  void PersistWorkingMemory (Transaction &tx);   // MEMORIES with kind='WORKING'
  void PersistAccumulators (Transaction &tx);    // ACCUMULATORS table

  Config config_;
  std::shared_ptr<Store> store_;
  std::unique_ptr<IOperation> root_operation_;

  std::unique_ptr<ProcessorContext> context_;
};

} // namespace cortext
