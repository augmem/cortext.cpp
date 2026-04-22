#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "cortext/consolidation_mode.hpp"
#include "cortext/export.hpp"
#include "cortext/stop_token.hpp"

namespace cortext
{

class Cortext;
class ObjectStore;
class Store;

// Forward declaration for ProcessorOutput fields
namespace operations
{
enum class Metric;
}

namespace internal
{
class StreamingTextProbeSession;
}

/// @brief High-level entrypoint for Cortext with tri-modal process stubs.
class CORTEXT_EXPORT Cortext
{
public:
  /// @brief Simplified output metrics from processor.
  struct ProcessorOutput
  {
    std::optional<double> composite_score;
    std::optional<double> threshold;
    std::optional<bool> decision;
    double effective_focus = 0.0;
    double coherence = 0.0;
    double emotion_intensity = 0.0;
    double valence = 0.5;
    double arousal = 0.0;
    std::unordered_map<int, double> metrics; // Metric enum cast to int -> value
    std::unordered_map<std::string, double> operation_ms;
    std::optional<long long> stored_embedding_id;  // Set if stored to memory
  };
  
  /// @brief Context returned from processing calls, containing hydrated
  /// memories.
  struct Context
  {
    struct Memory
    {
      std::string source_id;
      long long id = 0;
      std::uint64_t timestamp = 0;
      long long retrieved_count = 0;
      long long used_count = 0;
      std::vector<std::vector<unsigned char>> content;  ///< Blobs from signals (one per signal in slot)
      std::string modality;
      std::string mimetype;

      // Flattened metrics (formerly HistoricalMetrics)
      double relevance = 0.0;
      double mismatch = 0.0;
      double surprise = 0.0;
      double rarity = 0.0;
      double drift = 0.0;
      double contradiction = 0.0;
      double utility = 0.0;
      double periphery = 0.0;
      double coverage = 0.0;
      double salience = 0.0;
      double valence = 0.5;
      double arousal = 0.0;
      double composite_score = 0.0;
      double threshold_t = 0.0;
    };

    std::vector<Memory> working_memory;    ///< Active WM slots (conversation context)
    std::vector<Memory> retrieved_memory;  ///< Long-term retrieval results (injected context)
    bool should_interrupt = false;
    bool interrupt_aborted = false;
    bool at_boundary = false;
    bool consolidation_recommended = false;
    bool consolidation_required = false;
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
    ProcessorOutput output;
    double encode_ms = 0.0;
    double process_ms = 0.0;
    double hydrate_ms = 0.0;
    double total_ms = 0.0;
  };

  /// @brief Three-knob configuration preserved across the system.
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
    std::string label_bank_path;
  };

  /// @brief Factory to create a Cortext instance.
  /// @param cfg Three-knob configuration.
  /// @param db_path SQLite database path (e.g. ":memory:" or a file path).
  /// @param models_dir Models directory root (e.g. "models").
  /// @return A unique_ptr to a Cortext instance.
  static std::unique_ptr<Cortext> Create (const Config &cfg,
                                          const std::string &db_path,
                                          const std::string &models_dir);

  /// @brief Factory using SQLite for metadata and caller-supplied object store.
  static std::unique_ptr<Cortext> Create (
      const Config &cfg, const std::string &db_path,
      std::shared_ptr<ObjectStore> object_store, const std::string &models_dir);

  /// @brief Factory to create a Cortext instance using a caller-supplied store.
  /// @param cfg Three-knob configuration.
  /// @param store Caller-owned database store. Cortext keeps a shared reference
  /// and does not close externally supplied stores.
  /// @param models_dir Models directory root (e.g. "models").
  /// @return A unique_ptr to a Cortext instance.
  static std::unique_ptr<Cortext> Create (const Config &cfg,
                                          std::shared_ptr<Store> store,
                                          const std::string &models_dir);

  /// @brief Factory with caller-supplied database and object stores.
  /// @param cfg Three-knob configuration.
  /// @param store Caller-owned database store.
  /// @param object_store Caller-owned content-addressed object store.
  /// @param models_dir Models directory root (e.g. "models").
  /// @return A unique_ptr to a Cortext instance.
  static std::unique_ptr<Cortext> Create (
      const Config &cfg, std::shared_ptr<Store> store,
      std::shared_ptr<ObjectStore> object_store, const std::string &models_dir);

  /// @brief Process text input.
  /// @param text The text content to process.
  /// @param source_id Identifier for the signal source (e.g., "chat/user").
  /// @return Context with retrieved memories and processing output.
  Context ProcessText (const std::string &text, const std::string &source_id);

  /// @brief Process text input with an explicit timestamp (ms since epoch).
  /// @param text The text content to process.
  /// @param source_id Identifier for the signal source (e.g., "chat/user").
  /// @param timestamp Milliseconds since Unix epoch (must be monotonic).
  /// @return Context with retrieved memories and processing output.
  Context ProcessTextAt (const std::string &text, const std::string &source_id,
                         std::uint64_t timestamp);


  /// @brief Process audio PCM input.
  /// @param pcm Pointer to PCM float samples (16kHz mono expected).
  /// @param num_samples Number of samples.
  /// @param source_id Identifier for the signal source.
  /// @return Context with retrieved memories and processing output.
  Context ProcessAudio (const float *pcm, std::size_t num_samples,
                        const std::string &source_id);

  /// @brief Process image input.
  /// @param data Raw pixel data (RGB/RGBA).
  /// @param width Image width.
  /// @param height Image height.
  /// @param channels Number of channels (3 for RGB, 4 for RGBA).
  /// @param source_id Identifier for the signal source.
  /// @return Context with retrieved memories and processing output.
  Context ProcessImage (const std::uint8_t *data, int width, int height,
                        int channels, const std::string &source_id);

  /// @brief Attempt to trigger consolidation if conditions allow.
  /// @param mode Shallow (embedding-only), deep (local summary/labels backend),
  /// or both.
  Context Consolidate (ConsolidationMode mode = ConsolidationMode::Both);
  Context Consolidate (StopToken stop_token,
                       ConsolidationMode mode = ConsolidationMode::Both);

  /// @brief Flush/commit any pending episode writes.
  void Flush ();

#if defined(CORTEXT_TESTING)
  /// @brief Test-only helper to hydrate arbitrary memory ids.
  Context DebugHydrateForTest (const std::vector<long long> &candidate_ids,
                               const std::vector<long long> &used_ids);
#endif

  // Content persistence API (best-effort; no-op if id unavailable).
  // These helpers allow callers to persist raw content once a memory id
  // exists. Implemented using sqlite-objstore for payloads and SQLite
  // memory_index for metadata.
  static std::string MakeContentKey (long long embedding_id);
  static std::string MakeAudioMimePcmF32 ();
  static std::string MakeImageMimePng ();

  ~Cortext ();

  Cortext (const Cortext &) = delete;
  Cortext &operator= (const Cortext &) = delete;
  Cortext (Cortext &&) = delete;
  Cortext &operator= (Cortext &&) = delete;

private:
  friend class internal::StreamingTextProbeSession;
  struct Impl;
  explicit Cortext (const Config &cfg, const std::string &db_path,
                    const std::string &models_dir);
  explicit Cortext (const Config &cfg, const std::string &db_path,
                    std::shared_ptr<ObjectStore> object_store,
                    const std::string &models_dir);
  explicit Cortext (const Config &cfg, std::shared_ptr<Store> store,
                    const std::string &models_dir);
  explicit Cortext (const Config &cfg, std::shared_ptr<Store> store,
                    std::shared_ptr<ObjectStore> object_store,
                    const std::string &models_dir);
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext
