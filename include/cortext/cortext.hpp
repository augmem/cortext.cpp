#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "cortext/export.hpp"
#include "cortext/retention.hpp"
#include "cortext/stop_token.hpp"

namespace cortext
{

class Cortext;
class Clock;
class ObjectStore;
class Store;

// Forward declaration for ProcessorOutput fields
namespace operations
{
enum class Metric;
}

namespace internal
{
class ReplayIngress;
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
    std::optional<long long> stored_memory_id;     // Set if stored to memory
    std::optional<long long> stored_signal_id;     // Set if stored to signal
    bool signal_filter_evaluated = false;
    bool signal_filter_accepted = true;
    std::string signal_filter_modality;
    std::string signal_filter_reason;
    double signal_filter_score = 0.0;
    double signal_filter_threshold = 0.0;
    double signal_filter_quiet_seconds = 0.0;
    bool soft_anchor_enabled = false;
    int soft_anchor_state_count = 0;
    int soft_anchor_link_count = 0;
    int soft_anchor_create_count = 0;
    int soft_anchor_update_count = 0;
    int soft_anchor_none_count = 0;
    double soft_anchor_last_update_us = 0.0;
    double soft_anchor_mean_update_us = 0.0;
  };
  
  /// @brief Context returned from processing calls, containing hydrated
  /// memories.
  struct Context
  {
    struct Memory
    {
      struct SoftAnchor
      {
        std::string id;
        double strength = 0.0;
        double likelihood = 0.0;
        std::string label;
        std::string tier;
        double score = 0.0;
        double margin = 0.0;
        double entropy = 0.0;
      };

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
      std::vector<SoftAnchor> soft_anchors;
    };

    std::vector<Memory> working_memory;    ///< Active WM slots (conversation context)
    std::vector<Memory> retrieved_memory;  ///< Long-term retrieval results (injected context)
    std::vector<float> embedding;          ///< Canonical embedding used for this process call.
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
    /// @brief Boundary trigger label when at_boundary (e.g. explicit_turn,
    /// timeout, drift). Empty when no boundary was taken.
    std::optional<std::string> boundary_type;
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
    bool signal_filter_audio_enabled = true;
    bool signal_filter_image_enabled = true;
    bool signal_filter_text_enabled = false;
  };

  /// @brief Optional source media payload to persist with an audio/image signal.
  ///
  /// Processing still uses the canonical audio/image input passed to
  /// ProcessAudio/ProcessImage. When Media contains data, those bytes and this
  /// MIME type become the durable payload instead of the canonical processing
  /// representation. Media bytes are copied during the call.
  struct Media
  {
    const std::uint8_t *data = nullptr;
    std::size_t size = 0;
    std::string mimetype;
  };

  /// @brief Factory to create a Cortext instance.
  /// @param cfg Three-knob configuration.
  /// @param db_path SQLite database path (e.g. ":memory:" or a file path).
  /// @return A unique_ptr to a Cortext instance.
  static std::unique_ptr<Cortext> Create (const Config &cfg,
                                          const std::string &db_path);

  /// @brief Factory to create a Cortext instance with a caller-supplied clock.
  static std::unique_ptr<Cortext> Create (const Config &cfg,
                                          const std::string &db_path,
                                          std::shared_ptr<Clock> clock);

  /// @brief Factory using SQLite for metadata and caller-supplied object store.
  static std::unique_ptr<Cortext> Create (
      const Config &cfg, const std::string &db_path,
      std::shared_ptr<ObjectStore> object_store);

  /// @brief Factory using SQLite metadata, caller-supplied object store, and
  /// caller-supplied clock.
  static std::unique_ptr<Cortext> Create (
      const Config &cfg, const std::string &db_path,
      std::shared_ptr<ObjectStore> object_store, std::shared_ptr<Clock> clock);

  /// @brief Factory to create a Cortext instance using a caller-supplied store.
  /// @param cfg Three-knob configuration.
  /// @param store Caller-owned database store. Cortext keeps a shared reference
  /// and does not close externally supplied stores.
  /// @return A unique_ptr to a Cortext instance.
  static std::unique_ptr<Cortext> Create (const Config &cfg,
                                          std::shared_ptr<Store> store);

  /// @brief Factory with caller-supplied database store and clock.
  static std::unique_ptr<Cortext> Create (const Config &cfg,
                                          std::shared_ptr<Store> store,
                                          std::shared_ptr<Clock> clock);

  /// @brief Factory with caller-supplied database and object stores.
  /// @param cfg Three-knob configuration.
  /// @param store Caller-owned database store.
  /// @param object_store Caller-owned content-addressed object store.
  /// @return A unique_ptr to a Cortext instance.
  static std::unique_ptr<Cortext> Create (
      const Config &cfg, std::shared_ptr<Store> store,
      std::shared_ptr<ObjectStore> object_store);

  /// @brief Factory with caller-supplied database store, object store, and
  /// clock.
  static std::unique_ptr<Cortext> Create (
      const Config &cfg, std::shared_ptr<Store> store,
      std::shared_ptr<ObjectStore> object_store, std::shared_ptr<Clock> clock);

  /// @brief Process text input.
  /// @param text The text content to process.
  /// @param source_id Opaque source stream identifier used for provenance and
  /// exact same-source grouping, not for semantic behavior (e.g.,
  /// "conversation/main").
  /// @param retention Natural by default; episode algorithms decide. Pass
  /// Durable for explicit turn commit, Boundary for explicit turn edge only,
  /// or Ephemeral for no-storage query.
  /// @return Context with retrieved memories and processing output.
  Context ProcessText (const std::string &text, const std::string &source_id,
                       Retention retention = Retention::Natural);

  /// @brief Process text input with an explicit timestamp (ms since epoch).
  /// @param text The text content to process.
  /// @param source_id Opaque source stream identifier used for provenance and
  /// exact same-source grouping, not for semantic behavior (e.g.,
  /// "conversation/main").
  /// @param timestamp Milliseconds since Unix epoch (must be monotonic).
  /// @param retention Natural by default; episode algorithms decide. Pass
  /// Durable for explicit turn commit, Boundary for explicit turn edge only,
  /// or Ephemeral for no-storage query.
  /// @return Context with retrieved memories and processing output.
  Context ProcessTextAt (const std::string &text, const std::string &source_id,
                         std::uint64_t timestamp,
                         Retention retention = Retention::Natural);


  /// @brief Process audio PCM input.
  /// @param pcm Pointer to PCM float samples (16kHz mono expected).
  /// @param num_samples Number of samples.
  /// @param source_id Opaque source stream identifier used for provenance and
  /// exact same-source grouping, not for semantic behavior.
  /// @param retention Natural by default; episode algorithms decide. Pass
  /// Durable for explicit turn commit, Boundary for explicit turn edge only,
  /// or Ephemeral for no-storage query.
  /// @return Context with retrieved memories and processing output.
  Context ProcessAudio (const float *pcm, std::size_t num_samples,
                        const std::string &source_id,
                        Retention retention = Retention::Natural);

  /// @brief Process audio PCM while storing caller-provided source media.
  ///
  /// The PCM buffer remains the canonical processing input. If media contains
  /// bytes, those bytes are persisted as the signal/memory payload using
  /// media.mimetype. If media is empty, no audio payload is persisted.
  Context ProcessAudio (const float *pcm, std::size_t num_samples,
                        const std::string &source_id, const Media &media,
                        Retention retention = Retention::Natural);

  /// @brief Process image input.
  /// @param data Raw pixel data (RGB/RGBA).
  /// @param width Image width.
  /// @param height Image height.
  /// @param channels Number of channels (3 for RGB, 4 for RGBA).
  /// @param source_id Opaque source stream identifier used for provenance and
  /// exact same-source grouping, not for semantic behavior.
  /// @param retention Natural by default; episode algorithms decide. Pass
  /// Durable for explicit turn commit, Boundary for explicit turn edge only,
  /// or Ephemeral for no-storage query.
  /// @return Context with retrieved memories and processing output.
  Context ProcessImage (const std::uint8_t *data, int width, int height,
                        int channels, const std::string &source_id,
                        Retention retention = Retention::Natural);

  /// @brief Process image pixels while storing caller-provided source media.
  ///
  /// The pixel buffer remains the canonical processing input. If media contains
  /// bytes, those bytes are persisted as the signal/memory payload using
  /// media.mimetype. If media is empty, no image payload is persisted.
  Context ProcessImage (const std::uint8_t *data, int width, int height,
                        int channels, const std::string &source_id,
                        const Media &media,
                        Retention retention = Retention::Natural);

  /// @brief Encode text and return the configured encoder's raw embedding.
  ///
  /// This is an embedding-only call: it does not run signal filters, mutate
  /// processor state, write to storage, retrieve memories, or trigger
  /// consolidation.
  std::vector<float> EmbedText (const std::string &text) const;

  /// @brief Encode audio PCM and return the configured encoder's raw embedding.
  ///
  /// Audio uses the same public input contract as ProcessAudio: 16 kHz mono
  /// float32 PCM. The call is embedding-only and does not mutate memory state.
  std::vector<float> EmbedAudio (const float *pcm,
                                 std::size_t num_samples) const;

  /// @brief Encode image pixels and return the configured encoder's raw embedding.
  ///
  /// Image data uses the same public input contract as ProcessImage: row-major
  /// RGB/RGBA-style bytes with explicit width, height, and channel count. The
  /// call is embedding-only and does not mutate memory state.
  std::vector<float> EmbedImage (const std::uint8_t *data, int width,
                                 int height, int channels) const;

  /// @brief Attempt to trigger consolidation if conditions allow.
  Context Consolidate ();
  Context Consolidate (StopToken stop_token);

  /// @brief Flush/commit any pending episode writes.
  void Flush ();

  /// @brief Reset volatile in-memory processor state.
  ///
  /// Durable memories, signals, embeddings, and object-store content are
  /// retained. The processor is rebuilt against the same store and model
  /// dependencies, reloading persisted adaptive state as a newly created
  /// handle would.
  void Reset ();

#if defined(CORTEXT_TESTING)
  /// @brief Test-only helper to hydrate arbitrary memory ids.
  Context DebugHydrateForTest (const std::vector<long long> &candidate_ids,
                               const std::vector<long long> &used_ids);
  Context DebugHydrateForTest (const std::vector<long long> &candidate_ids,
                               const std::vector<long long> &used_ids,
                               const std::vector<float> &query_embedding);
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
  friend class internal::ReplayIngress;
  friend class internal::StreamingTextProbeSession;
  struct Impl;
  explicit Cortext (const Config &cfg, const std::string &db_path,
                    std::shared_ptr<Clock> clock);
  explicit Cortext (const Config &cfg, const std::string &db_path,
                    std::shared_ptr<ObjectStore> object_store);
  explicit Cortext (const Config &cfg, const std::string &db_path,
                    std::shared_ptr<ObjectStore> object_store,
                    std::shared_ptr<Clock> clock);
  explicit Cortext (const Config &cfg, std::shared_ptr<Store> store,
                    std::shared_ptr<Clock> clock);
  explicit Cortext (const Config &cfg, std::shared_ptr<Store> store,
                    std::shared_ptr<ObjectStore> object_store);
  explicit Cortext (const Config &cfg, std::shared_ptr<Store> store,
                    std::shared_ptr<ObjectStore> object_store,
                    std::shared_ptr<Clock> clock);
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext
