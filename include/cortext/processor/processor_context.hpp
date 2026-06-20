#pragma once

#include "cortext/data/centroids.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/processor/accumulator_state.hpp"
#include "cortext/signal.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cortext
{

struct BoundaryDiagnostics
{
  double coherence_prev = 0.0;
  double coherence_curr = 1.0;
  double coh_drop = 0.0;
  double coh_drop_norm = 0.0;
  double d_step = 0.0;
  double eta_prev = 0.0;
  double drift_spike = 0.0;
  double drift_norm = 0.0;
  double surprisal_raw = 0.0;
  double surprisal_norm = 0.0;
  double topic_shift = 0.0;
  double topic_norm = 0.0;
  double boundary_rate_ema = 0.0;
  double boundary_rate_mult = 1.0;
  double boundary_threshold = 0.0;
  double boundary_floor = 0.0;
  double boundary_score = 0.0;
  bool should_flush = false;
};

/// @brief Holds the long-lived, evolving state of the SignalProcessor.
///
/// This includes all dynamic variables such as EWMAs, rolling windows,
/// dynamic thresholds, and counters that persist across multiple signals.
///
/// State is organized into logical groups:
/// - General: signals_processed, u_t, initialization flags
/// - Focus: weight_relevance, attention_width, recent_context (Alg 1, 2, 15)
/// - Sensitivity: base_rate, emotion weights/centroids, rate_target (Alg 3, 4, 16)
/// - Stability: half_lives, hysteresis, retention history (Alg 5, 6, 17)
/// - Threshold: T_dynamic, recent_scores, rate control (Alg 8)
/// - Blender: metric weights, RLS state (Alg 7)
/// - Consolidation: last_consolidation_ts, last_retrieval_ts (Alg 28)
/// - Influence: sustained_influence (Alg 19)
/// - Working Memory: wm_slots, acceptance state (Alg 24)
///
/// Note: This struct could be refactored into smaller sub-structs (e.g.,
/// FocusState, SensitivityState, StabilityState) to improve encapsulation,
/// but this would be a breaking change requiring updates to all operations.
struct ProcessorContext
{
  // --- Observed write-rate window ---
  class WriteRateWindow
  {
  public:
    WriteRateWindow () : capacity_ (30) {}
    explicit WriteRateWindow (size_t cap) : capacity_ (cap) {}
    void
    SetCapacity (size_t cap)
    {
      capacity_ = (cap == 0) ? 1 : cap;
      if (!timestamps_.empty ())
        {
          const uint64_t latest = timestamps_.back ();
          const uint64_t window_ms = static_cast<uint64_t> (capacity_) * 1000ULL;
          while (!timestamps_.empty ()
                 && latest > timestamps_.front ()
                        && (latest - timestamps_.front ()) > window_ms)
            {
              timestamps_.pop_front ();
            }
        }
      cached_rate_.reset ();
    }
    void
    Record (uint64_t ts)
    {
      if (!timestamps_.empty () && ts <= timestamps_.back ())
        {
          return; // ignore non-increasing timestamps
        }
      timestamps_.push_back (ts);
      const uint64_t window_ms = static_cast<uint64_t> (capacity_) * 1000ULL;
      while (!timestamps_.empty ()
             && ts > timestamps_.front ()
                    && (ts - timestamps_.front ()) > window_ms)
        {
          timestamps_.pop_front ();
        }
      cached_rate_.reset ();
    }
    double
    RatePerMinute (double alpha) const
    {
      if (cached_rate_.has_value ())
        {
          return *cached_rate_;
        }
      if (timestamps_.size () < 2)
        {
          cached_rate_ = 0.0;
          return 0.0;
        }
      // Compute EWMA of intervals (seconds)
      constexpr double kMillisToSeconds = 1e-3;
      constexpr double kMinInterval = 0.1;
      constexpr double kMaxInterval = 10.0;
      const double a = (alpha <= 0.0) ? 0.25 : (alpha >= 1.0 ? 1.0 : alpha);
      double ema = 0.0;
      bool first = true;
      for (size_t i = 1; i < timestamps_.size (); ++i)
        {
          double dt = static_cast<double> (timestamps_[i] - timestamps_[i - 1])
                      * kMillisToSeconds;
          if (dt < kMinInterval)
            dt = kMinInterval;
          if (dt > kMaxInterval)
            dt = kMaxInterval;
          if (first)
            {
              ema = dt;
              first = false;
            }
          else
            {
              ema = (1.0 - a) * ema + a * dt;
            }
        }
      const double eps = 1e-9;
      const double rate = (ema > eps) ? (60.0 / ema) : 0.0;
      cached_rate_ = std::isfinite (rate) ? rate : 0.0;
      return *cached_rate_;
    }
    std::vector<uint64_t>
    GetTimestamps () const
    {
      return std::vector<uint64_t> (timestamps_.begin (), timestamps_.end ());
    }
    void
    SetTimestamps (const std::vector<uint64_t> &timestamps)
    {
      timestamps_.assign (timestamps.begin (), timestamps.end ());
      if (!timestamps_.empty ())
        {
          const uint64_t latest = timestamps_.back ();
          const uint64_t window_ms = static_cast<uint64_t> (capacity_) * 1000ULL;
          while (!timestamps_.empty ()
                 && latest > timestamps_.front ()
                        && (latest - timestamps_.front ()) > window_ms)
            {
              timestamps_.pop_front ();
            }
        }
      cached_rate_.reset ();
    }

  private:
    std::deque<uint64_t> timestamps_;
    size_t capacity_;
    mutable std::optional<double> cached_rate_;
  };

  // --- Recent ID LRU for novelty (Algorithm 27 helper) ---
  class RecentIdsLru
  {
  public:
    explicit RecentIdsLru (size_t cap = 1024) : capacity_ (cap) {}
    void
    SetCapacity (size_t cap)
    {
      capacity_ = (cap == 0) ? 1 : cap;
      while (queue_.size () > capacity_)
        {
          const long long old = queue_.front ();
          queue_.pop_front ();
          set_.erase (old);
        }
    }
    void
    RecordIds (const std::vector<long long> &ids)
    {
      for (const long long id : ids)
        {
          if (set_.find (id) != set_.end ())
            {
              continue;
            }
          set_.insert (id);
          queue_.push_back (id);
          if (queue_.size () > capacity_)
            {
              const long long old = queue_.front ();
              queue_.pop_front ();
              set_.erase (old);
            }
        }
    }
    const std::unordered_set<long long> &
    GetIdSet () const
    {
      return set_;
    }
    size_t
    Size () const
    {
      return set_.size ();
    }

  private:
    std::unordered_set<long long> set_;
    std::deque<long long> queue_;
    size_t capacity_;
  };

  // ======================================================================
  // General State
  // ======================================================================
  int signals_processed = 0;
  double u_t = 0.0;
  double outcome_pred = 0.0;
  double delta_reward = 0.0;
  double neuromod_ach = 0.0;
  double neuromod_ne = 0.0;
  double neuromod_da = 0.0;
  double encode_bias = 0.5;
  double retrieval_bias = 0.5;
  double osc_phase = 0.0;
  double last_used_rate = 0.0;
  double last_used_flag = 0.0;
  bool focus_priors_initialized = false;
  bool sensitivity_priors_initialized = false;
  bool stability_priors_initialized = false;
  int last_interrupt_tick = -1000000;
  int blender_update_count = 0;
  WriteRateWindow write_rate_window_;
  RecentIdsLru recent_ids_lru_;

  // ======================================================================
  // Embedding Prediction Error State (Section 3.1.4)
  // ======================================================================
  /// @brief Previous signal embedding (optional utility)
  std::optional<Eigen::VectorXf> last_embedding;
  /// @brief EMA expectation state for prediction error
  std::optional<Eigen::VectorXf> x_pred_ema;
  /// @brief Prediction error SSE history for ΔSSE utility
  std::optional<double> prediction_error_sse;
  std::optional<double> prediction_error_sse_prev;

  // ======================================================================
  // Focus-Related State (Algorithms 1, 2, 15)
  // ======================================================================
  double weight_relevance_prior = 0.5;
  double coverage_gain_floor_prior = 0.65;
  double mismatch_weight_prior = 0.5;
  double attention_width_prior = 1.57;
  double weight_relevance = 0.5;
  double attention_width = 1.57;
  double coverage_gain_floor = 0.65;
  double mismatch_weight = 0.5;
  std::deque<Eigen::VectorXf> recent_context_embeddings;

  /// @brief Embedding ids whose persisted memories currently have non-zero
  /// predictive pre-activation and therefore need per-turn decay.
  std::unordered_set<long long> predictive_pre_activation_embedding_ids;
  /// @brief Embedding ids whose persisted memories currently carry retrieval
  /// induced forgetting suppression and therefore need recovery updates.
  std::unordered_set<long long> retrieval_suppression_embedding_ids;

  // ======================================================================
  // Sensitivity-Related State (Algorithms 3, 4, 16)
  // ======================================================================
  double base_rate_prior = 0.2;
  double weight_novelty_prior = 0.3;
  double weight_surprise_prior = 0.2;
  double weight_valence_prior = 0.4;
  double weight_arousal_prior = 0.0;
  double weight_emotion_prior = 0.2;
  double emotion_gain_prior = 1.0;
  double score_gain_prior = 1.0;
  double rate_target_prior = 0.2;
  std::optional<data::Centroids> centroids;
  double rate_target = 0.0;
  uint64_t last_signal_timestamp = 0;
  double weight_novelty = 0.3;
  double weight_surprise = 0.2;
  double weight_valence = 0.4;
  double weight_arousal = 0.0;
  double emotion_gain = 1.0;
  double score_gain = 1.0;

  // Emotion state (Algorithm 4, persisted EWMA values)
  double emotion_intensity_ewma = 0.0;
  double flashbulb_rate_ewma = 0.0;
  double valence_ewma = 0.5;
  double arousal_ewma = 0.0;
  std::deque<double> recent_emotion_intensities;

  // Mood state (Algorithm 4b, persisted tonic state)
  // Order: [anger, fear, joy, love, sadness, surprise]
  std::array<double, 6> mood_vector = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  uint64_t last_mood_ts = 0;

  // ======================================================================
  // Stability-Related State (Algorithms 5, 6, 17)
  // ======================================================================
  double hysteresis_band_prior = 0.02;
  double half_life_prior = 120.0;
  double rate_decay_prior = 0.60;
  double periphery_half_life_prior = 120.0;
  double salience_half_life_prior = 120.0;
  double drift_weight_prior = 0.5;
  std::deque<double> observed_retention_history;
  double half_life = 120.0;
  double rate_decay = 0.60;
  double periphery_half_life = 120.0;
  double salience_half_life = 120.0;
  double drift_weight = 0.5;
  double retention_ema = 0.0;
  double delta_half_life_adj = 0.0;

  // ======================================================================
  // Threshold & Score Tracking State (Algorithm 8)
  // ======================================================================
  std::deque<double> recent_scores;
  double T_dynamic = 0.2;
  double T_target = 0.2;
  double hysteresis = 0.05;
  double dt_ema = 0.0;
  double m_rate = 0.0;
  double rho_hat_prev = 0.0;
  int rate_ticks = 0;
  uint64_t last_rate_timestamp = 0;
  double reliability = 1.0;  // ESS-based reliability measure
  double boundary_rate_ema = 0.0;

  // ======================================================================
  // Metric Weight Blending State (Algorithm 7)
  // ======================================================================
  std::unordered_map<operations::Metric, double> blender_state;
  std::vector<operations::Metric> blender_order;
  std::vector<std::vector<double> > blender_P;
  bool blender_ready = false;

  // ======================================================================
  // Consolidation State (Algorithms 28, 28b)
  // ======================================================================
  uint64_t last_consolidation_ts = 0;
  uint64_t last_retrieval_ts = 0;
  int consolidation_count = 0;
  int memories_since_consolidation = 0;
  bool is_processing_signal = false;

  // Eviction frontier scans are corpus-size-sensitive maintenance work. They
  // are still run after consolidation and under pressure, but not every turn.
  int last_eviction_scan_signal = -1;
  uint64_t last_eviction_scan_consolidation_ts = 0;

  // Retention aggregates scan the memory table. Keep them periodic so
  // stability adaptation does not add a corpus-size term to every signal.
  int last_retention_scan_signal = -1;
  uint64_t last_retention_scan_consolidation_ts = 0;

  // Storage pressure reads touch filesystem/SQLite metadata and are used by
  // multiple operations in one turn. Cache briefly to avoid per-signal jitter.
  bool storage_pressure_cache_valid = false;
  int storage_pressure_cache_signal = -1;
  std::string storage_pressure_cache_key;
  bool storage_pressure_cache_active = false;
  long long storage_pressure_cache_used_bytes = 0;
  long long storage_pressure_cache_threshold_bytes = 0;

  // ======================================================================
  // Episode Tracking State (Algorithm 12)
  // ======================================================================
  uint64_t episode_start_ts = 0;

  // ======================================================================
  // Influence Feedback State (Algorithm 19)
  // ======================================================================
  double sustained_influence = 0.0;

  // ======================================================================
  // Working Memory State (Algorithm 24, Section 6.1)
  // ======================================================================
  /// @brief Working memory slot holding a coherent memory unit (Section 6.1.1)
  struct WMSlot
  {
    // v2 persistence fields
    int64_t memory_id = 0;               ///< DB row ID (for updates)
    int64_t embedding_id = 0;            ///< Current representative embedding row
    std::string source_id;               ///< Opaque source stream identifier
    std::vector<std::vector<unsigned char>> blob_ids;  ///< Content refs for hydration
    std::string modality = "text";       ///< Content type ("text", "audio", "image")
    int64_t start_ts = 0;                ///< Memory start timestamp (ms)
    bool metadata_dirty = true;          ///< Slot metadata needs DB update
    bool embedding_dirty = true;         ///< Representative embedding changed
    bool signal_records_dirty = true;    ///< Attached signal rows changed
    std::size_t persisted_signal_record_count = 0;

    // Core slot state
    Eigen::VectorXf embedding;           ///< e_rep (representative embedding)
    double strength = 0.0;               ///< Slot activation strength
    double last_ts = 0.0;                ///< Last access timestamp (seconds)
    int pos_index = 0;                   ///< Position index

    // Memory-level metadata (Section 6.1.1)
    int n_signals = 0;                   ///< Number of signals in memory
    double s_max = 0.0;                  ///< Max signal score
    double s_avg = 0.0;                  ///< Average signal score
    double drift_acc = 0.0;              ///< Accumulated drift (D_acc)
    double mem_elapsed = 0.0;            ///< Memory duration in seconds
    double s_emotion_max = 0.0;          ///< Max emotion intensity
    double s_arousal_avg = 0.0;          ///< Average arousal

    std::vector<SignalRecord> signal_records;  ///< Ordered signal records
  };
  std::vector<WMSlot> wm_slots;
  bool wm_slots_dirty = false;           ///< Active slot set changed since persist
  bool wm_last_accepted = false;
  bool wm_last_chunked = false;

  // ======================================================================
  // Memory Accumulation State (Section 4.4)
  // ======================================================================
  /// @brief Per-opaque-source-stream accumulators for memory formation.
  ///
  /// Maps exact source_id → AccumulatorState. Each stream accumulates signals
  /// into coherent memories before making write decisions. This implements
  /// Event Segmentation Theory (Zacks & Swallow, 2007) for grouping
  /// signals into natural "thought units".
  std::unordered_map<std::string, AccumulatorState> accumulator_states;

  /// @brief Stream of written memory representatives (e_rep).
  std::deque<Eigen::VectorXf> memory_stream;

  /// @brief Recent memory centroids for interrupt gate context (Section 8.2)
  ///
  /// Stores μ_acc from recently written memories. Used by interrupt gate
  /// to compute context centroid for novelty and marginal utility.
  /// ctx_window ← recent_memory_centroids (not individual signals)
  std::deque<Eigen::VectorXf> recent_memory_centroids;

  // ======================================================================
  // Soft Anchor State (ingress-time formation)
  // ======================================================================
  struct SoftAnchorState
  {
    std::string anchor_id;
    std::string status = "provisional";
    std::string last_source_id;
    Eigen::VectorXf semantic_centroid;
    Eigen::VectorXf entity_centroid;
    Eigen::VectorXf full_centroid;
    double semantic_radius = 0.0;
    double entity_radius = 0.0;
    double full_radius = 0.0;
    double anchor_strength = 0.0;
    int support_count = 0;
    int contradiction_count = 0;
    int first_step = 0;
    int last_step = 0;
    uint64_t first_ts = 0;
    uint64_t last_ts = 0;
    int64_t last_boundary_id = 0;
    std::deque<long long> recent_memory_ids;
  };

  struct SoftAnchorLink
  {
    long long memory_id = 0;
    std::string anchor_id;
    double anchor_strength = 0.0;
    std::string anchor_label = "none";
    std::string evidence_kind = "unknown";
    std::string memory_tier = "WM";
    double score = 0.0;
    double margin = 0.0;
    double entropy = 0.0;
    int support_count = 0;
    int contradiction_count = 0;
    int created_step = 0;
    int updated_step = 0;
  };

  std::vector<SoftAnchorState> soft_anchor_states;
  std::vector<SoftAnchorLink> soft_anchor_last_links;
  bool soft_anchor_enabled = false;
  int soft_anchor_next_id = 1;
  int soft_anchor_update_count = 0;
  int soft_anchor_last_link_count = 0;
  int soft_anchor_last_state_count = 0;
  int soft_anchor_last_create_count = 0;
  int soft_anchor_last_update_count = 0;
  int soft_anchor_last_none_count = 0;
  double soft_anchor_last_update_us = 0.0;
  double soft_anchor_total_update_us = 0.0;
  std::deque<double> soft_anchor_recent_update_us;

  // ======================================================================
  // Index and Procedural Stores (CLS extensions)
  // ======================================================================
  struct AssociationCacheEntry
  {
    long long memory_id = 0;
    long long embedding_id = 0;
    Eigen::VectorXf embedding;
    float embedding_norm = 0.0f;
    bool is_association = false;
  };

  struct RetrievalSurfaceEntry
  {
    long long memory_id = 0;
    long long embedding_id = 0;
    long long created_at = 0;
    long long start_ts = 0;
    long long last_access = 0;
    long long event_ts = 0;
    long long retrieved_count = 0;
    long long used_count = 0;
    std::string kind;
    std::string source_id;
    std::string modality;
    double source_reliability = -1.0;
    int source_contradiction_count = 0;
    double emotional_intensity = 0.0;
    double arousal_avg = 0.0;
    double pre_activation = 0.0;
    bool is_association = false;
    bool vector_seed_eligible = true;
    Eigen::VectorXf embedding;
    Eigen::VectorXf context_embedding;
    float embedding_norm = 0.0f;
    std::uint32_t seed_visit_epoch = 0;
  };

  struct RetrievalSurfaceSourceIndexEntry
  {
    long long memory_id = 0;
    long long start_ts = 0;

    bool
    operator== (const RetrievalSurfaceSourceIndexEntry &other) const
    {
      return memory_id == other.memory_id && start_ts == other.start_ts;
    }
  };

  struct AssociationFanoutEdge
  {
    long long memory_id = 0;
    long long embedding_id = 0;
    std::string edge_type;
    double weight = 1.0;
    long long last_reinforced = 0;
  };

  struct AssociationFanoutCache
  {
    bool valid = false;
    long long edge_count = 0;
    long long source_sum = 0;
    long long target_sum = 0;
    long long source_max = 0;
    long long target_max = 0;
    long long weight_sum_micros = 0;
    long long last_reinforced_sum = 0;
    int last_fingerprint_check_signal = -1;
    int consolidation_count = -1;
    std::unordered_map<long long, std::vector<AssociationFanoutEdge>>
        out_by_source;
    std::unordered_map<long long, std::vector<AssociationFanoutEdge>>
        in_by_target;
  };

  void
  UpsertAssociationCache (long long memory_id, long long embedding_id,
                          const Eigen::VectorXf &embedding,
                          bool is_association)
  {
    if (memory_id <= 0 || embedding_id <= 0 || embedding.size () == 0)
      {
        return;
      }
    const float norm = embedding.norm ();
    if (norm <= 1e-9f)
      {
        return;
      }
    auto it = association_cache_index.find (memory_id);
    if (it != association_cache_index.end ())
      {
        auto &entry = association_cache[it->second];
        entry.embedding_id = embedding_id;
        entry.embedding = embedding;
        entry.embedding_norm = norm;
        entry.is_association = is_association;
        return;
      }
    AssociationCacheEntry entry;
    entry.memory_id = memory_id;
    entry.embedding_id = embedding_id;
    entry.embedding = embedding;
    entry.embedding_norm = norm;
    entry.is_association = is_association;
    association_cache_index[memory_id] = association_cache.size ();
    association_cache.push_back (std::move (entry));
  }

  void
  UpsertRetrievalSurface (RetrievalSurfaceEntry entry)
  {
    if (entry.memory_id <= 0 || entry.embedding_id <= 0
        || entry.embedding.size () == 0)
      {
        return;
      }
    entry.embedding_norm = entry.embedding.norm ();
    if (entry.embedding_norm <= 1e-9f)
      {
        return;
      }
    entry.is_association = (entry.kind == "ASSOCIATION");
    auto it = retrieval_surface_index.find (entry.memory_id);
    if (it != retrieval_surface_index.end ())
      {
        auto &existing = retrieval_surface_cache[it->second];
        const bool reindex_source
            = RetrievalSurfaceSourceIndexable (existing)
              != RetrievalSurfaceSourceIndexable (entry)
              || existing.source_id != entry.source_id
              || existing.start_ts != entry.start_ts;
        if (reindex_source && RetrievalSurfaceSourceIndexable (existing))
          {
            RemoveRetrievalSurfaceSourceIndex (existing.memory_id,
                                               existing.source_id);
          }
        retrieval_surface_embedding_index.erase (existing.embedding_id);
        existing = std::move (entry);
        retrieval_surface_embedding_index[existing.embedding_id] = it->second;
        if (reindex_source && RetrievalSurfaceSourceIndexable (existing))
          {
            AddRetrievalSurfaceSourceIndex (existing);
          }
        return;
      }
    const size_t index = retrieval_surface_cache.size ();
    retrieval_surface_index[entry.memory_id] = index;
    retrieval_surface_embedding_index[entry.embedding_id] = index;
    retrieval_surface_cache.push_back (std::move (entry));
    AddRetrievalSurfaceSourceIndex (retrieval_surface_cache.back ());
  }

  void
  RemoveRetrievalSurface (long long memory_id)
  {
    auto it = retrieval_surface_index.find (memory_id);
    if (it == retrieval_surface_index.end ())
      {
        return;
      }
    const size_t index = it->second;
    const long long removed_memory_id
        = retrieval_surface_cache[index].memory_id;
    const std::string removed_source_id
        = retrieval_surface_cache[index].source_id;
    const bool removed_source_indexable
        = RetrievalSurfaceSourceIndexable (retrieval_surface_cache[index]);
    if (removed_source_indexable)
      {
        RemoveRetrievalSurfaceSourceIndex (removed_memory_id,
                                           removed_source_id);
      }
    retrieval_surface_embedding_index.erase (
        retrieval_surface_cache[index].embedding_id);
    const size_t last = retrieval_surface_cache.size () - 1;
    if (index != last)
      {
        retrieval_surface_cache[index]
            = std::move (retrieval_surface_cache[last]);
        retrieval_surface_index[retrieval_surface_cache[index].memory_id]
            = index;
        retrieval_surface_embedding_index[
            retrieval_surface_cache[index].embedding_id]
            = index;
      }
    retrieval_surface_cache.pop_back ();
    retrieval_surface_index.erase (it);
  }

  static bool
  RetrievalSurfaceSourceIndexable (const RetrievalSurfaceEntry &entry)
  {
    return entry.memory_id > 0 && entry.start_ts > 0
           && !entry.source_id.empty () && !entry.is_association
           && entry.kind != "WORKING";
  }

  void
  AddRetrievalSurfaceSourceIndex (const RetrievalSurfaceEntry &entry)
  {
    if (!RetrievalSurfaceSourceIndexable (entry))
      {
        return;
      }
    auto &entries = retrieval_surface_source_index[entry.source_id];
    if (!entries.empty ())
      {
        const auto &last_entry = entries.back ();
        if (entry.start_ts < last_entry.start_ts)
          {
            retrieval_surface_source_index_dirty.insert (entry.source_id);
          }
      }
    entries.push_back ({ entry.memory_id, entry.start_ts });
  }

  void
  RemoveRetrievalSurfaceSourceIndex (long long memory_id,
                                     const std::string &source_id)
  {
    if (memory_id <= 0 || source_id.empty ())
      {
        return;
      }
    auto it = retrieval_surface_source_index.find (source_id);
    if (it == retrieval_surface_source_index.end ())
      {
        return;
      }
    auto &entries = it->second;
    entries.erase (
        std::remove_if (
            entries.begin (), entries.end (),
            [&] (const RetrievalSurfaceSourceIndexEntry &entry) {
              return entry.memory_id == memory_id;
            }),
        entries.end ());
    if (entries.empty ())
      {
        retrieval_surface_source_index.erase (it);
        retrieval_surface_source_index_dirty.erase (source_id);
      }
  }

  void
  SortRetrievalSurfaceSourceIndex (const std::string &source_id)
  {
    auto it = retrieval_surface_source_index.find (source_id);
    if (it == retrieval_surface_source_index.end ())
      {
        retrieval_surface_source_index_dirty.erase (source_id);
        return;
      }
    auto &entries = it->second;
    std::sort (
        entries.begin (), entries.end (),
        [] (const RetrievalSurfaceSourceIndexEntry &a,
            const RetrievalSurfaceSourceIndexEntry &b) {
      if (a.start_ts != b.start_ts)
        {
          return a.start_ts < b.start_ts;
        }
      return a.memory_id < b.memory_id;
    });
    entries.erase (
        std::unique (entries.begin (), entries.end ()),
        entries.end ());
    retrieval_surface_source_index_dirty.erase (source_id);
  }

  void
  SortRetrievalSurfaceSourceIndexes ()
  {
    for (auto &entry : retrieval_surface_source_index)
      {
        std::sort (entry.second.begin (), entry.second.end (),
                   [] (const RetrievalSurfaceSourceIndexEntry &a,
                       const RetrievalSurfaceSourceIndexEntry &b) {
          if (a.start_ts != b.start_ts)
            {
              return a.start_ts < b.start_ts;
            }
          return a.memory_id < b.memory_id;
        });
        entry.second.erase (
            std::unique (entry.second.begin (), entry.second.end ()),
            entry.second.end ());
      }
    retrieval_surface_source_index_dirty.clear ();
  }

  void
  EnsureRetrievalSurfaceSourceIndexSorted (const std::string &source_id)
  {
    if (retrieval_surface_source_index_dirty.find (source_id)
        == retrieval_surface_source_index_dirty.end ())
      {
        return;
      }
    SortRetrievalSurfaceSourceIndex (source_id);
  }

  void
  UpdateRetrievalSurfacePreActivationByEmbedding (long long embedding_id,
                                                  double pre_activation)
  {
    auto it = retrieval_surface_embedding_index.find (embedding_id);
    if (it == retrieval_surface_embedding_index.end ())
      {
        return;
      }
    retrieval_surface_cache[it->second].pre_activation = pre_activation;
  }

  void
  DecayRetrievalSurfacePreActivationByEmbedding (long long embedding_id,
                                                 double pad)
  {
    auto it = retrieval_surface_embedding_index.find (embedding_id);
    if (it == retrieval_surface_embedding_index.end ())
      {
        return;
      }
    auto &entry = retrieval_surface_cache[it->second];
    entry.pre_activation = std::max (0.0, entry.pre_activation * pad);
    if (entry.pre_activation <= 1e-6)
      {
        entry.pre_activation = 0.0;
      }
  }

  void
  BoostRetrievalSurfacePreActivationByEmbedding (long long embedding_id,
                                                 double pad, double delta)
  {
    auto it = retrieval_surface_embedding_index.find (embedding_id);
    if (it == retrieval_surface_embedding_index.end ())
      {
        return;
      }
    auto &entry = retrieval_surface_cache[it->second];
    entry.pre_activation = std::min (
        1.0, std::max (0.0, entry.pre_activation) * pad + delta);
  }

  void
  UpdateRetrievalSurfaceUsageByEmbedding (long long embedding_id,
                                          long long retrieved_count,
                                          long long used_count,
                                          long long last_access)
  {
    auto it = retrieval_surface_embedding_index.find (embedding_id);
    if (it == retrieval_surface_embedding_index.end ())
      {
        return;
      }
    auto &entry = retrieval_surface_cache[it->second];
    entry.retrieved_count = retrieved_count;
    entry.used_count = used_count;
    entry.last_access = last_access;
  }

  void
  UpdateRetrievalSurfaceUsageByMemory (long long memory_id,
                                       long long retrieved_count,
                                       long long used_count,
                                       long long last_access)
  {
    auto it = retrieval_surface_index.find (memory_id);
    if (it == retrieval_surface_index.end ())
      {
        return;
      }
    auto &entry = retrieval_surface_cache[it->second];
    entry.retrieved_count = retrieved_count;
    entry.used_count = used_count;
    entry.last_access = last_access;
  }

  std::unordered_map<std::string, std::vector<long long>> index_store;
  std::unordered_map<long long, std::string> index_reverse;
  std::unordered_map<std::string, std::unordered_map<long long, double>> procedural_store;
  std::vector<AssociationCacheEntry> association_cache;
  std::unordered_map<long long, size_t> association_cache_index;
  std::vector<RetrievalSurfaceEntry> retrieval_surface_cache;
  std::uint32_t retrieval_surface_seed_visit_epoch = 0;
  std::unordered_map<long long, size_t> retrieval_surface_index;
  std::unordered_map<long long, size_t> retrieval_surface_embedding_index;
  std::unordered_map<std::string,
                     std::vector<RetrievalSurfaceSourceIndexEntry>>
      retrieval_surface_source_index;
  std::unordered_set<std::string> retrieval_surface_source_index_dirty;
  AssociationFanoutCache association_fanout_cache;

};

} // namespace cortext
