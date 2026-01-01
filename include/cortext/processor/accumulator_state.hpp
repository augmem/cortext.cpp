#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <vector>

namespace cortext
{

/**
 * @brief Individual signal record for SIGNALS table population (Section 4.4)
 *
 * Tracks per-signal metadata during memory accumulation for later persistence.
 */
struct SignalRecord
{
  Eigen::VectorXf embedding;              ///< Signal embedding (256d)
  uint64_t timestamp = 0;                 ///< Signal arrival time (ms since epoch)
  std::string modality;                   ///< "text" | "audio" | "image"
  std::string mime;                       ///< MIME type
  std::vector<unsigned char> blob_id;     ///< objstore hash (may be empty)
  double score = 0.0;                     ///< Composite score
  int serial_position = 0;                ///< Order within memory (0-based)
};

/**
 * @brief Per-source-stream memory accumulator state (Section 4.4)
 *
 * Maintains running statistics for grouping signals into coherent memories.
 * Each source_id has its own accumulator.
 */
struct AccumulatorState
{
  int64_t episode_id = 0;       ///< FK to episodes table (v2 schema)
  Eigen::VectorXf mu_acc;       ///< Running mean embedding (256d)
  Eigen::VectorXf c_t;          ///< Temporal context vector (slow drift)
  double drift_acc = 0.0;       ///< Accumulated drift within group (D_acc)
  double s_sum = 0.0;           ///< Sum of signal scores in group
  double s_max = 0.0;           ///< Max signal score in group
  int n_signals = 0;            ///< Count of signals in group
  Eigen::VectorXf e_peak;       ///< Accumulator centroid at highest score
  uint64_t t_start = 0;         ///< Timestamp of accumulation start
  uint64_t last_write_ts = 0;   ///< Timestamp of last write (for refractory)
  uint64_t last_signal_ts = 0;  ///< Timestamp of previous signal (for gap)
  double eta_acc = 0.0;         ///< Drift EWMA (η_acc)
  double coherence_prev = 0.0;  ///< Previous coherence value
  std::vector<Eigen::VectorXf> acc_signals_window;  ///< Coherence ring buffer

  // Boundary calibration (local normalization within episode)
  double boundary_surprisal_mean = 0.0;
  double boundary_surprisal_var = 0.0;
  double boundary_drift_spike_mean = 0.0;
  double boundary_drift_spike_var = 0.0;
  double boundary_coh_drop_mean = 0.0;
  double boundary_coh_drop_var = 0.0;
  double boundary_topic_shift_mean = 0.0;
  double boundary_topic_shift_var = 0.0;

  // Emotional metadata (Section 6.1.1)
  double s_emotion_max = 0.0;   ///< Peak emotion intensity in memory
  double s_arousal_sum = 0.0;   ///< Sum of arousal values (for computing avg)

  // Signal tracking for SIGNALS table (Section 4.4)
  std::vector<SignalRecord> signals;  ///< Tracked signals for persistence

  // v2 aggregated blob tracking (working-memory.plan.md Section 2)
  std::vector<std::vector<unsigned char>> blob_ids;  ///< Aggregated objstore refs
  std::string primary_modality;                       ///< "text"|"audio"|"image"

  // Streaming integration (Section 10)
  Eigen::VectorXf prev_x;           ///< Previous accumulator centroid for refractory
  Eigen::VectorXf x_last_check;     ///< Accumulator centroid at last pacing check
  double drift_accum = 0.0;         ///< Cumulative drift since last interrupt
  double drift_at_last_interrupt = 0.0;  ///< Snapshot for refractory delta
  double drift_acc_pacing = 0.0;    ///< Drift accumulator for pacing gate
  bool pending_interrupt_abort = false;  ///< Pending interrupt-induced abort

  /**
   * @brief Reset accumulator for new memory accumulation
   * @param first_embedding Initial embedding for the new memory
   * @param timestamp Current timestamp
   */
  void
  Reset (const Eigen::VectorXf &first_embedding, uint64_t timestamp)
  {
    mu_acc = first_embedding;
    e_peak = first_embedding;
    c_t = first_embedding;
    drift_acc = 0.0;
    s_sum = 0.0;
    s_max = 0.0;
    n_signals = 1;
    t_start = timestamp;
    last_signal_ts = timestamp;
    eta_acc = 0.0;
    coherence_prev = 0.0;
    boundary_surprisal_mean = 0.0;
    boundary_surprisal_var = 0.0;
    boundary_drift_spike_mean = 0.0;
    boundary_drift_spike_var = 0.0;
    boundary_coh_drop_mean = 0.0;
    boundary_coh_drop_var = 0.0;
    boundary_topic_shift_mean = 0.0;
    boundary_topic_shift_var = 0.0;
    // Reset emotional metadata
    s_emotion_max = 0.0;
    s_arousal_sum = 0.0;
    // Clear signal tracking
    signals.clear ();
    // Clear v2 blob tracking
    blob_ids.clear ();
    primary_modality.clear ();
    acc_signals_window.clear ();
    pending_interrupt_abort = false;
    // Note: last_write_ts is preserved across accumulations
  }

  /**
   * @brief Reset accumulator state after a boundary (no new signal yet).
   * @param timestamp Current timestamp (ms since epoch)
   */
  void
  ResetForNextUnit (uint64_t timestamp)
  {
    mu_acc = Eigen::VectorXf ();
    e_peak = Eigen::VectorXf ();
    drift_acc = 0.0;
    s_sum = 0.0;
    s_max = 0.0;
    n_signals = 0;
    t_start = timestamp;
    last_signal_ts = timestamp;
    eta_acc = 0.0;
    coherence_prev = 0.0;
    boundary_surprisal_mean = 0.0;
    boundary_surprisal_var = 0.0;
    boundary_drift_spike_mean = 0.0;
    boundary_drift_spike_var = 0.0;
    boundary_coh_drop_mean = 0.0;
    boundary_coh_drop_var = 0.0;
    boundary_topic_shift_mean = 0.0;
    boundary_topic_shift_var = 0.0;
    s_emotion_max = 0.0;
    s_arousal_sum = 0.0;
    signals.clear ();
    blob_ids.clear ();
    primary_modality.clear ();
    acc_signals_window.clear ();
    pending_interrupt_abort = false;
  }

  /**
   * @brief Accumulate a new signal into the memory (embedding + drift only)
   * @param embedding Signal embedding
   * @param drift Drift magnitude of the signal
   *
   * Updates running mean: μ_acc = ((n-1) × μ_acc + x_t) / n
   * Accumulates drift: D_acc += drift_mag_t / 2
   * Score aggregation is handled separately on the accumulator composite.
   */
  void
  Accumulate (const Eigen::VectorXf &embedding, double drift)
  {
    // Update running mean (Welford-style incremental mean)
    if (n_signals == 0)
      {
        mu_acc = embedding;
      }
    else
      {
        // μ_acc = ((n-1) × μ_acc + x_t) / n
        // Equivalent to: μ_acc = μ_acc + (x_t - μ_acc) / n
        mu_acc = mu_acc + (embedding - mu_acc) / static_cast<float> (n_signals + 1);
      }

    // Accumulate drift
    drift_acc += drift * 0.5;

    n_signals++;
  }
};

} // namespace cortext
