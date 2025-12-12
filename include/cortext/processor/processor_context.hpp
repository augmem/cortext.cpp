#pragma once

#include "cortext/operations/metrics.hpp"
#include "cortext/signal.hpp"
#include <Eigen/Dense>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cortext
{

/// @brief Holds the long-lived, evolving state of the SignalProcessor.
///
/// This includes all dynamic variables such as EWMAs, rolling windows,
/// dynamic thresholds, and counters that persist across multiple signals.
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
      // Trim if needed
      while (timestamps_.size () > capacity_)
        {
          timestamps_.pop_front ();
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
      while (timestamps_.size () > capacity_)
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
      constexpr double kMinInterval = 0.1;
      constexpr double kMaxInterval = 10.0;
      const double a = (alpha <= 0.0) ? 0.25 : (alpha >= 1.0 ? 1.0 : alpha);
      double ema = 0.0;
      bool first = true;
      for (size_t i = 1; i < timestamps_.size (); ++i)
        {
          double dt
              = static_cast<double> (timestamps_[i] - timestamps_[i - 1]);
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

  // --- General State ---
  int signals_processed = 0;
  double u_t = 0.0; // Smoothed uncertainty
  // One-time initialization guards for prior-setting operations.
  bool focus_priors_initialized = false;
  bool sensitivity_priors_initialized = false;
  bool stability_priors_initialized = false;
  // Algorithm 27 refractory tick (last interrupt position)
  int last_interrupt_tick = -1000000;
  // Algorithm 7 RLS update throttling counter
  int blender_update_count = 0;
  // Observed write-rate window
  WriteRateWindow write_rate_window_;
  // Rolling set of recent memory IDs for novelty computation
  RecentIdsLru recent_ids_lru_;

  // --- Focus-Related State (Algs 1 & 2) ---
  // Priors (set once at initialization)
  double weight_relevance_prior = 0.5;
  double coverage_gain_floor_prior = 0.65;
  double mismatch_weight_prior = 0.5;
  double attention_width_prior = 1.57; // PI / 2

  // Dynamic values (updated each signal)
  double weight_relevance = 0.5;
  double attention_width = 1.57;
  std::deque<Eigen::VectorXf> recent_context_embeddings;

  // --- Sensitivity-Related Priors (Alg 3) ---
  double base_rate_prior = 0.2; // writes/min
  double weight_novelty_prior = 0.3;
  double weight_surprise_prior = 0.2;
  double weight_valence_prior = 0.4;
  double weight_arousal_prior = 0.0;
  double weight_emotion_prior = 0.2;
  double emotion_gain_prior = 1.0;
  double score_gain_prior = 1.0;
  double rate_target_prior = 0.2; // writes/min

  // Optional emotion centroids for Algorithm 4
  // (anger,fear,joy,love,sadness,surprise)
  std::vector<Eigen::VectorXf> emotion_centroids; // size 6 if provided

  // --- Sensitivity Dynamic State (Alg 4) ---
  // Estimated target write rate (dynamic) and last-seen timestamp for rate
  // calc
  double rate_target = 0.0; // writes/min (EWMA)
  uint64_t last_signal_timestamp = 0;
  // Dynamic novelty weight (Alg 16 adjusts from usage feedback)
  double weight_novelty = 0.3;

  // --- Stability-Related Priors (Alg 5) ---
  double hysteresis_band_prior = 0.02;
  double half_life_prior = 120.0;
  double rate_decay_prior = 0.60;
  double periphery_half_life_prior = 120.0;
  double salience_half_life_prior = 120.0;
  double drift_weight_prior = 0.5;

  // --- Stability Dynamic State (Alg 6) ---
  // Rolling history of observed retention (seconds)
  std::deque<double> observed_retention_history;
  // Dynamic half-life parameters (seconds)
  double half_life = 120.0;
  double rate_decay = 0.60;
  double periphery_half_life = 120.0;
  double salience_half_life = 120.0;

  // --- Dynamic Threshold State (Alg 8) ---
  // Rolling history of composite scores used for threshold adaptation
  std::deque<double> recent_scores;
  // Current dynamic threshold value and hysteresis band
  double T_dynamic = 0.2;
  double hysteresis = 0.05;

  // --- Algorithm 7 (Metric Weight Blending) State ---
  // Online blender state (RLS with forgetting)
  std::unordered_map<operations::Metric, double> blender_state;
  std::vector<operations::Metric> blender_order;
  std::vector<std::vector<double> > blender_P;
  bool blender_ready = false;

  // --- Algorithm 8 (Rate Control State) ---
  // Exponential moving average of inter-arrival time (seconds)
  double dt_ema = 0.0;
  // Smoothed observed write rate estimate (writes per minute)
  double m_rate = 0.0;
  // Number of rate-update ticks observed
  int rate_ticks = 0;
  // Timestamp of last rate update (same units as Signal.timestamp)
  uint64_t last_rate_timestamp = 0;

  // --- Consolidation (Algorithm 28/28b) State ---
  // Last time consolidation started/completed (used for interval trigger)
  uint64_t last_consolidation_ts = 0;
  // Last time a retrieval event occurred (used for idle gating)
  uint64_t last_retrieval_ts = 0;

  // --- Algorithm 19 (Influence Feedback) State ---
  // Sustained (longer-horizon) influence EWMA
  double sustained_influence = 0.0;

  // --- Algorithm 24 (Working Memory) State ---
  struct WMSlot
  {
    Eigen::VectorXf embedding;
    double strength = 0.0;
    double last_ts = 0.0; // seconds
    int pos_index = 0;
  };
  std::vector<WMSlot> wm_slots;
  bool wm_last_accepted = false;
  bool wm_last_chunked = false;
};

} // namespace cortext
