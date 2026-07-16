#include "cortext/operations/working_memory.hpp"
#include "signal_record_rollback_internal.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <vector>

namespace cortext::operations
{

namespace
{
inline double
ComputeMeanCosWindow (const std::deque<Eigen::VectorXf> &ctx,
                      const Eigen::VectorXf &emb, int window)
{
  if (ctx.empty () || emb.size () == 0 || window <= 0)
    {
      return 1.0;
    }
  const int n_total = static_cast<int> (ctx.size ());
  const int start = std::max (0, n_total - window);
  double sum = 0.0;
  int count = 0;
  for (int i = start; i < n_total; ++i)
    {
      const auto &e = ctx[static_cast<size_t> (i)];
      if (e.size () != emb.size ())
        {
          continue;
        }
      sum += core::CosineSimilarity (e, emb);
      ++count;
    }
  if (count <= 0)
    {
      return 1.0;
    }
  return sum / static_cast<double> (count);
}

inline double
RelevanceToTask (const Eigen::VectorXf &q,
                 const std::deque<Eigen::VectorXf> &task_ctx,
                 double focus, double sensitivity, double stability)
{
  if (q.size () == 0 || task_ctx.empty ())
    {
      return core::WMTaskRelevanceFallback (focus, sensitivity, stability);
    }
  Eigen::VectorXf mean = Eigen::VectorXf::Zero (q.size ());
  int count = 0;
  for (const auto &e : task_ctx)
    {
      if (e.size () != q.size ())
        {
          continue;
        }
      mean += e;
      ++count;
    }
  if (count <= 0)
    {
      return core::WMTaskRelevanceFallback (focus, sensitivity, stability);
    }
  mean /= static_cast<float> (count);
  const double cos = core::CosineSimilarity (q, mean);
  return core::Clamp ((cos + 1.0) * 0.5, 0.0, 1.0);
}

} // namespace

void
WorkingMemory::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const auto &signal = context.GetSignal ();
  const bool preserve_input_trace = RetentionForcesWrite (signal.retention);
  const double wm_strength_max = core::WMStrengthMax (
      cfg.focus, cfg.sensitivity, cfg.stability);

  // Maintenance: decay strengths based on elapsed time.
  // NOTE: We do NOT evict slots during passive decay. Slots should only be
  // replaced when at capacity and a new signal is accepted. This prevents
  // WM from going empty when signals are rejected or during idle periods.
  // last_ts remains the access timestamp used by recency. strength_ts advances
  // independently so each elapsed interval is charged exactly once.
  const double cost_per_slot
      = core::WMMaintenanceCostPerSlot (cfg.sensitivity, cfg.focus);
  // Convert timestamp from milliseconds to seconds for decay calculations
  const double now_s = static_cast<double> (signal.timestamp) / 1000.0;
  for (auto &slot : p_ctx.wm_slots)
    {
      const double last_decay
          = slot.strength_ts > 0.0 ? slot.strength_ts : slot.last_ts;
      const double dt = now_s - last_decay;
      double strength_after = slot.strength;
      double strength_ts_after = slot.strength_ts;
      if (dt > 0.0)
        {
          // Decay strength but floor at a minimum to preserve the slot. The
          // active floor applies only while charging positive elapsed time;
          // a knob change must not recharge an already-weaker slot.
          strength_after
              = std::max (core::WMStrengthFloor (
                              cfg.focus, cfg.sensitivity, cfg.stability),
                          slot.strength - cost_per_slot * dt);
          strength_ts_after = now_s;
        }
      if (slot.strength != strength_after
          || slot.strength_ts != strength_ts_after)
        {
          slot.metadata_dirty = true;
        }
      slot.strength = strength_after;
      slot.strength_ts = strength_ts_after;
    }

  // Complexity penalty from entropy over strengths.
  const int ctx_window = static_cast<int> (core::NCtx (cfg.stability));
  double mean_cos_window = 1.0;
  if (const auto &rep_emb_opt = context.GetRepresentativeEmbedding ();
      rep_emb_opt.has_value () && rep_emb_opt->size () > 0)
    {
      mean_cos_window = ComputeMeanCosWindow (
          p_ctx.recent_context_embeddings, *rep_emb_opt, ctx_window);
    }
  const double manifold_complexity
      = core::Clamp ((1.0 - mean_cos_window) * 0.5, 0.0, 1.0);
  const double complexity_penalty
      = manifold_complexity * core::WMComplexityScale (cfg.sensitivity);

  // Gate decision
  p_ctx.wm_last_accepted = false;
  p_ctx.wm_last_chunked = false;

  // Section 6.1.3: Memory-Level Gating
  // Working memory gating evaluates coherent memories at accumulation boundaries,
  // not individual signals. Only proceed if a memory write was accepted.
  if (!context.GetAccumulatorWriteDecision ())
    {
      return; // No memory boundary - skip WM gating
    }

  // Consolidation signals drive maintenance jobs through the normal
  // operation sequence,
  // but they are not ingress memories. They must not evict or insert
  // working-memory input traces.
  if (signal.force_consolidation)
    {
      return;
    }

  // Get memory-level data: e_rep and S_window
  const auto &rep_emb_opt = context.GetRepresentativeEmbedding ();
  if (!rep_emb_opt.has_value () || rep_emb_opt->size () == 0)
    {
      return; // No representative embedding available
    }
  const Eigen::VectorXf &e_rep = *rep_emb_opt;

  // Get accumulator state for memory metadata
  auto acc_it = p_ctx.accumulator_states.find (signal.source_id);
  if (acc_it == p_ctx.accumulator_states.end ())
    {
      return; // No accumulator state
    }
  const auto &acc = acc_it->second;
  const int acc_signal_count = acc.signals.empty ()
                                   ? acc.n_signals
                                   : static_cast<int> (acc.signals.size ());

  // Section 8.1: memory_benefit = α × S_window + β × relevance + γ × novelty
  const double S_window = context.GetWindowScore ().value_or (0.0);
  const Eigen::VectorXf &task_query
      = (acc.mu_acc.size () > 0) ? acc.mu_acc : e_rep;
  const double relevance = RelevanceToTask (
      task_query, p_ctx.recent_context_embeddings, cfg.focus,
      cfg.sensitivity, cfg.stability);

  double max_cos = -1.0;
  for (const auto &slot : p_ctx.wm_slots)
    {
      if (slot.embedding.size () != e_rep.size () || slot.embedding.size () == 0)
        {
          continue;
        }
      const double c = core::CosineSimilarity (slot.embedding, e_rep);
      max_cos = std::max (max_cos, c);
    }
  const double novelty_to_set
      = (max_cos <= -1.0)
            ? 1.0
            : core::Clamp ((1.0 - max_cos) * 0.5, 0.0, 1.0);

  const auto benefit_weights
      = core::WMBenefitScoringWeights (cfg.focus, cfg.sensitivity,
                                       cfg.stability);
  const double alpha = benefit_weights.window;
  const double beta = benefit_weights.relevance;
  const double gamma = benefit_weights.novelty;

  const double benefit = core::Clamp01 (
      alpha * core::Clamp01 (S_window) + beta * relevance + gamma * novelty_to_set);

  // gate_threshold = lerp(0.1, 0.4, F) per Algorithm 24 spec
  // Higher Focus (narrower attention) => stricter gating (0.4)
  // Lower Focus (wider attention) => permissive gating (0.1)
  const double gate_threshold = core::WMGateThreshold (cfg.focus);
  const double margin = benefit - gate_threshold;
  const int k = static_cast<int> (p_ctx.wm_slots.size ());
  const int base_capacity
      = std::max (1, core::WMBaseCapacity (cfg.sensitivity, cfg.focus));
  const double over_ratio
      = (k > base_capacity)
            ? (static_cast<double> (k - base_capacity)
               / static_cast<double> (base_capacity))
            : 0.0;
  const double capacity_pressure
      = 1.0
        + std::pow (over_ratio, core::WMCapacityPressureExponent (
                                    cfg.focus, cfg.sensitivity,
                                    cfg.stability));
  const double raw_cost
      = (cost_per_slot * static_cast<double> (k) + complexity_penalty)
        * capacity_pressure;
  const double cost_total = core::WMCostSaturator (raw_cost);

  if (!preserve_input_trace && margin < cost_total)
    {
      // Reject
      telemetry::LogDebug ("cortext.working_memory", {
        telemetry::Attribute::Bool ("accepted", false),
        telemetry::Attribute::String ("reason", "margin_below_cost"),
        telemetry::Attribute::Int64 ("wm_slot_count", static_cast<int64_t> (p_ctx.wm_slots.size ())),
        telemetry::Attribute::Double ("benefit", benefit),
        telemetry::Attribute::Double ("S_window", S_window),
        telemetry::Attribute::Double ("relevance", relevance),
        telemetry::Attribute::Double ("novelty_to_set", novelty_to_set),
        telemetry::Attribute::Double ("gate_threshold", gate_threshold),
        telemetry::Attribute::Double ("margin", margin),
        telemetry::Attribute::Double ("cost_total", cost_total),
        telemetry::Attribute::Double ("raw_cost", raw_cost),
        telemetry::Attribute::Double ("over_ratio", over_ratio),
        telemetry::Attribute::Double ("capacity_pressure", capacity_pressure),
        telemetry::Attribute::Int64 ("base_capacity", base_capacity),
        telemetry::Attribute::Double ("now_s", now_s),
        telemetry::Attribute::Double ("cost_per_slot", cost_per_slot)
      });
      return;
    }

  // Section 6.1.4: Try to chunk into best-matching slot using e_rep.
  int best_idx = -1;
  double best_sim = -1.0;
  for (int i = 0; !preserve_input_trace && i < static_cast<int> (p_ctx.wm_slots.size ()); ++i)
    {
      const auto &slot = p_ctx.wm_slots[static_cast<size_t> (i)];
      // v2: Only chunk into same-source slots (working-memory.plan.md)
      if (!slot.source_id.empty () && slot.source_id != signal.source_id)
        {
          continue;
        }
      if (slot.embedding.size () != e_rep.size ())
        {
          continue;
        }
      const double sim
          = core::Clamp (core::CosineSimilarity (slot.embedding, e_rep),
                         constants::kNormalizedMin, constants::kNormalizedMax);
      if (sim > best_sim)
        {
          best_sim = sim;
          best_idx = i;
        }
    }

  const double chunk_threshold = core::WMChunkingThreshold (cfg.focus);
  if (best_idx >= 0 && best_sim >= chunk_threshold)
    {
      auto &slot = p_ctx.wm_slots[static_cast<size_t> (best_idx)];
      const int old_n = slot.n_signals;
      const double add_strength = core::WMStrengthBase (cfg.stability) * benefit;
      const double alpha = std::max (constants::kTiny, slot.strength);
      const double beta = std::max (constants::kTiny, add_strength);
      Eigen::VectorXf new_vec = alpha * slot.embedding + beta * e_rep;
      const double norm = new_vec.norm ();
      if (norm > constants::kNormEpsilon)
        {
          new_vec = new_vec / static_cast<float> (norm);
        }
      slot.embedding = new_vec;
      slot.strength = std::min (
          wm_strength_max,
          std::max (constants::kNormalizedMin, slot.strength + add_strength));
      slot.last_ts = now_s;
      slot.strength_ts = now_s;

      // Update memory-level metadata (Section 6.1.1)
      const int acc_n = std::max (1, acc_signal_count);
      slot.n_signals = old_n + acc_signal_count;
      slot.s_max = std::max (slot.s_max, acc.s_max);
      const double acc_avg = acc.s_sum / static_cast<double> (acc_n);
      const double weighted_sum
          = slot.s_avg * static_cast<double> (std::max (1, old_n))
            + acc_avg * static_cast<double> (acc_n);
      slot.s_avg
          = weighted_sum / static_cast<double> (std::max (1, slot.n_signals));
      slot.drift_acc += acc.drift_acc;
      slot.s_emotion_max = std::max (slot.s_emotion_max, acc.s_emotion_max);
      const double acc_arousal_avg
          = acc.s_arousal_sum / static_cast<double> (acc_n);
      const double weighted_arousal
          = slot.s_arousal_avg * static_cast<double> (std::max (1, old_n))
            + acc_arousal_avg * static_cast<double> (acc_n);
      slot.s_arousal_avg
          = weighted_arousal / static_cast<double> (std::max (1, slot.n_signals));
      const int64_t acc_start_ts = static_cast<int64_t> (acc.t_start);
      if (slot.start_ts == 0
          || (acc_start_ts > 0 && acc_start_ts < slot.start_ts))
        {
          slot.start_ts = acc_start_ts;
        }
      if (slot.start_ts > 0
          && signal.timestamp >= static_cast<std::uint64_t> (slot.start_ts))
        {
          slot.mem_elapsed = std::max (
              slot.mem_elapsed,
              static_cast<double> (signal.timestamp - slot.start_ts) / 1000.0);
        }

      // v2: Append blob_ids from accumulated signals to slot
      for (const auto &rec : acc.signals)
        {
          if (!rec.blob_id.empty ())
            {
              slot.blob_ids.push_back (rec.blob_id);
            }
        }
      // v2: Preserve ordered signal records (offset serial positions)
      for (const auto &rec : acc.signals)
        {
          SignalRecord merged = rec;
          merged.serial_position = rec.serial_position + old_n;
          slot.signal_records.push_back (std::move (merged));
        }
      slot.metadata_dirty = true;
      slot.embedding_dirty = true;
      slot.signal_records_dirty = true;
      p_ctx.wm_slots_dirty = true;

      p_ctx.wm_last_accepted = true;
      p_ctx.wm_last_chunked = true;
      telemetry::LogDebug ("cortext.working_memory", {
        telemetry::Attribute::Bool ("accepted", true),
        telemetry::Attribute::Bool ("chunked", true),
        telemetry::Attribute::Int64 ("wm_slot_count", static_cast<int64_t> (p_ctx.wm_slots.size ())),
        telemetry::Attribute::Int64 ("chunked_into_slot", static_cast<int64_t> (best_idx)),
        telemetry::Attribute::Double ("best_sim", best_sim),
        telemetry::Attribute::Double ("chunk_threshold", chunk_threshold),
        telemetry::Attribute::Double ("benefit", benefit),
        telemetry::Attribute::Double ("S_window", S_window),
        telemetry::Attribute::Double ("relevance", relevance),
        telemetry::Attribute::Double ("novelty_to_set", novelty_to_set),
        telemetry::Attribute::Double ("gate_threshold", gate_threshold),
        telemetry::Attribute::Double ("margin", margin),
        telemetry::Attribute::Double ("cost_total", cost_total),
        telemetry::Attribute::Double ("capacity_pressure", capacity_pressure),
        telemetry::Attribute::Int64 ("base_capacity", base_capacity),
        telemetry::Attribute::Double ("cost_per_slot", cost_per_slot),
        telemetry::Attribute::Double ("complexity_penalty", complexity_penalty),
        telemetry::Attribute::Double ("w_alpha", alpha),
        telemetry::Attribute::Double ("w_beta", beta),
        telemetry::Attribute::Double ("w_gamma", gamma),
        telemetry::Attribute::Double ("alpha", alpha),
        telemetry::Attribute::Double ("beta", beta),
        telemetry::Attribute::Double ("gamma", gamma),
        telemetry::Attribute::Double ("now_s", now_s)
      });
      return;
    }

  // Input-based rehearsal: boost strength for "close" slots without merging.
  // Slots with similarity in [rehearsal_threshold, chunk_threshold)
  // get a strength boost but embedding is NOT merged.
  const double rehearsal_threshold = core::WMRehearsalThreshold (cfg.focus);
  if (!preserve_input_trace && best_idx >= 0 && best_sim >= rehearsal_threshold)
    {
      auto &slot = p_ctx.wm_slots[static_cast<size_t> (best_idx)];
      const double rehearsal_rate = core::WMRehearsalRate (cfg.sensitivity);
      const double boost = rehearsal_rate
                           * core::WMRehearsalBaseDelta (
                               cfg.focus, cfg.sensitivity, cfg.stability);
      slot.strength = std::min (wm_strength_max, slot.strength + boost);
      slot.last_ts = now_s;
      slot.strength_ts = now_s;
      slot.metadata_dirty = true;
      p_ctx.wm_slots_dirty = true;
      // Continue to insert logic - rehearsal doesn't prevent new slot creation
    }

  // Retrieval-based rehearsal: boost WM slots that match retrieved memories.
  // Uses chunking_threshold as the similarity requirement per spec.
  const auto &retrieved = context.GetRetrievedMemoryEmbeddings ();
  if (!retrieved.empty ())
    {
      const double retrieval_threshold = core::WMChunkingThreshold (cfg.focus);
      const double rehearsal_rate = core::WMRehearsalRate (cfg.sensitivity);
      const double boost = rehearsal_rate
                           * core::WMRehearsalBaseDelta (
                               cfg.focus, cfg.sensitivity, cfg.stability);

      for (auto &slot : p_ctx.wm_slots)
        {
          for (const auto &[retrieved_id, retrieved_vec] : retrieved)
            {
              if (slot.embedding.size () != retrieved_vec.size ())
                continue;
              const double sim
                  = core::CosineSimilarity (slot.embedding, retrieved_vec);
              if (sim >= retrieval_threshold)
                {
                  slot.strength
                      = std::min (wm_strength_max, slot.strength + boost);
                  slot.last_ts = now_s;
                  slot.strength_ts = now_s;
                  slot.metadata_dirty = true;
                  p_ctx.wm_slots_dirty = true;
                  break; // Only boost once per slot per memory
                }
            }
        }
    }

  // Insert new slot (evict highest eviction_score if capacity reached).
  // Eviction considers both dedication (strength * T-derived factor) and recency.
  const int capacity
      = std::max (1, core::WMBaseCapacity (cfg.sensitivity, cfg.focus));
  if (static_cast<int> (p_ctx.wm_slots.size ()) >= capacity)
    {
      int evict_idx = -1;
      if (preserve_input_trace)
        {
          int64_t oldest_start_ts = std::numeric_limits<int64_t>::max ();
          double oldest_last_ts = std::numeric_limits<double>::max ();
          for (int i = 0; i < static_cast<int> (p_ctx.wm_slots.size ()); ++i)
            {
              const auto &slot = p_ctx.wm_slots[static_cast<size_t> (i)];
              const int64_t start_ts
                  = slot.start_ts > 0
                        ? slot.start_ts
                        : static_cast<int64_t> (slot.last_ts * 1000.0);
              if (start_ts < oldest_start_ts
                  || (start_ts == oldest_start_ts
                      && slot.last_ts < oldest_last_ts))
                {
                  oldest_start_ts = start_ts;
                  oldest_last_ts = slot.last_ts;
                  evict_idx = i;
                }
            }
        }
      else
        {
          const double dedication_strength
              = core::WMSlotDedicationStrength (cfg.stability);
          double max_eviction_score = -std::numeric_limits<double>::infinity ();
          constexpr double kTieEpsilon = 1e-12;

          for (int i = 0; i < static_cast<int> (p_ctx.wm_slots.size ()); ++i)
            {
              const auto &slot = p_ctx.wm_slots[static_cast<size_t> (i)];

              // Dedication: higher strength + higher T = more dedicated
              const double dedication = std::clamp (
                  slot.strength * dedication_strength / wm_strength_max,
                  0.0, 1.0);

              // Recency: recent access = lower eviction priority
              const double elapsed = std::max (0.0, now_s - slot.last_ts);
              const double recency
                  = std::exp (-elapsed / core::WMRecencyTauSeconds (
                                            cfg.focus, cfg.sensitivity,
                                            cfg.stability));

              // Eviction score: high = weak + old = evict first
              const double eviction_score = (1.0 - dedication) * (1.0 - recency);

              bool prefer_slot = false;
              if (eviction_score > max_eviction_score + kTieEpsilon)
                {
                  prefer_slot = true;
                }
              else if (std::abs (eviction_score - max_eviction_score)
                       <= kTieEpsilon)
                {
                  const auto &current
                      = p_ctx.wm_slots[static_cast<size_t> (evict_idx)];
                  if (slot.strength < current.strength - kTieEpsilon)
                    {
                      prefer_slot = true;
                    }
                  else if (std::abs (slot.strength - current.strength)
                           <= kTieEpsilon)
                    {
                      prefer_slot = slot.last_ts < current.last_ts - kTieEpsilon;
                    }
                }

              if (prefer_slot)
                {
                  max_eviction_score = eviction_score;
                  evict_idx = i;
                }
            }
        }

      if (evict_idx >= 0)
        {
          signal_record_rollback_internal::PreserveWorkingMemorySlotBeforeErase (
              p_ctx, static_cast<std::size_t> (evict_idx));
          p_ctx.wm_slots.erase (p_ctx.wm_slots.begin ()
                                + static_cast<long> (evict_idx));
          p_ctx.wm_slots_dirty = true;
        }
    }

  // Normalize e_rep for storage
  Eigen::VectorXf vec = e_rep;
  const double nrm = vec.norm ();
  if (nrm > constants::kNormEpsilon)
    {
      vec = vec / static_cast<float> (nrm);
    }

  // Create new WMSlot with memory-level metadata (Section 6.1.1)
  ProcessorContext::WMSlot slot;
  slot.embedding = vec;
  slot.strength = std::max (
      constants::kNormalizedMin,
      core::WMInitialSlotStrength (cfg.focus, cfg.sensitivity,
                                   cfg.stability, benefit));
  slot.last_ts = now_s;
  slot.strength_ts = now_s;
  slot.pos_index = p_ctx.signals_processed;

  // Memory-level metadata from accumulator (Section 6.1.1)
  slot.n_signals = acc_signal_count;
  slot.s_max = acc.s_max;
  slot.s_avg = acc_signal_count > 0
                   ? acc.s_sum / static_cast<double> (acc_signal_count)
                   : 0.0;
  slot.drift_acc = acc.drift_acc;
  slot.mem_elapsed
      = signal.timestamp >= acc.t_start
            ? static_cast<double> (signal.timestamp - acc.t_start) / 1000.0
            : 0.0;
  slot.s_emotion_max = acc.s_emotion_max;
  slot.s_arousal_avg
      = acc_signal_count > 0
            ? acc.s_arousal_sum / static_cast<double> (acc_signal_count)
            : 0.0;

  // v2 persistence fields (working-memory.plan.md Section 4)
  slot.source_id = signal.source_id;
  if (!acc.primary_modality.empty ())
    {
      slot.modality = acc.primary_modality;
    }
  else
    {
      slot.modality
          = acc.signals.empty () ? signal.modality : acc.signals[0].modality;
    }
  slot.start_ts = static_cast<int64_t> (acc.t_start);
  slot.signal_records = acc.signals;
  for (const auto &rec : acc.signals)
    {
      if (!rec.blob_id.empty ())
        {
          slot.blob_ids.push_back (rec.blob_id);
        }
    }

  p_ctx.wm_slots.push_back (std::move (slot));
  p_ctx.wm_slots_dirty = true;
  p_ctx.wm_last_accepted = true;
  p_ctx.wm_last_chunked = false;

  telemetry::LogDebug ("cortext.working_memory", {
    telemetry::Attribute::Bool ("accepted", true),
    telemetry::Attribute::Bool ("chunked", false),
    telemetry::Attribute::Int64 ("wm_slot_count", static_cast<int64_t> (p_ctx.wm_slots.size ())),
    telemetry::Attribute::Double ("benefit", benefit),
    telemetry::Attribute::Double ("S_window", S_window),
    telemetry::Attribute::Double ("relevance", relevance),
    telemetry::Attribute::Double ("novelty_to_set", novelty_to_set),
    telemetry::Attribute::Double ("gate_threshold", gate_threshold),
    telemetry::Attribute::Double ("margin", margin),
    telemetry::Attribute::Double ("cost_total", cost_total),
    telemetry::Attribute::Double ("raw_cost", raw_cost),
    telemetry::Attribute::Double ("over_ratio", over_ratio),
    telemetry::Attribute::Double ("capacity_pressure", capacity_pressure),
    telemetry::Attribute::Int64 ("base_capacity", base_capacity),
    telemetry::Attribute::Double ("cost_per_slot", cost_per_slot),
    telemetry::Attribute::Double ("complexity_penalty", complexity_penalty),
    telemetry::Attribute::Double ("w_alpha", alpha),
    telemetry::Attribute::Double ("w_beta", beta),
    telemetry::Attribute::Double ("w_gamma", gamma),
    telemetry::Attribute::Double ("alpha", alpha),
    telemetry::Attribute::Double ("beta", beta),
    telemetry::Attribute::Double ("gamma", gamma),
    telemetry::Attribute::Double ("now_s", now_s)
  });
}

} // namespace cortext::operations
