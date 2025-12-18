#include "cortext/operations/working_memory.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace cortext::operations
{

namespace
{
/// @brief Computes WM strength base contribution scale based on Stability.
inline double
WMStrengthBase (double T)
{
  return core::Lerp (constants::kWMBaseMin, constants::kWMBaseMax, T);
}

inline double
ComputeNormalizedEntropy (const std::vector<double> &weights)
{
  if (weights.empty ())
    {
      return 0.0;
    }
  double sum = 0.0;
  for (double w : weights)
    {
      if (w > 0.0)
        sum += w;
    }
  if (sum <= std::numeric_limits<double>::epsilon ())
    {
      return 0.0;
    }
  const double inv_sum = 1.0 / sum;
  const int n = static_cast<int> (weights.size ());
  double H = 0.0;
  for (double w : weights)
    {
      if (w <= 0.0)
        continue;
      const double p = w * inv_sum;
      H -= p * std::log (p);
    }
  const double H_max = std::log (static_cast<double> (n));
  if (H_max <= std::numeric_limits<double>::epsilon ())
    {
      return 0.0;
    }
  return core::Clamp (H / H_max, 0.0, 1.0);
}

} // namespace

void
WorkingMemory::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const auto &signal = context.GetSignal ();

  // Maintenance: decay strengths based on elapsed time.
  // NOTE: We do NOT evict slots during passive decay. Slots should only be
  // replaced when at capacity and a new signal is accepted. This prevents
  // WM from going empty when signals are rejected or during idle periods.
  // NOTE: We do NOT update last_ts during passive decay - recency should only
  // reflect when a slot was actually accessed (chunked, rehearsed, or inserted).
  const double cost_per_slot
      = core::WMMaintenanceCostPerSlot (cfg.sensitivity);
  // Convert timestamp from milliseconds to seconds for decay calculations
  const double now_s = static_cast<double> (signal.timestamp) / 1000.0;
  for (auto &slot : p_ctx.wm_slots)
    {
      const double last = slot.last_ts;
      const double dt = std::max (0.0, now_s - last);
      // Decay strength but floor at a minimum to preserve slot
      // Slot will be replaced via eviction when capacity is reached
      const double strength_after
          = std::max (0.01, slot.strength - cost_per_slot * dt);
      slot.strength = strength_after;
      // Do NOT update last_ts here - only update when slot is accessed
    }

  // Complexity penalty from entropy over strengths.
  double complexity_penalty = 0.0;
  {
    std::vector<double> strengths;
    strengths.reserve (p_ctx.wm_slots.size ());
    for (const auto &slot : p_ctx.wm_slots)
      {
        strengths.push_back (std::max (0.0, slot.strength));
      }
    const double H = ComputeNormalizedEntropy (strengths);
    complexity_penalty = H * core::WMComplexityScale (cfg.sensitivity);
  }

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

  // Section 6.1.3: memory_benefit = α × S_window + β × relevance + γ × novelty
  // For simplicity, use S_window as the primary benefit (already computed in write_gate)
  const double S_window = context.GetWindowScore ().value_or (0.0);
  const double benefit = core::Clamp01 (S_window);

  // gate_threshold = lerp(0.1, 0.4, F) per Algorithm 24 spec
  // Higher Focus (narrower attention) => stricter gating (0.4)
  // Lower Focus (wider attention) => permissive gating (0.1)
  const double gate_threshold = core::WMGateThreshold (cfg.focus);
  const double margin = benefit - gate_threshold;
  // Only charge maintenance cost for existing slots, not the prospective new one.
  // This prevents a bootstrap problem where empty WM requires unreasonably high
  // composite scores to accept the first item.
  const double cost_total
      = cost_per_slot * static_cast<double> (p_ctx.wm_slots.size ())
        + complexity_penalty;

  if (margin < cost_total)
    {
      // Reject
      telemetry::LogDebug ("cortext.working_memory", {
        telemetry::Attribute::Bool ("accepted", false),
        telemetry::Attribute::String ("reason", "margin_below_cost"),
        telemetry::Attribute::Int64 ("wm_slot_count", static_cast<int64_t> (p_ctx.wm_slots.size ())),
        telemetry::Attribute::Double ("benefit", benefit),
        telemetry::Attribute::Double ("S_window", S_window),
        telemetry::Attribute::Double ("gate_threshold", gate_threshold),
        telemetry::Attribute::Double ("margin", margin),
        telemetry::Attribute::Double ("cost_total", cost_total),
        telemetry::Attribute::Double ("now_s", now_s),
        telemetry::Attribute::Double ("cost_per_slot", cost_per_slot)
      });
      return;
    }

  // Section 6.1.4: Try to chunk into best-matching slot using e_rep.
  int best_idx = -1;
  double best_sim = -1.0;
  for (int i = 0; i < static_cast<int> (p_ctx.wm_slots.size ()); ++i)
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
      const double add_strength = WMStrengthBase (cfg.stability) * benefit;
      const double alpha = std::max (constants::kTiny, slot.strength);
      const double beta = std::max (constants::kTiny, add_strength);
      Eigen::VectorXf new_vec = alpha * slot.embedding + beta * e_rep;
      const double norm = new_vec.norm ();
      if (norm > constants::kNormEpsilon)
        {
          new_vec = new_vec / static_cast<float> (norm);
        }
      slot.embedding = new_vec;
      slot.strength
          = std::min (constants::kStrengthMax,
                      std::max (constants::kNormalizedMin,
                                slot.strength + add_strength));
      slot.last_ts = now_s;

      // Update memory-level metadata (Section 6.1.1)
      slot.n_signals += acc.n_signals;
      slot.s_max = std::max (slot.s_max, acc.s_max);
      slot.s_avg = (slot.s_avg + acc.s_sum / std::max (1, acc.n_signals)) / 2.0;
      slot.drift_acc += acc.drift_acc;
      slot.s_emotion_max = std::max (slot.s_emotion_max, acc.s_emotion_max);
      if (acc.n_signals > 0)
        {
          const double new_arousal_avg
              = acc.s_arousal_sum / static_cast<double> (acc.n_signals);
          slot.s_arousal_avg = (slot.s_arousal_avg + new_arousal_avg) / 2.0;
        }

      // v2: Append blob_ids from accumulated signals to slot
      for (const auto &rec : acc.signals)
        {
          if (!rec.blob_id.empty ())
            {
              slot.blob_ids.push_back (rec.blob_id);
            }
        }

      p_ctx.wm_last_accepted = true;
      p_ctx.wm_last_chunked = true;
      telemetry::LogDebug ("cortext.working_memory", {
        telemetry::Attribute::Bool ("accepted", true),
        telemetry::Attribute::Bool ("chunked", true),
        telemetry::Attribute::Int64 ("wm_slot_count", static_cast<int64_t> (p_ctx.wm_slots.size ())),
        telemetry::Attribute::Int64 ("chunked_into_slot", static_cast<int64_t> (best_idx)),
        telemetry::Attribute::Double ("best_sim", best_sim),
        telemetry::Attribute::Double ("chunk_threshold", chunk_threshold),
        telemetry::Attribute::Double ("now_s", now_s)
      });
      return;
    }

  // Input-based rehearsal: boost strength for "close" slots without merging.
  // Slots with similarity in [rehearsal_threshold, chunk_threshold)
  // get a strength boost but embedding is NOT merged.
  const double rehearsal_threshold = core::WMRehearsalThreshold (cfg.focus);
  if (best_idx >= 0 && best_sim >= rehearsal_threshold)
    {
      auto &slot = p_ctx.wm_slots[static_cast<size_t> (best_idx)];
      const double rehearsal_rate = core::WMRehearsalRate (cfg.sensitivity);
      const double boost = rehearsal_rate * constants::kWMRehearsalBaseDelta;
      slot.strength = std::min (constants::kStrengthMax, slot.strength + boost);
      slot.last_ts = now_s;
      // Continue to insert logic - rehearsal doesn't prevent new slot creation
    }

  // Retrieval-based rehearsal: boost WM slots that match retrieved memories.
  // Uses chunking_threshold as the similarity requirement per spec.
  const auto &retrieved = context.GetRetrievedMemoryEmbeddings ();
  if (!retrieved.empty ())
    {
      const double retrieval_threshold = core::WMChunkingThreshold (cfg.focus);
      const double rehearsal_rate = core::WMRehearsalRate (cfg.sensitivity);
      const double boost = rehearsal_rate * constants::kWMRehearsalBaseDelta;

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
                      = std::min (constants::kStrengthMax, slot.strength + boost);
                  slot.last_ts = now_s;
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
      const double dedication_strength
          = core::WMSlotDedicationStrength (cfg.stability);
      int evict_idx = -1;
      double max_eviction_score = -std::numeric_limits<double>::infinity ();

      for (int i = 0; i < static_cast<int> (p_ctx.wm_slots.size ()); ++i)
        {
          const auto &slot = p_ctx.wm_slots[static_cast<size_t> (i)];

          // Dedication: higher strength + higher T = more dedicated
          const double dedication = std::clamp (
              slot.strength * dedication_strength / constants::kStrengthMax,
              0.0, 1.0);

          // Recency: recent access = lower eviction priority
          const double elapsed = std::max (0.0, now_s - slot.last_ts);
          const double recency
              = std::exp (-elapsed / constants::kWMRecencyTauSeconds);

          // Eviction score: high = weak + old = evict first
          const double eviction_score = (1.0 - dedication) * (1.0 - recency);

          if (eviction_score > max_eviction_score)
            {
              max_eviction_score = eviction_score;
              evict_idx = i;
            }
        }

      if (evict_idx >= 0)
        {
          p_ctx.wm_slots.erase (p_ctx.wm_slots.begin ()
                                + static_cast<long> (evict_idx));
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
  slot.strength
      = std::min (constants::kStrengthMax,
                  std::max (constants::kNormalizedMin,
                            WMStrengthBase (cfg.stability)
                                * (constants::kOneHalf
                                   + constants::kOneHalf * benefit)));
  slot.last_ts = now_s;
  slot.pos_index = p_ctx.signals_processed;

  // Memory-level metadata from accumulator (Section 6.1.1)
  slot.n_signals = acc.n_signals;
  slot.s_max = acc.s_max;
  slot.s_avg = acc.n_signals > 0 ? acc.s_sum / static_cast<double> (acc.n_signals) : 0.0;
  slot.drift_acc = acc.drift_acc;
  slot.mem_elapsed
      = static_cast<double> (signal.timestamp - acc.t_start) / 1000.0;
  slot.s_emotion_max = acc.s_emotion_max;
  slot.s_arousal_avg
      = acc.n_signals > 0
            ? acc.s_arousal_sum / static_cast<double> (acc.n_signals)
            : 0.0;

  // v2 persistence fields (working-memory.plan.md Section 4)
  slot.source_id = signal.source_id;
  slot.modality
      = acc.signals.empty () ? signal.modality : acc.signals[0].modality;
  slot.start_ts = static_cast<int64_t> (acc.t_start);
  for (const auto &rec : acc.signals)
    {
      if (!rec.blob_id.empty ())
        {
          slot.blob_ids.push_back (rec.blob_id);
        }
    }

  p_ctx.wm_slots.push_back (std::move (slot));
  p_ctx.wm_last_accepted = true;
  p_ctx.wm_last_chunked = false;

  telemetry::LogDebug ("cortext.working_memory", {
    telemetry::Attribute::Bool ("accepted", true),
    telemetry::Attribute::Bool ("chunked", false),
    telemetry::Attribute::Int64 ("wm_slot_count", static_cast<int64_t> (p_ctx.wm_slots.size ())),
    telemetry::Attribute::Double ("benefit", benefit),
    telemetry::Attribute::Double ("S_window", S_window),
    telemetry::Attribute::Double ("gate_threshold", gate_threshold),
    telemetry::Attribute::Double ("cost_total", cost_total),
    telemetry::Attribute::Double ("now_s", now_s)
  });
}

} // namespace cortext::operations
