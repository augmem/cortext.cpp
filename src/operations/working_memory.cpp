#include "cortext/operations/working_memory.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
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
WorkingMemory::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const auto &signal = context.GetSignal ();

  // Maintenance: decay strengths and update timestamps.
  const double cost_per_slot
      = core::WMMaintenanceCostPerSlot (cfg.sensitivity);
  const double now_s = static_cast<double> (signal.timestamp);
  {
    std::vector<ProcessorContext::WMSlot> kept;
    kept.reserve (p_ctx.wm_slots.size ());
    for (auto &slot : p_ctx.wm_slots)
      {
        const double last = slot.last_ts;
        const double dt = std::max (0.0, now_s - last);
        const double strength_after
            = std::max (0.0, slot.strength - cost_per_slot * dt);
        slot.strength = strength_after;
        slot.last_ts = now_s;
        if (strength_after > 0.0 && slot.embedding.size () > 0)
          {
            kept.push_back (slot);
          }
      }
    p_ctx.wm_slots.swap (kept);
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

  if (signal.embedding.size () == 0)
    {
      return; // nothing to gate
    }

  const double benefit = core::Clamp01 (
      context.GetCompositeScore ().value_or (p_ctx.weight_relevance));
  // gate_threshold = lerp(0.1, 0.4, F) per Algorithm 24 spec
  // Higher Focus (narrower attention) => stricter gating (0.4)
  // Lower Focus (wider attention) => permissive gating (0.1)
  const double gate_threshold = core::WMGateThreshold (cfg.focus);
  const double margin = benefit - gate_threshold;
  const double cost_total
      = cost_per_slot * (static_cast<double> (p_ctx.wm_slots.size ()) + 1.0)
        + complexity_penalty;

  if (margin < cost_total)
    {
      // Reject
      return;
    }

  // Try to chunk into best-matching slot.
  int best_idx = -1;
  double best_sim = -1.0;
  for (int i = 0; i < static_cast<int> (p_ctx.wm_slots.size ()); ++i)
    {
      const auto &slot = p_ctx.wm_slots[static_cast<size_t> (i)];
      if (slot.embedding.size () != signal.embedding.size ())
        {
          continue;
        }
      const double sim
          = core::Clamp (core::CosineSimilarity (slot.embedding, signal.embedding),
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
      Eigen::VectorXf new_vec
          = alpha * slot.embedding + beta * signal.embedding;
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
      p_ctx.wm_last_accepted = true;
      p_ctx.wm_last_chunked = true;
      return;
    }

  // Input-based rehearsal (task 4.2): boost strength for "close" slots without
  // merging. Slots with similarity in [rehearsal_threshold, chunk_threshold)
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

  // Retrieval-based rehearsal (task 4.3): boost WM slots that match retrieved
  // memories. Uses chunking_threshold as the similarity requirement per spec.
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
                  break; // Only boost once per slot per signal
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

  Eigen::VectorXf vec = signal.embedding;
  const double nrm = vec.norm ();
  if (nrm > constants::kNormEpsilon)
    {
      vec = vec / static_cast<float> (nrm);
    }
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
  p_ctx.wm_slots.push_back (std::move (slot));
  p_ctx.wm_last_accepted = true;
  p_ctx.wm_last_chunked = false;
}

} // namespace cortext::operations
