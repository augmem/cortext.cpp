#include "cortext/operations/interrupt_gate.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/processor/processor_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <any>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_set>

namespace cortext::operations
{
namespace
{

// File-scoped constants (not shared broadly across operations)
constexpr double kCovMin = 0.40;
constexpr double kCovMax = 0.60;
constexpr double kRelMax = 0.35;
constexpr double kRelMin = 0.25;
constexpr double kRedMin = 0.15;
constexpr double kRedMax = 0.25;
constexpr double kCohMin = 0.15;
constexpr double kCohMax = 0.25;
constexpr double kDupMin = 0.88;
constexpr double kDupMax = 0.96;
constexpr double kDupBase = 0.98;
constexpr double kDupSlope = 0.02;
constexpr double kTauMuMin = 0.08;
constexpr double kTauMuMax = 0.18;
constexpr double kTauMuMinusS = 0.4;
constexpr double kTauMuPlusT = 0.4;
constexpr double kTauRefracMin = 24.0;
constexpr double kTauRefracMax = 96.0;
constexpr double kTauRefracSMin = 1.4;
constexpr double kTauRefracSMax = 1.0;
constexpr double kExpEpsilon = 1e-6;
constexpr double kBoundaryFMin = 1.3;
constexpr double kBoundaryFMax = 2.0;
constexpr double kBoundarySMin = 1.1;
constexpr double kBoundarySMax = 0.9;

inline double
Clamp01 (double v)
{
  return cortext::core::Clamp<double> (
      v, cortext::operations::constants::kNormalizedMin,
      cortext::operations::constants::kNormalizedMax);
}

inline double
CosSim (const Eigen::VectorXf &a, const Eigen::VectorXf &b)
{
  return cortext::core::CosineSimilarity (a, b);
}

Eigen::VectorXf
ComputeCentroid (const std::vector<Eigen::VectorXf> &vecs)
{
  if (vecs.empty ())
    {
      return Eigen::VectorXf (); // empty
    }
  const int dim = static_cast<int> (vecs.front ().size ());
  Eigen::VectorXf sum = Eigen::VectorXf::Zero (dim);
  for (const auto &v : vecs)
    {
      if (v.size () == sum.size ())
        {
          sum += v;
        }
    }
  const float n = static_cast<float> (vecs.size ());
  if (n > 0.0f)
    {
      sum /= n;
    }
  const float norm = sum.norm ();
  if (norm > 0.0f)
    {
      return sum / norm;
    }
  return sum;
}

double
ComputeRedundancy (const Eigen::VectorXf &candidate,
                   const std::vector<Eigen::VectorXf> &included)
{
  if (included.empty ())
    {
      return 0.0;
    }
  double max_overlap = -1.0;
  for (const auto &e : included)
    {
      max_overlap = std::max (max_overlap, CosSim (candidate, e));
    }
  if (max_overlap < 0.0)
    {
      return 0.0;
    }
  return Clamp01 (max_overlap);
}

double
ComputeCoverageGain (const Eigen::VectorXf &centroid,
                     const Eigen::VectorXf &candidate,
                     const std::vector<Eigen::VectorXf> &included)
{
  // Simple, bounded proxy: how much the candidate improves similarity
  // relative to the most-overlapping included vector.
  const double sim_ctx = Clamp01 (CosSim (centroid, candidate));
  const double redundancy = ComputeRedundancy (candidate, included);
  const double gain = std::max (0.0, sim_ctx - redundancy);
  return Clamp01 (gain);
}

} // namespace

void
ComputeMniGateDecision::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();

  // Retrieve candidates (id -> embedding)
  const auto &raw_candidates = context.GetRetrievedMemoryEmbeddings ();
  if (raw_candidates.empty ())
    {
      context.SetInterruptAllowed (false);
      // Diagnostics defaults
      context.SetMniJaccard (constants::kNormalizedMax);
      context.SetMniBestMu (constants::kNormalizedMin);
      context.SetMniDupThresh (constants::kNormalizedMin);
      context.SetMniTauJaccardEff (constants::kNormalizedMin);
      context.SetMniTauMuEff (constants::kNormalizedMin);
      return;
    }

  // Section 8.3.1: Write Exclusion Filter
  // Filter out candidates stored during the current signal processing cycle
  // to prevent self-triggering interrupts.
  Store *store = context.GetStore ();
  if (!store)
    {
      // No store available - cannot perform timestamp filtering.
      // Deny interrupt to prevent recursive interruptions.
      context.SetInterruptAllowed (false);
      context.SetMniJaccard (constants::kNormalizedMax);
      context.SetMniBestMu (constants::kNormalizedMin);
      context.SetMniDupThresh (constants::kNormalizedMin);
      context.SetMniTauJaccardEff (constants::kNormalizedMin);
      context.SetMniTauMuEff (constants::kNormalizedMin);
      return;
    }

  // Section 8.3.1: Use t_acc_start from accumulator state for write exclusion.
  // This ensures all memories written during the current accumulation cycle
  // are excluded from retrieval, preventing self-triggering interrupts.
  const auto &signal = context.GetSignal ();
  uint64_t write_exclusion_ts = signal.timestamp;  // Fallback to signal timestamp
  auto acc_it = p_ctx.accumulator_states.find (signal.source_id);
  if (acc_it != p_ctx.accumulator_states.end () && acc_it->second.t_start > 0)
    {
      write_exclusion_ts = acc_it->second.t_start;
    }

  // Query created_at timestamps for all candidate IDs
  std::unordered_map<long long, Eigen::VectorXf> candidates;
  {
    // Build IN clause for candidate IDs
    std::string sql = "SELECT embedding_id, created_at FROM embeddings "
                      "WHERE embedding_id IN (";
    std::vector<std::any> params;
    params.reserve (raw_candidates.size ());
    bool first = true;
    for (const auto &kv : raw_candidates)
      {
        if (!first)
          sql += ",";
        first = false;
        sql += "?";
        params.push_back (kv.first);
      }
    sql += ")";

    try
      {
        auto rows = store->Execute (sql, params);

        // Build set of eligible IDs (created_at < write_exclusion_ts)
        std::unordered_set<long long> eligible_ids;
        eligible_ids.reserve (rows.size ());
        for (const auto &row : rows)
          {
            auto it_id = row.find ("embedding_id");
            auto it_ts = row.find ("created_at");
            if (it_id == row.end () || it_ts == row.end ())
              continue;
            if (it_id->second.type () != typeid (long long))
              continue;

            const long long id = std::any_cast<long long> (it_id->second);
            uint64_t created_at = 0;
            if (it_ts->second.type () == typeid (long long))
              {
                created_at
                    = static_cast<uint64_t> (std::any_cast<long long> (it_ts->second));
              }

            // Include only if created_at < write_exclusion_ts
            if (created_at < write_exclusion_ts)
              {
                eligible_ids.insert (id);
              }
          }

        // Build filtered candidates map
        for (const auto &kv : raw_candidates)
          {
            if (eligible_ids.find (kv.first) != eligible_ids.end ())
              {
                candidates.emplace (kv.first, kv.second);
              }
          }
      }
    catch (...)
      {
        // Query failed - deny interrupt to be safe
        context.SetInterruptAllowed (false);
        context.SetMniJaccard (constants::kNormalizedMax);
        context.SetMniBestMu (constants::kNormalizedMin);
        context.SetMniDupThresh (constants::kNormalizedMin);
        context.SetMniTauJaccardEff (constants::kNormalizedMin);
        context.SetMniTauMuEff (constants::kNormalizedMin);
        return;
      }
  }

  // If all candidates were filtered out, deny interrupt
  if (candidates.empty ())
    {
      context.SetInterruptAllowed (false);
      context.SetMniJaccard (constants::kNormalizedMax);
      context.SetMniBestMu (constants::kNormalizedMin);
      context.SetMniDupThresh (constants::kNormalizedMin);
      context.SetMniTauJaccardEff (constants::kNormalizedMin);
      context.SetMniTauMuEff (constants::kNormalizedMin);
      return;
    }

  // Build included set from WM slots
  std::vector<Eigen::VectorXf> included_vecs;
  included_vecs.reserve (p_ctx.wm_slots.size ());
  for (const auto &slot : p_ctx.wm_slots)
    {
      if (slot.embedding.size () > 0)
        {
          included_vecs.push_back (slot.embedding);
        }
    }

  // Section 8.2: Build context vectors from recent_memory_centroids (μ_acc)
  // These are memory-level centroids, not individual signal embeddings.
  std::vector<Eigen::VectorXf> ctx_vecs;
  ctx_vecs.reserve (p_ctx.recent_memory_centroids.size ());
  for (const auto &v : p_ctx.recent_memory_centroids)
    {
      if (v.size () > 0)
        {
          ctx_vecs.push_back (v);
        }
    }
  if (ctx_vecs.empty ())
    {
      ctx_vecs = included_vecs;
    }
  if (ctx_vecs.empty ())
    {
      // No context to compare against → do not interrupt.
      context.SetInterruptAllowed (false);
      context.SetMniJaccard (constants::kNormalizedMax);
      context.SetMniBestMu (constants::kNormalizedMin);
      context.SetMniDupThresh (constants::kNormalizedMin);
      context.SetMniTauJaccardEff (constants::kNormalizedMin);
      context.SetMniTauMuEff (constants::kNormalizedMin);
      return;
    }

  const Eigen::VectorXf ctx_centroid = ComputeCentroid (ctx_vecs);
  const double coherence = Clamp01 (context.GetCoherence ());
  const double coherence_penalty = Clamp01 (constants::kNormalizedMax - coherence);

  // Knob-derived parameters
  const double F = cortext::core::Clamp<double> (cfg.focus,
                                                 constants::kNormalizedMin,
                                                 constants::kNormalizedMax);
  const double S = cortext::core::Clamp<double> (cfg.sensitivity,
                                                 constants::kNormalizedMin,
                                                 constants::kNormalizedMax);
  const double T = cortext::core::Clamp<double> (cfg.stability,
                                                 constants::kNormalizedMin,
                                                 constants::kNormalizedMax);

  // Retrieval threshold (Section 8.1)
  const double retrieval_thresh = cortext::core::RetrievalThreshold (F);

  // Derived weights (raw)
  const double w_cov_raw = cortext::core::Lerp (kCovMin, kCovMax, F);
  const double w_rel_raw = cortext::core::Lerp (kRelMax, kRelMin, F);
  const double w_red_raw = cortext::core::Lerp (kRedMin, kRedMax, S);
  const double w_coh_raw = cortext::core::Lerp (kCohMin, kCohMax, S);
  const double total_w
      = std::max (constants::kNormEpsilon,
                  w_cov_raw + w_rel_raw + w_red_raw + w_coh_raw);
  const double w_cov = w_cov_raw / total_w;
  const double w_rel = w_rel_raw / total_w;
  const double w_red = w_red_raw / total_w;
  const double w_coh = w_coh_raw / total_w;

  // Duplicate suppression threshold
  const double dup_thresh
      = cortext::core::Lerp (kDupMin, kDupMax, F)
        * (kDupBase + kDupSlope
                            * cortext::core::Clamp<double> (
                                T, constants::kNormalizedMin,
                                constants::kNormalizedMax));
  // Novelty threshold (Section 8.1)
  const double tau_novelty = cortext::core::TauNovelty (F, S, T);
  // MU threshold
  const double tau_mu = cortext::core::Lerp (kTauMuMin, kTauMuMax, F)
                        * (constants::kNormalizedMax - kTauMuMinusS * S)
                        * (constants::kNormalizedMax + kTauMuPlusT * T);

  // Refractory multiplier based on drift accumulation (Section 10)
  // Delta = cumulative drift since last interrupt
  const double Delta = std::max (0.0, p_ctx.drift_accum
                                          - p_ctx.drift_at_last_interrupt);
  const double tau_refrac = cortext::core::Lerp (kTauRefracMin, kTauRefracMax, T)
                            * cortext::core::Lerp (kTauRefracSMin, kTauRefracSMax,
                                                   S);
  const double k_refrac
      = cortext::core::Lerp (constants::kWeight20, constants::kGainSmall, T)
        * cortext::core::Lerp (0.8, 1.2, F);
  const double M_refrac
      = 1.0 + k_refrac * std::exp (-Delta / std::max (kExpEpsilon, tau_refrac));

  const double tau_novelty_eff = tau_novelty * M_refrac;
  const double tau_m_eff = tau_mu * M_refrac;

  // Boundary multiplier
  const double boundary_mult
      = cortext::core::Lerp (kBoundaryFMin, kBoundaryFMax, F)
        * cortext::core::Lerp (kBoundarySMin, kBoundarySMax, S);
  const bool at_boundary = context.GetAtBoundary ();

  // Limit candidates to top K by relevance (Section 8.3)
  const int K = cortext::core::InterruptCandidateCount (F);

  std::vector<std::pair<long long, double>> candidate_relevances;
  candidate_relevances.reserve (candidates.size ());
  for (const auto &kv : candidates)
    {
      const auto &cand = kv.second;
      if (cand.size () > 0 && cand.size () == ctx_centroid.size ())
        {
          const double sim = Clamp01 (CosSim (ctx_centroid, cand));
          candidate_relevances.emplace_back (kv.first, sim);
        }
    }

  const size_t k_size
      = std::min (static_cast<size_t> (K), candidate_relevances.size ());
  if (k_size < candidate_relevances.size ())
    {
      std::partial_sort (candidate_relevances.begin (),
                         candidate_relevances.begin ()
                             + static_cast<ptrdiff_t> (k_size),
                         candidate_relevances.end (),
                         [] (const auto &a, const auto &b) {
                           return a.second > b.second; // descending by relevance
                         });
      candidate_relevances.resize (k_size);
    }

  std::unordered_set<long long> top_k_ids;
  top_k_ids.reserve (k_size);
  for (const auto &cr : candidate_relevances)
    {
      top_k_ids.insert (cr.first);
    }

  // Iterate top K candidates and compute MU and overlaps
  double best_mu = -std::numeric_limits<double>::infinity ();
  double max_relevance = 0.0;
  double max_semantic_overlap = 0.0;

  for (const auto &kv : candidates)
    {
      if (top_k_ids.find (kv.first) == top_k_ids.end ())
        {
          continue; // Skip candidates not in top K
        }
      const auto &cand = kv.second;
      if (cand.size () == 0 || cand.size () != ctx_centroid.size ())
        {
          continue;
        }

      const double sim_ctx = Clamp01 (CosSim (ctx_centroid, cand));
      const double redundancy = ComputeRedundancy (cand, included_vecs);
      max_semantic_overlap = std::max (max_semantic_overlap, redundancy);
      const double cov_gain
          = ComputeCoverageGain (ctx_centroid, cand, included_vecs);

      const double mu = w_cov * cov_gain + w_rel * sim_ctx - w_red * redundancy
                        - w_coh * coherence_penalty;

      best_mu = std::max (best_mu, mu);
      max_relevance = std::max (max_relevance, sim_ctx);
    }

  // Section 8.3: Compute embedding novelty: 1 - max(cos(candidate, ctx_window))
  // Uses memory centroids (μ_acc) per Section 8.2
  double max_embedding_novelty = 0.0;
  const auto &ctx_window = p_ctx.recent_memory_centroids;

  for (const auto &kv : candidates)
    {
      if (top_k_ids.find (kv.first) == top_k_ids.end ())
        {
          continue;
        }
      const auto &cand = kv.second;
      if (cand.size () == 0)
        {
          continue;
        }

      // Find maximum cosine similarity to any embedding in context window
      double max_sim_to_ctx = 0.0;
      for (const auto &ctx_emb : ctx_window)
        {
          if (ctx_emb.size () == cand.size ())
            {
              const double sim = CosSim (cand, ctx_emb);
              max_sim_to_ctx = std::max (max_sim_to_ctx, sim);
            }
        }

      // Embedding novelty = 1 - max_similarity
      const double candidate_novelty = 1.0 - Clamp01 (max_sim_to_ctx);
      max_embedding_novelty = std::max (max_embedding_novelty, candidate_novelty);
    }

  // If context window is empty, all candidates are fully novel
  if (ctx_window.empty ())
    {
      max_embedding_novelty = 1.0;
    }

  // Keep candidate ID list for LRU recording
  std::vector<long long> A;
  A.reserve (candidates.size ());
  for (const auto &kv : candidates)
    {
      A.push_back (kv.first);
    }

  const bool allow_interrupt
      = (max_relevance >= retrieval_thresh)
        && ((max_embedding_novelty >= tau_novelty_eff) || (best_mu >= tau_m_eff))
        && (max_semantic_overlap < dup_thresh)
        && (at_boundary || best_mu >= boundary_mult * tau_m_eff);

  context.SetInterruptAllowed (allow_interrupt);
  if (allow_interrupt)
    {
      p_ctx.last_interrupt_tick = p_ctx.signals_processed;
      p_ctx.drift_at_last_interrupt = p_ctx.drift_accum; // Update drift baseline
    }

  // Diagnostics (field names unchanged for API compatibility)
  context.SetMniJaccard (max_embedding_novelty);     // Now stores embedding novelty
  context.SetMniBestMu (best_mu);
  context.SetMniDupThresh (dup_thresh);
  context.SetMniTauJaccardEff (tau_novelty_eff);     // Now stores tau_novelty_eff
  context.SetMniTauMuEff (tau_m_eff);

  // Record used and retrieved IDs into the LRU after decision is finalized.
  {
    std::vector<long long> ids_to_record;
    ids_to_record.reserve (A.size ());
    // Record candidates
    for (const auto &id : A)
      {
        ids_to_record.push_back (id);
      }
    // Record used memory IDs
    for (const auto &e : context.GetMemoryUsageEvents ())
      {
        if (e.used)
          {
            ids_to_record.push_back (static_cast<long long> (e.embedding_id));
          }
      }
    if (!ids_to_record.empty ())
      {
        p_ctx.recent_ids_lru_.RecordIds (ids_to_record);
      }
  }

  telemetry::LogDebug ("cortext.interrupt_gate", {
    telemetry::Attribute::Double ("best_mu", best_mu),
    telemetry::Attribute::Double ("jaccard", max_embedding_novelty),
    telemetry::Attribute::Double ("dup_thresh", dup_thresh),
    telemetry::Attribute::Double ("refractory_mult", M_refrac),
    telemetry::Attribute::Bool ("mni_decision", allow_interrupt),
    telemetry::Attribute::Int64 ("wm_slots_count", static_cast<int64_t> (included_vecs.size ())),
    telemetry::Attribute::Double ("max_semantic_overlap", max_semantic_overlap),
    telemetry::Attribute::Int64 ("ctx_window_size", static_cast<int64_t> (ctx_window.size ()))
  });
}

} // namespace cortext::operations
