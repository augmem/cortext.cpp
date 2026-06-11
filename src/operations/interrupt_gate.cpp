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
#include <optional>
#include <string>
#include <unordered_set>

namespace cortext::operations
{
namespace
{

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
  return cortext::core::Map01 (max_overlap);
}

double
ComputeRawOverlap (const Eigen::VectorXf &candidate,
                   const std::vector<Eigen::VectorXf> &included)
{
  if (included.empty ())
    {
      return -1.0;
    }
  double max_overlap = -1.0;
  for (const auto &e : included)
    {
      max_overlap = std::max (max_overlap, CosSim (candidate, e));
    }
  return max_overlap;
}

double
ComputeCoverageGain (const Eigen::VectorXf &centroid,
                     const Eigen::VectorXf &candidate,
                     const std::vector<Eigen::VectorXf> &included)
{
  const double redundancy = ComputeRedundancy (candidate, included);
  (void)centroid;
  return Clamp01 (1.0 - redundancy);
}

} // namespace

void
ComputeMniGateDecision::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const auto &signal = context.GetSignal ();
  context.SetSelectedCandidateId (std::nullopt);

  const double F = cortext::core::Clamp<double> (cfg.focus,
                                                 constants::kNormalizedMin,
                                                 constants::kNormalizedMax);
  const double S = cortext::core::Clamp<double> (cfg.sensitivity,
                                                 constants::kNormalizedMin,
                                                 constants::kNormalizedMax);
  const double T = cortext::core::Clamp<double> (cfg.stability,
                                                 constants::kNormalizedMin,
                                                 constants::kNormalizedMax);
  const double F_eff = cortext::core::FocusBias (F);
  const double S_eff = cortext::core::SensitivityBias (S);
  const double retrieval_thresh
      = cortext::core::RetrievalThresholdInterrupt (F, S);
  const double boundary_mult_base
      = cortext::core::InterruptBoundaryMultiplier (F, S, T);

  context.SetInterruptGateHasCandidates (false);
  context.SetInterruptGateBlockedNoStore (false);
  context.SetInterruptGateRelPass (false);
  context.SetInterruptGateNoveltyPass (false);
  context.SetInterruptGateMuPass (false);
  context.SetInterruptGateNoveltyMuPass (false);
  context.SetInterruptGateDupPass (false);
  context.SetInterruptGateBoundaryMuPass (false);
  context.SetInterruptGateRelStar (constants::kNormalizedMin);
  context.SetInterruptGateRetrievalThresh (retrieval_thresh);
  context.SetInterruptGateBoundaryMultEff (boundary_mult_base);
  context.SetInterruptGateAffectDrive (constants::kNormalizedMin);

  auto acc_it = p_ctx.accumulator_states.find (signal.source_id);
  if (acc_it == p_ctx.accumulator_states.end ())
    {
      p_ctx.accumulator_states[signal.source_id] = AccumulatorState{};
      acc_it = p_ctx.accumulator_states.find (signal.source_id);
    }
  auto &acc = acc_it->second;

  // Retrieve candidates (id -> embedding)
  const auto &raw_candidates = context.GetRetrievedMemoryEmbeddings ();
  if (raw_candidates.empty ())
    {
      context.SetInterruptGateHasCandidates (false);
      context.SetInterruptAllowed (false);
      // Diagnostics defaults
      context.SetMniJaccard (constants::kNormalizedMax);
      context.SetMniBestMu (constants::kNormalizedMin);
      context.SetMniDupThresh (constants::kNormalizedMin);
      context.SetMniOverlapStar (-1.0);
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
      context.SetInterruptGateHasCandidates (true);
      context.SetInterruptGateBlockedNoStore (true);
      context.SetInterruptAllowed (false);
      context.SetMniJaccard (constants::kNormalizedMax);
      context.SetMniBestMu (constants::kNormalizedMin);
      context.SetMniDupThresh (constants::kNormalizedMin);
      context.SetMniOverlapStar (-1.0);
      context.SetMniTauJaccardEff (constants::kNormalizedMin);
      context.SetMniTauMuEff (constants::kNormalizedMin);
      return;
    }

  // Section 8.3.1: Use t_acc_start from accumulator state for write exclusion.
  // This ensures all memories written during the current accumulation cycle
  // are excluded from retrieval, preventing self-triggering interrupts.
  uint64_t write_exclusion_ts = signal.timestamp;  // Fallback to signal timestamp
  if (auto cached = context.GetWriteExclusionTs ())
    {
      write_exclusion_ts = *cached;
    }
  else if (acc.t_start > 0)
    {
      write_exclusion_ts = acc.t_start;
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
        context.SetInterruptGateHasCandidates (true);
        context.SetInterruptAllowed (false);
        context.SetMniJaccard (constants::kNormalizedMax);
        context.SetMniBestMu (constants::kNormalizedMin);
        context.SetMniDupThresh (constants::kNormalizedMin);
        context.SetMniOverlapStar (-1.0);
        context.SetMniTauJaccardEff (constants::kNormalizedMin);
        context.SetMniTauMuEff (constants::kNormalizedMin);
        return;
      }
  }

  // If all candidates were filtered out, deny interrupt
  if (candidates.empty ())
    {
      context.SetInterruptGateHasCandidates (false);
      context.SetInterruptAllowed (false);
      context.SetMniJaccard (constants::kNormalizedMax);
      context.SetMniBestMu (constants::kNormalizedMin);
      context.SetMniDupThresh (constants::kNormalizedMin);
      context.SetMniOverlapStar (-1.0);
      context.SetMniTauJaccardEff (constants::kNormalizedMin);
      context.SetMniTauMuEff (constants::kNormalizedMin);
      return;
    }

  context.SetInterruptGateHasCandidates (true);

  // Build WM set (used for duplicate suppression and overlap checks)
  std::vector<Eigen::VectorXf> wm_vecs;
  wm_vecs.reserve (p_ctx.wm_slots.size ());
  for (const auto &slot : p_ctx.wm_slots)
    {
      if (slot.embedding.size () > 0)
        {
          wm_vecs.push_back (slot.embedding);
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
  if (ctx_vecs.empty () && acc.mu_acc.size () > 0)
    {
      Eigen::VectorXf mu_acc = acc.mu_acc;
      const float norm = mu_acc.norm ();
      if (norm > 0.0f)
        {
          mu_acc /= norm;
        }
      ctx_vecs.push_back (mu_acc);
    }
  if (ctx_vecs.empty () && !wm_vecs.empty ())
    {
      ctx_vecs = wm_vecs;
    }
  if (ctx_vecs.empty ())
    {
      // No context to compare against → do not interrupt.
      context.SetInterruptAllowed (false);
      context.SetMniJaccard (constants::kNormalizedMax);
      context.SetMniBestMu (constants::kNormalizedMin);
      context.SetMniDupThresh (constants::kNormalizedMin);
      context.SetMniOverlapStar (-1.0);
      context.SetMniTauJaccardEff (constants::kNormalizedMin);
      context.SetMniTauMuEff (constants::kNormalizedMin);
      return;
    }

  const Eigen::VectorXf ctx_centroid = ComputeCentroid (ctx_vecs);

  // Use WM + recent memory centroids as the inclusion set for redundancy/coverage.
  std::vector<Eigen::VectorXf> included_vecs = wm_vecs;
  if (!ctx_vecs.empty ())
    {
      included_vecs.insert (included_vecs.end (), ctx_vecs.begin (),
                            ctx_vecs.end ());
    }
  const double coherence = Clamp01 (context.GetCoherence ());
  const double coherence_penalty = Clamp01 (constants::kNormalizedMax - coherence);
  const double surprisal
      = Clamp01 (context.GetMetric (operations::Metric::embedding_surprisal)
                     .value_or (0.0));
  const double salience
      = Clamp01 (
          context.GetMetric (operations::Metric::salience).value_or (0.0));
  const double emotion_intensity
      = Clamp01 (context.GetEmotionIntensity ());
  const double arousal = Clamp01 (context.GetArousal ());

  const auto affect_weights = cortext::core::AffectDriveWeights (S);
  const double affect_gain = cortext::core::AffectGain (S);
  const double affect_drive = Clamp01 (
      affect_gain
      * (affect_weights.arousal_weight * arousal
         + affect_weights.emotion_weight * emotion_intensity
         + affect_weights.salience_weight * salience));
  const bool affect_interrupt = context.GetConfig ().affect_interrupt;
  const double affect_drive_used = affect_interrupt ? affect_drive : 0.0;
  const double affect_relax
      = 1.0 - cortext::core::InterruptAffectRelaxCoeff (S) * affect_drive_used;
  double retrieval_thresh_eff = retrieval_thresh * affect_relax;
  const double boundary_mult_eff = boundary_mult_base * affect_relax;

  const auto scoring_weights
      = cortext::core::InterruptGateCandidateScoringWeights (F, S, T);
  const double w_cov_raw = scoring_weights.coverage_raw;
  const double w_rel_raw = scoring_weights.relevance_raw;
  const double w_red_raw = scoring_weights.redundancy_raw;
  const double w_coh_raw = scoring_weights.coherence_raw;
  const double w_surp_raw = scoring_weights.surprise_raw;
  const double w_cov = scoring_weights.coverage;
  const double w_rel = scoring_weights.relevance;
  const double w_red = scoring_weights.redundancy;
  const double w_coh = scoring_weights.coherence;
  const double w_surp = scoring_weights.surprise;

  // Duplicate suppression threshold
  const double dup_thresh = cortext::core::DupThresh (F, T);
  // Novelty threshold (Section 8.1)
  const double tau_novelty = cortext::core::TauNovelty (F, S, T);
  // MU threshold
  const double tau_mu = cortext::core::InterruptMuThreshold (F, S, T);

  // Refractory multiplier based on drift accumulation (Section 10)
  // Delta = cumulative drift since last interrupt
  const double Delta = std::max (0.0, acc.drift_accum
                                          - acc.drift_at_last_interrupt);
  const double tau_refrac = cortext::core::InterruptRefractoryTau (S, T);
  const double k_refrac = cortext::core::InterruptRefractoryGain (F, T);
  const double M_refrac
      = 1.0
        + k_refrac
              * std::exp (
                  -Delta / std::max (constants::kTiny, tau_refrac));

  const double tau_novelty_eff = tau_novelty * M_refrac;
  const double win_coh = std::max (1.0, static_cast<double> (core::WinCoh (T)));
  const double acc_maturity
      = Clamp01 (static_cast<double> (acc.n_signals) / win_coh);
  const double maturity_scale
      = core::InterruptMaturityScale (acc_maturity, T);
  const double tau_m_eff = tau_mu * M_refrac * maturity_scale;

  // Boundary multiplier (precomputed at start of operation)
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
          const double sim = cortext::core::Map01 (CosSim (ctx_centroid, cand));
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
  double rel_star = 0.0;
  long long candidate_star_id = 0;
  Eigen::VectorXf candidate_star;
  bool has_candidate_star = false;
  double best_novelty = 0.0;

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

      const double sim_ctx = cortext::core::Map01 (CosSim (ctx_centroid, cand));

      const double redundancy = ComputeRedundancy (cand, included_vecs);
      double cov_gain = ComputeCoverageGain (ctx_centroid, cand, included_vecs);
      cov_gain = std::max (cov_gain, p_ctx.coverage_gain_floor);

      double novelty_ctx = 1.0;
      if (!ctx_vecs.empty ())
        {
          double max_sim_to_ctx = -1.0;
          for (const auto &ctx_emb : ctx_vecs)
            {
              if (ctx_emb.size () == cand.size ())
                {
                  max_sim_to_ctx = std::max (max_sim_to_ctx, CosSim (cand, ctx_emb));
                }
            }
          novelty_ctx = cortext::core::Clamp ((1.0 - max_sim_to_ctx) * 0.5,
                                              constants::kNormalizedMin,
                                              constants::kNormalizedMax);
        }

      const double surprise_bonus = surprisal * novelty_ctx * acc_maturity;
      const double mu = w_cov * cov_gain + w_rel * sim_ctx - w_red * redundancy
                        - w_coh * coherence_penalty + w_surp * surprise_bonus;

      if (mu > best_mu)
        {
          best_mu = mu;
          rel_star = sim_ctx;
          candidate_star_id = kv.first;
          candidate_star = cand;
          has_candidate_star = true;
          best_novelty = novelty_ctx;
        }
    }

  if (!has_candidate_star)
    {
      context.SetInterruptAllowed (false);
      context.SetSelectedCandidateId (std::nullopt);
      context.SetMniJaccard (constants::kNormalizedMin);
      context.SetMniBestMu (constants::kNormalizedMin);
      context.SetMniDupThresh (dup_thresh);
      context.SetMniOverlapStar (-1.0);
      context.SetMniTauJaccardEff (tau_novelty_eff);
      context.SetMniTauMuEff (tau_m_eff);
      return;
    }

  const double overlap_star = wm_vecs.empty ()
                                  ? -1.0
                                  : ComputeRawOverlap (candidate_star, wm_vecs);

  // Section 8.3: Compute embedding novelty for candidate_star
  // Uses memory centroids (μ_acc) per Section 8.2
  const double novelty_star = best_novelty;

  // Keep candidate ID list for LRU recording
  std::vector<long long> A;
  A.reserve (candidates.size ());
  for (const auto &kv : candidates)
    {
      A.push_back (kv.first);
    }

  const bool rel_pass = (rel_star >= retrieval_thresh_eff);
  const bool novelty_pass = (novelty_star >= tau_novelty_eff);
  const bool mu_pass = (best_mu >= tau_m_eff);
  const bool novelty_mu_pass = (novelty_pass || mu_pass);
  const bool dup_pass = (overlap_star < dup_thresh);
  const bool boundary_mu_pass
      = (at_boundary || best_mu >= boundary_mult_eff * tau_m_eff);
  const bool allow_interrupt
      = rel_pass && novelty_mu_pass && dup_pass && boundary_mu_pass;

  context.SetInterruptGateRelPass (rel_pass);
  context.SetInterruptGateNoveltyPass (novelty_pass);
  context.SetInterruptGateMuPass (mu_pass);
  context.SetInterruptGateNoveltyMuPass (novelty_mu_pass);
  context.SetInterruptGateDupPass (dup_pass);
  context.SetInterruptGateBoundaryMuPass (boundary_mu_pass);
  context.SetInterruptGateRelStar (rel_star);
  context.SetInterruptGateRetrievalThresh (retrieval_thresh_eff);
  context.SetInterruptGateBoundaryMultEff (boundary_mult_eff);
  context.SetInterruptGateAffectDrive (affect_drive_used);

  context.SetInterruptAllowed (allow_interrupt);
  context.SetSelectedCandidateId (allow_interrupt
                                      ? std::make_optional (candidate_star_id)
                                      : std::nullopt);
  if (allow_interrupt)
    {
      p_ctx.last_interrupt_tick = p_ctx.signals_processed;
      acc.drift_at_last_interrupt = acc.drift_accum; // Update drift baseline
    }

  context.SetMniJaccard (novelty_star);     // Now stores embedding novelty
  context.SetMniBestMu (best_mu);
  context.SetMniDupThresh (dup_thresh);
  context.SetMniOverlapStar (overlap_star);
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
    // Record used memory ID (selected candidate)
    if (allow_interrupt && candidate_star_id > 0)
      {
        ids_to_record.push_back (candidate_star_id);
      }
    if (!ids_to_record.empty ())
      {
        p_ctx.recent_ids_lru_.RecordIds (ids_to_record);
      }
  }

  telemetry::LogDebug ("cortext.interrupt_gate", {
    telemetry::Attribute::Double ("F", F),
    telemetry::Attribute::Double ("S", S),
    telemetry::Attribute::Double ("T", T),
    telemetry::Attribute::Double ("F_eff", F_eff),
    telemetry::Attribute::Double ("S_eff", S_eff),
    telemetry::Attribute::Double ("coherence", coherence),
    telemetry::Attribute::Double ("coherence_penalty", coherence_penalty),
    telemetry::Attribute::Double ("surprisal", surprisal),
    telemetry::Attribute::Double ("acc_maturity", acc_maturity),
    telemetry::Attribute::Double ("maturity_scale", maturity_scale),
    telemetry::Attribute::Double ("retrieval_thresh", retrieval_thresh),
    telemetry::Attribute::Double ("retrieval_thresh_eff", retrieval_thresh_eff),
    telemetry::Attribute::Double ("affect_drive", affect_drive_used),
    telemetry::Attribute::Double ("w_cov_raw", w_cov_raw),
    telemetry::Attribute::Double ("w_rel_raw", w_rel_raw),
    telemetry::Attribute::Double ("w_red_raw", w_red_raw),
    telemetry::Attribute::Double ("w_coh_raw", w_coh_raw),
    telemetry::Attribute::Double ("w_surp_raw", w_surp_raw),
    telemetry::Attribute::Double ("w_cov", w_cov),
    telemetry::Attribute::Double ("w_rel", w_rel),
    telemetry::Attribute::Double ("w_red", w_red),
    telemetry::Attribute::Double ("w_coh", w_coh),
    telemetry::Attribute::Double ("w_surp", w_surp),
    telemetry::Attribute::Double ("dup_thresh", dup_thresh),
    telemetry::Attribute::Double ("tau_novelty", tau_novelty),
    telemetry::Attribute::Double ("tau_mu", tau_mu),
    telemetry::Attribute::Double ("tau_novelty_eff", tau_novelty_eff),
    telemetry::Attribute::Double ("tau_mu_eff", tau_m_eff),
    telemetry::Attribute::Double ("Delta", Delta),
    telemetry::Attribute::Double ("tau_refrac", tau_refrac),
    telemetry::Attribute::Double ("k_refrac", k_refrac),
    telemetry::Attribute::Double ("refractory_mult", M_refrac),
    telemetry::Attribute::Double ("boundary_mult", boundary_mult_base),
    telemetry::Attribute::Bool ("at_boundary", at_boundary),
    telemetry::Attribute::Int64 ("candidate_limit", static_cast<int64_t> (K)),
    telemetry::Attribute::Int64 ("candidate_count", static_cast<int64_t> (candidates.size ())),
    telemetry::Attribute::Int64 ("top_k_count", static_cast<int64_t> (k_size)),
    telemetry::Attribute::Double ("best_mu", best_mu),
    telemetry::Attribute::Double ("rel_star", rel_star),
    telemetry::Attribute::Double ("novelty_star", novelty_star),
    telemetry::Attribute::Double ("overlap_star", overlap_star),
    telemetry::Attribute::Bool ("mni_decision", allow_interrupt),
    telemetry::Attribute::Int64 ("wm_slots_count", static_cast<int64_t> (included_vecs.size ())),
    telemetry::Attribute::Double ("max_semantic_overlap", overlap_star),
    telemetry::Attribute::Int64 ("ctx_window_size", static_cast<int64_t> (ctx_vecs.size ()))
  });
}

} // namespace cortext::operations
