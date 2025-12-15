#include "cortext/operations/interrupt_gate.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/processor/processor_context.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

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
constexpr double kTauJMinusS = 0.15;
constexpr double kTauJPlusT = 0.3;
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
ComputeMniGateDecision::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();

  // Retrieve candidates (id -> embedding)
  const auto &candidates = context.GetRetrievedMemoryEmbeddings ();
  if (candidates.empty ())
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

  // Build context vectors for centroid from recent context; fallback to
  // included
  std::vector<Eigen::VectorXf> ctx_vecs;
  ctx_vecs.reserve (p_ctx.recent_context_embeddings.size ());
  for (const auto &v : p_ctx.recent_context_embeddings)
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

  // Thresholds
  const double retrieval_thresh = cortext::core::Clamp<double> (
      context.GetThresholdTDynamic (), constants::kNormalizedMin,
      constants::kNormalizedMax);

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

  // Duplicate suppression and derived thresholds
  const double dup_thresh
      = cortext::core::Lerp (kDupMin, kDupMax, F)
        * (kDupBase + kDupSlope
                            * cortext::core::Clamp<double> (
                                T, constants::kNormalizedMin,
                                constants::kNormalizedMax));
  const double tau_jaccard
      = cortext::core::Lerp (constants::kGainMedium, constants::kWeightHigh, F)
        * (constants::kNormalizedMax - kTauJMinusS * S)
        * (constants::kNormalizedMax + kTauJPlusT * T);
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

  const double tau_j_eff = tau_jaccard * M_refrac;
  const double tau_m_eff = tau_mu * M_refrac;

  // Boundary multiplier
  const double boundary_mult
      = cortext::core::Lerp (kBoundaryFMin, kBoundaryFMax, F)
        * cortext::core::Lerp (kBoundarySMin, kBoundarySMax, S);
  const bool at_boundary = context.GetAtBoundary ();

  // Iterate candidates and compute MU and overlaps
  double best_mu = -std::numeric_limits<double>::infinity ();
  double max_relevance = 0.0;
  double max_semantic_overlap = 0.0;

  for (const auto &kv : candidates)
    {
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

  // Compute Jaccard novelty against recent ID LRU if available.
  // A = candidate IDs; B = recent_ids_lru_ set.
  std::vector<long long> A;
  A.reserve (candidates.size ());
  for (const auto &kv : candidates)
    {
      A.push_back (kv.first);
    }
  const auto &Bset = p_ctx.recent_ids_lru_.GetIdSet ();
  double jaccard = 0.0;
  if (!A.empty ())
    {
      size_t intersection = 0;
      for (const auto &id : A)
        {
          if (Bset.find (id) != Bset.end ())
            {
              ++intersection;
            }
        }
      const size_t union_size = A.size () + Bset.size () - intersection;
      jaccard = (union_size > 0) ? (1.0
                                    - static_cast<double> (intersection)
                                          / static_cast<double> (union_size))
                                 : 0.0;
    }
  else
    {
      jaccard = 0.0; // empty candidate set -> no novelty
    }

  const bool allow_interrupt
      = (max_relevance >= retrieval_thresh)
        && ((jaccard >= tau_j_eff) || (best_mu >= tau_m_eff))
        && (max_semantic_overlap < dup_thresh)
        && (at_boundary || best_mu >= boundary_mult * tau_m_eff);

  context.SetInterruptAllowed (allow_interrupt);
  if (allow_interrupt)
    {
      p_ctx.last_interrupt_tick = p_ctx.signals_processed;
      p_ctx.drift_at_last_interrupt = p_ctx.drift_accum;  // Update drift baseline
    }

  // Diagnostics
  context.SetMniJaccard (jaccard);
  context.SetMniBestMu (best_mu);
  context.SetMniDupThresh (dup_thresh);
  context.SetMniTauJaccardEff (tau_j_eff);
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
}

} // namespace cortext::operations
