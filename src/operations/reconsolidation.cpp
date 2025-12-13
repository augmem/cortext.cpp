#include "cortext/operations/reconsolidation.hpp"

#include "cortext/buffered_write_instruction.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/schema.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace cortext::operations
{
namespace
{
constexpr double kDriftClampMax = 0.3;
constexpr double kDriftClampMin = 0.0;
constexpr double kDriftSkipEpsilon = 0.001;
constexpr double kUncertaintyBumpCap = 0.2;
inline Eigen::VectorXf
Unit (const Eigen::VectorXf &v)
{
  const double n = v.norm ();
  if (n <= constants::kNormEpsilon)
    {
      return v;
    }
  return v / static_cast<float> (n);
}

inline std::vector<float>
ToFloatVector (const Eigen::VectorXf &v)
{
  std::vector<float> out;
  out.resize (static_cast<size_t> (v.size ()));
  for (int i = 0; i < v.size (); ++i)
    {
      out[static_cast<size_t> (i)] = v[i];
    }
  return out;
}
} // namespace

void
ApplyReconsolidation::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();

  // Require a current context embedding to drift toward.
  if (p_ctx.recent_context_embeddings.empty ())
    {
      return;
    }
  const Eigen::VectorXf &x_cur = p_ctx.recent_context_embeddings.back ();
  const Eigen::VectorXf u_cur = Unit (x_cur);

  const auto &retrieved = context.GetRetrievedMemoryEmbeddings ();
  if (retrieved.empty ())
    {
      return;
    }

  const double S = cfg.sensitivity;
  const double T = cfg.stability;
  const double recon_gain = core::ReconsolidationGain (T);
  // Ripple not used (no neighbor graph), kept for parity.
  (void)core::RippleDecay (T);
  const double lability_susc = core::LabilitySusceptibility (S, T);

  double max_drift = 0.0;
  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);

  for (const auto &kv : retrieved)
    {
      const long long embedding_id = kv.first;
      const Eigen::VectorXf &m = kv.second;
      if (m.size () == 0 || m.size () != u_cur.size ())
        {
          continue;
        }
      const Eigen::VectorXf u_m = Unit (m);
      // contextual_relevance ∈ [0,1]
      double contextual_relevance = core::CosineSimilarity (u_m, u_cur);
      if (contextual_relevance < constants::kNormalizedMin)
        contextual_relevance = constants::kNormalizedMin;
      if (contextual_relevance > constants::kNormalizedMax)
        contextual_relevance = constants::kNormalizedMax;

      // Without DB reads available here, assume current lability based on
      // susceptibility (elapsed≈0 at retrieval time).
      const double current_lability = lability_susc;

      double drift_mag
          = (1.0 - T) * S * current_lability * contextual_relevance;
      drift_mag *= recon_gain;
      // Safety clamp
      drift_mag
          = std::min (kDriftClampMax, std::max (kDriftClampMin, drift_mag));
      max_drift = std::max (max_drift, drift_mag);

      // Ensure feedback row exists.
      {
        BufferedWriteInstruction op;
        op.query = "INSERT OR IGNORE INTO memory_feedback (embedding_id) "
                   "VALUES (?)";
        op.params = { embedding_id };
        context.AddWriteInstruction (std::move (op));
      }

      // Update lability fields and set original_embedding if first-time.
      {
        BufferedWriteInstruction op;
        op.query = "UPDATE memory_feedback "
                   "SET original_embedding = COALESCE(original_embedding, ?), "
                   "    lability_state = ?, "
                   "    lability_ts = ? "
                   "WHERE embedding_id = ?";
        op.params
            = { ToFloatVector (u_m), current_lability, now_ts, embedding_id };
        context.AddWriteInstruction (std::move (op));
      }

      if (drift_mag < kDriftSkipEpsilon)
        {
          continue; // No embedding update when drift too small
        }

      // Blend toward current context and normalize.
      Eigen::VectorXf blended = static_cast<float> (1.0 - drift_mag) * u_m
                                + static_cast<float> (drift_mag) * u_cur;
      blended = Unit (blended);

      // Upsert into embeddings table.
      {
        BufferedWriteInstruction op;
        op.query
            = "INSERT OR REPLACE INTO embeddings (embedding_id, embedding) "
              "VALUES (?, ?)";
        op.params = { embedding_id, ToFloatVector (blended) };
        context.AddWriteInstruction (std::move (op));
      }
    }

  // Increase uncertainty proportional to max drift (cap 0.2).
  const double bump = std::min (kUncertaintyBumpCap, max_drift);
  p_ctx.u_t = core::Clamp (p_ctx.u_t + bump, constants::kNormalizedMin,
                           constants::kNormalizedMax);
}

void
ApplyReconsolidation::CollectSchema (cortext::store::SchemaRegistry &registry) const
{
  // Reconsolidation relies on memory_feedback and embeddings, which are core tables.
  // However, if we want to be explicit about table requirements or indexes for lability,
  // we could add them here. For now, core tables are handled by RegisterCoreSchema.
  (void)registry;
}

} // namespace cortext::operations
