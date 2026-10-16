#include "cortext/operations/reconsolidation.hpp"
#include "historical_surface_search_cache_internal.hpp"
#include "constructive_recall_internal.hpp"
#include "retrieval_trace_state.hpp"
#include "neuromodulator_internal.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <any>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <string>
#include <vector>

namespace cortext::operations
{

namespace
{
using SteadyClock = std::chrono::steady_clock;

double
ElapsedMillis (SteadyClock::time_point start)
{
  return std::chrono::duration<double, std::milli> (SteadyClock::now () - start)
      .count ();
}

/// @brief Normalizes a vector to unit length.
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

void
RefreshProcessorSurfaces (Transaction &tx, ProcessorContext *p_ctx,
                          long long memory_id, long long old_embedding_id,
                          long long new_embedding_id,
                          const Eigen::VectorXf &embedding,
                          bool current_surface_written)
{
  if (!p_ctx)
    {
      return;
    }

  auto cache_it = p_ctx->retrieval_surface_index.find (memory_id);
  if (cache_it != p_ctx->retrieval_surface_index.end ())
    {
      auto &entry = p_ctx->retrieval_surface_cache[cache_it->second];
      if (entry.embedding_id > 0)
        {
          p_ctx->retrieval_surface_embedding_index.erase (entry.embedding_id);
        }
      entry.embedding_id = new_embedding_id;
      entry.embedding = embedding;
      entry.embedding_norm = embedding.norm ();
      p_ctx->retrieval_surface_embedding_index[new_embedding_id]
          = cache_it->second;
    }

  if (old_embedding_id > 0 && old_embedding_id != new_embedding_id)
    {
      auto existing = p_ctx->retrieval_surface_embedding_index.find (
          old_embedding_id);
      if (existing == p_ctx->retrieval_surface_embedding_index.end ())
        {
          auto rows = tx.Execute (
              "SELECT memory_id FROM memories "
              "WHERE embedding_id = ? AND memory_id != ? "
              "ORDER BY memory_id ASC LIMIT 1",
              { old_embedding_id, memory_id });
          if (!rows.empty () && rows[0].count ("memory_id") == 1)
            {
              const long long sibling_memory_id = store::AnyToLongLong (
                  rows[0].at ("memory_id")).value_or (0);
              auto sibling_it = p_ctx->retrieval_surface_index.find (
                  sibling_memory_id);
              if (sibling_memory_id > 0
                  && sibling_it != p_ctx->retrieval_surface_index.end ())
                {
                  p_ctx->retrieval_surface_embedding_index[old_embedding_id]
                      = sibling_it->second;
                }
            }
        }
    }

  if (current_surface_written)
    {
      historical_surface_search_cache_internal::UpsertCurrent (
          *p_ctx,
          { new_embedding_id, memory_id, 0, std::string (), std::string (),
            embedding });
    }
  retrieval_trace::RecordSurfaceUpsert (
      memory_id, new_embedding_id, ToFloatVector (embedding));
}

long long
LoadCurrentSurfaceEmbeddingId (Transaction &tx, long long memory_id,
                               long long fallback_embedding_id)
{
  const auto latest = constructive_recall::LoadLatestReconstruction (
      tx, memory_id);
  auto rows = tx.Execute (
      "SELECT embedding_id, created_at "
      "FROM current_memory_embeddings WHERE memory_id = ?",
      { memory_id });
  if (!rows.empty () && rows[0].count ("embedding_id") == 1)
    {
      const long long embedding_id = store::AnyToLongLong (
          rows[0].at ("embedding_id")).value_or (0);
      const bool current_surface_fresh
          = !latest.has_value () || embedding_id == latest->embedding_id;
      if (embedding_id > 0 && current_surface_fresh)
        {
          return embedding_id;
        }
    }
  if (latest.has_value () && latest->embedding_id > 0)
    {
      return latest->embedding_id;
    }
  return fallback_embedding_id;
}

long long
WriteEmbeddingInPlaceOrFork (Transaction &tx, long long memory_id,
                             long long embedding_id,
                             const Eigen::VectorXf &embedding,
                             long long timestamp,
                             ProcessorContext *p_ctx = nullptr)
{
  if (p_ctx != nullptr)
    {
      // Replacing or forking a joined base row changes both the vector and its
      // join metadata. Fall back to authoritative SQL rather than risk a
      // partially updated pre-filter population.
      historical_surface_search_cache_internal::
          InvalidateHistoricalPreserveCurrent (*p_ctx);
    }
  auto ref_rows = tx.Execute (
      "SELECT "
      "  (SELECT COUNT(*) FROM memories WHERE embedding_id = ?) "
      "+ (SELECT COUNT(*) FROM signals WHERE embedding_id = ?) "
      "+ (SELECT COUNT(*) FROM memory_reconstructions "
      "   WHERE embedding_id = ?) AS cnt",
      { embedding_id, embedding_id, embedding_id });
  long long ref_count = 0;
  if (!ref_rows.empty () && ref_rows[0].count ("cnt") == 1)
    {
      ref_count = store::AnyToLongLong (ref_rows[0].at ("cnt")).value_or (0);
    }
  if (ref_count <= 1)
    {
      tx.Execute ("DELETE FROM embeddings WHERE embedding_id = ?",
                  { embedding_id });
      tx.Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { embedding_id, ToFloatVector (embedding), timestamp });
      tx.Execute (
          "DELETE FROM current_memory_embeddings WHERE memory_id = ?",
          { memory_id });
      tx.Execute (
          "INSERT INTO current_memory_embeddings("
          "memory_id, embedding, embedding_id, created_at"
          ") VALUES (?, ?, ?, ?)",
          { memory_id, ToFloatVector (embedding), embedding_id, timestamp });
      RefreshProcessorSurfaces (tx, p_ctx, memory_id, embedding_id,
                                embedding_id, embedding, true);
      return embedding_id;
    }

  tx.Execute ("INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
              { ToFloatVector (embedding), timestamp });
  auto id_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
  long long replacement_embedding_id = 0;
  if (!id_rows.empty () && id_rows[0].count ("id") == 1)
    {
      replacement_embedding_id
          = store::AnyToLongLong (id_rows[0].at ("id")).value_or (0);
    }
  if (replacement_embedding_id > 0)
    {
      tx.Execute ("UPDATE memories SET embedding_id = ? WHERE memory_id = ?",
                  { replacement_embedding_id, memory_id });
      tx.Execute (
          "DELETE FROM current_memory_embeddings WHERE memory_id = ?",
          { memory_id });
      tx.Execute (
          "INSERT INTO current_memory_embeddings("
          "memory_id, embedding, embedding_id, created_at"
          ") VALUES (?, ?, ?, ?)",
          { memory_id, ToFloatVector (embedding), replacement_embedding_id,
            timestamp });
      RefreshProcessorSurfaces (tx, p_ctx, memory_id, embedding_id,
                                replacement_embedding_id, embedding, true);
    }
  return replacement_embedding_id;
}

struct RetrievedReconCandidate
{
  long long memory_id = 0;
  long long embedding_id = 0;
  Eigen::VectorXf embedding;
};

} // namespace

void
ApplyReconsolidation::Execute (OperationContext &context, Transaction &tx) const
{
  if (context.GetSignal ().retention == Retention::Ephemeral)
    {
      return;
    }
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
  const auto &retrieved_records = context.GetRetrievedMemoryCandidates ();
  std::vector<RetrievedReconCandidate> retrieved_candidates;
  if (!retrieved_records.empty ())
    {
      retrieved_candidates.reserve (retrieved_records.size ());
      for (const auto &candidate : retrieved_records)
        {
          if (candidate.memory_id > 0 && candidate.embedding_id > 0
              && candidate.embedding.size () > 0)
            {
              retrieved_candidates.push_back (
                  { candidate.memory_id, candidate.embedding_id,
                    candidate.embedding });
            }
        }
    }
  else
    {
      retrieved_candidates.reserve (retrieved.size ());
      for (const auto &kv : retrieved)
        {
          if (kv.first > 0 && kv.second.size () > 0)
            {
              retrieved_candidates.push_back ({ 0, kv.first, kv.second });
            }
        }
    }
  if (retrieved_candidates.empty ())
    {
      return;
    }

  const double S = cfg.sensitivity;
  const double T = cfg.stability;
  const double F = cfg.focus;
  const double recon_gain = core::ReconsolidationGain (T);
  const double recon_mod_scale
      = neuromodulation::ReconsolidationScale (p_ctx.neuromod_ach);
  const double tau_labile = core::TauLabile (T);
  const double lability_susc = core::LabilitySusceptibility (S, T);
  const double drift_skip_epsilon = core::ReconsolidationDriftSkipEpsilon (
      F, S, T);
  const int primary_reconstruction_limit
      = core::ReconsolidationPrimaryReconstructionLimit (F, S, T);
  const double uncertainty_relevance_weight
      = core::ReconsolidationUncertaintyRelevanceWeight (F, S, T);
  const constructive_recall::ReconstructionUpdatePolicy
      reconstruction_update_policy {
        core::ReconstructionHistoryLimit (F, S, T),
        core::ReconstructionPruneBatchLimit (F, S, T),
        core::ReconstructionMinUpdateIntervalMs (F, S, T),
        true
      };

  double max_drift = 0.0;
  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);

  int primary_reconstruction_considered = 0;

  for (const auto &retrieved_candidate : retrieved_candidates)
    {
      const long long embedding_id = retrieved_candidate.embedding_id;
      const Eigen::VectorXf &m = retrieved_candidate.embedding;
      if (m.size () == 0 || m.size () != u_cur.size ())
        {
          continue;
        }

      long long memory_id = 0;
      double stored_lability = 0.0;
      long long lability_ts = 0;
      const auto metadata_start = SteadyClock::now ();
      auto mem_rows = retrieved_candidate.memory_id > 0
                          ? tx.Execute (
                                "SELECT memory_id, lability_state, "
                                "lability_ts "
                                "FROM memories WHERE memory_id = ?",
                                { retrieved_candidate.memory_id })
                          : tx.Execute (
                                "SELECT memory_id, lability_state, "
                                "lability_ts "
                                "FROM memories WHERE embedding_id = ?",
                                { embedding_id });
      context.AddOperationTiming ("Reconsolidation.metadata_sql",
                                  ElapsedMillis (metadata_start));
      if (!mem_rows.empty ())
        {
          auto it_mem = mem_rows[0].find ("memory_id");
          if (it_mem != mem_rows[0].end ())
            {
              if (it_mem->second.type () == typeid (long long))
                {
                  memory_id = std::any_cast<long long> (it_mem->second);
                }
              else if (it_mem->second.type () == typeid (int))
                {
                  memory_id = static_cast<long long> (
                      std::any_cast<int> (it_mem->second));
                }
            }
          auto state_it = mem_rows[0].find ("lability_state");
          if (state_it != mem_rows[0].end ())
            {
              if (state_it->second.type () == typeid (double))
                {
                  stored_lability = std::any_cast<double> (state_it->second);
                }
              else if (state_it->second.type () == typeid (float))
                {
                  stored_lability = static_cast<double> (
                      std::any_cast<float> (state_it->second));
                }
            }
          auto ts_it = mem_rows[0].find ("lability_ts");
          if (ts_it != mem_rows[0].end ())
            {
              if (ts_it->second.type () == typeid (long long))
                {
                  lability_ts = std::any_cast<long long> (ts_it->second);
                }
              else if (ts_it->second.type () == typeid (int))
                {
                  lability_ts = std::any_cast<int> (ts_it->second);
                }
            }
        }
      if (memory_id <= 0)
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

      double current_lability = lability_susc;
      if (lability_ts > 0 && now_ts > lability_ts)
        {
          const double dt_s
              = static_cast<double> (now_ts - lability_ts) / 1000.0;
          const double base
              = (stored_lability > 0.0) ? stored_lability : lability_susc;
          current_lability
              = base
                * std::exp (-dt_s
                            / std::max (tau_labile,
                                        constants::kNormEpsilon));
        }
      else if (stored_lability > 0.0)
        {
          current_lability = stored_lability;
        }

      const double drift_mag = core::ReconsolidationPrimaryDriftMagnitude (
          F, S, T, current_lability, contextual_relevance, recon_mod_scale);
      max_drift = std::max (max_drift, drift_mag);

      // Note: embeddings row already exists - no need to ensure it exists

      // v2: Update lability_state, original_centroid, and lability_ts in
      // memories table (merged from memory_feedback)
      const auto lability_update_start = SteadyClock::now ();
      tx.Execute (
          "UPDATE memories "
          "SET lability_state = COALESCE(?, 0.0), "
          "    original_centroid = COALESCE(original_centroid, ?), "
          "    lability_ts = ? "
          "WHERE memory_id = ?",
          { current_lability, ToFloatVector (u_m), now_ts, memory_id });
      context.AddOperationTiming (
          "Reconsolidation.primary_lability_update_sql",
          ElapsedMillis (lability_update_start));

      if (drift_mag < drift_skip_epsilon)
        {
          continue; // No embedding update when drift too small
        }

      // Blend toward current context and normalize.
      Eigen::VectorXf blended = static_cast<float> (1.0 - drift_mag) * u_m
                                + static_cast<float> (drift_mag) * u_cur;
      blended = Unit (blended);

      const bool persist_primary_reconstruction
          = primary_reconstruction_considered < primary_reconstruction_limit;
      if (persist_primary_reconstruction)
        {
          ++primary_reconstruction_considered;
        }
      if (persist_primary_reconstruction && constructive_recall::Disabled ())
        {
          const auto primary_append_start = SteadyClock::now ();
          WriteEmbeddingInPlaceOrFork (tx, memory_id, embedding_id, blended,
                                       now_ts, &p_ctx);
          context.AddOperationTiming ("Reconsolidation.primary_append",
                                      ElapsedMillis (primary_append_start));
        }
      else if (persist_primary_reconstruction)
        {
          const auto primary_append_start = SteadyClock::now ();
          const auto blob_id
              = constructive_recall::LoadCurrentBlobId (tx, memory_id);
          const double uncertainty
              = core::Clamp (current_lability
                                 * (1.0 - uncertainty_relevance_weight
                                                * contextual_relevance),
                             0.0, 1.0);
          constructive_recall::AppendReconstructionWithEmbedding (
              tx, memory_id, blended, blob_id, now_ts, uncertainty,
              "reconsolidation", 1.0, contextual_relevance,
              reconstruction_update_policy, &p_ctx);
          if (auto current = constructive_recall::LoadCurrentEmbedding (
                  tx, memory_id, embedding_id,
                  static_cast<int> (blended.size ())))
            {
              const long long current_embedding_id = LoadCurrentSurfaceEmbeddingId (
                  tx, memory_id, embedding_id);
              const bool current_surface_written
                  = reconstruction_update_policy.update_current_surface
                    && !constructive_recall::
                           CurrentSurfaceWritesDisabled ();
              RefreshProcessorSurfaces (tx, &p_ctx, memory_id, embedding_id,
                                        current_embedding_id, *current,
                                        current_surface_written);
            }
          context.AddOperationTiming ("Reconsolidation.primary_append",
                                      ElapsedMillis (primary_append_start));
        }
    }

  (void)max_drift;

  telemetry::LogDebug ("cortext.reconsolidation", {
    telemetry::Attribute::Double ("recon_gain", recon_gain),
    telemetry::Attribute::Double ("recon_mod_scale", recon_mod_scale),
    telemetry::Attribute::Double ("max_drift", max_drift)
  });
}

} // namespace cortext::operations
