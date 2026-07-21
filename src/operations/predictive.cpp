#include "cortext/operations/predictive.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
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
#include <string>
#include <unordered_set>
#include <vector>

namespace cortext::operations
{

namespace
{
using SteadyClock = std::chrono::steady_clock;
constexpr std::size_t kPredictiveSqlChunkSize = 400;
constexpr int kPredictiveDbDecayIntervalSignals = 64;

struct PredictiveCandidate
{
  long long memory_id = 0;
  long long embedding_id = 0;
  Eigen::VectorXf embedding;
};

double
VectorNorm (const Eigen::VectorXf &v)
{
  double sum_sq = 0.0;
  for (Eigen::Index i = 0; i < v.size (); ++i)
    {
      const double x = static_cast<double> (v.coeff (i));
      sum_sq += x * x;
    }
  return std::sqrt (sum_sq);
}

inline Eigen::VectorXf
Unit (const Eigen::VectorXf &v)
{
  const double n = VectorNorm (v);
  if (n <= constants::kNormEpsilon)
    {
      return v;
    }
  return v / static_cast<float> (n);
}

double
ElapsedMillis (SteadyClock::time_point start)
{
  return std::chrono::duration<double, std::milli> (SteadyClock::now () - start)
      .count ();
}

std::string
Placeholders (std::size_t count)
{
  std::string text;
  text.reserve (count * 2);
  for (std::size_t i = 0; i < count; ++i)
    {
      if (i > 0)
        {
          text += ",";
        }
      text += "?";
    }
  return text;
}

std::vector<std::any>
ChunkParams (const std::vector<long long> &ids, std::size_t begin,
             std::size_t end)
{
  std::vector<std::any> params;
  params.reserve (end - begin);
  for (std::size_t i = begin; i < end; ++i)
    {
      params.push_back (ids[i]);
    }
  return params;
}

void
DecayActivePreActivation (OperationContext &context, Transaction &tx,
                          double pad)
{
  auto &active_ids = context.GetProcessorContext ()
                         .predictive_pre_activation_embedding_ids;
  auto &active_memory_ids = context.GetProcessorContext ()
                                .predictive_pre_activation_memory_ids;
  if (active_ids.empty () && active_memory_ids.empty ())
    {
      context.AddOperationTiming ("Predictive.decay_active_sql", 0.0);
      return;
    }

  const auto start = SteadyClock::now ();
  auto &p_ctx = context.GetProcessorContext ();
  const bool flush_db_decay
      = p_ctx.signals_processed <= 1
        || (p_ctx.signals_processed % kPredictiveDbDecayIntervalSignals) == 0;

  std::vector<long long> memory_ids (active_memory_ids.begin (),
                                     active_memory_ids.end ());
  std::unordered_set<long long> still_active_memory_ids;
  still_active_memory_ids.reserve (memory_ids.size ());

  for (std::size_t begin = 0; begin < memory_ids.size ();
       begin += kPredictiveSqlChunkSize)
    {
      const std::size_t end
          = std::min (memory_ids.size (), begin + kPredictiveSqlChunkSize);
      const std::string placeholders = Placeholders (end - begin);

      std::vector<std::any> update_params;
      update_params.reserve ((end - begin) + 2);
      update_params.push_back (pad);
      update_params.push_back (pad);
      for (std::size_t i = begin; i < end; ++i)
        {
          update_params.push_back (memory_ids[i]);
        }

      if (flush_db_decay)
        {
          tx.Execute ("UPDATE memories "
                      "SET pre_activation = CASE "
                      "  WHEN pre_activation * ? <= 1e-6 "
                      "    THEN 0.0 "
                      "  ELSE pre_activation * ? "
                      "END "
                      "WHERE memory_id IN (" + placeholders + ") "
                      "  AND pre_activation > 0.0",
                      update_params);
        }

      std::vector<long long> cache_miss_memory_ids;
      cache_miss_memory_ids.reserve (end - begin);
      for (std::size_t i = begin; i < end; ++i)
        {
          const long long memory_id = memory_ids[i];
          p_ctx.DecayRetrievalSurfacePreActivationByMemory (memory_id, pad);
          auto cache_it = p_ctx.retrieval_surface_index.find (memory_id);
          if (cache_it == p_ctx.retrieval_surface_index.end ())
            {
              if (flush_db_decay)
                {
                  cache_miss_memory_ids.push_back (memory_id);
                }
              else
                {
                  still_active_memory_ids.insert (memory_id);
                }
              continue;
            }
          const auto &entry
              = p_ctx.retrieval_surface_cache[cache_it->second];
          if (entry.pre_activation > 1e-6)
            {
              still_active_memory_ids.insert (memory_id);
            }
        }
      if (flush_db_decay && !cache_miss_memory_ids.empty ())
        {
          const std::string miss_placeholders = Placeholders (
              cache_miss_memory_ids.size ());
          auto rows = tx.Execute (
              "SELECT memory_id FROM memories "
              "WHERE memory_id IN (" + miss_placeholders + ") "
              "  AND pre_activation > 0.0 "
              "  AND pre_activation > 1e-6",
              ChunkParams (cache_miss_memory_ids, 0,
                           cache_miss_memory_ids.size ()));
          for (const auto &row : rows)
            {
              auto it = row.find ("memory_id");
              if (it == row.end ())
                {
                  continue;
                }
              const auto id = store::AnyToLongLong (it->second);
              if (id.has_value () && *id > 0)
                {
                  still_active_memory_ids.insert (*id);
                }
            }
        }
    }
  active_memory_ids.swap (still_active_memory_ids);

  std::vector<long long> ids (active_ids.begin (), active_ids.end ());
  std::unordered_set<long long> still_active;
  still_active.reserve (ids.size ());

  for (std::size_t begin = 0; begin < ids.size ();
       begin += kPredictiveSqlChunkSize)
    {
      const std::size_t end
          = std::min (ids.size (), begin + kPredictiveSqlChunkSize);
      const std::string placeholders = Placeholders (end - begin);

      std::vector<std::any> update_params;
      update_params.reserve ((end - begin) + 2);
      update_params.push_back (pad);
      update_params.push_back (pad);
      for (std::size_t i = begin; i < end; ++i)
        {
          update_params.push_back (ids[i]);
        }

      if (flush_db_decay)
        {
          tx.Execute ("UPDATE memories "
                      "SET pre_activation = CASE "
                      "  WHEN pre_activation * ? <= 1e-6 "
                      "    THEN 0.0 "
                      "  ELSE pre_activation * ? "
                      "END "
                      "WHERE embedding_id IN (" + placeholders + ") "
                      "  AND pre_activation > 0.0",
                      update_params);
        }
      std::vector<long long> cache_miss_ids;
      cache_miss_ids.reserve (end - begin);
      for (std::size_t i = begin; i < end; ++i)
        {
          const long long id = ids[i];
          p_ctx.DecayRetrievalSurfacePreActivationByEmbedding (id, pad);
          auto cache_it = p_ctx.retrieval_surface_embedding_index.find (id);
          if (cache_it == p_ctx.retrieval_surface_embedding_index.end ())
            {
              if (flush_db_decay)
                {
                  cache_miss_ids.push_back (id);
                }
              else
                {
                  still_active.insert (id);
                }
              continue;
            }
          const auto &entry
              = p_ctx.retrieval_surface_cache[cache_it->second];
          if (entry.pre_activation > 1e-6)
            {
              still_active.insert (id);
            }
        }
      if (flush_db_decay && !cache_miss_ids.empty ())
        {
          const std::string miss_placeholders = Placeholders (
              cache_miss_ids.size ());
          auto rows = tx.Execute (
              "SELECT embedding_id FROM memories "
              "WHERE embedding_id IN (" + miss_placeholders + ") "
              "  AND pre_activation > 0.0 "
              "  AND pre_activation > 1e-6",
              ChunkParams (cache_miss_ids, 0, cache_miss_ids.size ()));
          for (const auto &row : rows)
            {
              auto it = row.find ("embedding_id");
              if (it == row.end ())
                {
                  continue;
                }
              const auto id = store::AnyToLongLong (it->second);
              if (id.has_value () && *id > 0)
                {
                  still_active.insert (*id);
                }
            }
        }
    }

  active_ids.swap (still_active);
  context.AddOperationTiming ("Predictive.decay_active_sql",
                              ElapsedMillis (start));
}

inline double
Clamp01 (double v)
{
  if (v < constants::kNormalizedMin)
    return constants::kNormalizedMin;
  if (v > constants::kNormalizedMax)
    return constants::kNormalizedMax;
  return v;
}

} // namespace

void
ApplyPredictivePreActivation::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const double pad = core::PredictivePreActivationDecay (cfg.stability);

  // Keep predictive state transient without scanning every memory row.
  DecayActivePreActivation (context, tx, pad);

  // Need some recent context to estimate a trajectory.
  if (p_ctx.recent_context_embeddings.empty ())
    {
      return;
    }
  const int available
      = static_cast<int> (p_ctx.recent_context_embeddings.size ());
  const int want = core::PredictionHorizon (cfg.focus);
  const int take = std::max (1, std::min (available, want));

  // Predicted direction: mean of last `take` embeddings, then normalized.
  Eigen::VectorXf pred = Eigen::VectorXf::Zero (
      p_ctx.recent_context_embeddings.back ().size ());
  for (int i = available - take; i < available; ++i)
    {
      pred += p_ctx.recent_context_embeddings[static_cast<size_t> (i)];
    }
  if (pred.size () == 0)
    {
      return;
    }
  pred = Unit (pred);
  // Fallback to last context if degenerate.
  if (VectorNorm (pred) <= 1e-9f)
    {
      pred = Unit (p_ctx.recent_context_embeddings.back ());
    }

  const auto &retrieved = context.GetRetrievedMemoryEmbeddings ();
  const auto &retrieved_records = context.GetRetrievedMemoryCandidates ();
  std::vector<PredictiveCandidate> candidates;
  if (!retrieved_records.empty ())
    {
      candidates.reserve (retrieved_records.size ());
      for (const auto &record : retrieved_records)
        {
          if (record.memory_id > 0 && record.embedding_id > 0
              && record.embedding.size () > 0)
            {
              candidates.push_back (
                  { record.memory_id, record.embedding_id,
                    record.embedding });
            }
        }
    }
  else
    {
      candidates.reserve (retrieved.size ());
      for (const auto &kv : retrieved)
        {
          if (kv.first > 0 && kv.second.size () > 0)
            {
              candidates.push_back ({ 0, kv.first, kv.second });
            }
        }
    }
  if (candidates.empty ())
    {
      return;
    }

  const double conf_thresh = core::PredictionConfidenceThreshold (cfg.focus);
  const double base_delta = core::PredictiveBaseDelta (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const double update_rate_on_surprise
      = core::PredictiveSurpriseUpdateRate (cfg.sensitivity, cfg.stability);
  const double surp_sens
      = core::PredictiveSurpriseSensitivity (cfg.sensitivity, cfg.stability);

  // Optional surprise modulation from metrics (map to [0,1]).
  double surprise_01 = constants::kNormalizedMin;
  if (auto m = context.GetMetric (operations::Metric::surprise))
    {
      double v = *m;
      if (std::isnan (v) || std::isinf (v))
        {
          v = constants::kNormalizedMin;
        }
      if (v < constants::kNormalizedMin)
        {
          surprise_01 = Clamp01 (
              (v + constants::kNormalizedMax) / constants::kTwo);
        }
      else
        {
          surprise_01 = Clamp01 (v);
        }
    }

  int boost_count = 0;
  for (const auto &candidate : candidates)
    {
      const Eigen::VectorXf &vec = candidate.embedding;
      if (vec.size () == 0 || vec.size () != pred.size ())
        {
          continue;
        }
      double sim_pred = core::CosineSimilarity (pred, vec);
      sim_pred = Clamp01 (sim_pred);
      if (sim_pred < conf_thresh)
        {
          continue;
        }

      // Compute modulated delta; keep safe bounds.
      double delta = base_delta * (1.0 + constants::kQuarter * surp_sens
                                              * surprise_01);
      delta += update_rate_on_surprise * surprise_01;
      delta = core::Clamp (
          delta, 0.0,
          core::PredictiveDeltaCap (cfg.focus, cfg.sensitivity, cfg.stability));

      if (candidate.memory_id > 0)
        {
          tx.Execute (
              "UPDATE memories "
              "SET pre_activation = MIN(?, COALESCE(pre_activation, 0.0) * ? + ?) "
              "WHERE memory_id = ?;",
              { 1.0, pad, delta, candidate.memory_id });
          p_ctx.BoostRetrievalSurfacePreActivationByMemory (
              candidate.memory_id, pad, delta);
          p_ctx.predictive_pre_activation_memory_ids.insert (
              candidate.memory_id);
        }
      else
        {
          tx.Execute (
              "UPDATE memories "
              "SET pre_activation = MIN(?, COALESCE(pre_activation, 0.0) * ? + ?) "
              "WHERE embedding_id = ?;",
              { 1.0, pad, delta, candidate.embedding_id });
          p_ctx.BoostRetrievalSurfacePreActivationByEmbedding (
              candidate.embedding_id, pad, delta);
          p_ctx.predictive_pre_activation_embedding_ids.insert (
              candidate.embedding_id);
        }
      ++boost_count;
    }

  // Debug logging
  telemetry::LogDebug ("cortext.predictive", {
    telemetry::Attribute::Double ("conf_thresh", conf_thresh),
    telemetry::Attribute::Double ("pad", pad),
    telemetry::Attribute::Double ("base_delta", base_delta),
    telemetry::Attribute::Double ("update_rate_on_surprise", update_rate_on_surprise),
    telemetry::Attribute::Double ("surp_sens", surp_sens),
    telemetry::Attribute::Double ("surprise_01", surprise_01),
    telemetry::Attribute::Double ("prediction_norm", VectorNorm (pred)),
    telemetry::Attribute::Int64 ("boost_count", static_cast<int64_t> (boost_count))
  });
}

} // namespace cortext::operations
