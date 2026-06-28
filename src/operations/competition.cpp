#include "cortext/operations/competition.hpp"

#include "neuromodulator_internal.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cortext::operations
{

namespace
{
using SteadyClock = std::chrono::steady_clock;
constexpr std::size_t kCompetitionSqlChunkSize = 400;

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

/// @brief Computes suppression per retrieval based on Stability.
inline double
SuppressionPerRetrieval (double F, double S, double T,
                         double winning_activation)
{
  const double base = core::RetrievalCompetitionSuppressionBase (F, S, T);
  const double term = core::Clamp (constants::kNormalizedMax - winning_activation,
                                   constants::kNormalizedMin,
                                   constants::kNormalizedMax);
  return base * term;
}

/// @brief Clamps a value to [0, 1].
inline double
Clamp01 (double v)
{
  if (v < constants::kNormalizedMin)
    return constants::kNormalizedMin;
  if (v > constants::kNormalizedMax)
    return constants::kNormalizedMax;
  return v;
}
void
ApplyRIFRecovery (OperationContext &context, Transaction &tx, long long now_ts,
                  double recovery_time)
{
  auto &active_ids
      = context.GetProcessorContext ().retrieval_suppression_embedding_ids;
  auto &active_memory_ids
      = context.GetProcessorContext ().retrieval_suppression_memory_ids;
  if (active_ids.empty () && active_memory_ids.empty ())
    {
      return;
    }

  const auto start = SteadyClock::now ();
  std::vector<long long> memory_ids (active_memory_ids.begin (),
                                     active_memory_ids.end ());
  std::unordered_set<long long> still_active_memory_ids;
  still_active_memory_ids.reserve (memory_ids.size ());

  for (std::size_t begin = 0; begin < memory_ids.size ();
       begin += kCompetitionSqlChunkSize)
    {
      const std::size_t end
          = std::min (memory_ids.size (), begin + kCompetitionSqlChunkSize);
      const std::string placeholders = Placeholders (end - begin);

      std::vector<std::any> update_params;
      update_params.reserve ((end - begin) + 12);
      update_params.push_back (now_ts);
      update_params.push_back (now_ts);
      update_params.push_back (recovery_time);
      update_params.push_back (now_ts);
      update_params.push_back (recovery_time);
      update_params.push_back (now_ts);
      update_params.push_back (now_ts);
      update_params.push_back (recovery_time);
      update_params.push_back (now_ts);
      update_params.push_back (recovery_time);
      update_params.push_back (now_ts);
      for (std::size_t i = begin; i < end; ++i)
        {
          update_params.push_back (memory_ids[i]);
        }

      tx.Execute (
          "UPDATE memories "
          "SET strength = MAX(0.0, strength + ("
          "  suppression * ("
          "    CASE "
          "      WHEN (? - COALESCE(suppression_ts, 0)) <= 0 THEN 0.0 "
          "      WHEN (? - COALESCE(suppression_ts, 0)) >= ? THEN 1.0 "
          "      ELSE ((? - COALESCE(suppression_ts, 0)) * 1.0 / ?) "
          "    END"
          "  )"
          ")), "
          "suppression = MAX(0.0, suppression - (suppression * ("
          "    CASE "
          "      WHEN (? - COALESCE(suppression_ts, 0)) <= 0 THEN 0.0 "
          "      WHEN (? - COALESCE(suppression_ts, 0)) >= ? THEN 1.0 "
          "      ELSE ((? - COALESCE(suppression_ts, 0)) * 1.0 / ?) "
          "    END"
          "  ))), "
          "suppression_ts = ? "
          "WHERE memory_id IN (" + placeholders + ") "
          "  AND COALESCE(suppression, 0.0) > 0.0",
          update_params);

      auto rows = tx.Execute (
          "SELECT memory_id FROM memories "
          "WHERE memory_id IN (" + placeholders + ") "
          "  AND COALESCE(suppression, 0.0) > 1e-9",
          ChunkParams (memory_ids, begin, end));
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
  active_memory_ids.swap (still_active_memory_ids);

  std::vector<long long> ids (active_ids.begin (), active_ids.end ());
  std::unordered_set<long long> still_active;
  still_active.reserve (ids.size ());

  for (std::size_t begin = 0; begin < ids.size ();
       begin += kCompetitionSqlChunkSize)
    {
      const std::size_t end
          = std::min (ids.size (), begin + kCompetitionSqlChunkSize);
      const std::string placeholders = Placeholders (end - begin);

      std::vector<std::any> update_params;
      update_params.reserve ((end - begin) + 12);
      update_params.push_back (now_ts);
      update_params.push_back (now_ts);
      update_params.push_back (recovery_time);
      update_params.push_back (now_ts);
      update_params.push_back (recovery_time);
      update_params.push_back (now_ts);
      update_params.push_back (now_ts);
      update_params.push_back (recovery_time);
      update_params.push_back (now_ts);
      update_params.push_back (recovery_time);
      update_params.push_back (now_ts);
      for (std::size_t i = begin; i < end; ++i)
        {
          update_params.push_back (ids[i]);
        }

      tx.Execute (
          "UPDATE memories "
          "SET strength = MAX(0.0, strength + ("
          "  suppression * ("
          "    CASE "
          "      WHEN (? - COALESCE(suppression_ts, 0)) <= 0 THEN 0.0 "
          "      WHEN (? - COALESCE(suppression_ts, 0)) >= ? THEN 1.0 "
          "      ELSE ((? - COALESCE(suppression_ts, 0)) * 1.0 / ?) "
          "    END"
          "  )"
          ")), "
          "suppression = MAX(0.0, suppression - (suppression * ("
          "    CASE "
          "      WHEN (? - COALESCE(suppression_ts, 0)) <= 0 THEN 0.0 "
          "      WHEN (? - COALESCE(suppression_ts, 0)) >= ? THEN 1.0 "
          "      ELSE ((? - COALESCE(suppression_ts, 0)) * 1.0 / ?) "
          "    END"
          "  ))), "
          "suppression_ts = ? "
          "WHERE embedding_id IN (" + placeholders + ") "
          "  AND COALESCE(suppression, 0.0) > 0.0",
          update_params);

      auto rows = tx.Execute (
          "SELECT embedding_id FROM memories "
          "WHERE embedding_id IN (" + placeholders + ") "
          "  AND COALESCE(suppression, 0.0) > 1e-9",
          ChunkParams (ids, begin, end));
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

  active_ids.swap (still_active);
  context.AddOperationTiming ("Competition.rif_recovery_active_sql",
                              ElapsedMillis (start));
}

struct Candidate
{
  long long memory_id = 0;
  long long embedding_id = 0;
  Eigen::VectorXf vec;
  double activation = 0.0;
};

std::vector<Candidate>
ScoreCandidates (const std::vector<Candidate> &retrieved,
                 const Eigen::VectorXf &x_ctx)
{
  std::vector<Candidate> cands;
  cands.reserve (retrieved.size ());
  for (const auto &candidate : retrieved)
    {
      const Eigen::VectorXf &v = candidate.vec;
      if (v.size () == 0 || v.size () != x_ctx.size ())
        {
          continue;
        }
      double act = core::CosineSimilarity (v, x_ctx);
      act = Clamp01 (act);
      Candidate scored = candidate;
      scored.activation = act;
      cands.push_back (std::move (scored));
    }
  std::sort (cands.begin (), cands.end (),
             [] (const Candidate &a, const Candidate &b) {
               return a.activation > b.activation;
             });
  return cands;
}

void
ApplyLateralInhibition (const std::vector<Candidate> &winners,
                        const std::vector<Candidate> &losers, double radius,
                        double focus, double sensitivity, double stability,
                        double competition_scale,
                        long long now_ts, Transaction &tx,
                        std::unordered_set<long long> &active_ids,
                        std::unordered_set<long long> &active_memory_ids,
                        int &suppressed_count)
{
  for (const auto &loser : losers)
    {
      double total_supp = 0.0;
      for (const auto &winner : winners)
        {
          double sim_lw = core::CosineSimilarity (loser.vec, winner.vec);
          sim_lw = Clamp01 (sim_lw);
          if (sim_lw < radius)
            {
              continue;
            }
          const double spr = SuppressionPerRetrieval (
              focus, sensitivity, stability, winner.activation);
          total_supp += spr * competition_scale;
        }
      if (total_supp <= std::numeric_limits<double>::epsilon ())
        {
          continue;
        }
      // v2: RIF state merged into memories table (suppression, suppression_ts)
      tx.Execute ("UPDATE memories "
                  "SET suppression = suppression + ?, "
                  "    suppression_ts = ?, "
                  "    suppression_count = suppression_count + 1, "
                  "    strength = MAX(0.0, strength - ?) "
                  + std::string (loser.memory_id > 0
                                     ? "WHERE memory_id = ?;"
                                     : "WHERE embedding_id = ?;"),
                  { total_supp, now_ts, total_supp,
                    loser.memory_id > 0 ? loser.memory_id
                                        : loser.embedding_id });
      if (loser.memory_id > 0)
        {
          active_memory_ids.insert (loser.memory_id);
        }
      else if (loser.embedding_id > 0)
        {
          active_ids.insert (loser.embedding_id);
        }
      ++suppressed_count;
    }
}

} // namespace

void
ApplyRetrievalCompetition::Execute (OperationContext &context,
                                    Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  if (p_ctx.recent_context_embeddings.empty ())
    {
      return;
    }
  const Eigen::VectorXf &x_ctx = p_ctx.recent_context_embeddings.back ();
  const auto &retrieved = context.GetRetrievedMemoryEmbeddings ();
  const auto &retrieved_records = context.GetRetrievedMemoryCandidates ();
  std::vector<Candidate> retrieval_candidates;
  if (!retrieved_records.empty ())
    {
      retrieval_candidates.reserve (retrieved_records.size ());
      for (const auto &candidate : retrieved_records)
        {
          if (candidate.embedding_id > 0 && candidate.embedding.size () > 0)
            {
              retrieval_candidates.push_back (
                  { candidate.memory_id, candidate.embedding_id,
                    candidate.embedding, 0.0 });
            }
        }
    }
  else
    {
      retrieval_candidates.reserve (retrieved.size ());
      for (const auto &kv : retrieved)
        {
          if (kv.first > 0 && kv.second.size () > 0)
            {
              retrieval_candidates.push_back ({ 0, kv.first, kv.second, 0.0 });
            }
        }
    }
  if (retrieval_candidates.empty ())
    {
      return;
    }
  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);
  const double recovery_time = core::RetrievalCompetitionRecoverySeconds (
                                   cfg.stability)
                               * 1000.0;
  ApplyRIFRecovery (context, tx, now_ts, recovery_time);
  const double competition_scale
      = neuromodulation::RetrievalCompetitionScale (p_ctx.neuromod_ne);

  std::vector<Candidate> cands = ScoreCandidates (retrieval_candidates, x_ctx);
  if (cands.empty ())
    {
      return;
    }
  const int k = std::min (core::RetrievalCompetitionWinnerCount (cfg.focus),
                          static_cast<int> (cands.size ()));
  const double radius = core::RetrievalCompetitionInhibitionRadius (cfg.focus);
  std::vector<Candidate> winners (cands.begin (), cands.begin () + k);
  std::vector<Candidate> losers (cands.begin () + k, cands.end ());
  if (winners.empty () || losers.empty ())
    {
      return;
    }
  int suppressed_count = 0;
  auto &active_suppression_ids = p_ctx.retrieval_suppression_embedding_ids;
  auto &active_suppression_memory_ids = p_ctx.retrieval_suppression_memory_ids;
  ApplyLateralInhibition (winners, losers, radius, cfg.focus, cfg.sensitivity,
                          cfg.stability, competition_scale, now_ts, tx,
                          active_suppression_ids,
                          active_suppression_memory_ids, suppressed_count);

  // Debug logging
  telemetry::LogDebug ("cortext.competition", {
    telemetry::Attribute::Int64 ("winner_count", static_cast<int64_t> (k)),
    telemetry::Attribute::Double ("inhibition_radius", radius),
    telemetry::Attribute::Double ("competition_scale", competition_scale),
    telemetry::Attribute::Int64 ("suppressed_count", static_cast<int64_t> (suppressed_count))
  });
}

} // namespace cortext::operations
