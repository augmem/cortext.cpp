#include "cortext/operations/competition.hpp"

#include "../experimental_env.hpp"
#include "neuromodulator_internal.hpp"
#include "execution_cache_sidecar_internal.hpp"
#include "rif_state_internal.hpp"
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
#include <utility>
#include <vector>

namespace cortext::operations
{

namespace
{
using SteadyClock = std::chrono::steady_clock;

bool
ProfileWorkCountersEnabled ()
{
  return internal::experimental_env::Flag ("CORTEXT_PROFILE_WORK_COUNTERS");
}

double
ElapsedMillis (SteadyClock::time_point start)
{
  return std::chrono::duration<double, std::milli> (SteadyClock::now () - start)
      .count ();
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
  const auto start = SteadyClock::now ();
  auto sidecar = execution_cache_sidecar_internal::Ensure (
      context.GetProcessorContext ());
  if (!sidecar->rif_active_epoch.valid)
    {
      sidecar->rif_active_epoch.active_rows = rif_state_internal::CountRows (
          tx,
          "SELECT COUNT(*) AS row_count FROM rif_active_state a "
          "JOIN rif_recovery_clock c ON c.generation = a.generation "
          "WHERE c.singleton = 1");
      sidecar->rif_active_epoch.pending_rebuild = true;
    }
  const auto calibration_memory_ids
      = sidecar->rif_active_epoch.calibration_memory_ids;
  const auto result = rif_state_internal::AdvanceRecovery (
      tx, now_ts, recovery_time, calibration_memory_ids,
      sidecar->rif_active_epoch.limits.row_batch_size);
  sidecar->rif_active_epoch.row_batch_high_water = std::max (
      sidecar->rif_active_epoch.row_batch_high_water,
      result.maximum_statement_rows);
  sidecar->rif_active_epoch.calibration_memory_ids.clear ();
  rif_active_epoch_cache_internal::StageClock (
      sidecar->rif_active_epoch, result.clock.generation,
      result.clock.log_factor, result.clock.last_ts);
  rif_active_epoch_cache_internal::StageMemories (
      sidecar->rif_active_epoch, result.changed_memory_ids);
  if (result.generation_reset)
    sidecar->rif_active_epoch.active_rows = 0;
  else
    sidecar->rif_active_epoch.active_rows
        -= std::min (sidecar->rif_active_epoch.active_rows,
                     result.expired_rows);
  auto &p_ctx = context.GetProcessorContext ();
  p_ctx.retrieval_suppression_embedding_ids.clear ();
  p_ctx.retrieval_suppression_memory_ids.clear ();
  context.AddOperationTiming ("Competition.rif_recovery_active_sql",
                              ElapsedMillis (start));
  if (ProfileWorkCountersEnabled ())
    {
      const double touched
          = static_cast<double> (result.expired_rows + result.retired_rows + 1);
      context.AddOperationTiming ("Competition.rows_visited", touched);
      context.AddOperationTiming ("Competition.rows_touched", touched);
      context.AddOperationTiming ("Competition.rows_visited_activity",
                                  touched > 0.0 ? 1.0 : 0.0);
      context.AddOperationTiming ("Competition.rows_touched_activity",
                                  touched > 0.0 ? 1.0 : 0.0);
    }
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
                        rif_active_epoch_cache_internal::State &epoch,
                        std::size_t &active_rows,
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
      const auto memory_id = rif_state_internal::ResolveMemoryId (
          tx, loser.memory_id, loser.embedding_id);
      if (!memory_id.has_value ())
        continue;
      if (rif_state_internal::SuppressMemory (
              tx, *memory_id, total_supp, now_ts))
        ++active_rows;
      rif_active_epoch_cache_internal::StageMemory (
          epoch, *memory_id);
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
      context.AddOperationTiming ("Competition.score_candidates", 0.0);
      if (ProfileWorkCountersEnabled ())
        {
          context.AddOperationTiming ("Competition.candidate_count", 0.0);
          context.AddOperationTiming ("Competition.candidate_activity", 0.0);
        }
      return;
    }
  if (ProfileWorkCountersEnabled ())
    {
      context.AddOperationTiming (
          "Competition.candidate_count",
          static_cast<double> (retrieval_candidates.size ()));
      context.AddOperationTiming ("Competition.candidate_activity", 1.0);
    }
  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);
  const double recovery_time = core::RetrievalCompetitionRecoverySeconds (
                                   cfg.stability)
                               * 1000.0;
  ApplyRIFRecovery (context, tx, now_ts, recovery_time);
  const double competition_scale
      = neuromodulation::RetrievalCompetitionScale (p_ctx.neuromod_ne);

  const auto score_start = SteadyClock::now ();
  std::vector<Candidate> cands = ScoreCandidates (retrieval_candidates, x_ctx);
  context.AddOperationTiming ("Competition.score_candidates",
                              ElapsedMillis (score_start));
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
  auto sidecar = execution_cache_sidecar_internal::Ensure (p_ctx);
  ApplyLateralInhibition (winners, losers, radius, cfg.focus, cfg.sensitivity,
                          cfg.stability, competition_scale, now_ts, tx,
                          sidecar->rif_active_epoch,
                          sidecar->rif_active_epoch.active_rows,
                          suppressed_count);

  // Debug logging
  telemetry::LogDebug ("cortext.competition", {
    telemetry::Attribute::Int64 ("winner_count", static_cast<int64_t> (k)),
    telemetry::Attribute::Double ("inhibition_radius", radius),
    telemetry::Attribute::Double ("competition_scale", competition_scale),
    telemetry::Attribute::Int64 ("suppressed_count", static_cast<int64_t> (suppressed_count))
  });
}

} // namespace cortext::operations
