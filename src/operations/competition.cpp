#include "cortext/operations/competition.hpp"

#include "cortext/store/store.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/schema.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cortext::operations
{

namespace
{
/// @brief Computes the number of winners based on Focus.
inline int
ComputeWinnersK (double F)
{
  return static_cast<int> (std::round (
      core::Lerp (constants::kWinnersKMax, constants::kWinnersKMin, F)));
}

/// @brief Computes inhibition radius based on Focus.
inline double
InhibitionRadius (double F)
{
  return core::Lerp (constants::kInhibitionRadiusMin,
                     constants::kInhibitionRadiusMax, F);
}

/// @brief Computes suppression per retrieval based on Stability.
inline double
SuppressionPerRetrieval (double T, double winning_activation)
{
  const double base = core::Lerp (constants::kSuppressionBaseMax,
                                  constants::kSuppressionBaseMin, T);
  const double term = core::Clamp (constants::kNormalizedMax - winning_activation,
                                   constants::kNormalizedMin,
                                   constants::kNormalizedMax);
  return base * term;
}

/// @brief Computes lateral inhibition strength based on Focus and Sensitivity.
inline double
LateralInhibitionStrength (double F, double S)
{
  return F * (constants::kNormalizedMax + constants::kLateralInhibitionSMult * S);
}

/// @brief Computes number of competition iterations based on Focus.
inline int
CompetitionIterations (double F)
{
  return static_cast<int> (std::round (
      core::Lerp (constants::kCompetitionIterMin,
                  constants::kCompetitionIterMax, F)));
}

/// @brief Computes RIF recovery time in seconds based on Stability.
inline double
RecoveryTimeSeconds (double T)
{
  return core::Lerp (constants::kRecoveryTimeMinSeconds,
                     constants::kRecoveryTimeMaxSeconds, T);
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
ApplyRIFRecovery (Transaction &tx, long long now_ts, double recovery_time)
{
  tx.Execute ("UPDATE embeddings "
              "SET strength = MAX(0.0, strength + ("
              "  SELECT suppression * ("
              "    CASE "
              "      WHEN (? - ts) <= 0 THEN 0.0 "
              "      WHEN (? - ts) >= ? THEN 1.0 "
              "      ELSE ((? - ts) * 1.0 / ?) "
              "    END"
              "  )"
              "  FROM rif_state WHERE rif_state.embedding_id = "
              "embeddings.embedding_id"
              ")) "
              "WHERE EXISTS (SELECT 1 FROM rif_state WHERE "
              "rif_state.embedding_id = embeddings.embedding_id);",
              { now_ts, now_ts, recovery_time, now_ts, recovery_time });

  tx.Execute ("UPDATE rif_state "
              "SET suppression = MAX(0.0, suppression - (suppression * ("
              "    CASE "
              "      WHEN (? - ts) <= 0 THEN 0.0 "
              "      WHEN (? - ts) >= ? THEN 1.0 "
              "      ELSE ((? - ts) * 1.0 / ?) "
              "    END"
              "  ))), "
              "    ts = ?;",
              { now_ts, now_ts, recovery_time, now_ts, recovery_time, now_ts });
}

struct Candidate
{
  long long id;
  Eigen::VectorXf vec;
  double activation;
};

std::vector<Candidate>
ScoreCandidates (const std::unordered_map<long long, Eigen::VectorXf> &retrieved,
                 const Eigen::VectorXf &x_ctx)
{
  std::vector<Candidate> cands;
  cands.reserve (retrieved.size ());
  for (const auto &kv : retrieved)
    {
      const long long id = kv.first;
      const Eigen::VectorXf &v = kv.second;
      if (v.size () == 0 || v.size () != x_ctx.size ())
        {
          continue;
        }
      double act = core::CosineSimilarity (v, x_ctx);
      act = Clamp01 (act);
      cands.push_back (Candidate{ id, v, act });
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
                        double lat_strength, double iter_mult, double stability,
                        long long now_ts, Transaction &tx, int &suppressed_count)
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
          const double proximity = (1.0 - radius) > constants::kNormEpsilon
                                       ? (sim_lw - radius) / (1.0 - radius)
                                       : constants::kNormalizedMin;
          const double spr
              = SuppressionPerRetrieval (stability, winner.activation);
          total_supp += spr * lat_strength * proximity;
        }
      total_supp *= iter_mult;
      if (total_supp < 0.0)
        total_supp = 0.0;
      if (total_supp > constants::kOneHalf)
        total_supp = constants::kOneHalf;
      if (total_supp <= std::numeric_limits<double>::epsilon ())
        {
          continue;
        }
      // Note: embeddings row already exists from storage
      tx.Execute ("INSERT OR IGNORE INTO rif_state (embedding_id) VALUES (?);",
                  { loser.id });
      tx.Execute ("UPDATE rif_state SET suppression = suppression + ?, ts = ? "
                  "WHERE embedding_id = ?;",
                  { total_supp, now_ts, loser.id });
      tx.Execute ("UPDATE embeddings "
                  "SET strength = MAX(0.0, strength - ?) "
                  "WHERE embedding_id = ?;",
                  { total_supp, loser.id });
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
  if (retrieved.empty ())
    {
      return;
    }
  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);
  const double recovery_time = RecoveryTimeSeconds (cfg.stability);
  ApplyRIFRecovery (tx, now_ts, recovery_time);

  std::vector<Candidate> cands = ScoreCandidates (retrieved, x_ctx);
  if (cands.empty ())
    {
      return;
    }
  const int k = std::min (ComputeWinnersK (cfg.focus),
                          static_cast<int> (cands.size ()));
  const double radius = InhibitionRadius (cfg.focus);
  const double lat_strength
      = LateralInhibitionStrength (cfg.focus, cfg.sensitivity);
  const int iters = CompetitionIterations (cfg.focus);
  const double iter_mult = static_cast<double> (iters) / 3.0;
  std::vector<Candidate> winners (cands.begin (), cands.begin () + k);
  std::vector<Candidate> losers (cands.begin () + k, cands.end ());
  if (winners.empty () || losers.empty ())
    {
      return;
    }
  int suppressed_count = 0;
  ApplyLateralInhibition (winners, losers, radius, lat_strength, iter_mult,
                          cfg.stability, now_ts, tx, suppressed_count);

  // Debug logging
  telemetry::LogDebug ("cortext.competition", {
    telemetry::Attribute::Int64 ("winner_count", static_cast<int64_t> (k)),
    telemetry::Attribute::Double ("inhibition_radius", radius),
    telemetry::Attribute::Int64 ("suppressed_count", static_cast<int64_t> (suppressed_count))
  });
}

void
ApplyRetrievalCompetition::CollectSchema (
    cortext::store::SchemaRegistry &registry) const
{
  registry.Register ({
      20, // Competition & RIF
      "Retrieval Induced Forgetting (rif_state)",
      {
          "CREATE TABLE IF NOT EXISTS rif_state ("
          "  embedding_id INTEGER PRIMARY KEY,"
          "  suppression REAL NOT NULL DEFAULT 0.0,"
          "  ts INTEGER DEFAULT 0"
          ")",
      },
  });
}

} // namespace cortext::operations
