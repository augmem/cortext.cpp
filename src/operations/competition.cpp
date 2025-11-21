#include "cortext/operations/competition.hpp"

#include "cortext/buffered_write_instruction.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
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
inline int
ComputeWinnersK (double F)
{
  // winners_k = round(lerp(7, 3, F))
  return static_cast<int> (std::round (core::Lerp (7.0, 3.0, F)));
}

inline double
InhibitionRadius (double F)
{
  // inhibition_radius = lerp(0.5, 0.85, F)
  return core::Lerp (0.5, 0.85, F);
}

inline double
SuppressionPerRetrieval (double T, double winning_activation)
{
  // suppression_per_retrieval = lerp(0.1, 0.01, T) * (1 - winning_activation)
  const double base = core::Lerp (0.1, 0.01, T);
  double term = 1.0 - winning_activation;
  if (term < 0.0)
    term = 0.0;
  if (term > 1.0)
    term = 1.0;
  return base * term;
}

inline double
LateralInhibitionStrength (double F, double S)
{
  // lateral_inhibition_strength = F * (1 + 0.5*S)
  return F * (1.0 + 0.5 * S);
}

inline int
CompetitionIterations (double F)
{
  // competition_iterations = round(lerp(3, 10, F))
  return static_cast<int> (std::round (core::Lerp (3.0, 10.0, F)));
}

inline double
RecoveryTimeSeconds (double T)
{
  // recovery_time_RIF = lerp(60, 600, T) seconds
  return core::Lerp (60.0, 600.0, T);
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
ApplyRetrievalCompetition::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();

  // Need current context embedding and retrieved candidates.
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

  // Ensure persistence tables exist (idempotent).
  {
    BufferedWriteInstruction op;
    op.query = "CREATE TABLE IF NOT EXISTS memory_feedback ("
               "  embedding_id INTEGER PRIMARY KEY,"
               "  retrieved_count INTEGER NOT NULL DEFAULT 0,"
               "  used_count INTEGER NOT NULL DEFAULT 0,"
               "  contextual_gain REAL NOT NULL DEFAULT 0.0,"
               "  use_frequency REAL NOT NULL DEFAULT 0.0,"
               "  strength REAL NOT NULL DEFAULT 1.0"
               ");";
    context.AddWriteInstruction (std::move (op));
  }
  {
    BufferedWriteInstruction op;
    op.query = "CREATE TABLE IF NOT EXISTS rif_state ("
               "  embedding_id INTEGER PRIMARY KEY,"
               "  suppression REAL NOT NULL DEFAULT 0.0,"
               "  ts INTEGER DEFAULT 0"
               ");";
    context.AddWriteInstruction (std::move (op));
  }

  // 1) Recovery step for all rows in rif_state based on elapsed time.
  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);
  const double recovery_time = RecoveryTimeSeconds (cfg.stability);
  {
    // Restore strength by suppression * frac; do nothing if no rif_state row.
    BufferedWriteInstruction op;
    op.query = "UPDATE memory_feedback "
               "SET strength = MAX(0.0, strength + ("
               "  SELECT suppression * ("
               "    CASE "
               "      WHEN (? - ts) <= 0 THEN 0.0 "
               "      WHEN (? - ts) >= ? THEN 1.0 "
               "      ELSE ((? - ts) * 1.0 / ?) "
               "    END"
               "  )"
               "  FROM rif_state WHERE rif_state.embedding_id = "
               "memory_feedback.embedding_id"
               ")) "
               "WHERE EXISTS (SELECT 1 FROM rif_state WHERE "
               "rif_state.embedding_id = memory_feedback.embedding_id);";
    op.params = { now_ts, now_ts, recovery_time, now_ts, recovery_time };
    context.AddWriteInstruction (std::move (op));
  }
  {
    // Decay rif_state.suppression by the same recovered fraction and update
    // ts.
    BufferedWriteInstruction op;
    op.query = "UPDATE rif_state "
               "SET suppression = MAX(0.0, suppression - (suppression * ("
               "    CASE "
               "      WHEN (? - ts) <= 0 THEN 0.0 "
               "      WHEN (? - ts) >= ? THEN 1.0 "
               "      ELSE ((? - ts) * 1.0 / ?) "
               "    END"
               "  ))), "
               "    ts = ?;";
    op.params = { now_ts, recovery_time, now_ts, recovery_time, now_ts };
    context.AddWriteInstruction (std::move (op));
  }

  // 2) Compute activations against current context for all candidates.
  struct Candidate
  {
    long long id;
    Eigen::VectorXf vec;
    double activation;
  };
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
  if (cands.empty ())
    {
      return;
    }

  std::sort (cands.begin (), cands.end (),
             [] (const Candidate &a, const Candidate &b) {
               return a.activation > b.activation;
             });

  const int k = std::min (ComputeWinnersK (cfg.focus),
                          static_cast<int> (cands.size ()));
  const double radius = InhibitionRadius (cfg.focus);
  const double lat_strength
      = LateralInhibitionStrength (cfg.focus, cfg.sensitivity);
  const int iters = CompetitionIterations (cfg.focus);
  const double iter_mult = static_cast<double> (iters) / 3.0; // mild scale

  // Winners are top-k; losers are the rest.
  std::vector<Candidate> winners (cands.begin (), cands.begin () + k);
  std::vector<Candidate> losers (cands.begin () + k, cands.end ());
  if (winners.empty () || losers.empty ())
    {
      return; // nothing to inhibit
    }

  // 3) Apply lateral inhibition to losers near any winner.
  for (const auto &loser : losers)
    {
      double total_supp = 0.0;
      for (const auto &winner : winners)
        {
          // Proximity in [0,1] based on cosine similarity to the winner,
          // thresholded by inhibition radius.
          double sim_lw = core::CosineSimilarity (loser.vec, winner.vec);
          sim_lw = Clamp01 (sim_lw);
          if (sim_lw < radius)
            {
              continue;
            }
          // Map similarity above radius → [0,1]
          const double proximity = (1.0 - radius) > constants::kNormEpsilon
                                       ? (sim_lw - radius) / (1.0 - radius)
                                       : constants::kNormalizedMin;
          // Per winner suppression contribution.
          const double spr
              = SuppressionPerRetrieval (cfg.stability, winner.activation);
          total_supp += spr * lat_strength * proximity;
        }
      // Scale by iterations multiplier; clamp to a safe cap.
      total_supp *= iter_mult;
      if (total_supp < 0.0)
        total_supp = 0.0;
      if (total_supp > constants::kOneHalf)
        total_supp = constants::kOneHalf; // safety to avoid large jumps

      if (total_supp <= std::numeric_limits<double>::epsilon ())
        {
          continue;
        }

      // Ensure rows exist, then persist suppression and reduce strength.
      {
        BufferedWriteInstruction op;
        op.query = "INSERT OR IGNORE INTO memory_feedback (embedding_id) "
                   "VALUES (?);";
        op.params = { loser.id };
        context.AddWriteInstruction (std::move (op));
      }
      {
        BufferedWriteInstruction op;
        op.query = "INSERT OR IGNORE INTO rif_state (embedding_id) "
                   "VALUES (?);";
        op.params = { loser.id };
        context.AddWriteInstruction (std::move (op));
      }
      {
        BufferedWriteInstruction op;
        op.query
            = "UPDATE rif_state SET suppression = suppression + ?, ts = ? "
              "WHERE embedding_id = ?;";
        op.params = { total_supp, now_ts, loser.id };
        context.AddWriteInstruction (std::move (op));
      }
      {
        BufferedWriteInstruction op;
        op.query = "UPDATE memory_feedback "
                   "SET strength = MAX(0.0, strength - ?) "
                   "WHERE embedding_id = ?;";
        op.params = { total_supp, loser.id };
        context.AddWriteInstruction (std::move (op));
      }
    }
}

} // namespace cortext::operations
