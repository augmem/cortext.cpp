#include "cortext/operations/consolidation.hpp"
#include "cortext/buffered_write_instruction.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include <algorithm>
#include <cmath>
#include <string>

namespace cortext::operations
{

void
EvaluateConsolidation::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();

  // Triggers (Algorithm 28)
  const uint64_t now_ts = context.GetSignal ().timestamp;
  const double rate_target = (p_ctx.rate_target > 0.0)
                                 ? p_ctx.rate_target
                                 : p_ctx.rate_target_prior;
  const double m_rate = p_ctx.m_rate;
  const bool trigger_rate
      = (rate_target > 0.0)
        && (m_rate < constants::kOneHalf * rate_target);

  const int interval_req = core::ConsolidationIntervalSeconds (cfg.stability);
  const bool has_last_cons = p_ctx.last_consolidation_ts > 0 && now_ts > 0;
  const bool trigger_interval
      = has_last_cons
            ? (static_cast<int64_t> (now_ts - p_ctx.last_consolidation_ts)
               > static_cast<int64_t> (interval_req))
            : false;

  const bool any_trigger = trigger_rate || trigger_interval;
  if (!any_trigger)
    {
      // No event when no triggers fire.
      return;
    }

  // Scheduling (Algorithm 28b)
  const int tokens_in_flight = context.GetTokensInFlight ();
  const int retrieval_queue_depth = context.GetRetrievalQueueDepth ();
  double idle_for = 0.0;
  if (p_ctx.last_retrieval_ts > 0 && now_ts > p_ctx.last_retrieval_ts)
    {
      idle_for = static_cast<double> (now_ts - p_ctx.last_retrieval_ts);
    }
  const int idle_required = core::IdleRequiredSeconds (cfg.stability);
  const bool idle_basic_ok
      = (tokens_in_flight == 0) && (retrieval_queue_depth == 0);
  // Interval trigger: require idle_basic_ok only (allow prompt start).
  // Rate trigger: require full idle window to avoid contention with retrieval.
  const bool idle_ok
      = trigger_interval
            ? idle_basic_ok
            : (idle_basic_ok
               && (idle_for >= static_cast<double> (idle_required)));

  const std::string reason
      = trigger_rate ? std::string ("rate") : std::string ("interval");
  const std::string action
      = idle_ok ? std::string ("start") : std::string ("defer");

  // Ensure events table exists.
  {
    BufferedWriteInstruction op;
    op.query = "CREATE TABLE IF NOT EXISTS consolidation_events ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT,"
               "ts INTEGER,"
               "reason TEXT,"
               "action TEXT,"
               "m_rate REAL,"
               "rate_target REAL,"
               "idle_for REAL,"
               "tokens_in_flight INTEGER,"
               "retrieval_queue_depth INTEGER"
               ");";
    context.AddWriteInstruction (std::move (op));
  }

  // Emit event row.
  {
    BufferedWriteInstruction op;
    op.query = "INSERT INTO consolidation_events("
               "ts, reason, action, m_rate, rate_target, idle_for, "
               "tokens_in_flight, retrieval_queue_depth"
               ") VALUES (?, ?, ?, ?, ?, ?, ?, ?);";
    op.params = { static_cast<long long> (now_ts),
                  reason,
                  action,
                  m_rate,
                  rate_target,
                  idle_for,
                  tokens_in_flight,
                  retrieval_queue_depth };
    context.AddWriteInstruction (std::move (op));
  }

  // Start signal: set flag and update last_consolidation_ts.
  if (idle_ok)
    {
      context.SetConsolidationShouldStart (true);
      p_ctx.last_consolidation_ts = now_ts;
    }
}

} // namespace cortext::operations

namespace cortext::operations
{

void
EnqueueExtractionJobs::Execute (OperationContext &context) const
{
  const auto &cfg = context.GetConfig ();
  const double F = cfg.focus;
  const double T = cfg.stability;
  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);

  // Ensure jobs table exists regardless of gating to allow
  // observability/tests.
  {
    BufferedWriteInstruction op;
    op.query = "CREATE TABLE IF NOT EXISTS extraction_jobs ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT,"
               "summary_id TEXT UNIQUE,"
               "prompt TEXT,"
               "status TEXT NOT NULL DEFAULT 'queued',"
               "created_at INTEGER"
               ");";
    context.AddWriteInstruction (std::move (op));
  }

  // Gate by stability (Algorithm 29c): extraction_enabled = (T > 0.2)
  if (T <= 0.2)
    {
      return;
    }

  const int min_cluster = core::MinClusterSizeForExtraction (F);
  const int batch_size = core::ExtractionBatchSize (T);

  // Insert-or-ignore queued jobs for eligible summaries, limited by batch
  // size. Prompt is built from algorithms.md template: includes Summary and
  // Context.
  {
    BufferedWriteInstruction op;
    op.query
        = "INSERT OR IGNORE INTO extraction_jobs(summary_id, prompt, status, "
          "created_at) "
          "SELECT cs.summary_id, "
          "       'Analyze this consolidated memory cluster and extract:\\n'"
          "       || '1. Named entities (people, places, organizations, "
          "concepts)\\n'"
          "       || '2. Relationships between entities (co-occurs, implies, "
          "contradicts)\\n'"
          "       || '3. Key themes or topics\\n\\n'"
          "       || 'Summary: ' || cs.summary_text || '\\n\\n'"
          "       || 'Context: ' || COALESCE(group_concat(cs2.source_text, "
          "char(10)), '') AS prompt, "
          "       'queued' AS status, "
          "       ?1 AS created_at "
          "FROM consolidation_summaries cs "
          "LEFT JOIN consolidation_sources cs2 "
          "  ON cs2.summary_id = cs.summary_id "
          "WHERE cs.cluster_size >= ?2 "
          "  AND NOT EXISTS (SELECT 1 FROM extraction_jobs ej "
          "                  WHERE ej.summary_id = cs.summary_id) "
          "GROUP BY cs.summary_id, cs.summary_text "
          "LIMIT ?3;";
    op.params = { now_ts, static_cast<long long> (min_cluster),
                  static_cast<long long> (batch_size) };
    context.AddWriteInstruction (std::move (op));
  }
}

} // namespace cortext::operations
namespace cortext::operations
{

void
ScoreConsolidation::Execute (OperationContext &context) const
{
  const auto &cfg = context.GetConfig ();
  const double F = cfg.focus;
  const double S = cfg.sensitivity;
  const double T = cfg.stability;
  const uint64_t now_ts = context.GetSignal ().timestamp;

  // Floor derived from knobs (no magic numbers).
  const double floor_cutoff = core::PeripheryCutoff (T);

  // Ensure memory_feedback exists (Alg 14/18 table shape).
  {
    BufferedWriteInstruction op;
    op.query = "CREATE TABLE IF NOT EXISTS memory_feedback ("
               "embedding_id INTEGER PRIMARY KEY,"
               "retrieved_count INTEGER NOT NULL DEFAULT 0,"
               "used_count INTEGER NOT NULL DEFAULT 0,"
               "contextual_gain REAL NOT NULL DEFAULT 0.0,"
               "use_frequency REAL NOT NULL DEFAULT 0.0,"
               "strength REAL NOT NULL DEFAULT 1.0"
               ")";
    context.AddWriteInstruction (std::move (op));
  }

  // Ensure consolidation_candidates table exists.
  {
    BufferedWriteInstruction op;
    op.query = "CREATE TABLE IF NOT EXISTS consolidation_candidates ("
               "embedding_id INTEGER PRIMARY KEY,"
               "score REAL NOT NULL,"
               "created_at INTEGER,"
               "reason TEXT"
               ")";
    context.AddWriteInstruction (std::move (op));
  }

  // Insert or update candidates whose score is below floor.
  // score = T*strength - F*redundancy + S*connectivity + T*stability
  // redundancy/connectivity/stability default to 0.0 when absent.
  {
    BufferedWriteInstruction op;
    op.query
        = "INSERT INTO consolidation_candidates(embedding_id, score, "
          "created_at, reason) "
          "SELECT mf.embedding_id, "
          "       ((?1 * mf.strength) "
          "        - (?2 * 0.0) "
          "        + (?3 * 0.0) "
          "        + (?4 * 0.0)) AS computed_score, "
          "       ?5 AS created_at, "
          "       'score_below_floor' AS reason "
          "FROM memory_feedback mf "
          "WHERE ((?1 * mf.strength) - (?2 * 0.0) + (?3 * 0.0) + (?4 * 0.0)) "
          "      < ?6 "
          "ON CONFLICT(embedding_id) DO UPDATE SET "
          "  score=excluded.score, "
          "  created_at=excluded.created_at, "
          "  reason=excluded.reason;";
    op.params = { T, F, S, T, static_cast<long long> (now_ts), floor_cutoff };
    context.AddWriteInstruction (std::move (op));
  }
}

} // namespace cortext::operations
