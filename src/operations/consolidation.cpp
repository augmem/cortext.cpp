#include "cortext/operations/consolidation.hpp"
#include "cortext/store/store.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/operations/extraction.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/schema.hpp"
#include <algorithm>
#include <any>
#include <cmath>
#include <string>

namespace cortext::operations
{

namespace
{

bool
CheckRateTrigger (double rate_target, double m_rate)
{
  return (rate_target > 0.0)
        && (m_rate < constants::kOneHalf * rate_target);
}

bool
CheckIntervalTrigger (uint64_t last_consolidation_ts, uint64_t now_ts,
                      int interval_req)
{
  const bool has_last_cons = last_consolidation_ts > 0 && now_ts > 0;
  return has_last_cons
            ? (static_cast<int64_t> (now_ts - last_consolidation_ts)
               > static_cast<int64_t> (interval_req))
            : false;
}

bool
CheckCapacityTrigger (Store *store, long long consolidation_threshold,
                      long long &db_size)
{
  db_size = 0;
  if (!store)
    {
      return false;
    }
  try
    {
      auto rows = store->Execute (
          "SELECT COUNT(*) AS c FROM embeddings", {});
      if (!rows.empty () && rows[0].count ("c") == 1)
        {
          const auto &v = rows[0].at ("c");
          if (v.type () == typeid (long long))
            db_size = std::any_cast<long long> (v);
        }
    }
  catch (...)
    {
      db_size = 0;
    }
  return (db_size > consolidation_threshold);
}

bool
CheckIdleCondition (int tokens_in_flight, int retrieval_queue_depth,
                    double idle_for, int idle_required, bool trigger_interval)
{
  const bool idle_basic_ok
      = (tokens_in_flight == 0) && (retrieval_queue_depth == 0);
  return trigger_interval
            ? idle_basic_ok
            : (idle_basic_ok
               && (idle_for >= static_cast<double> (idle_required)));
}

} // namespace

void
EvaluateConsolidation::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  Store *store = context.GetStore ();
  const uint64_t now_ts = context.GetSignal ().timestamp;
  const double rate_target = (p_ctx.rate_target > 0.0)
                                 ? p_ctx.rate_target
                                 : p_ctx.rate_target_prior;
  const double m_rate = p_ctx.m_rate;
  const bool trigger_rate = CheckRateTrigger (rate_target, m_rate);
  const int interval_req = core::ConsolidationIntervalSeconds (cfg.stability);
  const bool trigger_interval
      = CheckIntervalTrigger (p_ctx.last_consolidation_ts, now_ts, interval_req);
  long long db_size = 0;
  const long long consolidation_threshold
      = core::ConsolidationThresholdCount (cfg.stability);
  const bool trigger_capacity
      = CheckCapacityTrigger (store, consolidation_threshold, db_size);

  const bool any_trigger = trigger_rate || trigger_interval || trigger_capacity;
  if (!any_trigger)
    {
      return;
    }
  const int tokens_in_flight = context.GetTokensInFlight ();
  const int retrieval_queue_depth = context.GetRetrievalQueueDepth ();
  double idle_for = 0.0;
  if (p_ctx.last_retrieval_ts > 0 && now_ts > p_ctx.last_retrieval_ts)
    {
      idle_for = static_cast<double> (now_ts - p_ctx.last_retrieval_ts);
    }
  const int idle_required = core::IdleRequiredSeconds (cfg.stability);
  const bool idle_ok
      = CheckIdleCondition (tokens_in_flight, retrieval_queue_depth,
                            idle_for, idle_required, trigger_interval);

  // NOTE: consolidation_events table removed (undocumented).
  // Event logging removed - consolidation decisions are tracked via
  // last_consolidation_ts in processor_state.

  // Start signal: set flag and update last_consolidation_ts.
  if (idle_ok)
    {
      context.SetConsolidationShouldStart (true);
      p_ctx.last_consolidation_ts = now_ts;
    }
}

void
EvaluateConsolidation::CollectSchema (cortext::store::SchemaRegistry &registry) const
{
  // No-op: consolidation_events table removed (undocumented).
  // All tables now in core schema.cpp migration 0.
  (void)registry;
}

void
EnqueueExtractionJobs::Execute (OperationContext &context, Transaction &tx) const
{
  if (!context.GetConsolidationShouldStart ())
    {
      return;
    }

  const auto &requests = context.GetExtractionRequests ();
  if (requests.empty ())
    {
      return;
    }

  const auto &cfg = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();
  const uint64_t now_ts = context.GetSignal ().timestamp;

  // Check interval since last extraction.
  const int interval = core::ExtractionIntervalSeconds (cfg.stability);
  if (p_ctx.last_extraction_ts > 0
      && (now_ts - p_ctx.last_extraction_ts)
             < static_cast<uint64_t> (interval))
    {
      return;
    }

  // Respect max_per_cycle limit.
  const int max_per_cycle = core::MaxExtractionsPerCycle (cfg.stability);
  const int count = std::min (static_cast<int> (requests.size ()), max_per_cycle);

  // Invoke extraction callback if registered.
  auto *callback = context.GetExtractionCallback ();
  if (callback)
    {
      std::vector<operations::ExtractionRequest> batch (requests.begin (),
                                                        requests.begin () + count);
      (*callback) (batch);
    }

  p_ctx.last_extraction_ts = now_ts;
}

void
EnqueueExtractionJobs::CollectSchema (cortext::store::SchemaRegistry &registry) const
{
  // No-op: extraction_jobs table removed (undocumented).
  // All tables now in core schema.cpp migration 0.
  (void)registry;
}

void
ScoreConsolidation::Execute (OperationContext &context, Transaction &tx) const
{
  const auto &cfg = context.GetConfig ();
  const double F = cfg.focus;
  const double S = cfg.sensitivity;
  const double T = cfg.stability;
  const uint64_t now_ts = context.GetSignal ().timestamp;

  // Floor derived from knobs (no magic numbers).
  const double floor_cutoff = core::PeripheryCutoff (T);

  // Update connectivity metric from graph edge count (Section 9.2).
  // Connectivity = normalized count of edges where this embedding participates.
  // Normalized by max observed edge count to keep values in [0, 1].
  tx.Execute ("WITH edge_counts AS ("
              "  SELECT e.embedding_id, "
              "         (SELECT COUNT(*) FROM graph_edges ge "
              "          WHERE ge.source_id = 'emb:' || e.embedding_id "
              "             OR ge.target_id = 'emb:' || e.embedding_id) AS cnt "
              "  FROM embeddings e"
              "), max_cnt AS ("
              "  SELECT MAX(cnt) AS m FROM edge_counts WHERE cnt > 0"
              ") "
              "UPDATE embeddings SET connectivity = ("
              "  SELECT CASE WHEN (SELECT m FROM max_cnt) > 0 "
              "              THEN CAST(ec.cnt AS REAL) / (SELECT m FROM max_cnt) "
              "              ELSE 0.0 END "
              "  FROM edge_counts ec WHERE ec.embedding_id = "
              "embeddings.embedding_id"
              ");",
              {});

  // Insert or update candidates whose score is below floor.
  // score = T*strength - F*redundancy + S*connectivity + T*stability
  // Uses unified embeddings table which contains per-embedding state.
  tx.Execute ("INSERT INTO consolidation_candidates(embedding_id, score, "
              "created_at, reason) "
              "SELECT e.embedding_id, "
              "       ((?1 * COALESCE(e.strength, 1.0)) "
              "        - (?2 * COALESCE(e.redundancy, 0.0)) "
              "        + (?3 * COALESCE(e.connectivity, 0.0)) "
              "        + (?4 * COALESCE(e.stability, 0.0))) AS computed_score, "
              "       ?5 AS created_at, "
              "       'score_below_floor' AS reason "
              "FROM embeddings e "
              "WHERE ((?1 * COALESCE(e.strength, 1.0)) "
              "       - (?2 * COALESCE(e.redundancy, 0.0)) "
              "       + (?3 * COALESCE(e.connectivity, 0.0)) "
              "       + (?4 * COALESCE(e.stability, 0.0))) < ?6 "
              "ON CONFLICT(embedding_id) DO UPDATE SET "
              "  score=excluded.score, "
              "  created_at=excluded.created_at, "
              "  reason=excluded.reason;",
              { T, F, S, T, static_cast<long long> (now_ts), floor_cutoff });
}

void
ScoreConsolidation::CollectSchema (cortext::store::SchemaRegistry &registry) const
{
  // No-op: consolidation_candidates table now in core schema.cpp migration 0.
  (void)registry;
}

} // namespace cortext::operations
