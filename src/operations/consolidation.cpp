#include "cortext/operations/consolidation.hpp"
#include "cortext/store/store.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/operations/extraction.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <any>
#include <cmath>
#include <string>

namespace cortext::operations
{

namespace
{

// Checks if consolidation should trigger due to rate falling below target.
// Triggers when measured rate falls below 50% of the knob-derived target.
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
      // v2: Count memories (primary metadata table) instead of embeddings
      auto rows = store->Execute (
          "SELECT COUNT(*) AS c FROM memories", {});
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
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  Store *store = context.GetStore ();
  const uint64_t now_ts = context.GetSignal ().timestamp;
  // Derive rate_target from knobs per algorithms.md Section 7.1:
  // rate_consolidate = (1/max(interval,1)) × (0.3+0.7T) × (1−0.5S)
  const double rate_target
      = core::ConsolidationRate (cfg.stability, cfg.sensitivity);
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
  double idle_for_ms = 0.0;
  if (p_ctx.last_retrieval_ts > 0 && now_ts > p_ctx.last_retrieval_ts)
    {
      idle_for_ms = static_cast<double> (now_ts - p_ctx.last_retrieval_ts);
    }
  // Convert idle_required from seconds to milliseconds for consistent comparison
  const int idle_required_ms = core::IdleRequiredSeconds (cfg.stability) * 1000;
  const bool idle_ok
      = CheckIdleCondition (tokens_in_flight, retrieval_queue_depth,
                            idle_for_ms, idle_required_ms, trigger_interval);

  // NOTE: consolidation_events table removed (undocumented).
  // Event logging removed - consolidation decisions are tracked via
  // last_consolidation_ts in processor_state.

  // Start signal: set flag and update last_consolidation_ts.
  if (idle_ok)
    {
      context.SetConsolidationShouldStart (true);
      p_ctx.last_consolidation_ts = now_ts;
    }

  telemetry::LogDebug("cortext.evaluate_consolidation", {
    telemetry::Attribute::Bool("rate_trigger", trigger_rate),
    telemetry::Attribute::Bool("interval_trigger", trigger_interval),
    telemetry::Attribute::Bool("consolidation_start", idle_ok)
  });
}

void
EnqueueExtractionJobs::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
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

  // Respect both batch_size and max_per_cycle limits.
  // ExtractionBatchSize(T): how many to process per batch (8-32)
  // MaxExtractionsPerCycle(T): max extractions per consolidation cycle (20-5)
  const int batch_size = core::ExtractionBatchSize (cfg.stability);
  const int max_per_cycle = core::MaxExtractionsPerCycle (cfg.stability);
  const int count = std::min ({ static_cast<int> (requests.size ()),
                                batch_size, max_per_cycle });

  // Invoke extraction callback if registered.
  auto *callback = context.GetExtractionCallback ();
  if (callback)
    {
      std::vector<operations::ExtractionRequest> batch (requests.begin (),
                                                        requests.begin () + count);
      (*callback) (batch);
    }

  p_ctx.last_extraction_ts = now_ts;

  telemetry::LogDebug("cortext.enqueue_extraction_jobs", {
    telemetry::Attribute::Int64("jobs_queued", count)
  });
}

void
ScoreConsolidation::Execute (OperationContext &context, Transaction &tx) const
{
  const auto &cfg = context.GetConfig ();
  const double F = cfg.focus;
  const double S = cfg.sensitivity;
  const double T = cfg.stability;
  const uint64_t now_ts = context.GetSignal ().timestamp;
  (void)now_ts;

  // Floor derived from knobs (no magic numbers).
  const double floor_cutoff = core::PeripheryCutoff (T);

  // V2: Update connectivity metric from ASSOCIATIONS edge count (Section 9.2).
  // Connectivity = normalized count of edges where this memory participates.
  // Normalized by max observed edge count to keep values in [0, 1].
  tx.Execute ("WITH edge_counts AS ("
              "  SELECT m.memory_id, m.embedding_id, "
              "         (SELECT COUNT(*) FROM associations a "
              "          WHERE a.source_memory_id = m.memory_id "
              "             OR a.target_memory_id = m.memory_id) AS cnt "
              "  FROM memories m"
              "), max_cnt AS ("
              "  SELECT COALESCE(MAX(cnt), 0) AS m FROM edge_counts"
              ") "
              "UPDATE memories SET connectivity = ("
              "  SELECT CASE WHEN (SELECT m FROM max_cnt) > 0 "
              "              THEN CAST(ec.cnt AS REAL) / (SELECT m FROM max_cnt) "
              "              ELSE 0.0 END "
              "  FROM edge_counts ec WHERE ec.embedding_id = "
              "memories.embedding_id"
              ");",
              {});

  // v2: Select candidates whose score is below floor.
  // score = T*strength - F*redundancy + S*connectivity + T*stability
  // Uses unified memories table which contains per-memory state.
  auto rows = tx.Execute (
      "SELECT m.embedding_id, "
      "       ((?1 * COALESCE(m.strength, 1.0)) "
      "        - (?2 * COALESCE(m.redundancy, 0.0)) "
      "        + (?3 * COALESCE(m.connectivity, 0.0)) "
      "        + (?4 * COALESCE(m.stability, 0.0))) AS computed_score, "
      "       e.embedding "
      "FROM memories m "
      "JOIN embeddings e ON m.embedding_id = e.embedding_id "
      "WHERE ((?1 * COALESCE(m.strength, 1.0)) "
      "       - (?2 * COALESCE(m.redundancy, 0.0)) "
      "       + (?3 * COALESCE(m.connectivity, 0.0)) "
      "       + (?4 * COALESCE(m.stability, 0.0))) < ?5 "
      "ORDER BY computed_score ASC;",
      { T, F, S, T, floor_cutoff });

  std::vector<ConsolidationCandidate> candidates;
  if (!rows.empty())
    {
      candidates.reserve (rows.size ());
      int emb_dim = 256; // Default assumption

      for (const auto &row : rows)
        {
          auto it_id = row.find ("embedding_id");
          auto it_score = row.find ("computed_score");
          auto it_emb = row.find ("embedding");

          if (it_id == row.end () || it_score == row.end () || it_emb == row.end ())
            {
              continue;
            }

          if (it_id->second.type () != typeid (long long))
            {
              continue;
            }

          ConsolidationCandidate c;
          c.embedding_id = std::any_cast<long long> (it_id->second);

          if (it_score->second.type () == typeid (double))
            {
              c.score = std::any_cast<double> (it_score->second);
            }
          else if (it_score->second.type () == typeid (long long))
            {
              c.score = static_cast<double> (
                  std::any_cast<long long> (it_score->second));
            }
          else
            {
              continue;
            }

          if (!core::DecodeFloatBlob (it_emb->second, emb_dim, c.embedding))
            {
              continue;
            }
          emb_dim = static_cast<int> (c.embedding.size ());

          candidates.push_back (std::move (c));
        }
    }

  long long candidate_count = static_cast<long long>(candidates.size());
  context.SetConsolidationCandidates (std::move (candidates));

  telemetry::LogDebug("cortext.score_consolidation", {
    telemetry::Attribute::Int64("candidate_count", candidate_count),
    telemetry::Attribute::Int64("selected_count", candidate_count)
  });
}

} // namespace cortext::operations
