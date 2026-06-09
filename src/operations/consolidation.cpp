#include "cortext/internal/cancellation.hpp"
#include "cortext/operations/consolidation.hpp"
#include "cortext/store/store.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/operations/extraction.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/consolidation_mode.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <any>
#include <cmath>
#include <string>

namespace cortext::operations
{

void
EvaluateConsolidation::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const uint64_t now_ts = context.GetSignal ().timestamp;
  const auto mode = context.GetSignal ().consolidation_mode;

  if (!mode.has_value ())
    {
      return;
    }

  context.SetConsolidationShouldStart (true);
  p_ctx.last_consolidation_ts = now_ts;
  p_ctx.consolidation_count += 1;
  p_ctx.memories_since_consolidation = 0;
  telemetry::LogDebug ("cortext.evaluate_consolidation", {
    telemetry::Attribute::Bool ("consolidation_start", true),
    telemetry::Attribute::String ("mode", ConsolidationModeLabel (*mode))
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
  const auto mode = context.GetSignal ().consolidation_mode.value_or (
      ConsolidationMode::Both);
  if (mode == ConsolidationMode::Shallow)
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
             < static_cast<uint64_t> (std::max (interval, 0) * 1000))
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
  const double F_raw = cfg.focus;
  const double S_raw = cfg.sensitivity;
  const double T = cfg.stability;
  const double F_eff = core::FocusBias (F_raw);
  const double S_eff = core::SensitivityBias (S_raw);
  const uint64_t now_ts = context.GetSignal ().timestamp;
  (void)now_ts;
  const bool force_consolidation
      = context.GetSignal ().consolidation_mode.has_value ();
  const auto mode = context.GetSignal ().consolidation_mode.value_or (
      ConsolidationMode::Both);
  const bool deep_mode = (mode != ConsolidationMode::Shallow);

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
              "UPDATE memories SET connectivity = COALESCE(("
              "  SELECT CASE WHEN (SELECT m FROM max_cnt) > 0 "
              "              THEN CAST(ec.cnt AS REAL) / (SELECT m FROM max_cnt) "
              "              ELSE 0.0 END "
              "  FROM edge_counts ec WHERE ec.embedding_id = "
              "memories.embedding_id"
              "), 0.0);",
              {});

  // v2: Select candidates whose score is below floor.
  // score = T*strength - F*redundancy + S*connectivity + T*stability
  // Uses unified memories table which contains per-memory state.
  const double tag_weight = core::ConsolidationCandidateTagWeight (
      F_raw, S_raw, T);
  const std::string blob_filter
      = deep_mode ? " AND m.blob_id IS NOT NULL " : " ";
  const std::string candidate_scope
      = "WHERE m.kind = 'LONG_TERM' "
        "  AND m.cluster_id IS NULL ";
  const std::string query = std::string (
      "SELECT m.embedding_id, "
      "       ((?1 * COALESCE(m.strength, 1.0)) "
      "        - (?2 * COALESCE(m.redundancy, 0.0)) "
      "        + (?3 * COALESCE(m.connectivity, 0.0)) "
      "        + (?4 * COALESCE(m.stability, 0.0)) "
      "        + (?5 * CASE WHEN m.tag_expires_at > ?6 "
      "                THEN COALESCE(m.tag_strength, 0.0) ELSE 0.0 END)) "
      "        AS computed_score, "
      "       e.embedding "
      "FROM memories m "
      "JOIN embeddings e ON m.embedding_id = e.embedding_id "
      )
      + candidate_scope
      + blob_filter
      + "  AND ((?1 * COALESCE(m.strength, 1.0)) "
        "       - (?2 * COALESCE(m.redundancy, 0.0)) "
        "       + (?3 * COALESCE(m.connectivity, 0.0)) "
        "       + (?4 * COALESCE(m.stability, 0.0)) "
        "       + (?5 * CASE WHEN m.tag_expires_at > ?6 "
        "               THEN COALESCE(m.tag_strength, 0.0) ELSE 0.0 END)) < ?7 "
        "ORDER BY computed_score ASC;";
  auto rows = tx.Execute (
      query,
      { T, F_eff, S_eff, T, tag_weight, static_cast<long long> (now_ts),
        floor_cutoff });
  internal::ThrowIfStopRequested ();
  const int min_cluster_size = core::MinClusterSize (F_raw);
  if (force_consolidation
      && static_cast<int> (rows.size ()) < min_cluster_size)
    {
      const int fallback_limit
          = std::max ({ core::WRet (T), min_cluster_size, 1 });
      const std::string fallback_query = std::string (
          "SELECT m.embedding_id, "
          "       ((?1 * COALESCE(m.strength, 1.0)) "
          "        - (?2 * COALESCE(m.redundancy, 0.0)) "
          "        + (?3 * COALESCE(m.connectivity, 0.0)) "
          "        + (?4 * COALESCE(m.stability, 0.0)) "
          "        + (?5 * CASE WHEN m.tag_expires_at > ?6 "
          "                THEN COALESCE(m.tag_strength, 0.0) ELSE 0.0 END)) "
          "        AS computed_score, "
          "       e.embedding "
          "FROM memories m "
          "JOIN embeddings e ON m.embedding_id = e.embedding_id "
          )
          + candidate_scope
          + blob_filter
          + "ORDER BY computed_score ASC LIMIT ?7;";
      rows = tx.Execute (
          fallback_query,
          { T, F_eff, S_eff, T, tag_weight, static_cast<long long> (now_ts),
            fallback_limit });
      internal::ThrowIfStopRequested ();
    }

  std::vector<ConsolidationCandidate> candidates;
  if (!rows.empty())
    {
      candidates.reserve (rows.size ());
      int emb_dim = 256; // Default assumption

      for (const auto &row : rows)
        {
          internal::ThrowIfStopRequested ();
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
