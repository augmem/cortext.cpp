#include "cortext/internal/cancellation.hpp"
#include "cortext/operations/consolidation.hpp"
#include "cortext/store/store.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <any>
#include <chrono>
#include <cmath>
#include <string>

namespace cortext::operations
{

void
EvaluateConsolidation::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;

  if (context.GetSignal ().retention == Retention::Ephemeral
      || !context.GetSignal ().force_consolidation)
    {
      return;
    }

  context.SetConsolidationShouldStart (true);
  telemetry::LogDebug ("cortext.evaluate_consolidation", {
    telemetry::Attribute::Bool ("consolidation_start", true)
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
  // Floor derived from knobs (no magic numbers).
  const double floor_cutoff = core::PeripheryCutoff (T);

  // v2: Select candidates whose score is below floor.
  // score = T*strength - F*redundancy + S*connectivity + T*stability
  // Uses unified memories table which contains per-memory state.
  const double tag_weight = core::ConsolidationCandidateTagWeight (
      F_raw, S_raw, T);
  const std::string query
      = "WITH edge_counts AS ("
        "  SELECT m.memory_id, "
        "         (SELECT COUNT(*) FROM associations a "
        "          WHERE a.source_memory_id = m.memory_id "
        "             OR a.target_memory_id = m.memory_id) AS cnt "
        "  FROM memories m"
        "), max_cnt AS ("
        "  SELECT COALESCE(MAX(cnt), 0) AS maximum FROM edge_counts"
        "), scored AS ("
        "  SELECT m.memory_id, "
        "         COALESCE(cme.embedding_id, m.embedding_id) AS embedding_id, "
        "         ((?1 * COALESCE(m.strength, 1.0)) "
        "          - (?2 * COALESCE(m.redundancy, 0.0)) "
        "          + (?3 * CASE WHEN max_cnt.maximum > 0 "
        "                 THEN CAST(ec.cnt AS REAL) / max_cnt.maximum "
        "                 ELSE 0.0 END) "
        "          + (?4 * COALESCE(m.stability, 0.0)) "
        "          + (?5 * CASE WHEN m.tag_expires_at > ?6 "
        "                  THEN COALESCE(m.tag_strength, 0.0) "
        "                  ELSE 0.0 END)) AS computed_score, "
        "         COALESCE(cme.embedding, e.embedding) AS embedding, "
        "         m.created_at "
        "  FROM memories m "
        "  JOIN embeddings e ON m.embedding_id = e.embedding_id "
        "  LEFT JOIN current_memory_embeddings cme "
        "    ON cme.memory_id = m.memory_id "
        "  LEFT JOIN edge_counts ec ON ec.memory_id = m.memory_id "
        "  CROSS JOIN max_cnt "
        "  WHERE m.kind = 'LONG_TERM' "
        "    AND m.cluster_id IS NULL"
        ") "
        "SELECT memory_id, embedding_id, computed_score, embedding "
        "FROM scored WHERE computed_score < ?7 "
        "ORDER BY computed_score ASC, created_at ASC, memory_id ASC;";
  const auto candidate_query_start = std::chrono::steady_clock::now ();
  auto rows = tx.Execute (
      query,
      { T, F_eff, S_eff, T, tag_weight, static_cast<long long> (now_ts),
        floor_cutoff });
  context.AddOperationTiming (
      "ScoreConsolidation.candidate_query_sql",
      std::chrono::duration<double, std::milli> (
          std::chrono::steady_clock::now () - candidate_query_start)
          .count ());
  internal::ThrowIfStopRequested ();
  std::vector<ConsolidationCandidate> candidates;
  if (!rows.empty())
    {
      const auto decode_start = std::chrono::steady_clock::now ();
      candidates.reserve (rows.size ());
      int emb_dim = 256; // Default assumption

      for (const auto &row : rows)
        {
          internal::ThrowIfStopRequested ();
          auto it_memory_id = row.find ("memory_id");
          auto it_id = row.find ("embedding_id");
          auto it_score = row.find ("computed_score");
          auto it_emb = row.find ("embedding");

          if (it_memory_id == row.end () || it_id == row.end ()
              || it_score == row.end () || it_emb == row.end ())
            {
              continue;
            }

          if (it_memory_id->second.type () != typeid (long long)
              || it_id->second.type () != typeid (long long))
            {
              continue;
            }

          ConsolidationCandidate c;
          c.memory_id = std::any_cast<long long> (it_memory_id->second);
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
      context.AddOperationTiming (
          "ScoreConsolidation.decode_candidates",
          std::chrono::duration<double, std::milli> (
              std::chrono::steady_clock::now () - decode_start)
              .count ());
    }

  long long candidate_count = static_cast<long long>(candidates.size());
  context.SetConsolidationCandidates (std::move (candidates));

  telemetry::LogDebug("cortext.score_consolidation", {
    telemetry::Attribute::Int64("candidate_count", candidate_count),
    telemetry::Attribute::Int64("selected_count", candidate_count)
  });
}

} // namespace cortext::operations
