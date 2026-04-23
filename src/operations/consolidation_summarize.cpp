#include "cortext/internal/cancellation.hpp"
#include "cortext/operations/consolidation_summarize.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/operations/extraction.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/object_store.hpp"
#include "cortext/store/store.hpp"
#include "cortext/summarizer/summarizer.hpp"
#include "cortext/consolidation_mode.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "eviction_ablation.hpp"
#include "eviction_policy.hpp"
#include <algorithm>
#include <any>
#include <cctype>
#include <ctime>
#include <cmath>
#include <optional>
#include <regex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace cortext::operations
{

namespace
{

/// @brief Generate a unique summary ID.
std::string
GenerateSummaryId (uint64_t ts, int counter)
{
  std::ostringstream ss;
  ss << "summary_" << ts << "_" << counter;
  return ss.str ();
}

std::string
GenerateAssociativeCueId (uint64_t ts, int counter)
{
  std::ostringstream ss;
  ss << "associative_cue_" << ts << "_" << counter;
  return ss.str ();
}

/// @brief Execute a write query on the transaction.
void
AddWrite (Transaction &tx, const std::string &q,
          const std::vector<std::any> &p = {})
{
  tx.Execute (q, p);
}

/// @brief Extract blob bytes from any.
std::vector<unsigned char>
GetBlobBytes (const std::any &val)
{
  if (val.type () == typeid (std::vector<unsigned char>))
    {
      return std::any_cast<std::vector<unsigned char>> (val);
    }
  if (val.type () == typeid (std::vector<char>))
    {
      const auto &blob = std::any_cast<std::vector<char>> (val);
      return std::vector<unsigned char> (blob.begin (), blob.end ());
    }
  return {};
}

/// @brief Convert string payload to blob bytes.
std::vector<unsigned char>
StringToBlob (const std::string &text)
{
  return std::vector<unsigned char> (text.begin (), text.end ());
}

long long
AnyToInt64 (const std::any &val)
{
  if (val.type () == typeid (long long))
    {
      return std::any_cast<long long> (val);
    }
  if (val.type () == typeid (int))
    {
      return static_cast<long long> (std::any_cast<int> (val));
    }
  return 0;
}

std::string
TrimAsciiWhitespace (const std::string &text)
{
  std::size_t start = 0;
  while (start < text.size ()
         && std::isspace (static_cast<unsigned char> (text[start])))
    {
      ++start;
    }

  std::size_t end = text.size ();
  while (end > start
         && std::isspace (static_cast<unsigned char> (text[end - 1])))
    {
      --end;
    }

  return text.substr (start, end - start);
}

std::string
SanitizeSummaryText (std::string summary)
{
  if (summary.empty ())
    {
      return summary;
    }

  summary = std::regex_replace (
      summary, std::regex (R"(Excerpt\s+\d+\s+(mentions|indicates)\s+)",
                           std::regex_constants::icase),
      "");
  summary = std::regex_replace (
      summary,
      std::regex (R"(,\s*which is a user\b)", std::regex_constants::icase), "");
  summary = std::regex_replace (
      summary,
      std::regex (R"(\bthe user is focusing on\b)",
                  std::regex_constants::icase),
      "focuses on");
  summary = std::regex_replace (
      summary, std::regex (R"(\bthe user is\b)", std::regex_constants::icase),
      "");
  summary = std::regex_replace (
      summary, std::regex (R"(\bthe user\b)", std::regex_constants::icase),
      "");
  summary = std::regex_replace (
      summary,
      std::regex (R"(\bthe assistant\b)", std::regex_constants::icase), "");
  summary = std::regex_replace (summary, std::regex (R"([ \t]+)"), " ");
  summary = std::regex_replace (summary, std::regex (R"(\s+\.)"), ".");
  summary = std::regex_replace (summary, std::regex (R"(\n{3,})"), "\n\n");
  return TrimAsciiWhitespace (summary);
}

bool
StartsWithAscii (const std::string &text, const std::string &prefix)
{
  return text.size () >= prefix.size ()
         && std::equal (prefix.begin (), prefix.end (), text.begin ());
}

std::string
FormatSourceTextForSummary (const std::string &source_id,
                            const std::string &raw_text)
{
  std::string text = TrimAsciiWhitespace (raw_text);
  if (text.empty ())
    {
      return {};
    }

  if (source_id.rfind ("chat/user", 0) == 0)
    {
      if (StartsWithAscii (text, "User:"))
        {
          return text;
        }
      return "User: " + text;
    }

  if (source_id.rfind ("chat/assistant", 0) == 0)
    {
      if (StartsWithAscii (text, "Assistant:"))
        {
          return text;
        }
      return "Assistant: " + text;
    }

  return text;
}

long long
FindExistingDeepSummaryForSources (Transaction &tx,
                                   const std::vector<long long> &source_memory_ids)
{
  if (source_memory_ids.empty ())
    {
      return 0;
    }

  std::string sql =
      "SELECT s.memory_id "
      "FROM memories s "
      "JOIN associations a ON a.source_memory_id = s.memory_id "
      "WHERE s.kind = 'ASSOCIATION' "
      "  AND s.source_id LIKE 'summary_%' "
      "  AND a.edge_type = 'derived_from' "
      "GROUP BY s.memory_id "
      "HAVING COUNT(*) = ? "
      "   AND SUM(CASE WHEN a.target_memory_id IN (";
  std::vector<std::any> params;
  params.reserve (source_memory_ids.size () + 2);
  params.push_back (static_cast<long long> (source_memory_ids.size ()));
  for (size_t i = 0; i < source_memory_ids.size (); ++i)
    {
      if (i > 0)
        {
          sql += ",";
        }
      sql += "?";
      params.push_back (source_memory_ids[i]);
    }
  sql += ") THEN 1 ELSE 0 END) = ? "
         "ORDER BY s.memory_id DESC LIMIT 1";
  params.push_back (static_cast<long long> (source_memory_ids.size ()));

  auto rows = tx.Execute (sql, params);
  if (rows.empty ())
    {
      return 0;
    }

  auto it = rows[0].find ("memory_id");
  if (it == rows[0].end ())
    {
      return 0;
    }
  return AnyToInt64 (it->second);
}

long long
FindExistingAssociativeCueForSources (
    Transaction &tx, const std::vector<long long> &source_memory_ids)
{
  if (source_memory_ids.empty ())
    {
      return 0;
    }

  std::string sql =
      "SELECT s.memory_id "
      "FROM memories s "
      "JOIN associations a ON a.source_memory_id = s.memory_id "
      "WHERE s.kind = 'ASSOCIATION' "
      "  AND s.source_id LIKE 'associative_cue_%' "
      "  AND a.edge_type = 'derived_from' "
      "GROUP BY s.memory_id "
      "HAVING COUNT(*) = ? "
      "   AND SUM(CASE WHEN a.target_memory_id IN (";
  std::vector<std::any> params;
  params.reserve (source_memory_ids.size () + 2);
  params.push_back (static_cast<long long> (source_memory_ids.size ()));
  for (size_t i = 0; i < source_memory_ids.size (); ++i)
    {
      if (i > 0)
        {
          sql += ",";
        }
      sql += "?";
      params.push_back (source_memory_ids[i]);
    }
  sql += ") THEN 1 ELSE 0 END) = ? "
         "ORDER BY s.memory_id DESC LIMIT 1";
  params.push_back (static_cast<long long> (source_memory_ids.size ()));

  auto rows = tx.Execute (sql, params);
  if (rows.empty ())
    {
      return 0;
    }

  auto it = rows[0].find ("memory_id");
  if (it == rows[0].end ())
    {
      return 0;
    }
  return AnyToInt64 (it->second);
}

struct SummaryRecord
{
  long long memory_id = 0;
  long long embedding_id = 0;
  std::string source_id;
  Eigen::VectorXf embedding;
};

std::optional<SummaryRecord>
LoadSummaryRecord (Transaction &tx, long long memory_id, int expected_dim)
{
  auto rows = tx.Execute (
      "SELECT m.memory_id, m.embedding_id, m.source_id, e.embedding "
      "FROM memories m "
      "JOIN embeddings e ON e.embedding_id = m.embedding_id "
      "WHERE m.memory_id = ?",
      { memory_id });
  if (rows.empty ())
    {
      return std::nullopt;
    }

  const auto &row = rows[0];
  auto it_memory = row.find ("memory_id");
  auto it_embedding_id = row.find ("embedding_id");
  auto it_source_id = row.find ("source_id");
  auto it_embedding = row.find ("embedding");
  if (it_memory == row.end () || it_embedding_id == row.end ()
      || it_embedding == row.end ())
    {
      return std::nullopt;
    }

  SummaryRecord record;
  record.memory_id = AnyToInt64 (it_memory->second);
  record.embedding_id = AnyToInt64 (it_embedding_id->second);
  if (it_source_id != row.end ()
      && it_source_id->second.type () == typeid (std::string))
    {
      record.source_id = std::any_cast<std::string> (it_source_id->second);
    }
  if (record.memory_id <= 0 || record.embedding_id <= 0)
    {
      return std::nullopt;
    }
  if (!core::DecodeFloatBlob (it_embedding->second, expected_dim,
                              record.embedding))
    {
      return std::nullopt;
    }
  return record;
}

bool
HasLabelEdges (Transaction &tx, long long association_memory_id)
{
  if (association_memory_id <= 0)
    {
      return false;
    }
  auto rows = tx.Execute (
      "SELECT 1 AS present FROM associations "
      "WHERE source_memory_id = ? AND edge_type = 'has_label' LIMIT 1",
      { association_memory_id });
  return !rows.empty ();
}

long long
InsertAssociativeCue (OperationContext &context, Transaction &tx,
                      const ClusterInfo &cluster, const std::string &cue_id,
                      uint64_t now_ts)
{
  AddWrite (tx,
            "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
            { cluster.centroid, static_cast<long long> (now_ts) });

  auto id_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
  long long centroid_embedding_id = 0;
  if (!id_rows.empty () && id_rows[0].count ("id"))
    {
      centroid_embedding_id = AnyToInt64 (id_rows[0].at ("id"));
    }
  if (centroid_embedding_id <= 0)
    {
      return 0;
    }

  const std::string label
      = "associative cue " + std::to_string (cluster.cluster_id);
  AddWrite (tx,
            "INSERT INTO memories "
            "(embedding_id, source_id, kind, label, start_ts, n_signals, created_at) "
            "VALUES (?, ?, 'ASSOCIATION', ?, ?, ?, ?)",
            { centroid_embedding_id, cue_id, label,
              static_cast<long long> (now_ts),
              static_cast<long long> (cluster.embedding_ids.size ()),
              static_cast<long long> (now_ts) });

  auto mem_id_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
  long long centroid_memory_id = 0;
  if (!mem_id_rows.empty () && mem_id_rows[0].count ("id"))
    {
      centroid_memory_id = AnyToInt64 (mem_id_rows[0].at ("id"));
    }
  if (centroid_memory_id <= 0)
    {
      return 0;
    }

  Eigen::VectorXf centroid_vec (
      static_cast<Eigen::Index> (cluster.centroid.size ()));
  for (size_t i = 0; i < cluster.centroid.size (); ++i)
    {
      centroid_vec (static_cast<Eigen::Index> (i)) = cluster.centroid[i];
    }
  context.GetProcessorContext ().UpsertSummaryCache (
      centroid_memory_id, centroid_embedding_id, centroid_vec, true, false);
  return centroid_memory_id;
}

void
AttachClusterSources (Transaction &tx, long long association_memory_id,
                      int cluster_id,
                      const std::vector<std::pair<long long, long long>> &source_links)
{
  for (const auto &[emb_id, src_memory_id] : source_links)
    {
      internal::ThrowIfStopRequested ();
      AddWrite (tx,
                "UPDATE memories SET cluster_id = ? "
                "WHERE embedding_id = ?",
                { cluster_id, emb_id });

      if (src_memory_id > 0 && association_memory_id > 0)
        {
          AddWrite (tx,
                    "INSERT OR IGNORE INTO associations "
                    "(source_memory_id, target_memory_id, edge_type, weight) "
                    "VALUES (?, ?, 'derived_from', 1.0)",
                    { association_memory_id, src_memory_id });
        }
    }
}

void
QueueExtractionIfNeeded (std::vector<ExtractionRequest> &requests,
                         const ConsolidationSummarizeParams &params,
                         const ClusterInfo &cluster,
                         const std::string &association_source_id,
                         const std::string &summary_text,
                         const std::vector<std::string> &source_texts,
                         uint64_t now_ts)
{
  if (static_cast<int> (cluster.embedding_ids.size ())
      < params.min_cluster_size_for_extraction)
    {
      return;
    }

  ExtractionRequest req;
  req.summary_id = association_source_id;
  req.summary_text = summary_text;
  req.cluster_size = static_cast<int> (cluster.embedding_ids.size ());
  req.created_at = now_ts;
  req.source_texts = source_texts;
  requests.push_back (std::move (req));
}

} // namespace

ConsolidationSummarizeParams
ConsolidationSummarizeParams::FromKnobs (double F, double S, double T)
{
  (void)S;
  ConsolidationSummarizeParams p;
  p.min_cluster_size_for_extraction = core::MinClusterSizeForExtraction (F);
  const double F_eff = core::FocusBias (F);
  const double T_eff = core::Clamp (T, 0.0, 1.0);
  p.max_source_texts = std::max (
      2, static_cast<int> (std::round (core::Lerp (3.0, 8.0, F_eff))));
  p.max_total_chars
      = static_cast<int> (std::round (core::Lerp (1200.0, 3600.0, T_eff)));
  p.max_text_chars
      = static_cast<int> (std::round (core::Lerp (300.0, 900.0, F_eff)));
  p.max_summary_words = 0;
  return p;
}

void
ConsolidationSummarize::Execute (OperationContext &context, Transaction &tx) const
{
  if (!context.GetConsolidationShouldStart ())
    {
      return;
    }
  const auto mode = ParseConsolidationMode (context.GetSignal ().source_id);
  if (mode == ConsolidationMode::Shallow)
    {
      return;
    }

  const auto &clusters = context.GetConsolidationClusters ();
  if (clusters.empty ())
    {
      return;
    }

  if (!context.GetStore ())
    {
      return;
    }

  const auto &cfg = context.GetConfig ();
  auto params
      = ConsolidationSummarizeParams::FromKnobs (cfg.focus, cfg.sensitivity,
                                                  cfg.stability);
  auto &p_ctx = context.GetProcessorContext ();
  const uint64_t now_ts = context.GetSignal ().timestamp;

  const auto eviction_override = eviction::GetEvictionAblationOverride ();
  const auto eviction_frontier = eviction_policy::ResolveEvictionFrontier (
      tx, cfg.stability, static_cast<long long> (p_ctx.last_consolidation_ts),
      eviction_override);
  const bool compression_allowed = eviction_frontier.storage_gate_active
                                   && eviction_frontier.storage_allows_eviction;
  if (!compression_allowed)
    {
      telemetry::LogInfo (
          "cortext.consolidation_summarize.skipped_storage_pressure",
          { telemetry::Attribute::Bool (
                "storage_pressure_active",
                eviction_frontier.storage_gate_active),
            telemetry::Attribute::Int64 (
                "storage_used_bytes",
                eviction_frontier.storage_used_bytes),
            telemetry::Attribute::Int64 (
                "storage_threshold_bytes",
                eviction_frontier.storage_threshold_bytes) });
    }

  std::vector<ExtractionRequest> extraction_requests;
  int summary_sequence = 0;
  int cue_sequence = 0;
  int summary_count = 0;
  int summaries_with_summarizer = 0;
  int associative_cues = 0;
  int clusters_skipped_eviction_frontier = 0;
  int clusters_skipped_summarizer_unavailable = 0;

  // Get deep summarizer backend selected during Cortext construction.
  Summarizer *summarizer = context.GetSummarizer ();

  for (const auto &cluster : clusters)
    {
      internal::ThrowIfStopRequested ();

      // Collect all source texts and find most representative memory
      struct SourceItem
      {
        std::string text;
        double sim;
        long long start_ts;
        bool chat_source;
      };
      std::vector<SourceItem> source_items;
      source_items.reserve (cluster.embedding_ids.size ());
      std::vector<std::string> source_texts;
      std::vector<std::pair<long long, long long>> source_links;
      source_links.reserve (cluster.embedding_ids.size ());
      std::vector<long long> source_memory_ids;
      source_memory_ids.reserve (cluster.embedding_ids.size ());
      std::vector<long long> eviction_source_memory_ids;
      eviction_source_memory_ids.reserve (cluster.embedding_ids.size ());
      std::unordered_set<long long> seen_source_memory_ids;
      std::unordered_set<long long> seen_eviction_source_memory_ids;

      // Convert cluster centroid to Eigen for comparison.
      Eigen::VectorXf centroid_eigen (
          static_cast<Eigen::Index> (cluster.centroid.size ()));
      for (size_t i = 0; i < cluster.centroid.size (); ++i)
        {
          centroid_eigen (static_cast<Eigen::Index> (i)) = cluster.centroid[i];
        }

      for (long long emb_id : cluster.embedding_ids)
        {
          internal::ThrowIfStopRequested ();
          // Query embedding, memory row, blob_id, source_id, and start_ts.
          auto rows = tx.Execute (
              "SELECT e.embedding, m.memory_id, m.blob_id, m.source_id, "
              "m.start_ts, m.kind "
              "FROM embeddings e "
              "LEFT JOIN memories m ON e.embedding_id = m.embedding_id "
              "WHERE e.embedding_id = ?",
              { emb_id });

          if (rows.empty ())
            {
              continue;
            }

          const auto &row = rows[0];
          double sim = -1.0;
          auto it_emb = row.find ("embedding");
          if (it_emb != row.end ())
            {
              Eigen::VectorXf emb;
              if (core::DecodeFloatBlob (
                      it_emb->second,
                      static_cast<int> (cluster.centroid.size ()), emb))
                {
                  sim = core::CosineSimilarity (centroid_eigen, emb);
                }
            }

          // Try to fetch text from objstore.
          std::string text;
          auto it_blob = row.find ("blob_id");
          if (it_blob != row.end ())
            {
              auto blob_id = GetBlobBytes (it_blob->second);
              if (!blob_id.empty ())
                {
                  auto data = GetObject (context.GetObjectTransaction (), tx,
                                         blob_id);
                  if (data)
                    {
                      text.assign (data->begin (), data->end ());
                    }
                }
            }

          std::string source_id;
          auto it_source = row.find ("source_id");
          if (it_source != row.end ()
              && it_source->second.type () == typeid (std::string))
            {
              source_id = std::any_cast<std::string> (it_source->second);
            }

          std::string memory_kind;
          auto it_kind = row.find ("kind");
          if (it_kind != row.end ()
              && it_kind->second.type () == typeid (std::string))
            {
              memory_kind = std::any_cast<std::string> (it_kind->second);
            }

          long long memory_id = 0;
          auto it_memory_id = row.find ("memory_id");
          if (it_memory_id != row.end ())
            {
              memory_id = AnyToInt64 (it_memory_id->second);
            }
          if (memory_id > 0)
            {
              source_links.emplace_back (emb_id, memory_id);
              if (seen_source_memory_ids.insert (memory_id).second)
                {
                  source_memory_ids.push_back (memory_id);
                }
              if (memory_kind == "LONG_TERM"
                  && seen_eviction_source_memory_ids.insert (memory_id).second)
                {
                  eviction_source_memory_ids.push_back (memory_id);
                }
            }

          long long start_ts = 0;
          auto it_start_ts = row.find ("start_ts");
          if (it_start_ts != row.end ())
            {
              if (it_start_ts->second.type () == typeid (long long))
                {
                  start_ts = std::any_cast<long long> (it_start_ts->second);
                }
              else if (it_start_ts->second.type () == typeid (int))
                {
                  start_ts = std::any_cast<int> (it_start_ts->second);
                }
            }

          text = FormatSourceTextForSummary (source_id, text);

          // Collect text for summarizer
          if (!text.empty ())
            {
              const bool chat_source
                  = (source_id.rfind ("chat/user", 0) == 0
                     || source_id.rfind ("chat/assistant", 0) == 0);
              source_items.push_back (SourceItem{ std::move (text), sim,
                                                  start_ts, chat_source });
            }
        }

      if (!source_items.empty ())
        {
          internal::ThrowIfStopRequested ();
          const int chat_source_count = static_cast<int> (
              std::count_if (source_items.begin (), source_items.end (),
                             [] (const SourceItem &item) {
                               return item.chat_source;
                             }));
          const int effective_max_source_texts
              = (chat_source_count >= 4)
                    ? std::max (params.max_source_texts, 6)
                    : params.max_source_texts;
          std::sort (source_items.begin (), source_items.end (),
                     [] (const SourceItem &a, const SourceItem &b) {
                       return a.sim > b.sim;
                     });
          std::vector<SourceItem> selected_items;
          selected_items.reserve (
              static_cast<size_t> (effective_max_source_texts));
          int total_chars = 0;
          for (const auto &item : source_items)
            {
              internal::ThrowIfStopRequested ();
              if (static_cast<int> (selected_items.size ())
                  >= effective_max_source_texts)
                {
                  break;
                }
              if (total_chars >= params.max_total_chars)
                {
                  break;
                }
              std::string trimmed = item.text;
              if (static_cast<int> (trimmed.size ()) > params.max_text_chars)
                {
                  trimmed.resize (
                      static_cast<size_t> (params.max_text_chars));
                }
              if (trimmed.empty ())
                {
                  continue;
                }
              int remaining = params.max_total_chars - total_chars;
              if (remaining <= 0)
                {
                  break;
                }
              if (static_cast<int> (trimmed.size ()) > remaining)
                {
                  trimmed.resize (static_cast<size_t> (remaining));
                }
              if (trimmed.empty ())
                {
                  break;
                }
              total_chars += static_cast<int> (trimmed.size ());
              SourceItem selected = item;
              selected.text = std::move (trimmed);
              selected_items.push_back (std::move (selected));
            }

          std::sort (selected_items.begin (), selected_items.end (),
                     [] (const SourceItem &a, const SourceItem &b) {
                       if (a.start_ts == b.start_ts)
                         {
                           return a.sim > b.sim;
                         }
                       return a.start_ts < b.start_ts;
                     });
          for (auto &item : selected_items)
            {
              source_texts.push_back (std::move (item.text));
            }
        }

      if (source_texts.empty ())
        {
          throw std::runtime_error (
              "Consolidation label routing requires source text for "
              + std::to_string (cluster.cluster_id));
        }

      long long cue_memory_id = FindExistingAssociativeCueForSources (
          tx, source_memory_ids);
      std::string cue_id;
      if (cue_memory_id > 0)
        {
          auto existing_cue = LoadSummaryRecord (
              tx, cue_memory_id,
              static_cast<int> (cluster.centroid.size ()));
          if (existing_cue.has_value ())
            {
              cue_id = existing_cue->source_id;
              context.GetProcessorContext ().UpsertSummaryCache (
                  existing_cue->memory_id,
                  existing_cue->embedding_id,
                  existing_cue->embedding,
                  true,
                  false);
            }
        }

      if (cue_memory_id <= 0 || cue_id.empty ())
        {
          cue_id = GenerateAssociativeCueId (now_ts, cue_sequence++);
          cue_memory_id = InsertAssociativeCue (context, tx, cluster,
                                                cue_id, now_ts);
          if (cue_memory_id > 0)
            {
              ++associative_cues;
            }
        }

      AttachClusterSources (tx, cue_memory_id, cluster.cluster_id,
                            source_links);
      if (cue_memory_id > 0 && !HasLabelEdges (tx, cue_memory_id))
        {
          QueueExtractionIfNeeded (extraction_requests, params, cluster,
                                   cue_id, "", source_texts, now_ts);
        }

      bool cluster_compression_allowed = compression_allowed;
      if (cluster_compression_allowed)
        {
          const auto evictable_source_ids
              = eviction_policy::LoadEvictableMemoryIds (
                  tx, eviction_frontier, eviction_source_memory_ids);
          cluster_compression_allowed
              = !eviction_source_memory_ids.empty ()
                && evictable_source_ids.size ()
                       >= eviction_source_memory_ids.size ();
        }
      if (cluster_compression_allowed
          && (!summarizer || !summarizer->IsAvailable ()))
        {
          ++clusters_skipped_summarizer_unavailable;
          cluster_compression_allowed = false;
        }

      if (!cluster_compression_allowed)
        {
          ++clusters_skipped_eviction_frontier;
          continue;
        }

      long long centroid_memory_id = FindExistingDeepSummaryForSources (
          tx, source_memory_ids);
      if (centroid_memory_id > 0)
        {
          if (auto existing_summary = LoadSummaryRecord (
                  tx, centroid_memory_id,
                  static_cast<int> (cluster.centroid.size ()));
              existing_summary.has_value ())
            {
              context.GetProcessorContext ().UpsertSummaryCache (
                  existing_summary->memory_id,
                  existing_summary->embedding_id,
                  existing_summary->embedding,
                  true,
                  false);
            }

          for (const auto &[emb_id, src_memory_id] : source_links)
            {
              internal::ThrowIfStopRequested ();
              AddWrite (tx,
                        "UPDATE memories SET cluster_id = ? "
                        "WHERE embedding_id = ?",
                        { cluster.cluster_id, emb_id });
              AddWrite (tx,
                        "INSERT OR IGNORE INTO associations "
                        "(source_memory_id, target_memory_id, edge_type, weight) "
                        "VALUES (?, ?, 'derived_from', 1.0)",
                        { centroid_memory_id, src_memory_id });
            }
          continue;
        }

      std::string summary_id = GenerateSummaryId (now_ts, summary_sequence++);

      // Use the configured abstractive summarizer backend.
      std::string summary_text;
      try
        {
          internal::ThrowIfStopRequested ();
          summary_text = summarizer->SummarizeTextsLimited (
              source_texts, params.max_summary_words);
          summary_text = SanitizeSummaryText (summary_text);
        }
      catch (const internal::CancellationError &)
        {
          throw;
        }
      catch (const std::exception &e)
        {
          throw std::runtime_error (
              "Consolidation summarization failed for " + summary_id + ": "
              + e.what ());
        }
      if (summary_text.empty ())
        {
          throw std::runtime_error (
              "Consolidation summarization returned empty text for "
              + summary_id);
        }
      summary_count++;
      summaries_with_summarizer++;

      // 2. Store summary text as blob for downstream consolidation.
      std::vector<unsigned char> summary_blob_id;
      summary_blob_id = PutObject (context.GetObjectTransaction (), tx,
                                   StringToBlob (summary_text));
      if (summary_blob_id.empty ())
        {
          throw std::runtime_error (
              "Consolidation summarization failed to persist blob for "
              + summary_id);
        }

      // 3. Convert centroid to blob for storage.
      std::vector<float> centroid_blob = cluster.centroid;

      // 4. Create embeddings entry for centroid (v2: minimal table).
      AddWrite (tx,
                "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
                { centroid_blob, static_cast<long long> (now_ts) });

      // Get the auto-assigned embedding_id
      auto id_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
      long long centroid_embedding_id = 0;
      if (!id_rows.empty () && id_rows[0].count ("id"))
        {
          auto val = id_rows[0].at ("id");
          if (val.type () == typeid (long long))
            {
              centroid_embedding_id = std::any_cast<long long> (val);
            }
          else if (val.type () == typeid (int))
            {
              centroid_embedding_id = std::any_cast<int> (val);
            }
        }

      // 5. Create MEMORIES row for centroid with kind='ASSOCIATION' (v2 schema).
      // Store summary_text as label + blob, n_signals as cluster size.
      AddWrite (tx,
                "INSERT INTO memories "
                "(embedding_id, source_id, kind, label, start_ts, n_signals, blob_id, created_at) "
                "VALUES (?, ?, 'ASSOCIATION', ?, ?, ?, ?, ?)",
                { centroid_embedding_id, summary_id, summary_text,
                  static_cast<long long> (now_ts),
                  static_cast<long long> (cluster.embedding_ids.size ()),
                  summary_blob_id, static_cast<long long> (now_ts) });

      // Get the memory_id for the centroid memory
      auto mem_id_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
      centroid_memory_id = 0;
      if (!mem_id_rows.empty () && mem_id_rows[0].count ("id"))
        {
          auto val = mem_id_rows[0].at ("id");
          if (val.type () == typeid (long long))
            {
              centroid_memory_id = std::any_cast<long long> (val);
            }
          else if (val.type () == typeid (int))
            {
              centroid_memory_id = std::any_cast<int> (val);
            }
        }

      if (centroid_memory_id > 0 && centroid_embedding_id > 0
          && !cluster.centroid.empty ())
        {
          Eigen::VectorXf centroid_vec (
              static_cast<Eigen::Index> (cluster.centroid.size ()));
          for (size_t i = 0; i < cluster.centroid.size (); ++i)
            {
              centroid_vec (static_cast<Eigen::Index> (i))
                  = cluster.centroid[i];
            }
          context.GetProcessorContext ().UpsertSummaryCache (
              centroid_memory_id, centroid_embedding_id, centroid_vec, true, false);
        }

      // 6. Update cluster_id in memories for source embeddings and create
      // ASSOCIATIONS (derived_from edges).
      for (const auto &[emb_id, src_memory_id] : source_links)
        {
          internal::ThrowIfStopRequested ();
          // Update cluster_id
          AddWrite (tx,
                    "UPDATE memories SET cluster_id = ? "
                    "WHERE embedding_id = ?",
                    { cluster.cluster_id, emb_id });

          // Create derived_from edge: centroid <- source
          if (src_memory_id > 0 && centroid_memory_id > 0)
            {
              AddWrite (tx,
                        "INSERT OR IGNORE INTO associations "
                        "(source_memory_id, target_memory_id, edge_type, weight) "
                        "VALUES (?, ?, 'derived_from', 1.0)",
                        { centroid_memory_id, src_memory_id });
            }
        }

    }

  // 9. Pass extraction requests to next operation.
  const int jobs_queued = static_cast<int> (extraction_requests.size ());
  context.SetExtractionRequests (std::move (extraction_requests));

  telemetry::LogInfo (
      "cortext.consolidation_summarize",
      { telemetry::Attribute::Int64 ("summary_count", summary_count),
        telemetry::Attribute::Int64 ("extraction_jobs_queued", jobs_queued),
        telemetry::Attribute::Int64 ("summaries_with_summarizer",
                                     summaries_with_summarizer),
        telemetry::Attribute::Int64 ("summaries_fallback", 0),
        telemetry::Attribute::Int64 ("associative_cues", associative_cues),
        telemetry::Attribute::Int64 ("clusters_skipped_eviction_frontier",
                                     clusters_skipped_eviction_frontier),
        telemetry::Attribute::Int64 (
            "clusters_skipped_summarizer_unavailable",
            clusters_skipped_summarizer_unavailable),
        telemetry::Attribute::Int64 ("storage_used_bytes",
                                     eviction_frontier.storage_used_bytes),
        telemetry::Attribute::Int64 ("storage_threshold_bytes",
                                     eviction_frontier.storage_threshold_bytes) });
}

} // namespace cortext::operations
