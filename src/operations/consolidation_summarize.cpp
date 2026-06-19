#include "cortext/internal/cancellation.hpp"
#include "cortext/operations/consolidation_summarize.hpp"
#include <limits>
#include <cstdint>

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/operations/extraction.hpp"
#include "cortext/operations/label_utils.hpp"
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
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <optional>
#include <regex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
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

bool
EnvBool (const char *name, bool fallback = false)
{
  const char *value = std::getenv (name);
  if (value == nullptr)
    {
      return fallback;
    }
  std::string text (value);
  std::transform (text.begin (), text.end (), text.begin (),
                  [] (unsigned char c) {
                    return static_cast<char> (std::tolower (c));
                  });
  return text == "1" || text == "true" || text == "yes" || text == "on";
}

int
EnvInt (const char *name, int fallback, int min_value, int max_value)
{
  const char *raw = std::getenv (name);
  if (raw == nullptr || *raw == '\0')
    {
      return fallback;
    }

  char *end = nullptr;
  const long parsed = std::strtol (raw, &end, 10);
  if (end == raw)
    {
      return fallback;
    }
  return std::clamp (static_cast<int> (parsed), min_value, max_value);
}

bool
StmLtmAuditEnabled ()
{
  return EnvBool ("CORTEXT_STM_LTM_AUDIT", false);
}

std::string
AuditJoinStrings (const std::vector<std::string> &values)
{
  std::string out;
  for (const auto &value : values)
    {
      if (!out.empty ())
        {
          out += "\n";
        }
      out += value;
    }
  return out;
}

std::string
AuditJoinInt64s (const std::vector<long long> &values)
{
  std::string out;
  for (long long value : values)
    {
      if (!out.empty ())
        {
          out += ",";
        }
      out += std::to_string (value);
    }
  return out;
}

void
EnsureStmLtmAuditTable (Transaction &tx)
{
  AddWrite (
      tx,
      "CREATE TABLE IF NOT EXISTS stm_ltm_relabel_audit ("
      "  summary_id TEXT PRIMARY KEY,"
      "  created_at INTEGER,"
      "  cluster_size INTEGER,"
      "  source_memory_count INTEGER,"
      "  source_text_count INTEGER,"
      "  source_blob_count INTEGER,"
      "  source_memory_ids TEXT,"
      "  stm_graph_count INTEGER,"
      "  stm_item_count INTEGER,"
      "  stm_label_edge_count INTEGER,"
      "  current_label_count INTEGER,"
      "  current_labels TEXT,"
      "  refined_label_count INTEGER DEFAULT 0,"
      "  refined_labels TEXT DEFAULT '',"
      "  kept_label_count INTEGER DEFAULT 0,"
      "  added_label_count INTEGER DEFAULT 0,"
      "  removed_label_count INTEGER DEFAULT 0,"
      "  removed_labels TEXT DEFAULT '',"
      "  has_label_edges_after INTEGER DEFAULT 0,"
      "  derived_from_edges INTEGER DEFAULT 0,"
      "  relation_count INTEGER DEFAULT 0,"
      "  relation_edges_created INTEGER DEFAULT 0,"
      "  relation_edges_skipped_non_durable_endpoint INTEGER DEFAULT 0,"
      "  relation_edges_skipped_missing_endpoint INTEGER DEFAULT 0,"
      "  relation_edges_skipped_unsupported_predicate INTEGER DEFAULT 0,"
      "  relation_endpoint_direct_hits INTEGER DEFAULT 0,"
      "  relation_endpoint_repair_hits INTEGER DEFAULT 0,"
      "  relation_endpoint_created_labels INTEGER DEFAULT 0,"
      "  relation_endpoint_relation_backed_labels INTEGER DEFAULT 0,"
      "  relation_endpoint_rejected_count INTEGER DEFAULT 0,"
      "  relation_endpoint_rejected_non_durable INTEGER DEFAULT 0,"
      "  relation_endpoint_rejected_ungrounded INTEGER DEFAULT 0,"
      "  fact_assertions_touched INTEGER DEFAULT 0,"
      "  source_memories_with_content INTEGER DEFAULT 0"
      ")");
}

void
AuditQueuedRelabelRequest (
    Transaction &tx, const ProcessorContext &p_ctx,
    const std::string &association_source_id, uint64_t now_ts,
    const ClusterInfo &cluster, const std::vector<long long> &source_memory_ids,
    const std::vector<std::string> &source_texts,
    const std::vector<ExtractionSourceBlob> &source_blobs,
    const std::vector<std::string> &current_labels)
{
  if (!StmLtmAuditEnabled ())
    {
      return;
    }

  int stm_item_count = 0;
  int stm_label_edge_count = 0;
  for (const auto &[source_id, graph] : p_ctx.short_term_graphs)
    {
      (void)source_id;
      stm_item_count += static_cast<int> (graph.items.size ());
      stm_label_edge_count += static_cast<int> (graph.label_edges.size ());
    }

  EnsureStmLtmAuditTable (tx);
  AddWrite (
      tx,
      "INSERT OR REPLACE INTO stm_ltm_relabel_audit ("
      "summary_id, created_at, cluster_size, source_memory_count, "
      "source_text_count, source_blob_count, source_memory_ids, "
      "stm_graph_count, stm_item_count, stm_label_edge_count, "
      "current_label_count, current_labels) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      { association_source_id,
        static_cast<long long> (now_ts),
        static_cast<long long> (cluster.embedding_ids.size ()),
        static_cast<long long> (source_memory_ids.size ()),
        static_cast<long long> (source_texts.size ()),
        static_cast<long long> (source_blobs.size ()),
        AuditJoinInt64s (source_memory_ids),
        static_cast<long long> (p_ctx.short_term_graphs.size ()),
        static_cast<long long> (stm_item_count),
        static_cast<long long> (stm_label_edge_count),
        static_cast<long long> (current_labels.size ()),
        AuditJoinStrings (current_labels) });
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

std::string
FormatSourceTextForSummary (const std::string &raw_text)
{
  std::string text = TrimAsciiWhitespace (raw_text);
  if (text.empty ())
    {
      return {};
    }
  return text;
}

std::string
CanonicalEvidenceText (const std::string &text)
{
  std::string out;
  out.reserve (text.size ());
  bool previous_space = false;
  for (unsigned char c : text)
    {
      if (std::isalnum (c) != 0)
        {
          out.push_back (static_cast<char> (std::tolower (c)));
          previous_space = false;
        }
      else
        {
          if (!out.empty () && !previous_space)
            {
              out.push_back (' ');
              previous_space = true;
            }
        }
    }
  if (!out.empty () && out.back () == ' ')
    {
      out.pop_back ();
    }
  return out;
}

std::vector<std::string>
SplitCanonicalTokens (const std::string &tokens)
{
  std::vector<std::string> out;
  std::string current;
  for (char c : tokens)
    {
      if (std::isspace (static_cast<unsigned char> (c)) != 0)
        {
          if (!current.empty ())
            {
              out.push_back (current);
              current.clear ();
            }
          continue;
        }
      current.push_back (c);
    }
  if (!current.empty ())
    {
      out.push_back (current);
    }
  return out;
}

bool
LabelAppearsInSourceTexts (const std::string &label_key,
                           const std::string &canonical_source_text)
{
  if (canonical_source_text.empty ())
    {
      return true;
    }
  const std::string label = CanonicalLabelTokenKey (label_key);
  if (label.empty ())
    {
      return false;
    }
  const std::string haystack = " " + canonical_source_text + " ";
  const std::string needle = " " + label + " ";
  if (haystack.find (needle) != std::string::npos)
    {
      return true;
    }

  const auto label_parts = SplitCanonicalTokens (label);
  if (label_parts.size () < 2)
    {
      return false;
    }
  for (const auto &part : label_parts)
    {
      if (haystack.find (" " + part + " ") == std::string::npos)
        {
          return false;
        }
    }
  return true;
}

std::string
JoinCanonicalSourceTexts (const std::vector<std::string> &source_texts)
{
  std::string combined;
  for (const auto &text : source_texts)
    {
      if (text.empty ())
        {
          continue;
        }
      if (!combined.empty ())
        {
          combined += " ";
        }
      combined += text;
    }
  return CanonicalEvidenceText (combined);
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
      "WHERE s.kind = 'LONG_TERM' "
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

std::string
MakeSourceMemoryKey (std::vector<long long> source_memory_ids)
{
  if (source_memory_ids.empty ())
    {
      return {};
    }
  std::sort (source_memory_ids.begin (), source_memory_ids.end ());
  source_memory_ids.erase (
      std::unique (source_memory_ids.begin (), source_memory_ids.end ()),
      source_memory_ids.end ());
  return AuditJoinInt64s (source_memory_ids);
}

struct PendingSummaryAttachment
{
  int cluster_id = 0;
  std::vector<std::pair<long long, long long>> source_links;
};

struct PendingSummaryJob
{
  std::string summary_id;
  std::string source_key;
  std::vector<std::string> source_texts;
  std::vector<float> centroid;
  std::size_t embedding_count = 0;
  std::vector<PendingSummaryAttachment> attachments;
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

std::vector<std::string>
LoadCurrentLabels (Transaction &tx, long long association_memory_id)
{
  if (association_memory_id <= 0)
    {
      return {};
    }
  auto rows = tx.Execute (
      "SELECT l.label FROM associations a "
      "JOIN memories l ON l.memory_id = a.target_memory_id "
      "WHERE a.source_memory_id = ? AND a.edge_type = 'has_label' "
      "  AND l.kind = 'LABEL' "
      "ORDER BY a.weight DESC, l.label ASC",
      { association_memory_id });
  std::vector<std::string> labels;
  labels.reserve (rows.size ());
  for (const auto &row : rows)
    {
      auto it = row.find ("label");
      if (it != row.end () && it->second.type () == typeid (std::string))
        {
          labels.push_back (std::any_cast<std::string> (it->second));
        }
    }
  return labels;
}

std::vector<std::string>
MergeShadowCurrentLabels (
    const ProcessorContext &p_ctx, const SignalProcessor::Config &cfg,
    std::vector<std::string> current_labels,
    const std::vector<Eigen::VectorXf> &source_embeddings,
    const std::vector<std::string> &source_texts)
{
  if (source_embeddings.empty () || p_ctx.short_term_graphs.empty ())
    {
      return current_labels;
    }

  std::unordered_set<std::string> seen;
  seen.reserve (current_labels.size ());
  for (const auto &label : current_labels)
    {
      const std::string key = NormalizeLabelKey (label);
      if (!key.empty ())
        {
          seen.insert (key);
        }
    }

  const double min_match = core::STMLabelConsolidationMinSimilarity (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const int max_shadow_labels = std::max (
      1, core::STMLabelConsolidationMaxLabels (cfg.focus, cfg.stability));
  const int max_ungrounded_labels = std::max (
      0, core::STMLabelConsolidationMaxUngrounded (
             cfg.focus, cfg.sensitivity, cfg.stability));
  const std::string canonical_source_text = JoinCanonicalSourceTexts (
      source_texts);
  struct RankedLabel
  {
    double score = 0.0;
    std::string label;
    bool grounded = false;
  };
  std::vector<RankedLabel> ranked;

  for (const auto &[source_id, short_term_graph] : p_ctx.short_term_graphs)
    {
      (void)source_id;
      for (const auto &edge : short_term_graph.label_edges)
        {
          if (edge.label.empty () || edge.signal_embedding.size () == 0)
            {
              continue;
            }
          double best_sim = -1.0;
          for (const auto &source_embedding : source_embeddings)
            {
              if (source_embedding.size () != edge.signal_embedding.size ())
                {
                  continue;
                }
              best_sim = std::max (
                  best_sim,
                  core::CosineSimilarity (source_embedding,
                                          edge.signal_embedding));
            }
          if (best_sim < min_match)
            {
              continue;
            }
          const std::string key = NormalizeLabelKey (edge.label);
          if (key.empty () || seen.find (key) != seen.end ())
            {
              continue;
            }
          const bool grounded = LabelAppearsInSourceTexts (
              key, canonical_source_text);
          if (!grounded && max_ungrounded_labels <= 0)
            {
              continue;
            }
          RankedLabel candidate;
          candidate.score = best_sim * core::Clamp (edge.weight, 0.0, 1.0);
          candidate.label = edge.label;
          candidate.grounded = grounded;
          ranked.push_back (std::move (candidate));
          seen.insert (key);
        }
    }

  if (ranked.empty ())
    {
      return current_labels;
    }
  std::stable_sort (ranked.begin (), ranked.end (),
                    [] (const RankedLabel &a, const RankedLabel &b) {
                      if (a.grounded != b.grounded)
                        {
                          return a.grounded;
                        }
                      return a.score > b.score;
                    });
  int added = 0;
  int ungrounded_added = 0;
  for (const auto &candidate : ranked)
    {
      if (added >= max_shadow_labels)
        {
          break;
        }
      if (!candidate.grounded)
        {
          if (ungrounded_added >= max_ungrounded_labels)
            {
              continue;
            }
          ++ungrounded_added;
        }
      current_labels.push_back (candidate.label);
      ++added;
    }
  return current_labels;
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
                      const std::vector<std::pair<long long, long long>> &source_links,
                      double derived_source_edge_weight)
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
                    "VALUES (?, ?, 'derived_from', ?)",
                    { association_memory_id, src_memory_id,
                      derived_source_edge_weight });
        }
    }
}

long long
PersistDeepSummary (OperationContext &context, Transaction &tx,
                    const PendingSummaryJob &job,
                    const std::string &summary_text, uint64_t now_ts,
                    double derived_source_edge_weight)
{
  std::vector<unsigned char> summary_blob_id;
  summary_blob_id = PutObject (context.GetObjectTransaction (), tx,
                               StringToBlob (summary_text));
  if (summary_blob_id.empty ())
    {
      throw std::runtime_error (
          "Consolidation summarization failed to persist blob for "
          + job.summary_id);
    }

  AddWrite (tx,
            "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
            { job.centroid, static_cast<long long> (now_ts) });

  auto id_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
  long long centroid_embedding_id = 0;
  if (!id_rows.empty () && id_rows[0].count ("id"))
    {
      centroid_embedding_id = AnyToInt64 (id_rows[0].at ("id"));
    }
  if (centroid_embedding_id <= 0)
    {
      throw std::runtime_error (
          "Consolidation summarization failed to persist embedding for "
          + job.summary_id);
    }

  AddWrite (tx,
            "INSERT INTO memories "
            "(embedding_id, source_id, kind, label, start_ts, n_signals, blob_id, created_at) "
            "VALUES (?, ?, 'LONG_TERM', ?, ?, ?, ?, ?)",
            { centroid_embedding_id, job.summary_id, summary_text,
              static_cast<long long> (now_ts),
              static_cast<long long> (job.embedding_count), summary_blob_id,
              static_cast<long long> (now_ts) });

  auto mem_id_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
  long long centroid_memory_id = 0;
  if (!mem_id_rows.empty () && mem_id_rows[0].count ("id"))
    {
      centroid_memory_id = AnyToInt64 (mem_id_rows[0].at ("id"));
    }
  if (centroid_memory_id <= 0)
    {
      throw std::runtime_error (
          "Consolidation summarization failed to persist memory for "
          + job.summary_id);
    }

  if (!job.centroid.empty ())
    {
      Eigen::VectorXf centroid_vec (
          static_cast<Eigen::Index> (job.centroid.size ()));
      for (size_t i = 0; i < job.centroid.size (); ++i)
        {
          centroid_vec (static_cast<Eigen::Index> (i)) = job.centroid[i];
        }
      context.GetProcessorContext ().UpsertSummaryCache (
          centroid_memory_id, centroid_embedding_id, centroid_vec, false, false);
    }

  for (const auto &attachment : job.attachments)
    {
      AttachClusterSources (tx, centroid_memory_id, attachment.cluster_id,
                            attachment.source_links,
                            derived_source_edge_weight);
    }
  return centroid_memory_id;
}

void
QueueExtractionIfNeeded (std::vector<ExtractionRequest> &requests,
                         const ConsolidationSummarizeParams &params,
                         const ClusterInfo &cluster,
                         const std::string &association_source_id,
                         const std::string &summary_text,
                         const std::vector<std::string> &source_texts,
                         const std::vector<ExtractionSourceBlob> &source_blobs,
                         const std::vector<std::string> &current_labels,
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
  req.source_blobs = source_blobs;
  req.current_labels = current_labels;
  requests.push_back (std::move (req));
}

} // namespace

ConsolidationSummarizeParams
ConsolidationSummarizeParams::FromKnobs (double F, double S, double T)
{
  ConsolidationSummarizeParams p;
  p.min_cluster_size_for_extraction = core::MinClusterSizeForExtraction (F);
  const auto budget = core::ConsolidationSummaryEvidenceBudgetForKnobs (
      F, S, T);
  p.max_source_texts = budget.max_source_texts;
  p.max_total_chars = budget.max_total_chars;
  p.max_text_chars = budget.max_text_chars;
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
  const auto mode = context.GetSignal ().consolidation_mode.value_or (
      ConsolidationMode::Both);
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
  const double derived_source_edge_weight = core::DerivedSourceEdgeWeight (
      cfg.focus, cfg.sensitivity, cfg.stability);
  auto &p_ctx = context.GetProcessorContext ();
  const uint64_t now_ts = context.GetSignal ().timestamp;

  const auto eviction_override = eviction::GetEvictionAblationOverride ();
  const auto eviction_frontier = eviction_policy::ResolveEvictionFrontier (
      tx, cfg.stability, static_cast<long long> (std::min<std::uint64_t> (
          p_ctx.last_consolidation_ts,
          static_cast<std::uint64_t> (
              std::numeric_limits<long long>::max ()))),
      eviction_override);
  const bool storage_compression_allowed
      = eviction_frontier.storage_gate_active
        && eviction_frontier.storage_allows_eviction;
  if (!storage_compression_allowed)
    {
      telemetry::LogInfo (
          "cortext.consolidation_summarize.storage_pressure_status",
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
  int clusters_skipped_summarizer_unavailable = 0;

  // Get deep summarizer backend selected during Cortext construction.
  Summarizer *summarizer = context.GetSummarizer ();
  const int summary_batch_size = EnvInt (
      "CORTEXT_CONSOLIDATION_SUMMARY_BATCH_SIZE", 8, 1, 128);
  std::vector<PendingSummaryJob> pending_summary_jobs;
  std::unordered_map<std::string, std::size_t> pending_summary_by_sources;

  auto flush_pending_summaries = [&] {
    if (pending_summary_jobs.empty ())
      {
        return;
      }
    if (!summarizer || !summarizer->IsAvailable ())
      {
        pending_summary_jobs.clear ();
        pending_summary_by_sources.clear ();
        return;
      }

    std::vector<Summarizer::BatchTextItem> items;
    items.reserve (pending_summary_jobs.size ());
    for (const auto &job : pending_summary_jobs)
      {
        Summarizer::BatchTextItem item;
        item.id = job.summary_id;
        item.texts = job.source_texts;
        item.max_words = params.max_summary_words;
        items.push_back (std::move (item));
      }

    std::vector<std::string> summaries;
    try
      {
        internal::ThrowIfStopRequested ();
        const auto summarize_start = std::chrono::steady_clock::now ();
        summaries = summarizer->SummarizeTextBatches (items);
        const auto summarize_end = std::chrono::steady_clock::now ();
        const double batch_ms = std::chrono::duration<double, std::milli> (
                                    summarize_end - summarize_start)
                                    .count ();
        context.AddOperationTiming (
            "ConsolidationSummarize.summarizer_batch_call", batch_ms);
        context.AddOperationTiming ("ConsolidationSummarize.summarizer_call",
                                    batch_ms);
      }
    catch (const internal::CancellationError &)
      {
        throw;
      }
    catch (const std::exception &e)
      {
        throw std::runtime_error (
            "Consolidation summarization batch failed: "
            + std::string (e.what ()));
      }

    if (summaries.size () != pending_summary_jobs.size ())
      {
        throw std::runtime_error (
            "Consolidation summarization batch returned "
            + std::to_string (summaries.size ()) + " summaries for "
            + std::to_string (pending_summary_jobs.size ()) + " jobs");
      }

    for (std::size_t i = 0; i < pending_summary_jobs.size (); ++i)
      {
        auto summary_text = SanitizeSummaryText (std::move (summaries[i]));
        const auto &job = pending_summary_jobs[i];
        if (summary_text.empty ())
          {
            throw std::runtime_error (
                "Consolidation summarization returned empty text for "
                + job.summary_id);
          }
        PersistDeepSummary (context, tx, job, summary_text, now_ts,
                            derived_source_edge_weight);
        ++summary_count;
        ++summaries_with_summarizer;
      }

    pending_summary_jobs.clear ();
    pending_summary_by_sources.clear ();
  };

  for (const auto &cluster : clusters)
    {
      internal::ThrowIfStopRequested ();

      // Collect all source texts and find most representative memory
      struct SourceItem
      {
        std::string text;
        double sim;
        long long start_ts;
      };
      std::vector<SourceItem> source_items;
      source_items.reserve (cluster.embedding_ids.size ());
      std::vector<std::string> source_texts;
      std::vector<ExtractionSourceBlob> source_blobs;
      std::vector<Eigen::VectorXf> source_embeddings;
      std::vector<std::pair<long long, long long>> source_links;
      source_links.reserve (cluster.embedding_ids.size ());
      std::vector<long long> source_memory_ids;
      source_memory_ids.reserve (cluster.embedding_ids.size ());
      std::unordered_set<long long> seen_source_memory_ids;
      long long latest_source_ts = 0;
      const bool source_blobs_enabled
          = !EnvBool ("CORTEXT_DISABLE_SOURCE_BLOBS", false);

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
          // Query embedding, memory row, blob_id, and start_ts.
          auto rows = tx.Execute (
              "SELECT e.embedding, m.memory_id, m.blob_id, "
              "m.start_ts, m.kind, m.modality, "
              "(SELECT s.mime FROM signals s WHERE s.memory_id = m.memory_id "
              " ORDER BY s.serial_position ASC LIMIT 1) AS mime "
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
                  source_embeddings.push_back (std::move (emb));
                }
            }

          // Try to fetch text from objstore.
          std::string text;
          std::vector<unsigned char> payload_bytes;
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
                      payload_bytes = *data;
                      text.assign (data->begin (), data->end ());
                    }
                }
            }

          std::string modality;
          auto it_modality = row.find ("modality");
          if (it_modality != row.end ()
              && it_modality->second.type () == typeid (std::string))
            {
              modality = std::any_cast<std::string> (it_modality->second);
            }

          std::string mime;
          auto it_mime = row.find ("mime");
          if (it_mime != row.end ()
              && it_mime->second.type () == typeid (std::string))
            {
              mime = std::any_cast<std::string> (it_mime->second);
            }
          if (!source_blobs_enabled && modality != "text")
            {
              payload_bytes.clear ();
              text.clear ();
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
          latest_source_ts = std::max (latest_source_ts, start_ts);

          text = FormatSourceTextForSummary (text);

          if (source_blobs_enabled && !payload_bytes.empty ()
              && !modality.empty ())
            {
              ExtractionSourceBlob source_blob;
              source_blob.bytes = std::move (payload_bytes);
              source_blob.modality = modality;
              source_blob.mime = mime;
              if (modality == "audio")
                {
                  source_blob.sample_rate = 16000;
                  source_blob.num_samples = source_blob.bytes.size ()
                                            / sizeof (float);
                }
              source_blobs.push_back (std::move (source_blob));
            }

          // Collect text for summarizer
          if (!text.empty ())
            {
              source_items.push_back (SourceItem{ std::move (text), sim,
                                                  start_ts });
            }
        }

      if (!source_items.empty ())
        {
          internal::ThrowIfStopRequested ();
          std::sort (source_items.begin (), source_items.end (),
                     [] (const SourceItem &a, const SourceItem &b) {
                       return a.sim > b.sim;
                     });
          std::vector<SourceItem> selected_items;
          selected_items.reserve (
              static_cast<size_t> (params.max_source_texts));
          int total_chars = 0;
          for (const auto &item : source_items)
            {
              internal::ThrowIfStopRequested ();
              if (static_cast<int> (selected_items.size ())
                  >= params.max_source_texts)
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

      if (source_texts.empty () && source_blobs.empty ())
        {
          throw std::runtime_error (
              "Consolidation label routing requires source evidence for "
              + std::to_string (cluster.cluster_id));
        }
      const uint64_t evidence_ts
          = latest_source_ts > 0 ? static_cast<uint64_t> (latest_source_ts)
                                 : now_ts;

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
          cue_id = GenerateAssociativeCueId (evidence_ts, cue_sequence++);
          cue_memory_id = InsertAssociativeCue (context, tx, cluster,
                                                cue_id, evidence_ts);
          if (cue_memory_id > 0)
            {
              ++associative_cues;
            }
        }

      AttachClusterSources (tx, cue_memory_id, cluster.cluster_id,
                            source_links, derived_source_edge_weight);
      if (cue_memory_id > 0)
        {
          const bool stm_label_handoff_enabled
              = !EnvBool ("CORTEXT_DISABLE_STM_LABEL_HANDOFF", false);
          const auto current_labels
              = stm_label_handoff_enabled
                    ? MergeShadowCurrentLabels (
                        p_ctx, context.GetConfig (),
                        LoadCurrentLabels (tx, cue_memory_id), source_embeddings,
                        source_texts)
                    : LoadCurrentLabels (tx, cue_memory_id);
          AuditQueuedRelabelRequest (
              tx, p_ctx, cue_id, evidence_ts, cluster, source_memory_ids,
              source_texts, source_blobs, current_labels);
          QueueExtractionIfNeeded (extraction_requests, params, cluster,
                                   cue_id, "", source_texts, source_blobs,
                                   current_labels, evidence_ts);
        }

      if (source_texts.empty ())
        {
          ++clusters_skipped_summarizer_unavailable;
          continue;
        }

      if (!summarizer || !summarizer->IsAvailable ())
        {
          ++clusters_skipped_summarizer_unavailable;
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
                  false,
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
                        "VALUES (?, ?, 'derived_from', ?)",
                        { centroid_memory_id, src_memory_id,
                          derived_source_edge_weight });
            }
          continue;
        }

      const std::string source_key = MakeSourceMemoryKey (source_memory_ids);
      if (!source_key.empty ())
        {
          auto pending_it = pending_summary_by_sources.find (source_key);
          if (pending_it != pending_summary_by_sources.end ())
            {
              pending_summary_jobs[pending_it->second].attachments.push_back (
                  PendingSummaryAttachment{ cluster.cluster_id,
                                            std::move (source_links) });
              continue;
            }
        }

      PendingSummaryJob job;
      job.summary_id = GenerateSummaryId (now_ts, summary_sequence++);
      job.source_key = source_key;
      job.source_texts = std::move (source_texts);
      job.centroid = cluster.centroid;
      job.embedding_count = cluster.embedding_ids.size ();
      job.attachments.push_back (
          PendingSummaryAttachment{ cluster.cluster_id,
                                    std::move (source_links) });
      pending_summary_jobs.push_back (std::move (job));
      if (!source_key.empty ())
        {
          pending_summary_by_sources[source_key]
              = pending_summary_jobs.size () - 1;
        }
      if (static_cast<int> (pending_summary_jobs.size ())
          >= summary_batch_size)
        {
          flush_pending_summaries ();
        }

    }

  flush_pending_summaries ();

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
        telemetry::Attribute::Bool ("storage_compression_allowed",
                                    storage_compression_allowed),
        telemetry::Attribute::Int64 (
            "clusters_skipped_summarizer_unavailable",
            clusters_skipped_summarizer_unavailable),
        telemetry::Attribute::Int64 ("storage_used_bytes",
                                     eviction_frontier.storage_used_bytes),
        telemetry::Attribute::Int64 ("storage_threshold_bytes",
                                     eviction_frontier.storage_threshold_bytes) });
}

} // namespace cortext::operations
