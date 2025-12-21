#include "cortext/operations/process_extraction_results.hpp"

#include "cortext/store/store.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/encoder/encoder.hpp"
#include "cortext/extractor/extractor.hpp"
#include "cortext/operations/extraction.hpp"
#include "cortext/processor/operation_context.hpp"
#include <algorithm>
#include <any>
#include <cctype>
#include <optional>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>

namespace cortext::operations
{

namespace
{

/// @brief Add a write instruction to the transaction.
void
AddWrite (Transaction &tx, const std::string &q,
          const std::vector<std::any> &p = {})
{
  tx.Execute (q, p);
}

/// @brief Default JSON schema for extraction.
const nlohmann::json kExtractionSchema = nlohmann::json::parse (R"({
  "type": "object",
  "properties": {
    "labels": {
      "type": "array",
      "minItems": 1,
      "items": {"type": "string"}
    },
    "relations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "subject": {"type": "string"},
          "predicate": {"type": "string"},
          "object": {"type": "string"},
          "confidence": {"type": "number"}
        },
        "required": ["subject", "predicate", "object"]
      }
    }
  },
  "required": ["labels"]
})");

constexpr int kEmbeddingDim = 256;

std::string
TrimLabel (const std::string &label)
{
  auto start = label.begin ();
  auto end = label.end ();
  while (start != end && std::isspace (static_cast<unsigned char> (*start)))
    {
      ++start;
    }
  while (end != start)
    {
      auto prev = end;
      --prev;
      if (!std::isspace (static_cast<unsigned char> (*prev)))
        {
          break;
        }
      end = prev;
    }
  return std::string (start, end);
}

std::string
NormalizeLabelKey (const std::string &label)
{
  std::string trimmed = TrimLabel (label);
  std::string out;
  out.reserve (trimmed.size ());
  for (unsigned char c : trimmed)
    {
      out.push_back (static_cast<char> (std::tolower (c)));
    }
  return out;
}

std::optional<Eigen::VectorXf>
LoadEmbedding (Transaction &tx, long long embedding_id,
               int expected_dim = kEmbeddingDim)
{
  auto rows = tx.Execute (
      "SELECT embedding FROM embeddings WHERE embedding_id = ?",
      { embedding_id });
  if (rows.empty ())
    {
      return std::nullopt;
    }

  auto it = rows[0].find ("embedding");
  if (it == rows[0].end ())
    {
      return std::nullopt;
    }

  Eigen::VectorXf out;
  if (!core::DecodeFloatBlob (it->second, expected_dim, out))
    {
      return std::nullopt;
    }
  return out;
}

double
ComputeLabelSalience (const std::string &label,
                      const std::optional<Eigen::VectorXf> &summary_embedding,
                      Encoder &encoder)
{
  if (!summary_embedding.has_value () || summary_embedding->size () == 0)
    {
      return 0.5;
    }

  std::vector<float> label_embedding;
  try
    {
      encoder.EncodeText (label, label_embedding);
    }
  catch (const std::exception &)
    {
      return 0.5;
    }

  if (label_embedding.empty ())
    {
      return 0.5;
    }

  Eigen::Map<const Eigen::VectorXf> label_vec (
      label_embedding.data (), static_cast<int> (label_embedding.size ()));
  const double cos = core::CosineSimilarity (label_vec, *summary_embedding);
  return core::Map01 (cos);
}

std::string
NormalizePredicate (const std::string &raw)
{
  std::string norm;
  norm.reserve (raw.size ());
  for (unsigned char c : raw)
    {
      if (std::isalnum (c))
        {
          norm.push_back (static_cast<char> (std::tolower (c)));
        }
      else if (c == '-' || c == ' ' || c == '_')
        {
          norm.push_back ('_');
        }
    }
  // Collapse repeated underscores.
  norm.erase (std::unique (norm.begin (), norm.end (),
                           [] (char a, char b) {
                             return a == '_' && b == '_';
                           }),
              norm.end ());
  if (!norm.empty () && norm.front () == '_')
    {
      norm.erase (norm.begin ());
    }
  if (!norm.empty () && norm.back () == '_')
    {
      norm.pop_back ();
    }
  return norm;
}

std::string
MapEdgeType (const std::string &predicate)
{
  const std::string norm = NormalizePredicate (predicate);
  if (norm == "co_occurs" || norm == "co_occurs_with" || norm == "cooccurs"
      || norm == "co_occur")
    {
      return "co_occurs";
    }
  if (norm == "implies" || norm == "implication" || norm == "imply")
    {
      return "implies";
    }
  if (norm == "contradicts" || norm == "contradiction" || norm == "contradict")
    {
      return "contradicts";
    }
  if (norm == "reinforces" || norm == "reinforce")
    {
      return "reinforces";
    }
  if (norm == "causes" || norm == "cause")
    {
      return "causes";
    }
  if (norm == "similar_to" || norm == "similar" || norm == "similarto")
    {
      return "similar_to";
    }
  return {};
}

} // namespace

void
ProcessExtractionResults::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();

  // Get extractor (may be null if OGA disabled)
  Extractor *extractor = context.GetExtractor ();

  // Process extraction requests using in-process extractor if available
  const auto &requests = context.GetExtractionRequests ();
  if (extractor && extractor->IsAvailable () && !requests.empty ())
    {
      for (const auto &req : requests)
        {
          try
            {
              // Combine source texts for extraction
              std::string combined_text = req.summary_text;
              if (!req.source_texts.empty ())
                {
                  combined_text += "\n\nSource texts:\n";
                  for (const auto &txt : req.source_texts)
                    {
                      combined_text += txt + "\n---\n";
                    }
                }

              auto result
                  = extractor->ExtractFromText (combined_text, kExtractionSchema);
              result.summary_id = req.summary_id;

              // Add to pending results for processing
              p_ctx.pending_extraction_results.push_back (std::move (result));
            }
          catch (const std::exception &)
            {
              // Skip failed extractions
            }
        }
    }
  else if (!requests.empty ())
    {
      // Extractor not available - invoke callback if set
      auto *callback = context.GetExtractionCallback ();
      if (callback && *callback)
        {
          (*callback) (requests);
        }
    }

  // Process pending extraction results (from extractor or external callback)
  if (p_ctx.pending_extraction_results.empty ())
    {
      return;
    }

  const uint64_t now_ts = context.GetSignal ().timestamp;
  const int label_threshold
      = core::LabelFrequencyThreshold (context.GetConfig ().stability);
  std::unordered_map<std::string, int> label_counts;
  for (const auto &pending : p_ctx.pending_extraction_results)
    {
      for (const auto &label_entry : pending.labels)
        {
          const std::string label_key
              = NormalizeLabelKey (label_entry.label);
          if (label_key.empty ())
            {
              continue;
            }
          label_counts[label_key] += 1;
        }
    }

  for (const auto &result : p_ctx.pending_extraction_results)
    {
      // Find summary memory for has_label edges.
      long long summary_memory_id = 0;
      long long summary_embedding_id = 0;
      auto summary_rows = tx.Execute (
          "SELECT memory_id, embedding_id FROM memories "
          "WHERE source_id = ? AND kind = 'ASSOCIATION'",
          { result.summary_id });
      if (!summary_rows.empty ())
        {
          const auto &row = summary_rows[0];
          auto mem_it = row.find ("memory_id");
          if (mem_it != row.end ())
            {
              if (mem_it->second.type () == typeid (long long))
                {
                  summary_memory_id = std::any_cast<long long> (mem_it->second);
                }
              else if (mem_it->second.type () == typeid (int))
                {
                  summary_memory_id = std::any_cast<int> (mem_it->second);
                }
            }
          auto emb_it = row.find ("embedding_id");
          if (emb_it != row.end ())
            {
              if (emb_it->second.type () == typeid (long long))
                {
                  summary_embedding_id
                      = std::any_cast<long long> (emb_it->second);
                }
              else if (emb_it->second.type () == typeid (int))
                {
                  summary_embedding_id = std::any_cast<int> (emb_it->second);
                }
            }
        }

      const auto summary_embedding
          = summary_embedding_id > 0
                ? LoadEmbedding (tx, summary_embedding_id)
                : std::optional<Eigen::VectorXf> ();
      Encoder *encoder = context.GetConfig ().encoder;
      if (!encoder)
        {
          throw std::runtime_error (
              "ProcessExtractionResults requires a non-null Encoder");
        }

      // Track label -> memory_id for relation linking
      std::unordered_map<std::string, long long> label_memory_ids;

      // 1. Insert labels into MEMORIES (kind='LABEL').
      for (const auto &label_entry : result.labels)
        {
          const std::string label = TrimLabel (label_entry.label);
          const std::string label_key = NormalizeLabelKey (label_entry.label);
          if (label.empty () || label_key.empty ())
            {
              continue;
            }
          if (label_threshold > 1
              && label_counts[label_key] < label_threshold)
            {
              continue;
            }

          const double salience
              = ComputeLabelSalience (label, summary_embedding, *encoder);

          // Use label text as source_id for uniqueness.
          std::string source_id = label_key;

          // Check if this label already exists
          auto existing = tx.Execute (
              "SELECT memory_id FROM memories WHERE source_id = ? AND kind = 'LABEL'",
              { source_id });

          long long label_memory_id = 0;
          if (!existing.empty () && existing[0].count ("memory_id"))
            {
              // Label exists, get its memory_id
              auto val = existing[0].at ("memory_id");
              if (val.type () == typeid (long long))
                {
                  label_memory_id = std::any_cast<long long> (val);
                }
              else if (val.type () == typeid (int))
                {
                  label_memory_id = std::any_cast<int> (val);
                }

              // Update salience if higher
              AddWrite (tx,
                        "UPDATE memories SET s_max = MAX(s_max, ?) "
                        "WHERE memory_id = ?",
                        { salience, label_memory_id });
            }
          else
            {
              // Insert new label as MEMORIES row
              AddWrite (tx,
                        "INSERT INTO memories "
                        "(source_id, kind, label, start_ts, s_max, created_at) "
                        "VALUES (?, 'LABEL', ?, ?, ?, ?)",
                        { source_id, label,
                          static_cast<long long> (now_ts), salience,
                          static_cast<long long> (now_ts) });

              // Get the new memory_id
              auto id_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
              if (!id_rows.empty () && id_rows[0].count ("id"))
                {
                  auto val = id_rows[0].at ("id");
                  if (val.type () == typeid (long long))
                    {
                      label_memory_id = std::any_cast<long long> (val);
                    }
                  else if (val.type () == typeid (int))
                    {
                      label_memory_id = std::any_cast<int> (val);
                    }
                }
            }

          // Attach label to summary (has_label) and track for relation linking
          if (label_memory_id > 0)
            {
              label_memory_ids[label_key] = label_memory_id;
              if (summary_memory_id > 0)
                {
                  const double weight01 = core::Clamp (salience, 0.0, 1.0);
                  AddWrite (tx,
                            "INSERT OR REPLACE INTO associations "
                            "(source_memory_id, target_memory_id, edge_type, weight) "
                            "VALUES (?, ?, 'has_label', ?)",
                            { summary_memory_id, label_memory_id, weight01 });
                }
            }
        }

      // 2. Insert relations into ASSOCIATIONS.
      for (const auto &relation : result.relations)
        {
          // Look up subject and object memory_ids
          long long subject_id = 0;
          long long object_id = 0;

          auto it_subj
              = label_memory_ids.find (NormalizeLabelKey (relation.subject));
          if (it_subj != label_memory_ids.end ())
            {
              subject_id = it_subj->second;
            }

          auto it_obj
              = label_memory_ids.find (NormalizeLabelKey (relation.object));
          if (it_obj != label_memory_ids.end ())
            {
              object_id = it_obj->second;
            }

          const std::string edge_type = MapEdgeType (relation.predicate);

          // Only create association if both labels exist and edge is supported
          if (subject_id > 0 && object_id > 0 && !edge_type.empty ())
            {
              const double weight01
                  = core::Clamp (relation.confidence, 0.0, 1.0);
              AddWrite (tx,
                        "INSERT OR REPLACE INTO associations "
                        "(source_memory_id, target_memory_id, edge_type, weight) "
                        "VALUES (?, ?, ?, ?)",
                        { subject_id, object_id, edge_type, weight01 });
            }
        }
    }

  // 4. Clear pending results.
  p_ctx.pending_extraction_results.clear ();
}

} // namespace cortext::operations
