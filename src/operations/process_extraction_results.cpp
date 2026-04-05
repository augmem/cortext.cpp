#include "cortext/internal/cancellation.hpp"
#include "cortext/operations/process_extraction_results.hpp"

#include "../store/facts.hpp"
#include "cortext/store/store.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/encoder/encoder.hpp"
#include "cortext/extractor/extractor.hpp"
#include "cortext/operations/extraction.hpp"
#include "cortext/operations/label_utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <any>
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
    },
    "facts": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "subject": {"type": "string"},
          "predicate": {"type": "string"},
          "object": {"type": "string"},
          "confidence": {"type": "number"},
          "valid_start_ts": {"type": "integer"},
          "valid_end_ts": {"type": "integer"}
        },
        "required": ["subject", "predicate", "object"]
      }
    }
  },
  "required": ["labels"]
})");

constexpr int kEmbeddingDim = 256;

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

std::optional<std::vector<float>>
LoadEmbeddingVector (Transaction &tx, long long embedding_id,
                     int expected_dim = kEmbeddingDim)
{
  auto emb = LoadEmbedding (tx, embedding_id, expected_dim);
  if (!emb.has_value ())
    {
      return std::nullopt;
    }
  std::vector<float> out (static_cast<size_t> (emb->size ()));
  Eigen::Map<Eigen::VectorXf> out_vec (out.data (),
                                       static_cast<int> (out.size ()));
  out_vec = *emb;
  return out;
}

double
ComputeLabelSalience (const std::vector<float> *label_embedding,
                      const std::optional<Eigen::VectorXf> &summary_embedding)
{
  if (!summary_embedding.has_value () || summary_embedding->size () == 0
      || label_embedding == nullptr || label_embedding->empty ())
    {
      return 0.5;
    }
  if (static_cast<int> (label_embedding->size ()) != summary_embedding->size ())
    {
      return 0.5;
    }

  Eigen::Map<const Eigen::VectorXf> label_vec (
      label_embedding->data (),
      static_cast<int> (label_embedding->size ()));
  const double cos = core::CosineSimilarity (label_vec, *summary_embedding);
  return core::Map01 (cos);
}

std::optional<std::vector<float>>
EncodeLabelEmbedding (const std::string &label, Encoder &encoder)
{
  std::vector<float> embedding;
  try
    {
      encoder.EncodeText (label, embedding);
    }
  catch (const std::exception &)
    {
      return std::nullopt;
    }
  if (embedding.empty () || embedding.size () != kEmbeddingDim)
    {
      return std::nullopt;
    }
  return embedding;
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

std::optional<long long>
OptionalI64 (std::uint64_t value)
{
  return static_cast<long long> (value);
}

std::any
NullableAny (const std::optional<long long> &value)
{
  if (value.has_value ())
    {
      return std::any (*value);
    }
  return std::any ();
}

long long
ExtractInt64Field (const std::map<std::string, std::any> &row,
                   const char *field)
{
  auto it = row.find (field);
  if (it == row.end () || !it->second.has_value ())
    {
      return 0;
    }
  if (it->second.type () == typeid (long long))
    {
      return std::any_cast<long long> (it->second);
    }
  if (it->second.type () == typeid (int))
    {
      return std::any_cast<int> (it->second);
    }
  return 0;
}

double
RequiredSupersessionConfidence (double existing_confidence, double sensitivity,
                                double stability)
{
  const double s = core::SensitivityBias (sensitivity);
  const double t = core::Clamp (stability, 0.0, 1.0);
  const double margin = core::Lerp (0.18, 0.03, s)
                        * core::Lerp (1.10, 0.90, t);
  return core::Clamp (
      existing_confidence
          + margin * (1.0 - core::Clamp (existing_confidence, 0.0, 1.0)),
      0.0, 1.0);
}

double
ExtractDoubleField (const std::map<std::string, std::any> &row,
                    const char *field, double fallback = 0.0)
{
  auto it = row.find (field);
  if (it == row.end () || !it->second.has_value ())
    {
      return fallback;
    }
  if (it->second.type () == typeid (double))
    {
      return std::any_cast<double> (it->second);
    }
  if (it->second.type () == typeid (float))
    {
      return static_cast<double> (std::any_cast<float> (it->second));
    }
  if (it->second.type () == typeid (int))
    {
      return static_cast<double> (std::any_cast<int> (it->second));
    }
  if (it->second.type () == typeid (long long))
    {
      return static_cast<double> (std::any_cast<long long> (it->second));
    }
  return fallback;
}

} // namespace

void
ProcessExtractionResults::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();

  // Get extractor (may be null if OGA disabled)
  Extractor *extractor = context.GetExtractor ();

  // Process extraction requests using in-process extractor if available
  const auto &requests = context.GetExtractionRequests ();
  if (extractor && extractor->IsAvailable () && !requests.empty ())
    {
      for (const auto &req : requests)
        {
          internal::ThrowIfStopRequested ();
          try
            {
              // Extract from raw episodic source texts (not the compressed
              // summary). The summary serves as a retrieval node but
              // extraction works on the original evidence to avoid lossy
              // compression hiding extractable facts.
              std::string combined_text;
              if (!req.source_texts.empty ())
                {
                  for (const auto &txt : req.source_texts)
                    {
                      if (!combined_text.empty ())
                        combined_text += "\n---\n";
                      combined_text += txt;
                    }
                }
              else
                {
                  combined_text = req.summary_text;
                }

              auto result
                  = extractor->ExtractFromText (combined_text, kExtractionSchema);
              result.summary_id = req.summary_id;

              // Add to pending results for processing
              p_ctx.pending_extraction_results.push_back (std::move (result));
            }
          catch (const std::exception &e)
            {
              telemetry::LogWarn (
                  "cortext.extraction_failed",
                  { telemetry::Attribute::String ("summary_id",
                                                  req.summary_id),
                    telemetry::Attribute::Int64 ("cluster_size",
                                                 req.cluster_size),
                    telemetry::Attribute::String ("error", e.what ()) });
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

  int total_results = static_cast<int> (p_ctx.pending_extraction_results.size ());
  int total_labels = 0;
  int total_relations = 0;
  int total_facts = 0;
  std::vector<long long> touched_fact_ids;
  for (const auto &pending : p_ctx.pending_extraction_results)
    {
      total_labels += static_cast<int> (pending.labels.size ());
      total_relations += static_cast<int> (pending.relations.size ());
      total_facts += static_cast<int> (pending.facts.size ());
    }

  const uint64_t now_ts = context.GetSignal ().timestamp;
  Encoder *lifecycle_encoder = context.GetConfig ().encoder;
  const int label_threshold = core::LabelFrequencyThreshold (cfg.stability);
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

  struct ExistingLabel
  {
    long long memory_id = 0;
    long long embedding_id = 0;
    bool loaded = false;
  };

      auto &label_cache = p_ctx.label_embedding_cache;
      auto upsert_summary_cache = [&](long long memory_id,
                                      long long embedding_id,
                                      const std::vector<float> *embedding) {
        if (!embedding || embedding->empty ())
          {
            return;
          }
        Eigen::VectorXf vec (
            static_cast<Eigen::Index> (embedding->size ()));
        for (size_t i = 0; i < embedding->size (); ++i)
          {
            vec (static_cast<Eigen::Index> (i)) = (*embedding)[i];
          }
        p_ctx.UpsertSummaryCache (memory_id, embedding_id, vec, false, true);
      };
  std::unordered_map<std::string, ExistingLabel> existing_labels;
  auto get_existing = [&](const std::string &label_key) -> ExistingLabel & {
    auto it = existing_labels.find (label_key);
    if (it != existing_labels.end ())
      {
        return it->second;
      }
    ExistingLabel entry;
    auto rows = tx.Execute (
        "SELECT memory_id, embedding_id FROM memories "
        "WHERE source_id = ? AND kind = 'LABEL'",
        { label_key });
    if (!rows.empty ())
      {
        const auto &row = rows[0];
        auto mem_it = row.find ("memory_id");
        if (mem_it != row.end ())
          {
            if (mem_it->second.type () == typeid (long long))
              {
                entry.memory_id = std::any_cast<long long> (mem_it->second);
              }
            else if (mem_it->second.type () == typeid (int))
              {
                entry.memory_id = std::any_cast<int> (mem_it->second);
              }
          }
        auto emb_it = row.find ("embedding_id");
        if (emb_it != row.end ())
          {
            if (emb_it->second.type () == typeid (long long))
              {
                entry.embedding_id
                    = std::any_cast<long long> (emb_it->second);
              }
            else if (emb_it->second.type () == typeid (int))
              {
                entry.embedding_id = std::any_cast<int> (emb_it->second);
              }
          }
      }
    entry.loaded = true;
    auto [ins_it, _] = existing_labels.emplace (label_key, entry);
    return ins_it->second;
  };

  for (const auto &result : p_ctx.pending_extraction_results)
    {
      // Find summary memory for has_label edges.
      long long summary_memory_id = 0;
      long long summary_embedding_id = 0;
      long long summary_start_ts = 0;
      auto summary_rows = tx.Execute (
          "SELECT memory_id, embedding_id, start_ts FROM memories "
          "WHERE source_id = ? AND kind = 'ASSOCIATION'",
          { result.summary_id });
      if (!summary_rows.empty ())
        {
          const auto &row = summary_rows[0];
          summary_memory_id = ExtractInt64Field (row, "memory_id");
          summary_embedding_id = ExtractInt64Field (row, "embedding_id");
          summary_start_ts = ExtractInt64Field (row, "start_ts");
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
      bool inserted_any_label = false;
      struct FallbackLabel
      {
        std::string label;
        std::string label_key;
        double salience = 0.0;
        std::vector<float> embedding;
        bool has_embedding = false;
      };
      std::optional<FallbackLabel> fallback_label;

      auto insert_label = [&](const std::string &label,
                              const std::string &label_key,
                              double salience,
                              const std::vector<float> *label_embedding,
                              ExistingLabel &existing) -> bool {
        if (label.empty () || label_key.empty ())
          {
            return false;
          }

        // Use label text as source_id for uniqueness.
        std::string source_id = label_key;

        long long label_memory_id = existing.memory_id;
        long long existing_embedding_id = existing.embedding_id;

        if (label_memory_id > 0)
          {
            // Update salience if higher
            AddWrite (tx,
                      "UPDATE memories SET s_max = MAX(s_max, ?) "
                      "WHERE memory_id = ?",
                      { salience, label_memory_id });
          }
        else
          {
            long long embedding_id = 0;
            if (label_embedding != nullptr && !label_embedding->empty ())
              {
                AddWrite (tx,
                          "INSERT INTO embeddings (embedding, created_at) "
                          "VALUES (?, ?)",
                          { *label_embedding,
                            static_cast<long long> (now_ts) });
                auto emb_rows
                    = tx.Execute ("SELECT last_insert_rowid() AS id", {});
                if (!emb_rows.empty () && emb_rows[0].count ("id"))
                  {
                    auto emb_val = emb_rows[0].at ("id");
                    if (emb_val.type () == typeid (long long))
                      {
                        embedding_id
                            = std::any_cast<long long> (emb_val);
                      }
                    else if (emb_val.type () == typeid (int))
                      {
                        embedding_id = std::any_cast<int> (emb_val);
                      }
                  }
              }
            // Insert new label as MEMORIES row
            AddWrite (tx,
                      "INSERT INTO memories "
                      "(embedding_id, source_id, kind, label, start_ts, "
                      "s_max, created_at) "
                      "VALUES (?, ?, 'LABEL', ?, ?, ?, ?)",
                      { embedding_id > 0 ? std::any (embedding_id) : std::any (),
                        source_id, label,
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
            existing.memory_id = label_memory_id;
            existing.embedding_id = embedding_id;
            existing_embedding_id = embedding_id;
          }

        if (label_memory_id > 0 && existing_embedding_id == 0
            && label_embedding != nullptr && !label_embedding->empty ())
          {
            AddWrite (tx,
                      "INSERT INTO embeddings (embedding, created_at) "
                      "VALUES (?, ?)",
                      { *label_embedding,
                        static_cast<long long> (now_ts) });
            auto emb_rows
                = tx.Execute ("SELECT last_insert_rowid() AS id", {});
            long long embedding_id = 0;
            if (!emb_rows.empty () && emb_rows[0].count ("id"))
              {
                auto emb_val = emb_rows[0].at ("id");
                if (emb_val.type () == typeid (long long))
                  {
                    embedding_id = std::any_cast<long long> (emb_val);
                  }
                else if (emb_val.type () == typeid (int))
                  {
                    embedding_id = std::any_cast<int> (emb_val);
                  }
              }
            if (embedding_id > 0)
              {
                AddWrite (tx,
                          "UPDATE memories SET embedding_id = ? "
                          "WHERE memory_id = ?",
                          { embedding_id, label_memory_id });
                existing.embedding_id = embedding_id;
                upsert_summary_cache (label_memory_id, embedding_id,
                                      label_embedding);
              }
          }

        // Attach label to summary (has_label) and track for relation linking
        if (label_memory_id > 0)
          {
            label_memory_ids[label_key] = label_memory_id;
            if (existing.embedding_id > 0)
              {
                upsert_summary_cache (label_memory_id, existing.embedding_id,
                                      label_embedding);
              }
            if (summary_memory_id > 0)
              {
                const double weight01 = core::Clamp (salience, 0.0, 1.0);
                AddWrite (tx,
                          "INSERT OR REPLACE INTO associations "
                          "(source_memory_id, target_memory_id, edge_type, weight) "
                          "VALUES (?, ?, 'has_label', ?)",
                          { summary_memory_id, label_memory_id, weight01 });
              }
            return true;
          }
        return false;
      };

      // 1. Insert labels into MEMORIES (kind='LABEL').
      for (const auto &label_entry : result.labels)
        {
          const std::string label = TrimLabel (label_entry.label);
          const std::string label_key = NormalizeLabelKey (label_entry.label);
          if (label.empty () || label_key.empty ())
            {
              continue;
            }
          auto &existing = get_existing (label_key);
          const std::vector<float> *label_embedding_ptr = nullptr;
          auto cached_it = label_cache.find (label_key);
          if (cached_it != label_cache.end ())
            {
              label_embedding_ptr = &cached_it->second;
            }
          else if (existing.embedding_id > 0)
            {
              auto loaded = LoadEmbeddingVector (tx, existing.embedding_id);
              if (loaded.has_value ())
                {
                  auto [it, _] = label_cache.emplace (label_key,
                                                      std::move (*loaded));
                  label_embedding_ptr = &it->second;
                }
            }
          if (!label_embedding_ptr)
            {
              auto encoded = EncodeLabelEmbedding (label, *encoder);
              if (encoded.has_value ())
                {
                  auto [it, _] = label_cache.emplace (label_key,
                                                      std::move (*encoded));
                  label_embedding_ptr = &it->second;
                }
            }
          const double salience
              = ComputeLabelSalience (label_embedding_ptr, summary_embedding);
          if (label_threshold > 1
              && label_counts[label_key] < label_threshold)
            {
              if (!fallback_label || salience > fallback_label->salience)
                {
                  FallbackLabel candidate;
                  candidate.label = label;
                  candidate.label_key = label_key;
                  candidate.salience = salience;
                  if (label_embedding_ptr && !label_embedding_ptr->empty ())
                    {
                      candidate.embedding = *label_embedding_ptr;
                      candidate.has_embedding = true;
                    }
                  fallback_label = std::move (candidate);
                }
              continue;
            }

          if (insert_label (label, label_key, salience, label_embedding_ptr,
                            existing))
            {
              inserted_any_label = true;
            }
        }

      if (!inserted_any_label && fallback_label.has_value ())
        {
          const std::vector<float> *fallback_embedding
              = fallback_label->has_embedding
                    ? &fallback_label->embedding
                    : nullptr;
          if (insert_label (fallback_label->label,
                            fallback_label->label_key,
                            fallback_label->salience,
                            fallback_embedding,
                            get_existing (fallback_label->label_key)))
            {
              inserted_any_label = true;
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

      // 3. Insert fact assertions and evidence.
      std::vector<long long> evidence_memory_ids;
      if (summary_memory_id > 0)
        {
          evidence_memory_ids.push_back (summary_memory_id);
          auto evidence_rows = tx.Execute (
              "SELECT target_memory_id FROM associations "
              "WHERE source_memory_id = ? AND edge_type = 'derived_from'",
              { summary_memory_id });
          for (const auto &row : evidence_rows)
            {
              const long long source_memory_id
                  = ExtractInt64Field (row, "target_memory_id");
              if (source_memory_id > 0)
                {
                  evidence_memory_ids.push_back (source_memory_id);
                }
            }
        }

      for (const auto &fact : result.facts)
        {
          const std::string canonical_subject
              = store::NormalizeFactTerm (fact.subject);
          const std::string canonical_predicate
              = store::NormalizeFactPredicate (fact.predicate);
          const std::string canonical_object
              = store::NormalizeFactTerm (fact.object);
          if (canonical_subject.empty () || canonical_predicate.empty ()
              || canonical_object.empty () || summary_memory_id <= 0)
            {
              continue;
            }

          const std::optional<long long> valid_start_ts
              = fact.valid_start_ts.has_value ()
                    ? OptionalI64 (*fact.valid_start_ts)
                    : (summary_start_ts > 0
                           ? std::optional<long long> (summary_start_ts)
                           : std::nullopt);
          const std::optional<long long> valid_end_ts
              = fact.valid_end_ts.has_value ()
                    ? OptionalI64 (*fact.valid_end_ts)
                    : std::nullopt;
          const bool explicit_valid_start = fact.valid_start_ts.has_value ();
          const bool explicit_valid_end = fact.valid_end_ts.has_value ();
          const double confidence = core::Clamp (fact.confidence, 0.0, 1.0);

          long long fact_id = 0;
          const auto duplicate_rows
              = (!explicit_valid_start && !explicit_valid_end)
                    ? tx.Execute (
                          "SELECT fact_id, confidence FROM fact_assertions "
                          "WHERE canonical_subject = ? "
                          "  AND canonical_predicate = ? "
                          "  AND canonical_object = ? "
                          "  AND (superseded_at_ts IS NULL OR superseded_at_ts > ?) "
                          "ORDER BY recorded_at_ts DESC LIMIT 1",
                          { canonical_subject, canonical_predicate,
                            canonical_object,
                            static_cast<long long> (now_ts) })
                    : tx.Execute (
                          "SELECT fact_id, confidence FROM fact_assertions "
                          "WHERE canonical_subject = ? "
                          "  AND canonical_predicate = ? "
                          "  AND canonical_object = ? "
                          "  AND ((valid_start_ts IS NULL AND ? IS NULL) OR valid_start_ts = ?) "
                          "  AND ((valid_end_ts IS NULL AND ? IS NULL) OR valid_end_ts = ?) "
                          "  AND (superseded_at_ts IS NULL OR superseded_at_ts > ?) "
                          "ORDER BY recorded_at_ts DESC LIMIT 1",
                          { canonical_subject, canonical_predicate,
                            canonical_object,
                            NullableAny (valid_start_ts),
                            NullableAny (valid_start_ts),
                            NullableAny (valid_end_ts),
                            NullableAny (valid_end_ts),
                            static_cast<long long> (now_ts) });
          if (!duplicate_rows.empty ())
            {
              fact_id = ExtractInt64Field (duplicate_rows[0], "fact_id");
              const double merged_confidence
                  = std::max (confidence,
                              ExtractDoubleField (duplicate_rows[0],
                                                  "confidence", confidence));
              if (fact_id > 0)
                {
                  if (!explicit_valid_start && valid_start_ts.has_value ())
                    {
                      AddWrite (
                          tx,
                          "UPDATE fact_assertions "
                          "SET confidence = ?, "
                          "    confirmation_count = confirmation_count + 1, "
                          "    compressed_support_count = compressed_support_count + 1, "
                          "    last_confirmation_ts = ?, "
                          "    lifecycle_state = 'active', "
                          "    archived_at = NULL, "
                          "    valid_start_ts = CASE "
                          "      WHEN valid_start_ts IS NULL OR valid_start_ts > ? THEN ? "
                          "      ELSE valid_start_ts END "
                          "WHERE fact_id = ?",
                          { merged_confidence,
                            static_cast<long long> (now_ts),
                            *valid_start_ts,
                            *valid_start_ts,
                            fact_id });
                    }
                  else
                    {
                      AddWrite (tx,
                                "UPDATE fact_assertions "
                                "SET confidence = ?, "
                                "    confirmation_count = confirmation_count + 1, "
                                "    compressed_support_count = compressed_support_count + 1, "
                                "    last_confirmation_ts = ?, "
                                "    lifecycle_state = 'active', "
                                "    archived_at = NULL "
                                "WHERE fact_id = ?",
                                { merged_confidence,
                                  static_cast<long long> (now_ts), fact_id });
                    }
                }
            }
          else
            {
              auto conflicting_rows = tx.Execute (
                  "SELECT fact_id, confidence FROM fact_assertions "
                  "WHERE canonical_subject = ? "
                  "  AND canonical_predicate = ? "
                  "  AND canonical_object <> ? "
                  "  AND (superseded_at_ts IS NULL OR superseded_at_ts > ?) "
                  "  AND (valid_end_ts IS NULL OR valid_end_ts > ?)",
                  { canonical_subject, canonical_predicate, canonical_object,
                    static_cast<long long> (now_ts),
                    NullableAny (valid_start_ts) });

              double strongest_conflicting_confidence = 0.0;
              for (const auto &row : conflicting_rows)
                {
                  strongest_conflicting_confidence = std::max (
                      strongest_conflicting_confidence,
                      ExtractDoubleField (row, "confidence", 0.0));
                }
              const bool accept_conflicting_update
                  = conflicting_rows.empty ()
                    || confidence
                           >= RequiredSupersessionConfidence (
                               strongest_conflicting_confidence,
                               cfg.sensitivity, cfg.stability);

              for (const auto &row : conflicting_rows)
                {
                  const long long conflicting_fact_id
                      = ExtractInt64Field (row, "fact_id");
                  if (conflicting_fact_id <= 0)
                    {
                      continue;
                    }
                  if (!accept_conflicting_update)
                    {
                      continue;
                    }
                  AddWrite (
                      tx,
                      "UPDATE fact_assertions "
                      "SET valid_end_ts = CASE "
                      "      WHEN ? IS NULL THEN valid_end_ts "
                      "      WHEN valid_end_ts IS NULL OR valid_end_ts > ? THEN ? "
                      "      ELSE valid_end_ts END, "
                      "    superseded_at_ts = ?, "
                      "    challenge_count = challenge_count + 1, "
                      "    last_challenge_ts = ? "
                      "WHERE fact_id = ?",
                      { NullableAny (valid_start_ts), NullableAny (valid_start_ts),
                        NullableAny (valid_start_ts),
                        static_cast<long long> (now_ts),
                        static_cast<long long> (now_ts), conflicting_fact_id });
                  touched_fact_ids.push_back (conflicting_fact_id);
                  store::RefreshFactCache (tx, encoder, conflicting_fact_id,
                                           now_ts);
                }

              AddWrite (tx,
                        "INSERT INTO fact_assertions "
                        "(subject, predicate, object, canonical_subject, "
                        " canonical_predicate, canonical_object, valid_start_ts, "
                        " valid_end_ts, recorded_at_ts, superseded_at_ts, "
                        " confidence, summary_memory_id, created_at, "
                        " support_mass, source_diversity, contradiction_mass, "
                        " confirmation_count, challenge_count, "
                        " compressed_support_count, last_confirmation_ts, "
                        " last_challenge_ts, severity_class, lifecycle_state, "
                        " archived_at, last_maintenance_ts) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                        "        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        { fact.subject, fact.predicate, fact.object,
                          canonical_subject, canonical_predicate, canonical_object,
                          NullableAny (valid_start_ts),
                          NullableAny (valid_end_ts),
                          static_cast<long long> (now_ts),
                          accept_conflicting_update ? std::any ()
                                                    : std::any (
                                                          static_cast<long long> (
                                                              now_ts)),
                          confidence, summary_memory_id,
                          static_cast<long long> (now_ts),
                          0.0,
                          0LL,
                          strongest_conflicting_confidence,
                          1LL,
                          0LL,
                          0LL,
                          static_cast<long long> (now_ts),
                          std::any (),
                          store::PredicateSeverityClass (canonical_predicate),
                          accept_conflicting_update
                              ? std::any (std::string ("active"))
                              : std::any (std::string ("archived")),
                          accept_conflicting_update ? std::any ()
                                                    : std::any (
                                                          static_cast<long long> (
                                                              now_ts)),
                          0LL });
              auto fact_rows
                  = tx.Execute ("SELECT last_insert_rowid() AS id", {});
              if (!fact_rows.empty () && fact_rows[0].count ("id"))
                {
                  fact_id = ExtractInt64Field (fact_rows[0], "id");
                }
            }

          if (fact_id <= 0)
            {
              continue;
            }
          touched_fact_ids.push_back (fact_id);

          for (size_t i = 0; i < evidence_memory_ids.size (); ++i)
            {
              const long long evidence_memory_id = evidence_memory_ids[i];
              if (evidence_memory_id <= 0)
                {
                  continue;
                }
              const std::string evidence_type
                  = i == 0 ? "summary" : "episodic";
              const double support_weight = i == 0 ? 1.0 : 0.75;
              AddWrite (tx,
                        "INSERT OR IGNORE INTO fact_evidence "
                        "(fact_id, source_memory_id, evidence_type, support_weight) "
                        "VALUES (?, ?, ?, ?)",
                        { fact_id, evidence_memory_id, evidence_type,
                          support_weight });
            }

          store::RefreshFactCache (tx, encoder, fact_id, now_ts);
        }
    }

  if (!touched_fact_ids.empty ())
    {
      store::MaintainFactLifecycle (tx, lifecycle_encoder, now_ts,
                                    touched_fact_ids);
    }

  // 4. Clear pending results.
  p_ctx.pending_extraction_results.clear ();

  telemetry::LogInfo ("cortext.process_extraction_results", {
    telemetry::Attribute::Int64 ("results_processed", total_results),
    telemetry::Attribute::Int64 ("labels_seen", total_labels),
    telemetry::Attribute::Int64 ("relations_seen", total_relations),
    telemetry::Attribute::Int64 ("facts_seen", total_facts),
    telemetry::Attribute::Int64 ("label_threshold", label_threshold)
  });
}

} // namespace cortext::operations
