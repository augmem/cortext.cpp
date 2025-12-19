#include "cortext/operations/process_extraction_results.hpp"

#include "cortext/store/store.hpp"
#include "cortext/extractor/extractor.hpp"
#include "cortext/operations/extraction.hpp"
#include "cortext/processor/operation_context.hpp"
#include <any>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

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
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "type": {"type": "string"},
          "salience": {"type": "number"}
        },
        "required": ["name", "type"]
      }
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
  }
})");

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

  for (const auto &result : p_ctx.pending_extraction_results)
    {
      // Track entity name -> memory_id for relation linking
      std::unordered_map<std::string, long long> entity_memory_ids;

      // 1. Insert entities into MEMORIES (kind='LABEL').
      for (const auto &entity : result.entities)
        {
          // Create a MEMORIES entry for the entity label
          // source_id = entity type + name for uniqueness
          std::string source_id = entity.type + ":" + entity.name;

          // Check if this entity already exists
          auto existing = tx.Execute (
              "SELECT memory_id FROM memories WHERE source_id = ? AND kind = 'LABEL'",
              { source_id });

          long long entity_memory_id = 0;
          if (!existing.empty () && existing[0].count ("memory_id"))
            {
              // Entity exists, get its memory_id
              auto val = existing[0].at ("memory_id");
              if (val.type () == typeid (long long))
                {
                  entity_memory_id = std::any_cast<long long> (val);
                }
              else if (val.type () == typeid (int))
                {
                  entity_memory_id = std::any_cast<int> (val);
                }

              // Update salience if higher
              AddWrite (tx,
                        "UPDATE memories SET s_max = MAX(s_max, ?) "
                        "WHERE memory_id = ?",
                        { entity.salience, entity_memory_id });
            }
          else
            {
              // Insert new entity as MEMORIES row
              AddWrite (tx,
                        "INSERT INTO memories "
                        "(source_id, kind, label, start_ts, s_max, created_at) "
                        "VALUES (?, 'LABEL', ?, ?, ?, ?)",
                        { source_id, entity.name,
                          static_cast<long long> (now_ts), entity.salience,
                          static_cast<long long> (now_ts) });

              // Get the new memory_id
              auto id_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
              if (!id_rows.empty () && id_rows[0].count ("id"))
                {
                  auto val = id_rows[0].at ("id");
                  if (val.type () == typeid (long long))
                    {
                      entity_memory_id = std::any_cast<long long> (val);
                    }
                  else if (val.type () == typeid (int))
                    {
                      entity_memory_id = std::any_cast<int> (val);
                    }
                }
            }

          // Track for relation linking
          if (entity_memory_id > 0)
            {
              entity_memory_ids[entity.name] = entity_memory_id;
            }
        }

      // 2. Insert relations into ASSOCIATIONS.
      for (const auto &relation : result.relations)
        {
          // Look up subject and object memory_ids
          long long subject_id = 0;
          long long object_id = 0;

          auto it_subj = entity_memory_ids.find (relation.subject);
          if (it_subj != entity_memory_ids.end ())
            {
              subject_id = it_subj->second;
            }

          auto it_obj = entity_memory_ids.find (relation.object);
          if (it_obj != entity_memory_ids.end ())
            {
              object_id = it_obj->second;
            }

          // Only create association if both entities exist
          if (subject_id > 0 && object_id > 0)
            {
              AddWrite (tx,
                        "INSERT OR REPLACE INTO associations "
                        "(source_memory_id, target_memory_id, edge_type, weight) "
                        "VALUES (?, ?, ?, ?)",
                        { subject_id, object_id, relation.predicate,
                          relation.confidence });
            }
        }
    }

  // 4. Clear pending results.
  p_ctx.pending_extraction_results.clear ();
}

} // namespace cortext::operations
