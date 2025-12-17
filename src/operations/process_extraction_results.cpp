#include "cortext/operations/process_extraction_results.hpp"

#include "cortext/store/store.hpp"
#include "cortext/extractor/extractor.hpp"
#include "cortext/operations/extraction.hpp"
#include "cortext/processor/operation_context.hpp"
#include <any>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
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

/// @brief Generate a deterministic node_id from entity name and type.
std::string
GenerateNodeId (const std::string &name, const std::string &type)
{
  std::ostringstream ss;
  ss << type << "_" << name;
  return ss.str ();
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

  for (const auto &result : p_ctx.pending_extraction_results)
    {
      // 1. Insert entities into extraction_entities.
      for (const auto &entity : result.entities)
        {
          AddWrite (tx,
                    "INSERT OR REPLACE INTO extraction_entities"
                    "(summary_id, name, type, salience) "
                    "VALUES(?, ?, ?, ?)",
                    { result.summary_id, entity.name, entity.type,
                      entity.salience });

          // 2. Update entity_index for name→node_id mapping.
          std::string node_id = GenerateNodeId (entity.name, entity.type);
          AddWrite (tx,
                    "INSERT OR IGNORE INTO entity_index(name, node_id) "
                    "VALUES(?, ?)",
                    { entity.name, node_id });
        }

      // 3. Insert relations into extraction_relations.
      for (const auto &relation : result.relations)
        {
          AddWrite (tx,
                    "INSERT INTO extraction_relations"
                    "(summary_id, subject, predicate, object, confidence) "
                    "VALUES(?, ?, ?, ?, ?)",
                    { result.summary_id, relation.subject, relation.predicate,
                      relation.object, relation.confidence });
        }
    }

  // 4. Clear pending results.
  p_ctx.pending_extraction_results.clear ();
}

} // namespace cortext::operations
