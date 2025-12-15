#include "cortext/operations/process_extraction_results.hpp"

#include "cortext/buffered_write_instruction.hpp"
#include "cortext/operations/extraction.hpp"
#include "cortext/processor/operation_context.hpp"
#include <any>
#include <sstream>
#include <string>
#include <vector>

namespace cortext::operations
{

namespace
{

/// @brief Add a buffered write instruction.
void
AddWrite (OperationContext &ctx, const std::string &q,
          const std::vector<std::any> &p = {})
{
  BufferedWriteInstruction op;
  op.query = q;
  op.params = p;
  ctx.AddWriteInstruction (std::move (op));
}

/// @brief Generate a deterministic node_id from entity name and type.
std::string
GenerateNodeId (const std::string &name, const std::string &type)
{
  std::ostringstream ss;
  ss << type << "_" << name;
  return ss.str ();
}

} // namespace

void
ProcessExtractionResults::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();

  if (p_ctx.pending_extraction_results.empty ())
    {
      return;
    }

  for (const auto &result : p_ctx.pending_extraction_results)
    {
      // 1. Insert entities into extraction_entities.
      for (const auto &entity : result.entities)
        {
          AddWrite (context,
                    "INSERT OR REPLACE INTO extraction_entities"
                    "(summary_id, name, type, salience) "
                    "VALUES(?, ?, ?, ?)",
                    { result.summary_id, entity.name, entity.type,
                      entity.salience });

          // 2. Update entity_index for name→node_id mapping.
          std::string node_id = GenerateNodeId (entity.name, entity.type);
          AddWrite (context,
                    "INSERT OR IGNORE INTO entity_index(name, node_id) "
                    "VALUES(?, ?)",
                    { entity.name, node_id });
        }

      // 3. Insert relations into extraction_relations.
      for (const auto &relation : result.relations)
        {
          AddWrite (context,
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
