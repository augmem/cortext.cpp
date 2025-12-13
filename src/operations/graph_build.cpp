#include "cortext/operations/graph_build.hpp"

#include "cortext/buffered_write_instruction.hpp"
#include "cortext/processor/operation_context.hpp"
#include <string>

namespace cortext::operations
{

namespace
{
/// @brief Adds a buffered write instruction to the context.
void
Add (OperationContext &ctx, const std::string &q,
     const std::vector<std::any> &p = {})
{
  BufferedWriteInstruction op;
  op.query = q;
  op.params = p;
  ctx.AddWriteInstruction (std::move (op));
}

} // namespace

void
BuildGraphFromConsolidation::Execute (OperationContext &context) const
{
  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);

  // Ensure required tables exist. (Safe to emit repeatedly.)
  Add (context,
       "CREATE TABLE IF NOT EXISTS graph_nodes ("
       "  node_id TEXT PRIMARY KEY,"
       "  type TEXT NOT NULL,"
       "  label TEXT,"
       "  embedding_id INTEGER,"
       "  metadata TEXT,"
       "  created_at INTEGER"
       ");");
  Add (context,
       "CREATE TABLE IF NOT EXISTS graph_edges ("
       "  source_id TEXT NOT NULL,"
       "  target_id TEXT NOT NULL,"
       "  edge_type TEXT NOT NULL,"
       "  weight REAL NOT NULL DEFAULT 1.0,"
       "  decay_rate REAL,"
       "  last_reinforced INTEGER,"
       "  PRIMARY KEY (source_id, target_id, edge_type)"
       ");");
  Add (context,
       "CREATE TABLE IF NOT EXISTS entity_index ("
       "  name TEXT PRIMARY KEY,"
       "  node_id TEXT NOT NULL"
       ");");

  // 1) Create nodes for all summaries.
  //    Node id: summary:<summary_id>
  Add (context,
       "INSERT OR IGNORE INTO graph_nodes(node_id, type, label, created_at) "
       "SELECT 'summary:' || summary_id, 'summary', summary_id, ?1 "
       "FROM consolidation_summaries;",
       { now_ts });

  // 2) Ensure entity nodes exist for extraction_entities.
  Add (context,
       "INSERT OR IGNORE INTO graph_nodes(node_id, type, label, embedding_id, "
       "created_at) "
       "SELECT 'entity:' || name, 'entity', name, embedding_id, ?1 "
       "FROM extraction_entities;",
       { now_ts });

  // 2b) Maintain mapping table (name -> node_id).
  Add (context,
       "INSERT OR IGNORE INTO entity_index(name, node_id) "
       "SELECT name, 'entity:' || name FROM extraction_entities;");

  // 3) Mentions edges: summary -> entity, weight=salience
  Add (context,
       "INSERT OR REPLACE INTO graph_edges(source_id, target_id, edge_type, weight, "
       "last_reinforced) "
       "SELECT 'summary:' || e.summary_id, 'entity:' || e.name, 'mentions', "
       "COALESCE(e.salience, 1.0), ?1 "
       "FROM extraction_entities e;",
       { now_ts });

  // 4) Relation edges: entity(subject) -> entity(object), edge_type=predicate
  // Ensure both endpoints exist.
  Add (context,
       "INSERT OR IGNORE INTO graph_nodes(node_id, type, label, created_at) "
       "SELECT 'entity:' || subject, 'entity', subject, ?1 "
       "FROM extraction_relations;",
       { now_ts });
  Add (context,
       "INSERT OR IGNORE INTO graph_nodes(node_id, type, label, created_at) "
       "SELECT 'entity:' || object, 'entity', object, ?1 "
       "FROM extraction_relations;",
       { now_ts });

  Add (context,
       "INSERT OR REPLACE INTO graph_edges(source_id, target_id, edge_type, weight, "
       "last_reinforced) "
       "SELECT 'entity:' || subject, 'entity:' || object, predicate, "
       "COALESCE(confidence, 1.0), ?1 "
       "FROM extraction_relations;",
       { now_ts });

  // 5) Derived-from edges: summary -> embedded memory id (if available).
  Add (context,
       "INSERT OR IGNORE INTO graph_nodes(node_id, type, label, embedding_id, "
       "created_at) "
       "SELECT 'emb:' || source_embedding_id, 'memory', NULL, source_embedding_id, "
       "?1 "
       "FROM consolidation_sources "
       "WHERE source_embedding_id IS NOT NULL;",
       { now_ts });

  Add (context,
       "INSERT OR REPLACE INTO graph_edges(source_id, target_id, edge_type, weight, "
       "last_reinforced) "
       "SELECT 'summary:' || summary_id, 'emb:' || source_embedding_id, "
       "'derived_from', 1.0, ?1 "
       "FROM consolidation_sources "
       "WHERE source_embedding_id IS NOT NULL;",
       { now_ts });
}

} // namespace cortext::operations
