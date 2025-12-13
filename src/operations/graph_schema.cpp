#include "cortext/operations/graph_schema.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/schema.hpp"

namespace cortext::operations
{

void
EnsureGraphSchema::Execute (OperationContext &context) const
{
  // No-op: schema is now managed via CollectSchema().
  (void)context;
}

void
EnsureGraphSchema::CollectSchema (cortext::store::SchemaRegistry &registry) const
{
  registry.Register ({
      10, // ID allocated for Graph tables
      "Graph schema (nodes, edges, indexes, extraction)",
      {
          // Graph storage
          "CREATE TABLE IF NOT EXISTS graph_nodes ("
          "  node_id TEXT PRIMARY KEY,"
          "  type TEXT NOT NULL,"
          "  label TEXT,"
          "  embedding_id INTEGER,"
          "  metadata TEXT,"
          "  created_at INTEGER"
          ")",

          "CREATE TABLE IF NOT EXISTS graph_edges ("
          "  source_id TEXT NOT NULL,"
          "  target_id TEXT NOT NULL,"
          "  edge_type TEXT NOT NULL,"
          "  weight REAL NOT NULL DEFAULT 1.0,"
          "  decay_rate REAL,"
          "  last_reinforced INTEGER,"
          "  PRIMARY KEY (source_id, target_id, edge_type)"
          ")",

          "CREATE INDEX IF NOT EXISTS idx_graph_edges_source "
          "ON graph_edges(source_id)",
          "CREATE INDEX IF NOT EXISTS idx_graph_edges_target "
          "ON graph_edges(target_id)",
          "CREATE INDEX IF NOT EXISTS idx_graph_edges_type "
          "ON graph_edges(edge_type)",

          // Name→node mapping
          "CREATE TABLE IF NOT EXISTS entity_index ("
          "  name TEXT PRIMARY KEY,"
          "  node_id TEXT NOT NULL"
          ")",

          // Goal nodes
          "CREATE TABLE IF NOT EXISTS goal_nodes ("
          "  node_id TEXT PRIMARY KEY"
          ")",

          // Consolidation tables
          "CREATE TABLE IF NOT EXISTS consolidation_summaries ("
          "  summary_id TEXT PRIMARY KEY,"
          "  summary_text TEXT,"
          "  cluster_size INTEGER"
          ")",
          "CREATE TABLE IF NOT EXISTS consolidation_sources ("
          "  summary_id TEXT,"
          "  source_text TEXT,"
          "  source_embedding_id INTEGER"
          ")",

          // Extraction tables
          "CREATE TABLE IF NOT EXISTS extraction_entities ("
          "  summary_id TEXT NOT NULL,"
          "  name TEXT NOT NULL,"
          "  type TEXT NOT NULL,"
          "  salience REAL,"
          "  embedding_id INTEGER,"
          "  PRIMARY KEY (summary_id, name, type)"
          ")",

          "CREATE TABLE IF NOT EXISTS extraction_relations ("
          "  summary_id TEXT NOT NULL,"
          "  subject TEXT NOT NULL,"
          "  predicate TEXT NOT NULL,"
          "  object TEXT NOT NULL,"
          "  confidence REAL"
          ")",
      },
  });
}

} // namespace cortext::operations
