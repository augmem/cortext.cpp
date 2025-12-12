// tests/operations_graph_build.test.cpp
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cortext/operations/graph_build.hpp>
#include <cortext/operations/graph_schema.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;
using cortext::operations::BuildGraphFromConsolidation;
using cortext::operations::EnsureGraphSchema;

namespace
{
static Signal
MakeSignal (uint64_t ts)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}
} // namespace

TEST_CASE ("Alg30 builds nodes and edges from extraction outputs",
           "[operations][graph][alg30]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Seed input tables and rows (external worker outputs).
  store->Execute ("CREATE TABLE IF NOT EXISTS consolidation_summaries ("
                  "  summary_id TEXT PRIMARY KEY,"
                  "  summary_text TEXT,"
                  "  cluster_size INTEGER"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS consolidation_sources ("
                  "  summary_id TEXT,"
                  "  source_text TEXT,"
                  "  source_embedding_id INTEGER"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS extraction_entities ("
                  "  summary_id TEXT NOT NULL,"
                  "  name TEXT NOT NULL,"
                  "  type TEXT NOT NULL,"
                  "  salience REAL,"
                  "  embedding_id INTEGER,"
                  "  PRIMARY KEY (summary_id, name, type)"
                  ");");
  store->Execute ("CREATE TABLE IF NOT EXISTS extraction_relations ("
                  "  summary_id TEXT NOT NULL,"
                  "  subject TEXT NOT NULL,"
                  "  predicate TEXT NOT NULL,"
                  "  object TEXT NOT NULL,"
                  "  confidence REAL"
                  ");");

  store->Execute ("INSERT INTO consolidation_summaries(summary_id, summary_text, "
                  "cluster_size) VALUES (?,?,?)",
                  { std::string ("s1"), std::string ("summary one"), 10LL });
  store->Execute ("INSERT INTO consolidation_sources(summary_id, source_text, "
                  "source_embedding_id) VALUES (?,?,?)",
                  { std::string ("s1"), std::string ("source"), 42LL });
  store->Execute ("INSERT INTO extraction_entities(summary_id, name, type, "
                  "salience, embedding_id) VALUES (?,?,?,?,?)",
                  { std::string ("s1"), std::string ("Alice"),
                    std::string ("Person"), 0.7, 100LL });
  store->Execute ("INSERT INTO extraction_relations(summary_id, subject, predicate, "
                  "object, confidence) VALUES (?,?,?,?,?)",
                  { std::string ("s1"), std::string ("Alice"),
                    std::string ("KNOWS"), std::string ("Bob"), 0.8 });

  SignalProcessor::Config cfg;
  auto ops = std::make_unique<OperationSet> (
      std::make_unique<EnsureGraphSchema> (),
      std::make_unique<BuildGraphFromConsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (1234ULL));
  processor.Flush ();

  // Nodes
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS c FROM graph_nodes WHERE node_id IN "
        "('summary:s1','entity:Alice','entity:Bob','emb:42')",
        {});
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 4LL);
  }

  // mentions edge
  {
    auto rows = store->Execute (
        "SELECT weight FROM graph_edges "
        "WHERE source_id='summary:s1' AND target_id='entity:Alice' "
        "AND edge_type='mentions'",
        {});
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<double> (rows[0].at ("weight"))
             == Catch::Approx (0.7).margin (1e-6));
  }

  // relation edge
  {
    auto rows = store->Execute (
        "SELECT weight FROM graph_edges "
        "WHERE source_id='entity:Alice' AND target_id='entity:Bob' "
        "AND edge_type='KNOWS'",
        {});
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<double> (rows[0].at ("weight"))
             == Catch::Approx (0.8).margin (1e-6));
  }

  // derived_from edge
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS c FROM graph_edges "
        "WHERE source_id='summary:s1' AND target_id='emb:42' "
        "AND edge_type='derived_from'",
        {});
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 1LL);
  }

  // entity_index mapping
  {
    auto rows = store->Execute (
        "SELECT node_id FROM entity_index WHERE name='Alice'", {});
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<std::string> (rows[0].at ("node_id"))
             == "entity:Alice");
  }
}

