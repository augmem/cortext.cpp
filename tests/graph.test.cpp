// tests/graph.test.cpp
#include <any>
#include <catch2/catch_test_macros.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <string>

static inline std::string
j (const char *s)
{
  return std::string (s);
}

TEST_CASE ("Graph basic SQL functions", "[graph]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  store->Execute ("CREATE VIRTUAL TABLE graph USING graph()", {});

  // Initial counts
  auto nodes0 = store->Execute ("SELECT graph_count_nodes() AS c", {});
  REQUIRE (nodes0.size () == 1);
  REQUIRE (std::any_cast<long long> (nodes0[0].at ("c")) == 0LL);
  auto edges0 = store->Execute ("SELECT graph_count_edges() AS c", {});
  REQUIRE (edges0.size () == 1);
  REQUIRE (std::any_cast<long long> (edges0[0].at ("c")) == 0LL);

  // Add nodes
  store->Execute ("SELECT graph_node_add(?, ?) AS id",
                  { 1LL, j ("{\"name\":\"Alice\",\"age\":30}") });
  store->Execute ("SELECT graph_node_add(?, ?) AS id",
                  { 2LL, j ("{\"name\":\"Bob\",\"age\":25}") });

  // Add edge
  store->Execute ("SELECT graph_edge_add(?, ?, ?, ?) AS id",
                  { 1LL, 2LL, j ("KNOWS"), j ("{\"since\":2020}") });

  // Verify counts
  auto nodes = store->Execute ("SELECT graph_count_nodes() AS c", {});
  REQUIRE (nodes.size () == 1);
  REQUIRE (std::any_cast<long long> (nodes[0].at ("c")) == 2LL);
  auto edges = store->Execute ("SELECT graph_count_edges() AS c", {});
  REQUIRE (edges.size () == 1);
  REQUIRE (std::any_cast<long long> (edges[0].at ("c")) == 1LL);
}

TEST_CASE ("Graph Cypher create and match", "[graph]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  store->Execute ("CREATE VIRTUAL TABLE graph USING graph()", {});

  // Create nodes and relationship with Cypher
  store->Execute (
      "SELECT cypher_execute('CREATE (p:Person {name: \"Alice\", age: 30})')",
      {});
  store->Execute (
      "SELECT cypher_execute('CREATE (c:Company {name: \"Acme\"})')", {});
  store->Execute ("SELECT cypher_execute('CREATE (a:Person {name:\"Bob\"})-[:"
                  "FRIENDS {since:2021}]->(b:Person {name:\"Carol\"})')",
                  {});

  // MATCH with label and WHERE
  auto r1 = store->Execute ("SELECT cypher_execute('MATCH (p:Person) WHERE "
                            "p.age > 25 RETURN p') AS r",
                            {});
  REQUIRE (r1.size () == 1);
  REQUIRE (r1[0].count ("r") == 1);
  REQUIRE (std::any_cast<std::string> (r1[0].at ("r")).size () > 0);

  // Relationship pattern
  auto r2 = store->Execute (
      "SELECT cypher_execute('MATCH (a)-[r:FRIENDS]->(b) RETURN a,r,b') AS r",
      {});
  REQUIRE (r2.size () == 1);
  REQUIRE (r2[0].count ("r") == 1);
  REQUIRE (std::any_cast<std::string> (r2[0].at ("r")).size () > 0);
}
