// tests/operations_graph_schema.test.cpp
#include <any>
#include <catch2/catch_test_macros.hpp>

#include <cortext/operations/graph_schema.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;
using cortext::operations::EnsureGraphSchema;

namespace
{
static Signal
MakeSignal ()
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  s.timestamp = 1;
  s.source_id = "test";
  return s;
}
} // namespace

TEST_CASE ("EnsureGraphSchema creates required tables", "[operations][graph]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  auto ops = std::make_unique<OperationSet> (std::make_unique<EnsureGraphSchema> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal ());
  processor.Flush ();

  auto rows = store->Execute (
      "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", {});
  REQUIRE (!rows.empty ());

  auto has = [&rows] (const std::string &name) {
    for (const auto &r : rows)
      {
        auto it = r.find ("name");
        if (it != r.end () && it->second.type () == typeid (std::string))
          {
            if (std::any_cast<std::string> (it->second) == name)
              return true;
          }
      }
    return false;
  };

  REQUIRE (has ("graph_nodes"));
  REQUIRE (has ("graph_edges"));
  REQUIRE (has ("entity_index"));
  REQUIRE (has ("goal_nodes"));
  REQUIRE (has ("consolidation_summaries"));
  REQUIRE (has ("consolidation_sources"));
  REQUIRE (has ("extraction_entities"));
  REQUIRE (has ("extraction_relations"));
}

