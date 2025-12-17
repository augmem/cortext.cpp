#include <any>
#include <catch2/catch_test_macros.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;

struct InsertOp : IOperation
{
  void
  Execute (OperationContext & /*ctx*/, Transaction &tx) const override
  {
    tx.Execute ("INSERT INTO t(v) VALUES(?)", { std::string{ "hello" } });
  }
};

TEST_CASE ("SignalProcessor processes and flushes to SQLite", "[processor]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  // Create table upfront
  store->Execute (
      "CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT);");

  SignalProcessor::Config cfg;
  auto pipeline
      = std::make_unique<OperationSet> (std::make_unique<InsertOp> ());
  SignalProcessor proc (cfg, store, std::move (pipeline));

  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  s.timestamp = 0;
  s.source_id = "test";

  proc.Process (s);
  proc.Flush ();

  auto rows = store->Execute ("SELECT COUNT(*) AS c FROM t;");
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 1LL);
}
