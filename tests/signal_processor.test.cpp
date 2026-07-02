#include <any>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
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

struct RecordOrderOp : IOperation
{
  explicit RecordOrderOp (std::vector<int> *order, int id)
      : order (order), id (id)
  {
  }

  void
  Execute (OperationContext & /*ctx*/, Transaction & /*tx*/) const override
  {
    if (order)
      {
        order->push_back (id);
      }
  }

  std::vector<int> *order;
  int id = 0;
};

struct CaptureAccumulatorCountOp : IOperation
{
  explicit CaptureAccumulatorCountOp (std::size_t *count) : count (count) {}

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    if (count)
      {
        *count = ctx.GetProcessorContext ().accumulator_states.size ();
      }
  }

  std::size_t *count;
};

TEST_CASE ("SignalProcessor processes and flushes to SQLite", "[processor]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  // Create table upfront
  store->Execute (
      "CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT);");

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  auto pipeline
      = std::make_unique<DynamicOperationSet> (std::make_unique<InsertOp> ());
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

TEST_CASE ("SignalProcessor executes pipeline in order", "[processor][order]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  std::vector<int> order;
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<RecordOrderOp> (&order, 1),
      std::make_unique<RecordOrderOp> (&order, 2),
      std::make_unique<RecordOrderOp> (&order, 3));

  SignalProcessor proc (cfg, store, std::move (pipeline));

  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  s.timestamp = 0;
  s.source_id = "test";

  proc.Process (s);

  REQUIRE (order == std::vector<int> { 1, 2, 3 });
}

TEST_CASE ("SignalProcessor treats persisted accumulators as volatile",
           "[processor][accumulator]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  cortext::testing::InitializeCoreSchema (*store);
  store->Execute (
      "INSERT INTO accumulators(source_id, n, t_start, last_signal_ts) "
      "VALUES('stale/source', 3, 1000, 2000)",
      {});

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  std::size_t accumulator_count = 999;
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<CaptureAccumulatorCountOp> (&accumulator_count));

  SignalProcessor proc (cfg, store, std::move (pipeline));

  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  s.timestamp = 3000;
  s.source_id = "test";

  proc.Process (s);

  REQUIRE (accumulator_count == 0);
}
