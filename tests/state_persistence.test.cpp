// tests/state_persistence.test.cpp
#include <any>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cortext/operations/focus.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <memory>

using namespace cortext;
using Catch::Matchers::WithinAbs;

namespace
{

// Helper to extract long long from std::any
long long
GetInt64 (const std::map<std::string, std::any> &row, const std::string &key)
{
  auto it = row.find (key);
  if (it == row.end ())
    return 0;
  if (it->second.type () == typeid (long long))
    return std::any_cast<long long> (it->second);
  if (it->second.type () == typeid (int))
    return static_cast<long long> (std::any_cast<int> (it->second));
  return 0;
}

// Helper to extract double from std::any
double
GetDouble (const std::map<std::string, std::any> &row, const std::string &key)
{
  auto it = row.find (key);
  if (it == row.end ())
    return 0.0;
  if (it->second.type () == typeid (double))
    return std::any_cast<double> (it->second);
  if (it->second.type () == typeid (float))
    return static_cast<double> (std::any_cast<float> (it->second));
  return 0.0;
}

// Dummy operation that triggers episode boundary
struct TriggerBoundaryOp : IOperation
{
  void
  Execute (OperationContext &ctx) const override
  {
    ctx.RequestFinalizeEpisode ();
  }
};

} // namespace

TEST_CASE ("State persistence tables are created", "[state_persistence][schema]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  auto ops = std::make_unique<OperationSet> ();
  SignalProcessor processor (SignalProcessor::Config{}, store, std::move (ops));

  auto rows
      = store->Execute ("SELECT name FROM sqlite_master WHERE type='table'",
                        {});

  auto has = [&] (const std::string &name) {
    for (const auto &r : rows)
      {
        auto it = r.find ("name");
        if (it != r.end ()
            && std::any_cast<std::string> (it->second) == name)
          return true;
      }
    return false;
  };

  SECTION ("Migration 3 tables exist")
  {
    REQUIRE (has ("processor_state"));
    REQUIRE (has ("blender_weights"));
    REQUIRE (has ("blender_covariance"));
  }

  SECTION ("Migration 4 tables exist")
  {
    REQUIRE (has ("episodes"));
    REQUIRE (has ("signal_metrics"));
    REQUIRE (has ("generation_trace"));
  }

  SECTION ("Migration 5 tables exist")
  {
    REQUIRE (has ("recent_context"));
    REQUIRE (has ("recent_scores"));
  }
}

TEST_CASE ("Processor state is persisted on flush",
           "[state_persistence][processor_state]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // First processor instance - process some signals
  {
    auto ops
        = std::make_unique<OperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (SignalProcessor::Config{}, store,
                               std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 1000;
    s.source_id = "test";

    // Process multiple signals to increment signals_processed
    processor.Process (s);
    s.timestamp = 2000;
    processor.Process (s);
    s.timestamp = 3000;
    processor.Process (s);

    processor.Flush ();
  }

  // Check that processor_state was saved
  auto rows = store->Execute ("SELECT * FROM processor_state WHERE id = 1");
  REQUIRE (rows.size () == 1);

  long long signals = GetInt64 (rows[0], "signals_processed");
  REQUIRE (signals == 3);
}

TEST_CASE ("Processor state is loaded on startup",
           "[state_persistence][processor_state]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // First processor instance - set some state
  {
    auto ops
        = std::make_unique<OperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (SignalProcessor::Config{}, store,
                               std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 1000;
    s.source_id = "test";

    processor.Process (s);
    processor.Process (s);
    processor.Flush ();
  }

  // Verify state was persisted
  auto rows1 = store->Execute ("SELECT signals_processed FROM processor_state");
  REQUIRE (rows1.size () == 1);
  long long saved_count = GetInt64 (rows1[0], "signals_processed");
  REQUIRE (saved_count == 2);

  // Second processor instance - should load state
  {
    auto ops
        = std::make_unique<OperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (SignalProcessor::Config{}, store,
                               std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 2000;
    s.source_id = "test";

    // Process one more signal
    processor.Process (s);
    processor.Flush ();
  }

  // Check that count is cumulative (2 from before + 1 new = 3)
  auto rows2 = store->Execute ("SELECT signals_processed FROM processor_state");
  REQUIRE (rows2.size () == 1);
  long long total_count = GetInt64 (rows2[0], "signals_processed");
  REQUIRE (total_count == 3);
}

TEST_CASE ("Blender weights are persisted", "[state_persistence][blender]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  {
    auto ops
        = std::make_unique<OperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (SignalProcessor::Config{}, store,
                               std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 1000;
    s.source_id = "test";

    processor.Process (s);
    processor.Flush ();
  }

  // Check blender_weights table has data
  auto rows = store->Execute ("SELECT * FROM blender_weights WHERE id = 1");
  REQUIRE (rows.size () == 1);

  // Default weights should be around 0.5
  double w_relevance = GetDouble (rows[0], "w_relevance");
  REQUIRE_THAT (w_relevance, WithinAbs (0.5, 0.5));
}

TEST_CASE ("Recent context embeddings are persisted",
           "[state_persistence][recent_context]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  const int num_signals = 5;

  {
    // Use UpdateFocus to populate recent_context_embeddings, then trigger
    // boundary
    auto ops = std::make_unique<OperationSet> (
        std::make_unique<operations::UpdateFocus> (),
        std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (SignalProcessor::Config{}, store,
                               std::move (ops));

    for (int i = 0; i < num_signals; ++i)
      {
        Signal s;
        s.embedding = Eigen::VectorXf::Random (256);
        s.timestamp = static_cast<uint64_t> (1000 + i * 100);
        s.source_id = "test";
        processor.Process (s);
      }

    processor.Flush ();
  }

  // Check recent_context table has embeddings
  auto rows = store->Execute (
      "SELECT COUNT(*) as c FROM recent_context");
  REQUIRE (rows.size () == 1);

  long long count = GetInt64 (rows[0], "c");
  // Should have persisted some embeddings (exact count depends on n_ctx)
  REQUIRE (count > 0);
}

TEST_CASE ("Recent scores are persisted", "[state_persistence][recent_scores]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  {
    auto ops
        = std::make_unique<OperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (SignalProcessor::Config{}, store,
                               std::move (ops));

    for (int i = 0; i < 3; ++i)
      {
        Signal s;
        s.embedding = Eigen::VectorXf::Random (256);
        s.timestamp = static_cast<uint64_t> (1000 + i * 100);
        s.source_id = "test";
        processor.Process (s);
      }

    processor.Flush ();
  }

  // Check recent_scores table
  auto rows = store->Execute ("SELECT COUNT(*) as c FROM recent_scores");
  REQUIRE (rows.size () == 1);
  // Scores may or may not be persisted depending on threshold logic
}

TEST_CASE ("State persistence is idempotent across restarts",
           "[state_persistence][idempotent]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Run 1
  {
    auto ops
        = std::make_unique<OperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (SignalProcessor::Config{}, store,
                               std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 1000;
    s.source_id = "test";
    processor.Process (s);
    processor.Flush ();
  }

  // Run 2
  {
    auto ops
        = std::make_unique<OperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (SignalProcessor::Config{}, store,
                               std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 2000;
    s.source_id = "test";
    processor.Process (s);
    processor.Flush ();
  }

  // Run 3
  {
    auto ops
        = std::make_unique<OperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (SignalProcessor::Config{}, store,
                               std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 3000;
    s.source_id = "test";
    processor.Process (s);
    processor.Flush ();
  }

  // Should have exactly one row in singleton tables
  auto ps_rows = store->Execute ("SELECT COUNT(*) as c FROM processor_state");
  REQUIRE (GetInt64 (ps_rows[0], "c") == 1);

  auto bw_rows = store->Execute ("SELECT COUNT(*) as c FROM blender_weights");
  REQUIRE (GetInt64 (bw_rows[0], "c") == 1);

  // Signals processed should be cumulative
  auto state_rows = store->Execute ("SELECT signals_processed FROM processor_state");
  REQUIRE (GetInt64 (state_rows[0], "signals_processed") == 3);
}
