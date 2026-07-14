#include <any>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include "../src/operations/historical_surface_search_cache_internal.hpp"
#include <thread>

using namespace cortext;

namespace cortext::testing
{
std::size_t ProcessorRollbackSnapshotOwnerCountForTest ();
std::size_t ProcessorRollbackSnapshotReuseCountForTest ();
}

struct InsertOp : IOperation
{
  void
  Execute (OperationContext & /*ctx*/, Transaction &tx) const override
  {
    tx.Execute ("INSERT INTO t(v) VALUES(?)", { std::string{ "hello" } });
  }
};

struct InsertDeferredViolationOp : IOperation
{
  void
  Execute (OperationContext & /*ctx*/, Transaction &tx) const override
  {
    tx.Execute ("INSERT INTO deferred_child(parent_id) VALUES(?)", { 999LL });
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

struct ExerciseRollbackCacheRestoreOp : IOperation
{
  ExerciseRollbackCacheRestoreOp (bool *retrieval_restored,
                                  bool *volatile_restored)
      : retrieval_restored (retrieval_restored),
        volatile_restored (volatile_restored)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &processor_context = ctx.GetProcessorContext ();
    ++call_count;
    if (call_count == 1)
      {
        ProcessorContext::RetrievalSurfaceEntry cache_only_entry;
        cache_only_entry.memory_id = 77LL;
        cache_only_entry.embedding_id = 88LL;
        cache_only_entry.created_at = 1500LL;
        cache_only_entry.start_ts = 1500LL;
        cache_only_entry.kind = "LONG_TERM";
        cache_only_entry.source_id = "cache/only";
        cache_only_entry.modality = "text";
        cache_only_entry.embedding = Eigen::VectorXf::Constant (256, 0.25f);
        processor_context.UpsertRetrievalSurface (
            std::move (cache_only_entry));
        processor_context.index_store["stable"] = { 11LL, 12LL };
        processor_context.index_reverse[11LL] = "stable";
        processor_context.procedural_store["stable"][11LL] = 0.75;
        return;
      }
    if (call_count == 2)
      {
        processor_context.retrieval_surface_cache.clear ();
        processor_context.retrieval_surface_index.clear ();
        processor_context.retrieval_surface_embedding_index.clear ();
        processor_context.index_store["stable"] = { 99LL };
        processor_context.index_reverse[11LL] = "mutated";
        processor_context.procedural_store["stable"][11LL] = 0.0;
        throw std::runtime_error ("force rollback");
      }

    if (retrieval_restored)
      {
        *retrieval_restored
            = processor_context.retrieval_surface_index.contains (41LL)
              && processor_context.retrieval_surface_index.contains (77LL)
              && processor_context.retrieval_surface_embedding_index.contains (
                  51LL)
              && processor_context.retrieval_surface_embedding_index.contains (
                  88LL)
              && processor_context.retrieval_surface_cache.size () == 2
              && processor_context.retrieval_surface_cache[0].memory_id == 41LL
              && processor_context.retrieval_surface_cache[1].memory_id == 77LL;
      }
    if (volatile_restored)
      {
        const auto index_it = processor_context.index_store.find ("stable");
        const auto reverse_it = processor_context.index_reverse.find (11LL);
        const auto procedural_it
            = processor_context.procedural_store.find ("stable");
        *volatile_restored
            = index_it != processor_context.index_store.end ()
              && index_it->second == std::vector<long long> { 11LL, 12LL }
              && reverse_it != processor_context.index_reverse.end ()
              && reverse_it->second == "stable"
              && procedural_it != processor_context.procedural_store.end ()
              && procedural_it->second.contains (11LL)
              && procedural_it->second.at (11LL) == 0.75;
      }
  }

  bool *retrieval_restored;
  bool *volatile_restored;
  mutable int call_count = 0;
};

struct ExerciseWholeContextRollbackOp : IOperation
{
  explicit ExerciseWholeContextRollbackOp (bool *restored)
      : restored (restored)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &processor_context = ctx.GetProcessorContext ();
    ++call_count;
    if (call_count == 1)
      {
        AccumulatorState foreign;
        foreign.n_signals = 7;
        foreign.mu_acc = Eigen::VectorXf::Constant (256, 0.125f);
        foreign.primary_modality = "audio";
        processor_context.accumulator_states["foreign/source"]
            = std::move (foreign);
        processor_context.recent_scores = { 0.25, 0.75 };
        processor_context.soft_anchor_next_id = 17;
        return;
      }
    if (call_count == 2)
      {
        processor_context.accumulator_states.erase ("foreign/source");
        processor_context.accumulator_states["unexpected/source"].n_signals
            = 99;
        processor_context.recent_scores.clear ();
        processor_context.soft_anchor_next_id = 99;
        throw std::runtime_error ("force whole-context rollback");
      }

    const auto accumulator_it
        = processor_context.accumulator_states.find ("foreign/source");
    if (restored)
      {
        *restored
            = accumulator_it != processor_context.accumulator_states.end ()
              && accumulator_it->second.n_signals == 7
              && accumulator_it->second.mu_acc.size () == 256
              && accumulator_it->second.mu_acc[0] == 0.125f
              && accumulator_it->second.primary_modality == "audio"
              && !processor_context.accumulator_states.contains (
                  "unexpected/source")
              && processor_context.recent_scores
                     == std::deque<double> { 0.25, 0.75 }
              && processor_context.soft_anchor_next_id == 17;
      }
  }

  bool *restored;
  mutable int call_count = 0;
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

TEST_CASE ("SignalProcessor restores exact durable and cache-only state after "
           "rollback",
           "[processor][rollback][cache]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  cortext::testing::InitializeCoreSchema (*store);
  cortext::testing::SeedEmbeddingV2 (
      *store, 51LL, std::vector<float> (256, 0.125f), 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 41LL, 51LL, "seed/source",
                                  "LONG_TERM", 1.0, 1000LL);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  bool retrieval_restored = false;
  bool volatile_restored = false;
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<ExerciseRollbackCacheRestoreOp> (
          &retrieval_restored, &volatile_restored));
  SignalProcessor proc (cfg, store, std::move (pipeline));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 2000;
  proc.Process (signal);

  signal.timestamp = 3000;
  REQUIRE_THROWS (proc.Process (signal));

  signal.timestamp = 4000;
  proc.Process (signal);
  REQUIRE (retrieval_restored);
  REQUIRE (volatile_restored);
}

TEST_CASE ("SignalProcessor reuses lifecycle-owned exact rollback storage",
           "[processor][rollback][lifecycle]")
{
  const std::size_t baseline_owner_count
      = cortext::testing::ProcessorRollbackSnapshotOwnerCountForTest ();
  const std::size_t baseline_reuse_count
      = cortext::testing::ProcessorRollbackSnapshotReuseCountForTest ();
  bool restored = false;
  {
    auto uniq = SQLiteStore::Create (":memory:");
    std::shared_ptr<Store> store (std::move (uniq));
    cortext::testing::InitializeCoreSchema (*store);

    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    auto pipeline = std::make_unique<DynamicOperationSet> (
        std::make_unique<ExerciseWholeContextRollbackOp> (&restored));
    SignalProcessor proc (cfg, store, std::move (pipeline));

    Signal signal;
    signal.embedding = Eigen::VectorXf::Zero (256);
    signal.source_id = "test";
    signal.timestamp = 1000;
    proc.Process (signal);
    REQUIRE (
        cortext::testing::ProcessorRollbackSnapshotOwnerCountForTest ()
        == baseline_owner_count + 1);

    signal.timestamp = 2000;
    REQUIRE_THROWS (proc.Process (signal));

    signal.timestamp = 3000;
    proc.Process (signal);
    REQUIRE (restored);
    REQUIRE (
        cortext::testing::ProcessorRollbackSnapshotReuseCountForTest ()
        >= baseline_reuse_count + 1);
  }
  REQUIRE (cortext::testing::ProcessorRollbackSnapshotOwnerCountForTest ()
           == baseline_owner_count);
}

TEST_CASE ("SignalProcessor erases historical cache when destroyed on another "
           "thread",
           "[processor][cache][lifecycle]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  const std::size_t baseline_size = cache::RegistrySizeForTest ();
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  cortext::testing::InitializeCoreSchema (*store);
  cortext::testing::SeedEmbeddingV2 (
      *store, 51LL, std::vector<float> (256, 0.125f), 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 41LL, 51LL, "seed/source",
                                  "LONG_TERM", 1.0, 1000LL);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<RecordOrderOp> (nullptr, 0));
  auto processor = std::make_unique<SignalProcessor> (
      cfg, store, std::move (pipeline));
  REQUIRE (cache::RegistrySizeForTest () == baseline_size + 1);

  std::thread teardown (
      [owner = std::move (processor)] () mutable { owner.reset (); });
  teardown.join ();
  REQUIRE (cache::RegistrySizeForTest () == baseline_size);
}

TEST_CASE ("SignalProcessor rebuilds historical cache after force commit",
           "[processor][cache][consolidation][lifecycle]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  const std::size_t baseline_size = cache::RegistrySizeForTest ();
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  cortext::testing::InitializeCoreSchema (*store);
  cortext::testing::SeedEmbeddingV2 (
      *store, 51LL, std::vector<float> (256, 0.125f), 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 41LL, 51LL, "seed/source",
                                  "LONG_TERM", 1.0, 1000LL);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<RecordOrderOp> (nullptr, 0));
  SignalProcessor processor (cfg, store, std::move (pipeline));
  REQUIRE (cache::RegistrySizeForTest () == baseline_size + 1);

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 2000;
  signal.force_consolidation = true;
  processor.Process (signal);
  REQUIRE (cache::RegistrySizeForTest () == baseline_size + 1);

  signal.timestamp = 3000;
  signal.force_consolidation = false;
  processor.Process (signal);
  REQUIRE (cache::RegistrySizeForTest () == baseline_size + 1);
}

TEST_CASE ("SignalProcessor restores historical cache after force commit "
           "failure",
           "[processor][rollback][cache][consolidation]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  const std::size_t baseline_size = cache::RegistrySizeForTest ();
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  cortext::testing::InitializeCoreSchema (*store);
  cortext::testing::SeedEmbeddingV2 (
      *store, 51LL, std::vector<float> (256, 0.125f), 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 41LL, 51LL, "seed/source",
                                  "LONG_TERM", 1.0, 1000LL);
  store->Execute ("CREATE TABLE deferred_parent(id INTEGER PRIMARY KEY)", {});
  store->Execute (
      "CREATE TABLE deferred_child("
      "parent_id INTEGER REFERENCES deferred_parent(id) "
      "DEFERRABLE INITIALLY DEFERRED)",
      {});

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<InsertDeferredViolationOp> ());
  SignalProcessor processor (cfg, store, std::move (pipeline));
  REQUIRE (cache::RegistrySizeForTest () == baseline_size + 1);

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 2000;
  signal.force_consolidation = true;
  REQUIRE_THROWS (processor.Process (signal));
  REQUIRE (cache::RegistrySizeForTest () == baseline_size + 1);
  const auto child_rows
      = store->Execute ("SELECT COUNT(*) AS c FROM deferred_child", {});
  REQUIRE (child_rows.size () == 1);
  REQUIRE (std::any_cast<long long> (child_rows[0].at ("c")) == 0LL);
}
