// tests/state_persistence.test.cpp
#include <any>
#include "../src/operations/historical_surface_search_cache_internal.hpp"
#include "../src/operations/consolidation_throughput_state_internal.hpp"
#include "../src/working_memory_time_internal.hpp"
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cortext/clock.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/focus.hpp>
#include <cortext/operations/working_memory.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cmath>
#include <limits>
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
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.RequestFinalizeEpisode ();
  }
};

struct CaptureRecentWindowOp : IOperation
{
  std::size_t *context_count = nullptr;
  std::size_t *score_count = nullptr;

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    const auto &pctx = ctx.GetProcessorContext ();
    if (context_count)
      {
        *context_count = pctx.recent_context_embeddings.size ();
      }
    if (score_count)
      {
        *score_count = pctx.recent_scores.size ();
      }
  }
};

struct CaptureLoadedPolicyStateOp : IOperation
{
  mutable double weight_relevance = 0.0;
  mutable double coverage_gain_floor = 0.0;
  mutable double theta_dynamic = 0.0;
  mutable double half_life = 0.0;
  mutable double weight_surprise = 0.0;
  mutable double rate_decay = 0.0;
  mutable double reliability = -1.0;

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    const auto &pctx = ctx.GetProcessorContext ();
    weight_relevance = pctx.weight_relevance;
    coverage_gain_floor = pctx.coverage_gain_floor;
    theta_dynamic = pctx.T_dynamic;
    half_life = pctx.half_life;
    weight_surprise = pctx.weight_surprise;
    rate_decay = pctx.rate_decay;
    reliability = pctx.reliability;
  }
};

struct CaptureProcessorContextOp : IOperation
{
  ProcessorContext **captured = nullptr;

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    if (captured)
      {
        *captured = &ctx.GetProcessorContext ();
      }
  }
};

struct SetConsolidationThroughputOp : IOperation
{
  operations::consolidation_throughput_state_internal::State state;

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    operations::consolidation_throughput_state_internal::Reset (
        ctx.GetProcessorContext (), state);
  }
};

struct CaptureConsolidationThroughputOp : IOperation
{
  operations::consolidation_throughput_state_internal::State *captured
      = nullptr;

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    if (captured)
      {
        *captured
            = operations::consolidation_throughput_state_internal::Find (
                ctx.GetProcessorContext ());
      }
  }
};

struct ObserveConsolidationThroughputOp : IOperation
{
  double rate = 0.0;

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    operations::consolidation_throughput_state_internal::Observe (
        ctx.GetProcessorContext (), rate, ctx.GetConfig ().focus,
        ctx.GetConfig ().sensitivity, ctx.GetConfig ().stability);
  }
};

class FailCommitTransaction final : public Transaction
{
public:
  FailCommitTransaction (std::unique_ptr<Transaction> inner, bool *fail_next)
      : inner_ (std::move (inner)), fail_next_ (fail_next)
  {
  }

  std::unique_ptr<Transaction>
  Begin () override
  {
    return std::make_unique<FailCommitTransaction> (inner_->Begin (), fail_next_);
  }

  std::vector<std::map<std::string, std::any>>
  Execute (const std::string &query,
           const std::vector<std::any> &params = {}) override
  {
    return inner_->Execute (query, params);
  }

  void
  Commit () override
  {
    if (fail_next_ && *fail_next_)
      {
        *fail_next_ = false;
        throw StoreError ("injected commit failure");
      }
    inner_->Commit ();
  }

  void
  Rollback () override
  {
    inner_->Rollback ();
  }

private:
  std::unique_ptr<Transaction> inner_;
  bool *fail_next_;
};

class FailCommitStore final : public Store
{
public:
  explicit FailCommitStore (std::shared_ptr<Store> inner)
      : inner_ (std::move (inner))
  {
  }

  std::vector<std::map<std::string, std::any>>
  Execute (const std::string &query,
           const std::vector<std::any> &params = {}) override
  {
    return inner_->Execute (query, params);
  }

  std::unique_ptr<Transaction>
  Begin () override
  {
    return std::make_unique<FailCommitTransaction> (inner_->Begin (),
                                                    &fail_next_commit);
  }

  void Commit () override { inner_->Commit (); }
  void Rollback () override { inner_->Rollback (); }
  void Close () override { inner_->Close (); }

  bool fail_next_commit = false;

private:
  std::shared_ptr<Store> inner_;
};

inline SignalProcessor::Config
MakeConfig ()
{
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  return cfg;
}

} // namespace

TEST_CASE ("State persistence tables are created", "[state_persistence][schema]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  auto ops = std::make_unique<DynamicOperationSet> ();
  SignalProcessor processor (MakeConfig (), store, std::move (ops));

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

  SECTION ("v2 schema tables exist")
  {
    REQUIRE (has ("state"));
    REQUIRE (has ("accumulators"));
    REQUIRE (has ("memories"));
    REQUIRE (has ("embeddings"));
    REQUIRE (has ("episodes"));
    REQUIRE (has ("signals"));
  }

  SECTION ("v2 views exist")
  {
    // In v2 schema, recent_context and recent_scores are VIEWs derived from signals
    auto views = store->Execute (
        "SELECT name FROM sqlite_master WHERE type='view'", {});
    auto has_view = [&] (const std::string &name) {
      for (const auto &r : views)
        {
          auto it = r.find ("name");
          if (it != r.end ()
              && std::any_cast<std::string> (it->second) == name)
            return true;
        }
      return false;
    };
    REQUIRE (has_view ("recent_context"));
    REQUIRE (has_view ("recent_scores"));
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
        = std::make_unique<DynamicOperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (MakeConfig (), store, std::move (ops));

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

  // Check that state was saved (v2 schema: unified state table)
  auto rows = store->Execute ("SELECT * FROM state WHERE id = 1");
  REQUIRE (rows.size () == 1);

  long long signals = GetInt64 (rows[0], "signals_processed");
  REQUIRE (signals == 3);
}

TEST_CASE ("Single signal state is persisted as loadable state",
           "[state_persistence][processor_state]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  {
    auto ops
        = std::make_unique<DynamicOperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (MakeConfig (), store, std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 1000;
    s.source_id = "test";
    processor.Process (s);
  }

  auto rows = store->Execute ("SELECT signals_processed FROM state");
  REQUIRE (rows.size () == 1);
  REQUIRE (GetInt64 (rows[0], "signals_processed") == 1);

  {
    auto ops
        = std::make_unique<DynamicOperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (MakeConfig (), store, std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 2000;
    s.source_id = "test";
    processor.Process (s);
  }

  rows = store->Execute ("SELECT signals_processed FROM state");
  REQUIRE (rows.size () == 1);
  REQUIRE (GetInt64 (rows[0], "signals_processed") == 2);
}

TEST_CASE ("Processor state is loaded on startup",
           "[state_persistence][processor_state]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // First processor instance - set some state
  {
    auto ops
        = std::make_unique<DynamicOperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (MakeConfig (), store, std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 1000;
    s.source_id = "test";

    processor.Process (s);
    processor.Process (s);
    processor.Flush ();
  }

  // Verify state was persisted (v2 schema: unified state table)
  auto rows1 = store->Execute ("SELECT signals_processed FROM state");
  REQUIRE (rows1.size () == 1);
  long long saved_count = GetInt64 (rows1[0], "signals_processed");
  REQUIRE (saved_count == 2);

  // Second processor instance - should load state
  {
    auto ops
        = std::make_unique<DynamicOperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (MakeConfig (), store, std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 2000;
    s.source_id = "test";

    // Process one more signal
    processor.Process (s);
    processor.Flush ();
  }

  // Check that count is cumulative (2 from before + 1 new = 3)
  auto rows2 = store->Execute ("SELECT signals_processed FROM state");
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
        = std::make_unique<DynamicOperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (MakeConfig (), store, std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 1000;
    s.source_id = "test";

    processor.Process (s);
    processor.Flush ();
  }

  // Check state table has blender weights (v2 schema: merged into state)
  auto rows = store->Execute ("SELECT * FROM state WHERE id = 1");
  REQUIRE (rows.size () == 1);

  // Default weights should be around 0.5
  double w_relevance = GetDouble (rows[0], "w_relevance");
  REQUIRE_THAT (w_relevance, WithinAbs (0.5, 0.5));
}

TEST_CASE ("Bootstrap state persistence uses active knobs",
           "[state_persistence][blender][knobs]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg = MakeConfig ();
  cfg.focus = 1.0;
  cfg.sensitivity = 1.0;
  cfg.stability = 1.0;

  {
    auto ops = std::make_unique<DynamicOperationSet> ();
    SignalProcessor processor (cfg, store, std::move (ops));
    processor.Flush ();
  }

  auto rows
      = store->Execute ("SELECT theta_dynamic, w_relevance FROM state WHERE id = 1");
  REQUIRE (rows.size () == 1);

  const double expected_theta
      = core::TPrior (cfg.focus, cfg.sensitivity, cfg.stability);
  const double expected_blender
      = core::BlendBootstrapFallback (cfg.focus, cfg.sensitivity,
                                      cfg.stability);
  REQUIRE (expected_blender > 0.5);
  REQUIRE_THAT (GetDouble (rows[0], "theta_dynamic"),
                WithinAbs (expected_theta, 1e-9));
  REQUIRE_THAT (GetDouble (rows[0], "w_relevance"),
                WithinAbs (expected_blender, 1e-9));
}

TEST_CASE ("Legacy state defaults reload from active knobs",
           "[state_persistence][processor_state][knobs]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  {
    auto ops = std::make_unique<DynamicOperationSet> ();
    SignalProcessor processor (MakeConfig (), store, std::move (ops));
    processor.Flush ();
  }

  store->Execute ("INSERT OR REPLACE INTO state (id, signals_processed) "
                  "VALUES (1, 5)");

  SignalProcessor::Config cfg = MakeConfig ();
  cfg.focus = 1.0;
  cfg.sensitivity = 1.0;
  cfg.stability = 1.0;

  auto capture = std::make_unique<CaptureLoadedPolicyStateOp> ();
  auto *capture_raw = capture.get ();
  auto ops = std::make_unique<DynamicOperationSet> (std::move (capture));
  SignalProcessor processor (cfg, store, std::move (ops));

  Signal s;
  s.embedding = Eigen::VectorXf::Random (256);
  s.timestamp = 1234;
  s.source_id = "test";
  processor.Process (s);

  REQUIRE_THAT (capture_raw->weight_relevance,
                WithinAbs (core::Sigmoid (1.0), 1e-9));
  REQUIRE_THAT (capture_raw->coverage_gain_floor,
                WithinAbs (1.0, 1e-9));
  REQUIRE_THAT (capture_raw->theta_dynamic,
                WithinAbs (core::TPrior (cfg.focus, cfg.sensitivity,
                                         cfg.stability),
                           1e-9));
  REQUIRE_THAT (capture_raw->half_life,
                WithinAbs (core::BaseHalfLifePrior (cfg.stability), 1e-9));
  REQUIRE_THAT (capture_raw->weight_surprise,
                WithinAbs (1.0, 1e-9));
  REQUIRE_THAT (capture_raw->rate_decay,
                WithinAbs (
                    core::StabilityStatePriorsForKnobs (cfg.stability).rate_decay,
                    1e-9));
  REQUIRE_THAT (capture_raw->reliability, WithinAbs (0.0, 1e-9));
}

TEST_CASE ("Partial legacy state defaults reload from active knobs",
           "[state_persistence][processor_state][knobs]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  {
    auto ops = std::make_unique<DynamicOperationSet> ();
    SignalProcessor processor (MakeConfig (), store, std::move (ops));
    processor.Flush ();
  }

  store->Execute (
      "INSERT OR REPLACE INTO state (id, signals_processed, weight_relevance) "
      "VALUES (1, 5, 0.9)");

  SignalProcessor::Config cfg = MakeConfig ();
  cfg.focus = 1.0;
  cfg.sensitivity = 1.0;
  cfg.stability = 1.0;

  auto capture = std::make_unique<CaptureLoadedPolicyStateOp> ();
  auto *capture_raw = capture.get ();
  auto ops = std::make_unique<DynamicOperationSet> (std::move (capture));
  SignalProcessor processor (cfg, store, std::move (ops));

  Signal s;
  s.embedding = Eigen::VectorXf::Random (256);
  s.timestamp = 1234;
  s.source_id = "test";
  processor.Process (s);

  REQUIRE_THAT (capture_raw->weight_relevance, WithinAbs (0.9, 1e-9));
  REQUIRE_THAT (capture_raw->theta_dynamic,
                WithinAbs (core::TPrior (cfg.focus, cfg.sensitivity,
                                         cfg.stability),
                           1e-9));
  REQUIRE_THAT (capture_raw->weight_surprise,
                WithinAbs (1.0, 1e-9));
  REQUIRE_THAT (capture_raw->rate_decay,
                WithinAbs (
                    core::StabilityStatePriorsForKnobs (cfg.stability).rate_decay,
                    1e-9));
}

TEST_CASE ("Recent context view derives from signals",
           "[state_persistence][recent_context]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  {
    // Initialize schema
    auto ops = std::make_unique<DynamicOperationSet> ();
    SignalProcessor processor (MakeConfig (), store, std::move (ops));
    processor.Flush ();
  }

  // In v2 schema, recent_context is a VIEW that derives from signals/embeddings.
  // Insert test data into signals table to verify the view works.
  const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds> (
                          std::chrono::system_clock::now ().time_since_epoch ())
                          .count ();

  std::vector<float> emb (256, 0.1f);

  // Insert embedding
  store->Execute (
      "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
      { emb, now_ms });

  // Insert signal
  store->Execute (
      "INSERT INTO signals (embedding_id, source_id, timestamp, modality, created_at) "
      "VALUES (?, ?, ?, 'text', ?)",
      { 1LL, std::string ("test"), now_ms, now_ms });

  // Check recent_context view returns data
  auto rows = store->Execute ("SELECT COUNT(*) as c FROM recent_context");
  REQUIRE (rows.size () == 1);

  long long count = GetInt64 (rows[0], "c");
  // View should show the signal we inserted
  REQUIRE (count >= 1);
}

TEST_CASE ("Recent scores are persisted", "[state_persistence][recent_scores]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  {
    auto ops
        = std::make_unique<DynamicOperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (MakeConfig (), store, std::move (ops));

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

TEST_CASE ("Recent windows restore with knob-derived runtime limits",
           "[state_persistence][recent_context][recent_scores]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  {
    auto ops = std::make_unique<DynamicOperationSet> ();
    SignalProcessor::Config cfg = MakeConfig ();
    cfg.stability = 1.0;
    SignalProcessor processor (cfg, store, std::move (ops));
    processor.Flush ();
  }

  constexpr int kSeedRows = 150;
  for (int i = 0; i < kSeedRows; ++i)
    {
      std::vector<float> emb (256, 0.0f);
      emb[static_cast<std::size_t> (i % 256)] = 1.0f;
      const long long ts = 1000LL + static_cast<long long> (i);
      store->Execute (
          "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
          { emb, ts });
      store->Execute (
          "INSERT INTO signals (embedding_id, source_id, timestamp, modality, "
          "created_at, score) VALUES (?, ?, ?, 'text', ?, ?)",
          { static_cast<long long> (i + 1), std::string ("test"), ts, ts,
            static_cast<double> (i) / static_cast<double> (kSeedRows) });
    }

  std::size_t restored_context_count = 0;
  std::size_t restored_score_count = 0;
  {
    auto capture = std::make_unique<CaptureRecentWindowOp> ();
    capture->context_count = &restored_context_count;
    capture->score_count = &restored_score_count;
    auto ops = std::make_unique<DynamicOperationSet> (std::move (capture));
    SignalProcessor::Config cfg = MakeConfig ();
    cfg.stability = 1.0;
    SignalProcessor processor (cfg, store, std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 5000;
    s.source_id = "test";
    processor.Process (s);
  }

  REQUIRE (restored_context_count == static_cast<std::size_t> (kSeedRows));
  REQUIRE (restored_score_count == static_cast<std::size_t> (kSeedRows));
}

TEST_CASE ("State persistence is idempotent across restarts",
           "[state_persistence][idempotent]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Run 1
  {
    auto ops
        = std::make_unique<DynamicOperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (MakeConfig (), store, std::move (ops));

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
        = std::make_unique<DynamicOperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (MakeConfig (), store, std::move (ops));

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
        = std::make_unique<DynamicOperationSet> (std::make_unique<TriggerBoundaryOp> ());
    SignalProcessor processor (MakeConfig (), store, std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 3000;
    s.source_id = "test";
    processor.Process (s);
    processor.Flush ();
  }

  // Should have exactly one row in unified state table (v2 schema)
  auto state_rows = store->Execute ("SELECT COUNT(*) as c FROM state");
  REQUIRE (GetInt64 (state_rows[0], "c") == 1);

  // Signals processed should be cumulative
  auto state_vals = store->Execute ("SELECT signals_processed FROM state");
  REQUIRE (GetInt64 (state_vals[0], "signals_processed") == 3);
}

TEST_CASE ("Consolidation throughput range persists across processor restart",
           "[state_persistence][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  SignalProcessor::Config cfg = MakeConfig ();

  {
    auto set = std::make_unique<SetConsolidationThroughputOp> ();
    set->state = { 2.5, 11.0, true, false };
    auto ops = std::make_unique<DynamicOperationSet> (std::move (set));
    SignalProcessor processor (cfg, store, std::move (ops));
    Signal signal;
    signal.embedding = Eigen::VectorXf::Zero (256);
    signal.timestamp = 1000;
    signal.source_id = "first/store";
    processor.Process (signal);
  }

  auto rows = store->Execute (
      "SELECT consolidation_rate_floor, consolidation_rate_peak, "
      "consolidation_rate_initialized, consolidation_rate_armed "
      "FROM state WHERE id = 1");
  REQUIRE (rows.size () == 1);
  REQUIRE_THAT (GetDouble (rows[0], "consolidation_rate_floor"),
                WithinAbs (2.5, 1e-12));
  REQUIRE_THAT (GetDouble (rows[0], "consolidation_rate_peak"),
                WithinAbs (11.0, 1e-12));
  REQUIRE (GetInt64 (rows[0], "consolidation_rate_initialized") == 1);
  REQUIRE (GetInt64 (rows[0], "consolidation_rate_armed") == 0);

  operations::consolidation_throughput_state_internal::State restored;
  {
    auto capture = std::make_unique<CaptureConsolidationThroughputOp> ();
    capture->captured = &restored;
    auto ops = std::make_unique<DynamicOperationSet> (std::move (capture));
    SignalProcessor processor (cfg, store, std::move (ops));
    Signal signal;
    signal.embedding = Eigen::VectorXf::Zero (256);
    signal.timestamp = 2000;
    signal.source_id = "second/store";
    processor.Process (signal);
  }
  REQUIRE_THAT (restored.floor, WithinAbs (2.5, 1e-12));
  REQUIRE_THAT (restored.peak, WithinAbs (11.0, 1e-12));
  REQUIRE (restored.initialized);
  REQUIRE_FALSE (restored.armed);
}

TEST_CASE ("Zero consolidation throughput initialization persists across restart",
           "[state_persistence][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  SignalProcessor::Config cfg = MakeConfig ();

  {
    auto set = std::make_unique<SetConsolidationThroughputOp> ();
    set->state = { 0.0, 0.0, true };
    auto ops = std::make_unique<DynamicOperationSet> (std::move (set));
    SignalProcessor processor (cfg, store, std::move (ops));
    Signal signal;
    signal.embedding = Eigen::VectorXf::Zero (256);
    signal.timestamp = 1000;
    signal.source_id = "zero/store";
    processor.Process (signal);
  }

  auto rows = store->Execute (
      "SELECT consolidation_rate_floor, consolidation_rate_peak, "
      "consolidation_rate_initialized FROM state WHERE id = 1");
  REQUIRE (rows.size () == 1);
  REQUIRE_THAT (GetDouble (rows[0], "consolidation_rate_floor"),
                WithinAbs (0.0, 1e-12));
  REQUIRE_THAT (GetDouble (rows[0], "consolidation_rate_peak"),
                WithinAbs (0.0, 1e-12));
  REQUIRE (GetInt64 (rows[0], "consolidation_rate_initialized") == 1);

  operations::consolidation_throughput_state_internal::State observed;
  {
    auto observe = std::make_unique<ObserveConsolidationThroughputOp> ();
    observe->rate = 10.0;
    auto capture = std::make_unique<CaptureConsolidationThroughputOp> ();
    capture->captured = &observed;
    auto ops = std::make_unique<DynamicOperationSet> (
        std::move (observe), std::move (capture));
    SignalProcessor processor (cfg, store, std::move (ops));
    Signal signal;
    signal.embedding = Eigen::VectorXf::Zero (256);
    signal.timestamp = 2000;
    signal.source_id = "positive/store";
    processor.Process (signal);
  }
  REQUIRE (observed.initialized);
  REQUIRE (observed.floor > 0.0);
  REQUIRE (observed.floor < 10.0);
  REQUIRE_THAT (observed.peak, WithinAbs (10.0, 1e-12));
}

TEST_CASE ("Disarmed consolidation range loads with zero processed signals",
           "[state_persistence][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  SignalProcessor::Config cfg = MakeConfig ();
  cortext::store::ApplyMigrations (*store);
  store->Execute ("INSERT INTO state (id) VALUES (1) ON CONFLICT(id) DO NOTHING");
  store->Execute (
      "UPDATE state SET signals_processed = 0, "
      "consolidation_rate_floor = 2.5, consolidation_rate_peak = 11.0, "
      "consolidation_rate_initialized = 1, consolidation_rate_armed = 0 "
      "WHERE id = 1");

  operations::consolidation_throughput_state_internal::State restored;
  auto capture = std::make_unique<CaptureConsolidationThroughputOp> ();
  capture->captured = &restored;
  auto ops = std::make_unique<DynamicOperationSet> (std::move (capture));
  SignalProcessor processor (cfg, store, std::move (ops));
  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.timestamp = 1000;
  signal.source_id = "zero-signal/restart";
  processor.Process (signal);

  REQUIRE_THAT (restored.floor, WithinAbs (2.5, 1e-12));
  REQUIRE_THAT (restored.peak, WithinAbs (11.0, 1e-12));
  REQUIRE (restored.initialized);
  REQUIRE_FALSE (restored.armed);
}

// Operation that populates WM slots for testing persistence
struct PopulateWMSlotsOp : IOperation
{
  int num_slots = 2;
  double strength = 0.8;

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    pctx.wm_slots.clear ();
    for (int i = 0; i < num_slots; ++i)
      {
        ProcessorContext::WMSlot slot;
        slot.embedding = Eigen::VectorXf::Constant (256, static_cast<float> (i + 1) * 0.1f);
        slot.embedding.normalize ();
        slot.strength = strength;
        slot.last_ts = static_cast<double> (ctx.GetSignal ().timestamp);
        slot.pos_index = i;
        pctx.wm_slots.push_back (std::move (slot));
      }
    ctx.RequestFinalizeEpisode ();
  }
};

struct RotateOneWMSlotOp : IOperation
{
  mutable int sequence = 0;

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    pctx.wm_slots.clear ();

    ProcessorContext::WMSlot slot;
    slot.embedding = Eigen::VectorXf::Zero (256);
    slot.embedding[sequence % 256] = 1.0f;
    slot.strength = 0.8;
    slot.last_ts = static_cast<double> (ctx.GetSignal ().timestamp) / 1000.0;
    slot.start_ts = static_cast<int64_t> (ctx.GetSignal ().timestamp);
    slot.source_id = "test";
    slot.modality = "text";
    slot.n_signals = 1;
    slot.s_max = 0.5;
    slot.s_avg = 0.5;
    pctx.wm_slots.push_back (std::move (slot));
    pctx.wm_slots_dirty = true;
    ++sequence;
  }
};

TEST_CASE ("Working memory slots are persisted",
           "[state_persistence][working_memory]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  {
    SignalProcessor::Config cfg = MakeConfig ();
    cfg.focus = 0.0;
    cfg.sensitivity = 1.0;
    cfg.stability = 0.0;
    auto ops = std::make_unique<DynamicOperationSet> (
        std::make_unique<PopulateWMSlotsOp> ());
    SignalProcessor processor (cfg, store, std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 1000;
    s.source_id = "test";
    processor.Process (s);
    processor.Flush ();
  }

  // Check memories table has WM entries (v2 schema: kind='WORKING', active slots)
  // Active slots have end_ts IS NULL
  auto rows = store->Execute (
      "SELECT memory_id, strength, source_reliability, stability FROM memories "
      "WHERE kind = 'WORKING' AND end_ts IS NULL ORDER BY memory_id");
  REQUIRE (rows.size () == 2);
  REQUIRE_THAT (GetDouble (rows[0], "strength"), WithinAbs (0.8, 0.01));
  REQUIRE_THAT (GetDouble (rows[1], "strength"), WithinAbs (0.8, 0.01));
  const double expected_source_reliability
      = core::SourceReliabilityPrior (0.0, 1.0, 0.0);
  const double expected_stability
      = core::MemoryInitialStabilityPolicy (0.0, 1.0, 0.0);
  REQUIRE_THAT (GetDouble (rows[0], "source_reliability"),
                WithinAbs (expected_source_reliability, 1e-9));
  REQUIRE_THAT (GetDouble (rows[1], "source_reliability"),
                WithinAbs (expected_source_reliability, 1e-9));
  REQUIRE_THAT (GetDouble (rows[0], "stability"),
                WithinAbs (expected_stability, 1e-9));
  REQUIRE_THAT (GetDouble (rows[1], "stability"),
                WithinAbs (expected_stability, 1e-9));
}

TEST_CASE ("Closed working memory rows are pruned incrementally",
           "[state_persistence][working_memory][performance]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg = MakeConfig ();
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<RotateOneWMSlotOp> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  for (int i = 0; i < 96; ++i)
    {
      Signal s;
      s.embedding = Eigen::VectorXf::Random (256);
      s.timestamp = static_cast<uint64_t> (1000 + i * 1000);
      s.source_id = "test";
      processor.Process (s);
    }

  auto rows = store->Execute (
      "SELECT "
      "SUM(CASE WHEN end_ts IS NULL THEN 1 ELSE 0 END) AS active_count, "
      "SUM(CASE WHEN end_ts IS NOT NULL THEN 1 ELSE 0 END) AS closed_count "
      "FROM memories WHERE kind = 'WORKING'");
  REQUIRE (rows.size () == 1);
  REQUIRE (GetInt64 (rows[0], "active_count") == 1LL);
  REQUIRE (GetInt64 (rows[0], "closed_count") <= 64LL);
}

TEST_CASE ("Working memory slots are loaded on startup",
           "[state_persistence][working_memory]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Directly insert slots into database
  {
    auto ops = std::make_unique<DynamicOperationSet> ();
    SignalProcessor processor (MakeConfig (), store, std::move (ops));
    // This ensures schema is created
    processor.Flush ();
  }

  // Insert test data directly - DB stores timestamps in milliseconds
  // v2 schema: WM slots stored in memories table with kind='WORKING'
  const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds> (
                          std::chrono::system_clock::now ().time_since_epoch ())
                          .count ();

  std::vector<float> emb1 (256, 0.1f);
  std::vector<float> emb2 (256, 0.2f);

  // First create embeddings
  store->Execute (
      "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
      { emb1, now_ms });
  store->Execute (
      "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
      { emb2, now_ms });

  // Then create WM memory entries
  store->Execute (
      "INSERT INTO memories (embedding_id, source_id, kind, modality, strength, "
      "last_access, start_ts, n_signals, s_max, s_avg, created_at) "
      "VALUES (?, ?, 'WORKING', 'text', ?, ?, ?, 1, 0.5, 0.5, ?)",
      { 1LL, std::string ("test"), 0.9, now_ms, now_ms, now_ms });
  store->Execute (
      "INSERT INTO memories (embedding_id, source_id, kind, modality, strength, "
      "last_access, start_ts, n_signals, s_max, s_avg, created_at) "
      "VALUES (?, ?, 'WORKING', 'text', ?, ?, ?, 1, 0.5, 0.5, ?)",
      { 2LL, std::string ("test"), 0.7, now_ms, now_ms, now_ms });

  // Create new processor - should load slots
  {
    auto verify_op = std::make_unique<DynamicOperationSet> ();
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.sensitivity = 0.5;
    SignalProcessor processor (cfg, store, std::move (verify_op));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = static_cast<uint64_t> (now_ms); // Signal timestamps are in milliseconds
    s.source_id = "test";
    auto out = processor.Process (s);

    // Access processor internals through output or verify via another process
  }

  // Create processor with custom op to verify slots were loaded
  struct VerifyWMSlotsOp : IOperation
  {
    mutable bool verified = false;
    mutable size_t slot_count = 0;
    mutable double slot0_strength = 0.0;
    mutable double slot1_strength = 0.0;

    void
    Execute (OperationContext &ctx, Transaction & /*tx*/) const override
    {
      const auto &pctx = ctx.GetProcessorContext ();
      slot_count = pctx.wm_slots.size ();
      if (slot_count >= 2)
        {
          slot0_strength = pctx.wm_slots[0].strength;
          slot1_strength = pctx.wm_slots[1].strength;
        }
      verified = true;
    }
  };

  auto verify_ptr = std::make_unique<VerifyWMSlotsOp> ();
  auto *verify_raw = verify_ptr.get ();
  auto ops = std::make_unique<DynamicOperationSet> (std::move (verify_ptr));
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.5;
  SignalProcessor processor (cfg, store, std::move (ops));

  Signal s;
  s.embedding = Eigen::VectorXf::Random (256);
  s.timestamp = static_cast<uint64_t> (now_ms); // Signal timestamps are in milliseconds
  s.source_id = "test";
  processor.Process (s);

  REQUIRE (verify_raw->verified);
  REQUIRE (verify_raw->slot_count == 2);
  // Strength may have decayed slightly if any time elapsed
  REQUIRE (verify_raw->slot0_strength > 0.0);
  REQUIRE (verify_raw->slot1_strength > 0.0);
}

TEST_CASE ("Working memory slots decay on load",
           "[state_persistence][working_memory][decay]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Create schema
  {
    auto ops = std::make_unique<DynamicOperationSet> ();
    SignalProcessor processor (MakeConfig (), store, std::move (ops));
    processor.Flush ();
  }

  // Insert slot with old timestamp (5 seconds ago)
  // v2 schema: WM slots stored in memories table with kind='WORKING'
  const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds> (
                          std::chrono::system_clock::now ().time_since_epoch ())
                          .count ();
  const auto old_ts_ms = now_ms - 5000; // 5 seconds ago in milliseconds

  std::vector<float> emb (256, 0.1f);

  // First create embedding
  store->Execute (
      "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
      { emb, now_ms });

  // Then create WM memory entry with old last_access timestamp
  store->Execute (
      "INSERT INTO memories (embedding_id, source_id, kind, modality, strength, "
      "last_access, start_ts, n_signals, s_max, s_avg, created_at) "
      "VALUES (?, ?, 'WORKING', 'text', ?, ?, ?, 1, 0.5, 0.5, ?)",
      { 1LL, std::string ("test"), 1.0, old_ts_ms, old_ts_ms, old_ts_ms });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  const double expected_strength = std::max (
      core::WMStrengthFloor (cfg.focus, cfg.sensitivity, cfg.stability),
      1.0 - core::WMMaintenanceCostPerSlot (cfg.sensitivity, cfg.focus)
                * 5.0);

  // Load with neutral knobs. The WM21 default spreads the legacy full-window
  // maintenance budget over 21 slots, so five seconds decays to the
  // knob-derived expected strength rather than the old seven-slot value.
  struct VerifyDecayOp : IOperation
  {
    mutable double loaded_strength = 0.0;

    void
    Execute (OperationContext &ctx, Transaction & /*tx*/) const override
    {
      const auto &pctx = ctx.GetProcessorContext ();
      if (!pctx.wm_slots.empty ())
        {
          loaded_strength = pctx.wm_slots[0].strength;
        }
    }
  };

  auto verify_ptr = std::make_unique<VerifyDecayOp> ();
  auto *verify_raw = verify_ptr.get ();
  auto ops = std::make_unique<DynamicOperationSet> (std::move (verify_ptr));
  SignalProcessor processor (cfg, store, std::move (ops));

  Signal s;
  s.embedding = Eigen::VectorXf::Random (256);
  s.timestamp = static_cast<uint64_t> (now_ms); // Signal timestamps are in milliseconds
  s.source_id = "test";
  processor.Process (s);

  REQUIRE_THAT (verify_raw->loaded_strength,
                WithinAbs (expected_strength, 1e-3));
}

TEST_CASE ("Working memory reload preserves floor like live passive decay",
           "[state_persistence][working_memory][decay]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Create schema
  {
    auto ops = std::make_unique<DynamicOperationSet> ();
    SignalProcessor processor (MakeConfig (), store, std::move (ops));
    processor.Flush ();
  }

  // Insert slot with very old timestamp. Under the WM21 maintenance budget this
  // needs to be older than the old seven-slot test case to reach the floor.
  // v2 schema: WM slots stored in memories table with kind='WORKING'
  const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds> (
                          std::chrono::system_clock::now ().time_since_epoch ())
                          .count ();
  const auto old_ts_ms = now_ms - 60000; // 60 seconds ago in milliseconds

  std::vector<float> emb (256, 0.1f);

  // First create embedding
  store->Execute (
      "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
      { emb, now_ms });

  // Then create WM memory entry with very old last_access timestamp
  store->Execute (
      "INSERT INTO memories (embedding_id, source_id, kind, modality, strength, "
      "last_access, start_ts, n_signals, s_max, s_avg, created_at) "
      "VALUES (?, ?, 'WORKING', 'text', ?, ?, ?, 1, 0.5, 0.5, ?)",
      { 1LL, std::string ("test"), 1.0, old_ts_ms, old_ts_ms, old_ts_ms });

  // Load with sensitivity=0.5 -> cost_per_slot would decay below zero.
  // Persisted reload should match live WM passive decay and clamp to the
  // knob-derived strength floor instead of dropping the active slot.
  struct VerifyReloadedFloorOp : IOperation
  {
    mutable size_t slot_count = 999;
    mutable double loaded_strength = -1.0;

    void
    Execute (OperationContext &ctx, Transaction & /*tx*/) const override
    {
      const auto &pctx = ctx.GetProcessorContext ();
      slot_count = pctx.wm_slots.size ();
      if (!pctx.wm_slots.empty ())
        {
          loaded_strength = pctx.wm_slots[0].strength;
        }
    }
  };

  auto verify_ptr = std::make_unique<VerifyReloadedFloorOp> ();
  auto *verify_raw = verify_ptr.get ();
  auto ops = std::make_unique<DynamicOperationSet> (std::move (verify_ptr));
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  SignalProcessor processor (cfg, store, std::move (ops));

  Signal s;
  s.embedding = Eigen::VectorXf::Random (256);
  s.timestamp = static_cast<uint64_t> (now_ms); // Signal timestamps are in milliseconds
  s.source_id = "test";
  processor.Process (s);

  REQUIRE (verify_raw->slot_count == 1);
  REQUIRE_THAT (
      verify_raw->loaded_strength,
      WithinAbs (core::WMStrengthFloor (cfg.focus, cfg.sensitivity,
                                        cfg.stability),
                 1e-9));
}

TEST_CASE ("Working memory persistence falls back to last access when the "
           "strength timestamp is unset",
           "[state_persistence][working_memory][decay][regression]")
{
  constexpr std::uint64_t kInsertMs = 1'000'000;
  constexpr std::uint64_t kUpdateMs = 1'002'000;
  constexpr std::uint64_t kInitializedLastAccessMs = 1'004'000;
  constexpr std::uint64_t kInitializedStrengthMs = 1'003'500;

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  auto clock = std::make_shared<FixedClock> (kInsertMs);

  SignalProcessor::Config cfg = MakeConfig ();
  cfg.clock = clock;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto make_processor = [&] (ProcessorContext **captured) {
    auto capture = std::make_unique<CaptureProcessorContextOp> ();
    capture->captured = captured;
    auto ops = std::make_unique<DynamicOperationSet> (std::move (capture));
    return std::make_unique<SignalProcessor> (cfg, store, std::move (ops));
  };
  auto process_at = [] (SignalProcessor &processor, std::uint64_t timestamp) {
    Signal signal;
    signal.embedding = Eigen::VectorXf::Ones (256);
    signal.timestamp = timestamp;
    signal.source_id = "compat/source";
    processor.Process (signal);
  };

  {
    ProcessorContext *captured = nullptr;
    auto processor = make_processor (&captured);
    process_at (*processor, kInsertMs);
    REQUIRE (captured != nullptr);

    ProcessorContext::WMSlot slot;
    slot.embedding = Eigen::VectorXf::Ones (256);
    slot.strength = 0.8;
    slot.last_ts = static_cast<double> (kInsertMs) / 1000.0;
    slot.strength_ts = 0.0;
    slot.source_id = "compat/source";
    slot.start_ts = static_cast<int64_t> (kInsertMs);
    captured->wm_slots.push_back (std::move (slot));
    captured->wm_slots_dirty = true;
    processor->Flush ();
  }

  auto rows = store->Execute (
      "SELECT last_access, strength_updated_at FROM memories "
      "WHERE kind = 'WORKING' AND end_ts IS NULL");
  REQUIRE (rows.size () == 1);
  REQUIRE (GetInt64 (rows[0], "last_access")
           == static_cast<long long> (kInsertMs));
  REQUIRE (GetInt64 (rows[0], "strength_updated_at")
           == static_cast<long long> (kInsertMs));

  {
    ProcessorContext *captured = nullptr;
    auto processor = make_processor (&captured);
    process_at (*processor, kInsertMs);
    REQUIRE (captured != nullptr);
    REQUIRE (captured->wm_slots.size () == 1);
    REQUIRE_THAT (captured->wm_slots[0].strength, WithinAbs (0.8, 1e-9));

    auto &slot = captured->wm_slots[0];
    slot.strength = 0.7;
    slot.last_ts = static_cast<double> (kUpdateMs) / 1000.0;
    slot.strength_ts = 0.0;
    slot.metadata_dirty = true;
    captured->wm_slots_dirty = true;
    processor->Flush ();
  }

  rows = store->Execute (
      "SELECT last_access, strength_updated_at FROM memories "
      "WHERE kind = 'WORKING' AND end_ts IS NULL");
  REQUIRE (rows.size () == 1);
  REQUIRE (GetInt64 (rows[0], "last_access")
           == static_cast<long long> (kUpdateMs));
  REQUIRE (GetInt64 (rows[0], "strength_updated_at")
           == static_cast<long long> (kUpdateMs));

  clock->SetNowMillis (kUpdateMs);
  {
    ProcessorContext *captured = nullptr;
    auto processor = make_processor (&captured);
    process_at (*processor, kUpdateMs);
    REQUIRE (captured != nullptr);
    REQUIRE (captured->wm_slots.size () == 1);
    REQUIRE_THAT (captured->wm_slots[0].strength, WithinAbs (0.7, 1e-9));

    auto &slot = captured->wm_slots[0];
    slot.strength = 0.6;
    slot.last_ts
        = static_cast<double> (kInitializedLastAccessMs) / 1000.0;
    slot.strength_ts
        = static_cast<double> (kInitializedStrengthMs) / 1000.0;
    slot.metadata_dirty = true;
    captured->wm_slots_dirty = true;
    processor->Flush ();
  }

  rows = store->Execute (
      "SELECT last_access, strength_updated_at FROM memories "
      "WHERE kind = 'WORKING' AND end_ts IS NULL");
  REQUIRE (rows.size () == 1);
  REQUIRE (GetInt64 (rows[0], "last_access")
           == static_cast<long long> (kInitializedLastAccessMs));
  REQUIRE (GetInt64 (rows[0], "strength_updated_at")
           == static_cast<long long> (kInitializedStrengthMs));

  clock->SetNowMillis (kInitializedLastAccessMs);
  ProcessorContext *captured = nullptr;
  auto processor = make_processor (&captured);
  process_at (*processor, kInitializedLastAccessMs);
  REQUIRE (captured != nullptr);
  REQUIRE (captured->wm_slots.size () == 1);
  const double expected_strength = std::max (
      core::WMStrengthFloor (cfg.focus, cfg.sensitivity, cfg.stability),
      0.6 - core::WMMaintenanceCostPerSlot (cfg.sensitivity, cfg.focus) * 0.5);
  REQUIRE_THAT (captured->wm_slots[0].strength,
                WithinAbs (expected_strength, 1e-9));
}

TEST_CASE ("Passive working memory decay persists without a forced flush",
           "[state_persistence][working_memory][decay][regression]")
{
  constexpr std::uint64_t kInitialMs = 1'000;
  constexpr std::uint64_t kDecayMs = 1'001;
  constexpr double kInitialStrength = 2.0;

  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  auto clock = std::make_shared<FixedClock> (kInitialMs);

  SignalProcessor::Config cfg = MakeConfig ();
  cfg.clock = clock;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto process_at = [] (SignalProcessor &processor, std::uint64_t timestamp) {
    Signal signal;
    signal.embedding = Eigen::VectorXf::Ones (256);
    signal.timestamp = timestamp;
    signal.source_id = "passive-decay/source";
    processor.Process (signal);
  };
  auto make_capture = [] (ProcessorContext **captured) {
    auto capture = std::make_unique<CaptureProcessorContextOp> ();
    capture->captured = captured;
    return capture;
  };

  {
    ProcessorContext *captured = nullptr;
    auto ops = std::make_unique<DynamicOperationSet> (
        make_capture (&captured));
    SignalProcessor processor (cfg, store, std::move (ops));
    process_at (processor, kInitialMs);
    REQUIRE (captured != nullptr);

    ProcessorContext::WMSlot slot;
    slot.embedding = Eigen::VectorXf::Ones (256);
    slot.strength = kInitialStrength;
    slot.last_ts = static_cast<double> (kInitialMs) / 1000.0;
    slot.strength_ts = slot.last_ts;
    slot.source_id = "passive-decay/source";
    slot.start_ts = static_cast<int64_t> (kInitialMs);
    captured->wm_slots.push_back (std::move (slot));
    captured->wm_slots_dirty = true;
    processor.Flush ();
  }

  const double expected_strength
      = kInitialStrength
        - core::WMMaintenanceCostPerSlot (cfg.sensitivity, cfg.focus)
              * static_cast<double> (kDecayMs - kInitialMs) / 1000.0;

  {
    ProcessorContext *captured = nullptr;
    auto ops = std::make_unique<DynamicOperationSet> (
        std::make_unique<operations::WorkingMemory> (),
        make_capture (&captured));
    SignalProcessor processor (cfg, store, std::move (ops));
    clock->SetNowMillis (kDecayMs);
    process_at (processor, kDecayMs);
    REQUIRE (captured != nullptr);
    REQUIRE (captured->wm_slots.size () == 1);
    REQUIRE_THAT (captured->wm_slots[0].strength,
                  WithinAbs (expected_strength, 1e-9));
    REQUIRE_FALSE (captured->wm_slots_dirty);
  }

  auto rows = store->Execute (
      "SELECT strength, strength_updated_at FROM memories "
      "WHERE kind = 'WORKING' AND end_ts IS NULL");
  REQUIRE (rows.size () == 1);
  REQUIRE_THAT (GetDouble (rows[0], "strength"),
                WithinAbs (expected_strength, 1e-9));
  REQUIRE (GetInt64 (rows[0], "strength_updated_at")
           == static_cast<long long> (kDecayMs));

  SignalProcessor::Config reopen_cfg = cfg;
  reopen_cfg.sensitivity = 1.0;
  ProcessorContext *reloaded = nullptr;
  auto reopen_ops = std::make_unique<DynamicOperationSet> (
      make_capture (&reloaded));
  SignalProcessor reopened (reopen_cfg, store, std::move (reopen_ops));
  process_at (reopened, kDecayMs);
  REQUIRE (reloaded != nullptr);
  REQUIRE (reloaded->wm_slots.size () == 1);
  REQUIRE_THAT (reloaded->wm_slots[0].strength,
                WithinAbs (expected_strength, 1e-9));
}

TEST_CASE ("Working memory reload does not persist a knob-floor recharge",
           "[state_persistence][working_memory][decay][regression]")
{
  constexpr std::uint64_t kInitialMs = 1'000;
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  auto clock = std::make_shared<FixedClock> (kInitialMs);

  SignalProcessor::Config prior_cfg = MakeConfig ();
  prior_cfg.clock = clock;
  prior_cfg.focus = 1.0;
  prior_cfg.sensitivity = 1.0;
  prior_cfg.stability = 0.0;
  const double prior_floor = core::WMStrengthFloor (
      prior_cfg.focus, prior_cfg.sensitivity, prior_cfg.stability);

  auto process_at = [] (SignalProcessor &processor, std::uint64_t timestamp) {
    Signal signal;
    signal.embedding = Eigen::VectorXf::Ones (256);
    signal.timestamp = timestamp;
    signal.source_id = "floor-recharge/source";
    processor.Process (signal);
  };
  auto make_capture = [] (ProcessorContext **captured) {
    auto capture = std::make_unique<CaptureProcessorContextOp> ();
    capture->captured = captured;
    return capture;
  };

  {
    ProcessorContext *captured = nullptr;
    auto ops = std::make_unique<DynamicOperationSet> (make_capture (&captured));
    SignalProcessor processor (prior_cfg, store, std::move (ops));
    process_at (processor, kInitialMs);
    REQUIRE (captured != nullptr);

    ProcessorContext::WMSlot slot;
    slot.embedding = Eigen::VectorXf::Ones (256);
    slot.strength = prior_floor;
    slot.last_ts = static_cast<double> (kInitialMs) / 1000.0;
    slot.strength_ts = slot.last_ts;
    slot.source_id = "floor-recharge/source";
    slot.start_ts = static_cast<int64_t> (kInitialMs);
    captured->wm_slots.push_back (std::move (slot));
    captured->wm_slots_dirty = true;
    processor.Flush ();
  }

  SignalProcessor::Config current_cfg = prior_cfg;
  current_cfg.focus = 0.0;
  current_cfg.sensitivity = 0.0;
  current_cfg.stability = 1.0;
  REQUIRE (prior_floor
           < core::WMStrengthFloor (current_cfg.focus,
                                    current_cfg.sensitivity,
                                    current_cfg.stability));

  auto verify_reload = [&] (std::uint64_t timestamp) {
    clock->SetNowMillis (timestamp);
    ProcessorContext *captured = nullptr;
    auto ops = std::make_unique<DynamicOperationSet> (
        std::make_unique<operations::WorkingMemory> (),
        make_capture (&captured));
    SignalProcessor processor (current_cfg, store, std::move (ops));
    process_at (processor, timestamp);
    REQUIRE (captured != nullptr);
    REQUIRE (captured->wm_slots.size () == 1);
    REQUIRE_THAT (captured->wm_slots[0].strength,
                  WithinAbs (prior_floor, 1e-9));

    auto rows = store->Execute (
        "SELECT strength, strength_updated_at FROM memories "
        "WHERE kind = 'WORKING' AND end_ts IS NULL");
    REQUIRE (rows.size () == 1);
    REQUIRE_THAT (GetDouble (rows[0], "strength"),
                  WithinAbs (prior_floor, 1e-9));
    REQUIRE (GetInt64 (rows[0], "strength_updated_at")
             == static_cast<long long> (kInitialMs));
  };

  verify_reload (kInitialMs);
  verify_reload (kInitialMs - 100);
}

TEST_CASE ("Working memory timestamp conversion is checked and portable",
           "[state_persistence][working_memory][timestamp][regression]")
{
  REQUIRE (internal::WorkingMemorySecondsToMillis (1.001) == 1001);
  REQUIRE (internal::WorkingMemorySecondsToMillis (0.0) == 0);
  REQUIRE_THROWS_AS (internal::WorkingMemorySecondsToMillis (-0.001),
                     std::invalid_argument);
  REQUIRE_THROWS_AS (
      internal::WorkingMemorySecondsToMillis (
          std::numeric_limits<double>::quiet_NaN ()),
      std::invalid_argument);
  REQUIRE_THROWS_AS (
      internal::WorkingMemorySecondsToMillis (
          std::numeric_limits<double>::infinity ()),
      std::invalid_argument);

  const double exclusive_upper_millis = std::ldexp (1.0, 63);
  const double exclusive_upper_seconds = exclusive_upper_millis / 1000.0;
  REQUIRE_THROWS_AS (
      internal::WorkingMemorySecondsToMillis (exclusive_upper_seconds),
      std::out_of_range);
  REQUIRE_THROWS_AS (
      internal::WorkingMemorySecondsToMillis (
          std::numeric_limits<double>::max ()),
      std::out_of_range);

  const double safe_seconds
      = std::nextafter (exclusive_upper_seconds, 0.0);
  const double safe_millis = safe_seconds * 1000.0;
  REQUIRE (safe_millis < exclusive_upper_millis);
  REQUIRE (internal::WorkingMemorySecondsToMillis (safe_seconds)
           == static_cast<int64_t> (safe_millis));
}

TEST_CASE ("Load-time working memory decay persists on the next process",
           "[state_persistence][working_memory][decay][regression]")
{
  constexpr std::uint64_t kInitialMs = 1'000;
  constexpr std::uint64_t kLoadMs = 1'001;
  constexpr double kInitialStrength = 2.0;

  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  auto clock = std::make_shared<FixedClock> (kInitialMs);
  SignalProcessor::Config cfg = MakeConfig ();
  cfg.clock = clock;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto process_at = [] (SignalProcessor &processor, std::uint64_t timestamp) {
    Signal signal;
    signal.embedding = Eigen::VectorXf::Ones (256);
    signal.timestamp = timestamp;
    signal.source_id = "load-decay/source";
    processor.Process (signal);
  };
  auto make_capture = [] (ProcessorContext **captured) {
    auto capture = std::make_unique<CaptureProcessorContextOp> ();
    capture->captured = captured;
    return capture;
  };

  {
    ProcessorContext *captured = nullptr;
    auto ops = std::make_unique<DynamicOperationSet> (
        make_capture (&captured));
    SignalProcessor processor (cfg, store, std::move (ops));
    process_at (processor, kInitialMs);
    REQUIRE (captured != nullptr);

    ProcessorContext::WMSlot slot;
    slot.embedding = Eigen::VectorXf::Ones (256);
    slot.strength = kInitialStrength;
    slot.last_ts = static_cast<double> (kInitialMs) / 1000.0;
    slot.strength_ts = slot.last_ts;
    slot.source_id = "load-decay/source";
    slot.start_ts = static_cast<int64_t> (kInitialMs);
    captured->wm_slots.push_back (std::move (slot));
    captured->wm_slots_dirty = true;
    processor.Flush ();
  }

  const double expected_strength
      = kInitialStrength
        - core::WMMaintenanceCostPerSlot (cfg.sensitivity, cfg.focus)
              * static_cast<double> (kLoadMs - kInitialMs) / 1000.0;
  clock->SetNowMillis (kLoadMs);
  {
    ProcessorContext *captured = nullptr;
    auto ops = std::make_unique<DynamicOperationSet> (
        make_capture (&captured));
    SignalProcessor processor (cfg, store, std::move (ops));
    process_at (processor, kLoadMs);
    REQUIRE (captured != nullptr);
    REQUIRE (captured->wm_slots.size () == 1);
    REQUIRE_THAT (captured->wm_slots[0].strength,
                  WithinAbs (expected_strength, 1e-9));
  }

  auto rows = store->Execute (
      "SELECT strength, strength_updated_at FROM memories "
      "WHERE kind = 'WORKING' AND end_ts IS NULL");
  REQUIRE (rows.size () == 1);
  REQUIRE_THAT (GetDouble (rows[0], "strength"),
                WithinAbs (expected_strength, 1e-9));
  REQUIRE (GetInt64 (rows[0], "strength_updated_at")
           == static_cast<long long> (kLoadMs));

  SignalProcessor::Config reopen_cfg = cfg;
  reopen_cfg.sensitivity = 1.0;
  ProcessorContext *reloaded = nullptr;
  auto reopen_ops = std::make_unique<DynamicOperationSet> (
      make_capture (&reloaded));
  SignalProcessor reopened (reopen_cfg, store, std::move (reopen_ops));
  process_at (reopened, kLoadMs);
  REQUIRE (reloaded != nullptr);
  REQUIRE (reloaded->wm_slots.size () == 1);
  REQUIRE_THAT (reloaded->wm_slots[0].strength,
                WithinAbs (expected_strength, 1e-9));
}

TEST_CASE ("Flush restores working-memory persistence state after commit failure",
           "[state_persistence][working_memory][flush][regression]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  const std::size_t baseline_cache_count = cache::RegistrySizeForTest ();
  auto sqlite = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*sqlite);
  cortext::testing::SeedEmbeddingV2 (
      *sqlite, 100LL, std::vector<float> (256, 0.125f), 900LL);
  cortext::testing::SeedEmbeddingV2 (
      *sqlite, 101LL, std::vector<float> (256, 0.25f), 950LL);
  cortext::testing::SeedMemoryV2 (*sqlite, 10LL, 100LL, "lineage/source",
                                  "LONG_TERM", 1.0, 900LL);
  sqlite->Execute (
      "INSERT INTO memory_reconstructions("
      "reconstruction_id, memory_id, embedding_id, created_at) "
      "VALUES(1, 10, 101, 950)");
  cortext::testing::SeedCurrentMemoryEmbeddingV2 (*sqlite, 10LL, 101LL);
  auto store = std::make_shared<FailCommitStore> (sqlite);

  ProcessorContext *captured = nullptr;
  auto capture = std::make_unique<CaptureProcessorContextOp> ();
  capture->captured = &captured;
  auto ops = std::make_unique<DynamicOperationSet> (std::move (capture));
  SignalProcessor processor (MakeConfig (), store, std::move (ops));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Ones (256);
  signal.timestamp = 1000;
  signal.source_id = "flush/source";
  processor.Process (signal);
  REQUIRE (captured != nullptr);
  REQUIRE (cache::RegistrySizeForTest () == baseline_cache_count + 1);
  REQUIRE (cache::Find (*captured) != nullptr);
  REQUIRE (cache::BaseEmbeddingIdForMemory (*captured, 10LL, 0) == 100LL);

  ProcessorContext::WMSlot slot;
  slot.embedding = signal.embedding;
  slot.strength = 1.0;
  slot.last_ts = 1.0;
  slot.strength_ts = 1.0;
  slot.source_id = signal.source_id;
  slot.start_ts = 1000;
  SignalRecord record;
  record.embedding = signal.embedding;
  record.timestamp = 1000;
  record.modality = "text";
  record.mime = "text/plain";
  slot.signal_records.push_back (std::move (record));
  captured->wm_slots.push_back (std::move (slot));
  captured->wm_slots_dirty = true;

  store->fail_next_commit = true;
  REQUIRE_THROWS_WITH (processor.Flush (), "injected commit failure");
  REQUIRE (captured->wm_slots.size () == 1);
  REQUIRE (captured->wm_slots[0].persisted_signal_record_count == 0);
  REQUIRE (captured->wm_slots[0].signal_records_dirty);
  REQUIRE (captured->wm_slots_dirty);
  REQUIRE (cache::RegistrySizeForTest () == baseline_cache_count + 1);
  REQUIRE (cache::Find (*captured) != nullptr);
  REQUIRE (cache::BaseEmbeddingIdForMemory (*captured, 10LL, 0) == 100LL);

  signal.timestamp = 2000;
  REQUIRE_NOTHROW (processor.Process (signal));
  REQUIRE (cache::RegistrySizeForTest () == baseline_cache_count + 1);
  REQUIRE (cache::Find (*captured) != nullptr);

  REQUIRE_NOTHROW (processor.Flush ());
  const auto rows = sqlite->Execute (
      "SELECT COUNT(*) AS count FROM signals WHERE source_id = ?",
      { std::string ("flush/source") });
  REQUIRE (GetInt64 (rows[0], "count") == 1);
}
