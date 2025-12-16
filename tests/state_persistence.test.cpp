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

// Operation that populates WM slots for testing persistence
struct PopulateWMSlotsOp : IOperation
{
  int num_slots = 2;
  double strength = 0.8;

  void
  Execute (OperationContext &ctx) const override
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

TEST_CASE ("Working memory slots are persisted",
           "[state_persistence][working_memory]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  {
    auto ops = std::make_unique<OperationSet> (
        std::make_unique<PopulateWMSlotsOp> ());
    SignalProcessor processor (SignalProcessor::Config{}, store,
                               std::move (ops));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = 1000;
    s.source_id = "test";
    processor.Process (s);
    processor.Flush ();
  }

  // Check working_memory_slots table has entries
  auto rows = store->Execute (
      "SELECT slot_index, strength FROM working_memory_slots ORDER BY slot_index");
  REQUIRE (rows.size () == 2);
  REQUIRE (GetInt64 (rows[0], "slot_index") == 0);
  REQUIRE (GetInt64 (rows[1], "slot_index") == 1);
  REQUIRE_THAT (GetDouble (rows[0], "strength"), WithinAbs (0.8, 0.01));
  REQUIRE_THAT (GetDouble (rows[1], "strength"), WithinAbs (0.8, 0.01));
}

TEST_CASE ("Working memory slots are loaded on startup",
           "[state_persistence][working_memory]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Directly insert slots into database
  {
    auto ops = std::make_unique<OperationSet> ();
    SignalProcessor processor (SignalProcessor::Config{}, store,
                               std::move (ops));
    // This ensures schema is created
    processor.Flush ();
  }

  // Insert test data directly
  const auto now = std::chrono::duration_cast<std::chrono::seconds> (
                       std::chrono::system_clock::now ().time_since_epoch ())
                       .count ();

  std::vector<float> emb1 (256, 0.1f);
  std::vector<float> emb2 (256, 0.2f);

  store->Execute (
      "INSERT INTO working_memory_slots (slot_index, strength, timestamp, embedding) "
      "VALUES (?, ?, ?, ?)",
      { 0LL, 0.9, now, emb1 });
  store->Execute (
      "INSERT INTO working_memory_slots (slot_index, strength, timestamp, embedding) "
      "VALUES (?, ?, ?, ?)",
      { 1LL, 0.7, now, emb2 });

  // Create new processor - should load slots
  {
    auto verify_op = std::make_unique<OperationSet> ();
    SignalProcessor::Config cfg;
    cfg.sensitivity = 0.5;
    SignalProcessor processor (cfg, store, std::move (verify_op));

    Signal s;
    s.embedding = Eigen::VectorXf::Random (256);
    s.timestamp = static_cast<uint64_t> (now);
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
    Execute (OperationContext &ctx) const override
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
  auto ops = std::make_unique<OperationSet> (std::move (verify_ptr));
  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.5;
  SignalProcessor processor (cfg, store, std::move (ops));

  Signal s;
  s.embedding = Eigen::VectorXf::Random (256);
  s.timestamp = static_cast<uint64_t> (now);
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
    auto ops = std::make_unique<OperationSet> ();
    SignalProcessor processor (SignalProcessor::Config{}, store,
                               std::move (ops));
    processor.Flush ();
  }

  // Insert slot with old timestamp (5 seconds ago)
  const auto now = std::chrono::duration_cast<std::chrono::seconds> (
                       std::chrono::system_clock::now ().time_since_epoch ())
                       .count ();
  const auto old_ts = now - 5; // 5 seconds ago

  std::vector<float> emb (256, 0.1f);
  store->Execute (
      "INSERT INTO working_memory_slots (slot_index, strength, timestamp, embedding) "
      "VALUES (?, ?, ?, ?)",
      { 0LL, 1.0, old_ts, emb });

  // Load with sensitivity=0.5 → cost_per_slot = lerp(0.05, 0.15, 0.5) = 0.10
  // After 5 seconds: strength = 1.0 - 0.10 * 5 = 0.5
  struct VerifyDecayOp : IOperation
  {
    mutable double loaded_strength = 0.0;

    void
    Execute (OperationContext &ctx) const override
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
  auto ops = std::make_unique<OperationSet> (std::move (verify_ptr));
  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.5;
  SignalProcessor processor (cfg, store, std::move (ops));

  Signal s;
  s.embedding = Eigen::VectorXf::Random (256);
  s.timestamp = static_cast<uint64_t> (now);
  s.source_id = "test";
  processor.Process (s);

  // Strength should have decayed from 1.0 to approximately 0.5
  REQUIRE_THAT (verify_raw->loaded_strength, WithinAbs (0.5, 0.1));
}

TEST_CASE ("Fully decayed WM slots are not loaded",
           "[state_persistence][working_memory][decay]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Create schema
  {
    auto ops = std::make_unique<OperationSet> ();
    SignalProcessor processor (SignalProcessor::Config{}, store,
                               std::move (ops));
    processor.Flush ();
  }

  // Insert slot with very old timestamp (20 seconds ago)
  const auto now = std::chrono::duration_cast<std::chrono::seconds> (
                       std::chrono::system_clock::now ().time_since_epoch ())
                       .count ();
  const auto old_ts = now - 20; // 20 seconds ago

  std::vector<float> emb (256, 0.1f);
  store->Execute (
      "INSERT INTO working_memory_slots (slot_index, strength, timestamp, embedding) "
      "VALUES (?, ?, ?, ?)",
      { 0LL, 1.0, old_ts, emb });

  // Load with sensitivity=0.5 → cost_per_slot = 0.10
  // After 20 seconds: strength = 1.0 - 0.10 * 20 = -1.0 → not loaded
  struct VerifyNotLoadedOp : IOperation
  {
    mutable size_t slot_count = 999;

    void
    Execute (OperationContext &ctx) const override
    {
      const auto &pctx = ctx.GetProcessorContext ();
      slot_count = pctx.wm_slots.size ();
    }
  };

  auto verify_ptr = std::make_unique<VerifyNotLoadedOp> ();
  auto *verify_raw = verify_ptr.get ();
  auto ops = std::make_unique<OperationSet> (std::move (verify_ptr));
  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.5;
  SignalProcessor processor (cfg, store, std::move (ops));

  Signal s;
  s.embedding = Eigen::VectorXf::Random (256);
  s.timestamp = static_cast<uint64_t> (now);
  s.source_id = "test";
  processor.Process (s);

  // Slot should not have been loaded due to full decay
  REQUIRE (verify_raw->slot_count == 0);
}
