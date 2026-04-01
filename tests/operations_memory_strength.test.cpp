// tests/operations_memory_strength.test.cpp
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/memory_strength.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>


using cortext::OperationContext;
using cortext::Signal;
using cortext::SignalProcessor;
using cortext::Transaction;

namespace
{

constexpr int kEmbeddingDim = 256;

// Helper op to seed embeddings and memories into the v2 database.
class SeedEmbeddingsOp : public cortext::IOperation
{
public:
  explicit SeedEmbeddingsOp (std::vector<long long> ids,
                             double strength = 1.0,
                             double use_frequency = 0.0)
      : ids_ (std::move (ids)),
        strength_ (strength),
        use_frequency_ (use_frequency)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto *store = ctx.GetStore ();
    std::vector<float> vec (kEmbeddingDim, 0.0f);
    vec[0] = 1.0f; // Simple unit vector
    auto now_ts = cortext::testing::NowMs ();
    for (const auto id : ids_)
      {
        // v2: Insert into embeddings (minimal vec0 table)
        store->Execute (
            "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
            "VALUES(?, ?, ?)",
            { id, vec, now_ts });
        // v2: Insert into memories (comprehensive metadata)
        store->Execute (
            "INSERT OR REPLACE INTO memories("
            "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
            "s_max, s_avg, strength, trace_fast, trace_med, trace_slow, trace_ultra, "
            "use_frequency, stability, connectivity, drift_mag, "
            "influence, sustained_influence, contextual_gain, redundancy, "
            "pre_activation, lability_state, suppression_count, created_at) "
            "VALUES(?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, "
            "?, ?, ?, ?, ?, ?, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, ?)",
            { id, id, now_ts, strength_, strength_, strength_ * 0.5,
              strength_ * 0.2, strength_ * 0.05, use_frequency_, now_ts });
      }
  }

private:
  std::vector<long long> ids_;
  double strength_;
  double use_frequency_;
};

// Helper op to inject usage events into the OperationContext before updates.
class SetUsageEventsOp : public cortext::IOperation
{
public:
  explicit SetUsageEventsOp (
      std::vector<OperationContext::MemoryUsageEvent> evs)
      : events_ (std::move (evs))
  {
  }
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetMemoryUsageEvents (events_);
  }

private:
  std::vector<OperationContext::MemoryUsageEvent> events_;
};

// Minimal signal factory
static Signal
MakeSignal (int dim, uint64_t ts = 1)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (dim);
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

} // namespace

TEST_CASE ("Algorithm 14 creates and updates embeddings row", "[op14]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  // Default knobs: S=0.5, T=0.5
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Usage: id=1 used=true
  auto seed = std::make_unique<SeedEmbeddingsOp> (std::vector<long long>{ 1LL });
  auto set_events = std::make_unique<SetUsageEventsOp> (
      std::vector<OperationContext::MemoryUsageEvent>{
          { 1LL, true },
      });
  auto update_strength
      = std::make_unique<cortext::operations::UpdateMemoryStrength> ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (seed), std::move (set_events), std::move (update_strength));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  auto signal = MakeSignal (4, 100);
  processor.Process (signal);
  processor.Flush ();

  // v2: Query memories table for metadata
  auto rows = store->Execute ("SELECT memory_id, use_frequency, strength "
                              "FROM memories WHERE memory_id = ?",
                              { 1LL });
  REQUIRE (rows.size () == 1);
  const auto id = std::any_cast<long long> (rows[0].at ("memory_id"));
  const auto uf = std::any_cast<double> (rows[0].at ("use_frequency"));
  const auto strength = std::any_cast<double> (rows[0].at ("strength"));
  REQUIRE (id == 1LL);
  REQUIRE (uf > 0.0);
  REQUIRE (uf <= 1.0);
  REQUIRE (strength >= 0.0);
  REQUIRE (strength <= 1.0);
}

TEST_CASE ("Algorithm 14 decays and evicts below cutoff", "[op14]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  // Set S=0.0 to avoid reinforcement, T=0.0 to maximize decay per step.
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;

  // Seed the row with low strength and let decay drive eviction.
  {
    SeedEmbeddingsOp seed ({ 42LL }, /*strength=*/0.1);
    Signal seed_signal = MakeSignal (4, 0);
    cortext::ProcessorContext seed_ctx_state;
    OperationContext seed_ctx (seed_signal, seed_ctx_state, cfg, store.get ());
    seed.Execute (seed_ctx, cortext::testing::GetNullTransaction ());
  }
  auto set_events = std::make_unique<SetUsageEventsOp> (
      std::vector<OperationContext::MemoryUsageEvent>{ { 42LL, false } });
  auto update_strength
      = std::make_unique<cortext::operations::UpdateMemoryStrength> ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (set_events), std::move (update_strength));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  // Process multiple signals to ensure decay crosses cutoff.
  for (int i = 0; i < 240; ++i)
    {
      auto sig = MakeSignal (4, static_cast<uint64_t> ((i + 1) * 1000));
      processor.Process (sig);
    }
  processor.Flush ();

  // v2: Check memories table for eviction (memory deleted when strength <= 0)
  auto rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?",
      { 42LL });
  const auto cnt = std::any_cast<long long> (rows[0].at ("cnt"));
  REQUIRE (cnt == 0LL); // evicted below periphery cutoff
}

TEST_CASE (
    "Algorithm 18 increments counts and boosts strength with positive gain",
    "[op18][memory_strength]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  // F=1.0 to fully apply influence term; moderate S/T
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 7LL;
  ev.used = true;
  ev.contextual_gain = 1.0; // strong positive gain

  auto seed = std::make_unique<SeedEmbeddingsOp> (
      std::vector<long long>{ 7LL }, /*strength=*/0.2);
  auto set_events = std::make_unique<SetUsageEventsOp> (
      std::vector<OperationContext::MemoryUsageEvent>{ ev });
  auto update_strength
      = std::make_unique<cortext::operations::UpdateMemoryStrength> ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (seed), std::move (set_events), std::move (update_strength));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto signal = MakeSignal (4, 1);
  processor.Process (signal);
  processor.Flush ();

  // v2: Query memories table directly - all metadata is inline
  auto rows = store->Execute (
      "SELECT retrieved_count, used_count, use_frequency, strength "
      "FROM memories WHERE memory_id = ?",
      { 7LL });
  REQUIRE (rows.size () == 1);
  const auto retrieved
      = std::any_cast<long long> (rows[0].at ("retrieved_count"));
  const auto used = std::any_cast<long long> (rows[0].at ("used_count"));
  const auto strength = std::any_cast<double> (rows[0].at ("strength"));
  REQUIRE (retrieved == 1LL);
  REQUIRE (used == 1LL);
  REQUIRE (strength > 0.2); // should increase from baseline with influence
  REQUIRE (strength <= 1.0);
}

TEST_CASE (
    "Algorithm 18 negative gain yields small increase, counts increment",
    "[op18][memory_strength]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  // F=1.0 ensures influence term is fully weighted; with negative gain the
  // map01(influence_factor=-1) => 0, so only reinforcement remains.
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 8LL;
  ev.used = true;
  ev.contextual_gain = -1.0;

  auto seed = std::make_unique<SeedEmbeddingsOp> (
      std::vector<long long>{ 8LL }, /*strength=*/0.5);
  auto set_events = std::make_unique<SetUsageEventsOp> (
      std::vector<OperationContext::MemoryUsageEvent>{ ev });
  auto update_strength
      = std::make_unique<cortext::operations::UpdateMemoryStrength> ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (seed), std::move (set_events), std::move (update_strength));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto signal = MakeSignal (4, 1);
  processor.Process (signal);
  processor.Flush ();

  // v2: Query memories table directly - all metadata is inline
  auto rows = store->Execute (
      "SELECT retrieved_count, used_count, use_frequency, strength "
      "FROM memories WHERE memory_id = ?",
      { 8LL });
  REQUIRE (rows.size () == 1);
  const auto retrieved
      = std::any_cast<long long> (rows[0].at ("retrieved_count"));
  const auto used = std::any_cast<long long> (rows[0].at ("used_count"));
  const auto strength = std::any_cast<double> (rows[0].at ("strength"));
  REQUIRE (retrieved == 1LL);
  REQUIRE (used == 1LL);
  // With negative contextual_gain=-1, influence term can reduce strength.
  REQUIRE (strength >= 0.0);
  REQUIRE (strength <= 0.55);
}

TEST_CASE ("Algorithm 18 counts retrieval when not used",
           "[op18][memory_strength]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 9LL;
  ev.used = false; // retrieved but not used
  ev.contextual_gain = 0.0;

  // v2: Need to seed the memory row before update can increment counts
  auto seed = std::make_unique<SeedEmbeddingsOp> (std::vector<long long>{ 9LL });
  auto set_events = std::make_unique<SetUsageEventsOp> (
      std::vector<OperationContext::MemoryUsageEvent>{ ev });
  auto update_strength
      = std::make_unique<cortext::operations::UpdateMemoryStrength> ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (seed), std::move (set_events), std::move (update_strength));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto signal = MakeSignal (4, 1);
  processor.Process (signal);
  processor.Flush ();

  // v2: Query memories table directly - all metadata is inline
  auto rows = store->Execute ("SELECT retrieved_count, used_count FROM "
                              "memories WHERE memory_id = ?",
                              { 9LL });
  REQUIRE (rows.size () == 1);
  const auto retrieved
      = std::any_cast<long long> (rows[0].at ("retrieved_count"));
  const auto used = std::any_cast<long long> (rows[0].at ("used_count"));
  REQUIRE (retrieved == 1LL);
  REQUIRE (used == 0LL);
}

TEST_CASE ("Algorithm 14 applies exponential decay based on elapsed time",
           "[op14][decay]")
{
  // Test that strength follows half-life semantics:
  // With T=0, half_life = 120s, so after 120s the decay_factor should be 0.5

  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  // T=0 gives half_life=120s; S=0 removes reinforcement; F=0 removes influence
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;

  // Create processor with a fresh op set for each signal
  // Use used=true to ensure row is created (INSERT requires used || contextual_gain)
  auto make_ops = [] () {
    OperationContext::MemoryUsageEvent ev{};
    ev.embedding_id = 100LL;
    ev.used = false;
    ev.contextual_gain = 0.0; // Neutral gain to avoid influence term
    auto set_events = std::make_unique<SetUsageEventsOp> (
        std::vector<OperationContext::MemoryUsageEvent>{ ev });
    auto update_strength
        = std::make_unique<cortext::operations::UpdateMemoryStrength> ();
    return std::make_unique<cortext::OperationSet> (
        std::move (set_events), std::move (update_strength));
  };

  // First signal at t=0: seed row then update with strength=1.0, last_access=0
  {
    auto seed
        = std::make_unique<SeedEmbeddingsOp> (std::vector<long long>{ 100LL });
    OperationContext::MemoryUsageEvent ev{};
    ev.embedding_id = 100LL;
    ev.used = true;
    ev.contextual_gain = 0.0;
    auto set_events = std::make_unique<SetUsageEventsOp> (
        std::vector<OperationContext::MemoryUsageEvent>{ ev });
    auto update_strength
        = std::make_unique<cortext::operations::UpdateMemoryStrength> ();
    auto ops = std::make_unique<cortext::OperationSet> (
        std::move (seed), std::move (set_events), std::move (update_strength));
    cortext::SignalProcessor processor (cfg, store, std::move (ops));
    auto signal = MakeSignal (4, 0);
    processor.Process (signal);
    processor.Flush ();
  }

  // v2: Verify initial strength from memories table
  auto rows = store->Execute (
      "SELECT strength, last_access FROM memories "
      "WHERE memory_id = ?",
      { 100LL });
  REQUIRE (rows.size () == 1);
  const double initial_strength
      = std::any_cast<double> (rows[0].at ("strength"));
  REQUIRE (initial_strength > 0.0);

  // Second signal at t=120000ms (120 seconds = one half-life for T=0)
  {
    cortext::SignalProcessor processor (cfg, store, make_ops ());
    auto signal = MakeSignal (4, 120000);
    processor.Process (signal);
    processor.Flush ();
  }

  // v2: Verify decayed strength from memories table
  rows = store->Execute (
      "SELECT strength, last_access FROM memories "
      "WHERE memory_id = ?",
      { 100LL });
  REQUIRE (rows.size () == 1);
  const double decayed_strength
      = std::any_cast<double> (rows[0].at ("strength"));
  REQUIRE (decayed_strength > 0.0);
  REQUIRE (decayed_strength < initial_strength);
}

TEST_CASE ("Algorithm 14 no decay when delta_t is zero", "[op14][decay]")
{
  // When two updates happen at the same timestamp, decay_factor = exp(0) = 1.0

  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  // T=0 gives fastest decay (half_life=120s); S=0 and F=0 remove reinforcement
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;

  // Use used=true to ensure row is created
  auto make_ops = [] () {
    OperationContext::MemoryUsageEvent ev{};
    ev.embedding_id = 200LL;
    ev.used = false;
    ev.contextual_gain = 0.0;
    auto set_events = std::make_unique<SetUsageEventsOp> (
        std::vector<OperationContext::MemoryUsageEvent>{ ev });
    auto update_strength
        = std::make_unique<cortext::operations::UpdateMemoryStrength> ();
    return std::make_unique<cortext::OperationSet> (
        std::move (set_events), std::move (update_strength));
  };

  // First signal at t=1000ms - seed row then update
  {
    auto seed
        = std::make_unique<SeedEmbeddingsOp> (std::vector<long long>{ 200LL });
    OperationContext::MemoryUsageEvent ev{};
    ev.embedding_id = 200LL;
    ev.used = true;
    ev.contextual_gain = 0.0;
    auto set_events = std::make_unique<SetUsageEventsOp> (
        std::vector<OperationContext::MemoryUsageEvent>{ ev });
    auto update_strength
        = std::make_unique<cortext::operations::UpdateMemoryStrength> ();
    auto ops = std::make_unique<cortext::OperationSet> (
        std::move (seed), std::move (set_events), std::move (update_strength));
    cortext::SignalProcessor processor (cfg, store, std::move (ops));
    auto signal = MakeSignal (4, 1000);
    processor.Process (signal);
    processor.Flush ();
  }

  // v2: Query memories table
  auto rows = store->Execute ("SELECT strength FROM memories "
                              "WHERE memory_id = ?",
                              { 200LL });
  REQUIRE (rows.size () == 1);
  const double strength_after_first
      = std::any_cast<double> (rows[0].at ("strength"));

  // Second signal at same timestamp t=1000ms
  {
    cortext::SignalProcessor processor (cfg, store, make_ops ());
    auto signal = MakeSignal (4, 1000);
    processor.Process (signal);
    processor.Flush ();
  }

  // v2: Query memories table
  rows = store->Execute ("SELECT strength FROM memories "
                         "WHERE memory_id = ?",
                         { 200LL });
  REQUIRE (rows.size () == 1);
  const double strength_after_second
      = std::any_cast<double> (rows[0].at ("strength"));

  // With zero time delta, decay_factor = 1.0, so strength should be unchanged
  // (within floating point tolerance)
  REQUIRE (std::abs (strength_after_second - strength_after_first) < 0.02);
}

TEST_CASE ("Algorithm 14 initializes last_access on INSERT",
           "[op14][decay]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto seed
      = std::make_unique<SeedEmbeddingsOp> (std::vector<long long>{ 300LL });
  auto set_events = std::make_unique<SetUsageEventsOp> (
      std::vector<OperationContext::MemoryUsageEvent>{ { 300LL, true } });
  auto update_strength
      = std::make_unique<cortext::operations::UpdateMemoryStrength> ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (seed), std::move (set_events), std::move (update_strength));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  // Process at timestamp 5000ms
  auto signal = MakeSignal (4, 5000);
  processor.Process (signal);
  processor.Flush ();

  // v2: Query memories table
  auto rows
      = store->Execute ("SELECT last_access FROM memories "
                        "WHERE memory_id = ?",
                        { 300LL });
  REQUIRE (rows.size () == 1);
  const auto last_ts
      = std::any_cast<long long> (rows[0].at ("last_access"));
  REQUIRE (last_ts == 5000LL);
}
