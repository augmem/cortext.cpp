// tests/operations_memory_strength.test.cpp
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

// Helper op to seed embeddings into the database before operations run.
class SeedEmbeddingsOp : public cortext::IOperation
{
public:
  explicit SeedEmbeddingsOp (std::vector<long long> ids) : ids_ (std::move (ids))
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto *store = ctx.GetStore ();
    std::vector<float> vec (kEmbeddingDim, 0.0f);
    vec[0] = 1.0f; // Simple unit vector
    for (const auto id : ids_)
      {
        store->Execute (
            "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, type, "
            "strength, use_frequency, stability, connectivity, drift_mag, "
            "influence, sustained_influence, contextual_gain, redundancy, "
            "pre_activation, lability_state, suppression_count) "
            "VALUES (?, ?, 'memory', 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
            "0.0, 0.0, 0.0, 0)",
            { id, vec });
      }
  }

private:
  std::vector<long long> ids_;
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

  // Default knobs: S=0.5, T=0.5
  SignalProcessor::Config cfg;
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

  auto rows = store->Execute ("SELECT embedding_id, use_frequency, strength "
                              "FROM embeddings WHERE "
                              "embedding_id = ?",
                              { 1LL });
  REQUIRE (rows.size () == 1);
  const auto id = std::any_cast<long long> (rows[0].at ("embedding_id"));
  const auto uf = std::any_cast<double> (rows[0].at ("use_frequency"));
  const auto strength = std::any_cast<double> (rows[0].at ("strength"));
  REQUIRE (id == 1LL);
  REQUIRE (uf > 0.0);
  REQUIRE (uf <= 1.0);
  REQUIRE (strength >= 1.0);
}

TEST_CASE ("Algorithm 14 decays and evicts below cutoff", "[op14]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  // Set S=0.0 to avoid reinforcement, T=0.0 to maximize decay per step.
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;

  // First, create the row with used=false (will create row then decay).
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
      auto sig = MakeSignal (4, static_cast<uint64_t> (i + 1));
      processor.Process (sig);
    }
  processor.Flush ();

  auto rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM embeddings WHERE embedding_id = ?",
      { 42LL });
  const auto cnt = std::any_cast<long long> (rows[0].at ("cnt"));
  REQUIRE (cnt == 0LL); // evicted
}

TEST_CASE (
    "Algorithm 18 increments counts and boosts strength with positive gain",
    "[op18][memory_strength]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  // F=1.0 to fully apply influence term; moderate S/T
  SignalProcessor::Config cfg;
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 7LL;
  ev.used = true;
  ev.contextual_gain = 1.0; // strong positive gain

  auto seed = std::make_unique<SeedEmbeddingsOp> (std::vector<long long>{ 7LL });
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

  auto rows = store->Execute (
      "SELECT mf.retrieved_count, mf.used_count, em.use_frequency, em.strength "
      "FROM embeddings em "
      "JOIN memory_feedback mf ON em.embedding_id = mf.embedding_id "
      "WHERE em.embedding_id = ?",
      { 7LL });
  REQUIRE (rows.size () == 1);
  const auto retrieved
      = std::any_cast<long long> (rows[0].at ("retrieved_count"));
  const auto used = std::any_cast<long long> (rows[0].at ("used_count"));
  const auto strength = std::any_cast<double> (rows[0].at ("strength"));
  REQUIRE (retrieved == 1LL);
  REQUIRE (used == 1LL);
  REQUIRE (
      strength
      > 1.0); // should increase from baseline with reinforcement/influence
}

TEST_CASE (
    "Algorithm 18 negative gain yields small increase; counts increment",
    "[op18][memory_strength]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  // F=1.0 ensures influence term is fully weighted; with negative gain the
  // map01(influence_factor=-1) => 0, so only reinforcement remains.
  SignalProcessor::Config cfg;
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 8LL;
  ev.used = true;
  ev.contextual_gain = -1.0;

  auto seed = std::make_unique<SeedEmbeddingsOp> (std::vector<long long>{ 8LL });
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

  auto rows = store->Execute (
      "SELECT mf.retrieved_count, mf.used_count, em.use_frequency, em.strength "
      "FROM embeddings em "
      "JOIN memory_feedback mf ON em.embedding_id = mf.embedding_id "
      "WHERE em.embedding_id = ?",
      { 8LL });
  REQUIRE (rows.size () == 1);
  const auto retrieved
      = std::any_cast<long long> (rows[0].at ("retrieved_count"));
  const auto used = std::any_cast<long long> (rows[0].at ("used_count"));
  const auto strength = std::any_cast<double> (rows[0].at ("strength"));
  REQUIRE (retrieved == 1LL);
  REQUIRE (used == 1LL);
  REQUIRE (strength > 1.0);
  REQUIRE (strength < 1.2);
}

TEST_CASE ("Algorithm 18 counts retrieval when not used",
           "[op18][memory_strength]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 9LL;
  ev.used = false; // retrieved but not used
  ev.contextual_gain = 0.0;

  auto set_events = std::make_unique<SetUsageEventsOp> (
      std::vector<OperationContext::MemoryUsageEvent>{ ev });
  auto update_strength
      = std::make_unique<cortext::operations::UpdateMemoryStrength> ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (set_events), std::move (update_strength));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto signal = MakeSignal (4, 1);
  processor.Process (signal);
  processor.Flush ();

  auto rows = store->Execute ("SELECT retrieved_count, used_count FROM "
                              "memory_feedback WHERE embedding_id = ?",
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
  cfg.focus = 0.0;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;

  // Create processor with a fresh op set for each signal
  // Use used=true to ensure row is created (INSERT requires used || contextual_gain)
  auto make_ops = [] () {
    OperationContext::MemoryUsageEvent ev{};
    ev.embedding_id = 100LL;
    ev.used = true;
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

  // Verify initial strength
  auto rows = store->Execute (
      "SELECT strength, last_access FROM embeddings "
      "WHERE embedding_id = ?",
      { 100LL });
  REQUIRE (rows.size () == 1);
  const double initial_strength
      = std::any_cast<double> (rows[0].at ("strength"));
  REQUIRE (initial_strength >= 0.99);
  REQUIRE (initial_strength <= 1.01);

  // Second signal at t=120000ms (120 seconds = one half-life for T=0)
  {
    cortext::SignalProcessor processor (cfg, store, make_ops ());
    auto signal = MakeSignal (4, 120000);
    processor.Process (signal);
    processor.Flush ();
  }

  // Verify decayed strength: should be approximately 0.5 (one half-life)
  rows = store->Execute (
      "SELECT strength, last_access FROM embeddings "
      "WHERE embedding_id = ?",
      { 100LL });
  REQUIRE (rows.size () == 1);
  const double decayed_strength
      = std::any_cast<double> (rows[0].at ("strength"));
  // decay_factor = exp(-ln(2)/120 * 120) = exp(-ln(2)) = 0.5
  // Allow 10% tolerance for floating point and any minor numerical effects
  REQUIRE (decayed_strength >= 0.45);
  REQUIRE (decayed_strength <= 0.55);
}

TEST_CASE ("Algorithm 14 no decay when delta_t is zero", "[op14][decay]")
{
  // When two updates happen at the same timestamp, decay_factor = exp(0) = 1.0

  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  // T=0 gives fastest decay (half_life=120s); S=0 and F=0 remove reinforcement
  SignalProcessor::Config cfg;
  cfg.focus = 0.0;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;

  // Use used=true to ensure row is created
  auto make_ops = [] () {
    OperationContext::MemoryUsageEvent ev{};
    ev.embedding_id = 200LL;
    ev.used = true;
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

  auto rows = store->Execute ("SELECT strength FROM embeddings "
                              "WHERE embedding_id = ?",
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

  rows = store->Execute ("SELECT strength FROM embeddings "
                         "WHERE embedding_id = ?",
                         { 200LL });
  REQUIRE (rows.size () == 1);
  const double strength_after_second
      = std::any_cast<double> (rows[0].at ("strength"));

  // With zero time delta, decay_factor = 1.0, so strength should be unchanged
  // (within floating point tolerance)
  REQUIRE (std::abs (strength_after_second - strength_after_first) < 0.01);
}

TEST_CASE ("Algorithm 14 initializes last_access on INSERT",
           "[op14][decay]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
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

  auto rows
      = store->Execute ("SELECT last_access FROM embeddings "
                        "WHERE embedding_id = ?",
                        { 300LL });
  REQUIRE (rows.size () == 1);
  const auto last_ts
      = std::any_cast<long long> (rows[0].at ("last_access"));
  REQUIRE (last_ts == 5000LL);
}
