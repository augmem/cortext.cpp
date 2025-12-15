// tests/operations_memory_strength.test.cpp
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/memory_strength.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using cortext::BufferedWriteInstruction;
using cortext::OperationContext;
using cortext::Signal;
using cortext::SignalProcessor;

namespace
{

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
  Execute (OperationContext &ctx) const override
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

TEST_CASE ("Algorithm 14 creates and updates embeddings_meta row", "[op14]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  // Default knobs: S=0.5, T=0.5
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Usage: id=1 used=true
  auto set_events = std::make_unique<SetUsageEventsOp> (
      std::vector<OperationContext::MemoryUsageEvent>{
          { 1LL, true },
      });
  auto update_strength
      = std::make_unique<cortext::operations::UpdateMemoryStrength> ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (set_events), std::move (update_strength));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  auto signal = MakeSignal (4, 100);
  processor.Process (signal);
  processor.Flush ();

  auto rows = store->Execute ("SELECT embedding_id, use_frequency, strength "
                              "FROM embeddings_meta WHERE "
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
      "SELECT COUNT(*) AS cnt FROM embeddings_meta WHERE embedding_id = ?",
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

  auto rows = store->Execute (
      "SELECT mf.retrieved_count, mf.used_count, em.use_frequency, em.strength "
      "FROM embeddings_meta em "
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

  auto rows = store->Execute (
      "SELECT mf.retrieved_count, mf.used_count, em.use_frequency, em.strength "
      "FROM embeddings_meta em "
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
