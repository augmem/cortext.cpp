// tests/operations_memory_strength.test.cpp
#include "test_helpers.hpp"
#include "../src/operations/association_fanout_cache_internal.hpp"
#include "../src/operations/execution_cache_sidecar_internal.hpp"
#include "../src/operations/eviction_policy_override.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/memory_strength.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>


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

std::vector<unsigned char>
PutObjstoreBlob (cortext::Store &store,
                 const std::vector<unsigned char> &payload)
{
  auto rows = store.Execute ("SELECT objstore_put(?1) AS id", { payload });
  REQUIRE (rows.size () == 1);
  REQUIRE (rows[0].count ("id") == 1);
  auto blob_id = cortext::store::BlobFromAny (rows[0].at ("id"));
  REQUIRE_FALSE (blob_id.empty ());
  return blob_id;
}

long long
CountObjstoreBlob (cortext::Store &store,
                   const std::vector<unsigned char> &blob_id)
{
  auto rows = store.Execute (
      "SELECT COUNT(*) AS cnt FROM objstore_data WHERE id = ?",
      { blob_id });
  REQUIRE (rows.size () == 1);
  return std::any_cast<long long> (rows[0].at ("cnt"));
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
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
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
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::move (set_events), std::move (update_strength));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  cortext::operations::eviction::EvictionPolicyOverride override;
  override.consolidation_gate_enabled = false;
  cortext::operations::eviction::ScopedEvictionPolicyOverride gate_override (
      override);

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
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
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

TEST_CASE ("Algorithm 18 updates by memory_id when usage events provide it",
           "[op18][memory_strength]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> vec (kEmbeddingDim, 0.0f);
  vec[0] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (*store, 700LL, vec, 1);
  cortext::testing::SeedMemoryV2 (*store, 70LL, 700LL, "test", "LONG_TERM",
                                  0.2, 1);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  cortext::ProcessorContext pctx;
  cortext::ProcessorContext::RetrievalSurfaceEntry entry;
  entry.memory_id = 70LL;
  entry.embedding_id = 700LL;
  entry.created_at = 1LL;
  entry.start_ts = 1LL;
  entry.kind = "LONG_TERM";
  entry.source_id = "test";
  entry.modality = "text";
  entry.embedding = Eigen::Map<Eigen::VectorXf> (vec.data (), vec.size ());
  pctx.UpsertRetrievalSurface (std::move (entry));

  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 701LL;
  ev.memory_id = 70LL;
  ev.used = true;
  ev.contextual_gain = 1.0;

  auto signal = MakeSignal (4, 100);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetMemoryUsageEvents ({ ev });

  cortext::operations::UpdateMemoryStrength update_strength;
  auto tx = store->Begin ();
  update_strength.Execute (ctx, *tx);
  tx->Commit ();

  auto rows = store->Execute (
      "SELECT retrieved_count, used_count, strength "
      "FROM memories WHERE memory_id = ?",
      { 70LL });
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<long long> (rows[0].at ("retrieved_count")) == 1LL);
  REQUIRE (std::any_cast<long long> (rows[0].at ("used_count")) == 1LL);
  REQUIRE (std::any_cast<double> (rows[0].at ("strength")) >= 0.0);
  REQUIRE (std::any_cast<double> (rows[0].at ("strength")) <= 1.0);

  const auto cache_it = pctx.retrieval_surface_index.find (70LL);
  REQUIRE (cache_it != pctx.retrieval_surface_index.end ());
  const auto &cached = pctx.retrieval_surface_cache[cache_it->second];
  REQUIRE (cached.retrieved_count == 1LL);
  REQUIRE (cached.used_count == 1LL);
  REQUIRE (cached.last_access == 100LL);
}

TEST_CASE ("Algorithm 18 tolerates malformed persisted memory strength row",
           "[op18][memory_strength][robustness]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> vec (kEmbeddingDim, 0.0f);
  vec[0] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (*store, 901LL, vec, 1000);
  store->Execute (
      "INSERT INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, use_frequency, contextual_gain, "
      "retrieved_count, used_count, last_access, created_at, "
      "trace_fast, trace_med, trace_slow, trace_ultra) "
      "VALUES(901, 901, 'test', 'LONG_TERM', 1000, 1, 'text', "
      "0.5, 0.5, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      { std::string ("bad-strength"), std::string ("bad-use"),
        std::string ("bad-gain"), std::string ("bad-retrieved"),
        std::string ("bad-used"), std::string ("bad-access"),
        std::string ("bad-created"), std::string ("bad-fast"),
        std::string ("bad-med"), std::string ("bad-slow"),
        std::string ("bad-ultra") });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  cortext::ProcessorContext pctx;
  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 901LL;
  ev.memory_id = 901LL;
  ev.used = true;
  ev.contextual_gain = 1.0;

  auto signal = MakeSignal (4, 5000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetMemoryUsageEvents ({ ev });

  cortext::operations::UpdateMemoryStrength update_strength;
  auto tx = store->Begin ();
  tx->Execute (
      "INSERT INTO signals(signal_id, memory_id, source_id, embedding_id, "
      "timestamp, modality, created_at) "
      "VALUES(9901, 901, 'test', 901, 5000, 'text', 5000)",
      {});
  REQUIRE_NOTHROW (update_strength.Execute (ctx, *tx));
  tx->Commit ();

  auto signal_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM signals WHERE signal_id = 9901", {});
  REQUIRE (std::any_cast<long long> (signal_rows[0].at ("cnt")) == 1LL);

  auto memory_rows = store->Execute (
      "SELECT retrieved_count, used_count, last_access "
      "FROM memories WHERE memory_id = 901",
      {});
  REQUIRE (memory_rows.size () == 1);
  REQUIRE (std::any_cast<long long> (memory_rows[0].at ("retrieved_count"))
           == 1LL);
  REQUIRE (std::any_cast<long long> (memory_rows[0].at ("used_count"))
           == 1LL);
  REQUIRE (std::any_cast<long long> (memory_rows[0].at ("last_access"))
           == 5000LL);
}

TEST_CASE ("Algorithm 18 falls back from malformed last_access to created_at",
           "[op18][memory_strength][robustness]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> vec (kEmbeddingDim, 0.0f);
  vec[0] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (*store, 902LL, vec, 4999);
  store->Execute (
      "INSERT INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, use_frequency, contextual_gain, "
      "retrieved_count, used_count, last_access, created_at, "
      "trace_fast, trace_med, trace_slow, trace_ultra) "
      "VALUES(902, 902, 'test', 'LONG_TERM', 4999, 1, 'text', "
      "1.0, 1.0, 1.0, 0.0, 0.0, 0, 0, ?, 4999, 1.0, 0.0, 0.0, 0.0)",
      { std::string ("bad-access") });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;

  cortext::operations::eviction::EvictionPolicyOverride override;
  override.trace_count = 1;
  override.coupling_enabled = false;
  override.reinforcement
      = cortext::operations::eviction::ReinforcementStrength::Off;
  override.half_life = 1.0;
  cortext::operations::eviction::ScopedEvictionPolicyOverride scoped_override (
      override);

  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 902LL;
  ev.memory_id = 902LL;
  ev.used = false;

  cortext::ProcessorContext pctx;
  auto signal = MakeSignal (4, 5000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetMemoryUsageEvents ({ ev });

  cortext::operations::UpdateMemoryStrength update_strength;
  auto tx = store->Begin ();
  REQUIRE_NOTHROW (update_strength.Execute (ctx, *tx));
  tx->Commit ();

  auto rows = store->Execute (
      "SELECT strength, trace_fast, last_access FROM memories "
      "WHERE memory_id = 902",
      {});
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<double> (rows[0].at ("strength")) > 0.99);
  REQUIRE (std::any_cast<double> (rows[0].at ("trace_fast")) > 0.99);
  REQUIRE (std::any_cast<long long> (rows[0].at ("last_access")) == 5000LL);
}

TEST_CASE ("Algorithm 18 writes an eviction audit row before deleting long-term memory",
           "[op18][memory_strength][eviction_audit]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;

  {
    SeedEmbeddingsOp seed ({ 77LL }, /*strength=*/0.1);
    Signal seed_signal = MakeSignal (4, 0);
    cortext::ProcessorContext seed_ctx_state;
    OperationContext seed_ctx (seed_signal, seed_ctx_state, cfg, store.get ());
    seed.Execute (seed_ctx, cortext::testing::GetNullTransaction ());
  }

  auto set_events = std::make_unique<SetUsageEventsOp> (
      std::vector<OperationContext::MemoryUsageEvent>{ { 77LL, false } });
  auto update_strength
      = std::make_unique<cortext::operations::UpdateMemoryStrength> ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::move (set_events), std::move (update_strength));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  cortext::operations::eviction::EvictionPolicyOverride override;
  override.consolidation_gate_enabled = false;
  cortext::operations::eviction::ScopedEvictionPolicyOverride gate_override (
      override);
  for (int i = 0; i < 240; ++i)
    {
      auto sig = MakeSignal (4, static_cast<uint64_t> ((i + 1) * 1000));
      processor.Process (sig);
    }
  processor.Flush ();

  auto rows = store->Execute (
      "SELECT memory_id, kind, source_id, eviction_reason, evicted_at "
      "FROM memory_evictions WHERE memory_id = ?",
      { 77LL });
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<long long> (rows[0].at ("memory_id")) == 77LL);
  REQUIRE (std::any_cast<std::string> (rows[0].at ("kind")) == "LONG_TERM");
  REQUIRE (std::any_cast<std::string> (rows[0].at ("source_id")) == "test");
  REQUIRE (std::any_cast<std::string> (rows[0].at ("eviction_reason"))
           == "periphery_cutoff");
  REQUIRE (std::any_cast<long long> (rows[0].at ("evicted_at")) > 0LL);
}

TEST_CASE ("Algorithm 18 does not evict before storage budget is reached",
           "[op18][memory_strength][storage_gate]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;

  {
    SeedEmbeddingsOp seed ({ 88LL }, /*strength=*/0.1);
    Signal seed_signal = MakeSignal (4, 0);
    cortext::ProcessorContext seed_ctx_state;
    OperationContext seed_ctx (seed_signal, seed_ctx_state, cfg, store.get ());
    seed.Execute (seed_ctx, cortext::testing::GetNullTransaction ());
  }

  auto set_events = std::make_unique<SetUsageEventsOp> (
      std::vector<OperationContext::MemoryUsageEvent>{ { 88LL, false } });
  auto update_strength
      = std::make_unique<cortext::operations::UpdateMemoryStrength> ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::move (set_events), std::move (update_strength));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  cortext::operations::eviction::EvictionPolicyOverride override;
  override.consolidation_gate_enabled = false;
  override.storage_gate_enabled = true;
  override.min_storage_bytes = 1LL << 30;
  cortext::operations::eviction::ScopedEvictionPolicyOverride storage_override (
      override);

  for (int i = 0; i < 240; ++i)
    {
      auto sig = MakeSignal (4, static_cast<uint64_t> ((i + 1) * 1000));
      processor.Process (sig);
    }
  processor.Flush ();

  auto rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?",
      { 88LL });
  REQUIRE (std::any_cast<long long> (rows[0].at ("cnt")) == 1LL);

  auto eviction_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memory_evictions WHERE memory_id = ?",
      { 88LL });
  REQUIRE (std::any_cast<long long> (eviction_rows[0].at ("cnt")) == 0LL);
}

TEST_CASE ("Algorithm 18 evicts once storage budget is below threshold",
           "[op18][memory_strength][storage_gate]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;

  {
    SeedEmbeddingsOp seed ({ 89LL }, /*strength=*/0.1);
    Signal seed_signal = MakeSignal (4, 0);
    cortext::ProcessorContext seed_ctx_state;
    OperationContext seed_ctx (seed_signal, seed_ctx_state, cfg, store.get ());
    seed.Execute (seed_ctx, cortext::testing::GetNullTransaction ());
  }

  auto set_events = std::make_unique<SetUsageEventsOp> (
      std::vector<OperationContext::MemoryUsageEvent>{ { 89LL, false } });
  auto update_strength
      = std::make_unique<cortext::operations::UpdateMemoryStrength> ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::move (set_events), std::move (update_strength));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  cortext::operations::eviction::EvictionPolicyOverride override;
  override.consolidation_gate_enabled = false;
  override.storage_gate_enabled = true;
  override.min_storage_bytes = 0;
  cortext::operations::eviction::ScopedEvictionPolicyOverride storage_override (
      override);

  for (int i = 0; i < 240; ++i)
    {
      auto sig = MakeSignal (4, static_cast<uint64_t> ((i + 1) * 1000));
      processor.Process (sig);
    }
  processor.Flush ();

  auto rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?",
      { 89LL });
  REQUIRE (std::any_cast<long long> (rows[0].at ("cnt")) == 0LL);
}

TEST_CASE (
    "Algorithm 18 keeps shared embeddings referenced by surviving memories",
    "[op18][memory_strength][eviction]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> vec (kEmbeddingDim, 0.0f);
  vec[0] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (*store, 9900LL, vec, 1000);
  cortext::testing::SeedMemoryV2 (*store, 9901LL, 9900LL, "weak",
                                  "LONG_TERM", 0.1, 1000);
  cortext::testing::SeedMemoryV2 (*store, 9902LL, 9900LL, "strong",
                                  "LONG_TERM", 0.9, 1000);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  cortext::operations::eviction::EvictionPolicyOverride override;
  override.consolidation_gate_enabled = false;
  override.periphery_cutoff = 0.5;
  cortext::operations::eviction::ScopedEvictionPolicyOverride scoped_override (
      override);

  cortext::ProcessorContext pctx;
  auto signal = MakeSignal (4, 2000);
  OperationContext ctx (signal, pctx, cfg, store.get ());

  cortext::operations::UpdateMemoryStrength update_strength;
  auto tx = store->Begin ();
  update_strength.Execute (ctx, *tx);
  tx->Commit ();

  auto weak_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?",
      { 9901LL });
  REQUIRE (std::any_cast<long long> (weak_rows[0].at ("cnt")) == 0LL);

  auto strong_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?",
      { 9902LL });
  REQUIRE (std::any_cast<long long> (strong_rows[0].at ("cnt")) == 1LL);

  auto embedding_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM embeddings WHERE embedding_id = ?",
      { 9900LL });
  REQUIRE (std::any_cast<long long> (embedding_rows[0].at ("cnt")) == 1LL);
}

TEST_CASE ("Algorithm 18 eviction removes stale supersession eligibility",
           "[op18][memory_strength][eviction][supersession][regression]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf replacement_embedding
      = Eigen::VectorXf::Unit (kEmbeddingDim, 0);
  const Eigen::VectorXf target_embedding
      = Eigen::VectorXf::Unit (kEmbeddingDim, 1);
  cortext::testing::SeedEmbeddingV2 (*store, 9900, replacement_embedding,
                                    1000);
  cortext::testing::SeedEmbeddingV2 (*store, 9901, target_embedding, 900);
  cortext::testing::SeedMemoryV2 (*store, 100, 9900, "opaque/replacement",
                                  "LONG_TERM", 0.1, 1000);
  cortext::testing::SeedMemoryV2 (*store, 200, 9901, "opaque/target",
                                  "LONG_TERM", 0.9, 900);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, "
      "edge_type, weight, last_reinforced) "
      "VALUES(100, 200, 'supersedes', 1.0, 1000)");

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cortext::ProcessorContext pctx;
  pctx.UpsertRetrievalSurface (
      { 100, 9900, 1000, 1000, 0, 0, 0, 0, "LONG_TERM",
        "opaque/replacement", "image", -1.0, 0, 0.0, 0.0, 0.0, false,
        true, replacement_embedding });
  pctx.UpsertRetrievalSurface (
      { 200, 9901, 900, 900, 0, 0, 0, 0, "LONG_TERM", "opaque/target",
        "audio", -1.0, 0, 0.0, 0.0, 0.0, false, true,
        target_embedding });
  REQUIRE (cortext::operations::association_fanout_cache::Ensure (
      store.get (), pctx));
  const auto sidecar
      = cortext::operations::execution_cache_sidecar_internal::Find (pctx);
  REQUIRE (sidecar);
  REQUIRE (sidecar->supersession_eligibility.valid);
  REQUIRE (sidecar->supersession_eligibility.activation_ts_by_target.contains (
      200));

  cortext::operations::eviction::EvictionPolicyOverride override;
  override.consolidation_gate_enabled = false;
  override.periphery_cutoff = 0.5;
  cortext::operations::eviction::ScopedEvictionPolicyOverride scoped_override (
      override);
  auto signal = MakeSignal (4, 2000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  cortext::operations::UpdateMemoryStrength update_strength;
  auto tx = store->Begin ();
  update_strength.Execute (ctx, *tx);
  tx->Commit ();

  REQUIRE_FALSE (
      sidecar->supersession_eligibility.activation_ts_by_target.contains (
          200));
}

TEST_CASE (
    "Algorithm 18 deletes orphan objstore blobs for evicted memories",
    "[op18][memory_strength][eviction]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> vec (kEmbeddingDim, 0.0f);
  vec[0] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (*store, 9900LL, vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 9903LL, vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 9904LL, vec, 1000);
  cortext::testing::SeedMemoryV2 (*store, 9901LL, 9900LL, "weak",
                                  "LONG_TERM", 0.1, 1000);

  const auto memory_blob = PutObjstoreBlob (
      *store, std::vector<unsigned char>{ 1, 2, 3, 4 });
  const auto signal_blob = PutObjstoreBlob (
      *store, std::vector<unsigned char>{ 5, 6, 7, 8 });
  const auto reconstruction_blob = PutObjstoreBlob (
      *store, std::vector<unsigned char>{ 9, 10, 11, 12 });
  store->Execute ("UPDATE memories SET blob_id = ? WHERE memory_id = ?",
                  { memory_blob, 9901LL });
  store->Execute (
      "INSERT INTO signals("
      "signal_id, memory_id, source_id, embedding_id, blob_id, timestamp, "
      "modality, mime, serial_position, created_at) "
      "VALUES(?, ?, 'weak', ?, ?, ?, 'text', 'text/plain', 0, ?)",
      { 9904LL, 9901LL, 9900LL, signal_blob, 1100LL, 1100LL });
  store->Execute (
      "INSERT INTO memory_reconstructions("
      "reconstruction_id, memory_id, embedding_id, blob_id, created_at, "
      "uncertainty, trigger, source_confidence, context_similarity) "
      "VALUES(?, ?, ?, ?, ?, ?, 'retrieval', ?, ?)",
      { 9905LL, 9901LL, 9903LL, reconstruction_blob, 1200LL, 0.1, 0.9,
        0.8 });

  REQUIRE (CountObjstoreBlob (*store, memory_blob) == 1LL);
  REQUIRE (CountObjstoreBlob (*store, signal_blob) == 1LL);
  REQUIRE (CountObjstoreBlob (*store, reconstruction_blob) == 1LL);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  cortext::operations::eviction::EvictionPolicyOverride override;
  override.consolidation_gate_enabled = false;
  override.periphery_cutoff = 0.5;
  cortext::operations::eviction::ScopedEvictionPolicyOverride scoped_override (
      override);

  cortext::ProcessorContext pctx;
  auto signal = MakeSignal (4, 2000);
  OperationContext ctx (signal, pctx, cfg, store.get ());

  cortext::operations::UpdateMemoryStrength update_strength;
  auto tx = store->Begin ();
  update_strength.Execute (ctx, *tx);
  tx->Commit ();

  auto memory_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?",
      { 9901LL });
  REQUIRE (std::any_cast<long long> (memory_rows[0].at ("cnt")) == 0LL);

  REQUIRE (CountObjstoreBlob (*store, memory_blob) == 0LL);
  REQUIRE (CountObjstoreBlob (*store, signal_blob) == 0LL);
  REQUIRE (CountObjstoreBlob (*store, reconstruction_blob) == 0LL);
}

TEST_CASE (
    "Algorithm 18 keeps objstore blobs referenced by surviving memories",
    "[op18][memory_strength][eviction]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> vec (kEmbeddingDim, 0.0f);
  vec[0] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (*store, 9910LL, vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 9911LL, vec, 1000);
  cortext::testing::SeedMemoryV2 (*store, 9912LL, 9910LL, "weak",
                                  "LONG_TERM", 0.1, 1000);
  cortext::testing::SeedMemoryV2 (*store, 9913LL, 9911LL, "strong",
                                  "LONG_TERM", 0.9, 1000);

  const auto shared_blob = PutObjstoreBlob (
      *store, std::vector<unsigned char>{ 21, 22, 23, 24 });
  store->Execute ("UPDATE memories SET blob_id = ? WHERE memory_id IN (?, ?)",
                  { shared_blob, 9912LL, 9913LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  cortext::operations::eviction::EvictionPolicyOverride override;
  override.consolidation_gate_enabled = false;
  override.periphery_cutoff = 0.5;
  cortext::operations::eviction::ScopedEvictionPolicyOverride scoped_override (
      override);

  cortext::ProcessorContext pctx;
  auto signal = MakeSignal (4, 2000);
  OperationContext ctx (signal, pctx, cfg, store.get ());

  cortext::operations::UpdateMemoryStrength update_strength;
  auto tx = store->Begin ();
  update_strength.Execute (ctx, *tx);
  tx->Commit ();

  auto weak_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?",
      { 9912LL });
  REQUIRE (std::any_cast<long long> (weak_rows[0].at ("cnt")) == 0LL);

  auto strong_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?",
      { 9913LL });
  REQUIRE (std::any_cast<long long> (strong_rows[0].at ("cnt")) == 1LL);

  REQUIRE (CountObjstoreBlob (*store, shared_blob) == 1LL);
}

TEST_CASE (
    "Algorithm 18 deletes reconstructions and orphan reconstruction embeddings",
    "[op18][memory_strength][eviction]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> vec (kEmbeddingDim, 0.0f);
  vec[0] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (*store, 9900LL, vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 9903LL, vec, 1000);
  cortext::testing::SeedMemoryV2 (*store, 9901LL, 9900LL, "weak",
                                  "LONG_TERM", 0.1, 1000);
  store->Execute (
      "INSERT INTO memory_reconstructions("
      "reconstruction_id, memory_id, embedding_id, created_at, uncertainty, "
      "trigger, source_confidence, context_similarity) "
      "VALUES(?, ?, ?, ?, ?, 'retrieval', ?, ?)",
      { 1LL, 9901LL, 9903LL, 1100LL, 0.1, 0.9, 0.8 });
  cortext::testing::SeedSignalV2 (*store, 9905LL, 9904LL, "weak", 1200);
  store->Execute ("UPDATE signals SET memory_id = ? WHERE signal_id = ?",
                  { 9901LL, 9905LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  cortext::operations::eviction::EvictionPolicyOverride override;
  override.consolidation_gate_enabled = false;
  override.periphery_cutoff = 0.5;
  cortext::operations::eviction::ScopedEvictionPolicyOverride scoped_override (
      override);

  cortext::ProcessorContext pctx;
  auto signal = MakeSignal (4, 2000);
  OperationContext ctx (signal, pctx, cfg, store.get ());

  cortext::operations::UpdateMemoryStrength update_strength;
  auto tx = store->Begin ();
  update_strength.Execute (ctx, *tx);
  tx->Commit ();

  auto reconstruction_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memory_reconstructions "
      "WHERE memory_id = ?",
      { 9901LL });
  REQUIRE (
      std::any_cast<long long> (reconstruction_rows[0].at ("cnt")) == 0LL);

  auto reconstruction_embedding_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM embeddings WHERE embedding_id = ?",
      { 9903LL });
  REQUIRE (std::any_cast<long long> (
               reconstruction_embedding_rows[0].at ("cnt")) == 0LL);

  auto signal_embedding_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM embeddings WHERE embedding_id = ?",
      { 9904LL });
  REQUIRE (std::any_cast<long long> (
               signal_embedding_rows[0].at ("cnt")) == 0LL);
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
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
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
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
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
    return std::make_unique<cortext::DynamicOperationSet> (
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
    auto ops = std::make_unique<cortext::DynamicOperationSet> (
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
    return std::make_unique<cortext::DynamicOperationSet> (
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
    auto ops = std::make_unique<cortext::DynamicOperationSet> (
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
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
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
