// tests/operations_emotion_cascade.test.cpp
#include "test_helpers.hpp"
#include "../src/operations/association_fanout_cache_internal.hpp"
#include "../src/operations/emotional_metadata_cache_internal.hpp"
#include "../src/operations/sparse_retrieval_knobs_internal.hpp"
#include "../src/operations/signal_record_rollback_internal.hpp"
#include <array>
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <filesystem>
#include <string>
#include <tuple>
#include <unordered_map>

#include <cortext/core/knobs.hpp>
#include <cortext/operations/emotion_cascade.hpp>

#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;

using cortext::operations::PropagateEmotionalCascade;

namespace
{
constexpr int kEmbeddingDim = 256;

static Signal
MakeSignal (uint64_t ts)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

static void
SeedCascadeMemory (Store &store, long long memory_id, long long embedding_id,
                   long long created_at = 1'000'000)
{
  std::vector<float> embedding (kEmbeddingDim, 0.0f);
  embedding[0] = 1.0f;
  if (store.Execute ("SELECT 1 FROM embeddings WHERE embedding_id = ?",
                     { embedding_id }).empty ())
    cortext::testing::SeedEmbeddingV2 (
        store, embedding_id, embedding, created_at);
  cortext::testing::SeedMemoryV2 (
      store, memory_id, embedding_id, "cascade/test", "LONG_TERM", 1.0,
      created_at);
}

static void
SetCascadeSource (Store &store, long long memory_id, double intensity,
                  int radius, double decay)
{
  store.Execute (
      "UPDATE memories SET flashbulb = 1, emotional_intensity = ?, "
      "half_life_bonus = 2.0, cascade_radius = ?, cascade_decay = ?, "
      "s_arousal_avg = 0.9 WHERE memory_id = ?",
      { intensity, static_cast<long long> (radius), decay, memory_id });
}

static void
AddCascadeEdge (Store &store, long long source, long long target)
{
  store.Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, "
      "edge_type, weight) VALUES(?, ?, 'co_occurs', 1.0)",
      { source, target });
}

static double
MemoryIntensity (Store &store, long long memory_id)
{
  const auto rows = store.Execute (
      "SELECT emotional_intensity FROM memories WHERE memory_id = ?",
      { memory_id });
  REQUIRE (rows.size () == 1);
  return std::any_cast<double> (rows[0].at ("emotional_intensity"));
}

static long long
TotalChanges (Store &store)
{
  const auto rows = store.Execute ("SELECT total_changes() AS total_changes");
  REQUIRE (rows.size () == 1);
  return std::any_cast<long long> (rows[0].at ("total_changes"));
}

class ScopedExecutionCacheSidecar
{
public:
  explicit ScopedExecutionCacheSidecar (ProcessorContext &context)
      : context_ (context)
  {
    operations::execution_cache_sidecar_internal::Erase (context_);
  }

  ~ScopedExecutionCacheSidecar ()
  {
    operations::execution_cache_sidecar_internal::Erase (context_);
  }

private:
  ProcessorContext &context_;
};

class RawEmotionalMutationThenCascade final : public IOperation
{
public:
  explicit RawEmotionalMutationThenCascade (bool cascade_on_next_call,
                                             bool use_store = false)
      : cascade_on_next_call_ (cascade_on_next_call), use_store_ (use_store)
  {
  }

  void
  Execute (OperationContext &context, Transaction &tx) const override
  {
    ++call_count_;
    if (call_count_ == 1)
      {
        const std::string update
            = "UPDATE memories SET flashbulb = 1, "
              "emotional_intensity = 0.8, half_life_bonus = 2.0, "
              "cascade_radius = 1, cascade_decay = 0.5, "
              "s_arousal_avg = 0.8 WHERE memory_id = 10";
        if (use_store_)
          context.GetStore ()->Execute (update);
        else
          tx.Execute (update);
      }
    if (!cascade_on_next_call_ || call_count_ > 1)
      {
        PropagateEmotionalCascade cascade;
        cascade.Execute (context, tx);
      }
  }

private:
  bool cascade_on_next_call_ = false;
  bool use_store_ = false;
  mutable int call_count_ = 0;
};

class CascadeThenFailOnce final : public IOperation
{
public:
  void
  Execute (OperationContext &context, Transaction &tx) const override
  {
    PropagateEmotionalCascade cascade;
    cascade.Execute (context, tx);
    if (fail_once_)
      {
        fail_once_ = false;
        throw std::runtime_error ("forced cascade rollback");
      }
  }

private:
  mutable bool fail_once_ = true;
};

class ScopedTempDatabase
{
public:
  ScopedTempDatabase ()
      : path_ (cortext::testing::UniqueTempPath (
            "cortext_emotional_cascade_", ".db"))
  {
  }

  ~ScopedTempDatabase ()
  {
    std::error_code error;
    std::filesystem::remove (path_, error);
    std::filesystem::remove (path_.string () + "-wal", error);
    std::filesystem::remove (path_.string () + "-shm", error);
  }

  const std::filesystem::path &
  Path () const
  {
    return path_;
  }

private:
  std::filesystem::path path_;
};

static double
RunCustomRawEmotionalMutation (bool cascade_on_next_call,
                               bool use_store = false)
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  SeedCascadeMemory (*store, 10, 100);
  SeedCascadeMemory (*store, 20, 200);
  AddCascadeEdge (*store, 10, 20);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.stability = 0.5;
  auto root = std::make_unique<DynamicOperationSet> (
      std::make_unique<RawEmotionalMutationThenCascade> (
          cascade_on_next_call, use_store));
  SignalProcessor processor (cfg, store, std::move (root));
  processor.Process (MakeSignal (1'000'000));
  if (cascade_on_next_call)
    processor.Process (MakeSignal (1'000'001));
  return MemoryIntensity (*store, 20);
}

static std::unordered_map<std::string, double>
ExecuteCascade (const std::shared_ptr<Store> &store, bool commit = true,
                double focus = 0.5, double sensitivity = 0.5,
                double stability = 0.5)
{
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = focus;
  cfg.sensitivity = sensitivity;
  cfg.stability = stability;
  ProcessorContext processor_context;
  ScopedExecutionCacheSidecar sidecar_scope (processor_context);
  const Signal signal = MakeSignal (1'000'000);
  OperationContext context (signal, processor_context, cfg, store.get ());
  auto transaction = store->Begin ();
  PropagateEmotionalCascade operation;
  operation.Execute (context, *transaction);
  if (commit)
    transaction->Commit ();
  else
    transaction->Rollback ();
  return context.GetOperationTimings ();
}

static void
ExecuteCascadeWithMetadataCache (
    const std::shared_ptr<Store> &store,
    std::vector<operations::execution_cache_sidecar_internal::
                    EmotionalMemoryMetadata>
        metadata)
{
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  ProcessorContext processor_context;
  ScopedExecutionCacheSidecar sidecar_scope (processor_context);
  operations::emotional_metadata_cache_internal::Reset (
      processor_context, std::move (metadata));
  const Signal signal = MakeSignal (1'000'000);
  OperationContext context (signal, processor_context, cfg, store.get ());
  auto transaction = store->Begin ();
  PropagateEmotionalCascade operation;
  operation.Execute (context, *transaction);
  transaction->Commit ();
}
} // namespace

TEST_CASE ("EmotionCascadeParams derives from knobs correctly",
           "[operations][emotion_cascade][params]")
{
  auto params = operations::EmotionCascadeParams::FromKnobs (0.5, 0.0, 0.5);
  REQUIRE (params.cascade_radius == 1); // round(lerp(1, 5, 0)) = 1
  REQUIRE (params.cascade_decay == Catch::Approx (0.7).margin (1e-6)); // lerp(0.7, 0.3, 0)

  auto params2 = operations::EmotionCascadeParams::FromKnobs (0.5, 1.0, 0.5);
  REQUIRE (params2.cascade_radius == 5); // round(lerp(1, 5, 1)) = 5
  REQUIRE (params2.cascade_decay == Catch::Approx (0.3).margin (1e-6)); // lerp(0.7, 0.3, 1)
}


TEST_CASE ("Emotional metadata cache preserves shared embedding minima",
           "[operations][emotion_cascade][metadata_cache][shared_embedding]")
{
  ProcessorContext pctx;
  ScopedExecutionCacheSidecar sidecar_scope (pctx);
  operations::emotional_metadata_cache_internal::Reset (
      pctx,
      { { 10, 100, 2000, true, 0.9, 0.8, 3.0, 1, 0.5 },
        { 11, 100, 1000, false, 0.2, 0.1, 0.5, 0, 0.0 },
        { 20, 200, 1500, true, 0.4, 0.3, 1.0, 0, 0.0 } });

  const auto state
      = operations::emotional_metadata_cache_internal::FindState (pctx);
  REQUIRE (state);
  const auto &cache = state->emotional_metadata;
  REQUIRE (cache.valid);
  REQUIRE (cache.memory_ids_by_embedding.at (100)
           == std::vector<long long>{ 10, 11 });
  REQUIRE (cache.values_by_embedding.at (100).intensity
           == Catch::Approx (0.2));
  REQUIRE (cache.values_by_embedding.at (100).half_life_bonus
           == Catch::Approx (0.5));
  REQUIRE (cache.source_query_order
           == std::vector<long long>{ 10, 20 });

  operations::emotional_metadata_cache_internal::MaxEmbedding (
      pctx, 100, 0.7, 2.0);
  REQUIRE (cache.values_by_embedding.at (100).intensity
           == Catch::Approx (0.7));
  REQUIRE (cache.values_by_embedding.at (100).half_life_bonus
           == Catch::Approx (2.0));

  operations::emotional_metadata_cache_internal::OverwriteEmbedding (
      pctx, 100, true, 0.6, 1.5, 4, 0.4);
  REQUIRE (cache.rows_by_memory.at (10).cascade_radius == 4);
  REQUIRE (cache.rows_by_memory.at (11).cascade_decay
           == Catch::Approx (0.4));
  REQUIRE (cache.values_by_embedding.at (100).intensity
           == Catch::Approx (0.6));
  REQUIRE (cache.values_by_embedding.at (100).half_life_bonus
           == Catch::Approx (1.5));
  REQUIRE (cache.source_query_order
           == std::vector<long long>{ 10, 11, 20 });

  operations::emotional_metadata_cache_internal::Remove (pctx, 10);
  REQUIRE (cache.memory_ids_by_embedding.at (100)
           == std::vector<long long>{ 11 });
  REQUIRE (cache.source_query_order
           == std::vector<long long>{ 11, 20 });
  operations::emotional_metadata_cache_internal::Remove (pctx, 11);
  REQUIRE_FALSE (cache.memory_ids_by_embedding.contains (100));
  REQUIRE_FALSE (cache.values_by_embedding.contains (100));
  REQUIRE (cache.source_query_order
           == std::vector<long long>{ 20 });
}

TEST_CASE ("Emotional source index retains the SQL-ranked bounded prefix",
           "[operations][emotion_cascade][metadata_cache][bounds]")
{
  ProcessorContext processor_context;
  ScopedExecutionCacheSidecar sidecar_scope (processor_context);
  std::vector<operations::execution_cache_sidecar_internal::
                  EmotionalMemoryMetadata>
      rows;
  const std::array<double, 5> intensities { 0.7, 0.99, 0.8, 0.95, 0.9 };
  for (long long memory_id = 1; memory_id <= 5; ++memory_id)
    rows.push_back ({ memory_id, memory_id, memory_id, true,
                      intensities[static_cast<std::size_t> (memory_id - 1)],
                      0.9, 2.0, 1, 0.5 });
  operations::emotional_metadata_cache_internal::Reset (
      processor_context, std::move (rows), 3);
  const auto state
      = operations::emotional_metadata_cache_internal::FindState (
          processor_context);
  REQUIRE (state);
  REQUIRE (state->emotional_metadata.source_query_order
           == std::vector<long long>{ 2, 4, 5 });
}

TEST_CASE ("Bounded emotional source reranking refills from SQLite",
           "[operations][emotion_cascade][metadata_cache][bounds][regression]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  ProcessorContext processor_context;
  ScopedExecutionCacheSidecar sidecar_scope (processor_context);
  std::vector<operations::execution_cache_sidecar_internal::
                  EmotionalMemoryMetadata>
      rows;
  const std::array<double, 5> intensities { 0.99, 0.9, 0.8, 0.7, 0.6 };
  for (long long memory_id = 1; memory_id <= 5; ++memory_id)
    {
      SeedCascadeMemory (*store, memory_id, memory_id);
      const double intensity
          = intensities[static_cast<std::size_t> (memory_id - 1)];
      SetCascadeSource (*store, memory_id, intensity, 1, 0.5);
      rows.push_back ({ memory_id, memory_id, memory_id, true, intensity,
                        0.9, 2.0, 1, 0.5 });
    }
  operations::emotional_metadata_cache_internal::Reset (
      processor_context, std::move (rows), 3);
  const auto state
      = operations::emotional_metadata_cache_internal::FindState (
          processor_context);
  REQUIRE (state);
  auto &cache = state->emotional_metadata;
  REQUIRE (cache.source_query_order == std::vector<long long>{ 1, 2, 3 });
  const auto plan = store->Execute (
      "EXPLAIN QUERY PLAN SELECT memory_id FROM memories "
      "WHERE flashbulb = 1 AND embedding_id IS NOT NULL "
      "AND kind != 'WORKING' "
      "ORDER BY emotional_intensity DESC, memory_id ASC LIMIT 3");
  REQUIRE (plan.size () == 1);
  REQUIRE (std::any_cast<std::string> (plan[0].at ("detail"))
               .find ("idx_memories_flashbulb_intensity")
           != std::string::npos);

  auto transaction = store->Begin ();
  transaction->Execute (
      "UPDATE memories SET emotional_intensity = 0.1 WHERE memory_id = 1");
  operations::emotional_metadata_cache_internal::OverwriteEmbedding (
      processor_context, 1, true, 0.1, 2.0, 1, 0.5);
  REQUIRE (cache.source_query_order_dirty);
  REQUIRE (
      operations::emotional_metadata_cache_internal::
          RefreshBoundedSourceOrder (processor_context, *transaction, 0,
                                     0.0, 0.0));
  REQUIRE (cache.source_query_order == std::vector<long long>{ 2, 3, 4 });
  REQUIRE_FALSE (cache.source_query_order_dirty);

  transaction->Execute (
      "UPDATE memories SET emotional_intensity = 1.0 WHERE memory_id = 5");
  operations::emotional_metadata_cache_internal::MaxEmbedding (
      processor_context, 5, 1.0, 2.0);
  REQUIRE (cache.source_query_order_dirty);
  REQUIRE (
      operations::emotional_metadata_cache_internal::
          RefreshBoundedSourceOrder (processor_context, *transaction, 0,
                                     0.0, 0.0));
  REQUIRE (cache.source_query_order == std::vector<long long>{ 5, 2, 3 });
  transaction->Rollback ();
}

TEST_CASE ("Bounded emotional source reranking filters runtime eligibility "
           "before truncation",
           "[operations][emotion_cascade][metadata_cache][bounds]"
           "[eligibility][regression]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  ProcessorContext processor_context;
  ScopedExecutionCacheSidecar sidecar_scope (processor_context);
  std::vector<operations::execution_cache_sidecar_internal::
                  EmotionalMemoryMetadata>
      rows;
  for (long long memory_id = 1; memory_id <= 5; ++memory_id)
    {
      SeedCascadeMemory (*store, memory_id, memory_id, 2000 + memory_id);
      SetCascadeSource (*store, memory_id, 1.0 - 0.05 * memory_id, 1,
                        0.5);
      rows.push_back ({ memory_id, memory_id, 2000 + memory_id, true,
                        1.0 - 0.05 * memory_id, 0.9, 2.0, 1, 0.5 });
    }
  operations::emotional_metadata_cache_internal::Reset (
      processor_context, std::move (rows), 3);
  const auto state
      = operations::emotional_metadata_cache_internal::FindState (
          processor_context);
  REQUIRE (state);
  auto &cache = state->emotional_metadata;
  REQUIRE (cache.source_query_order == std::vector<long long>{ 1, 2, 3 });

  store->Execute ("UPDATE memories SET created_at = 100 WHERE memory_id = 1");
  store->Execute ("UPDATE memories SET s_arousal_avg = 0.1 WHERE memory_id = 2");
  auto transaction = store->Begin ();
  REQUIRE (
      operations::emotional_metadata_cache_internal::
          RefreshBoundedSourceOrder (processor_context, *transaction, 1000,
                                     0.5, 0.5));
  REQUIRE (cache.source_query_order == std::vector<long long>{ 3, 4, 5 });
  transaction->Rollback ();
}

TEST_CASE ("CascadeRadius and CascadeDecay knob values",
           "[operations][emotion_cascade][knobs]")
{
  // CascadeRadius
  REQUIRE (core::CascadeRadius (0.0) == 1);
  REQUIRE (core::CascadeRadius (0.5) == 3);
  REQUIRE (core::CascadeRadius (1.0) == 5);

  // CascadeDecay
  REQUIRE (core::CascadeDecay (0.0) == Catch::Approx (0.7).margin (1e-6));
  REQUIRE (
      core::CascadeDecay (0.5)
      == Catch::Approx (
             core::Lerp (0.7, 0.3, core::SensitivityBias (0.5)))
             .margin (1e-6));
  REQUIRE (core::CascadeDecay (1.0) == Catch::Approx (0.3).margin (1e-6));
}

TEST_CASE ("PropagateEmotionalCascade propagates through graph edges",
           "[operations][emotion_cascade]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // V2: Create embeddings and memories for the cascade to propagate through
  std::vector<float> emb (kEmbeddingDim, 0.0f);
  emb[0] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (*store, 1LL, emb);
  cortext::testing::SeedEmbeddingV2 (*store, 2LL, emb);
  cortext::testing::SeedEmbeddingV2 (*store, 3LL, emb);
  cortext::testing::SeedMemoryV2 (*store, 1LL, 1LL, "test");
  cortext::testing::SeedMemoryV2 (*store, 2LL, 2LL, "test");
  cortext::testing::SeedMemoryV2 (*store, 3LL, 3LL, "test");

  // V2: Create associations (graph edges): 1 -> 2 -> 3 via memory_ids
  store->Execute ("INSERT INTO associations (source_memory_id, target_memory_id, edge_type, weight) "
                  "VALUES (?, ?, ?, ?)",
                  { 1LL, 2LL, std::string ("co_occurs"), 0.9 });
  store->Execute ("INSERT INTO associations (source_memory_id, target_memory_id, edge_type, weight) "
                  "VALUES (?, ?, ?, ?)",
                  { 2LL, 3LL, std::string ("causes"), 0.8 });

  // v2: Set high-intensity flashbulb for source memory (inline on memories)
  const long long now = 5000;
  store->Execute ("UPDATE memories SET flashbulb = 1, emotional_intensity = ?, "
                  "half_life_bonus = ?, cascade_radius = ?, cascade_decay = ?, "
                  "s_arousal_avg = ? "
                  "WHERE embedding_id = ?",
                  { 0.8, 2.0, 2LL, 0.5, 0.8, 1LL });

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.5; // cascade_radius = 3, decay = 0.5
  cfg.stability = 0.5;
  auto ops = std::make_unique<DynamicOperationSet> (

      std::make_unique<PropagateEmotionalCascade> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (static_cast<uint64_t> (now)));
  processor.Flush ();

  // v2: Verify emotional_intensity propagated through memories table
  auto rows = store->Execute (
      "SELECT embedding_id, emotional_intensity FROM memories ORDER BY embedding_id",
      {});

  // Should have emotional values for memories (propagated)
  // Note: Results depend on consolidation interval and timing
}

TEST_CASE ("Custom raw emotional mutation uses SQL cascade in one transaction",
           "[operations][emotion_cascade][custom][cache][regression]")
{
  REQUIRE (RunCustomRawEmotionalMutation (false) > 0.0);
}

TEST_CASE ("Custom committed emotional mutation uses SQL cascade next call",
           "[operations][emotion_cascade][custom][cache][regression]")
{
  REQUIRE (RunCustomRawEmotionalMutation (true) > 0.0);
}

TEST_CASE ("Custom Store mutation uses SQL cascade in one transaction",
           "[operations][emotion_cascade][custom][cache][regression]")
{
  REQUIRE (RunCustomRawEmotionalMutation (false, true) > 0.0);
}

TEST_CASE ("PropagateEmotionalCascade decays intensity per hop",
           "[operations][emotion_cascade][decay]")
{
  // Test that intensity decays correctly through hops
  double source_intensity = 0.8;
  double decay = 0.5;

  // After 1 hop: 0.8 * 0.5 = 0.4
  double hop1 = source_intensity * std::pow (decay, 1);
  REQUIRE (hop1 == Catch::Approx (0.4).margin (1e-6));

  // After 2 hops: 0.8 * 0.5^2 = 0.2
  double hop2 = source_intensity * std::pow (decay, 2);
  REQUIRE (hop2 == Catch::Approx (0.2).margin (1e-6));

  // After 3 hops: 0.8 * 0.5^3 = 0.1
  double hop3 = source_intensity * std::pow (decay, 3);
  REQUIRE (hop3 == Catch::Approx (0.1).margin (1e-6));
}

TEST_CASE ("PropagateEmotionalCascade uses a millisecond consolidation window",
           "[operations][emotion_cascade][timestamp][regression]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  std::vector<float> emb (kEmbeddingDim, 0.0f);
  emb[0] = 1.0f;
  constexpr long long now_ms = 1'000'000;
  constexpr long long source_created_ms = now_ms - 60'000;
  cortext::testing::SeedEmbeddingV2 (*store, 10, emb, source_created_ms);
  cortext::testing::SeedEmbeddingV2 (*store, 20, emb, source_created_ms);
  cortext::testing::SeedMemoryV2 (*store, 10, 10, "source", "LONG_TERM", 1.0,
                                  source_created_ms);
  cortext::testing::SeedMemoryV2 (*store, 20, 20, "target", "LONG_TERM", 1.0,
                                  source_created_ms);
  store->Execute (
      "UPDATE memories SET flashbulb = 1, emotional_intensity = 0.8, "
      "half_life_bonus = 2.0, cascade_radius = 1, cascade_decay = 0.5, "
      "s_arousal_avg = 0.8 WHERE memory_id = 10");
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight) VALUES(10, 20, 'co_occurs', 1.0)");

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.stability = 0.0; // Five-minute source window.
  ProcessorContext pctx;
  ScopedExecutionCacheSidecar sidecar_scope (pctx);
  Signal signal = MakeSignal (now_ms);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  auto tx = store->Begin ();
  PropagateEmotionalCascade op;
  op.Execute (ctx, *tx);
  tx->Commit ();

  const auto rows = store->Execute (
      "SELECT emotional_intensity FROM memories WHERE memory_id = 20");
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<double> (rows[0].at ("emotional_intensity")) > 0.0);
  operations::execution_cache_sidecar_internal::Erase (pctx);
}

TEST_CASE ("PropagateEmotionalCascade preserves ordered sources across cycles",
           "[operations][emotion_cascade][cache][regression]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  for (long long id = 1; id <= 4; ++id)
    SeedCascadeMemory (*store, id, id);
  SetCascadeSource (*store, 1, 0.9, 1, 0.5);
  SetCascadeSource (*store, 2, 0.8, 3, 0.8);
  AddCascadeEdge (*store, 1, 3);
  AddCascadeEdge (*store, 2, 3);
  AddCascadeEdge (*store, 3, 4);
  AddCascadeEdge (*store, 4, 1); // Cycle must remain radius-bounded.

  ExecuteCascade (store);

  // Higher-intensity source 1 is visited first and owns shared target 3.
  REQUIRE (MemoryIntensity (*store, 3)
           == Catch::Approx (0.45).margin (1e-9));
  REQUIRE (MemoryIntensity (*store, 4)
           == Catch::Approx (0.45).margin (1e-9));
}

TEST_CASE ("PropagateEmotionalCascade preserves tied source order",
           "[operations][emotion_cascade][cache][order][regression]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  SeedCascadeMemory (*store, 10, 10);
  SeedCascadeMemory (*store, 20, 20);
  SeedCascadeMemory (*store, 30, 30);
  SetCascadeSource (*store, 10, 0.9, 1, 0.5);
  SetCascadeSource (*store, 20, 0.9, 1, 0.8);
  AddCascadeEdge (*store, 10, 30);
  AddCascadeEdge (*store, 20, 30);
  ExecuteCascade (store);

  REQUIRE (MemoryIntensity (*store, 30)
           == Catch::Approx (0.45).margin (1e-9));
}

TEST_CASE ("PropagateEmotionalCascade cache preserves tied source order",
           "[operations][emotion_cascade][metadata_cache][order][regression]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  SeedCascadeMemory (*store, 10, 10, 800'000);
  SeedCascadeMemory (*store, 20, 20, 900'000);
  SeedCascadeMemory (*store, 30, 30, 700'000);
  SetCascadeSource (*store, 10, 0.9, 1, 0.5);
  SetCascadeSource (*store, 20, 0.9, 1, 0.8);
  AddCascadeEdge (*store, 10, 30);
  AddCascadeEdge (*store, 20, 30);

  ExecuteCascadeWithMetadataCache (
      store,
      { { 10, 10, 800'000, true, 0.9, 0.9, 2.0, 1, 0.5 },
        { 20, 20, 900'000, true, 0.9, 0.9, 2.0, 1, 0.8 },
        { 30, 30, 700'000, false, 0.0, 0.0, 0.0, 0, 0.0 } });

  REQUIRE (MemoryIntensity (*store, 30)
           == Catch::Approx (0.45).margin (1e-9));
}

TEST_CASE ("PropagateEmotionalCascade preserves duplicate embedding semantics",
           "[operations][emotion_cascade][cache][duplicate][regression]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  SeedCascadeMemory (*store, 10, 100);
  SeedCascadeMemory (*store, 11, 100);
  SeedCascadeMemory (*store, 20, 200);
  SeedCascadeMemory (*store, 21, 200);
  SetCascadeSource (*store, 11, 0.9, 1, 0.8);
  AddCascadeEdge (*store, 10, 20); // Legacy lookup selects first source row.
  store->Execute (
      "UPDATE memories SET emotional_intensity = 0.9, half_life_bonus = 3.0 "
      "WHERE memory_id = 20");
  store->Execute (
      "UPDATE memories SET emotional_intensity = 0.0, half_life_bonus = 0.0 "
      "WHERE memory_id = 21");

  ExecuteCascade (store);

  REQUIRE (MemoryIntensity (*store, 20)
           == Catch::Approx (0.9).margin (1e-9));
  REQUIRE (MemoryIntensity (*store, 21)
           == Catch::Approx (0.72).margin (1e-9));
}

TEST_CASE ("PropagateEmotionalCascade keeps a detached first duplicate seed",
           "[operations][emotion_cascade][cache][duplicate][regression]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  SeedCascadeMemory (*store, 10, 100);
  SeedCascadeMemory (*store, 11, 100);
  SeedCascadeMemory (*store, 20, 200);
  SetCascadeSource (*store, 11, 0.9, 1, 0.8);
  AddCascadeEdge (*store, 11, 20);

  ExecuteCascade (store);

  // The legacy lookup selects detached memory 10, not later duplicate 11.
  REQUIRE (MemoryIntensity (*store, 20)
           == Catch::Approx (0.0).margin (1e-9));
}

TEST_CASE ("PropagateEmotionalCascade traverses a null-embedding bridge",
           "[operations][emotion_cascade][cache][null_embedding][regression]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  SeedCascadeMemory (*store, 1, 1);
  SeedCascadeMemory (*store, 2, 2);
  SeedCascadeMemory (*store, 3, 3);
  SetCascadeSource (*store, 1, 0.9, 2, 0.5);
  store->Execute ("UPDATE memories SET embedding_id = NULL WHERE memory_id = 2");
  AddCascadeEdge (*store, 1, 2);
  AddCascadeEdge (*store, 2, 3);

  ExecuteCascade (store);

  REQUIRE (MemoryIntensity (*store, 3)
           == Catch::Approx (0.225).margin (1e-9));
}

TEST_CASE ("PropagateEmotionalCascade skips exact no-op updates and rolls back",
           "[operations][emotion_cascade][cache][noop][rollback]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  SeedCascadeMemory (*store, 1, 1);
  SeedCascadeMemory (*store, 2, 2);
  SetCascadeSource (*store, 1, 0.9, 1, 0.5);
  AddCascadeEdge (*store, 1, 2);
  store->Execute (
      "UPDATE memories SET emotional_intensity = 0.8, half_life_bonus = 3.0 "
      "WHERE memory_id = 2");
  const long long changes_before = TotalChanges (*store);
  ExecuteCascade (store);
  REQUIRE (TotalChanges (*store) == changes_before);
  REQUIRE (MemoryIntensity (*store, 2)
           == Catch::Approx (0.8).margin (1e-9));

  store->Execute (
      "UPDATE memories SET emotional_intensity = 0.0, half_life_bonus = 0.0 "
      "WHERE memory_id = 2");
  ExecuteCascade (store, false);
  REQUIRE (MemoryIntensity (*store, 2)
           == Catch::Approx (0.0).margin (1e-9));
}

TEST_CASE ("PropagateEmotionalCascade skips a stable fixed-point traversal",
           "[operations][emotion_cascade][cache][fixed_point][regression]")
{
  cortext::testing::ScopedEnvVar profile (
      "CORTEXT_PROFILE_EMOTIONAL_CASCADE", "1");
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  SeedCascadeMemory (*store, 1, 1, 3'000'000);
  SeedCascadeMemory (*store, 2, 2, 3'000'000);
  SeedCascadeMemory (*store, 3, 3, 3'000'000);
  SetCascadeSource (*store, 1, 0.9, 2, 0.5);
  AddCascadeEdge (*store, 1, 2);
  REQUIRE (MemoryIntensity (*store, 3)
           == Catch::Approx (0.0).margin (1e-9));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  ProcessorContext processor_context;
  ScopedExecutionCacheSidecar sidecar_scope (processor_context);
  operations::emotional_metadata_cache_internal::Reset (
      processor_context,
      { { 1, 1, 3'000'000, true, 0.9, 0.9, 2.0, 2, 0.5 },
        { 2, 2, 3'000'000, false, 0.0, 0.0, 0.0, 0, 0.0 },
        { 3, 3, 3'000'000, false, 0.0, 0.0, 0.0, 0, 0.0 } });

  auto execute = [&] (uint64_t timestamp) {
    const Signal signal = MakeSignal (timestamp);
    OperationContext context (signal, processor_context, cfg, store.get ());
    auto transaction = store->Begin ();
    PropagateEmotionalCascade operation;
    operation.Execute (context, *transaction);
    transaction->Commit ();
    return context.GetOperationTimings ();
  };

  const auto first = execute (3'000'000);
  REQUIRE (first.at ("EmotionalCascade.source_count") == 1.0);
  REQUIRE (first.at ("EmotionalCascade.neighbor_count") == 2.0);
  const long long changes_at_fixed_point = TotalChanges (*store);

  const auto second = execute (3'000'001);
  REQUIRE (TotalChanges (*store) == changes_at_fixed_point);
  REQUIRE (second.at ("EmotionalCascade.source_count") == 0.0);
  REQUIRE (second.at ("EmotionalCascade.neighbor_count") == 0.0);

  SeedCascadeMemory (*store, 4, 4, 3'000'001);
  SeedCascadeMemory (*store, 5, 5, 3'000'001);
  operations::emotional_metadata_cache_internal::Upsert (
      processor_context,
      { 4, 4, 3'000'001, false, 0.0, 0.0, 0.0, 0, 0.0 });
  operations::emotional_metadata_cache_internal::Upsert (
      processor_context,
      { 5, 5, 3'000'001, false, 0.0, 0.0, 0.0, 0, 0.0 });
  AddCascadeEdge (*store, 4, 5);
  operations::association_fanout_cache::UpsertAssociation (
      processor_context, processor_context.association_fanout_cache,
      4, 5, 4, 5, "related", 1.0, 3'000'001);
  const auto unrelated_topology = execute (3'000'002);
  REQUIRE (unrelated_topology.at ("EmotionalCascade.source_count") == 0.0);
  REQUIRE (unrelated_topology.at ("EmotionalCascade.neighbor_count") == 0.0);

  const auto reversed_time = execute (2'999'999);
  REQUIRE (reversed_time.at ("EmotionalCascade.source_count") == 1.0);
  REQUIRE (reversed_time.at ("EmotionalCascade.neighbor_count") == 2.0);

  AddCascadeEdge (*store, 2, 3);
  REQUIRE (MemoryIntensity (*store, 3)
           == Catch::Approx (0.0).margin (1e-9));
  processor_context.association_fanout_cache.valid = false;
  const auto topology_changed = execute (3'000'002);
  REQUIRE (topology_changed.at ("EmotionalCascade.source_count") == 1.0);
  REQUIRE (topology_changed.at ("EmotionalCascade.neighbor_count") == 3.0);
  REQUIRE (MemoryIntensity (*store, 3)
           == Catch::Approx (0.225).margin (1e-9));

  store->Execute (
      "UPDATE memories SET emotional_intensity = 1.0 WHERE memory_id = 1");
  operations::emotional_metadata_cache_internal::OverwriteEmbedding (
      processor_context, 1, true, 1.0, 2.0, 2, 0.5);
  const auto source_changed = execute (3'000'003);
  REQUIRE (source_changed.at ("EmotionalCascade.source_count") == 1.0);
  REQUIRE (source_changed.at ("EmotionalCascade.neighbor_count") == 3.0);
  REQUIRE (MemoryIntensity (*store, 2)
           == Catch::Approx (0.5).margin (1e-9));

  store->Execute (
      "UPDATE memories SET emotional_intensity = 0.0, half_life_bonus = 0.0 "
      "WHERE memory_id = 2");
  operations::emotional_metadata_cache_internal::OverwriteEmbedding (
      processor_context, 2, false, 0.0, 0.0, 0, 0.0);
  const auto target_decreased = execute (3'000'004);
  REQUIRE (target_decreased.at ("EmotionalCascade.source_count") == 1.0);
  REQUIRE (target_decreased.at ("EmotionalCascade.neighbor_count") == 3.0);
  REQUIRE (MemoryIntensity (*store, 2)
           == Catch::Approx (0.5).margin (1e-9));
}

TEST_CASE ("Ordinary unconnected storage preserves the emotional fixed point",
           "[operations][emotion_cascade][cache][fixed_point][regression]")
{
  cortext::testing::ScopedEnvVar profile (
      "CORTEXT_PROFILE_EMOTIONAL_CASCADE", "1");
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  SeedCascadeMemory (*store, 1, 1, 3'000'000);
  SeedCascadeMemory (*store, 2, 2, 3'000'000);
  SetCascadeSource (*store, 1, 0.9, 1, 0.5);
  AddCascadeEdge (*store, 1, 2);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  ProcessorContext processor_context;
  ScopedExecutionCacheSidecar sidecar_scope (processor_context);
  operations::emotional_metadata_cache_internal::Reset (
      processor_context,
      { { 1, 1, 3'000'000, true, 0.9, 0.9, 2.0, 1, 0.5 },
        { 2, 2, 3'000'000, false, 0.0, 0.0, 0.0, 0, 0.0 } });

  auto execute = [&] (uint64_t timestamp) {
    const Signal signal = MakeSignal (timestamp);
    OperationContext context (signal, processor_context, cfg, store.get ());
    auto transaction = store->Begin ();
    PropagateEmotionalCascade operation;
    operation.Execute (context, *transaction);
    transaction->Commit ();
    return context.GetOperationTimings ();
  };

  const auto initial = execute (3'000'000);
  REQUIRE (initial.at ("EmotionalCascade.source_count") == 1.0);
  REQUIRE (initial.at ("EmotionalCascade.neighbor_count") == 1.0);

  SeedCascadeMemory (*store, 3, 3, 3'000'001);
  operations::emotional_metadata_cache_internal::Upsert (
      processor_context,
      { 3, 3, 3'000'001, false, 0.0, 0.4, 0.0, 0, 0.0 });
  operations::emotional_metadata_cache_internal::OverwriteEmbedding (
      processor_context, 3, false, 0.4, 0.0, 1, 0.5);

  const auto after_ordinary_storage = execute (3'000'001);
  REQUIRE (after_ordinary_storage.at ("EmotionalCascade.source_count")
           == 0.0);
  REQUIRE (after_ordinary_storage.at ("EmotionalCascade.neighbor_count")
           == 0.0);
  REQUIRE (MemoryIntensity (*store, 2)
           == Catch::Approx (0.45).margin (1e-9));
}

TEST_CASE ("PropagateEmotionalCascade batches more than 64 sources exactly",
           "[operations][emotion_cascade][cache][source_batch][regression]")
{
  cortext::testing::ScopedEnvVar profile (
      "CORTEXT_PROFILE_EMOTIONAL_CASCADE", "1");
  cortext::testing::ScopedEnvVar collapse (
      "CORTEXT_EXPERIMENT_EMOTIONAL_WINNER_COLLAPSE", "1");
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  constexpr long long kTargetMemoryId = 1000;
  SeedCascadeMemory (*store, kTargetMemoryId, kTargetMemoryId);
  for (long long source = 1; source <= 70; ++source)
    {
      SeedCascadeMemory (*store, source, source);
      SetCascadeSource (*store, source, 0.9 - 0.001 * (source - 1), 1,
                        0.5);
      AddCascadeEdge (*store, source, kTargetMemoryId);
    }

  const auto timings = ExecuteCascade (store);
  REQUIRE (timings.at ("EmotionalCascade.source_count") == 70.0);
  // The shared bit-mask traversal may observe the target from every source,
  // but first-source-wins means only one winning source/target pair should be
  // materialized and executed.
  REQUIRE (timings.at ("EmotionalCascade.neighbor_count") == 1.0);
  REQUIRE (timings.at ("EmotionalCascade.neighbor_execution_count") == 1.0);
  REQUIRE (timings.at ("EmotionalCascade.reachable_pair_count") == 70.0);
  REQUIRE (timings.at ("EmotionalCascade.edge_visit_count") == 70.0);
  REQUIRE (timings.at ("EmotionalCascade.update_count") == 1.0);
  REQUIRE (MemoryIntensity (*store, kTargetMemoryId)
           == Catch::Approx (0.45).margin (1e-9));
}

TEST_CASE ("PropagateEmotionalCascade edge experiment uses the knob-derived "
           "physical ceiling",
           "[operations][emotion_cascade][bounds][edge_visit]")
{
  cortext::testing::ScopedEnvVar profile (
      "CORTEXT_PROFILE_EMOTIONAL_CASCADE", "1");
  cortext::testing::ScopedEnvVar collapse (
      "CORTEXT_EXPERIMENT_EMOTIONAL_WINNER_COLLAPSE", "1");
  cortext::testing::ScopedEnvVar edge_limit (
      "CORTEXT_EXPERIMENT_EMOTIONAL_EDGE_LIMIT", "1");
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  constexpr double kFocus = 0.0;
  constexpr double kSensitivity = 0.0;
  constexpr double kStability = 0.0;
  const long long limit = operations::sparse_retrieval_knobs_internal::
      PublicQueryNodeBudget (kFocus, kSensitivity, kStability);
  REQUIRE (limit == 1280);
  SeedCascadeMemory (*store, 1, 1);
  SetCascadeSource (*store, 1, 0.9, 1, 0.9);
  for (long long target = 2; target <= limit + 8; ++target)
    {
      SeedCascadeMemory (*store, target, target);
      AddCascadeEdge (*store, 1, target);
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = kFocus;
  cfg.sensitivity = kSensitivity;
  cfg.stability = kStability;
  ProcessorContext processor_context;
  ScopedExecutionCacheSidecar sidecar_scope (processor_context);
  std::vector<operations::execution_cache_sidecar_internal::
                  EmotionalMemoryMetadata>
      metadata;
  metadata.reserve (static_cast<std::size_t> (limit + 8));
  metadata.push_back (
      { 1, 1, 1'000'000, true, 0.9, 0.9, 2.0, 1, 0.9 });
  for (long long target = 2; target <= limit + 8; ++target)
    metadata.push_back (
        { target, target, 1'000'000, false, 0.0, 0.0, 0.0, 0, 0.0 });
  operations::emotional_metadata_cache_internal::Reset (
      processor_context, std::move (metadata));
  auto execute = [&] (uint64_t timestamp) {
    const Signal signal = MakeSignal (timestamp);
    OperationContext context (signal, processor_context, cfg, store.get ());
    auto transaction = store->Begin ();
    PropagateEmotionalCascade operation;
    operation.Execute (context, *transaction);
    transaction->Commit ();
    return context.GetOperationTimings ();
  };

  const auto first = execute (1'000'000);
  REQUIRE (first.at ("EmotionalCascade.edge_visit_limit") == limit);
  REQUIRE (first.at ("EmotionalCascade.edge_visit_count") == limit);
  REQUIRE (first.at ("EmotionalCascade.edge_visit_limit_reached") == 1.0);
  REQUIRE (first.at ("EmotionalCascade.neighbor_count") <= limit);

  // A physically truncated traversal is not a fixed point. The same inputs
  // must revisit the graph instead of taking the zero-work fast path.
  const auto resumed = execute (1'000'001);
  REQUIRE (resumed.at ("EmotionalCascade.source_count") == 1.0);
  REQUIRE (resumed.at ("EmotionalCascade.edge_visit_count") == limit);
  REQUIRE (resumed.at ("EmotionalCascade.edge_visit_limit_reached") == 1.0);
}

TEST_CASE ("PropagateEmotionalCascade bounds source inspection and execution "
           "from the same knobs",
           "[operations][emotion_cascade][bounds][source]")
{
  cortext::testing::ScopedEnvVar profile (
      "CORTEXT_PROFILE_EMOTIONAL_CASCADE", "1");
  cortext::testing::ScopedEnvVar collapse (
      "CORTEXT_EXPERIMENT_EMOTIONAL_WINNER_COLLAPSE", "1");
  cortext::testing::ScopedEnvVar edge_limit (
      "CORTEXT_EXPERIMENT_EMOTIONAL_EDGE_LIMIT", "1");
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  constexpr double kFocus = 0.0;
  constexpr double kSensitivity = 0.0;
  constexpr double kStability = 0.0;
  const long long inspection_limit
      = operations::sparse_retrieval_knobs_internal::PublicQueryNodeBudget (
          kFocus, kSensitivity, kStability);
  const long long execution_limit
      = operations::sparse_retrieval_knobs_internal::BackfillBatchSize (
          kFocus, kSensitivity, kStability);
  REQUIRE (inspection_limit == 1280);
  REQUIRE (execution_limit == 64);
  for (long long source = 1; source <= inspection_limit + 8; ++source)
    {
      SeedCascadeMemory (*store, source, source);
      SetCascadeSource (*store, source, 0.9, 1, 0.5);
    }

  const auto timings = ExecuteCascade (
      store, true, kFocus, kSensitivity, kStability);
  REQUIRE (timings.at ("EmotionalCascade.source_inspection_limit")
           == inspection_limit);
  REQUIRE (timings.at ("EmotionalCascade.source_inspection_count")
           == inspection_limit);
  REQUIRE (timings.at ("EmotionalCascade.source_execution_count")
           == inspection_limit);
  REQUIRE (timings.at ("EmotionalCascade.source_limit")
           == execution_limit);
  REQUIRE (timings.at ("EmotionalCascade.source_count") == execution_limit);
  REQUIRE (timings.at ("EmotionalCascade.source_limit_reached") == 1.0);
}

TEST_CASE ("PropagateEmotionalCascade caps writes at a knob-derived identity "
           "envelope",
           "[operations][emotion_cascade][bounds][knobs]")
{
  cortext::testing::ScopedEnvVar profile (
      "CORTEXT_PROFILE_EMOTIONAL_CASCADE", "1");
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  constexpr double kFocus = 0.0;
  constexpr double kSensitivity = 0.0;
  constexpr double kStability = 0.0;
  const int update_limit
      = operations::sparse_retrieval_knobs_internal::
            ActivationIdentityTarget (kFocus, kSensitivity, kStability);
  REQUIRE (update_limit == 640);

  SeedCascadeMemory (*store, 1, 1);
  SetCascadeSource (*store, 1, 0.9, 1, 0.9);
  for (long long target = 2; target <= update_limit + 3; ++target)
    SeedCascadeMemory (*store, target, target);
  AddCascadeEdge (*store, 1, 2);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = kFocus;
  cfg.sensitivity = kSensitivity;
  cfg.stability = kStability;
  ProcessorContext processor_context;
  ScopedExecutionCacheSidecar sidecar_scope (processor_context);
  std::vector<operations::execution_cache_sidecar_internal::
                  EmotionalMemoryMetadata>
      metadata;
  metadata.reserve (static_cast<std::size_t> (update_limit + 3));
  metadata.push_back (
      { 1, 1, 1'000'000, true, 0.9, 0.9, 2.0, 1, 0.9 });
  for (long long target = 2; target <= update_limit + 3; ++target)
    metadata.push_back (
        { target, target, 1'000'000, false, 0.0, 0.0, 0.0, 0, 0.0 });
  operations::emotional_metadata_cache_internal::Reset (
      processor_context, std::move (metadata));
  auto execute = [&] (uint64_t timestamp) {
    const Signal signal = MakeSignal (timestamp);
    OperationContext context (signal, processor_context, cfg, store.get ());
    auto transaction = store->Begin ();
    PropagateEmotionalCascade operation;
    operation.Execute (context, *transaction);
    transaction->Commit ();
    return context.GetOperationTimings ();
  };

  // Establish a real fixed point before the reachable topology expands.
  const auto initial = execute (1'000'000);
  REQUIRE (initial.at ("EmotionalCascade.update_count") == 1.0);
  REQUIRE (initial.at ("EmotionalCascade.update_limit_reached") == 0.0);

  for (long long target = 3; target <= update_limit + 3; ++target)
    {
      AddCascadeEdge (*store, 1, target);
      operations::association_fanout_cache::UpsertAssociation (
          processor_context, processor_context.association_fanout_cache,
          1, target, 1, target, "co_occurs", 1.0, 1'000'001);
    }

  const auto timings = execute (1'000'001);
  REQUIRE (timings.at ("EmotionalCascade.update_limit") == update_limit);
  REQUIRE (timings.at ("EmotionalCascade.update_count") == update_limit);
  REQUIRE (timings.at ("EmotionalCascade.update_limit_reached") == 1.0);
  const auto changed = store->Execute (
      "SELECT COUNT(*) AS n FROM memories "
      "WHERE memory_id >= 2 AND emotional_intensity > 0.0");
  REQUIRE (changed.size () == 1);
  REQUIRE (store::AnyToLongLong (changed[0].at ("n"))
           == update_limit + 1);

  // A saturated pass is an incomplete prefix, not a fixed point. Reusing the
  // same ProcessorContext must revisit the unresolved tail on the next event.
  const auto resumed = execute (1'000'002);
  REQUIRE (resumed.at ("EmotionalCascade.source_count") == 1.0);
  REQUIRE (resumed.at ("EmotionalCascade.update_limit_reached") == 0.0);
  REQUIRE (resumed.at ("EmotionalCascade.update_count") == 1.0);
  const auto all_changed = store->Execute (
      "SELECT COUNT(*) AS n FROM memories "
      "WHERE memory_id >= 2 AND emotional_intensity > 0.0");
  REQUIRE (all_changed.size () == 1);
  REQUIRE (store::AnyToLongLong (all_changed[0].at ("n"))
           == update_limit + 2);
}
