// tests/operations_reconsolidation.test.cpp
#include <Eigen/Dense>
#include "test_helpers.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/algorithms.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/core/utils.hpp>
#include <cortext/operations/reconsolidation.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;
using cortext::operations::ApplyReconsolidation;

namespace
{

constexpr int kEmbeddingDim = 256;

// Helper op to seed embeddings and memories into the v2 database.
class SeedEmbeddingsOp : public IOperation
{
public:
  explicit SeedEmbeddingsOp (std::unordered_map<long long, Eigen::VectorXf> embs)
      : embeddings_ (std::move (embs))
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto *store = ctx.GetStore ();
    auto now_ts = cortext::testing::NowMs ();
    for (const auto &[id, emb] : embeddings_)
      {
        std::vector<float> vec (emb.data (), emb.data () + emb.size ());
        // v2: Insert into embeddings (minimal vec0 table)
        store->Execute (
            "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
            "VALUES(?, ?, ?)",
            { id, vec, now_ts });
        // v2: Insert into memories (comprehensive metadata including lability)
        store->Execute (
            "INSERT OR REPLACE INTO memories("
            "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
            "s_max, s_avg, strength, use_frequency, stability, connectivity, drift_mag, "
            "influence, sustained_influence, contextual_gain, redundancy, "
            "pre_activation, lability_state, suppression_count, lability_ts, created_at) "
            "VALUES(?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, "
            "1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, ?)",
            { id, id, now_ts, now_ts });
      }
  }

private:
  std::unordered_map<long long, Eigen::VectorXf> embeddings_;
};

// Helper op to preload current context and retrieved embeddings into context.
class SetupReconInputsOp : public IOperation
{
public:
  SetupReconInputsOp (Eigen::VectorXf cur,
                      std::unordered_map<long long, Eigen::VectorXf> retrieved)
      : cur_ (std::move (cur)), retrieved_ (std::move (retrieved))
  {
  }
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    pctx.recent_context_embeddings.clear ();
    pctx.recent_context_embeddings.push_back (cur_);
    ctx.SetRetrievedMemoryEmbeddings (retrieved_);
  }

private:
  Eigen::VectorXf cur_;
  std::unordered_map<long long, Eigen::VectorXf> retrieved_;
};

class SetNeuromodAchOp : public IOperation
{
public:
  explicit SetNeuromodAchOp (double ach) : ach_ (ach) {}

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.GetProcessorContext ().neuromod_ach = ach_;
  }

private:
  double ach_ = 0.0;
};

// Assert u_t increased from baseline 0.
class AssertUncertaintyIncreasedOp : public IOperation
{
public:
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    REQUIRE (pctx.u_t > 0.0);
  }
};

/// @brief Creates a 256-dim unit vector with value at specified index.
static Eigen::VectorXf
MakeUnitVec256 (int idx)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[idx] = 1.0f;
  return v;
}

static Signal
MakeSignal (const Eigen::VectorXf &emb, uint64_t ts = 1)
{
  Signal s;
  s.embedding = emb;
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

} // namespace

TEST_CASE ("Alg20 drifts embedding and writes lability fields",
           "[operations][recon]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0; // maximize plasticity
  cfg.stability = 0.0;   // minimize persistence

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  const Eigen::VectorXf mem = MakeUnitVec256 (0);

  // v2: Need to seed the memory before reconsolidation can update it
  std::unordered_map<long long, Eigen::VectorXf> retrieved{ { 1LL, mem } };
  auto seed = std::make_unique<SeedEmbeddingsOp> (retrieved);
  auto setup = std::make_unique<SetupReconInputsOp> (cur, retrieved);
  auto apply = std::make_unique<ApplyReconsolidation> ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::move (seed), std::move (setup), std::move (apply));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto s = MakeSignal (cur, /*ts=*/100);
  processor.Process (s);
  processor.Flush ();

  // v2: Memory row created in memories table
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?",
        { 1LL });
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("cnt")) == 1LL);
  }
  // v2: Lability fields are inline on memories table
  {
    auto rows = store->Execute (
        "SELECT lability_state, lability_ts FROM memories WHERE memory_id = ?",
        { 1LL });
    REQUIRE (rows.size () == 1);
    const double lab = std::any_cast<double> (rows[0].at ("lability_state"));
    const long long ts = std::any_cast<long long> (rows[0].at ("lability_ts"));
    REQUIRE (lab > 0.0);
    REQUIRE (ts == 100LL);
  }
}

TEST_CASE ("Alg20 structured reconsolidation does not overwrite shared embedding",
           "[operations][recon]")
{
  cortext::testing::ScopedEnvVar disable_constructive (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");

  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  Eigen::VectorXf mem = MakeUnitVec256 (0);
  mem[1] = 1.0f;
  mem.normalize ();
  cortext::testing::SeedEmbeddingV2 (*store, 420LL, mem, 1);
  cortext::testing::SeedMemoryV2 (*store, 100LL, 420LL, "target",
                                  "LONG_TERM", 1.0, 1);
  cortext::testing::SeedMemoryV2 (*store, 101LL, 420LL, "sibling",
                                  "LONG_TERM", 1.0, 1);
  store->Execute (
      "INSERT INTO signals(memory_id, embedding_id, source_id, timestamp, "
      "modality, created_at) VALUES (?, ?, ?, ?, 'text', ?)",
      { 100LL, 420LL, std::string ("target/signal"), 1LL, 1LL });

  ProcessorContext pctx;
  pctx.recent_context_embeddings.push_back (cur);
  pctx.UpsertAssociationCache (100LL, 420LL, mem, false);
  pctx.UpsertAssociationCache (101LL, 420LL, mem, false);
  ProcessorContext::RetrievalSurfaceEntry surface_entry;
  surface_entry.memory_id = 100LL;
  surface_entry.embedding_id = 420LL;
  surface_entry.embedding = mem;
  surface_entry.kind = "LONG_TERM";
  surface_entry.source_id = "target";
  surface_entry.start_ts = 1;
  pctx.UpsertRetrievalSurface (std::move (surface_entry));
  ProcessorContext::RetrievalSurfaceEntry sibling_surface;
  sibling_surface.memory_id = 101LL;
  sibling_surface.embedding_id = 420LL;
  sibling_surface.embedding = mem;
  sibling_surface.kind = "LONG_TERM";
  sibling_surface.source_id = "sibling";
  sibling_surface.start_ts = 1;
  pctx.UpsertRetrievalSurface (std::move (sibling_surface));
  OperationContext ctx (MakeSignal (cur, 100), pctx, cfg, store.get ());
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{ { 420LL, mem } });
  ctx.SetRetrievedMemoryCandidates (
      std::vector<OperationContext::RetrievedMemoryCandidate>{
        { 100LL, 420LL, mem, 1.0 } });

  ApplyReconsolidation op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto rows = store->Execute (
      "SELECT memory_id, embedding_id FROM memories "
      "WHERE memory_id IN (100, 101) ORDER BY memory_id",
      {});
  REQUIRE (rows.size () == 2);
  const long long target_embedding
      = std::any_cast<long long> (rows[0].at ("embedding_id"));
  const long long sibling_embedding
      = std::any_cast<long long> (rows[1].at ("embedding_id"));
  REQUIRE (target_embedding != 420LL);
  REQUIRE (sibling_embedding == 420LL);

  auto signal_rows = store->Execute (
      "SELECT embedding_id FROM signals WHERE source_id = ?",
      { std::string ("target/signal") });
  REQUIRE (signal_rows.size () == 1);
  REQUIRE (std::any_cast<long long> (signal_rows[0].at ("embedding_id"))
           == 420LL);

  auto old_embedding_rows = store->Execute (
      "SELECT embedding FROM embeddings WHERE embedding_id = ?",
      { 420LL });
  REQUIRE (old_embedding_rows.size () == 1);
  Eigen::VectorXf original_still_shared;
  REQUIRE (cortext::core::DecodeFloatBlob (
      old_embedding_rows[0].at ("embedding"), kEmbeddingDim,
      original_still_shared));
  REQUIRE (original_still_shared[1] == Catch::Approx (mem[1]));

  auto current_rows = store->Execute (
      "SELECT embedding_id FROM current_memory_embeddings WHERE memory_id = ?",
      { 100LL });
  REQUIRE (current_rows.size () == 1);
  REQUIRE (std::any_cast<long long> (current_rows[0].at ("embedding_id"))
           == target_embedding);

  auto cache_it = pctx.retrieval_surface_index.find (100LL);
  REQUIRE (cache_it != pctx.retrieval_surface_index.end ());
  REQUIRE (pctx.retrieval_surface_cache[cache_it->second].embedding_id
           == target_embedding);
  auto assoc_it = pctx.association_cache_index.find (100LL);
  REQUIRE (assoc_it != pctx.association_cache_index.end ());
  REQUIRE (pctx.association_cache[assoc_it->second].embedding_id
           == target_embedding);
  auto sibling_cache_it = pctx.retrieval_surface_embedding_index.find (420LL);
  REQUIRE (sibling_cache_it != pctx.retrieval_surface_embedding_index.end ());
  REQUIRE (pctx.retrieval_surface_cache[sibling_cache_it->second].memory_id
           == 101LL);
}

TEST_CASE ("Alg20 constructive reconsolidation refreshes processor caches",
           "[operations][recon]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  Eigen::VectorXf mem = MakeUnitVec256 (0);
  mem[1] = 1.0f;
  mem.normalize ();
  cortext::testing::SeedEmbeddingV2 (*store, 420LL, mem, 1);
  cortext::testing::SeedMemoryV2 (*store, 100LL, 420LL, "target",
                                  "LONG_TERM", 1.0, 1);

  ProcessorContext pctx;
  pctx.recent_context_embeddings.push_back (cur);
  pctx.UpsertAssociationCache (100LL, 420LL, mem, false);
  ProcessorContext::RetrievalSurfaceEntry surface_entry;
  surface_entry.memory_id = 100LL;
  surface_entry.embedding_id = 420LL;
  surface_entry.embedding = mem;
  surface_entry.kind = "LONG_TERM";
  surface_entry.source_id = "target";
  surface_entry.start_ts = 1;
  pctx.UpsertRetrievalSurface (std::move (surface_entry));

  OperationContext ctx (MakeSignal (cur, 2'000'000), pctx, cfg, store.get ());
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{ { 420LL, mem } });
  ctx.SetRetrievedMemoryCandidates (
      std::vector<OperationContext::RetrievedMemoryCandidate>{
        { 100LL, 420LL, mem, 1.0 } });

  ApplyReconsolidation op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto reconstruction_rows = store->Execute (
      "SELECT embedding_id FROM memory_reconstructions WHERE memory_id = ? "
      "ORDER BY reconstruction_id DESC LIMIT 1",
      { 100LL });
  REQUIRE (reconstruction_rows.size () == 1);
  const long long reconstruction_embedding_id
      = std::any_cast<long long> (reconstruction_rows[0].at ("embedding_id"));
  REQUIRE (reconstruction_embedding_id != 420LL);

  auto current_rows = store->Execute (
      "SELECT embedding, embedding_id FROM current_memory_embeddings "
      "WHERE memory_id = ?",
      { 100LL });
  REQUIRE (current_rows.size () == 1);
  REQUIRE (std::any_cast<long long> (current_rows[0].at ("embedding_id"))
           == reconstruction_embedding_id);
  Eigen::VectorXf current_surface;
  REQUIRE (cortext::core::DecodeFloatBlob (
      current_rows[0].at ("embedding"), kEmbeddingDim, current_surface));
  REQUIRE (current_surface (0) > mem (0));
  auto cache_it = pctx.retrieval_surface_index.find (100LL);
  REQUIRE (cache_it != pctx.retrieval_surface_index.end ());
  REQUIRE (pctx.retrieval_surface_cache[cache_it->second].embedding_id
           == reconstruction_embedding_id);
  REQUIRE (pctx.retrieval_surface_cache[cache_it->second].embedding (0)
           > mem (0));
  auto assoc_it = pctx.association_cache_index.find (100LL);
  REQUIRE (assoc_it != pctx.association_cache_index.end ());
  REQUIRE (pctx.association_cache[assoc_it->second].embedding_id
           == reconstruction_embedding_id);
  REQUIRE (pctx.association_cache[assoc_it->second].embedding (0)
           > mem (0));
}

TEST_CASE ("Alg20 no drift when S=0: embedding unchanged, lability updated",
           "[operations][recon]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.0; // no drift
  cfg.stability = 0.5;

  const Eigen::VectorXf cur = MakeUnitVec256 (1);
  const Eigen::VectorXf mem = MakeUnitVec256 (1);

  std::unordered_map<long long, Eigen::VectorXf> retrieved{ { 2LL, mem } };
  auto seed = std::make_unique<SeedEmbeddingsOp> (retrieved);
  auto setup = std::make_unique<SetupReconInputsOp> (cur, retrieved);
  auto apply = std::make_unique<ApplyReconsolidation> ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::move (seed), std::move (setup), std::move (apply));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto s = MakeSignal (cur, /*ts=*/42);
  processor.Process (s);
  processor.Flush ();

  // v2: Memory row exists (seeded) but drift_mag should be 0 when S=0
  {
    auto rows = store->Execute (
        "SELECT drift_mag FROM memories WHERE memory_id = ?",
        { 2LL });
    REQUIRE (rows.size () == 1);
    const double drift = std::any_cast<double> (rows[0].at ("drift_mag"));
    REQUIRE (drift == 0.0); // No drift when S=0
  }
  // v2: Lability fields are inline on memories table
  {
    auto rows = store->Execute (
        "SELECT lability_state, lability_ts FROM memories WHERE memory_id = ?",
        { 2LL });
    REQUIRE (rows.size () == 1);
    const double lab = std::any_cast<double> (rows[0].at ("lability_state"));
    const long long ts = std::any_cast<long long> (rows[0].at ("lability_ts"));
    REQUIRE (lab >= 0.0);
    REQUIRE (ts == 42LL);
  }
}

TEST_CASE ("Alg20 bumps uncertainty with positive drift",
           "[operations][recon]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  const Eigen::VectorXf mem = MakeUnitVec256 (0);

  auto seed = std::make_unique<SeedEmbeddingsOp> (
      std::unordered_map<long long, Eigen::VectorXf>{ { 3LL, mem } });
  auto setup = std::make_unique<SetupReconInputsOp> (
      cur, std::unordered_map<long long, Eigen::VectorXf>{ { 3LL, mem } });
  auto apply = std::make_unique<ApplyReconsolidation> ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::move (seed), std::move (setup), std::move (apply));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto s = MakeSignal (cur, /*ts=*/7);
  processor.Process (s);
  processor.Flush ();

  auto rows = store->Execute (
      "SELECT lability_state, lability_ts FROM memories WHERE embedding_id = ?",
      { 3LL });
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<double> (rows[0].at ("lability_state")) > 0.0);
  REQUIRE (std::any_cast<long long> (rows[0].at ("lability_ts")) == 7LL);
}

TEST_CASE ("High ACh increases reconsolidation drift",
           "[operations][recon][neuromod]")
{
  auto run_case = [] (bool disable_scale) {
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.focus = 0.5;
    cfg.sensitivity = 1.0;
    cfg.stability = 0.0;

    Eigen::VectorXf current = MakeUnitVec256 (0);
    current[1] = 0.9f;
    current.normalize ();
    const Eigen::VectorXf mem = MakeUnitVec256 (0);

    auto seed = std::make_unique<SeedEmbeddingsOp> (
        std::unordered_map<long long, Eigen::VectorXf>{ { 9LL, mem } });
    auto setup = std::make_unique<SetupReconInputsOp> (
        current, std::unordered_map<long long, Eigen::VectorXf>{ { 9LL, mem } });
    auto set_ach = std::make_unique<SetNeuromodAchOp> (1.0);
    auto apply = std::make_unique<ApplyReconsolidation> ();
    auto ops = std::make_unique<cortext::DynamicOperationSet> (
        std::move (seed), std::move (setup), std::move (set_ach),
        std::move (apply));

    cortext::SignalProcessor processor (cfg, store, std::move (ops));
    cortext::testing::ScopedEnvVar disable_constructive (
        "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
    std::optional<cortext::testing::ScopedEnvVar> disable_scale_guard;
    if (disable_scale)
      {
        disable_scale_guard.emplace (
            "CORTEXT_DISABLE_NEUROMOD_RECONSOLIDATION_SCALE", "1");
      }
    processor.Process (MakeSignal (current, 100));
    processor.Flush ();

    auto rows = store->Execute (
        "SELECT embedding FROM embeddings WHERE embedding_id = ?",
        { 9LL });
    REQUIRE (rows.size () == 1);
    Eigen::VectorXf updated;
    const bool decoded = cortext::core::DecodeFloatBlob (
        rows[0].at ("embedding"), kEmbeddingDim, updated);
    REQUIRE (decoded);
    return core::CosineSimilarity (updated, current);
  };

  const double sim_scaled = run_case (false);
  const double sim_unscaled = run_case (true);
  REQUIRE (sim_scaled > sim_unscaled);
}

// --- Ripple Effect Tests ---

TEST_CASE ("Alg20 ripple propagation reaches graph neighbors",
           "[operations][recon][ripple]")
{
  // Test directly with OperationContext to isolate the operation
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  // Initialize schema
  cortext::store::ApplyMigrations (*store);

  // Insert test data
  // Create a neighbor embedding (emb:10) connected to primary (emb:1)
  const Eigen::VectorXf neighbor_vec = MakeUnitVec256 (5);
  std::vector<float> neighbor_data (neighbor_vec.data (),
                                    neighbor_vec.data () + neighbor_vec.size ());
  auto now_ts = cortext::testing::NowMs ();
  // v2: Insert into embeddings (minimal)
  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) VALUES (?, ?, ?)",
      { 10LL, neighbor_data, now_ts });
  // v2: Insert into memories (comprehensive)
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, start_ts, "
      "n_signals, modality, s_max, s_avg, created_at) "
      "VALUES (?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, ?)",
      { 10LL, 10LL, now_ts, now_ts });

  // V2: Create a 'reinforces' edge from memory:1 to memory:10 in associations
  store->Execute (
      "INSERT INTO associations (source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, ?, ?)",
      { 1LL, 10LL, std::string ("reinforces"), 1.0 });

  // Create config, signal, and context directly
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0; // maximize plasticity
  cfg.stability = 0.0;   // maximize ripple (ripple_depth=2, ripple_decay=0.5)

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  const Eigen::VectorXf mem = MakeUnitVec256 (0);
  Signal s = MakeSignal (cur, /*ts=*/100);

  ProcessorContext pctx;
  pctx.recent_context_embeddings.push_back (cur);
  pctx.UpsertAssociationCache (10LL, 10LL, neighbor_vec, false);
  ProcessorContext::RetrievalSurfaceEntry neighbor_surface;
  neighbor_surface.memory_id = 10LL;
  neighbor_surface.embedding_id = 10LL;
  neighbor_surface.embedding = neighbor_vec;
  neighbor_surface.kind = "LONG_TERM";
  neighbor_surface.source_id = "neighbor";
  neighbor_surface.start_ts = static_cast<long long> (now_ts);
  pctx.UpsertRetrievalSurface (std::move (neighbor_surface));

  // Create OperationContext with the store pointer
  OperationContext ctx (s, pctx, cfg, store.get ());

  // Verify store is set in context
  REQUIRE (ctx.GetStore () != nullptr);

  // v2: Seed the primary memory before reconsolidation can update it
  std::vector<float> mem_data (mem.data (), mem.data () + mem.size ());
  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) VALUES (?, ?, ?)",
      { 1LL, mem_data, now_ts });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, start_ts, "
      "n_signals, modality, s_max, s_avg, created_at) "
      "VALUES (?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, ?)",
      { 1LL, 1LL, now_ts, now_ts });

  // Set up the retrieved embeddings
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, mem } });

  // Execute reconsolidation with a real transaction
  auto tx = store->Begin ();
  ApplyReconsolidation recon_op;
  recon_op.Execute (ctx, *tx);
  tx->Commit ();

  // v2: Verify primary reconsolidation worked (memory exists)
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?",
        { 1LL });
    REQUIRE (rows.size () == 1);
    auto cnt = std::any_cast<long long> (rows[0].at ("cnt"));
    REQUIRE (cnt == 1LL);
  }

  // v2: Verify neighbor (memory_id:10) received ripple update
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?",
        { 10LL });
    REQUIRE (rows.size () == 1);
    auto cnt = std::any_cast<long long> (rows[0].at ("cnt"));
    // Neighbor should have been written via ripple
    REQUIRE (cnt == 1LL);
  }
  {
    auto cache_it = pctx.retrieval_surface_index.find (10LL);
    REQUIRE (cache_it != pctx.retrieval_surface_index.end ());
    REQUIRE (pctx.retrieval_surface_cache[cache_it->second].embedding (0)
             > 0.0f);
    auto assoc_it = pctx.association_cache_index.find (10LL);
    REQUIRE (assoc_it != pctx.association_cache_index.end ());
    REQUIRE (pctx.association_cache[assoc_it->second].embedding (0)
             > 0.0f);
  }
}

TEST_CASE ("Alg20 ripple reconstruction forks shared neighbor embedding",
           "[operations][recon][ripple]")
{
  cortext::testing::ScopedEnvVar disable_constructive (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");

  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  const Eigen::VectorXf primary = MakeUnitVec256 (0);
  const Eigen::VectorXf neighbor = MakeUnitVec256 (5);
  const long long now_ts = 1000LL;

  cortext::testing::SeedEmbeddingV2 (*store, 1LL, primary, now_ts);
  cortext::testing::SeedMemoryV2 (*store, 1LL, 1LL, "primary",
                                  "LONG_TERM", 1.0, now_ts);
  cortext::testing::SeedEmbeddingV2 (*store, 900LL, neighbor, now_ts);
  cortext::testing::SeedMemoryV2 (*store, 200LL, 900LL, "neighbor",
                                  "LONG_TERM", 1.0, now_ts);
  cortext::testing::SeedMemoryV2 (*store, 201LL, 900LL, "sibling",
                                  "LONG_TERM", 1.0, now_ts);
  store->Execute (
      "INSERT INTO associations (source_memory_id, target_memory_id, "
      "edge_type, weight) VALUES (?, ?, ?, ?)",
      { 1LL, 200LL, std::string ("reinforces"), 1.0 });

  ProcessorContext pctx;
  pctx.recent_context_embeddings.push_back (cur);
  OperationContext ctx (MakeSignal (cur, 2000), pctx, cfg, store.get ());
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, primary } });

  auto tx = store->Begin ();
  ApplyReconsolidation recon_op;
  recon_op.Execute (ctx, *tx);
  tx->Commit ();

  auto rows = store->Execute (
      "SELECT memory_id, embedding_id FROM memories "
      "WHERE memory_id IN (200, 201) ORDER BY memory_id",
      {});
  REQUIRE (rows.size () == 2);
  const long long neighbor_embedding
      = std::any_cast<long long> (rows[0].at ("embedding_id"));
  const long long sibling_embedding
      = std::any_cast<long long> (rows[1].at ("embedding_id"));
  REQUIRE (neighbor_embedding != 900LL);
  REQUIRE (sibling_embedding == 900LL);

  auto current_rows = store->Execute (
      "SELECT embedding_id FROM current_memory_embeddings WHERE memory_id = ?",
      { 200LL });
  REQUIRE (current_rows.size () == 1);
  REQUIRE (std::any_cast<long long> (current_rows[0].at ("embedding_id"))
           == neighbor_embedding);
  REQUIRE (pctx.association_fanout_cache.valid);
}

TEST_CASE ("Alg20 ripple decay applied correctly per hop",
           "[operations][recon][ripple]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0; // maximize plasticity
  cfg.stability = 0.0;   // ripple_depth=2, ripple_decay=0.5

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  const Eigen::VectorXf mem = MakeUnitVec256 (0);
  const Eigen::VectorXf vec20 = MakeUnitVec256 (6);
  const Eigen::VectorXf vec21 = MakeUnitVec256 (7);

  // V2: Seed all embeddings needed for ripple propagation (including memory_id=1)
  std::unordered_map<long long, Eigen::VectorXf> allEmbs{
      { 1LL, mem }, { 20LL, vec20 }, { 21LL, vec21 }
  };
  auto seed = std::make_unique<SeedEmbeddingsOp> (allEmbs);
  auto setup = std::make_unique<SetupReconInputsOp> (
      cur, std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, mem } });
  auto apply = std::make_unique<ApplyReconsolidation> ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::move (seed), std::move (setup), std::move (apply));

  // Create processor first to initialize schema
  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  // Now insert associations after schema is initialized (memories already seeded above)
  // Create chain: memory 1 --reinforces--> memory 20 --reinforces--> memory 21

  // V2: Edge via ASSOCIATIONS: memory_id 1 -> 20 (depth 1)
  store->Execute (
      "INSERT INTO associations (source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, ?, ?)",
      { 1LL, 20LL, std::string ("reinforces"), 1.0 });
  // V2: Edge via ASSOCIATIONS: memory_id 20 -> 21 (depth 2 from memory 1)
  store->Execute (
      "INSERT INTO associations (source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, ?, ?)",
      { 20LL, 21LL, std::string ("reinforces"), 1.0 });

  auto s = MakeSignal (cur, /*ts=*/200);
  processor.Process (s);
  processor.Flush ();

  // v2: Verify depth-1 neighbor (memory_id:20) and depth-2 neighbor (memory_id:21) both updated
  double lab20 = 0.0, lab21 = 0.0;
  {
    auto rows = store->Execute (
        "SELECT lability_state FROM memories WHERE memory_id = ?",
        { 20LL });
    REQUIRE (rows.size () == 1);
    lab20 = std::any_cast<double> (rows[0].at ("lability_state"));
  }
  {
    auto rows = store->Execute (
        "SELECT lability_state FROM memories WHERE memory_id = ?",
        { 21LL });
    REQUIRE (rows.size () == 1);
    lab21 = std::any_cast<double> (rows[0].at ("lability_state"));
  }

  // Depth-1 neighbor should have higher lability than depth-2 (decay per hop)
  // With ripple_decay=0.5: depth-1 gets lability*0.5, depth-2 gets lability*0.25
  REQUIRE (lab20 > lab21);
  REQUIRE (lab20 > 0.0);
  REQUIRE (lab21 > 0.0);
}

TEST_CASE ("Alg20 RippleDepth knob affects traversal depth",
           "[operations][recon][ripple]")
{
  // With T=0.8, ripple_depth = round(lerp(2,1,0.8)) = round(1.2) = 1
  // So depth-2 neighbors should NOT be reached by ripple propagation
  // Note: T=1.0 would make drift_mag=0 (formula has 1-T factor), blocking all ripple
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0; // maximize plasticity
  cfg.stability = 0.8;   // HIGH stability: ripple_depth=1, but allows some drift

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  const Eigen::VectorXf mem = MakeUnitVec256 (0);
  const Eigen::VectorXf vec30 = MakeUnitVec256 (8);
  const Eigen::VectorXf vec31 = MakeUnitVec256 (9);

  // Seed all embeddings before operation runs
  std::unordered_map<long long, Eigen::VectorXf> allEmbs{
    { 1LL, mem }, { 30LL, vec30 }, { 31LL, vec31 }
  };
  auto seed = std::make_unique<SeedEmbeddingsOp> (allEmbs);
  auto setup = std::make_unique<SetupReconInputsOp> (
      cur, std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, mem } });
  auto apply = std::make_unique<ApplyReconsolidation> ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::move (seed), std::move (setup), std::move (apply));

  // Create processor first to initialize schema
  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  // V2: Insert ASSOCIATIONS edges after schema is initialized
  // Create chain: memory 1 --reinforces--> memory 30 --reinforces--> memory 31
  store->Execute (
      "INSERT INTO associations (source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, ?, ?)",
      { 1LL, 30LL, std::string ("reinforces"), 1.0 });
  store->Execute (
      "INSERT INTO associations (source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, ?, ?)",
      { 30LL, 31LL, std::string ("reinforces"), 1.0 });

  auto s = MakeSignal (cur, /*ts=*/300);
  processor.Process (s);
  processor.Flush ();

  // v2: Depth-1 neighbor (memory_id:30) should have lability_state > 0 (updated by ripple)
  {
    auto rows = store->Execute (
        "SELECT lability_state FROM memories WHERE memory_id = ?",
        { 30LL });
    REQUIRE (rows.size () == 1);
    const double lab = std::any_cast<double> (rows[0].at ("lability_state"));
    REQUIRE (lab > 0.0); // Ripple reached depth-1
  }

  // v2: Depth-2 neighbor (memory_id:31) should have lability_state = 0 (beyond ripple_depth=1)
  {
    auto rows = store->Execute (
        "SELECT lability_state FROM memories WHERE memory_id = ?",
        { 31LL });
    REQUIRE (rows.size () == 1);
    const double lab = std::any_cast<double> (rows[0].at ("lability_state"));
    REQUIRE (lab == 0.0); // Ripple did NOT reach depth-2
  }
}

TEST_CASE ("Alg20 ripple respects co_occurs edge type",
           "[operations][recon][ripple]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  const Eigen::VectorXf mem = MakeUnitVec256 (0);
  const Eigen::VectorXf vec40 = MakeUnitVec256 (10);

  // V2: Seed all embeddings needed for ripple propagation (including memory_id=1)
  std::unordered_map<long long, Eigen::VectorXf> allEmbs{
      { 1LL, mem }, { 40LL, vec40 }
  };
  auto seed = std::make_unique<SeedEmbeddingsOp> (allEmbs);
  auto setup = std::make_unique<SetupReconInputsOp> (
      cur, std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, mem } });
  auto apply = std::make_unique<ApplyReconsolidation> ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::move (seed), std::move (setup), std::move (apply));

  // Create processor first to initialize schema
  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  // Now insert associations after schema is initialized (memories already seeded)
  // Create neighbor connected via co_occurs (not reinforces)

  // V2: Edge via ASSOCIATIONS: memory 1 --co_occurs--> memory 40
  store->Execute (
      "INSERT INTO associations (source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, ?, ?)",
      { 1LL, 40LL, std::string ("co_occurs"), 1.0 });

  auto s = MakeSignal (cur, /*ts=*/400);
  processor.Process (s);
  processor.Flush ();

  // v2: Neighbor via co_occurs should also receive ripple
  {
    auto rows = store->Execute (
        "SELECT lability_state FROM memories WHERE memory_id = ?",
        { 40LL });
    REQUIRE (rows.size () == 1);
    const double lab = std::any_cast<double> (rows[0].at ("lability_state"));
    REQUIRE (lab > 0.0);
  }
}

TEST_CASE ("Alg20 bounds ripple reconstruction writes but keeps lability",
           "[operations][recon][ripple]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  const Eigen::VectorXf mem = MakeUnitVec256 (0);
  std::unordered_map<long long, Eigen::VectorXf> all_embs{ { 1LL, mem } };
  for (long long i = 0; i < 12; ++i)
    {
      all_embs.emplace (100LL + i,
                        MakeUnitVec256 (static_cast<int> (20 + i)));
    }

  auto seed = std::make_unique<SeedEmbeddingsOp> (all_embs);
  auto setup = std::make_unique<SetupReconInputsOp> (
      cur, std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, mem } });
  auto apply = std::make_unique<ApplyReconsolidation> ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::move (seed), std::move (setup), std::move (apply));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  for (long long i = 0; i < 8; ++i)
    {
      store->Execute (
          "INSERT INTO associations (source_memory_id, target_memory_id, "
          "edge_type, weight) VALUES (?, ?, ?, ?)",
          { 1LL, 100LL + i, std::string ("reinforces"), 1.0 });
    }
  for (long long i = 8; i < 12; ++i)
    {
      store->Execute (
          "INSERT INTO associations (source_memory_id, target_memory_id, "
          "edge_type, weight) VALUES (?, ?, ?, ?)",
          { 100LL, 100LL + i, std::string ("reinforces"), 1.0 });
    }

  processor.Process (MakeSignal (cur, /*ts=*/500));
  processor.Flush ();

  const int reconstruction_limit
      = core::ReconsolidationRippleReconstructionLimit (
          cfg.focus, cfg.sensitivity, cfg.stability);
  REQUIRE (reconstruction_limit < 12);

  auto recon_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memory_reconstructions "
      "WHERE memory_id >= 100 AND trigger = 'reconsolidation'",
      {});
  REQUIRE (recon_rows.size () == 1);
  REQUIRE (std::any_cast<long long> (recon_rows[0].at ("cnt"))
           == reconstruction_limit);

  auto lability_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memories "
      "WHERE memory_id >= 100 AND lability_state > 0.0",
      {});
  REQUIRE (lability_rows.size () == 1);
  REQUIRE (std::any_cast<long long> (lability_rows[0].at ("cnt")) == 12LL);
}
