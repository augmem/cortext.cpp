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
            "s_max, s_avg, strength, drift_mag, "
            "lability_state, suppression_count, lability_ts, created_at) "
            "VALUES(?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, "
            "1.0, 0.0, 0.0, 0, 0, ?)",
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
  auto signal = MakeSignal (cur, 100);
  OperationContext ctx (signal, pctx, cfg, store.get ());
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
#if defined(CORTEXT_EXPERIMENT_HOOKS)
  REQUIRE (target_embedding != 420LL);
#else
  REQUIRE (target_embedding == 420LL);
#endif
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
#if defined(CORTEXT_EXPERIMENT_HOOKS)
  REQUIRE (pctx.retrieval_surface_cache[sibling_cache_it->second].memory_id
           == 101LL);
#else
  REQUIRE (pctx.retrieval_surface_cache[sibling_cache_it->second].memory_id
           == 100LL);
#endif
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

  auto signal = MakeSignal (cur, 2'000'000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
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
#if defined(CORTEXT_EXPERIMENT_HOOKS)
  REQUIRE (sim_scaled > sim_unscaled);
#else
  REQUIRE (sim_scaled == Catch::Approx (sim_unscaled));
#endif
}
