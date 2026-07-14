#include "test_helpers.hpp"
#include "../src/operations/constructive_recall_internal.hpp"
#include "../src/operations/historical_surface_search_cache_internal.hpp"
#include "../src/operations/retrieval_trace_state.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cortext/core/algorithms.hpp>
#include <cortext/core/utils.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/operations/memory_storage.hpp>
#include <cortext/operations/reconsolidation.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/accumulator_state.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>

#include <Eigen/Dense>

#include <memory>
#include <unordered_map>
#include <vector>

using namespace cortext;

namespace
{

constexpr int kEmbeddingDim = 256;

Eigen::VectorXf
MakeVec (std::initializer_list<std::pair<int, float>> values)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  for (const auto &[idx, value] : values)
    {
      if (idx >= 0 && idx < kEmbeddingDim)
        {
          v[idx] = value;
        }
    }
  const float norm = v.norm ();
  if (norm > 1e-9f)
    {
      v /= norm;
    }
  return v;
}

std::vector<float>
ToFloatVec (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

Signal
MakeSignal (const Eigen::VectorXf &embedding, uint64_t ts)
{
  Signal s;
  s.embedding = embedding;
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

class ForceRetrievalGateOp : public IOperation
{
public:
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetShouldCheckRetrieval (true);
    auto &pctx = ctx.GetProcessorContext ();
    if (pctx.memory_stream.empty ())
      {
        pctx.memory_stream.push_back (ctx.GetSignal ().embedding);
      }
    auto &acc = pctx.accumulator_states[ctx.GetSignal ().source_id];
    acc.mu_acc = ctx.GetSignal ().embedding;
    acc.c_t = ctx.GetSignal ().embedding;
  }
};

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

Eigen::VectorXf
LoadEmbeddingById (Store &store, long long embedding_id)
{
  auto rows = store.Execute (
      "SELECT embedding FROM embeddings WHERE embedding_id = ?",
      { embedding_id });
  REQUIRE (rows.size () == 1);
  Eigen::VectorXf out;
  REQUIRE (core::DecodeFloatBlob (rows[0].at ("embedding"), kEmbeddingDim, out));
  return out;
}

} // namespace

TEST_CASE ("MemoryStorage seeds an initial reconstruction version",
           "[operations][constructive_recall]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  Signal s;
  s.embedding = MakeVec ({ { 0, 1.0f } });
  s.timestamp = 12345;
  s.source_id = "test";
  s.modality = "text";
  s.mimetype = "text/plain";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  AccumulatorState acc;
  acc.mu_acc = s.embedding;
  acc.n_signals = 1;
  acc.s_sum = 0.5;
  acc.s_max = 0.5;
  acc.t_start = s.timestamp - 1000;
  {
    SignalRecord rec;
    rec.embedding = s.embedding;
    rec.timestamp = s.timestamp;
    rec.modality = s.modality;
    rec.mime = s.mimetype;
    rec.score = 0.5;
    rec.serial_position = 0;
    acc.signals.push_back (std::move (rec));
  }
  pctx.accumulator_states[s.source_id] = std::move (acc);

  OperationContext ctx (s, pctx, cfg, store.get ());
  ctx.SetAccumulatorWriteDecision (true);
  ctx.SetRepresentativeEmbedding (s.embedding);

  operations::MemoryStorage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  const auto stored_id = ctx.GetStoredEmbeddingId ();
  REQUIRE (stored_id.has_value ());

  auto mem_rows = store->Execute (
      "SELECT memory_id FROM memories WHERE embedding_id = ?",
      { *stored_id });
  REQUIRE (mem_rows.size () == 1);
  const long long memory_id
      = cortext::store::AnyToLongLong (mem_rows[0].at ("memory_id")).value_or (0);
  REQUIRE (memory_id > 0);

  auto recon_rows = store->Execute (
      "SELECT embedding_id, uncertainty, trigger "
      "FROM memory_reconstructions WHERE memory_id = ? "
      "ORDER BY reconstruction_id",
      { memory_id });
  REQUIRE (recon_rows.size () == 1);
  REQUIRE (cortext::store::AnyToLongLong (recon_rows[0].at ("embedding_id"))
               .value_or (0)
           == *stored_id);
  REQUIRE (cortext::testing::GetDouble (recon_rows[0], "uncertainty")
           == Catch::Approx (0.0));
  REQUIRE (std::any_cast<std::string> (recon_rows[0].at ("trigger"))
           == "initial");
}

TEST_CASE ("Constructive recall bounds reconstruction history and keeps current surface",
           "[operations][constructive_recall]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf base = MakeVec ({ { 0, 1.0f } });
  cortext::testing::SeedEmbeddingV2 (*store, 1LL, base, 1);
  cortext::testing::SeedMemoryV2 (*store, 1LL, 1LL, "test", "LONG_TERM", 1.0,
                                  1);

  operations::constructive_recall::ReconstructionUpdatePolicy policy;
  policy.history_limit = 3;
  policy.prune_batch_limit = 16;
  policy.delete_orphan_embeddings = true;
  ProcessorContext pctx;
  namespace cache
      = operations::historical_surface_search_cache_internal;
  REQUIRE (cache::Reset (
      pctx, { { 1LL, 1LL, 1LL, "LONG_TERM", "test", base } }));

  Eigen::VectorXf latest = base;
  {
    auto tx = store->Begin ();
    operations::constructive_recall::AppendReconstructionWithEmbeddingId (
        *tx, 1LL, 1LL, {}, 1LL, 0.0, "initial", 1.0, 1.0, policy);
    for (int i = 0; i < 8; ++i)
      {
        latest = MakeVec ({ { i % 4, 1.0f }, { (i + 1) % 4, 0.1f } });
        operations::constructive_recall::AppendReconstructionWithEmbedding (
            *tx, 1LL, latest, {}, 10LL + i, 0.2, "retrieval", 0.8, 0.7,
            policy, &pctx);
      }
    tx->Commit ();
  }

  auto recon_rows = store->Execute (
      "SELECT reconstruction_id, embedding_id, trigger "
      "FROM memory_reconstructions WHERE memory_id = ? "
      "ORDER BY reconstruction_id DESC",
      { 1LL });
  REQUIRE (recon_rows.size () == 3);
  REQUIRE (std::any_cast<std::string> (recon_rows.front ().at ("trigger"))
           == "retrieval");

  const long long latest_embedding_id
      = cortext::store::AnyToLongLong (
            recon_rows.front ().at ("embedding_id")).value_or (0);
  REQUIRE (latest_embedding_id > 1LL);

  auto current_rows = store->Execute (
      "SELECT embedding_id FROM current_memory_embeddings WHERE memory_id = ?",
      { 1LL });
  REQUIRE (current_rows.size () == 1);
  REQUIRE (cortext::store::AnyToLongLong (
               current_rows[0].at ("embedding_id")).value_or (0)
           == latest_embedding_id);

  auto embedding_rows = store->Execute (
      "SELECT embedding_id FROM embeddings ORDER BY embedding_id", {});
  const auto cache_owner = cache::Find (pctx);
  REQUIRE (cache_owner != nullptr);
  REQUIRE (cache_owner->entries.size () == embedding_rows.size ());
  for (const auto &row : embedding_rows)
    {
      const long long embedding_id = cortext::store::AnyToLongLong (
          row.at ("embedding_id")).value_or (0);
      REQUIRE (cache_owner->embedding_index.count (embedding_id) == 1);
    }
  cache::Erase (pctx);

  const auto current = operations::constructive_recall::LoadCurrentEmbedding (
      store.get (), 1LL, 1LL, kEmbeddingDim);
  REQUIRE (current.has_value ());
  REQUIRE (core::CosineSimilarity (*current, latest)
           == Catch::Approx (1.0).margin (1e-5));

  auto old_embedding_rows = store->Execute (
      "SELECT embedding FROM embeddings WHERE embedding_id = ?",
      { 2LL });
  REQUIRE (old_embedding_rows.empty ());
  REQUIRE (core::CosineSimilarity (LoadEmbeddingById (*store, 1LL), base)
           == Catch::Approx (1.0).margin (1e-5));
}

TEST_CASE ("Constructive recall ignores non-positive prune env overrides",
           "[operations][constructive_recall]")
{
  cortext::testing::ScopedEnvVar history_limit (
      "CORTEXT_RECONSTRUCTION_HISTORY_LIMIT", "0");
  cortext::testing::ScopedEnvVar prune_batch_limit (
      "CORTEXT_RECONSTRUCTION_PRUNE_BATCH_LIMIT", "0");

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf base = MakeVec ({ { 0, 1.0f } });
  cortext::testing::SeedEmbeddingV2 (*store, 1LL, base, 1);
  cortext::testing::SeedMemoryV2 (*store, 1LL, 1LL, "test", "LONG_TERM", 1.0,
                                  1);

  operations::constructive_recall::ReconstructionUpdatePolicy policy;
  policy.history_limit = 3;
  policy.prune_batch_limit = 16;

  {
    auto tx = store->Begin ();
    for (int i = 0; i < 8; ++i)
      {
        const auto embedding
            = MakeVec ({ { i % 4, 1.0f }, { (i + 1) % 4, 0.1f } });
        operations::constructive_recall::AppendReconstructionWithEmbedding (
            *tx, 1LL, embedding, {}, 10LL + i, 0.2, "retrieval", 0.8, 0.7,
            policy);
      }
    tx->Commit ();
  }

  auto recon_rows = store->Execute (
      "SELECT reconstruction_id FROM memory_reconstructions "
      "WHERE memory_id = ? ORDER BY reconstruction_id DESC",
      { 1LL });
  REQUIRE (recon_rows.size () == 3);

  auto old_embedding_rows = store->Execute (
      "SELECT embedding FROM embeddings WHERE embedding_id = ?",
      { 2LL });
  REQUIRE (old_embedding_rows.size () == 1);
}

TEST_CASE ("Constructive recall cooldown skips dense rewrites",
           "[operations][constructive_recall]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf base = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf first = MakeVec ({ { 1, 1.0f } });
  const Eigen::VectorXf skipped = MakeVec ({ { 2, 1.0f } });
  const Eigen::VectorXf later = MakeVec ({ { 3, 1.0f } });

  cortext::testing::SeedEmbeddingV2 (*store, 1LL, base, 1);
  cortext::testing::SeedMemoryV2 (*store, 1LL, 1LL, "test", "LONG_TERM", 1.0,
                                  1);

  operations::constructive_recall::ReconstructionUpdatePolicy policy;
  policy.history_limit = 4;
  policy.prune_batch_limit = 16;
  policy.min_update_interval_ms = 1000;

  long long first_id = 0;
  long long skipped_id = 0;
  long long later_id = 0;
  {
    auto tx = store->Begin ();
    operations::constructive_recall::AppendReconstructionWithEmbeddingId (
        *tx, 1LL, 1LL, {}, 1000LL, 0.0, "initial", 1.0, 1.0, policy);
    first_id = operations::constructive_recall::AppendReconstructionWithEmbedding (
        *tx, 1LL, first, {}, 2500LL, 0.2, "retrieval", 0.8, 0.7,
        policy);
    skipped_id
        = operations::constructive_recall::AppendReconstructionWithEmbedding (
            *tx, 1LL, skipped, {}, 2800LL, 0.2, "retrieval", 0.8, 0.7,
            policy);
    later_id = operations::constructive_recall::AppendReconstructionWithEmbedding (
        *tx, 1LL, later, {}, 4000LL, 0.2, "retrieval", 0.8, 0.7,
        policy);
    tx->Commit ();
  }

  REQUIRE (first_id > 0);
  REQUIRE (skipped_id == 0);
  REQUIRE (later_id > first_id);

  auto recon_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memory_reconstructions WHERE memory_id = ?",
      { 1LL });
  REQUIRE (cortext::testing::GetInt64 (recon_rows[0], "cnt") == 3);

  const auto current = operations::constructive_recall::LoadCurrentEmbedding (
      store.get (), 1LL, 1LL, kEmbeddingDim);
  REQUIRE (current.has_value ());
  REQUIRE (core::CosineSimilarity (*current, later)
           == Catch::Approx (1.0).margin (1e-5));
}

TEST_CASE ("Constructive recall can load latest ledger version when surface is stale",
           "[operations][constructive_recall]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf base = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf first = MakeVec ({ { 1, 1.0f } });
  const Eigen::VectorXf latest = MakeVec ({ { 2, 1.0f } });

  cortext::testing::SeedEmbeddingV2 (*store, 1LL, base, 1);
  cortext::testing::SeedMemoryV2 (*store, 1LL, 1LL, "test", "LONG_TERM", 1.0,
                                  1);

  operations::constructive_recall::ReconstructionUpdatePolicy policy;
  policy.history_limit = 4;
  policy.prune_batch_limit = 16;
  policy.min_update_interval_ms = 0;

  {
    auto tx = store->Begin ();
    operations::constructive_recall::AppendReconstructionWithEmbedding (
        *tx, 1LL, first, {}, 1000LL, 0.2, "retrieval", 0.8, 0.7, policy);
    policy.update_current_surface = false;
    operations::constructive_recall::AppendReconstructionWithEmbedding (
        *tx, 1LL, latest, {}, 2000LL, 0.2, "reconsolidation", 0.8, 0.7,
        policy);
    tx->Commit ();
  }

  auto current_rows = store->Execute (
      "SELECT embedding_id FROM current_memory_embeddings WHERE memory_id = ?",
      { 1LL });
  REQUIRE (current_rows.size () == 1);

  auto latest_rows = store->Execute (
      "SELECT embedding_id FROM memory_reconstructions "
      "WHERE memory_id = ? ORDER BY reconstruction_id DESC LIMIT 1",
      { 1LL });
  REQUIRE (latest_rows.size () == 1);
  REQUIRE (cortext::store::AnyToLongLong (
               current_rows[0].at ("embedding_id")).value_or (0)
           != cortext::store::AnyToLongLong (
                  latest_rows[0].at ("embedding_id")).value_or (0));

  const auto loaded = operations::constructive_recall::LoadCurrentEmbedding (
      store.get (), 1LL, 1LL, kEmbeddingDim);
  REQUIRE (loaded.has_value ());
  REQUIRE (core::CosineSimilarity (*loaded, latest)
           == Catch::Approx (1.0).margin (1e-5));
}

TEST_CASE ("Constructive recall retrieval uses the latest reconstruction and appends a new version",
           "[operations][constructive_recall][graph]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf target_base = MakeVec ({ { 1, 1.0f } });
  const Eigen::VectorXf target_reconstructed = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf rival = MakeVec ({ { 0, 0.92f }, { 1, 0.39f } });

  cortext::testing::SeedEmbeddingV2 (*store, 11LL, target_base, 1);
  cortext::testing::SeedMemoryV2 (*store, 11LL, 11LL, "test", "LONG_TERM",
                                  1.0, 1);
  cortext::testing::SeedEmbeddingV2 (*store, 22LL, rival, 1);
  cortext::testing::SeedMemoryV2 (*store, 22LL, 22LL, "test", "LONG_TERM",
                                  1.0, 1);
  cortext::testing::SeedEmbeddingV2 (*store, 111LL, target_reconstructed, 2);
  {
    auto tx = store->Begin ();
    operations::constructive_recall::AppendReconstructionWithEmbeddingId (
        *tx, 11LL, 111LL, {}, 2LL, 0.0, "initial");
    tx->Commit ();
  }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cortext::testing::ScopedEnvVar disable_source_seed_expansion (
      "CORTEXT_DISABLE_SOURCE_SEED_GRAPH_EXPANSION", "1");

  auto run = [&] {
    operations::retrieval_trace::ClearLastRankedCandidates ();
    auto ops = std::make_unique<DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<operations::GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    processor.Process (MakeSignal (query, 4000000));
    processor.Flush ();
    return operations::retrieval_trace::GetLastRankedCandidates ();
  };

  {
    cortext::testing::ScopedEnvVar disable (
        "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
    const auto ranked = run ();
    REQUIRE_FALSE (ranked.empty ());
#if defined(CORTEXT_EXPERIMENT_HOOKS)
    REQUIRE (ranked.front ().memory_id == 22LL);
#else
    REQUIRE (ranked.front ().memory_id == 11LL);
#endif
    auto recon_rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM memory_reconstructions WHERE memory_id = ?",
        { 11LL });
#if defined(CORTEXT_EXPERIMENT_HOOKS)
    REQUIRE (cortext::testing::GetInt64 (recon_rows[0], "cnt") == 1);
#else
    REQUIRE (cortext::testing::GetInt64 (recon_rows[0], "cnt") == 2);
#endif
  }

  {
    cortext::testing::ScopedEnvVar enable ("CORTEXT_DISABLE_CONSTRUCTIVE_RECALL");
    cortext::testing::ScopedEnvVar disable_cooldown (
        "CORTEXT_RECONSTRUCTION_MIN_UPDATE_MS", "0");
    const auto ranked = run ();
    REQUIRE_FALSE (ranked.empty ());
    REQUIRE (ranked.front ().memory_id == 11LL);
    auto recon_rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM memory_reconstructions WHERE memory_id = ?",
        { 11LL });
    REQUIRE (cortext::testing::GetInt64 (recon_rows[0], "cnt") == 2);

    auto latest_rows = store->Execute (
        "SELECT trigger, uncertainty FROM memory_reconstructions "
        "WHERE memory_id = ? ORDER BY reconstruction_id DESC LIMIT 1",
        { 11LL });
    REQUIRE (latest_rows.size () == 1);
    REQUIRE (std::any_cast<std::string> (latest_rows[0].at ("trigger"))
             == "retrieval");
#if defined(CORTEXT_EXPERIMENT_HOOKS)
    REQUIRE (cortext::testing::GetDouble (latest_rows[0], "uncertainty") > 0.0);
#else
    REQUIRE (cortext::testing::GetDouble (latest_rows[0], "uncertainty")
             == Catch::Approx (0.0));
#endif
  }
}

TEST_CASE ("Reconsolidation appends a new reconstruction while preserving the evidence embedding",
           "[operations][constructive_recall][recon]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf evidence = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf current = MakeVec ({ { 0, 0.4f }, { 1, 0.9165f } });

  cortext::testing::SeedEmbeddingV2 (*store, 1LL, evidence, 1);
  cortext::testing::SeedMemoryV2 (*store, 1LL, 1LL, "test", "LONG_TERM", 1.0,
                                  1);
  {
    auto tx = store->Begin ();
    operations::constructive_recall::AppendReconstructionWithEmbeddingId (
        *tx, 1LL, 1LL, {}, 1LL, 0.0, "initial");
    tx->Commit ();
  }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<SetupReconInputsOp> (
          current,
          std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, evidence } }),
      std::make_unique<operations::ApplyReconsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (current, 4000000));
  processor.Flush ();

  auto mem_rows = store->Execute (
      "SELECT embedding_id FROM memories WHERE memory_id = ?",
      { 1LL });
  REQUIRE (mem_rows.size () == 1);
  REQUIRE (cortext::store::AnyToLongLong (mem_rows[0].at ("embedding_id"))
               .value_or (0)
           == 1LL);

  auto recon_rows = store->Execute (
      "SELECT reconstruction_id, embedding_id, trigger "
      "FROM memory_reconstructions WHERE memory_id = ? "
      "ORDER BY reconstruction_id",
      { 1LL });
  REQUIRE (recon_rows.size () == 2);
  REQUIRE (std::any_cast<std::string> (recon_rows.back ().at ("trigger"))
           == "reconsolidation");
  const long long latest_embedding_id
      = cortext::store::AnyToLongLong (recon_rows.back ().at ("embedding_id"))
            .value_or (0);
  REQUIRE (latest_embedding_id > 1LL);

  const Eigen::VectorXf evidence_after = LoadEmbeddingById (*store, 1LL);
  const Eigen::VectorXf reconstructed_after
      = LoadEmbeddingById (*store, latest_embedding_id);

  REQUIRE (core::CosineSimilarity (evidence_after, evidence)
           == Catch::Approx (1.0).margin (1e-5));
  REQUIRE (core::CosineSimilarity (reconstructed_after, current) > 0.45);
  REQUIRE (core::CosineSimilarity (reconstructed_after, current)
           > core::CosineSimilarity (evidence_after, current));
  REQUIRE (core::CosineSimilarity (reconstructed_after, evidence) < 0.999);
}
