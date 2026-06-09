#include "test_helpers.hpp"
#include "../src/operations/constructive_recall_internal.hpp"
#include "../src/operations/retrieval_debug_state.hpp"

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
  store->Execute (
      "INSERT INTO memory_reconstructions("
      "memory_id, embedding_id, created_at, uncertainty, trigger"
      ") VALUES(?, ?, ?, ?, 'initial')",
      { 11LL, 111LL, 2LL, 0.0 });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cortext::testing::ScopedEnvVar disable_source_seed_expansion (
      "CORTEXT_DISABLE_SOURCE_SEED_GRAPH_EXPANSION", "1");

  auto run = [&] {
    operations::retrieval_debug::ClearLastRankedCandidates ();
    auto ops = std::make_unique<OperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<operations::GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    processor.Process (MakeSignal (query, 10));
    processor.Flush ();
    return operations::retrieval_debug::GetLastRankedCandidates ();
  };

  {
    cortext::testing::ScopedEnvVar disable (
        "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
    const auto ranked = run ();
    REQUIRE_FALSE (ranked.empty ());
    REQUIRE (ranked.front ().memory_id == 22LL);
    auto recon_rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM memory_reconstructions WHERE memory_id = ?",
        { 11LL });
    REQUIRE (cortext::testing::GetInt64 (recon_rows[0], "cnt") == 1);
  }

  {
    cortext::testing::ScopedEnvVar enable ("CORTEXT_DISABLE_CONSTRUCTIVE_RECALL");
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
    REQUIRE (cortext::testing::GetDouble (latest_rows[0], "uncertainty") > 0.0);
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
  store->Execute (
      "INSERT INTO memory_reconstructions("
      "memory_id, embedding_id, created_at, uncertainty, trigger"
      ") VALUES(?, ?, ?, ?, 'initial')",
      { 1LL, 1LL, 1LL, 0.0 });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

  auto ops = std::make_unique<OperationSet> (
      std::make_unique<SetupReconInputsOp> (
          current,
          std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, evidence } }),
      std::make_unique<operations::ApplyReconsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (current, 100));
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
