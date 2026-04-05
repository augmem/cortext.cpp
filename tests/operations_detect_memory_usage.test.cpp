#include "test_helpers.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/sparse.hpp>
#include <cortext/operations/detect_memory_usage.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <optional>
#include <string>
#include <unordered_map>

using namespace cortext;

namespace
{
constexpr int kEmbeddingDim = 256;

Eigen::VectorXf
MakeVec ()
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = 1.0f;
  return v;
}

Signal
MakeSignal (uint64_t ts)
{
  Signal s;
  s.embedding = MakeVec ();
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

double
RunProceduralUpdate (double neuromod_da, bool disable_scale)
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  cortext::testing::SeedEmbeddingV2 (*store, 7LL, MakeVec (), 1);
  cortext::testing::SeedMemoryV2 (*store, 7LL, 7LL, "test", "LONG_TERM", 1.0,
                                  1);

  ProcessorContext pctx;
  pctx.neuromod_da = neuromod_da;
  pctx.delta_reward = 0.8;

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.procedural_enabled = true;

  auto signal = MakeSignal (10);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetInterruptAllowed (true);
  ctx.SetSelectedCandidateId (7LL);
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{ { 7LL, MakeVec () } });

  std::optional<cortext::testing::ScopedEnvVar> disable_guard;
  if (disable_scale)
    {
      disable_guard.emplace ("CORTEXT_DISABLE_NEUROMOD_VALUE_GAIN", "1");
    }

  operations::DetectMemoryUsage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  const int k_key = core::SparseKeySize (cfg.focus);
  const std::string key = core::SparseKey (signal.embedding, k_key);
  return pctx.procedural_store[key][7LL];
}

} // namespace

TEST_CASE ("High DA increases procedural value update gain",
           "[operations][detect_memory_usage][neuromod]")
{
  const double low_da = RunProceduralUpdate (0.0, false);
  const double high_da = RunProceduralUpdate (1.0, false);
  const double disabled = RunProceduralUpdate (1.0, true);

  REQUIRE (high_da > low_da);
  REQUIRE (disabled == Catch::Approx (0.8).margin (1e-6));
}
