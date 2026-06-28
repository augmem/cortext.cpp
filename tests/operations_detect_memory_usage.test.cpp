#include "test_helpers.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/sparse.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/detect_memory_usage.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <any>
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
AnyToDouble (const std::any &v)
{
  if (v.type () == typeid (double))
    {
      return std::any_cast<double> (v);
    }
  if (v.type () == typeid (float))
    {
      return static_cast<double> (std::any_cast<float> (v));
    }
  if (v.type () == typeid (long long))
    {
      return static_cast<double> (std::any_cast<long long> (v));
    }
  if (v.type () == typeid (int))
    {
      return static_cast<double> (std::any_cast<int> (v));
    }
  return 0.0;
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

  const int k_key
      = core::SparseKeySize (cfg.focus, cfg.sensitivity, cfg.stability);
  const std::string key = core::SparseKey (signal.embedding, k_key);
  return pctx.procedural_store[key][7LL];
}

double
RunReinforcementUpdate (double focus, double sensitivity, double stability,
                        bool selected)
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  cortext::testing::SeedEmbeddingV2 (*store, 7LL, MakeVec (), 1);
  cortext::testing::SeedEmbeddingV2 (*store, 8LL, MakeVec (), 1);
  cortext::testing::SeedMemoryV2 (*store, 7LL, 7LL, "test", "LONG_TERM", 1.0,
                                  1);
  cortext::testing::SeedMemoryV2 (*store, 8LL, 8LL, "test", "LONG_TERM", 1.0,
                                  1);

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = focus;
  cfg.sensitivity = sensitivity;
  cfg.stability = stability;
  cfg.reinforcement_enabled = true;

  auto signal = MakeSignal (10);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetInterruptAllowed (selected);
  if (selected)
    {
      ctx.SetSelectedCandidateId (7LL);
    }
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{
        { 7LL, MakeVec () },
        { 8LL, MakeVec () } });

  operations::DetectMemoryUsage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto rows = store->Execute (
      "SELECT weight FROM associations "
      "WHERE source_memory_id = 7 AND target_memory_id = 8 "
      "AND edge_type = 'reinforces'",
      {});
  REQUIRE (rows.size () == 1);
  return AnyToDouble (rows[0].at ("weight"));
}

} // namespace

TEST_CASE ("DetectMemoryUsage carries memory_id in usage events",
           "[operations][detect_memory_usage]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  cortext::testing::SeedEmbeddingV2 (*store, 700LL, MakeVec (), 1);
  cortext::testing::SeedMemoryV2 (*store, 70LL, 700LL, "test", "LONG_TERM",
                                  1.0, 1);

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);

  auto signal = MakeSignal (10);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetInterruptAllowed (true);
  ctx.SetSelectedCandidateId (700LL);
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{ { 700LL, MakeVec () } });

  operations::DetectMemoryUsage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  const auto &events = ctx.GetMemoryUsageEvents ();
  REQUIRE (events.size () == 1);
  REQUIRE (events[0].embedding_id == 700LL);
  REQUIRE (events[0].memory_id == 70LL);
  REQUIRE (events[0].used);
}

TEST_CASE ("DetectMemoryUsage matches structured selected candidates by memory id",
           "[operations][detect_memory_usage]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  cortext::testing::SeedEmbeddingV2 (*store, 700LL, MakeVec (), 1);
  cortext::testing::SeedMemoryV2 (*store, 70LL, 700LL, "weak", "LONG_TERM",
                                  1.0, 1);
  cortext::testing::SeedMemoryV2 (*store, 71LL, 700LL, "strong", "LONG_TERM",
                                  1.0, 1);

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);

  auto signal = MakeSignal (10);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetInterruptAllowed (true);
  ctx.SetSelectedCandidateId (71LL);
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{ { 700LL, MakeVec () } });
  ctx.SetRetrievedMemoryCandidates (
      std::vector<OperationContext::RetrievedMemoryCandidate>{
        { 70LL, 700LL, MakeVec (), 0.5 },
        { 71LL, 700LL, MakeVec (), 0.9 } });

  operations::DetectMemoryUsage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  const auto &events = ctx.GetMemoryUsageEvents ();
  REQUIRE (events.size () == 2);
  REQUIRE (events[0].memory_id == 70LL);
  REQUIRE_FALSE (events[0].used);
  REQUIRE (events[1].memory_id == 71LL);
  REQUIRE (events[1].used);
}

TEST_CASE ("High DA increases procedural value update gain",
           "[operations][detect_memory_usage][neuromod]")
{
  const double low_da = RunProceduralUpdate (0.0, false);
  const double high_da = RunProceduralUpdate (1.0, false);
  const double disabled = RunProceduralUpdate (1.0, true);

  REQUIRE (high_da > low_da);
  REQUIRE (disabled == Catch::Approx (0.8).margin (1e-6));
}

TEST_CASE ("Reinforcement edge update is knob-derived",
           "[operations][detect_memory_usage][reinforcement]")
{
  const double mid_selected
      = RunReinforcementUpdate (0.5, 0.5, 0.5, true);
  REQUIRE (mid_selected
           == Catch::Approx (
               core::ReinforcementCoRetrievalStep (0.5, 0.5, 0.5))
                  .margin (1e-6));

  const double mid_unselected
      = RunReinforcementUpdate (0.5, 0.5, 0.5, false);
  REQUIRE (mid_unselected
           == Catch::Approx (
               core::ReinforcementCoRetrievalStep (0.5, 0.5, 0.5)
               * core::ReinforcementUnselectedScale (0.5, 0.5, 0.5))
                  .margin (1e-6));
  REQUIRE (mid_unselected < mid_selected);

  const double conservative = RunReinforcementUpdate (1.0, 0.0, 0.0, true);
  const double eager = RunReinforcementUpdate (0.0, 1.0, 1.0, true);
  REQUIRE (eager > conservative);
}
