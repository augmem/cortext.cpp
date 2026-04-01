// tests/operations_influence_feedback.test.cpp
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/core/algorithms.hpp>
#include <cortext/operations/influence.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cmath>

using namespace cortext;
using cortext::operations::ApplyInfluenceFeedback;

namespace
{

constexpr int kEmbeddingDim = 256;

static void
SeedMemory (Store *store, long long id)
{
  std::vector<float> vec (kEmbeddingDim, 0.0f);
  vec[0] = 1.0f;
  const auto now_ts = cortext::testing::NowMs ();
  store->Execute (
      "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { id, vec, now_ts });
  store->Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, use_frequency, stability, connectivity, drift_mag, "
      "influence, sustained_influence, contextual_gain, redundancy, "
      "pre_activation, lability_state, suppression_count, created_at) "
      "VALUES(?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, "
      "1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, ?)",
      { id, id, now_ts, now_ts });
}

static Eigen::VectorXf
MakeUnitVec256 (int idx)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[idx] = 1.0f;
  return v;
}

// Helper op to attach embeddings and events to the context.
class SetupInfluenceInputsOp : public IOperation
{
public:
  SetupInfluenceInputsOp (
      Eigen::VectorXf prev, Eigen::VectorXf cur,
      std::vector<OperationContext::MemoryUsageEvent> events,
      std::unordered_map<long long, Eigen::VectorXf> embs)
      : prev_ (std::move (prev)),
        cur_ (std::move (cur)),
        events_ (std::move (events)),
        embeddings_ (std::move (embs))
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    pctx.recent_context_embeddings.clear ();
    pctx.recent_context_embeddings.push_back (prev_);
    pctx.recent_context_embeddings.push_back (cur_);
    ctx.SetMemoryUsageEvents (events_);
    ctx.SetRetrievedMemoryEmbeddings (embeddings_);
  }

private:
  Eigen::VectorXf prev_;
  Eigen::VectorXf cur_;
  std::vector<OperationContext::MemoryUsageEvent> events_;
  std::unordered_map<long long, Eigen::VectorXf> embeddings_;
};

} // namespace

TEST_CASE ("Alg19 influence persists per-memory", "[operations][influence]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  SeedMemory (store.get (), 101LL);

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.6;
  cfg.sensitivity = 0.7;
  cfg.stability = 0.5;

  const Eigen::VectorXf prev = MakeUnitVec256 (0);
  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  const Eigen::VectorXf mem = MakeUnitVec256 (0);

  OperationContext::MemoryUsageEvent ev{ 101LL, true, 0.8 };
  auto setup = std::make_unique<SetupInfluenceInputsOp> (
      prev, cur, std::vector<OperationContext::MemoryUsageEvent>{ ev },
      std::unordered_map<long long, Eigen::VectorXf>{ { 101LL, mem } });
  auto apply = std::make_unique<ApplyInfluenceFeedback> ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (setup), std::move (apply));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  Signal s;
  s.embedding = cur;
  s.timestamp = 2;
  s.source_id = "test";
  processor.Process (s);
  processor.Flush ();

  auto rows = store->Execute (
      "SELECT influence, sustained_influence FROM memories WHERE memory_id = ?",
      { 101LL });
  REQUIRE (rows.size () == 1);
  const double influence
      = std::any_cast<double> (rows[0].at ("influence"));
  const double sustained
      = std::any_cast<double> (rows[0].at ("sustained_influence"));

  // Expected influence: generation terms neutralized when no generation embeddings
  const double expected_influence = 0.5 * 0.8;
  REQUIRE (influence == Catch::Approx (expected_influence).margin (1e-6));

  const double L_sustain = std::round (core::Lerp (3.0, 5.0, cfg.stability));
  const double alpha_sustain = 2.0 / (L_sustain + 1.0);
  const double expected_sustained = alpha_sustain * expected_influence;
  REQUIRE (sustained == Catch::Approx (expected_sustained).margin (1e-6));
}

TEST_CASE ("Alg19 negative influence allowed", "[operations][influence]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  SeedMemory (store.get (), 202LL);

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  const Eigen::VectorXf prev = MakeUnitVec256 (0);
  const Eigen::VectorXf cur = MakeUnitVec256 (1);
  const Eigen::VectorXf mem = MakeUnitVec256 (2);

  OperationContext::MemoryUsageEvent ev{ 202LL, true, -1.0 };
  auto setup = std::make_unique<SetupInfluenceInputsOp> (
      prev, cur, std::vector<OperationContext::MemoryUsageEvent>{ ev },
      std::unordered_map<long long, Eigen::VectorXf>{ { 202LL, mem } });
  auto apply = std::make_unique<ApplyInfluenceFeedback> ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (setup), std::move (apply));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  Signal s;
  s.embedding = cur;
  s.timestamp = 2;
  s.source_id = "test";
  processor.Process (s);
  processor.Flush ();

  auto rows = store->Execute (
      "SELECT influence, sustained_influence FROM memories WHERE memory_id = ?",
      { 202LL });
  REQUIRE (rows.size () == 1);
  const double influence
      = std::any_cast<double> (rows[0].at ("influence"));
  const double sustained
      = std::any_cast<double> (rows[0].at ("sustained_influence"));
  REQUIRE (influence < 0.0);
  REQUIRE (sustained < 0.0);
}
