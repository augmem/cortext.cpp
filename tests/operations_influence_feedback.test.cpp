// tests/operations_influence_feedback.test.cpp
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/algorithms.hpp>
#include <cortext/operations/focus.hpp>
#include <cortext/operations/influence.hpp>
#include <cortext/operations/sensitivity.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;
using cortext::operations::ApplyInfluenceFeedback;
using cortext::operations::InitializeFocusPriors;
using cortext::operations::InitializeSensitivityPriors;

namespace
{

// Helper op to push embeddings (prev then cur) and attach events/embeddings.
class SetupInfluenceInputsOp : public IOperation
{
public:
  SetupInfluenceInputsOp (
      Eigen::VectorXf prev, Eigen::VectorXf cur,
      std::vector<OperationContext::MemoryUsageEvent> events,
      std::unordered_map<long long, Eigen::VectorXf> embs)
      : prev_ (std::move (prev)), cur_ (std::move (cur)),
        events_ (std::move (events)), embeddings_ (std::move (embs))
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
    // Store baselines as metrics for later assertion
    ctx.SetMetric (operations::Metric::aw_prev, pctx.attention_width);
    const double rate_prev = (pctx.rate_target == 0.0) ? pctx.rate_target_prior
                                                       : pctx.rate_target;
    ctx.SetMetric (operations::Metric::rate_prev, rate_prev);
    ctx.SetMetric (operations::Metric::hys_prev, pctx.hysteresis);
    // Ensure a well-defined baseline for rate_target
    if (pctx.rate_target == 0.0)
      {
        pctx.rate_target = pctx.rate_target_prior;
      }
  }

private:
  Eigen::VectorXf prev_;
  Eigen::VectorXf cur_;
  std::vector<OperationContext::MemoryUsageEvent> events_;
  std::unordered_map<long long, Eigen::VectorXf> embeddings_;
};

// Assert op reading baselines from metrics and comparing to current pctx
class AssertPositiveEffectsOp : public IOperation
{
public:
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    const auto &cfg = ctx.GetConfig ();
    const double kAttentionWidthMin = 0.1 * 3.14159;
    const double kAttentionWidthMax = 3.14159;
    const double aw_prev
        = core::Lerp (kAttentionWidthMin, kAttentionWidthMax, 1.0 - cfg.focus);
    REQUIRE (pctx.attention_width < aw_prev);
    REQUIRE (pctx.rate_target > pctx.rate_target_prior);
    REQUIRE (pctx.hysteresis > 0.05);
  }
};

class AssertNegativeEffectsOp : public IOperation
{
public:
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    const auto &cfg = ctx.GetConfig ();
    const double kAttentionWidthMin = 0.1 * 3.14159;
    const double kAttentionWidthMax = 3.14159;
    const double aw_prev
        = core::Lerp (kAttentionWidthMin, kAttentionWidthMax, 1.0 - cfg.focus);
    REQUIRE (pctx.attention_width > aw_prev);
    REQUIRE (pctx.rate_target
             <= Catch::Approx (pctx.rate_target_prior).epsilon (0.05));
  }
};

class AssertNoopEffectsOp : public IOperation
{
public:
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    const auto &cfg = ctx.GetConfig ();
    const double kAttentionWidthMin = 0.1 * 3.14159;
    const double kAttentionWidthMax = 3.14159;
    const double aw_prev
        = core::Lerp (kAttentionWidthMin, kAttentionWidthMax, 1.0 - cfg.focus);
    REQUIRE (pctx.attention_width == Catch::Approx (aw_prev));
    const double rt = pctx.rate_target;
    const double rtp = pctx.rate_target_prior;
    const bool ok
        = (std::abs (rt - 0.0) < 1e-6) || (std::abs (rt - rtp) < 1e-6);
    REQUIRE (ok);
    REQUIRE (pctx.hysteresis == Catch::Approx (0.05));
  }
};

static std::unique_ptr<cortext::OperationSet>
MakePipeline (const Eigen::VectorXf &prev, const Eigen::VectorXf &cur,
              double F, double S, double T,
              std::vector<OperationContext::MemoryUsageEvent> events,
              std::unordered_map<long long, Eigen::VectorXf> embeddings,
              std::unique_ptr<IOperation> assert_op)
{
  auto init_focus = std::make_unique<InitializeFocusPriors> ();
  auto init_sens = std::make_unique<InitializeSensitivityPriors> ();
  auto setup = std::make_unique<SetupInfluenceInputsOp> (
      prev, cur, std::move (events), std::move (embeddings));
  auto apply = std::make_unique<ApplyInfluenceFeedback> ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (init_focus), std::move (init_sens), std::move (setup),
      std::move (apply), std::move (assert_op));
  return ops;
}

} // namespace

TEST_CASE (
    "Alg19 positive mean influence narrows width, raises rate and hysteresis",
    "[operations][influence]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  SignalProcessor::Config cfg;
  cfg.focus = 0.6;
  cfg.sensitivity = 0.7;
  cfg.stability = 0.5;

  const Eigen::VectorXf prev = (Eigen::VectorXf (4) << 1, 0, 0, 0).finished ();
  const Eigen::VectorXf cur
      = (Eigen::VectorXf (4) << 1, 0.05f, 0, 0).finished ();
  const Eigen::VectorXf mem = (Eigen::VectorXf (4) << 1, 0, 0, 0).finished ();
  OperationContext::MemoryUsageEvent ev{ 101LL, true, 0.8 };

  auto ops = MakePipeline (
      prev, cur, cfg.focus, cfg.sensitivity, cfg.stability, { ev },
      std::unordered_map<long long, Eigen::VectorXf>{ { 101LL, mem } },
      std::make_unique<AssertPositiveEffectsOp> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  Signal s;
  s.embedding = cur;
  s.timestamp = 2;
  s.source_id = "test";
  processor.Process (s);
  processor.Flush ();
}

TEST_CASE (
    "Alg19 negative mean influence widens width and does not raise rate",
    "[operations][influence]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  const Eigen::VectorXf prev = (Eigen::VectorXf (4) << 1, 0, 0, 0).finished ();
  const Eigen::VectorXf cur = (Eigen::VectorXf (4) << 0, 1, 0, 0).finished ();
  const Eigen::VectorXf mem = (Eigen::VectorXf (4) << 0, 0, 1, 0).finished ();
  OperationContext::MemoryUsageEvent ev{ 202LL, true, -1.0 };

  auto ops = MakePipeline (
      prev, cur, cfg.focus, cfg.sensitivity, cfg.stability, { ev },
      std::unordered_map<long long, Eigen::VectorXf>{ { 202LL, mem } },
      std::make_unique<AssertNegativeEffectsOp> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  Signal s;
  s.embedding = cur;
  s.timestamp = 2;
  s.source_id = "test";
  processor.Process (s);
  processor.Flush ();
}

TEST_CASE ("Alg19 no embeddings → no change", "[operations][influence]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  const Eigen::VectorXf prev = (Eigen::VectorXf (4) << 1, 0, 0, 0).finished ();
  const Eigen::VectorXf cur = (Eigen::VectorXf (4) << 1, 0, 0, 0).finished ();
  OperationContext::MemoryUsageEvent ev{ 303LL, true, 0.6 };

  auto ops
      = MakePipeline (prev, cur, cfg.focus, cfg.sensitivity, cfg.stability,
                      { ev }, std::unordered_map<long long, Eigen::VectorXf>{},
                      std::make_unique<AssertNoopEffectsOp> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  Signal s;
  s.embedding = cur;
  s.timestamp = 1;
  s.source_id = "test";
  processor.Process (s);
  processor.Flush ();
}
