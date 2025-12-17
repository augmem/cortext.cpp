// tests/operations_precision_modulation.test.cpp
#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>

#include <cortext/core/knobs.hpp>
#include <cortext/operations/precision.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/signal.hpp>

using namespace cortext;
using cortext::operations::UpdatePrecisionDelta;

namespace
{
static Signal
MakeSignal ()
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  s.timestamp = 0;
  s.source_id = "test";
  return s;
}
} // namespace

TEST_CASE ("§5.5 sets negative Δ when retrieval precision below target",
           "[operations][precision][threshold]")
{
  ProcessorContext pc;
  SignalProcessor::Config cfg;
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto sig = MakeSignal ();
  OperationContext ctx (sig, pc, cfg);

  // retrieved=10, used=2 => precision=0.2
  std::vector<OperationContext::MemoryUsageEvent> events;
  for (int i = 0; i < 10; ++i)
    {
      events.push_back (OperationContext::MemoryUsageEvent{
          /*embedding_id=*/i + 1, /*used=*/(i < 2), /*contextual_gain=*/0.1 });
    }
  ctx.SetMemoryUsageEvents (events);

  const double target
      = cortext::core::TargetPrecision (cfg.focus, cfg.stability);
  REQUIRE (target > 0.2);

  UpdatePrecisionDelta op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetDeltaThresholdPrecision ().has_value ());
  const double delta = *ctx.GetDeltaThresholdPrecision ();
  REQUIRE (delta < 0.0);
}

TEST_CASE ("§5.5 sets zero Δ when retrieval precision meets/exceeds target",
           "[operations][precision][threshold]")
{
  ProcessorContext pc;
  SignalProcessor::Config cfg;
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 1.0;

  auto sig = MakeSignal ();
  OperationContext ctx (sig, pc, cfg);

  // With low focus and high stability, target_precision is modest.
  const double target
      = cortext::core::TargetPrecision (cfg.focus, cfg.stability);

  // retrieved=4, used=4 => precision=1.0 >= target
  ctx.SetMemoryUsageEvents ({
      { 1, true, 0.1 },
      { 2, true, 0.1 },
      { 3, true, 0.1 },
      { 4, true, 0.1 },
  });

  UpdatePrecisionDelta op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetDeltaThresholdPrecision ().has_value ());
  REQUIRE (*ctx.GetDeltaThresholdPrecision () == Catch::Approx (0.0));
  REQUIRE (1.0 >= target);
}

