#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/operations/write_gate.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::ComputeWriteGate;

TEST_CASE ("ComputeWriteGate accepts score above effective threshold",
           "[operations][write_gate]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;


  OperationContext ctx (s, pctx, cfg);

  // Set composite_score = 0.5, T_dynamic = 0.4, hysteresis = 0.05
  // effective_threshold = 0.4 - 0.05 = 0.35
  // 0.5 > 0.35 => should accept
  ctx.SetCompositeScore (0.5);
  ctx.SetThresholdTDynamic (0.4);
  ctx.SetThresholdHysteresis (0.05);

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetWriteDecision () == true);
}

TEST_CASE ("ComputeWriteGate rejects score below effective threshold",
           "[operations][write_gate]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;


  OperationContext ctx (s, pctx, cfg);

  // Set composite_score = 0.3, T_dynamic = 0.4, hysteresis = 0.05
  // effective_threshold = 0.4 - 0.05 = 0.35
  // 0.3 > 0.35 is false => should reject
  ctx.SetCompositeScore (0.3);
  ctx.SetThresholdTDynamic (0.4);
  ctx.SetThresholdHysteresis (0.05);

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetWriteDecision () == false);
}

TEST_CASE ("ComputeWriteGate rejects score exactly at effective threshold",
           "[operations][write_gate]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;


  OperationContext ctx (s, pctx, cfg);

  // Set composite_score = 0.35, T_dynamic = 0.4, hysteresis = 0.05
  // effective_threshold = 0.4 - 0.05 = 0.35
  // 0.35 > 0.35 is false (strict inequality) => should reject
  ctx.SetCompositeScore (0.35);
  ctx.SetThresholdTDynamic (0.4);
  ctx.SetThresholdHysteresis (0.05);

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetWriteDecision () == false);
}

TEST_CASE ("ComputeWriteGate defaults to reject when no composite score",
           "[operations][write_gate]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;


  OperationContext ctx (s, pctx, cfg);

  // No composite score set => defaults to 0.0
  // effective_threshold = 0.4 - 0.05 = 0.35
  // 0.0 > 0.35 is false => should reject
  ctx.SetThresholdTDynamic (0.4);
  ctx.SetThresholdHysteresis (0.05);

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetWriteDecision () == false);
}

TEST_CASE ("ComputeWriteGate accepts with zero threshold and hysteresis",
           "[operations][write_gate]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;


  OperationContext ctx (s, pctx, cfg);

  // Set composite_score = 0.01, T_dynamic = 0.0, hysteresis = 0.0
  // effective_threshold = 0.0 - 0.0 = 0.0
  // 0.01 > 0.0 => should accept
  ctx.SetCompositeScore (0.01);
  ctx.SetThresholdTDynamic (0.0);
  ctx.SetThresholdHysteresis (0.0);

  ComputeWriteGate op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetWriteDecision () == true);
}
