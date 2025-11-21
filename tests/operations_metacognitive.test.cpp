#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/algorithms.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/metacognitive.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using namespace cortext;
using cortext::operations::MetacognitiveMonitoring;

TEST_CASE ("Alg25 detects TOT when FOK high and retrieval low",
           "[operations][metacognitive][tot]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.timestamp = 1;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.8; // higher focus
  cfg.sensitivity = 0.2;
  cfg.stability = 0.7;
  std::vector<BufferedWriteInstruction> buf;

  MetacognitiveMonitoring op;
  OperationContext ctx (s, pctx, cfg, buf);
  ctx.SetFeelingOfKnowing (0.90); // high FOK
  ctx.SetCompositeScore (0.20);   // low retrieval
  op.Execute (ctx);

  const double tot_fok_cut = cortext::core::TOTFokCutoff (cfg.focus);
  const double tot_ret_cut = cortext::core::TOTRetrievalCutoff (cfg.focus);
  REQUIRE (0.90 > tot_fok_cut);
  REQUIRE (0.20 < tot_ret_cut);
  REQUIRE (ctx.GetMetacogTOTDetected () == true);
}

TEST_CASE ("Alg25 detects unknown when retrieval below threshold",
           "[operations][metacognitive][unknown]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.timestamp = 2;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;

  MetacognitiveMonitoring op;
  OperationContext ctx (s, pctx, cfg, buf);
  const double unk = cortext::core::UnknownThreshold (cfg.focus);
  ctx.SetFeelingOfKnowing (0.10);
  ctx.SetCompositeScore (unk - 0.05);
  op.Execute (ctx);
  REQUIRE (ctx.GetMetacogUnknownDetected () == true);
}

TEST_CASE ("Alg25 exposes parameter derivations per algorithms.md",
           "[operations][metacognitive][params]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (3);
  s.timestamp = 3;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.6;
  cfg.sensitivity = 0.3;
  cfg.stability = 0.8;
  std::vector<BufferedWriteInstruction> buf;

  MetacognitiveMonitoring op;
  OperationContext ctx (s, pctx, cfg, buf);
  ctx.SetFeelingOfKnowing (0.4);
  ctx.SetCompositeScore (0.7);
  op.Execute (ctx);

  const double expected_fok_th = cortext::core::FOKThreshold (cfg.focus);
  REQUIRE (ctx.GetMetacogFOKThreshold ()
           == Catch::Approx (expected_fok_th).epsilon (1e-9));

  const double expected_decay
      = cortext::core::ConfidenceDecayRate (cfg.stability);
  REQUIRE (ctx.GetMetacogConfidenceDecayRate ()
           == Catch::Approx (expected_decay).epsilon (1e-9));

  const int expected_latency
      = cortext::core::StrategySwitchLatencyMs (cfg.sensitivity);
  REQUIRE (ctx.GetMetacogStrategySwitchLatencyMs () == expected_latency);

  const double expected_cert
      = cortext::core::CertaintyRequirement (cfg.stability);
  REQUIRE (ctx.GetMetacogCertaintyRequirement ()
           == Catch::Approx (expected_cert).epsilon (1e-9));

  const double expected_meta_sens
      = cortext::core::MetacognitiveSensitivity (cfg.focus, cfg.sensitivity);
  REQUIRE (ctx.GetMetacogSensitivity ()
           == Catch::Approx (expected_meta_sens).epsilon (1e-9));
}
