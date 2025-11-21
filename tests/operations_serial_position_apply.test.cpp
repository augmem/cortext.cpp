#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/memory_strength.hpp>
#include <cortext/operations/serial_position.hpp>
#include <cortext/operations/serial_position_apply.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#
using namespace cortext;
using cortext::operations::ApplySerialPositionEffects;
using cortext::operations::ApplySerialPositionMultiplier;
using cortext::operations::UpdateMemoryStrength;
#
TEST_CASE ("Serial position multiplier reflects primacy/recency zones",
           "[operations][serial_position_apply]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  s.timestamp = 100;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;
  OperationContext ctx (s, pctx, cfg, buf);
#
  // Derive serial windows from Alg 26 op
  ApplySerialPositionEffects derive;
  derive.Execute (ctx);
#
  // Build memory usage events in order, marking all as used
  std::vector<OperationContext::MemoryUsageEvent> events
      = { { 1LL, true, std::nullopt },
          { 2LL, true, std::nullopt },
          { 3LL, true, std::nullopt } };
  ctx.SetMemoryUsageEvents (events);
#
  ApplySerialPositionMultiplier apply_mult;
  apply_mult.Execute (ctx);
#
  auto m = ctx.GetSerialPositionMultiplier ();
  REQUIRE (m.has_value ());
  REQUIRE (*m >= 1.0);
}
#
TEST_CASE ("Serial position multiplier is applied to reinforcement only",
           "[operations][serial_position_apply][memory_strength]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  s.timestamp = 200;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;
  OperationContext ctx (s, pctx, cfg, buf);
#
  // Explicitly set multiplier
  ctx.SetSerialPositionMultiplier (1.5);
#
  // One used event to force an UPDATE statement generation
  std::vector<OperationContext::MemoryUsageEvent> events
      = { { 42LL, true, 0.0 } };
  ctx.SetMemoryUsageEvents (events);
#
  UpdateMemoryStrength upd;
  upd.Execute (ctx);
#
  REQUIRE (!buf.empty ());
  const auto &op = buf.back ();
  // Ensure the SQL has the serial position factor "* ? *" in reinforcement
  REQUIRE (op.query.find ("+ (? * ? * ((1.0 - ?) * use_frequency + ? * ?))")
           != std::string::npos);
}
