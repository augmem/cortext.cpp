#include <catch2/catch_test_macros.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>

using namespace cortext;

struct SetCompositeScoreOp : IOperation
{
  void
  Execute (OperationContext &ctx) const override
  {
    ctx.SetCompositeScore (0.42);
  }
};

struct SetDeltaSensitivityOp : IOperation
{
  void
  Execute (OperationContext &ctx) const override
  {
    ctx.SetDeltaThresholdSensitivity (0.07);
  }
};

TEST_CASE ("OperationSet executes in order", "[operation_set]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  std::vector<BufferedWriteInstruction> buf;
  OperationContext ctx (s, pctx, cfg, buf);

  OperationSet set (std::make_unique<SetCompositeScoreOp> (),
                    std::make_unique<SetDeltaSensitivityOp> ());

  set.Execute (ctx);

  auto a = ctx.GetCompositeScore ();
  auto b = ctx.GetDeltaThresholdSensitivity ();
  REQUIRE (a.has_value ());
  REQUIRE (b.has_value ());
}
