#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>

using namespace cortext;

struct SetCompositeScoreOp : IOperation
{
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetCompositeScore (0.42);
  }
};

struct SetDeltaSensitivityOp : IOperation
{
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetDeltaThresholdSensitivity (0.07);
  }
};

struct RecordOrderOp : IOperation
{
  explicit RecordOrderOp (std::vector<int> *order, int id)
      : order (order), id (id)
  {
  }

  void
  Execute (OperationContext & /*ctx*/, Transaction & /*tx*/) const override
  {
    if (order)
      {
        order->push_back (id);
      }
  }

  std::vector<int> *order;
  int id = 0;
};

TEST_CASE ("OperationSet executes in order", "[operation_set]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);

  OperationContext ctx (s, pctx, cfg);

  OperationSet set (std::make_unique<SetCompositeScoreOp> (),
                    std::make_unique<SetDeltaSensitivityOp> ());

  set.Execute (ctx, cortext::testing::GetNullTransaction ());

  auto a = ctx.GetCompositeScore ();
  auto b = ctx.GetDeltaThresholdSensitivity ();
  REQUIRE (a.has_value ());
  REQUIRE (b.has_value ());
}

TEST_CASE ("OperationSet preserves explicit ordering", "[operation_set][order]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);

  OperationContext ctx (s, pctx, cfg);
  std::vector<int> order;

  OperationSet set (std::make_unique<RecordOrderOp> (&order, 1),
                    std::make_unique<RecordOrderOp> (&order, 2),
                    std::make_unique<RecordOrderOp> (&order, 3));

  set.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (order == std::vector<int> { 1, 2, 3 });
}
