#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>

#include <vector>

using namespace cortext;

namespace
{

// Tags for contract-aggregation tests.
struct TagA
{
};
struct TagB
{
};
struct TagC
{
};

struct RecordOp : IOperation
{
  RecordOp () = default;
  RecordOp (std::vector<int> *order, int id) : order (order), id (id) {}

  void
  Execute (OperationContext & /*ctx*/, Transaction & /*tx*/) const override
  {
    if (order)
      {
        order->push_back (id);
      }
  }

  std::vector<int> *order = nullptr;
  int id = 0;
};

struct RecordOp2 : RecordOp
{
  using RecordOp::RecordOp;
};

struct RecordOp3 : RecordOp
{
  using RecordOp::RecordOp;
};

struct ProducesA : Operation<Requires<>, Satisfies<TagA> >
{
  void
  Execute (OperationContext &, Transaction &) const override
  {
  }
};

struct ConsumesAProducesB : Operation<Requires<TagA>, Satisfies<TagB> >
{
  void
  Execute (OperationContext &, Transaction &) const override
  {
  }
};

struct ConsumesC : Operation<Requires<TagC>, Satisfies<> >
{
  void
  Execute (OperationContext &, Transaction &) const override
  {
  }
};

} // namespace

TEST_CASE ("OperationSet executes members in order", "[operation_set]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  OperationContext ctx (s, pctx, cfg);

  std::vector<int> order;
  OperationSet pipeline (RecordOp (&order, 1), RecordOp2 (&order, 2),
                     RecordOp3 (&order, 3));
  pipeline.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (order == std::vector<int>{ 1, 2, 3 });
}

TEST_CASE ("OperationSet records per-operation timings under distinct names",
           "[operation_set]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  OperationContext ctx (s, pctx, cfg);

  OperationSet<RecordOp, RecordOp2> pipeline;
  pipeline.Execute (ctx, cortext::testing::GetNullTransaction ());

  const auto &timings = ctx.GetOperationTimings ();
  REQUIRE (timings.size () == 2);
}

TEST_CASE ("Nested pipelines execute inline without a stage timing entry",
           "[operation_set]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  OperationContext ctx (s, pctx, cfg);

  std::vector<int> order;
  using Stage = OperationSet<RecordOp, RecordOp2>;
  OperationSet<Stage, RecordOp3> root (
      Stage (RecordOp (&order, 1), RecordOp2 (&order, 2)),
      RecordOp3 (&order, 3));
  root.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (order == std::vector<int>{ 1, 2, 3 });
  // Member ops are timed; the nested Stage wrapper itself is not.
  REQUIRE (ctx.GetOperationTimings ().size () == 3);
}

TEST_CASE ("OperationSet aggregates contracts at compile time", "[operation_set]")
{
  // Internally satisfied chain: A produced, then consumed.
  using Chained = OperationSet<ProducesA, ConsumesAProducesB>;
  STATIC_REQUIRE (Chained::kContractsComplete);
  STATIC_REQUIRE (IsSelfContained<Chained>);
  STATIC_REQUIRE (std::is_same_v<Chained::Input, Requires<> >);

  // Out of order: the consumer precedes the producer, so TagA is external.
  using Misordered = OperationSet<ConsumesAProducesB, ProducesA>;
  STATIC_REQUIRE (std::is_same_v<Misordered::Input, Requires<TagA> >);
  STATIC_REQUIRE (!IsSelfContained<Misordered>);

  // Unmet requirement surfaces as external input.
  using NeedsC = OperationSet<ProducesA, ConsumesC>;
  STATIC_REQUIRE (std::is_same_v<NeedsC::Input, Requires<TagC> >);

  // Nested pipelines propagate contracts upward.
  using Nested = OperationSet<OperationSet<ProducesA>, ConsumesAProducesB>;
  STATIC_REQUIRE (IsSelfContained<Nested>);

  // Unported members (no contract) mark aggregation incomplete.
  using Mixed = OperationSet<ProducesA, RecordOp>;
  STATIC_REQUIRE (!Mixed::kContractsComplete);
  STATIC_REQUIRE (!IsSelfContained<Mixed>);
}
