#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>


using cortext::OperationContext;
using cortext::ProcessorContext;
using cortext::Signal;
using cortext::SignalProcessor;

TEST_CASE ("OperationContext intermediate results and accessors", "[context]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  s.timestamp = 123;
  s.source_id = "test";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);


  OperationContext ctx (s, pctx, cfg);

  // Accessors return references
  REQUIRE (&ctx.GetSignal () == &s);
  REQUIRE (&ctx.GetProcessorContext () == &pctx);
  REQUIRE (&ctx.GetConfig () == &cfg);

  // Typed per-signal variables
  ctx.SetCompositeScore (0.75);
  auto comp = ctx.GetCompositeScore ();
  REQUIRE (comp.has_value ());
  REQUIRE (*comp == Catch::Approx (0.75));
}
