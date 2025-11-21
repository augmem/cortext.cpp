#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/buffered_write_instruction.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>

using cortext::BufferedWriteInstruction;
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
  std::vector<BufferedWriteInstruction> buf;

  OperationContext ctx (s, pctx, cfg, buf);

  // Accessors return references
  REQUIRE (&ctx.GetSignal () == &s);
  REQUIRE (&ctx.GetProcessorContext () == &pctx);
  REQUIRE (&ctx.GetConfig () == &cfg);

  // Typed per-signal variables
  ctx.SetCompositeScore (0.75);
  auto comp = ctx.GetCompositeScore ();
  REQUIRE (comp.has_value ());
  REQUIRE (*comp == Catch::Approx (0.75));

  // Write instruction buffer
  BufferedWriteInstruction ins{ "INSERT INTO t(v) VALUES(?)",
                                { std::string{ "x" } } };
  ctx.AddWriteInstruction (ins);
  REQUIRE (buf.size () == 1);
  REQUIRE (buf[0].query == "INSERT INTO t(v) VALUES(?)");
}
