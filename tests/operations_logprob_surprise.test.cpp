// tests/operations_logprob_surprise.test.cpp
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cortext/operations/logprob_surprise.hpp>
#include <cortext/operations/metrics.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/signal.hpp>

using namespace cortext;
using cortext::operations::Metric;
using cortext::operations::UpdateLogprobSurprise;

namespace
{
static Signal
MakeSignalWithNll (std::optional<double> mean_nll)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  s.mean_token_nll = mean_nll;
  s.timestamp = 0;
  s.source_id = "test";
  return s;
}
} // namespace

TEST_CASE ("Alg13 is a no-op when mean_token_nll is absent",
           "[operations][alg13][logprob]")
{
  ProcessorContext pc;
  SignalProcessor::Config cfg;
  std::vector<BufferedWriteInstruction> wb;

  auto sig = MakeSignalWithNll (std::nullopt);
  OperationContext ctx (sig, pc, cfg, wb);
  ctx.SetMetric (Metric::drift_mag, 0.5);
  ctx.SetMetric (Metric::surprise, 0.2);

  UpdateLogprobSurprise op;
  op.Execute (ctx);

  REQUIRE (ctx.GetMetric (Metric::surprise).has_value ());
  REQUIRE (ctx.GetMetric (Metric::surprise).value () == Catch::Approx (0.2));
}

TEST_CASE ("Alg13 fuses drift_mag with normalized mean_token_nll",
           "[operations][alg13][logprob]")
{
  ProcessorContext pc;
  SignalProcessor::Config cfg;
  std::vector<BufferedWriteInstruction> wb;

  // mean_token_nll=2.5 => logprob_surprisal=0.5
  auto sig = MakeSignalWithNll (2.5);
  OperationContext ctx (sig, pc, cfg, wb);
  ctx.SetMetric (Metric::drift_mag, 0.25);

  UpdateLogprobSurprise op;
  op.Execute (ctx);

  // surprise = 1 - (1 - 0.25)*(1 - 0.5) = 1 - 0.75*0.5 = 0.625
  REQUIRE (ctx.GetMetric (Metric::surprise).has_value ());
  REQUIRE (ctx.GetMetric (Metric::surprise).value ()
           == Catch::Approx (0.625).margin (1e-6));
}

