#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/drift_accumulation.hpp>
#include <cortext/operations/streaming_pacing.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <Eigen/Dense>

using namespace cortext;
using cortext::operations::UpdateDriftAccumulation;
using cortext::operations::CheckStreamingPacing;

namespace
{

Signal
MakeSignal (Eigen::VectorXf emb)
{
  Signal s;
  s.embedding = std::move (emb);
  s.timestamp = 1;
  s.source_id = "test";
  return s;
}

} // namespace

TEST_CASE ("UpdateDriftAccumulation initializes on first signal",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;


  Eigen::VectorXf emb = Eigen::VectorXf::Ones (4);
  Signal s = MakeSignal (emb);
  OperationContext ctx (s, pctx, cfg);

  REQUIRE_FALSE (pctx.last_pacing_check_embedding.has_value ());

  UpdateDriftAccumulation op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.last_pacing_check_embedding.has_value ());
  REQUIRE (pctx.drift_accum == 0.0);
  REQUIRE (ctx.GetDriftAccumSnapshot () == 0.0);
}

TEST_CASE ("UpdateDriftAccumulation accumulates drift",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;


  // First signal: initialize
  Eigen::VectorXf emb1 = Eigen::VectorXf::Zero (4);
  Signal s1 = MakeSignal (emb1);
  OperationContext ctx1 (s1, pctx, cfg);

  UpdateDriftAccumulation op;
  op.Execute (ctx1, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.drift_accum == 0.0);

  // Second signal: distance = ||[1,0,0,0] - [0,0,0,0]|| = 1.0
  Eigen::VectorXf emb2 = Eigen::VectorXf::Zero (4);
  emb2[0] = 1.0f;
  Signal s2 = MakeSignal (emb2);
  OperationContext ctx2 (s2, pctx, cfg);

  op.Execute (ctx2, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.drift_accum == Catch::Approx (1.0));
  REQUIRE (ctx2.GetDriftAccumSnapshot () == Catch::Approx (1.0));
}

TEST_CASE ("UpdateDriftAccumulation handles empty embedding",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;


  // Empty embedding should be skipped
  Eigen::VectorXf emb;
  Signal s = MakeSignal (emb);
  OperationContext ctx (s, pctx, cfg);

  UpdateDriftAccumulation op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE_FALSE (pctx.last_pacing_check_embedding.has_value ());
  REQUIRE (pctx.drift_accum == 0.0);
}

TEST_CASE ("UpdateDriftAccumulation handles dimension mismatch",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;


  // First signal: 4D
  Eigen::VectorXf emb1 = Eigen::VectorXf::Ones (4);
  Signal s1 = MakeSignal (emb1);
  OperationContext ctx1 (s1, pctx, cfg);

  UpdateDriftAccumulation op;
  op.Execute (ctx1, cortext::testing::GetNullTransaction ());

  pctx.drift_accum = 1.0; // Simulate accumulated drift

  // Second signal: 8D (dimension mismatch)
  Eigen::VectorXf emb2 = Eigen::VectorXf::Ones (8);
  Signal s2 = MakeSignal (emb2);
  OperationContext ctx2 (s2, pctx, cfg);

  op.Execute (ctx2, cortext::testing::GetNullTransaction ());

  // Should reset tracking
  REQUIRE (pctx.drift_accum == 0.0);
  REQUIRE (pctx.last_pacing_check_embedding->size () == 8);
}

TEST_CASE ("CheckStreamingPacing gates retrieval below threshold",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.5;  // threshold = 0.3
  cfg.focus = 0.5;        // max_wait = 1.25


  pctx.drift_accum = 0.1;  // Below threshold 0.3

  Signal s = MakeSignal (Eigen::VectorXf::Ones (4));
  OperationContext ctx (s, pctx, cfg);

  CheckStreamingPacing op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetShouldCheckRetrieval () == false);
  // Drift accum should NOT be reset
  REQUIRE (pctx.drift_accum == Catch::Approx (0.1));
}

TEST_CASE ("CheckStreamingPacing triggers retrieval above threshold",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.5;  // threshold = 0.3
  cfg.focus = 0.5;        // max_wait = 1.25


  pctx.drift_accum = 0.5;  // Above threshold 0.3
  pctx.last_pacing_check_embedding = Eigen::VectorXf::Zero (4);

  Eigen::VectorXf emb = Eigen::VectorXf::Ones (4);
  Signal s = MakeSignal (emb);
  OperationContext ctx (s, pctx, cfg);

  CheckStreamingPacing op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetShouldCheckRetrieval () == true);
  // Should reset after trigger
  REQUIRE (pctx.drift_accum == 0.0);
  // Should update last_pacing_check_embedding
  REQUIRE (pctx.last_pacing_check_embedding.has_value ());
}

TEST_CASE ("CheckStreamingPacing force check on max_wait_drift",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.0;  // threshold = 0.5
  cfg.focus = 1.0;        // max_wait = 0.5


  pctx.drift_accum = 0.6;  // Above max_wait 0.5 but below threshold 0.5
  pctx.last_pacing_check_embedding = Eigen::VectorXf::Zero (4);

  Signal s = MakeSignal (Eigen::VectorXf::Ones (4));
  OperationContext ctx (s, pctx, cfg);

  CheckStreamingPacing op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Should trigger because drift_accum (0.6) > max_wait (0.5)
  REQUIRE (ctx.GetShouldCheckRetrieval () == true);
  REQUIRE (pctx.drift_accum == 0.0);  // Reset after trigger
}

TEST_CASE ("CheckStreamingPacing default is true for backward compatibility",
           "[operations][streaming_pacing]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;


  Signal s = MakeSignal (Eigen::VectorXf::Ones (4));
  OperationContext ctx (s, pctx, cfg);

  // Before any operation runs, default should be true
  REQUIRE (ctx.GetShouldCheckRetrieval () == true);
}
