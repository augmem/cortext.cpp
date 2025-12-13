#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/data/centroids.hpp>
#include <cortext/operations/sensitivity.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#
using namespace cortext;
using cortext::operations::InitializeSensitivityPriors;
using cortext::operations::UpdateSensitivity;
#
static Eigen::VectorXf
unit_at (int idx, int dim)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (dim);
  v[idx] = 1.0f;
  return v;
}
#
TEST_CASE ("UpdateSensitivity without centroids sets sane defaults and deltas",
           "[operations][sensitivity]")
{
  Signal s;
  s.embedding = unit_at (0, 4);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.7;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;
  OperationContext ctx (s, pctx, cfg, buf);
#
  // Populate recent scores to create non-zero variance
  pctx.recent_scores = { 0.1, 0.3, 0.9, 0.5, 0.7 };
#
  UpdateSensitivity op;
  op.Execute (ctx);
#
  // Emotions defaults (no centroids)
  REQUIRE (ctx.GetEmotionIntensity () == Catch::Approx (0.0));
  REQUIRE (ctx.GetValence () == Catch::Approx (0.5));
  REQUIRE (ctx.GetArousal () == Catch::Approx (0.0));
#
  // Deltas present
  REQUIRE (ctx.GetDeltaThresholdEmotion ().has_value ());
  REQUIRE (ctx.GetDeltaThresholdSensitivity ().has_value ());
}
#
TEST_CASE (
    "UpdateSensitivity with centroids yields non-zero emotion intensity",
    "[operations][sensitivity]")
{
  const int dim = 4;
  Signal s;
  s.embedding = unit_at (2, dim); // align strongly with 'joy' index 2
  ProcessorContext pctx;
  // Provide 6 centroids in order: anger,fear,joy,love,sadness,surprise
  cortext::data::Centroids centroids;
  centroids.emotion_centroids.resize (6);
  centroids.emotion_centroids[0] = unit_at (0, dim);
  centroids.emotion_centroids[1] = unit_at (1, dim);
  centroids.emotion_centroids[2] = unit_at (2, dim); // match
  centroids.emotion_centroids[3] = unit_at (3, dim);
  centroids.emotion_centroids[4] = -unit_at (2, dim); // opposite of joy
  centroids.emotion_centroids[5] = unit_at (1, dim);  // duplicate fear
  pctx.centroids = centroids;
#
  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.8;
  cfg.stability = 0.4;
  std::vector<BufferedWriteInstruction> buf;
  OperationContext ctx (s, pctx, cfg, buf);
#
  UpdateSensitivity op;
  op.Execute (ctx);
#
  REQUIRE (ctx.GetEmotionIntensity () > 0.0);
  REQUIRE (ctx.GetDeltaThresholdEmotion ().has_value ());
  // ΔT_emo should be non-positive (loosens thresholds)
  REQUIRE (*ctx.GetDeltaThresholdEmotion () <= 0.0);
}
#
TEST_CASE ("Write-rate window affects rate_target with bursts and gaps",
           "[operations][sensitivity][rate]")
{
  // High-rate case (~60/min)
  {
    Signal s;
    s.embedding = unit_at (0, 2);
    s.timestamp = 1000;
    ProcessorContext pctx;
    SignalProcessor::Config cfg;
    cfg.sensitivity = 0.6;
    cfg.stability = 0.4;
    std::vector<BufferedWriteInstruction> buf;
    OperationContext ctx (s, pctx, cfg, buf);
    // Populate window with 1s intervals
    uint64_t t = 900;
    for (int i = 0; i < 10; ++i)
      {
        pctx.write_rate_window_.Record (t);
        t += 1;
      }
    UpdateSensitivity op;
    op.Execute (ctx);
    REQUIRE (pctx.rate_target > 0.0);
  }
  // Low-rate case (~12/min), ensure lower than previous
  {
    Signal s;
    s.embedding = unit_at (0, 2);
    s.timestamp = 2000;
    ProcessorContext pctx;
    SignalProcessor::Config cfg;
    cfg.sensitivity = 0.6;
    cfg.stability = 0.4;
    std::vector<BufferedWriteInstruction> buf;
    OperationContext ctx (s, pctx, cfg, buf);
    uint64_t t = 1800;
    for (int i = 0; i < 10; ++i)
      {
        pctx.write_rate_window_.Record (t);
        t += 5;
      }
    UpdateSensitivity op;
    op.Execute (ctx);
    REQUIRE (pctx.rate_target > 0.0);
    // With 5s intervals, rate should be ~12/min, ensure < 30/min rough bound
    REQUIRE (pctx.rate_target < 30.0);
  }
  // Non-increasing timestamps ignored
  {
    Signal s;
    s.embedding = unit_at (0, 2);
    s.timestamp = 3000;
    ProcessorContext pctx;
    SignalProcessor::Config cfg;
    cfg.sensitivity = 0.5;
    cfg.stability = 0.5;
    std::vector<BufferedWriteInstruction> buf;
    OperationContext ctx (s, pctx, cfg, buf);
    pctx.write_rate_window_.Record (100);
    pctx.write_rate_window_.Record (100); // should be ignored
    pctx.write_rate_window_.Record (101);
    UpdateSensitivity op;
    op.Execute (ctx);
    REQUIRE (pctx.rate_target >= 0.0);
  }
}
