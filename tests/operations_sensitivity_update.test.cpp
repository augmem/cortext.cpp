#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
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
TEST_CASE ("UpdateSensitivity without centroids sets sane defaults",
           "[operations][sensitivity]")
{
  Signal s;
  s.embedding = unit_at (0, 4);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.7;
  cfg.stability = 0.5;

  OperationContext ctx (s, pctx, cfg);
#
  UpdateSensitivity op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
#
  // Emotions defaults (no centroids)
  REQUIRE (ctx.GetEmotionIntensity () == Catch::Approx (0.0));
  REQUIRE (ctx.GetValence () == Catch::Approx (0.5));
  REQUIRE (ctx.GetArousal () == Catch::Approx (0.0));
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
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.8;
  cfg.stability = 0.4;

  OperationContext ctx (s, pctx, cfg);
#
  UpdateSensitivity op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());
#
  REQUIRE (ctx.GetEmotionIntensity () > 0.0);
}
#
