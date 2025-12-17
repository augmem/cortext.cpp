// tests/operations_mood.test.cpp
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/constants.hpp>
#include <cortext/operations/sensitivity.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include "test_helpers.hpp"
#include <array>
#include <cmath>

using namespace cortext;
using cortext::operations::UpdateMood;

namespace
{

static Signal
MakeSignal ()
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (4);
  s.timestamp = 1;
  s.source_id = "test";
  return s;
}

// Helper to set up emotion probabilities for testing
class SetupEmotionProbabilities : public IOperation
{
public:
  explicit SetupEmotionProbabilities (std::array<double, 6> probs)
      : probs_ (probs)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetEmotionProbabilities (probs_);
  }

private:
  std::array<double, 6> probs_;
};

} // namespace

TEST_CASE ("UpdateMood integrates emotion into mood vector",
           "[operations][mood]")
{
  Signal s = MakeSignal ();
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  OperationContext ctx (s, pctx, cfg);

  // Set emotion probabilities (softmax-like, sums to ~1)
  std::array<double, 6> e_t = { 0.1, 0.1, 0.5, 0.1, 0.1, 0.1 }; // joy-dominant
  ctx.SetEmotionProbabilities (e_t);

  // Initial mood should be zero
  for (size_t i = 0; i < 6; ++i)
    {
      REQUIRE (pctx.mood_vector[i] == 0.0);
    }

  UpdateMood op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // After first update: M = λ × 0 + α × e_t = α × e_t
  const double alpha_mood = core::AlphaMood (cfg.sensitivity);
  for (size_t i = 0; i < 6; ++i)
    {
      REQUIRE (pctx.mood_vector[i] == Catch::Approx (alpha_mood * e_t[i]));
    }

  // Delta threshold should be set
  REQUIRE (ctx.GetDeltaThresholdMood ().has_value ());
  REQUIRE (*ctx.GetDeltaThresholdMood () <= 0.0); // always non-positive
}

TEST_CASE ("UpdateMood decay dynamics with λ_mood", "[operations][mood]")
{
  Signal s = MakeSignal ();
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  OperationContext ctx (s, pctx, cfg);

  const double alpha = core::AlphaMood (cfg.sensitivity);
  const double lambda = core::LambdaMood (cfg.stability);

  // Set initial mood state
  pctx.mood_vector = { 0.5, 0.0, 0.3, 0.0, 0.0, 0.0 };
  std::array<double, 6> initial_mood = pctx.mood_vector;

  // Zero emotion input to test pure decay
  std::array<double, 6> e_t = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  ctx.SetEmotionProbabilities (e_t);

  UpdateMood op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // After update: M = λ × M_{t-1} + α × 0 = λ × M_{t-1}
  for (size_t i = 0; i < 6; ++i)
    {
      REQUIRE (pctx.mood_vector[i]
               == Catch::Approx (lambda * initial_mood[i]));
    }
}

TEST_CASE ("UpdateMood clamps per-dimension to [-1, 1]",
           "[operations][mood]")
{
  Signal s = MakeSignal ();
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.sensitivity = 1.0; // high reactivity: α = 0.20
  cfg.stability = 1.0;   // high persistence: λ = 0.999
  OperationContext ctx (s, pctx, cfg);

  // Start with mood near upper bound
  pctx.mood_vector = { 0.95, 0.0, 0.95, 0.0, 0.0, 0.0 };

  // Strong emotion input
  std::array<double, 6> e_t = { 1.0, 0.0, 1.0, 0.0, 0.0, 0.0 };
  ctx.SetEmotionProbabilities (e_t);

  UpdateMood op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // After update, values should be clamped to [-1, 1]
  for (size_t i = 0; i < 6; ++i)
    {
      REQUIRE (pctx.mood_vector[i] >= -1.0);
      REQUIRE (pctx.mood_vector[i] <= 1.0);
    }
}

TEST_CASE ("UpdateMood ΔT_mood calculation", "[operations][mood]")
{
  Signal s = MakeSignal ();
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.8;
  cfg.stability = 0.5;
  OperationContext ctx (s, pctx, cfg);

  // Set a known mood state
  pctx.mood_vector = { 0.3, 0.0, 0.4, 0.0, 0.0, 0.0 };

  // Zero emotion to not change mood magnitude
  std::array<double, 6> e_t = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  ctx.SetEmotionProbabilities (e_t);

  UpdateMood op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // ΔT_mood = -κ_mood × clamp(||M_t|| / √6, 0, 1)
  // Note: mood decays slightly, so recalculate magnitude
  const double lambda = core::LambdaMood (cfg.stability);
  const double decayed_mag
      = std::sqrt ((0.3 * lambda) * (0.3 * lambda)
                   + (0.4 * lambda) * (0.4 * lambda));
  // Normalize by √6 per paper Section 4.2.4
  const double m_norm = core::Clamp (decayed_mag / std::sqrt (6.0), 0.0, 1.0);
  const double kappa_mood = operations::constants::kGainMedium * cfg.sensitivity;
  const double expected_delta = -kappa_mood * m_norm;

  REQUIRE (ctx.GetDeltaThresholdMood ().has_value ());
  REQUIRE (*ctx.GetDeltaThresholdMood () == Catch::Approx (expected_delta));
}

TEST_CASE ("UpdateMood edge cases S=0 and S=1", "[operations][mood]")
{
  // S = 0: minimal reactivity (α = 0.01)
  SECTION ("S=0 minimal reactivity")
  {
    Signal s = MakeSignal ();
    ProcessorContext pctx;
    SignalProcessor::Config cfg;
    cfg.sensitivity = 0.0;
    cfg.stability = 0.5;
  
    OperationContext ctx (s, pctx, cfg);

    std::array<double, 6> e_t = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    ctx.SetEmotionProbabilities (e_t);

    UpdateMood op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    // α_mood(0) = 0.01
    REQUIRE (core::AlphaMood (0.0) == Catch::Approx (0.01));
    REQUIRE (pctx.mood_vector[0] == Catch::Approx (0.01));

    // ΔT_mood should be 0 since κ_mood = κ_base × 0 = 0
    REQUIRE (ctx.GetDeltaThresholdMood ().has_value ());
    REQUIRE (*ctx.GetDeltaThresholdMood () == Catch::Approx (0.0));
  }

  // S = 1: maximal reactivity (α = 0.20)
  SECTION ("S=1 maximal reactivity")
  {
    Signal s = MakeSignal ();
    ProcessorContext pctx;
    SignalProcessor::Config cfg;
    cfg.sensitivity = 1.0;
    cfg.stability = 0.5;
  
    OperationContext ctx (s, pctx, cfg);

    std::array<double, 6> e_t = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    ctx.SetEmotionProbabilities (e_t);

    UpdateMood op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    // α_mood(1) = 0.20
    REQUIRE (core::AlphaMood (1.0) == Catch::Approx (0.20));
    REQUIRE (pctx.mood_vector[0] == Catch::Approx (0.20));
  }
}

TEST_CASE ("UpdateMood edge cases T=0 and T=1", "[operations][mood]")
{
  // T = 0: fast decay (λ = 0.90)
  SECTION ("T=0 fast decay")
  {
    Signal s = MakeSignal ();
    ProcessorContext pctx;
    SignalProcessor::Config cfg;
    cfg.sensitivity = 0.5;
    cfg.stability = 0.0;
  
    OperationContext ctx (s, pctx, cfg);

    pctx.mood_vector = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    std::array<double, 6> e_t = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    ctx.SetEmotionProbabilities (e_t);

    UpdateMood op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    // λ_mood(0) = 0.90
    REQUIRE (core::LambdaMood (0.0) == Catch::Approx (0.90));
    REQUIRE (pctx.mood_vector[0] == Catch::Approx (0.90));
  }

  // T = 1: slow decay (λ = 0.999)
  SECTION ("T=1 slow decay")
  {
    Signal s = MakeSignal ();
    ProcessorContext pctx;
    SignalProcessor::Config cfg;
    cfg.sensitivity = 0.5;
    cfg.stability = 1.0;
  
    OperationContext ctx (s, pctx, cfg);

    pctx.mood_vector = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    std::array<double, 6> e_t = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    ctx.SetEmotionProbabilities (e_t);

    UpdateMood op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    // λ_mood(1) = 0.999
    REQUIRE (core::LambdaMood (1.0) == Catch::Approx (0.999));
    REQUIRE (pctx.mood_vector[0] == Catch::Approx (0.999));
  }
}

TEST_CASE ("UpdateMood accumulation over multiple signals",
           "[operations][mood]")
{
  Signal s = MakeSignal ();
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  OperationContext ctx (s, pctx, cfg);

  const double alpha = core::AlphaMood (cfg.sensitivity);
  const double lambda = core::LambdaMood (cfg.stability);

  // Consistent joy input over multiple steps
  std::array<double, 6> e_t = { 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 }; // pure joy
  ctx.SetEmotionProbabilities (e_t);

  UpdateMood op;

  // Run multiple iterations
  for (int i = 0; i < 10; ++i)
    {
      op.Execute (ctx, cortext::testing::GetNullTransaction ());
    }

  // Joy dimension should accumulate (converge toward α/(1-λ) if unclamped)
  REQUIRE (pctx.mood_vector[2] > 0.5); // accumulated joy

  // Other dimensions should remain near zero
  REQUIRE (pctx.mood_vector[0] == Catch::Approx (0.0).margin (0.01));
  REQUIRE (pctx.mood_vector[1] == Catch::Approx (0.0).margin (0.01));
}

TEST_CASE ("UpdateMood with mixed emotions", "[operations][mood]")
{
  Signal s = MakeSignal ();
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  OperationContext ctx (s, pctx, cfg);

  const double alpha = core::AlphaMood (cfg.sensitivity);

  // Mixed emotion input
  std::array<double, 6> e_t
      = { 0.3, 0.1, 0.2, 0.2, 0.1, 0.1 }; // anger + joy + love dominant
  ctx.SetEmotionProbabilities (e_t);

  UpdateMood op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Verify all dimensions updated proportionally
  for (size_t i = 0; i < 6; ++i)
    {
      REQUIRE (pctx.mood_vector[i] == Catch::Approx (alpha * e_t[i]));
    }

  // Compute expected magnitude
  double mag_sq = 0.0;
  for (size_t i = 0; i < 6; ++i)
    {
      mag_sq += pctx.mood_vector[i] * pctx.mood_vector[i];
    }
  const double mag = std::sqrt (mag_sq);
  // Normalize by √6 per paper Section 4.2.4
  const double m_norm = core::Clamp (mag / std::sqrt (6.0), 0.0, 1.0);

  // Verify ΔT_mood
  const double kappa_mood
      = operations::constants::kGainMedium * cfg.sensitivity;
  REQUIRE (ctx.GetDeltaThresholdMood ().has_value ());
  REQUIRE (*ctx.GetDeltaThresholdMood ()
           == Catch::Approx (-kappa_mood * m_norm));
}

TEST_CASE ("UpdateMood zero mood yields zero ΔT_mood", "[operations][mood]")
{
  Signal s = MakeSignal ();
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  OperationContext ctx (s, pctx, cfg);

  // Mood is already zero (default), zero emotion input
  std::array<double, 6> e_t = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  ctx.SetEmotionProbabilities (e_t);

  UpdateMood op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // ||M|| = 0, so ΔT_mood = 0
  REQUIRE (ctx.GetDeltaThresholdMood ().has_value ());
  REQUIRE (*ctx.GetDeltaThresholdMood () == Catch::Approx (0.0));
}

TEST_CASE ("UpdateMood max mood state normalization", "[operations][mood]")
{
  Signal s = MakeSignal ();
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.sensitivity = 1.0; // max sensitivity
  cfg.stability = 1.0;   // max stability (slow decay)
  OperationContext ctx (s, pctx, cfg);

  // Set all dimensions to max value 1.0
  // ||M|| = √6 when all dimensions are 1.0
  pctx.mood_vector = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

  // Zero emotion to minimize mood change
  std::array<double, 6> e_t = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  ctx.SetEmotionProbabilities (e_t);

  UpdateMood op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // After decay: λ = 0.999, each dimension ~ 0.999
  const double lambda = core::LambdaMood (cfg.stability);
  const double decayed_dim = 1.0 * lambda; // ~0.999
  const double decayed_mag = std::sqrt (6.0 * decayed_dim * decayed_dim);
  // m_norm = ||M|| / √6 should be close to 1.0 (clamped)
  const double m_norm = core::Clamp (decayed_mag / std::sqrt (6.0), 0.0, 1.0);

  // m_norm should be approximately equal to decayed_dim (~0.999)
  REQUIRE (m_norm == Catch::Approx (decayed_dim).epsilon (0.001));

  // Verify ΔT_mood uses normalized value
  const double kappa_mood
      = operations::constants::kGainMedium * cfg.sensitivity;
  const double expected_delta = -kappa_mood * m_norm;

  REQUIRE (ctx.GetDeltaThresholdMood ().has_value ());
  REQUIRE (*ctx.GetDeltaThresholdMood () == Catch::Approx (expected_delta));
  // At max S=1.0 and near-max mood, ΔT_mood should be bounded
  REQUIRE (*ctx.GetDeltaThresholdMood () >= -kappa_mood);
}
