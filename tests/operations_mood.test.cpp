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
constexpr double kUniformProb = 1.0 / 6.0;

static Signal
MakeSignal ()
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (4);
  s.timestamp = 1;
  s.source_id = "test";
  return s;
}

static std::array<double, 6>
Centered (const std::array<double, 6> &probs)
{
  std::array<double, 6> centered{};
  for (size_t i = 0; i < 6; ++i)
    {
      centered[i] = probs[i] - kUniformProb;
    }
  return centered;
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
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  OperationContext ctx (s, pctx, cfg);

  // Set emotion probabilities (softmax-like, sums to ~1)
  std::array<double, 6> p_c = { 0.1, 0.1, 0.5, 0.1, 0.1, 0.1 }; // joy-dominant
  ctx.SetEmotionProbabilities (p_c);

  // Initial mood should be zero
  for (size_t i = 0; i < 6; ++i)
    {
      REQUIRE (pctx.mood_vector[i] == 0.0);
    }

  UpdateMood op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // After first update: M = λ × 0 + α × e_t = α × e_t
  const double alpha_mood = core::AlphaMood (cfg.sensitivity);
  const auto centered = Centered (p_c);
  for (size_t i = 0; i < 6; ++i)
    {
      REQUIRE (pctx.mood_vector[i]
               == Catch::Approx (alpha_mood * centered[i]));
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
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  OperationContext ctx (s, pctx, cfg);

  s.timestamp = 31'000;
  pctx.last_mood_ts = 1'000;
  const double delta_s = (s.timestamp - pctx.last_mood_ts) / 1000.0;
  const double lambda = core::LambdaMood (delta_s, cfg.stability);

  // Set initial mood state
  pctx.mood_vector = { 0.5, 0.0, 0.3, 0.0, 0.0, 0.0 };
  std::array<double, 6> initial_mood = pctx.mood_vector;

  // Zero emotion input to test pure decay
  std::array<double, 6> p_c = { kUniformProb, kUniformProb, kUniformProb,
                                kUniformProb, kUniformProb, kUniformProb };
  ctx.SetEmotionProbabilities (p_c);

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
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 1.0; // high reactivity: α = 0.20
  cfg.stability = 1.0;   // high persistence: λ = 0.999
  OperationContext ctx (s, pctx, cfg);

  // Start with mood near upper bound
  pctx.mood_vector = { 0.95, 0.0, 0.95, 0.0, 0.0, 0.0 };

  // Strong emotion input
  std::array<double, 6> p_c = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  ctx.SetEmotionProbabilities (p_c);

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
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.8;
  cfg.stability = 0.5;
  OperationContext ctx (s, pctx, cfg);

  // Set a known mood state
  pctx.mood_vector = { 0.3, 0.0, 0.4, 0.0, 0.0, 0.0 };

  // Zero emotion to not change mood magnitude
  std::array<double, 6> p_c = { kUniformProb, kUniformProb, kUniformProb,
                                kUniformProb, kUniformProb, kUniformProb };
  ctx.SetEmotionProbabilities (p_c);

  UpdateMood op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // ΔT_mood = -κ_mood × clamp(||M_t|| / √6, 0, 1)
  // Note: mood decays slightly, so recalculate magnitude
  s.timestamp = 1'000;
  pctx.last_mood_ts = 0;
  const double lambda = core::LambdaMood (0.0, cfg.stability);
  const double decayed_mag
      = std::sqrt ((0.3 * lambda) * (0.3 * lambda)
                   + (0.4 * lambda) * (0.4 * lambda));
  // Normalize by √6 per paper Section 4.2.4
  const double m_norm = core::Clamp (decayed_mag / std::sqrt (6.0), 0.0, 1.0);
  const double kappa_mood
      = operations::constants::kGainMedium
        * core::SensitivityBias (cfg.sensitivity);
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
    cortext::testing::RequireEncoder (cfg);
    cfg.sensitivity = 0.0;
    cfg.stability = 0.5;
  
    OperationContext ctx (s, pctx, cfg);

    std::array<double, 6> p_c = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    ctx.SetEmotionProbabilities (p_c);

    UpdateMood op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    // α_mood(0) = 0.01
    REQUIRE (core::AlphaMood (0.0) == Catch::Approx (0.01));
    REQUIRE (pctx.mood_vector[0] == Catch::Approx (0.01 * (1.0 - kUniformProb)));

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
    cortext::testing::RequireEncoder (cfg);
    cfg.sensitivity = 1.0;
    cfg.stability = 0.5;
  
    OperationContext ctx (s, pctx, cfg);

    std::array<double, 6> p_c = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    ctx.SetEmotionProbabilities (p_c);

    UpdateMood op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    // α_mood(1) = 0.20
    REQUIRE (core::AlphaMood (1.0) == Catch::Approx (0.20));
    REQUIRE (pctx.mood_vector[0] == Catch::Approx (0.20 * (1.0 - kUniformProb)));
  }
}

TEST_CASE ("UpdateMood edge cases T=0 and T=1", "[operations][mood]")
{
  // T = 0: fast decay (λ = 0.90)
  SECTION ("T=0 fast decay")
  {
    Signal s = MakeSignal ();
    s.timestamp = 31'000;
    ProcessorContext pctx;
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.sensitivity = 0.5;
    cfg.stability = 0.0;
  
    OperationContext ctx (s, pctx, cfg);

    pctx.mood_vector = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    std::array<double, 6> p_c = { kUniformProb, kUniformProb, kUniformProb,
                                  kUniformProb, kUniformProb, kUniformProb };
    ctx.SetEmotionProbabilities (p_c);

    const double delta_s = 30.0;
    pctx.last_mood_ts = 1'000;

    UpdateMood op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    const double lambda = core::LambdaMood (delta_s, cfg.stability);
    REQUIRE (pctx.mood_vector[0] == Catch::Approx (lambda));
  }

  // T = 1: slow decay (λ = 0.999)
  SECTION ("T=1 slow decay")
  {
    Signal s = MakeSignal ();
    s.timestamp = 31'000;
    ProcessorContext pctx;
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.sensitivity = 0.5;
    cfg.stability = 1.0;
  
    OperationContext ctx (s, pctx, cfg);

    pctx.mood_vector = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    std::array<double, 6> p_c = { kUniformProb, kUniformProb, kUniformProb,
                                  kUniformProb, kUniformProb, kUniformProb };
    ctx.SetEmotionProbabilities (p_c);

    const double delta_s = 30.0;
    pctx.last_mood_ts = 1'000;

    UpdateMood op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());

    const double lambda = core::LambdaMood (delta_s, cfg.stability);
    REQUIRE (pctx.mood_vector[0] == Catch::Approx (lambda));
  }
}

TEST_CASE ("UpdateMood accumulation over multiple signals",
           "[operations][mood]")
{
  Signal s = MakeSignal ();
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  OperationContext ctx (s, pctx, cfg);

  // Consistent joy input over multiple steps
  std::array<double, 6> p_c = { 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 }; // pure joy
  ctx.SetEmotionProbabilities (p_c);

  UpdateMood op;

  // Run multiple iterations
  for (int i = 0; i < 10; ++i)
    {
      op.Execute (ctx, cortext::testing::GetNullTransaction ());
    }

  // Joy dimension should accumulate (converge toward α/(1-λ) if unclamped)
  REQUIRE (pctx.mood_vector[2] > 0.5); // accumulated joy

  // Other dimensions should remain near zero
  REQUIRE (pctx.mood_vector[0] < 0.0);
  REQUIRE (pctx.mood_vector[1] < 0.0);
}

TEST_CASE ("UpdateMood with mixed emotions", "[operations][mood]")
{
  Signal s = MakeSignal ();
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  OperationContext ctx (s, pctx, cfg);

  const double alpha = core::AlphaMood (cfg.sensitivity);

  // Mixed emotion input
  std::array<double, 6> p_c
      = { 0.3, 0.1, 0.2, 0.2, 0.1, 0.1 }; // anger + joy + love dominant
  ctx.SetEmotionProbabilities (p_c);

  UpdateMood op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Verify all dimensions updated proportionally
  const auto centered = Centered (p_c);
  for (size_t i = 0; i < 6; ++i)
    {
      REQUIRE (pctx.mood_vector[i]
               == Catch::Approx (alpha * centered[i]));
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
      = operations::constants::kGainMedium
        * core::SensitivityBias (cfg.sensitivity);
  REQUIRE (ctx.GetDeltaThresholdMood ().has_value ());
  REQUIRE (*ctx.GetDeltaThresholdMood ()
           == Catch::Approx (-kappa_mood * m_norm));
}

TEST_CASE ("UpdateMood zero mood yields zero ΔT_mood", "[operations][mood]")
{
  Signal s = MakeSignal ();
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  OperationContext ctx (s, pctx, cfg);

  // Mood is already zero (default), zero emotion input
  std::array<double, 6> p_c = { kUniformProb, kUniformProb, kUniformProb,
                                kUniformProb, kUniformProb, kUniformProb };
  ctx.SetEmotionProbabilities (p_c);

  UpdateMood op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // ||M|| = 0, so ΔT_mood = 0
  REQUIRE (ctx.GetDeltaThresholdMood ().has_value ());
  REQUIRE (*ctx.GetDeltaThresholdMood () == Catch::Approx (0.0));
}

TEST_CASE ("UpdateMood max mood state normalization", "[operations][mood]")
{
  Signal s = MakeSignal ();
  s.timestamp = 31'000;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 1.0; // max sensitivity
  cfg.stability = 1.0;   // max stability (slow decay)
  OperationContext ctx (s, pctx, cfg);

  // Set all dimensions to max value 1.0
  // ||M|| = √6 when all dimensions are 1.0
  pctx.mood_vector = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

  // Zero emotion to minimize mood change
  std::array<double, 6> p_c = { kUniformProb, kUniformProb, kUniformProb,
                                kUniformProb, kUniformProb, kUniformProb };
  ctx.SetEmotionProbabilities (p_c);

  pctx.last_mood_ts = 1'000;
  UpdateMood op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  double mag_sq = 0.0;
  for (double v : pctx.mood_vector)
    {
      mag_sq += v * v;
    }
  const double m_norm
      = core::Clamp (std::sqrt (mag_sq) / std::sqrt (6.0), 0.0, 1.0);

  // Verify ΔT_mood uses normalized value
  const double kappa_mood
      = operations::constants::kGainMedium
        * core::SensitivityBias (cfg.sensitivity);
  const double expected_delta = -kappa_mood * m_norm;

  REQUIRE (ctx.GetDeltaThresholdMood ().has_value ());
  REQUIRE (*ctx.GetDeltaThresholdMood () == Catch::Approx (expected_delta));
  // At max S=1.0 and near-max mood, ΔT_mood should be bounded
  REQUIRE (*ctx.GetDeltaThresholdMood () >= -kappa_mood);
}
