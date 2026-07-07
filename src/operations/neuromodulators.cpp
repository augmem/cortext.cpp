#include "cortext/operations/neuromodulators.hpp"

#include "neuromodulator_internal.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "../experimental_env.hpp"
#include <cmath>

namespace cortext::operations
{

namespace
{
constexpr double kTwoPi = 6.283185307179586;
}

void
UpdateNeuromodulators::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();

  const double F = cfg.focus;
  const double S = cfg.sensitivity;
  const double T = cfg.stability;
  const double S_eff = core::SensitivityBias (S);
  const auto policy = core::NeuromodulatorPolicyForKnobs (F, S, T);

  const double novelty
      = context.GetMetric (operations::Metric::rarity).value_or (0.0);
  const double surprisal
      = context.GetMetric (operations::Metric::embedding_surprisal).value_or (0.0);
  const double arousal = context.GetArousal ();
  const double retrieval_pressure
      = core::Clamp (
          static_cast<double> (context.GetRetrievalQueueDepth ())
              / policy.retrieval_pressure_depth,
          0.0, 1.0);

  p_ctx.neuromod_ach
      = core::Clamp (policy.ach_base + policy.ach_novelty_gain * novelty
                         - policy.ach_retrieval_pressure_gain
                               * retrieval_pressure,
                     0.0, 1.0);
  p_ctx.neuromod_ne
      = core::Clamp (policy.ne_base + policy.ne_surprisal_gain * surprisal
                         + policy.ne_arousal_gain * arousal,
                     0.0, 1.0);
  p_ctx.neuromod_da
      = core::Clamp (policy.da_base + std::max (0.0, p_ctx.delta_reward), 0.0,
                     1.0);

  // Oscillatory gating (no discrete modes)
  const uint64_t now_ts = context.GetSignal ().timestamp;
  double delta_t = 0.0;
  if (p_ctx.last_signal_timestamp > 0 && now_ts > p_ctx.last_signal_timestamp)
    {
      delta_t = static_cast<double> (now_ts - p_ctx.last_signal_timestamp) / 1000.0;
    }
  const double omega = policy.oscillation_rate;
  p_ctx.osc_phase = std::fmod (p_ctx.osc_phase + omega * delta_t, kTwoPi);
  const double osc_t = 0.5 + 0.5 * std::sin (p_ctx.osc_phase);
  const bool oscillator_disabled = internal::experimental_env::Flag (
      "CORTEXT_DISABLE_ENCODE_RETRIEVE_OSCILLATOR");

  const double encode_bias
      = p_ctx.neuromod_ach
        * (policy.encode_sensitivity_floor
           + policy.encode_sensitivity_gain * S_eff);
  const double oscillation_multiplier
      = oscillator_disabled
            ? 1.0
            : (policy.encode_oscillation_floor
               + policy.encode_oscillation_gain * osc_t);
  p_ctx.encode_bias = encode_bias * oscillation_multiplier;
  p_ctx.encode_bias = core::Clamp (p_ctx.encode_bias, 0.0, 1.0);
  p_ctx.retrieval_bias = core::Clamp (1.0 - p_ctx.encode_bias, 0.0, 1.0);
  const double write_threshold_scale
      = neuromodulation::WriteThresholdScale (p_ctx.neuromod_ne);
  const double reconsolidation_scale
      = neuromodulation::ReconsolidationScale (p_ctx.neuromod_ach);
  const double retrieval_competition_scale
      = neuromodulation::RetrievalCompetitionScale (p_ctx.neuromod_ne);
  const double value_update_gain
      = neuromodulation::ValueUpdateGain (p_ctx.neuromod_da);

  telemetry::LogDebug ("cortext.neuromodulators", {
    telemetry::Attribute::Double ("ACh", p_ctx.neuromod_ach),
    telemetry::Attribute::Double ("NE", p_ctx.neuromod_ne),
    telemetry::Attribute::Double ("DA", p_ctx.neuromod_da),
    telemetry::Attribute::Double ("encode_bias", p_ctx.encode_bias),
    telemetry::Attribute::Double ("retrieval_bias", p_ctx.retrieval_bias),
    telemetry::Attribute::Double ("write_threshold_scale",
                                  write_threshold_scale),
    telemetry::Attribute::Double ("reconsolidation_scale",
                                  reconsolidation_scale),
    telemetry::Attribute::Double ("retrieval_competition_scale",
                                  retrieval_competition_scale),
    telemetry::Attribute::Double ("value_update_gain", value_update_gain),
    telemetry::Attribute::Double ("osc_phase", p_ctx.osc_phase)
  });
}

} // namespace cortext::operations
