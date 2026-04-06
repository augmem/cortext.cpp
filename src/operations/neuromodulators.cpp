#include "cortext/operations/neuromodulators.hpp"

#include "neuromodulator_internal.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
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
  const double F_eff = core::FocusBias (F);
  const double S_eff = core::SensitivityBias (S);

  const double novelty
      = context.GetMetric (operations::Metric::rarity).value_or (0.0);
  const double surprisal
      = context.GetMetric (operations::Metric::embedding_surprisal).value_or (0.0);
  const double arousal = context.GetArousal ();
  const double retrieval_pressure
      = core::Clamp (
          static_cast<double> (context.GetRetrievalQueueDepth ()) / 10.0, 0.0,
          1.0);

  const double ACh_base = core::Clamp (
      0.15 + 0.55 * S_eff + 0.25 * (1.0 - T) - 0.15 * F_eff, 0.0, 1.0);
  const double NE_base
      = core::Clamp (0.10 + 0.60 * S_eff + 0.20 * (1.0 - T), 0.0, 1.0);
  const double DA_base
      = core::Clamp (0.10 + 0.40 * F_eff + 0.30 * T, 0.0, 1.0);

  p_ctx.neuromod_ach
      = core::Clamp (ACh_base + 0.35 * novelty - 0.20 * retrieval_pressure,
                     0.0, 1.0);
  p_ctx.neuromod_ne
      = core::Clamp (NE_base + 0.50 * surprisal + 0.30 * arousal, 0.0, 1.0);
  p_ctx.neuromod_da
      = core::Clamp (DA_base + std::max (0.0, p_ctx.delta_reward), 0.0, 1.0);

  // Oscillatory gating (no discrete modes)
  const uint64_t now_ts = context.GetSignal ().timestamp;
  double delta_t = 0.0;
  if (p_ctx.last_signal_timestamp > 0 && now_ts > p_ctx.last_signal_timestamp)
    {
      delta_t = static_cast<double> (now_ts - p_ctx.last_signal_timestamp) / 1000.0;
    }
  const double omega = core::Lerp (0.03, 0.12, S_eff) * core::Lerp (1.2, 0.8, T);
  p_ctx.osc_phase = std::fmod (p_ctx.osc_phase + omega * delta_t, kTwoPi);
  const double osc_t = 0.5 + 0.5 * std::sin (p_ctx.osc_phase);

  const double encode_bias = p_ctx.neuromod_ach * (0.7 + 0.3 * S_eff);
  p_ctx.encode_bias = encode_bias * (0.6 + 0.4 * osc_t);
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
