#include "cortext/operations/spike_bypass.hpp"

#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

void
CheckSpikeBypass::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  const auto &config = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();
  const auto &signal = context.GetSignal ();

  // Get composite score
  const double score = context.GetCompositeScore ().value_or (0.0);

  // Get dynamic threshold
  const double T_dynamic = context.GetThresholdTDynamic ();

  // Compute spike margin (Section 4.4.4), scaled by memory maturity and coherence
  // to avoid micro-memories when context is still forming.
  const double spike_margin
      = core::SpikeMargin (config.sensitivity, config.stability);

  int n_signals = 0;
  auto it = p_ctx.accumulator_states.find (signal.source_id);
  if (it != p_ctx.accumulator_states.end ())
    {
      n_signals = std::max (0, it->second.n_signals);
    }

  const double mem_tau = std::max (
      1.0, static_cast<double> (core::WinMemCtx (config.stability)));
  const double mem_maturity
      = 1.0 - std::exp (-static_cast<double> (n_signals) / mem_tau);

  const double coherence = context.GetAccumulatorCoherence ();
  const double coherence_scale
      = 1.0 + (1.0 - config.stability) * core::Clamp (coherence, 0.0, 1.0);
  const double spike_margin_eff
      = spike_margin * (1.0 + (1.0 - mem_maturity)) * coherence_scale;
  const double spike_threshold = T_dynamic + spike_margin_eff;

  // Check for spike bypass
  const bool spike = score > spike_threshold;

  // If spike, force flush even for single-signal units.
  bool spike_bypass = false;
  if (spike)
    {
      spike_bypass = true;
      context.SetFlushRequired (true);
      telemetry::AddCounter ("cortext.accumulator.spike_bypass_total", 1);
    }

  context.SetSpikeBypass (spike_bypass);

  telemetry::RecordHistogram ("cortext.accumulator.spike_margin", spike_margin);
  if (spike)
    {
      telemetry::AddCounter ("cortext.accumulator.spike_detected_total", 1);
    }

  telemetry::LogDebug ("cortext.spike_bypass", {
    telemetry::Attribute::Double ("score_t", score),
    telemetry::Attribute::Double ("spike_margin", spike_margin),
    telemetry::Attribute::Double ("spike_margin_eff", spike_margin_eff),
    telemetry::Attribute::Double ("mem_maturity", mem_maturity),
    telemetry::Attribute::Double ("mem_tau", mem_tau),
    telemetry::Attribute::Double ("coherence", coherence),
    telemetry::Attribute::Double ("coherence_scale", coherence_scale),
    telemetry::Attribute::Double ("spike_threshold", spike_threshold),
    telemetry::Attribute::Bool ("spike_bypass", spike_bypass)
  });
}

} // namespace cortext::operations
