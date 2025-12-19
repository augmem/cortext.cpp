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
  const auto &signal = context.GetSignal ();
  auto &p_ctx = context.GetProcessorContext ();
  const auto &config = context.GetConfig ();
  const std::string &source_id = signal.source_id;

  // Get composite score
  const double score = context.GetCompositeScore ().value_or (0.0);

  // Get dynamic threshold
  const double T_dynamic = context.GetThresholdTDynamic ();

  // Compute spike margin (Section 4.4.4)
  const double spike_margin = core::SpikeMargin (config.sensitivity);
  const double spike_threshold = T_dynamic + spike_margin;

  // Check for spike bypass
  const bool spike = score > spike_threshold;

  // If spike and we have accumulated signals, force flush
  bool spike_bypass = false;
  if (spike)
    {
      auto it = p_ctx.accumulator_states.find (source_id);
      if (it != p_ctx.accumulator_states.end () && it->second.n_signals > 1)
        {
          // Force flush - spike captures surrounding context
          spike_bypass = true;
          context.SetFlushRequired (true);
          telemetry::AddCounter ("cortext.accumulator.spike_bypass_total", 1);
        }
    }

  context.SetSpikeBypass (spike_bypass);

  telemetry::RecordHistogram ("cortext.accumulator.spike_margin", spike_margin);
  if (spike)
    {
      telemetry::AddCounter ("cortext.accumulator.spike_detected_total", 1);
    }

  telemetry::LogDebug ("cortext.spike_bypass", {
    telemetry::Attribute::Double ("score_t", score),
    telemetry::Attribute::Double ("spike_threshold", spike_threshold),
    telemetry::Attribute::Bool ("spike_bypass", spike_bypass)
  });
}

} // namespace cortext::operations
