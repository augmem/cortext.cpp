#include "cortext/operations/metacognitive.hpp"

#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <cmath>

namespace cortext::operations
{

void
MetacognitiveMonitoring::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const double F = core::Clamp01 (cfg.focus);
  const double S = core::Clamp01 (cfg.sensitivity);
  const double T = core::Clamp01 (cfg.stability);

  double retrieval_strength = 0.0;
  const auto &usage_events = context.GetMemoryUsageEvents ();
  for (const auto &event : usage_events)
    {
      if (!event.contextual_gain.has_value ())
        {
          continue;
        }
      retrieval_strength = std::max (
          retrieval_strength, core::Map01 (*event.contextual_gain));
    }
  const double retrieval = core::Clamp01 (retrieval_strength);

  const double FOK = core::Clamp01 (1.0 - p_ctx.u_t);
  context.SetFeelingOfKnowing (FOK);
  p_ctx.fok_state = FOK;
  p_ctx.retrieval_strength = retrieval;
  p_ctx.metacognitive_confidence = FOK;

  // Derived thresholds and params
  const double fok_threshold = core::FOKThreshold (F);
  const double tot_fok_cut = core::TOTFokCutoff (F);
  const double tot_ret_cut = core::TOTRetrievalCutoff (F);
  const double unknown_cut = core::UnknownThreshold (F);
  const double conf_decay = core::ConfidenceDecayRate (T);
  const int switch_latency_ms = core::StrategySwitchLatencyMs (S);
  const double certainty_req = core::CertaintyRequirement (T);
  const double meta_sens = core::MetacognitiveSensitivity (F, S);

  // Decisions
  const bool tot_detected = (FOK > tot_fok_cut) && (retrieval < tot_ret_cut);
  const bool unknown_detected = (retrieval < unknown_cut);

  // Expose outputs
  context.SetMetacogFOKThreshold (fok_threshold);
  context.SetMetacogTOTDetected (tot_detected);
  context.SetMetacogUnknownDetected (unknown_detected);
  context.SetMetacogConfidenceDecayRate (conf_decay);
  context.SetMetacogStrategySwitchLatencyMs (switch_latency_ms);
  context.SetMetacogCertaintyRequirement (certainty_req);
  context.SetMetacogSensitivity (meta_sens);

  telemetry::LogDebug ("cortext.metacognitive",
                       { telemetry::Attribute::Double ("fok_score", FOK),
                         telemetry::Attribute::Bool ("tot_state", tot_detected),
                         telemetry::Attribute::Bool ("unknown_state",
                                                     unknown_detected) });
}

} // namespace cortext::operations
