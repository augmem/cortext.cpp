#include "cortext/operations/metacognitive.hpp"

#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include <cmath>

namespace cortext::operations
{

namespace
{
inline double
Clamp01 (double v)
{
  if (v < 0.0)
    return 0.0;
  if (v > 1.0)
    return 1.0;
  return v;
}
} // namespace

void
MetacognitiveMonitoring::Execute (OperationContext &context) const
{
  const auto &cfg = context.GetConfig ();
  const double F = std::max (0.0, std::min (1.0, cfg.focus));
  const double S = std::max (0.0, std::min (1.0, cfg.sensitivity));
  const double T = std::max (0.0, std::min (1.0, cfg.stability));

  // Inputs
  const double retrieval
      = Clamp01 (context.GetCompositeScore ().value_or (0.0));
  const double FOK = Clamp01 (context.GetFeelingOfKnowing ().value_or (0.0));

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
}

} // namespace cortext::operations
