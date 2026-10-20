#include "cortext/operations/neuromodulators.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

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

  p_ctx.encode_bias = core::Clamp (
      p_ctx.neuromod_ach
          * (policy.encode_sensitivity_floor
             + policy.encode_sensitivity_gain * S_eff),
      0.0, 1.0);
  p_ctx.retrieval_bias = core::Clamp (1.0 - p_ctx.encode_bias, 0.0, 1.0);

  telemetry::LogDebug ("cortext.neuromodulators", {
    telemetry::Attribute::Double ("ACh", p_ctx.neuromod_ach),
    telemetry::Attribute::Double ("encode_bias", p_ctx.encode_bias),
    telemetry::Attribute::Double ("retrieval_bias", p_ctx.retrieval_bias)
  });
}

} // namespace cortext::operations
