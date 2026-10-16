#include "cortext/operations/stability.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

void
InitializeStabilityPriors::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto &p_ctx = context.GetProcessorContext ();
  const double T = context.GetConfig ().stability;

  if (p_ctx.stability_priors_initialized)
    {
      return;
    }

  p_ctx.hysteresis_band_prior = core::BaseBandPrior (T);
  p_ctx.stability_priors_initialized = true;

  telemetry::LogDebug("cortext.initialize_stability_priors", {
    telemetry::Attribute::Double("T", T),
    telemetry::Attribute::Double("hysteresis_band_prior", p_ctx.hysteresis_band_prior)
  });
}

} // namespace cortext::operations
