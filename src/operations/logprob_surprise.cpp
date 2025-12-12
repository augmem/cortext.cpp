#include "cortext/operations/logprob_surprise.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/processor/operation_context.hpp"
#include <cmath>

namespace cortext::operations
{

void
UpdateLogprobSurprise::Execute (OperationContext &context) const
{
  const auto &sig = context.GetSignal ();
  if (!sig.mean_token_nll.has_value ())
    {
      return;
    }

  // Algorithm 13:
  // mean_logp = mean(-log(token_logprobs)) ≈ mean_token_nll
  // logprob_surprisal = clamp(mean_logp / 5.0, 0, 1)
  // surprise = 1 - (1 - drift_mag) * (1 - logprob_surprisal)

  const double mean_nll = *sig.mean_token_nll;
  if (!std::isfinite (mean_nll))
    {
      return;
    }

  const double logprob_surprisal
      = core::Clamp (mean_nll / 5.0, constants::kNormalizedMin,
                     constants::kNormalizedMax);

  const double drift_mag
      = context.GetMetric (operations::Metric::drift_mag).value_or (0.0);
  const double drift01
      = core::Clamp (drift_mag, constants::kNormalizedMin,
                     constants::kNormalizedMax);

  const double surprise
      = constants::kNormalizedMax
        - (constants::kNormalizedMax - drift01)
              * (constants::kNormalizedMax - logprob_surprisal);

  context.SetMetric (operations::Metric::surprise,
                     core::Clamp (surprise, constants::kNormalizedMin,
                                  constants::kNormalizedMax));
}

} // namespace cortext::operations

