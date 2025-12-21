#include "cortext/operations/focus_spread.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace cortext::operations
{

void
ComputeFocusSpread::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  const auto &cfg = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();
  const auto &x = context.GetSignal ().embedding;
  const double attention_width
      = core::Clamp (p_ctx.attention_width,
                     static_cast<double> (core::kAttentionWidthMin),
                     static_cast<double> (core::kAttentionWidthMax));
  const double width_scale
      = static_cast<double> (core::kAttentionWidthMax) / attention_width;

  const auto &stream = p_ctx.memory_stream;
  const int n = static_cast<int> (stream.size ());
  if (n < 2)
    {
      context.SetMetric (operations::Metric::focus_spread, 0.0);
      return;
    }

  const int k = std::min (core::KNeighbors (cfg.stability), n);
  if (k < 2)
    {
      context.SetMetric (operations::Metric::focus_spread, 0.0);
      return;
    }

  std::vector<double> sims;
  sims.reserve (static_cast<size_t> (n));
  for (int i = 0; i < n; ++i)
    {
      const auto &emb
          = stream[static_cast<size_t> (i)];
      if (emb.size () == x.size () && x.size () > 0)
        {
          const double sim = core::CosineSimilarity (x, emb);
          const double scaled
              = core::Clamp (sim * width_scale, -1.0, 1.0);
          sims.push_back (scaled);
        }
    }
  if (static_cast<int> (sims.size ()) < 2)
    {
      context.SetMetric (operations::Metric::focus_spread, 0.0);
      return;
    }

  const int k_eff = std::min (k, static_cast<int> (sims.size ()));
  if (k_eff < 2)
    {
      context.SetMetric (operations::Metric::focus_spread, 0.0);
      return;
    }

  if (static_cast<int> (sims.size ()) > k_eff)
    {
      std::nth_element (sims.begin (), sims.begin () + k_eff, sims.end (),
                        std::greater<double> ());
      sims.resize (static_cast<size_t> (k_eff));
    }

  // Softmax with numerical stability: subtract max similarity.
  const double max_s = *std::max_element (sims.begin (), sims.end ());
  double denom = 0.0;
  for (double &s : sims)
    {
      s = std::exp (s - max_s);
      denom += s;
    }
  if (denom <= 0.0)
    {
      context.SetMetric (operations::Metric::focus_spread, 0.0);
      return;
    }

  double entropy = 0.0;
  for (double s : sims)
    {
      const double p = s / denom;
      entropy -= (p > 0.0) ? (p * std::log (p)) : 0.0;
    }

  const double norm = std::log (static_cast<double> (k_eff));
  if (norm <= 0.0)
    {
      context.SetMetric (operations::Metric::focus_spread, 0.0);
      return;
    }

  const double focus_spread = core::Clamp (entropy / norm, 0.0, 1.0);
  context.SetMetric (operations::Metric::focus_spread, focus_spread);

  telemetry::LogDebug("cortext.compute_focus_spread", {
    telemetry::Attribute::Double("entropy", entropy),
    telemetry::Attribute::Double("focus_spread", focus_spread)
  });
}

} // namespace cortext::operations
