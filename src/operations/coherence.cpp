#include "cortext/operations/coherence.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include <vector>

namespace cortext::operations
{

namespace
{
constexpr double kEpsilonNoise = 0.02;  // Noise floor for drift (Section 4.4.2)
}  // namespace

void
ComputeCoherence::Execute (OperationContext &context,
                           Transaction & /*tx*/) const
{
  const auto &signal = context.GetSignal ();
  auto &p_ctx = context.GetProcessorContext ();
  const auto &config = context.GetConfig ();
  const std::string &source_id = signal.source_id;

  // Get accumulator state
  auto it = p_ctx.accumulator_states.find (source_id);
  if (it == p_ctx.accumulator_states.end () || it->second.n_signals == 0)
    {
      // No accumulator state - use defaults
      context.SetAccumulatorCoherence (1.0);
      context.SetAccumulatorDriftStep (0.0);
      return;
    }

  auto &acc = it->second;

  // Get drift magnitude
  const double drift_mag = context.GetMetric (Metric::drift_mag).value_or (0.0);

  // Compute d_step with noise floor (Section 4.4.2)
  const double d_step = std::max (drift_mag - kEpsilonNoise, 0.0);

  // Update η_acc via EWMA
  const double alpha = core::AlphaEtaAcc (config.stability);
  acc.eta_acc = core::Ewma (acc.eta_acc, d_step, alpha);

  // Compute coherence as cosine similarity to accumulator mean
  double coherence = 1.0;
  if (acc.mu_acc.size () > 0 && signal.embedding.size () > 0)
    {
      coherence = core::CosineSimilarity (signal.embedding, acc.mu_acc);
      // Map to [0, 1] range for consistency
      coherence = core::Clamp ((coherence + 1.0) / 2.0, 0.0, 1.0);
    }

  // Store for boundary detection
  context.SetAccumulatorCoherence (coherence);
  context.SetAccumulatorDriftStep (d_step);

  // --- Structural Coherence (Section 3.1.1) ---
  // raw = var([cos(x_t, c) for c in context_window])
  // coherence_struct_t = 1 - clamp(raw, 0, 1)
  double coherence_struct = 1.0;  // Default: high coherence
  const int ctx_window = static_cast<int> (core::NCtx (config.stability));
  const int n_ctx = static_cast<int> (p_ctx.recent_context_embeddings.size ());
  if (n_ctx >= 2 && signal.embedding.size () > 0)
    {
      const int start = std::max (0, n_ctx - ctx_window);
      std::vector<double> cos_sims;
      cos_sims.reserve (static_cast<size_t> (n_ctx - start));

      for (int i = start; i < n_ctx; ++i)
        {
          const auto &c
              = p_ctx.recent_context_embeddings[static_cast<size_t> (i)];
          if (c.size () == signal.embedding.size ())
            {
              cos_sims.push_back (core::CosineSimilarity (signal.embedding, c));
            }
        }

      if (cos_sims.size () >= 2)
        {
          // Compute variance of cosine similarities
          double sum = 0.0;
          for (double v : cos_sims)
            {
              sum += v;
            }
          const double mean = sum / static_cast<double> (cos_sims.size ());
          double accum = 0.0;
          for (double v : cos_sims)
            {
              const double diff = v - mean;
              accum += diff * diff;
            }
          const double var_raw
              = accum / static_cast<double> (cos_sims.size ());
          coherence_struct = 1.0 - core::Clamp (var_raw, 0.0, 1.0);
        }
    }
  context.SetStructuralCoherence (coherence_struct);

  telemetry::RecordHistogram ("cortext.accumulator.coherence", coherence);
  telemetry::RecordHistogram ("cortext.accumulator.d_step", d_step);
  telemetry::RecordHistogram ("cortext.accumulator.eta_acc", acc.eta_acc);
  telemetry::RecordHistogram ("cortext.structural_coherence", coherence_struct);

  telemetry::LogDebug ("cortext.coherence", {
    telemetry::Attribute::Double ("d_step", d_step),
    telemetry::Attribute::Double ("eta_acc", acc.eta_acc),
    telemetry::Attribute::Double ("coherence_t", coherence),
    telemetry::Attribute::Double ("coherence_struct", coherence_struct)
  });
}

} // namespace cortext::operations
