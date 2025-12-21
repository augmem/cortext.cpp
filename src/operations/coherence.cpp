#include "cortext/operations/coherence.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include <deque>
#include <vector>

namespace cortext::operations
{

namespace
{
constexpr double kEpsilonNoise = 0.02;  // Noise floor for drift (Section 4.4.2)

inline Eigen::VectorXf
ComputeMean (const std::deque<Eigen::VectorXf> &embs, int start, int end)
{
  if (start >= end)
    {
      return Eigen::VectorXf ();
    }
  const int dim = static_cast<int> (embs[static_cast<size_t> (start)].size ());
  Eigen::VectorXf mean = Eigen::VectorXf::Zero (dim);
  const int count = end - start;
  for (int i = start; i < end; ++i)
    {
      mean += embs[static_cast<size_t> (i)];
    }
  mean /= static_cast<float> (count);
  return mean;
}

inline double
ComputeLaggedDriftMag (const std::deque<Eigen::VectorXf> &embs, double T)
{
  const int n_ctx_total = static_cast<int> (embs.size ());
  const int ctx_window = static_cast<int> (core::NCtx (T));
  const int k_ctx = core::KCtx (T);
  if (n_ctx_total < ctx_window + k_ctx)
    {
      return 0.0;
    }
  const Eigen::VectorXf mean_recent
      = ComputeMean (embs, n_ctx_total - ctx_window, n_ctx_total);
  const Eigen::VectorXf mean_prev
      = ComputeMean (embs, n_ctx_total - k_ctx - ctx_window,
                     n_ctx_total - k_ctx);
  if (mean_recent.size () == 0 || mean_prev.size () == 0
      || mean_recent.size () != mean_prev.size ())
    {
      return 0.0;
    }
  const Eigen::VectorXf nr = (mean_recent.norm () > 0.0f)
                                 ? (mean_recent / mean_recent.norm ())
                                 : mean_recent;
  const Eigen::VectorXf np = (mean_prev.norm () > 0.0f)
                                 ? (mean_prev / mean_prev.norm ())
                                 : mean_prev;
  return (nr - np).norm ();
}
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
  AccumulatorState *acc = nullptr;
  if (it != p_ctx.accumulator_states.end () && it->second.n_signals > 0)
    {
      acc = &it->second;
    }

  if (!acc)
    {
      // No accumulator state - use defaults for accumulator-specific signals.
      context.SetAccumulatorCoherence (1.0);
      context.SetAccumulatorDriftStep (0.0);
    }
  else
    {
      // Get drift magnitude
      double drift_mag = 0.0;
      if (auto v = context.GetMetric (Metric::drift_mag))
        {
          drift_mag = *v;
        }
      else
        {
          drift_mag
              = ComputeLaggedDriftMag (p_ctx.recent_context_embeddings,
                                       config.stability);
          context.SetMetric (Metric::drift_mag, drift_mag);
        }

      // Compute d_step with noise floor (Section 4.4.2)
      const double d_step = std::max (drift_mag - kEpsilonNoise, 0.0);

      const double eta_prev = acc->eta_acc;
      context.SetAccumulatorEtaPrev (eta_prev);

      // Update η_acc via EWMA
      const double alpha = core::AlphaEtaAcc (config.stability);
      acc->eta_acc = core::Ewma (eta_prev, d_step, alpha);

      // Compute coherence as mean cosine over the accumulator window (raw [-1,1])
      const int win_coh = core::WinCoh (config.stability);
      auto &window = acc->acc_signals_window;
      if (win_coh > 0 && static_cast<int> (window.size ()) > win_coh)
        {
          window.erase (window.begin (),
                        window.begin ()
                            + (static_cast<long> (window.size ()) - win_coh));
        }

      double coherence = 1.0;
      if (!window.empty () && signal.embedding.size () > 0)
        {
          double sum = 0.0;
          int count = 0;
          for (const auto &emb : window)
            {
              if (emb.size () == signal.embedding.size ())
                {
                  sum += core::CosineSimilarity (signal.embedding, emb);
                  ++count;
                }
            }
          if (count > 0)
            {
              coherence = sum / static_cast<double> (count);
            }
        }

      // Store for boundary detection
      context.SetAccumulatorCoherence (coherence);
      context.SetAccumulatorDriftStep (d_step);

      // Append current embedding after coherence calculation (unless accumulator was reset)
      if (acc->n_signals > 0 && signal.embedding.size () > 0)
        {
          window.push_back (signal.embedding);
          if (win_coh > 0 && static_cast<int> (window.size ()) > win_coh)
            {
              window.erase (window.begin (),
                            window.begin ()
                                + (static_cast<long> (window.size ()) - win_coh));
            }
        }
    }

  // --- Structural Coherence (Section 3.1.1) ---
  // raw = var([cos(x_t, c) for c in context_window])
  // coherence_struct_t = 1 - clamp(raw, 0, 1)
  double coherence_struct = 0.5;  // Default: neutral coherence when context < 2
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
  context.SetCoherence (coherence_struct);

  const double coherence = context.GetAccumulatorCoherence ();
  const double d_step = context.GetAccumulatorDriftStep ();
  const double eta_acc = acc ? acc->eta_acc : 0.0;

  telemetry::RecordHistogram ("cortext.accumulator.coherence", coherence);
  telemetry::RecordHistogram ("cortext.accumulator.d_step", d_step);
  telemetry::RecordHistogram ("cortext.accumulator.eta_acc", eta_acc);
  telemetry::RecordHistogram ("cortext.structural_coherence", coherence_struct);

  telemetry::LogDebug ("cortext.coherence", {
    telemetry::Attribute::Double ("d_step", d_step),
    telemetry::Attribute::Double ("eta_acc", eta_acc),
    telemetry::Attribute::Double ("coherence_t", coherence),
    telemetry::Attribute::Double ("coherence_struct", coherence_struct)
  });
}

} // namespace cortext::operations
