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

void
ComputeCoherence::Execute (OperationContext &context,
                           Transaction & /*tx*/) const
{
  const auto &signal = context.GetSignal ();
  auto &p_ctx = context.GetProcessorContext ();
  const auto &config = context.GetConfig ();
  const std::string &source_id = signal.source_id;
  const double drift_noise_floor = core::DriftNoiseFloor (config.stability);
  double drift_cos = 1.0;

  // Get accumulator state
  auto it = p_ctx.accumulator_states.find (source_id);
  AccumulatorState *acc = nullptr;
  if (it != p_ctx.accumulator_states.end ())
    {
      acc = &it->second;
    }

  // Drift step from consecutive signals (independent of accumulation state).
  double d_step = 0.0;
  if (acc && acc->prev_x.size () > 0
      && acc->prev_x.size () == signal.embedding.size ()
      && signal.embedding.size () > 0)
    {
      drift_cos = core::CosineSimilarity (signal.embedding, acc->prev_x);
      d_step = std::max (1.0 - drift_cos - drift_noise_floor, 0.0);
    }

  // Coherence defaults when no active accumulator window is available.
  double coherence = 1.0;
  if (acc && acc->n_signals > 0)
    {
      // Compute centroid coherence window
      Eigen::VectorXf mu_prev = acc->mu_acc;
      Eigen::VectorXf mu_curr;
      if (mu_prev.size () == signal.embedding.size ()
          && signal.embedding.size () > 0)
        {
          mu_curr = mu_prev
                    + (signal.embedding - mu_prev)
                          / static_cast<float> (acc->n_signals + 1);
        }
      else
        {
          mu_prev = signal.embedding;
          mu_curr = signal.embedding;
        }

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

      if (!window.empty () && mu_curr.size () > 0)
        {
          double sum = 0.0;
          int count = 0;
          for (const auto &emb : window)
            {
              if (emb.size () == mu_curr.size ())
                {
                  sum += core::CosineSimilarity (mu_curr, emb);
                  ++count;
                }
            }
          if (count > 0)
            {
              coherence = sum / static_cast<double> (count);
            }
        }

      // Append current centroid after coherence calculation (unless accumulator was reset)
      if (mu_curr.size () > 0)
        {
          window.push_back (mu_curr);
          if (win_coh > 0 && static_cast<int> (window.size ()) > win_coh)
            {
              window.erase (window.begin (),
                            window.begin ()
                                + (static_cast<long> (window.size ()) - win_coh));
            }
        }
    }

  context.SetAccumulatorCoherence (coherence);
  context.SetAccumulatorDriftStep (d_step);
  context.SetMetric (Metric::drift_mag, d_step);

  // --- Structural Coherence (Section 3.1.1) ---
  // raw = var([cos(x_t, c) for c in context_window])
  // coherence_struct_t = 1 - clamp(raw, 0, 1)
  double coherence_struct
      = core::CoherenceNeutral (config.stability);  // stability-derived neutral
  const int ctx_window = static_cast<int> (core::NCtx (config.stability));
  const int n_ctx = static_cast<int> (p_ctx.recent_context_embeddings.size ());
  double var_raw = -1.0;
  int cos_count = 0;
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
          var_raw = accum / static_cast<double> (cos_sims.size ());
          coherence_struct = 1.0 - core::Clamp (var_raw, 0.0, 1.0);
        }
      cos_count = static_cast<int> (cos_sims.size ());
    }
  context.SetStructuralCoherence (coherence_struct);
  context.SetCoherence (coherence_struct);

  const double coherence_val = context.GetAccumulatorCoherence ();
  const double d_step_val = context.GetAccumulatorDriftStep ();
  const double eta_acc = acc ? acc->eta_acc : 0.0;
  const int win_coh = core::WinCoh (config.stability);
  const int acc_window_size
      = acc ? static_cast<int> (acc->acc_signals_window.size ()) : 0;
  double mu_prev_norm = -1.0;
  double mu_curr_norm = -1.0;
  double mu_cos = -1.0;
  if (acc && acc->mu_acc.size () > 0 && signal.embedding.size () > 0)
    {
      const Eigen::VectorXf mu_prev = acc->mu_acc;
      Eigen::VectorXf mu_curr = mu_prev;
      if (acc->n_signals > 0 && mu_prev.size () == signal.embedding.size ())
        {
          mu_curr = mu_prev
                    + (signal.embedding - mu_prev)
                          / static_cast<float> (acc->n_signals + 1);
        }
      mu_prev_norm = static_cast<double> (mu_prev.norm ());
      mu_curr_norm = static_cast<double> (mu_curr.norm ());
      mu_cos = core::CosineSimilarity (mu_curr, mu_prev);
    }

  telemetry::RecordHistogram ("cortext.accumulator.coherence", coherence_val);
  telemetry::RecordHistogram ("cortext.accumulator.d_step", d_step_val);
  telemetry::RecordHistogram ("cortext.accumulator.eta_acc", eta_acc);
  telemetry::RecordHistogram ("cortext.structural_coherence", coherence_struct);

  telemetry::LogDebug ("cortext.coherence", {
    telemetry::Attribute::Int64 ("win_coh", static_cast<int64_t> (win_coh)),
    telemetry::Attribute::Int64 ("acc_window_size", static_cast<int64_t> (acc_window_size)),
    telemetry::Attribute::Double ("mu_prev_norm", mu_prev_norm),
    telemetry::Attribute::Double ("mu_curr_norm", mu_curr_norm),
    telemetry::Attribute::Double ("mu_cos", mu_cos),
    telemetry::Attribute::Double ("drift_cos", drift_cos),
    telemetry::Attribute::Double ("drift_noise_floor", core::DriftNoiseFloor (config.stability)),
    telemetry::Attribute::Double ("d_step", d_step_val),
    telemetry::Attribute::Double ("eta_acc", eta_acc),
    telemetry::Attribute::Double ("coherence_t", coherence_val),
    telemetry::Attribute::Double ("coherence_struct", coherence_struct),
    telemetry::Attribute::Int64 ("ctx_window", static_cast<int64_t> (ctx_window)),
    telemetry::Attribute::Int64 ("n_ctx", static_cast<int64_t> (n_ctx)),
    telemetry::Attribute::Int64 ("cos_count", static_cast<int64_t> (cos_count)),
    telemetry::Attribute::Double ("var_raw", var_raw)
  });
}

} // namespace cortext::operations
