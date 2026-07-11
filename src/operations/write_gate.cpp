#include "cortext/operations/write_gate.hpp"

#include "neuromodulator_internal.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include <cmath>

namespace cortext::operations
{

void
ComputeWriteGate::Execute (OperationContext &context,
                           Transaction &tx) const
{
  (void)tx;
  const auto &signal = context.GetSignal ();
  auto &p_ctx = context.GetProcessorContext ();
  const auto &config = context.GetConfig ();
  const std::string &source_id = signal.source_id;

  // Check if flush is required
  const bool flush = context.GetFlushRequired ();
  const bool spike_bypass = context.GetSpikeBypass ();

  if (signal.retention == Retention::Ephemeral)
    {
      context.SetAccumulatorWriteDecision (false);
      context.SetWriteDecision (false);
      telemetry::AddCounter ("cortext.retention.ephemeral_no_store_total", 1);
      telemetry::LogDebug ("cortext.write_gate", {
        telemetry::Attribute::String ("retention", "ephemeral"),
        telemetry::Attribute::Bool ("write_accumulator", false)
      });
      return;
    }

  if (!flush && !spike_bypass)
    {
      // No flush trigger - no write decision
      context.SetAccumulatorWriteDecision (false);
      return;
    }

  // Get accumulator state
  auto it = p_ctx.accumulator_states.find (source_id);
  if (it == p_ctx.accumulator_states.end () || it->second.n_signals == 0)
    {
      // No accumulator state - fall back to per-signal gate
      context.SetAccumulatorWriteDecision (false);
      return;
    }

  auto &acc = it->second;

  // Compute window score (Section 4.4.5)
  const int n = std::max (acc.n_signals, 1);
  const double s_avg = acc.s_sum / static_cast<double> (n);
  const double n_ctx = core::NCtx (config.stability);
  const double coverage
      = std::min (static_cast<double> (n) / n_ctx, 1.0);

  const double alpha = core::WindowScoreAlpha (config.focus);
  const double beta = core::WindowScoreCoverageBeta (config.sensitivity);
  const double S_window
      = alpha * acc.s_max + (1.0 - alpha) * s_avg + beta * coverage;

  context.SetWindowScore (S_window);

  // Compute write refractory multiplier
  const double T_dynamic = context.GetThresholdTDynamic ();
  const double tau_refrac = core::WriteRefractoryTau (config.stability);
  const double k_refrac = core::WriteRefractoryK (config.stability);
  const double dt_write
      = std::max (
          0.0,
          (static_cast<double> (signal.timestamp)
           - static_cast<double> (acc.last_write_ts))
              / 1000.0);

  double M_write_refrac = 1.0;
  if (acc.last_write_ts > 0)
    {
      M_write_refrac = 1.0 + k_refrac * std::exp (-dt_write / tau_refrac);
    }

  // Effective accumulator threshold
  const double neuromod_ne = core::Clamp (p_ctx.neuromod_ne, 0.0, 1.0);
  const double write_scale = neuromodulation::WriteThresholdScale (neuromod_ne);
  const double theta_accumulator = T_dynamic * M_write_refrac * write_scale;

  // Final write decision
  const bool force_write = spike_bypass || RetentionForcesWrite (signal.retention);
  const bool write_accumulator
      = force_write || (flush && (S_window > theta_accumulator));

  context.SetAccumulatorWriteDecision (write_accumulator);
  context.SetWriteDecision (write_accumulator);

  const double rho = core::RepresentativeBlendRho (config.focus);
  if (write_accumulator)
    {
      // Compute representative embedding (Section 4.4.5)
      Eigen::VectorXf e_rep;
      if (acc.mu_acc.size () == 0 && acc.e_peak.size () == 0)
        {
          e_rep = acc.mu_acc;
        }
      else if (acc.mu_acc.size () == 0)
        {
          e_rep = acc.e_peak;
        }
      else if (acc.e_peak.size () == 0 || acc.e_peak.size () != acc.mu_acc.size ())
        {
          e_rep = acc.mu_acc;
        }
      else
        {
          e_rep = rho * acc.mu_acc + (1.0 - rho) * acc.e_peak;
        }

      // Normalize
      const float norm = e_rep.norm ();
      if (norm > 1e-9f)
        {
          e_rep /= norm;
        }

      context.SetRepresentativeEmbedding (e_rep);

      if (e_rep.size () > 0)
        {
          p_ctx.memory_stream.push_back (e_rep);
          const size_t max_memory_stream
              = static_cast<size_t> (std::max (
                  core::KNeighbors (config.stability),
                  static_cast<int> (core::NCtx (config.stability))
                      + core::KCtx (config.stability)));
          while (p_ctx.memory_stream.size () > max_memory_stream)
            {
              p_ctx.memory_stream.pop_front ();
            }
        }

      // Section 8.2: Populate recent_memory_centroids for interrupt gate context.
      // Store normalized accumulator centroid (μ_acc) for memory-level context.
      if (acc.mu_acc.size () > 0)
        {
          Eigen::VectorXf mu_norm = acc.mu_acc;
          const float mu_norm_val = mu_norm.norm ();
          if (mu_norm_val > 1e-9f)
            {
              mu_norm /= mu_norm_val;
            }
          p_ctx.recent_memory_centroids.push_back (mu_norm);

          // Trim to win_mem_ctx(T) size
          const size_t max_size
              = static_cast<size_t> (core::WinMemCtx (config.stability));
          while (p_ctx.recent_memory_centroids.size () > max_size)
            {
              p_ctx.recent_memory_centroids.pop_front ();
            }
        }

      // Update last_write_ts; accumulator reset happens after persistence.
      acc.last_write_ts = signal.timestamp;

      telemetry::AddCounter ("cortext.accumulator.write_accept_total", 1);
    }
  else
    {
      // No write; accumulator reset happens after persistence.
      telemetry::AddCounter ("cortext.accumulator.write_reject_total", 1);
    }

  telemetry::RecordHistogram ("cortext.accumulator.S_window", S_window);
  telemetry::RecordHistogram ("cortext.accumulator.theta_accumulator", theta_accumulator);
  telemetry::RecordHistogram ("cortext.accumulator.M_write_refrac", M_write_refrac);
  telemetry::RecordHistogram ("cortext.accumulator.coverage", coverage);

  telemetry::LogDebug ("cortext.write_gate", {
    telemetry::Attribute::Bool ("flush", flush),
    telemetry::Attribute::Bool ("spike_bypass", spike_bypass),
    telemetry::Attribute::Double ("n_signals", static_cast<double> (n)),
    telemetry::Attribute::Double ("n_ctx", n_ctx),
    telemetry::Attribute::Double ("coverage", coverage),
    telemetry::Attribute::Double ("alpha", alpha),
    telemetry::Attribute::Double ("beta", beta),
    telemetry::Attribute::Double ("s_max", acc.s_max),
    telemetry::Attribute::Double ("s_avg", s_avg),
    telemetry::Attribute::Double ("S_window", S_window),
    telemetry::Attribute::Double ("T_dynamic", T_dynamic),
    telemetry::Attribute::Double ("tau_refrac", tau_refrac),
    telemetry::Attribute::Double ("k_refrac", k_refrac),
    telemetry::Attribute::Double ("dt_write", dt_write),
    telemetry::Attribute::Double ("refractory_mult", M_write_refrac),
    telemetry::Attribute::Double ("neuromod_ne", neuromod_ne),
    telemetry::Attribute::Double ("write_scale", write_scale),
    telemetry::Attribute::Double ("theta_accumulator", theta_accumulator),
    telemetry::Attribute::Double ("rho", rho),
    telemetry::Attribute::Bool ("force_write", force_write),
    telemetry::Attribute::Bool ("write_accumulator", write_accumulator)
  });
}

} // namespace cortext::operations
