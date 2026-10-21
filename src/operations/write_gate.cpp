#include "cortext/operations/write_gate.hpp"

#include "../experimental_env.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include <cmath>

namespace cortext::operations
{

namespace
{

enum class WriteGateProfileReason
{
  Ephemeral = 1,
  NoTrigger = 2,
  NoAccumulator = 3,
  Forced = 4,
  Closed = 5,
};

bool
ProfileWriteGateEnabled ()
{
  return internal::experimental_env::Flag ("CORTEXT_PROFILE_WORK_COUNTERS");
}

void
RecordWriteGateProfile (OperationContext &context, bool flush,
                        bool accumulator_available,
                        double n_signals, double coverage,
                        double window_score, bool force_write,
                        bool write_accumulator, WriteGateProfileReason reason)
{
  if (!ProfileWriteGateEnabled ())
    return;

  context.AddOperationTiming ("WriteGate.flush_trigger", flush ? 1.0 : 0.0);
  context.AddOperationTiming ("WriteGate.accumulator_available",
                              accumulator_available ? 1.0 : 0.0);
  context.AddOperationTiming ("WriteGate.n_signals", n_signals);
  context.AddOperationTiming ("WriteGate.coverage", coverage);
  context.AddOperationTiming ("WriteGate.window_score", window_score);
  context.AddOperationTiming ("WriteGate.force_write",
                              force_write ? 1.0 : 0.0);
  context.AddOperationTiming ("WriteGate.write_accumulator",
                              write_accumulator ? 1.0 : 0.0);
  context.AddOperationTiming ("WriteGate.reason_code",
                              static_cast<double> (reason));
}

} // namespace

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

  if (signal.retention == Retention::Ephemeral)
    {
      context.SetAccumulatorWriteDecision (false);
      context.SetWriteDecision (false);
      RecordWriteGateProfile (
          context, flush, false, 0.0, 0.0, 0.0, false,
          false, WriteGateProfileReason::Ephemeral);
      telemetry::AddCounter ("cortext.retention.ephemeral_no_store_total", 1);
      telemetry::LogDebug ("cortext.write_gate", {
        telemetry::Attribute::String ("retention", "ephemeral"),
        telemetry::Attribute::Bool ("write_accumulator", false)
      });
      return;
    }

  if (!flush)
    {
      // No flush trigger - no write decision
      context.SetAccumulatorWriteDecision (false);
      RecordWriteGateProfile (
          context, flush, false, 0.0, 0.0, 0.0, false,
          false, WriteGateProfileReason::NoTrigger);
      return;
    }

  // Get accumulator state
  auto it = p_ctx.accumulator_states.find (source_id);
  if (it == p_ctx.accumulator_states.end () || it->second.n_signals == 0)
    {
      // No accumulator state - fall back to per-signal gate
      context.SetAccumulatorWriteDecision (false);
      RecordWriteGateProfile (
          context, flush, false, 0.0, 0.0, 0.0, false,
          false, WriteGateProfileReason::NoAccumulator);
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

  // Every closed unit is stored. A boundary (natural or capacity) or a
  // Durable turn already decided the unit is complete; a score threshold here only
  // discarded finished units: under per-conversation Natural ingest it dropped
  // 7-11% of gold turns and lowered packet recall. S_window still feeds
  // working-memory admission.
  const bool force_write = RetentionForcesWrite (signal.retention);

  RecordWriteGateProfile (
      context, flush, true, static_cast<double> (n), coverage,
      S_window, force_write, true,
      force_write ? WriteGateProfileReason::Forced
                  : WriteGateProfileReason::Closed);

  context.SetAccumulatorWriteDecision (true);
  context.SetWriteDecision (true);

  const double rho = core::RepresentativeBlendRho (config.focus);
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

  // Accumulator reset happens after persistence.
  telemetry::AddCounter ("cortext.accumulator.write_accept_total", 1);

  telemetry::RecordHistogram ("cortext.accumulator.S_window", S_window);
  telemetry::RecordHistogram ("cortext.accumulator.coverage", coverage);

  telemetry::LogDebug ("cortext.write_gate", {
    telemetry::Attribute::Bool ("flush", flush),
    telemetry::Attribute::Double ("n_signals", static_cast<double> (n)),
    telemetry::Attribute::Double ("n_ctx", n_ctx),
    telemetry::Attribute::Double ("coverage", coverage),
    telemetry::Attribute::Double ("alpha", alpha),
    telemetry::Attribute::Double ("beta", beta),
    telemetry::Attribute::Double ("s_max", acc.s_max),
    telemetry::Attribute::Double ("s_avg", s_avg),
    telemetry::Attribute::Double ("S_window", S_window),
    telemetry::Attribute::Double ("rho", rho),
    telemetry::Attribute::Bool ("force_write", force_write)
  });
}

} // namespace cortext::operations
