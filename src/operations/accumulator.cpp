#include "cortext/operations/accumulator.hpp"

#include "cortext/core/knobs.hpp"
#include "cortext/processor/accumulator_state.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/object_store.hpp"
#include "cortext/signal.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "../experimental_env.hpp"

#include <algorithm>
#include <chrono>
#include <string>

namespace cortext::operations
{

namespace
{
using SteadyClock = std::chrono::steady_clock;

double
ElapsedMillis (SteadyClock::time_point start)
{
  return std::chrono::duration<double, std::milli> (SteadyClock::now () - start)
      .count ();
}

/// @brief Create a SignalRecord from the current signal and context
SignalRecord
CreateSignalRecord (const Signal &signal, double score, int serial_position,
                    OperationContext &context, Transaction &tx)
{
  SignalRecord rec;
  rec.embedding = signal.embedding;
  rec.timestamp = signal.timestamp;
  rec.modality = signal.modality;
  rec.mime = signal.mimetype;
  rec.score = score;
  rec.serial_position = serial_position;
  if ((signal.modality == "text" || signal.mimetype == "text/plain")
      && signal.payload && !signal.payload->empty ())
    {
      rec.text_payload.assign (
          reinterpret_cast<const char *> (signal.payload->data ()),
          signal.payload->size ());
    }
  if (signal.payload && !signal.payload->empty ())
    {
      const auto put_start = SteadyClock::now ();
      rec.blob_id
          = PutObject (context.GetObjectTransaction (), tx, *signal.payload);
      context.AddOperationTiming ("UpdateAccumulator.signal_payload_put",
                                  ElapsedMillis (put_start));
    }
  return rec;
}
} // namespace

void
UpdateAccumulator::Execute (OperationContext &context,
                            Transaction &tx) const
{
  const auto &signal = context.GetSignal ();
  auto &p_ctx = context.GetProcessorContext ();
  const auto &config = context.GetConfig ();
  const std::string &source_id = signal.source_id;
  const int64_t episode_id
      = (p_ctx.episode_start_ts > 0)
            ? static_cast<int64_t> (p_ctx.episode_start_ts)
            : 0;
  // Ephemeral queries must not grow storable accumulator units.
  // True for Natural/Boundary/Durable (everything that may persist signals).
  const bool track_storable_signal = signal.retention != Retention::Ephemeral;
  context.SetInterruptAborted (false);

  // Use drift step for accumulation (Section 4.4.2)
  const double drift_step = context.GetAccumulatorDriftStep ();
  const double drift_gain
      = core::AccumulatorDriftGain (config.sensitivity, config.stability);

  // Get emotional metadata from context (Section 6.1.1)
  const double emotion_intensity = p_ctx.emotion_intensity_ewma;
  const double arousal = p_ctx.arousal_ewma;

  // Get or create accumulator state for this source
  auto it = p_ctx.accumulator_states.find (source_id);
  if (it == p_ctx.accumulator_states.end ())
    {
      // New source - initialize accumulator state
      AccumulatorState state;
      state.Reset (signal.embedding, signal.timestamp);
      state.drift_acc = drift_step * drift_gain;
      state.episode_id = episode_id;
      // Initialize temporal context with normalized embedding
      if (state.c_t.size () > 0)
        {
          const float norm = state.c_t.norm ();
          if (norm > 1e-9f)
            {
              state.c_t /= norm;
            }
        }

      // Track emotional metadata (Section 6.1.1)
      state.s_emotion_max = emotion_intensity;
      state.s_arousal_sum = arousal;

      // Track first signal for SIGNALS table (Section 4.4)
      if (track_storable_signal)
        {
          state.signals.push_back (
              CreateSignalRecord (signal, 0.0, 0, context, tx));
        }

      // Track primary modality (v2: first modality wins)
      if (track_storable_signal)
        {
          state.primary_modality = signal.modality;
        }
      if (state.mu_acc.size () > 0)
        {
          state.acc_signals_window.push_back (state.mu_acc);
          const int win_coh = core::WinCoh (config.stability);
          if (win_coh > 0
              && static_cast<int> (state.acc_signals_window.size ()) > win_coh)
            {
              state.acc_signals_window.erase (
                  state.acc_signals_window.begin (),
                  state.acc_signals_window.begin ()
                      + (static_cast<long> (state.acc_signals_window.size ())
                         - win_coh));
            }
        }

      p_ctx.accumulator_states[source_id] = std::move (state);

      context.SetWriteExclusionTs (
          p_ctx.accumulator_states[source_id].t_start);
      telemetry::AddCounter ("cortext.accumulator.new_source_total", 1);
      return;
    }

  // Existing source - accumulate
  auto &acc = it->second;

  if (acc.pending_interrupt_abort)
    {
      const bool disable_accept = internal::experimental_env::Flag (
          "CORTEXT_DISABLE_INTERRUPT_ACCEPT");
      bool accept = false;
      const bool can_compare
          = (acc.pending_interrupt_embedding.size () > 0
             && acc.pending_interrupt_embedding.size ()
                    == signal.embedding.size ()
             && acc.mu_acc.size () == signal.embedding.size ()
             && acc.mu_acc.size () > 0);
      if (!disable_accept && can_compare)
        {
          const double sim_selected
              = core::CosineSimilarity (signal.embedding,
                                        acc.pending_interrupt_embedding);
          const double sim_mu
              = core::CosineSimilarity (signal.embedding, acc.mu_acc);
          const double margin_base = core::InterruptAbortAcceptanceMargin (
              config.focus, config.sensitivity, config.stability);
          accept = (sim_selected > (sim_mu + margin_base));
        }

      if (accept)
        {
          acc.ResetForNextUnit (signal.timestamp);
          context.SetInterruptAborted (true);
          telemetry::AddCounter ("cortext.accumulator.interrupt_abort_total", 1);
        }
      else
        {
          telemetry::AddCounter ("cortext.accumulator.interrupt_resume_total", 1);
        }
      acc.pending_interrupt_abort = false;
      acc.pending_interrupt_embedding = Eigen::VectorXf ();
    }

  // Check if this is first signal after reset (n_signals == 0 means post-reset)
  if (acc.n_signals == 0)
    {
      const Eigen::VectorXf prev_ctx = acc.c_t;
      acc.Reset (signal.embedding, signal.timestamp);
      acc.drift_acc = drift_step * drift_gain;
      acc.episode_id = episode_id;
      if (prev_ctx.size () == acc.c_t.size () && prev_ctx.size () > 0)
        {
          const double alpha_c = core::AccumulatorTemporalContextAlpha (
              config.sensitivity, config.stability);
          acc.c_t = prev_ctx * static_cast<float> (1.0 - alpha_c)
                    + acc.c_t * static_cast<float> (alpha_c);
        }
      if (acc.c_t.size () > 0)
        {
          const float norm = acc.c_t.norm ();
          if (norm > 1e-9f)
            {
              acc.c_t /= norm;
            }
        }

      // Track emotional metadata (Section 6.1.1)
      acc.s_emotion_max = emotion_intensity;
      acc.s_arousal_sum = arousal;

      // Track first signal for SIGNALS table (Section 4.4)
      if (track_storable_signal)
        {
          acc.signals.push_back (
              CreateSignalRecord (signal, 0.0, 0, context, tx));
        }

      // Track primary modality (v2: first modality wins)
      if (track_storable_signal)
        {
          acc.primary_modality = signal.modality;
        }
      if (acc.mu_acc.size () > 0)
        {
          acc.acc_signals_window.push_back (acc.mu_acc);
          const int win_coh = core::WinCoh (config.stability);
          if (win_coh > 0
              && static_cast<int> (acc.acc_signals_window.size ()) > win_coh)
            {
              acc.acc_signals_window.erase (
                  acc.acc_signals_window.begin (),
                  acc.acc_signals_window.begin ()
                      + (static_cast<long> (acc.acc_signals_window.size ())
                         - win_coh));
            }
        }
      context.SetWriteExclusionTs (acc.t_start);
      return;
    }

  // Accumulate into memory
  acc.Accumulate (signal.embedding, drift_step, drift_gain);

  // Update temporal context (slow drift)
  const double alpha_c = core::AccumulatorTemporalContextAlpha (
      config.sensitivity, config.stability);
  if (acc.c_t.size () == 0 || acc.c_t.size () != acc.mu_acc.size ())
    {
      acc.c_t = acc.mu_acc;
    }
  else
    {
      acc.c_t = acc.c_t * static_cast<float> (1.0 - alpha_c)
                + acc.mu_acc * static_cast<float> (alpha_c);
    }
  if (acc.c_t.size () > 0)
    {
      const float norm = acc.c_t.norm ();
      if (norm > 1e-9f)
        {
          acc.c_t /= norm;
        }
    }

  // Track emotional metadata (Section 6.1.1)
  acc.s_emotion_max = std::max (acc.s_emotion_max, emotion_intensity);
  acc.s_arousal_sum += arousal;

  // Track signal for SIGNALS table (Section 4.4)
  // Serial position is the current signal count before accumulation
  if (track_storable_signal)
    {
      const int serial_pos = static_cast<int> (acc.signals.size ());
      acc.signals.push_back (
          CreateSignalRecord (signal, 0.0, serial_pos, context, tx));
    }

  // Track primary modality (v2: first modality wins)
  if (track_storable_signal && acc.primary_modality.empty ())
    {
      acc.primary_modality = signal.modality;
    }

  telemetry::RecordHistogram ("cortext.accumulator.n_signals",
                              static_cast<double> (acc.n_signals));
  telemetry::RecordHistogram ("cortext.accumulator.drift_acc", acc.drift_acc);
  telemetry::LogDebug ("cortext.accumulator", {
    telemetry::Attribute::Int64 ("signal_count", acc.n_signals),
    telemetry::Attribute::Double ("running_mean_norm", acc.mu_acc.norm ()),
    telemetry::Attribute::Double ("drift_sum", acc.drift_acc),
    telemetry::Attribute::Double ("emotion_max", acc.s_emotion_max),
    telemetry::Attribute::Double ("arousal_sum", acc.s_arousal_sum),
    telemetry::Attribute::Int64 ("tracked_signals", static_cast<int64_t> (acc.signals.size ()))
  });

  context.SetWriteExclusionTs (acc.t_start);
}

} // namespace cortext::operations
