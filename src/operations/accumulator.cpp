#include "cortext/operations/accumulator.hpp"

#include "cortext/core/knobs.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/processor/accumulator_state.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/signal.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include <algorithm>

namespace cortext::operations
{

namespace
{
/// @brief Create a SignalRecord from the current signal and context
SignalRecord
CreateSignalRecord (const Signal &signal, double score, int serial_position)
{
  SignalRecord rec;
  rec.embedding = signal.embedding;
  rec.timestamp = signal.timestamp;
  rec.modality = signal.modality;
  rec.mime = signal.mimetype;
  rec.score = score;
  rec.serial_position = serial_position;
  // blob_id will be populated later if content is stored
  return rec;
}
} // namespace

void
UpdateAccumulator::Execute (OperationContext &context,
                            Transaction & /*tx*/) const
{
  const auto &signal = context.GetSignal ();
  auto &p_ctx = context.GetProcessorContext ();
  const std::string &source_id = signal.source_id;

  // Get composite score (default to 0.0)
  const double score = context.GetCompositeScore ().value_or (0.0);

  // Get drift magnitude from metrics
  const double drift_mag = context.GetMetric (Metric::drift_mag).value_or (0.0);

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
      state.s_sum = score;
      state.s_max = score;

      // Track emotional metadata (Section 6.1.1)
      state.s_emotion_max = emotion_intensity;
      state.s_arousal_sum = arousal;

      // Track first signal for SIGNALS table (Section 4.4)
      state.signals.push_back (CreateSignalRecord (signal, score, 0));

      // Track primary modality (v2: first modality wins)
      state.primary_modality = signal.modality;

      p_ctx.accumulator_states[source_id] = std::move (state);

      telemetry::AddCounter ("cortext.accumulator.new_source_total", 1);
      return;
    }

  // Existing source - accumulate
  auto &acc = it->second;

  // Check if this is first signal after reset (n_signals == 0 means post-reset)
  if (acc.n_signals == 0)
    {
      acc.Reset (signal.embedding, signal.timestamp);
      acc.s_sum = score;
      acc.s_max = score;

      // Track emotional metadata (Section 6.1.1)
      acc.s_emotion_max = emotion_intensity;
      acc.s_arousal_sum = arousal;

      // Track first signal for SIGNALS table (Section 4.4)
      acc.signals.push_back (CreateSignalRecord (signal, score, 0));

      // Track primary modality (v2: first modality wins)
      acc.primary_modality = signal.modality;
      return;
    }

  // Accumulate into memory
  acc.Accumulate (signal.embedding, score, drift_mag);
  acc.last_signal_ts = signal.timestamp;

  // Track emotional metadata (Section 6.1.1)
  acc.s_emotion_max = std::max (acc.s_emotion_max, emotion_intensity);
  acc.s_arousal_sum += arousal;

  // Track signal for SIGNALS table (Section 4.4)
  // Serial position is the current signal count before accumulation
  const int serial_pos = static_cast<int> (acc.signals.size ());
  acc.signals.push_back (CreateSignalRecord (signal, score, serial_pos));

  // Track primary modality (v2: first modality wins)
  if (acc.primary_modality.empty ())
    {
      acc.primary_modality = signal.modality;
    }

  telemetry::RecordHistogram ("cortext.accumulator.n_signals",
                              static_cast<double> (acc.n_signals));
  telemetry::RecordHistogram ("cortext.accumulator.drift_acc", acc.drift_acc);
  telemetry::RecordHistogram ("cortext.accumulator.s_sum", acc.s_sum);

  telemetry::LogDebug ("cortext.accumulator", {
    telemetry::Attribute::Int64 ("signal_count", acc.n_signals),
    telemetry::Attribute::Double ("running_mean_norm", acc.mu_acc.norm ()),
    telemetry::Attribute::Double ("drift_sum", acc.drift_acc),
    telemetry::Attribute::Double ("score_sum", acc.s_sum),
    telemetry::Attribute::Double ("emotion_max", acc.s_emotion_max),
    telemetry::Attribute::Double ("arousal_sum", acc.s_arousal_sum),
    telemetry::Attribute::Int64 ("tracked_signals", static_cast<int64_t> (acc.signals.size ()))
  });
}

} // namespace cortext::operations
