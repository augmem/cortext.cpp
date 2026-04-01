#include "cortext/operations/accumulator_reset.hpp"

#include "cortext/processor/accumulator_state.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext::operations
{

void
ResetAccumulatorAfterFlush::Execute (OperationContext &context,
                                     Transaction & /*tx*/) const
{
  const bool flush = context.GetFlushRequired ();
  const bool spike_bypass = context.GetSpikeBypass ();
  if (!flush && !spike_bypass)
    {
      return;
    }

  auto &p_ctx = context.GetProcessorContext ();
  const auto &signal = context.GetSignal ();
  auto it = p_ctx.accumulator_states.find (signal.source_id);
  if (it == p_ctx.accumulator_states.end ())
    {
      return;
    }

  it->second.ResetForNextUnit (signal.timestamp);
}

void
ResetAccumulatorOnInterrupt::Execute (OperationContext &context,
                                      Transaction & /*tx*/) const
{
  const bool interrupt_allowed = context.GetInterruptAllowed ();
  if (!interrupt_allowed)
    {
      return;
    }

  const bool flush = context.GetFlushRequired ();
  const bool spike_bypass = context.GetSpikeBypass ();
  if (flush || spike_bypass)
    {
      return;
    }

  auto &p_ctx = context.GetProcessorContext ();
  const auto &signal = context.GetSignal ();
  auto it = p_ctx.accumulator_states.find (signal.source_id);
  if (it == p_ctx.accumulator_states.end ())
    {
      return;
    }

  if (it->second.n_signals == 0)
    {
      return;
    }

  const auto selected_id = context.GetSelectedCandidateId ();
  if (!selected_id.has_value ())
    {
      return;
    }
  const auto &candidates = context.GetRetrievedMemoryEmbeddings ();
  auto cand_it = candidates.find (*selected_id);
  if (cand_it == candidates.end () || cand_it->second.size () == 0)
    {
      return;
    }

  it->second.pending_interrupt_abort = true;
  it->second.pending_interrupt_embedding = cand_it->second;
  telemetry::AddCounter ("cortext.accumulator.interrupt_pending_total", 1);
  telemetry::LogDebug ("cortext.accumulator.interrupt_pending", {
    telemetry::Attribute::String ("source_id", signal.source_id),
    telemetry::Attribute::Double ("n_signals", it->second.n_signals)
  });
}

} // namespace cortext::operations
