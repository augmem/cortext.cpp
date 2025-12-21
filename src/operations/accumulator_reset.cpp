#include "cortext/operations/accumulator_reset.hpp"

#include "cortext/processor/accumulator_state.hpp"
#include "cortext/processor/operation_context.hpp"

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

} // namespace cortext::operations
