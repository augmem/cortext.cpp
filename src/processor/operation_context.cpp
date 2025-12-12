#include "cortext/processor/operation_context.hpp"

namespace cortext
{

OperationContext::OperationContext (
    const Signal &signal, ProcessorContext &context,
    const SignalProcessor::Config &config,
    std::vector<BufferedWriteInstruction> &write_buffer)
    : OperationContext (signal, context, config, write_buffer, nullptr)
{
}

OperationContext::OperationContext (
    const Signal &signal, ProcessorContext &context,
    const SignalProcessor::Config &config,
    std::vector<BufferedWriteInstruction> &write_buffer, Store *store)
    : signal_ (signal), context_ (context), config_ (config),
      write_buffer_ (write_buffer), store_ (store)
{
}

void
OperationContext::AddWriteInstruction (BufferedWriteInstruction op)
{
  write_buffer_.push_back (std::move (op));
}

void
OperationContext::SetMemoryUsageEvents (
    std::vector<OperationContext::MemoryUsageEvent> events)
{
  memory_usage_events_ = std::move (events);
}

const std::vector<OperationContext::MemoryUsageEvent> &
OperationContext::GetMemoryUsageEvents () const
{
  return memory_usage_events_;
}

// Note: Getters/Setters for new fields are inline in the header.

} // namespace cortext
