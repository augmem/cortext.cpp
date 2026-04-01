#include "cortext/processor/operation_context.hpp"

namespace cortext
{

OperationContext::OperationContext (const Signal &signal,
                                    ProcessorContext &context,
                                    const SignalProcessor::Config &config)
    : OperationContext (signal, context, config, nullptr)
{
}

OperationContext::OperationContext (const Signal &signal,
                                    ProcessorContext &context,
                                    const SignalProcessor::Config &config,
                                    Store *store)
    : signal_ (signal), context_ (context), config_ (config), store_ (store)
{
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

void
OperationContext::AddOperationTiming (std::string_view op_type, double ms)
{
  if (op_type.empty ())
    {
      return;
    }
  const std::string key (op_type);
  operation_timings_ms_[key] += ms;
}

// Note: Getters/Setters for new fields are inline in the header.

} // namespace cortext
