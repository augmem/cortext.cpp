/// @file
#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Records retrieval usage counters for every retrieved memory.
///
/// For each memory usage event, increments `retrieved_count`, increments
/// `used_count` when the memory was used, stamps `last_used` (used only) and
/// `last_access`, and mirrors the counters into the processor retrieval
/// surface. These counters are exposed through hydrated memories.
class RecordMemoryUsage
    : public Operation<Requires<tags::MemoryUsageEvents>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
