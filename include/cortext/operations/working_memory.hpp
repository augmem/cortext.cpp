#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Implements Algorithm 24: Working Memory Gates & Maintenance.
///
/// Maintains a small set of working-memory slots in ProcessorContext by
/// decaying slot strengths over time and gating the current signal embedding
/// either by chunking into the best-matching slot or by inserting a new slot.
class WorkingMemory : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
