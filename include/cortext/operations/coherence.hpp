#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Implements Algorithm 10: Coherence / Integration.
///
/// coherence_t = 1 − clamp(var(cos(x_t, context_window)), 0, 1)
class ComputeCoherence : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
