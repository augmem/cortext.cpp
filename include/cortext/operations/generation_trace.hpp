#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Records output embeddings for influence feedback (Algorithm 19).
///
/// Persists the current embedding to the generation_trace table along with
/// semantic drift measurement for tracking output trajectory.
class RecordGenerationTrace : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
