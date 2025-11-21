#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 16: Sensitivity Feedback Adjustment.
///
/// Applies per-memory feedback to adjust the novelty weight based on
/// novelty reward, contextual gain, and redundancy (fallback 0).
class ApplySensitivityFeedback : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
