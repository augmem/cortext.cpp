#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Updates the smoothed uncertainty `u(t)`.
///
/// This operation implements the fallback uncertainty calculation from
/// `algorithms.md` Section 0.4, which is based on maturity. A more complex
/// implementation would also use structural metrics.
class UpdateUncertainty : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
