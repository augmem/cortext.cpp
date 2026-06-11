#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Updates the smoothed uncertainty `u(t)`.
///
/// This operation implements the fallback uncertainty calculation from
/// `algorithms.md` Section 0.4, which is based on maturity. A more complex
/// implementation would also use structural metrics.
class UpdateUncertainty
    : public Operation<Requires<tags::MetricValues, tags::StructuralCoherence>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
