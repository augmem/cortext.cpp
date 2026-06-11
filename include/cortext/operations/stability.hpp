#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Implements Algorithm 5: Stability Priors.
class InitializeStabilityPriors
    : public Operation<Requires<>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

/// @brief Implements Algorithm 6: Stability dynamic update.
class UpdateStability
    : public Operation<Requires<tags::DeltaHalfLifeAdjustment>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
