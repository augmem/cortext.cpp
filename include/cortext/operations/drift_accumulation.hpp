#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithm 10.4a: Drift Accumulation Tracking.
///
/// Computes: drift_accum += cosine_dist(μ_acc, prev_x)
/// Runs after ComputeCoherence to leverage context state.
class UpdateDriftAccumulation
    : public Operation<Requires<>, Satisfies<tags::DriftAccumSnapshot> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
