#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithm 7 (part 1): Fit metric weights via online RLS with
/// forgetting.
///
/// - Maintains weights, order, and covariance matrix in ProcessorContext.
/// - Uses φ(T) = 0.90 + 0.09T for forgetting.
/// - Observed target defaults to metric "relevance" normalized to [0,1].
class FitMetricWeightsRLS
    : public Operation<Requires<tags::MetricValues>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

/// @brief Algorithm 7 (part 2): Compute composite score from metrics and
/// weights.
///
/// - Bootstraps metric weights from knobs (F,S,T).
/// - Blends bootstrap and RLS weights using confidence schedule
/// τ_rls=lerp(20,80,T).
/// - Ensures non-negativity, normalizes weights to sum to 1, and sets
/// composite score in [0,1].
class ComputeCompositeScore
    : public Operation<Requires<tags::MetricValues>, Satisfies<tags::CompositeScore> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
