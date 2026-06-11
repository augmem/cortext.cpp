#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/**
 * @brief Updates accumulator score aggregates after composite scoring.
 *
 * Uses the accumulator-level composite score (derived from μ_acc) to
 * update s_sum, s_max, and e_peak, and patches the latest signal record.
 * This keeps decisions grounded on the current unflushed signal group.
 */
class UpdateAccumulatorScores
    : public Operation<Requires<tags::CompositeScore>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
