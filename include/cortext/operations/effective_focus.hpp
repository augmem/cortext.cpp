#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Computes effective focus F_eff = F × (0.5 + 0.5 × coherence_t).
///
/// Falls back to coherence_t=1.0 if not set, yielding F_eff=F.
class ComputeEffectiveFocus
    : public Operation<Requires<tags::MetricValues, tags::StructuralCoherence>, Satisfies<tags::EffectiveFocus> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
