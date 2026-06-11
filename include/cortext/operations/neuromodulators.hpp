#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Computes neuromodulator-like control signals and oscillatory gating.
class UpdateNeuromodulators
    : public Operation<Requires<tags::MetricValues, tags::Arousal>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
