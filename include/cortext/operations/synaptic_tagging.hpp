#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Tags nearby memories after high surprise/arousal events.
class ApplySynapticTagging
    : public Operation<Requires<tags::Arousal, tags::MetricValues>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
