#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithm 23: Emotional Consolidation Tags.
///
/// When emotion intensity and arousal exceed knob-derived thresholds, persist
/// emotional tags for memories used in the current signal, including
/// flashbulb markers and consolidation parameters.
class ApplyEmotionalConsolidation
    : public Operation<Requires<tags::StoredEmbeddingId>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
