#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 23: Emotional Consolidation Tags.
///
/// When emotion intensity and arousal exceed knob-derived thresholds, persist
/// emotional tags for memories used in the current signal, including
/// flashbulb markers and consolidation parameters.
class ApplyEmotionalConsolidation : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
  void CollectSchema (cortext::store::SchemaRegistry &registry) const override;
};

} // namespace cortext::operations
