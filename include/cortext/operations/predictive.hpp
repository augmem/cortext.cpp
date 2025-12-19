#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 22: Predictive Memory Pre-activation.
///
/// Uses the recent trajectory of context embeddings to predict the next query
/// direction and slightly boosts the strength of retrieved candidates that are
/// well-aligned with the predicted direction. The boost magnitude is small,
/// knob-derived, and optionally modulated by surprise.
class ApplyPredictivePreActivation : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
