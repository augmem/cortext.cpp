#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 27: Marginal Utility Novelty (MNI) Interrupt Gate.
///
/// Computes a stream-agnostic gate decision to allow retrieval interruption
/// based on MU, duplicate suppression, refractory scaling, and boundary
/// checks.
class ComputeMniGateDecision : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
