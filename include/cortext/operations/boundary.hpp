#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Implements Algorithm 12: Trajectory Drift / Episode Boundary.
///
/// Compares drift magnitude between two recent windows against a knob-derived
/// threshold and requests episode finalization when exceeded.
class CheckEpisodeBoundary : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
