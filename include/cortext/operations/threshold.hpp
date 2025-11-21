#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Implements Algorithm 8: Adaptive Threshold Evolution.
///
/// Updates the dynamic write threshold `T_dynamic` and hysteresis band using a
/// Bayesian blend of priors and observed scores, with schedules derived from
/// the Stability knob and smoothed uncertainty u(t).
class UpdateThreshold : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
