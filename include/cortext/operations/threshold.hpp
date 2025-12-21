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
  void Execute (OperationContext &context, Transaction &tx) const override;
};

/// @brief Updates rate-control state after the write decision.
///
/// Uses the current write decision to update rate EWMA, bias-corrected
/// estimate, and timestamps per Section 6 (post-write rate update).
class UpdateRateState : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
