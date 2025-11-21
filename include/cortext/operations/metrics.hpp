#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Enumeration of all metric types used in the cortext system
enum class Metric
{
  relevance,
  mismatch,
  surprise,
  rarity,
  drift,
  contradiction,
  utility,
  periphery,
  coverage,
  salience,
  valence,
  arousal,
  focus_spread,
  drift_mag,
  // Test/debugging metrics
  aw_prev,
  rate_prev,
  hys_prev
};

/// @brief Compute all 12 composite metrics (Algorithm 7 inputs) and set them
/// on the OperationContext as [-1,1] values.
///
/// Metrics:
/// - relevance, mismatch, surprise, rarity, drift, contradiction,
///   utility, periphery, coverage, salience, valence, arousal
class ComputeMetrics : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
