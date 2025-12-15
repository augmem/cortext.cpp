#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Persists signal metrics to the signal_metrics table for observability.
///
/// Records all computed metrics from Algorithm 7 composite scoring along with
/// the composite score, threshold, and write decision for each processed signal.
class PersistSignalMetrics : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
