#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 7+8 gate: Decides whether a signal should be stored.
///
/// Write decision formula: composite_score > (T_dynamic - hysteresis)
///
/// This operation must run AFTER:
/// - ComputeCompositeScore (provides composite_score)
/// - UpdateThreshold (provides T_dynamic and hysteresis)
///
/// Sets the write_decision flag in OperationContext, which is then:
/// - Exposed in SignalProcessor::Output::write_decision
/// - Recorded in signal_metrics.write_decision for observability
class ComputeWriteGate : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
