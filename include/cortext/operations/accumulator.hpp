#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/**
 * @brief Section 4.4.1: Updates memory accumulator state for a signal.
 *
 * This operation:
 * 1. Gets or creates AccumulatorState for the signal's exact source stream
 * 2. Updates running mean: μ_acc = ((n-1) × μ_acc + x_t) / n
 * 3. Accumulates drift: D_acc += drift_mag_t
 * 4. Tracks signal metadata for later score aggregation
 * 5. Updates last_signal_ts for gap detection
 *
 * Must run AFTER:
 * - UpdateEmbeddingPredictionError (provides drift_mag in context)
 */
class UpdateAccumulator : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
