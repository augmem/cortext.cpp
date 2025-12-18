#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/**
 * @brief Section 4.4.2: Computes drift and coherence for memory boundary detection.
 *
 * This operation:
 * 1. Computes d_step = max(drift_mag - ε_noise, 0)
 * 2. Updates η_acc via EWMA with AlphaEtaAcc(T)
 * 3. Computes coherence_t = cosine similarity to accumulator mean
 * 4. Stores d_step and coherence for boundary detection
 *
 * Must run AFTER:
 * - UpdateAccumulator (accumulator state exists)
 */
class ComputeCoherence : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
