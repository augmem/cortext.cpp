#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/**
 * @brief Section 4.4.3: Detects natural memory boundaries.
 *
 * Boundary score combines drift spike and coherence drop:
 *   drift_spike = (d_step - η_acc) / max(η_acc, ε)
 *   coh_drop = max(0, coherence_prev - coherence_t)
 *   boundary_score = w_drift × sigmoid(drift_spike) + w_coh × coh_drop
 *
 * Flush triggers (any one):
 * - boundary_score > b_thresh(F, S)
 * - memory_elapsed > max_mem_time(T)
 * - D_acc > max_mem_drift(S)
 * - signal_gap > gap_threshold(T)
 *
 * Must run AFTER:
 * - ComputeCoherence (provides d_step, coherence)
 */
class DetectBoundary : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
