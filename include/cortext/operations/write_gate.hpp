#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/**
 * @brief Section 4.4.5: Memory-level write gate with window scoring.
 *
 * Computes memory-level score:
 *   s_avg = s_sum / max(n, 1)
 *   coverage = min(n / n_ctx(T), 1.0)
 *   S_window = α(F) × s_max + (1-α(F)) × s_avg + β(S) × coverage
 *
 * Write refractory suppresses rapid successive writes:
 *   M_write_refrac = 1.0 + k × exp(-Δt_write / τ)
 *   θ_accumulator = θ_dynamic × M_write_refrac
 *
 * Final decision:
 *   write_accumulator = (flush OR spike_bypass) AND (S_window > θ_accumulator)
 *
 * On write: computes representative embedding and resets accumulator:
 *   e_rep = normalize(ρ(F) × μ_acc + (1-ρ(F)) × e_peak)
 *
 * Must run AFTER:
 * - DetectBoundary (provides flush decision)
 * - CheckSpikeBypass (provides spike_bypass)
 * - UpdateThreshold (provides θ_dynamic)
 */
class ComputeWriteGate : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
