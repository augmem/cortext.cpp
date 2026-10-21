#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/**
 * @brief Section 4.4.5: Memory-level write commit with window scoring.
 *
 * Every unit that closes (boundary or capacity flush, Durable turn) is
 * stored. The window score feeds working-memory admission:
 *   s_avg = s_sum / max(n, 1)
 *   coverage = min(n / n_ctx(T), 1.0)
 *   S_window = α(F) × s_max + (1-α(F)) × s_avg + β(S) × coverage
 *
 * Decision:
 *   write_accumulator = flush AND open unit AND NOT Ephemeral
 *
 * On write: computes the representative embedding:
 *   e_rep = normalize(ρ(F) × μ_acc + (1-ρ(F)) × e_peak)
 *
 * Must run AFTER:
 * - DetectBoundary and CheckCapacityFlush (provide the flush decision)
 */
class ComputeWriteGate
    : public Operation<Requires<tags::FlushRequired>, Satisfies<tags::AccumulatorWriteDecision, tags::RepresentativeEmbedding, tags::WindowScore, tags::WriteDecision> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
