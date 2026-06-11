#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Section 4.4.4: Checks for flashbulb flush (spike bypass).
///
/// High-salience signals bypass accumulation and flush immediately:
///   spike_bypass = score_t > (θ_dynamic + spike_margin(S))
///
/// When spike_bypass triggers with n > 1, forces segment flush to
/// capture surrounding context as a coherent memory unit.
///
/// Must run AFTER:
/// - UpdateThreshold (provides θ_dynamic)
/// - ComputeCompositeScore (provides score_t)
class CheckSpikeBypass
    : public Operation<Requires<tags::AccumulatorWindowState, tags::BoundaryDecision, tags::CompositeScore, tags::ThresholdState>, Satisfies<tags::SpikeBypass, tags::BoundaryDecision, tags::FlushRequired> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
