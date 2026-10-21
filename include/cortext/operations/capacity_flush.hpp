#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Closes a unit that reached its signal capacity.
///
/// A unit holding max_signals_per_memory(F,S,T) signals whose boundary score
/// reaches the fallback floor flushes as a "capacity" boundary and finalizes
/// its episode. Ephemeral probes never close a unit.
///
/// Must run AFTER DetectBoundary (provides the boundary score).
class CheckCapacityFlush
    : public Operation<Requires<tags::AccumulatorWindowState, tags::BoundaryDecision>, Satisfies<tags::BoundaryDecision, tags::FlushRequired> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
