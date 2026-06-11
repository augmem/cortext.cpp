#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Reset accumulator state after a flush/spike decision.
class ResetAccumulatorAfterFlush
    : public Operation<Requires<tags::FlushRequired, tags::SpikeBypass>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

/// @brief Reset accumulator state after an allowed interrupt to avoid
///        persisting partial thoughts.
class ResetAccumulatorOnInterrupt
    : public Operation<Requires<tags::InterruptAllowed, tags::RetrievedMemoryEmbeddings, tags::SelectedCandidateId>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
