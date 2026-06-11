#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Implements Algorithm 24: Working Memory Gates & Maintenance.
///
/// Maintains a small set of working-memory slots in ProcessorContext by
/// decaying slot strengths over time and gating the current signal embedding
/// either by chunking into the best-matching slot or by inserting a new slot.
class WorkingMemory
    : public Operation<Requires<tags::AccumulatorWriteDecision, tags::RepresentativeEmbedding, tags::RetrievedMemoryEmbeddings, tags::WindowScore>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
