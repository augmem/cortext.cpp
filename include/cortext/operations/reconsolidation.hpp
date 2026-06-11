#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithm 20: Memory Reconsolidation During Retrieval.
///
/// For each retrieved memory embedding, compute a drift magnitude based on
/// knobs S,T and current lability, then blend the memory toward the current
/// context embedding. Persist lability state/timestamp and store the blended
/// embedding. Increases uncertainty proportional to the maximum drift
/// observed.
class ApplyReconsolidation
    : public Operation<Requires<tags::RetrievedMemoryEmbeddings>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
