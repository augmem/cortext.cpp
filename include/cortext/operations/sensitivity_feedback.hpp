#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithm 16: Sensitivity Feedback Adjustment.
///
/// Applies per-memory feedback to adjust the novelty weight based on
/// novelty reward, contextual gain, and redundancy (fallback 0).
class ApplySensitivityFeedback
    : public Operation<Requires<tags::MemoryUsageEvents, tags::RetrievedMemoryEmbeddings>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
