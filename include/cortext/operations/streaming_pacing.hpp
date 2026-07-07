#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithm 10.4b: Streaming Pacing Gate.
///
/// Gates retrieval operations based on accumulated drift:
/// - If drift_acc_pacing > pacing_thresh(S): should_check_retrieval = true
/// - If drift_acc_pacing > max_wait_drift(F): force check regardless
/// - Boundary flush also triggers retrieval checks
/// - Ephemeral signals are query-style ingress and always trigger retrieval
/// - Resets drift_acc_pacing and updates x_last_check on trigger
class CheckStreamingPacing
    : public Operation<Requires<tags::FlushRequired>, Satisfies<tags::ShouldCheckRetrieval> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
