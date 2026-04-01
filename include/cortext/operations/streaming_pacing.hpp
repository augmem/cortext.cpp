#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 10.4b: Streaming Pacing Gate.
///
/// Gates retrieval operations based on accumulated drift:
/// - If drift_acc_pacing > pacing_thresh(S): should_check_retrieval = true
/// - If drift_acc_pacing > max_wait_drift(F): force check regardless
/// - Boundary flush also triggers retrieval checks
/// - Resets drift_acc_pacing and updates x_last_check on trigger
class CheckStreamingPacing : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
