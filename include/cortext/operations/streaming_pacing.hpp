#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 10.4b: Streaming Pacing Gate.
///
/// Gates retrieval operations based on accumulated drift:
/// - If drift_accum > pacing_thresh(S): should_check_retrieval = true
/// - If drift_accum > max_wait_drift(F): force check regardless
/// - Resets drift_accum and updates last_pacing_check_embedding on trigger
class CheckStreamingPacing : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
