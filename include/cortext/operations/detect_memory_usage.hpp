#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Detects implicit memory usage via semantic similarity.
///
/// Implements human-like implicit feedback: when previously-retrieved memories
/// appear semantically similar to the current input, they are marked as "used".
///
/// Detection mechanism:
/// 1. Compare current signal embedding against cached recently-retrieved
/// memories
/// 2. If similarity >= MemoryUsageThreshold(F), mark as "used"
/// 3. Populate MemoryUsageEvents with (embedding_id, used, contextual_gain)
///
/// This enables the closed-loop feedback system (Algorithms 14-19) to function
/// without explicit user feedback - usage is inferred from semantic patterns.
///
/// Pipeline order: Must run AFTER initialization priors, BEFORE feedback ops.
///
/// Knob effects:
/// - Focus (F): Higher F → stricter similarity threshold → fewer "used"
///   detections
/// - Stability (T): Higher T → longer cache duration → memories tracked longer
class DetectMemoryUsage : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations
