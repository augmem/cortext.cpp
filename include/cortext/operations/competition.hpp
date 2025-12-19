/// @file
#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 21: Retrieval Competition & Interference.
///
/// Selects top-k "winner" memories by activation against the current context
/// embedding, then applies lateral inhibition to "loser" memories that are
/// within an inhibition radius of any winner. Suppression is accumulated in a
/// transient `rif_state` table and immediately reduces
/// `memory_feedback.strength`. A time-based recovery step restores strength
/// proportionally to elapsed time since last suppression (RIF recovery), then
/// decays the stored suppression.
class ApplyRetrievalCompetition : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
