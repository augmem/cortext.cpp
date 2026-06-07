#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Ingress-time Soft Anchor state update.
///
/// This operation runs after MemoryStorage and before retrieval. It consumes
/// only the current stored memory signal plus prior Soft Anchor state; it does
/// not read retrieved candidates and does not alter retrieval ranking.
class UpdateSoftAnchor : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
