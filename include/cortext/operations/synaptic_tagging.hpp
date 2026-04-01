#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Tags nearby memories after high surprise/arousal events.
class ApplySynapticTagging : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
