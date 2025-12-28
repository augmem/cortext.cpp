#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Computes neuromodulator-like control signals and oscillatory gating.
class UpdateNeuromodulators : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
