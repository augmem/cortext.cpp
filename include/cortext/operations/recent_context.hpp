#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Updates the recent context buffer with the current signal embedding.
///
/// This aligns with Appendix D step 2 (update buffers) by appending x_t and
/// trimming to n_ctx(T) + k_ctx(T).
class UpdateRecentContext : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
