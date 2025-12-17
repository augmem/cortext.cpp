/// @file
#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Initialize baked centroid vectors into ProcessorContext.
///
/// Populates `ProcessorContext::centroids` once so downstream operations can
/// compute affect and emotion projections without runtime IO.
class InitializeEmbeddedCentroids : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations


