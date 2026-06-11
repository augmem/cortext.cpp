#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Ingress short-term memory graph used for live provisional labels.
///
/// This operation records a bounded, per-source short-term graph whose recent
/// signal items act as nodes and whose clustered label matches act as
/// provisional edges. Consolidation treats these labels as candidates for
/// Gemma refinement, not as final durable graph edges.
class UpdateShortTermMemoryShadow
    : public Operation<Requires<tags::BoundaryDecision>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
