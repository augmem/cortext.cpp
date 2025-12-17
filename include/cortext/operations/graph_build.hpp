/// @file
#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 30: Graph Construction (explicit tables).
///
/// Builds graph nodes/edges from consolidation summaries and extraction outputs.
/// This operation assumes:
/// - `consolidation_summaries` is populated with `summary_id` and `summary_text`
/// - `extraction_entities` and `extraction_relations` are populated by an
///   external worker processing `extraction_jobs`
///
/// Node id conventions:
/// - summary:  "summary:<summary_id>"
/// - entity:   "entity:<name>"
/// - memory:   "emb:<embedding_id>"
class BuildGraphFromConsolidation : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations

