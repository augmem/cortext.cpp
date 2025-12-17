#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 29d: Process extraction results from external LLM.
///
/// Processes `pending_extraction_results` from ProcessorContext and persists
/// extracted entities and relations to the database for knowledge graph
/// construction.
///
/// For each extraction result:
/// - Inserts entities into `extraction_entities` table
/// - Inserts relations into `extraction_relations` table
/// - Updates `entity_index` for name→node_id mapping
///
/// Input:
/// - ProcessorContext::pending_extraction_results (populated by external
/// callback)
///
/// Output:
/// - extraction_entities table populated
/// - extraction_relations table populated
/// - entity_index table updated
/// - pending_extraction_results cleared
class ProcessExtractionResults : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
