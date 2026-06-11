#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithm 29d: Process extraction results from external LLM.
///
/// Processes `pending_extraction_results` from ProcessorContext and persists
/// extracted labels and relations to the database for knowledge graph
/// construction.
///
/// For each extraction result:
/// - Inserts labels into `MEMORIES` (kind='LABEL')
/// - Inserts relations into `associations`
/// - Updates label salience (s_max) and links labels to summaries via has_label
///
/// Input:
/// - ProcessorContext::pending_extraction_results (populated by external
/// callback)
///
/// Output:
/// - label memories populated
/// - associations populated
/// - pending_extraction_results cleared
class ProcessExtractionResults
    : public Operation<Requires<tags::ExtractionRequests>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
