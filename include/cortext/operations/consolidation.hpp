#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 28/28b: Consolidation Triggers & Scheduling.
///
/// Evaluates whether background consolidation should start based on
/// rate/interval triggers (Alg 28) and idle gating (Alg 28b). Emits an
/// event row into `consolidation_events` for start/defer actions.
class EvaluateConsolidation : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

/// @brief Algorithm 29: Consolidation Scoring.
///
/// Computes a consolidation score for each memory based on knob-derived
/// weights and marks low-score memories as candidates for merge.
///
/// score_consolidate(m) = T*strength(m) − F*redundancy(m)
///                        + S*connectivity(m) + T*stability(m)
///
/// Current implementation uses available columns in memory_feedback; when
/// redundancy/connectivity/stability are absent they are treated as 0.
class ScoreConsolidation : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

/// @brief Algorithm 29c: Entity and Relation Extraction (enqueue jobs).
///
/// Queues semantic extraction jobs for consolidated summaries based on knob-
/// derived gating and batching rules. This operation does not perform model
/// inference; it only inserts rows into `extraction_jobs` with a prompt built
/// from the summary text and its clustered source texts.
class EnqueueExtractionJobs : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
