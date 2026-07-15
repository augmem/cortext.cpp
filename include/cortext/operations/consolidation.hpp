#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithm 28: Explicit consolidation execution gate.
///
/// Starts the consolidation pipeline only for a non-ephemeral signal whose
/// caller set `force_consolidation`. Throughput-derived maintenance urgency is
/// reported separately through `consolidation_state`; it never starts work.
class EvaluateConsolidation
    : public Operation<Requires<>, Satisfies<tags::ConsolidationShouldStart, tags::ConsolidationCandidates> >
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
class ScoreConsolidation
    : public Operation<Requires<>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
