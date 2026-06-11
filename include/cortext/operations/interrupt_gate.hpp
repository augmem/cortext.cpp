#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithm 27: Marginal Utility Novelty (MNI) Interrupt Gate.
///
/// Computes a stream-agnostic gate decision to allow retrieval interruption
/// based on MU, duplicate suppression, refractory scaling, and boundary
/// checks.
class ComputeMniGateDecision
    : public Operation<Requires<tags::Arousal, tags::BoundaryDecision, tags::Coherence, tags::EmotionIntensity, tags::MetricValues, tags::RetrievedMemoryEmbeddings, tags::WriteExclusionTs>, Satisfies<tags::InterruptAllowed, tags::SelectedCandidateId, tags::MniGateDiagnostics> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
