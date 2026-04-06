#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor.hpp"
#include "cortext/store/store.hpp"

namespace cortext
{
class OperationContext;
}

namespace cortext::operations::meta_learning
{

bool Disabled ();

double ResolveAttentionWidthPrior (Transaction &tx,
                                   const SignalProcessor::Config &config);

double ResolveRateTargetPrior (Transaction &tx,
                               const SignalProcessor::Config &config);

double ResolveHysteresisBandPrior (Transaction &tx,
                                   const SignalProcessor::Config &config);

void LearnFromOutcome (OperationContext &context, Transaction &tx);

} // namespace cortext::operations::meta_learning

namespace cortext::operations
{

class ApplyMetaLearning : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
