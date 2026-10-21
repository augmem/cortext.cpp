#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Implements Algorithm 3: Sensitivity Priors.
class InitializeSensitivityPriors
    : public Operation<Requires<>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

/// @brief Implements Algorithm 4: Sensitivity Dynamic Update per Signal.
class UpdateSensitivity
    : public Operation<Requires<>, Satisfies<tags::Arousal, tags::Valence, tags::EmotionIntensity, tags::EmotionProbabilities, tags::SensitivityThresholdDeltas> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
