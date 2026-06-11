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

/// @brief Implements Algorithm 4b: Mood Integrator (Tonic State).
///
/// Maintains a persistent background mood state (M_t) distinct from
/// instantaneous emotion (e_t). The mood decays slowly and reacts to
/// emotion events, providing a threshold bias via ΔThreshold_mood_t.
class UpdateMood
    : public Operation<Requires<tags::EmotionProbabilities>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
