#pragma once

#include "cortext/core/algorithms.hpp"
#include "../experimental_env.hpp"

namespace cortext::operations::neuromodulation
{

inline bool
EffectScalesDisabled ()
{
  return internal::experimental_env::Flag (
      "CORTEXT_DISABLE_NEUROMODULATOR_EFFECT_SCALES");
}

inline double
WriteThresholdScale (double neuromod_ne)
{
  if (EffectScalesDisabled ()
      || internal::experimental_env::Flag (
          "CORTEXT_DISABLE_NEUROMOD_WRITE_SCALE"))
    {
      return 1.0;
    }
  return core::Clamp (1.0 - 0.3 * core::Clamp (neuromod_ne, 0.0, 1.0), 0.7,
                      1.0);
}

inline double
ReconsolidationScale (double neuromod_ach)
{
  if (EffectScalesDisabled ()
      || internal::experimental_env::Flag (
          "CORTEXT_DISABLE_NEUROMOD_RECONSOLIDATION_SCALE"))
    {
      return 1.0;
    }
  return 1.0 + 0.4 * core::Clamp (neuromod_ach, 0.0, 1.0);
}

inline double
RetrievalCompetitionScale (double neuromod_ne)
{
  if (EffectScalesDisabled ()
      || internal::experimental_env::Flag (
          "CORTEXT_DISABLE_NEUROMOD_COMPETITION_SCALE"))
    {
      return 1.0;
    }
  return 1.0 + 0.5 * core::Clamp (neuromod_ne, 0.0, 1.0);
}

inline double
ValueUpdateGain (double neuromod_da)
{
  if (EffectScalesDisabled ()
      || internal::experimental_env::Flag (
          "CORTEXT_DISABLE_NEUROMOD_VALUE_GAIN"))
    {
      return 1.0;
    }
  return 0.5 + 0.5 * core::Clamp (neuromod_da, 0.0, 1.0);
}

} // namespace cortext::operations::neuromodulation
