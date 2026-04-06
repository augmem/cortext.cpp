#pragma once

#include "cortext/core/algorithms.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace cortext::operations::neuromodulation
{

inline bool
EnvFlag (const char *name)
{
  const char *value = std::getenv (name);
  if (!value)
    {
      return false;
    }
  std::string s (value);
  std::transform (s.begin (), s.end (), s.begin (),
                  [] (unsigned char c) {
                    return static_cast<char> (std::tolower (c));
                  });
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

inline double
WriteThresholdScale (double neuromod_ne)
{
  if (EnvFlag ("CORTEXT_DISABLE_NEUROMOD_WRITE_SCALE"))
    {
      return 1.0;
    }
  return core::Clamp (1.0 - 0.3 * core::Clamp (neuromod_ne, 0.0, 1.0), 0.7,
                      1.0);
}

inline double
ReconsolidationScale (double neuromod_ach)
{
  if (EnvFlag ("CORTEXT_DISABLE_NEUROMOD_RECONSOLIDATION_SCALE"))
    {
      return 1.0;
    }
  return 1.0 + 0.4 * core::Clamp (neuromod_ach, 0.0, 1.0);
}

inline double
RetrievalCompetitionScale (double neuromod_ne)
{
  if (EnvFlag ("CORTEXT_DISABLE_NEUROMOD_COMPETITION_SCALE"))
    {
      return 1.0;
    }
  return 1.0 + 0.5 * core::Clamp (neuromod_ne, 0.0, 1.0);
}

inline double
ValueUpdateGain (double neuromod_da)
{
  if (EnvFlag ("CORTEXT_DISABLE_NEUROMOD_VALUE_GAIN"))
    {
      return 1.0;
    }
  return 0.5 + 0.5 * core::Clamp (neuromod_da, 0.0, 1.0);
}

} // namespace cortext::operations::neuromodulation
