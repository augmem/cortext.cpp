#pragma once

namespace cortext
{

/// @brief Consolidation execution mode.
enum class ConsolidationMode
{
  Shallow = 0,
  Deep = 1,
  Both = 2
};

inline const char *
ConsolidationModeLabel (ConsolidationMode mode)
{
  switch (mode)
    {
    case ConsolidationMode::Shallow:
      return "shallow";
    case ConsolidationMode::Deep:
      return "deep";
    case ConsolidationMode::Both:
    default:
      return "both";
    }
}

} // namespace cortext
