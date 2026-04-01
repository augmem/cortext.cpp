#pragma once

#include "cortext/core/utils.hpp"
#include <string>

namespace cortext
{

/// @brief Consolidation execution mode.
enum class ConsolidationMode
{
  Shallow = 0,
  Deep = 1,
  Both = 2
};

inline bool
IsConsolidationSignal (const std::string &source_id)
{
  return core::StartsWith (source_id, "cortext/consolidate");
}

inline ConsolidationMode
ParseConsolidationMode (const std::string &source_id)
{
  if (!IsConsolidationSignal (source_id))
    {
      return ConsolidationMode::Both;
    }
  if (source_id == "cortext/consolidate"
      || source_id == "cortext/consolidate/both")
    {
      return ConsolidationMode::Both;
    }
  if (source_id == "cortext/consolidate/shallow")
    {
      return ConsolidationMode::Shallow;
    }
  if (source_id == "cortext/consolidate/deep")
    {
      return ConsolidationMode::Deep;
    }
  return ConsolidationMode::Both;
}

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

inline std::string
ConsolidationSourceId (ConsolidationMode mode)
{
  switch (mode)
    {
    case ConsolidationMode::Shallow:
      return "cortext/consolidate/shallow";
    case ConsolidationMode::Deep:
      return "cortext/consolidate/deep";
    case ConsolidationMode::Both:
    default:
      return "cortext/consolidate";
    }
}

} // namespace cortext
