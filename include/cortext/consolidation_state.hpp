#pragma once

namespace cortext
{

/// @brief Caller-facing urgency for an explicit consolidation pass.
enum class ConsolidationState
{
  None,
  Recommended,
  Required,
};

} // namespace cortext
