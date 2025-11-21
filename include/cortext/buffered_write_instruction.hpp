#pragma once

#include <any>
#include <string>
#include <vector>

namespace cortext
{

/// @brief Represents a deferred database write operation.
///
/// Operations in the pipeline produce these instructions, which are buffered
/// and executed in a single transaction at the end of an episode.
struct BufferedWriteInstruction
{
  std::string query;
  std::vector<std::any> params;
};

} // namespace cortext
