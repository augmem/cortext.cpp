#pragma once

#include "cortext/processor/operation.hpp"

#include <memory>

namespace cortext
{

/// @brief Build the private production operation graph.
std::unique_ptr<IOperation> BuildRootOperationSet (bool probe_mode);

} // namespace cortext
