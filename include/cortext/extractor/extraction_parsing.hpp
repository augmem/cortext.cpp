#pragma once

#include "cortext/operations/extraction.hpp"

#include <string>

namespace cortext
{

/// @brief Parse a model's extraction response (JSON or loose braces) into an
/// ExtractionResult. Defined in gemma_extractor.cpp; shared by the provider
/// adapters so every transport parses model output identically.
operations::ExtractionResult
ParseExtractionResponse (const std::string &content);

} // namespace cortext
