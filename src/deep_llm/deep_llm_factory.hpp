#pragma once

#include "cortext/extractor/extractor.hpp"
#include "cortext/summarizer/summarizer.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace cortext::internal
{

struct DeepLlmSelection
{
  std::unique_ptr<Extractor> extractor;
  std::unique_ptr<Summarizer> summarizer;
  std::string backend_name;
  std::filesystem::path extractor_model_path;
  std::filesystem::path summarizer_model_path;
};

std::optional<std::filesystem::path>
ResolveGemmaDeepLlmModelPath (const std::filesystem::path &models_dir);

std::optional<DeepLlmSelection>
TryCreateDeepLlmSelection (const std::filesystem::path &models_dir,
                           std::string *error_out);

/// Create the deep-consolidation stack: Gemma-4-E2B via LiteRT-LM for both
/// summarization and extraction. This is the one engine-resolved stack;
/// alternatives are supplied through the Cortext::Create inference-injection
/// seam (Summarizer/Extractor providers), not by backend configuration.
DeepLlmSelection
CreateDeepLlmSelection (const std::filesystem::path &models_dir);

} // namespace cortext::internal
