#pragma once

#include "cortext/extractor/extractor.hpp"
#include "cortext/summarizer/summarizer.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace cortext::internal
{

enum class DeepLlmBackend
{
  Auto,
  Gemma,
  Lfm2,
  Mixed
};

struct DeepLlmSelection
{
  std::unique_ptr<Extractor> extractor;
  std::unique_ptr<Summarizer> summarizer;
  std::string backend_name;
  std::filesystem::path extractor_model_path;
  std::filesystem::path summarizer_model_path;
};

DeepLlmBackend ResolveDeepLlmBackendOverride ();
std::string DescribeDeepLlmBackend (DeepLlmBackend backend);

std::optional<std::filesystem::path>
ResolveGemmaDeepLlmModelPath (const std::filesystem::path &models_dir);

std::optional<std::filesystem::path>
ResolveLfm2SummarizerModelPath (const std::filesystem::path &models_dir);

std::optional<std::filesystem::path>
ResolveLfm2ExtractorModelPath (const std::filesystem::path &models_dir);

std::optional<DeepLlmSelection>
TryCreateDeepLlmSelection (const std::filesystem::path &models_dir,
                           DeepLlmBackend backend, std::string *error_out);

DeepLlmSelection
CreateDeepLlmSelection (const std::filesystem::path &models_dir);

} // namespace cortext::internal
