#include "deep_llm_factory.hpp"

#include "cortext/extractor/gemma_extractor.hpp"
#include "cortext/summarizer/gemma_summarizer.hpp"

#include <stdexcept>
#include <utility>

namespace cortext::internal
{

std::optional<std::filesystem::path>
ResolveGemmaDeepLlmModelPath (const std::filesystem::path &models_dir)
{
  const auto candidate
      = models_dir / "gemma4-e2b-litert" / "gemma-4-E2B-it.litertlm";
  if (std::filesystem::exists (candidate))
    {
      return candidate;
    }
  return std::nullopt;
}

std::optional<DeepLlmSelection>
TryCreateDeepLlmSelection (const std::filesystem::path &models_dir,
                           std::string *error_out)
{
#if defined(CORTEXT_DISABLE_LITERT)
  if (error_out != nullptr)
    {
      *error_out = "Gemma backend unavailable: LiteRT-LM disabled at build "
                   "time; inject a Summarizer/Extractor instead";
    }
  (void)models_dir;
  return std::nullopt;
#else
  auto gemma_path = ResolveGemmaDeepLlmModelPath (models_dir);
  if (!gemma_path)
    {
      if (error_out != nullptr)
        {
          *error_out
              = "Gemma backend unavailable: no gemma4-e2b-litert model found";
        }
      return std::nullopt;
    }

  auto extractor = std::make_unique<GemmaExtractor> (gemma_path->string ());
  auto summarizer = std::make_unique<GemmaSummarizer> (gemma_path->string ());
  if (!extractor->IsAvailable () || !summarizer->IsAvailable ())
    {
      if (error_out != nullptr)
        {
          *error_out
              = "Gemma backend unavailable: model failed to initialize at "
                + gemma_path->string ();
        }
      return std::nullopt;
    }

  DeepLlmSelection selection;
  selection.backend_name = "Gemma/LiteRT-LM";
  selection.extractor_model_path = *gemma_path;
  selection.summarizer_model_path = *gemma_path;
  selection.extractor = std::move (extractor);
  selection.summarizer = std::move (summarizer);
  return selection;
#endif
}

DeepLlmSelection
CreateDeepLlmSelection (const std::filesystem::path &models_dir)
{
  std::string error;
  auto selection = TryCreateDeepLlmSelection (models_dir, &error);
  if (!selection)
    {
      throw std::runtime_error ("Deep consolidation stack unavailable: "
                                + error);
    }
  return std::move (*selection);
}

} // namespace cortext::internal
