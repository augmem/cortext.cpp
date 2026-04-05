#include "deep_llm_factory.hpp"

#include "cortext/extractor/gemma_extractor.hpp"
#include "cortext/summarizer/gemma_summarizer.hpp"
#include "lfm2_llama_backend.hpp"
#include "llama_cpp_support.hpp"

#include <cstdlib>
#include <sstream>
#include <utility>
#include <vector>

namespace cortext::internal
{

namespace
{

std::string
GetEnvOrDefault (const char *name)
{
  const char *value = std::getenv (name);
  if (value == nullptr || *value == '\0')
    {
      return {};
    }
  return value;
}

void
AppendUniquePath (std::vector<std::filesystem::path> &paths,
                  const std::filesystem::path &path)
{
  if (path.empty ())
    {
      return;
    }
  for (const auto &existing : paths)
    {
      if (existing == path)
        {
          return;
        }
    }
  paths.push_back (path);
}

std::vector<std::filesystem::path>
DeepLlmSearchRoots (const std::filesystem::path &models_dir)
{
  std::vector<std::filesystem::path> roots;
  AppendUniquePath (roots, models_dir);
  if (models_dir.filename () == "imagebind")
    {
      AppendUniquePath (roots, models_dir.parent_path ());
    }
  return roots;
}

std::optional<std::filesystem::path>
FindFirstExistingModel (const std::vector<std::filesystem::path> &roots,
                        const std::vector<std::filesystem::path> &dirs,
                        const std::vector<std::filesystem::path> &filenames)
{
  for (const auto &root : roots)
    {
      for (const auto &dir : dirs)
        {
          const auto base = dir.empty () ? root : root / dir;
          for (const auto &filename : filenames)
            {
              const auto candidate = base / filename;
              if (std::filesystem::exists (candidate))
                {
                  return candidate;
                }
            }
        }
    }
  return std::nullopt;
}

std::string
DescribeLiquidTextModel (const std::filesystem::path &model_path)
{
  const std::string path = model_path.string ();
  if (path.find ("LFM2.5") != std::string::npos)
    {
      return "LFM2.5";
    }
  return "LFM2";
}

std::optional<DeepLlmSelection>
TryCreateGemmaSelection (const std::filesystem::path &models_dir,
                         std::string *error_out)
{
#if defined(CORTEXT_DISABLE_LITERT)
  if (error_out != nullptr)
    {
      *error_out = "Gemma backend unavailable: LiteRT-LM disabled at build time";
    }
  (void)models_dir;
  return std::nullopt;
#else
  auto gemma_path = ResolveGemmaDeepLlmModelPath (models_dir);
  if (!gemma_path)
    {
      if (error_out != nullptr)
        {
          *error_out = "Gemma backend unavailable: no gemma3n-e2b-litert model found";
        }
      return std::nullopt;
    }

  auto extractor = std::make_unique<GemmaExtractor> (gemma_path->string ());
  auto summarizer = std::make_unique<GemmaSummarizer> (gemma_path->string ());
  if (!extractor->IsAvailable () || !summarizer->IsAvailable ())
    {
      if (error_out != nullptr)
        {
          *error_out = "Gemma backend unavailable: model failed to initialize at "
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

std::optional<DeepLlmSelection>
TryCreateLfm2Selection (const std::filesystem::path &models_dir,
                        std::string *error_out)
{
  if (!kLlamaCppAvailable)
    {
      if (error_out != nullptr)
        {
          *error_out
              = "Liquid GGUF backend unavailable: llama.cpp library not found at build time";
        }
      return std::nullopt;
    }

  auto summarizer_path = ResolveLfm2SummarizerModelPath (models_dir);
  auto extractor_path = ResolveLfm2ExtractorModelPath (models_dir);
  if (!summarizer_path || !extractor_path)
    {
      if (error_out != nullptr)
        {
          std::ostringstream error;
          error << "Liquid GGUF backend unavailable: preferred model files not found";
          if (!summarizer_path)
            {
              error << " (missing summarizer)";
            }
          if (!extractor_path)
            {
              error << " (missing extractor)";
            }
          *error_out = error.str ();
        }
      return std::nullopt;
    }

  auto summarizer
      = std::make_unique<Lfm2LlamaSummarizer> (summarizer_path->string ());
  auto extractor
      = std::make_unique<Lfm2LlamaExtractor> (extractor_path->string ());
  if (!summarizer->IsAvailable () || !extractor->IsAvailable ())
    {
      if (error_out != nullptr)
        {
          *error_out
              = "Liquid GGUF backend unavailable: model initialization failed";
        }
      return std::nullopt;
    }

  DeepLlmSelection selection;
  const std::string backend_family
      = DescribeLiquidTextModel (*summarizer_path);
  selection.backend_name = backend_family + "/llama.cpp";
  selection.extractor_model_path = *extractor_path;
  selection.summarizer_model_path = *summarizer_path;
  selection.extractor = std::move (extractor);
  selection.summarizer = std::move (summarizer);
  return selection;
}

std::optional<DeepLlmSelection>
TryCreateMixedSelection (const std::filesystem::path &models_dir,
                         std::string *error_out)
{
#if defined(CORTEXT_DISABLE_LITERT)
  if (error_out != nullptr)
    {
      *error_out = "Mixed backend unavailable: LiteRT-LM disabled at build time";
    }
  (void)models_dir;
  return std::nullopt;
#else
  if (!kLlamaCppAvailable)
    {
      if (error_out != nullptr)
        {
          *error_out
              = "Mixed backend unavailable: llama.cpp library not found at build time";
        }
      return std::nullopt;
    }

  auto gemma_path = ResolveGemmaDeepLlmModelPath (models_dir);
  if (!gemma_path)
    {
      if (error_out != nullptr)
        {
          *error_out = "Mixed backend unavailable: no gemma3n-e2b-litert model found";
        }
      return std::nullopt;
    }

  auto extractor_path = ResolveLfm2ExtractorModelPath (models_dir);
  if (!extractor_path)
    {
      if (error_out != nullptr)
        {
          *error_out = "Mixed backend unavailable: no preferred Liquid extractor model found";
        }
      return std::nullopt;
    }

  auto summarizer = std::make_unique<GemmaSummarizer> (gemma_path->string ());
  auto extractor = std::make_unique<Lfm2LlamaExtractor> (
      extractor_path->string ());
  if (!summarizer->IsAvailable () || !extractor->IsAvailable ())
    {
      if (error_out != nullptr)
        {
          *error_out = "Mixed backend unavailable: model initialization failed";
        }
      return std::nullopt;
    }

  DeepLlmSelection selection;
  selection.backend_name = "Gemma+" + DescribeLiquidTextModel (*extractor_path);
  selection.extractor_model_path = *extractor_path;
  selection.summarizer_model_path = *gemma_path;
  selection.extractor = std::move (extractor);
  selection.summarizer = std::move (summarizer);
  return selection;
#endif
}

} // namespace

DeepLlmBackend
ResolveDeepLlmBackendOverride ()
{
  const std::string override = GetEnvOrDefault ("CORTEXT_DEEP_LLM_BACKEND");
  if (override.empty () || override == "auto")
    {
      return DeepLlmBackend::Auto;
    }
  if (override == "gemma")
    {
      return DeepLlmBackend::Gemma;
    }
  if (override == "lfm2")
    {
      return DeepLlmBackend::Lfm2;
    }
  if (override == "mixed" || override == "hybrid")
    {
      return DeepLlmBackend::Mixed;
    }
  throw std::runtime_error ("Unsupported CORTEXT_DEEP_LLM_BACKEND value: "
                            + override);
}

std::string
DescribeDeepLlmBackend (DeepLlmBackend backend)
{
  switch (backend)
    {
    case DeepLlmBackend::Auto:
      return "auto";
    case DeepLlmBackend::Gemma:
      return "gemma";
    case DeepLlmBackend::Lfm2:
      return "lfm2";
    case DeepLlmBackend::Mixed:
      return "mixed";
    }
  return "unknown";
}

std::optional<std::filesystem::path>
ResolveGemmaDeepLlmModelPath (const std::filesystem::path &models_dir)
{
  const auto roots = DeepLlmSearchRoots (models_dir);
  for (const auto &root : roots)
    {
      const std::filesystem::path gemma_root = root / "gemma3n-e2b-litert";
      if (!std::filesystem::is_directory (gemma_root))
        {
          continue;
        }
      for (const auto &candidate :
           { std::filesystem::path ("gemma-3n-E2B-it-int4.litertlm"),
             std::filesystem::path ("gemma-3n-E2B-it-int4-Web.litertlm"),
             std::filesystem::path ("gemma-3n-E2B-it-int4.mediatek.mt6993.litertlm") })
        {
          const auto path = gemma_root / candidate;
          if (std::filesystem::exists (path))
            {
              return path;
            }
        }
    }
  return std::nullopt;
}

std::optional<std::filesystem::path>
ResolveLfm2SummarizerModelPath (const std::filesystem::path &models_dir)
{
  const std::string override
      = GetEnvOrDefault ("CORTEXT_LFM2_SUMMARIZER_MODEL");
  if (!override.empty ())
    {
      const std::filesystem::path path (override);
      if (std::filesystem::exists (path))
        {
          return path;
      }
      return std::nullopt;
    }

  const auto roots = DeepLlmSearchRoots (models_dir);
  if (auto preferred = FindFirstExistingModel (
          roots,
          { std::filesystem::path (),
            std::filesystem::path ("lfm2.5-350m-gguf"),
            std::filesystem::path ("LFM2.5-350M-GGUF") },
          { std::filesystem::path ("LFM2.5-350M-Q4_K_M.gguf"),
            std::filesystem::path ("LFM2.5-350M-Q8_0.gguf"),
            std::filesystem::path ("LFM2.5-350M-Q5_K_M.gguf"),
            std::filesystem::path ("LFM2.5-350M-Q6_K.gguf"),
            std::filesystem::path ("LFM2.5-350M-Q4_0.gguf"),
            std::filesystem::path ("LFM2.5-350M-BF16.gguf"),
            std::filesystem::path ("LFM2.5-350M-F16.gguf") }))
    {
      return preferred;
    }

  return FindFirstExistingModel (
      roots,
      { std::filesystem::path (),
        std::filesystem::path ("lfm2-2.6b-transcript-gguf"),
        std::filesystem::path ("LFM2-2.6B-Transcript-GGUF") },
      { std::filesystem::path ("LFM2-2.6B-Transcript-Q4_K_M.gguf") });
}

std::optional<std::filesystem::path>
ResolveLfm2ExtractorModelPath (const std::filesystem::path &models_dir)
{
  const std::string override = GetEnvOrDefault ("CORTEXT_LFM2_EXTRACT_MODEL");
  if (!override.empty ())
    {
      const std::filesystem::path path (override);
      if (std::filesystem::exists (path))
        {
          return path;
      }
      return std::nullopt;
    }

  const auto roots = DeepLlmSearchRoots (models_dir);
  if (auto preferred = FindFirstExistingModel (
          roots,
          { std::filesystem::path (),
            std::filesystem::path ("lfm2.5-350m-gguf"),
            std::filesystem::path ("LFM2.5-350M-GGUF") },
          { std::filesystem::path ("LFM2.5-350M-Q4_K_M.gguf"),
            std::filesystem::path ("LFM2.5-350M-Q8_0.gguf"),
            std::filesystem::path ("LFM2.5-350M-Q5_K_M.gguf"),
            std::filesystem::path ("LFM2.5-350M-Q6_K.gguf"),
            std::filesystem::path ("LFM2.5-350M-Q4_0.gguf"),
            std::filesystem::path ("LFM2.5-350M-BF16.gguf"),
            std::filesystem::path ("LFM2.5-350M-F16.gguf") }))
    {
      return preferred;
    }

  return FindFirstExistingModel (
      roots,
      { std::filesystem::path (),
        std::filesystem::path ("lfm2-1.2b-extract-gguf"),
        std::filesystem::path ("LFM2-1.2B-Extract-GGUF") },
      { std::filesystem::path ("LFM2-1.2B-Extract-Q4_K_M.gguf") });
}

std::optional<DeepLlmSelection>
TryCreateDeepLlmSelection (const std::filesystem::path &models_dir,
                           DeepLlmBackend backend, std::string *error_out)
{
  switch (backend)
    {
    case DeepLlmBackend::Gemma:
      return TryCreateGemmaSelection (models_dir, error_out);
    case DeepLlmBackend::Lfm2:
      return TryCreateLfm2Selection (models_dir, error_out);
    case DeepLlmBackend::Mixed:
      return TryCreateMixedSelection (models_dir, error_out);
    case DeepLlmBackend::Auto:
      break;
    }

  std::string mixed_error;
  std::string lfm2_error;
  if (auto selection
      = TryCreateLfm2Selection (models_dir, &lfm2_error))
    {
      return selection;
    }

  if (auto selection
      = TryCreateMixedSelection (models_dir, &mixed_error))
    {
      return selection;
    }

  std::string gemma_error;
  if (auto selection
      = TryCreateGemmaSelection (models_dir, &gemma_error))
    {
      return selection;
    }

  if (error_out != nullptr)
    {
      *error_out = mixed_error + "; " + gemma_error + "; " + lfm2_error;
    }
  return std::nullopt;
}

DeepLlmSelection
CreateDeepLlmSelection (const std::filesystem::path &models_dir)
{
  const DeepLlmBackend backend = ResolveDeepLlmBackendOverride ();
  std::string error;
  auto selection = TryCreateDeepLlmSelection (models_dir, backend, &error);
  if (!selection)
    {
      throw std::runtime_error ("No deep LLM backend available for "
                                + DescribeDeepLlmBackend (backend) + ": "
                                + error);
    }
  return std::move (*selection);
}

} // namespace cortext::internal
