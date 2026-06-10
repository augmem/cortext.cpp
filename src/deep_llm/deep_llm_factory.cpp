#include "deep_llm_factory.hpp"

#include "cortext/extractor/gemma_extractor.hpp"
#include "cortext/providers/adapters.hpp"
#include "cortext/providers/registry.hpp"
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
          *error_out = "Gemma backend unavailable: no gemma4-e2b-litert model found";
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
          *error_out = "Mixed backend unavailable: no gemma4-e2b-litert model found";
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
      for (const auto &candidate :
           { root / "gemma4-e2b-litert" / "gemma-4-E2B-it.litertlm" })
        {
          if (std::filesystem::exists (candidate))
            {
              return candidate;
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
            std::filesystem::path ("lfm2.5-1.2b-instruct-gguf"),
            std::filesystem::path ("LFM2.5-1.2B-Instruct-GGUF") },
          { std::filesystem::path ("LFM2.5-1.2B-Instruct-Q4_K_M.gguf"),
            std::filesystem::path ("LFM2.5-1.2B-Instruct-Q8_0.gguf"),
            std::filesystem::path ("LFM2.5-1.2B-Instruct-Q5_K_M.gguf"),
            std::filesystem::path ("LFM2.5-1.2B-Instruct-Q6_K.gguf"),
            std::filesystem::path ("LFM2.5-1.2B-Instruct-Q4_0.gguf"),
            std::filesystem::path ("LFM2.5-1.2B-Instruct-BF16.gguf"),
            std::filesystem::path ("LFM2.5-1.2B-Instruct-F16.gguf") }))
    {
      return preferred;
    }

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
      return TryCreateGemmaSelection (models_dir, error_out);
    }

  return std::nullopt;
}

namespace
{

std::shared_ptr<providers::InferenceProvider>
ResolveRoleProviderOrThrow (providers::Role role, const std::string &uri)
{
  std::string error;
  auto provider = providers::ResolveProvider (uri, role, &error);
  if (provider == nullptr)
    {
      const char *var = role == providers::Role::Summarizer
                            ? "CORTEXT_SUMMARIZER"
                            : "CORTEXT_EXTRACTOR";
      throw std::runtime_error (std::string (var) + "=" + uri
                                + " could not be resolved: " + error);
    }
  return std::shared_ptr<providers::InferenceProvider> (std::move (provider));
}

void
InstallSummarizerProvider (DeepLlmSelection &selection,
                           const std::string &uri)
{
  auto provider
      = ResolveRoleProviderOrThrow (providers::Role::Summarizer, uri);
  selection.summarizer_model_path = provider->Identity ().endpoint;
  selection.backend_name += "+summarizer:" + provider->Identity ().scheme;
  selection.summarizer
      = std::make_unique<providers::ProviderSummarizer> (std::move (provider));
}

void
InstallExtractorProvider (DeepLlmSelection &selection, const std::string &uri)
{
  auto provider
      = ResolveRoleProviderOrThrow (providers::Role::Extractor, uri);
  selection.extractor_model_path = provider->Identity ().endpoint;
  selection.backend_name += "+extractor:" + provider->Identity ().scheme;
  selection.extractor
      = std::make_unique<providers::ProviderExtractor> (std::move (provider));
}

} // namespace

DeepLlmSelection
CreateDeepLlmSelection (const std::filesystem::path &models_dir)
{
  using providers::Role;
  const auto summarizer_uri = providers::RoleUriFromEnvironment (
      Role::Summarizer);
  const auto extractor_uri = providers::RoleUriFromEnvironment (
      Role::Extractor);

  // Fully provider-specified setups are true replacement DI: build the
  // selection entirely from providers without touching local model
  // discovery, so remote-only deployments need no local weights at all.
  if (summarizer_uri && extractor_uri)
    {
      DeepLlmSelection selection;
      selection.backend_name = "providers";
      InstallSummarizerProvider (selection, *summarizer_uri);
      InstallExtractorProvider (selection, *extractor_uri);
      return selection;
    }

  const DeepLlmBackend backend = ResolveDeepLlmBackendOverride ();
  std::string error;
  auto selection = TryCreateDeepLlmSelection (models_dir, backend, &error);
  if (!selection)
    {
      throw std::runtime_error ("No deep LLM backend available for "
                                + DescribeDeepLlmBackend (backend) + ": "
                                + error);
    }
  // Partial override: the local factory supplies the role without an env
  // uri, so mixed setups (e.g. remote summarizer + local constrained
  // extractor) compose naturally.
  if (summarizer_uri)
    {
      InstallSummarizerProvider (*selection, *summarizer_uri);
    }
  if (extractor_uri)
    {
      InstallExtractorProvider (*selection, *extractor_uri);
    }
  return std::move (*selection);
}

} // namespace cortext::internal
