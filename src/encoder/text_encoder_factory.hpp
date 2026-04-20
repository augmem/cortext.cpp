#pragma once

#include "cortext/encoder/embeddinggemma.hpp"
#include "cortext/encoder/imagebind.hpp"

#include <filesystem>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cortext::internal
{

struct TextEncoderSelection
{
  std::unique_ptr<Encoder> encoder;
  std::string backend_name;
  std::filesystem::path resolved_path;
};

inline std::string
DescribeEmbeddingGemmaBackend (
    const std::filesystem::path &model_path)
{
  const std::string ext = model_path.extension ().string ();
  if (ext == ".gguf")
    {
      return "EmbeddingGemma/llama.cpp";
    }
  if (ext == ".tflite")
    {
      return "EmbeddingGemma/LiteRT";
    }
  if (ext == ".onnx")
    {
      return "EmbeddingGemma/ONNX";
    }
  return "EmbeddingGemma";
}

inline void
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

inline std::optional<EmbeddingGemmaConfig>
ResolveEmbeddingGemmaConfigForBackend (const std::filesystem::path &models_dir,
                                       const std::string &backend)
{
  std::vector<std::filesystem::path> candidates;

  std::filesystem::path tokenizer_name;
  bool require_tokenizer = true;
  std::vector<std::filesystem::path> model_candidates;
  if (backend == "onnx")
    {
      tokenizer_name = "tokenizer.model";
      AppendUniquePath (candidates, models_dir);
      AppendUniquePath (candidates, models_dir / "embeddinggemma-300m-onnx");
      if (models_dir.filename () == "imagebind")
        {
          AppendUniquePath (
              candidates, models_dir.parent_path () / "embeddinggemma-300m-onnx");
        }
      model_candidates = {
        std::filesystem::path ("onnx/model_q4.onnx"),
        std::filesystem::path ("onnx/model_quantized.onnx"),
        std::filesystem::path ("onnx/model.onnx"),
      };
    }
  else if (backend == "llama.cpp")
    {
      require_tokenizer = false;
      AppendUniquePath (candidates, models_dir);
      AppendUniquePath (candidates, models_dir / "llama_cpp");
      if (models_dir.filename () == "imagebind")
        {
          AppendUniquePath (candidates, models_dir.parent_path () / "llama_cpp");
        }
      model_candidates = {
        std::filesystem::path ("embeddinggemma-300M-Q8_0.gguf"),
        std::filesystem::path ("embeddinggemma-300M-Q4_K_M.gguf"),
      };
    }
  else if (backend == "litert")
    {
      tokenizer_name = "sentencepiece.model";
      AppendUniquePath (candidates, models_dir);
      AppendUniquePath (candidates, models_dir / "embeddinggemma-300m-litert");
      if (models_dir.filename () == "imagebind")
        {
          AppendUniquePath (
              candidates, models_dir.parent_path () / "embeddinggemma-300m-litert");
        }
      model_candidates = {
        std::filesystem::path ("embeddinggemma-300M_seq256_mixed-precision.tflite"),
      };
    }
  else
    {
      throw std::runtime_error ("Unsupported EmbeddingGemma backend: " + backend);
    }

  for (const auto &dir : candidates)
    {
      const auto tokenizer
          = require_tokenizer ? dir / tokenizer_name : std::filesystem::path ();
      if (require_tokenizer && !std::filesystem::exists (tokenizer))
        {
          continue;
        }

      for (const auto &model_rel : model_candidates)
        {
          const auto model = dir / model_rel;
          if (!std::filesystem::exists (model))
            {
              continue;
            }

          EmbeddingGemmaConfig cfg;
          cfg.model_path = model.string ();
          cfg.tokenizer_path = tokenizer.string ();
          return cfg;
        }
    }

  return std::nullopt;
}

inline std::optional<EmbeddingGemmaConfig>
ResolveEmbeddingGemmaConfig (const std::filesystem::path &models_dir)
{
  const char *backend_override = std::getenv ("CORTEXT_EMBEDDINGGEMMA_BACKEND");
  if (backend_override != nullptr && *backend_override != '\0')
    {
      return ResolveEmbeddingGemmaConfigForBackend (models_dir, backend_override);
    }

  std::vector<std::string> backends = { "llama.cpp" };
#if !defined(CORTEXT_DISABLE_LITERT)
  backends.push_back ("litert");
#endif
#if defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_ORT)
  backends.push_back ("onnx");
#endif

  for (const auto &backend : backends)
    {
      if (auto cfg = ResolveEmbeddingGemmaConfigForBackend (models_dir, backend))
        {
          return cfg;
        }
    }

  return std::nullopt;
}

inline std::optional<std::filesystem::path>
ResolveImageBindDir (const std::filesystem::path &models_dir)
{
  const auto has_imagebind = [] (const std::filesystem::path &dir) {
    return std::filesystem::exists (dir / "text_encoder.onnx")
           || std::filesystem::exists (dir / "text_encoder_int8.onnx");
  };

  std::vector<std::filesystem::path> candidates;
  AppendUniquePath (candidates, models_dir);
  AppendUniquePath (candidates, models_dir / "imagebind");
  if (models_dir.filename () == "embeddinggemma-300m-onnx")
    {
      AppendUniquePath (candidates, models_dir.parent_path () / "imagebind");
    }

  for (const auto &dir : candidates)
    {
      if (has_imagebind (dir))
        {
          return dir;
        }
    }

  return std::nullopt;
}

inline TextEncoderSelection
CreatePreferredTextEncoder (const std::string &models_dir)
{
  const std::filesystem::path root (models_dir);

#if defined(CORTEXT_ENABLE_EMBEDDINGGEMMA)
  if (auto gemma = ResolveEmbeddingGemmaConfig (root))
    {
      TextEncoderSelection selection;
      const std::filesystem::path model_path (gemma->model_path);
      selection.backend_name = DescribeEmbeddingGemmaBackend (model_path);
      selection.resolved_path = model_path;
      selection.encoder
          = std::make_unique<EmbeddingGemmaEncoder> (std::move (*gemma));
      return selection;
    }
#endif

#if defined(CORTEXT_ENABLE_IMAGEBIND_ORT)
  if (auto imagebind = ResolveImageBindDir (root))
    {
      TextEncoderSelection selection;
      selection.backend_name = "ImageBind";
      selection.resolved_path = *imagebind;
      selection.encoder = std::make_unique<ImageBindEncoder> (imagebind->string ());
      return selection;
    }
#endif

  throw std::runtime_error (
      "No supported text encoder models found under " + root.string () + ".");
}

} // namespace cortext::internal
