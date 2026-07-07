#pragma once

#include "cortext/models/aist_gguf_encoder.hpp"

#include <filesystem>
#include <memory>
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

/// Create the engine's text encoder. AIST is Cortext's required embedding
/// model: it is resolved via CORTEXT_AIST_MODEL_PATH or the internal default
/// search roots, and there is no fallback encoder. A database's stored
/// embeddings are pinned to this encoder's fingerprint, so silently swapping
/// encoders is never correct; failing loudly here is intentional.
inline TextEncoderSelection
CreatePreferredTextEncoder ()
{
  const std::vector<std::filesystem::path> search_roots = {
    "models",
    "../models",
    "../../models",
  };

  for (const auto &root : search_roots)
    if (auto aist = ResolveAistGgufModelPath (root))
      {
        AistGgufConfig cfg;
        cfg.model_path = aist->string ();
        TextEncoderSelection selection;
        selection.backend_name = "AIST-87M-GGUF";
        selection.resolved_path = *aist;
        selection.encoder = std::make_unique<AistGgufEncoder> (cfg);
        return selection;
      }

  throw std::runtime_error (
      "AIST GGUF model not found. Cortext requires the AIST encoder; set "
      "CORTEXT_AIST_MODEL_PATH to an AIST .gguf file when using a non-default "
      "install layout.");
}

} // namespace cortext::internal
