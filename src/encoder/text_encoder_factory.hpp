#pragma once

#include "cortext/models/aist_gguf_encoder.hpp"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace cortext::internal
{

struct TextEncoderSelection
{
  std::unique_ptr<Encoder> encoder;
  std::string backend_name;
  std::filesystem::path resolved_path;
};

/// Create the engine's text encoder. AIST is Cortext's required embedding
/// model: it is resolved under the models directory (q8_0 preferred) or via
/// CORTEXT_AIST_MODEL_PATH, and there is no fallback encoder. A database's
/// stored embeddings are pinned to this encoder's fingerprint, so silently
/// swapping encoders is never correct; failing loudly here is intentional.
inline TextEncoderSelection
CreatePreferredTextEncoder (const std::string &models_dir)
{
  const std::filesystem::path root (models_dir);

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
      "AIST GGUF model not found under " + root.string ()
      + ". Cortext requires the AIST encoder (models/AIST-87M-GGUF/, or set "
        "CORTEXT_AIST_MODEL_PATH).");
}

} // namespace cortext::internal
