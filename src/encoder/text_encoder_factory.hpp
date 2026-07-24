#pragma once

#include "cortext/models/aist_embedded_model.hpp"
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
/// model: it is resolved via CORTEXT_AIST_MODEL_PATH, on-disk search roots, or
/// (when this build embeds it) the baked-in library payload. There is no
/// alternate encoder. A database's stored embeddings are pinned to this
/// encoder's fingerprint, so silently swapping encoders is never correct;
/// failing loudly here is intentional.
inline TextEncoderSelection
CreatePreferredTextEncoder ()
{
  // Explicit env / override path is handled inside ResolveAistGgufModelPath
  // when given a search root; also try empty-root env-only via models/.
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

  // Linked Git shards inside libcortext: assemble full GGUF into the cache once.
  if (auto embedded = MaterializeEmbeddedAistModel ())
    {
      AistGgufConfig cfg;
      cfg.model_path = embedded->string ();
      // Vocab is written beside the model tree by materialize
      // (.../mdbr-leaf-ir/vocab.txt) for ResolveAistTokenizerPath.
      (void)MaterializeEmbeddedAistVocab ();
      TextEncoderSelection selection;
      selection.backend_name = "AIST-87M-GGUF";
      selection.resolved_path = *embedded;
      selection.encoder = std::make_unique<AistGgufEncoder> (cfg);
      return selection;
    }

  throw std::runtime_error (
      "AIST GGUF model not found. This Cortext build has no linked AIST shards "
      "and no on-disk assets; set CORTEXT_AIST_MODEL_PATH to an AIST .gguf file, "
      "or rebuild with -DCORTEXT_EMBED_AIST_MODEL=ON (default shared library).");
}

} // namespace cortext::internal
