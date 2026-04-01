#pragma once

#include <cstdint>

namespace cortext::internal
{

struct EmbeddingGemmaTextEncodeProfileSnapshot
{
  std::uint64_t calls = 0;
  double ensure_initialized_ms = 0.0;
  double tokenize_ms = 0.0;
  double tensor_create_ms = 0.0;
  double run_ms = 0.0;
  double copy_ms = 0.0;
  double normalize_ms = 0.0;
};

void ResetEmbeddingGemmaTextEncodeProfile ();
EmbeddingGemmaTextEncodeProfileSnapshot
GetEmbeddingGemmaTextEncodeProfileSnapshot ();

#if !defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_ORT)
inline void
ResetEmbeddingGemmaTextEncodeProfile ()
{
}

inline EmbeddingGemmaTextEncodeProfileSnapshot
GetEmbeddingGemmaTextEncodeProfileSnapshot ()
{
  return {};
}
#endif

} // namespace cortext::internal
