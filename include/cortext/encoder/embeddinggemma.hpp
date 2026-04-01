#pragma once

#include "cortext/encoder/encoder.hpp"

#include <memory>
#include <string>
#include <vector>

namespace cortext
{

struct EmbeddingGemmaConfig
{
  std::string model_path;
  std::string tokenizer_path;
  int max_length = 256;
  int num_threads = 1;
};

/// @brief EmbeddingGemma text encoder with pluggable local backends.
///
/// Backend selection is internal and derived from the configured model artifact
/// or local override. Supported artifact types currently include ONNX, GGUF via
/// llama.cpp, and LiteRT/TFLite. When the selected runtime is unavailable, all
/// methods throw std::runtime_error.
class EmbeddingGemmaEncoder : public Encoder
{
public:
  explicit EmbeddingGemmaEncoder (EmbeddingGemmaConfig config);
  ~EmbeddingGemmaEncoder () override;

  void EncodeText (const std::string &text,
                   std::vector<float> &out_embedding) override;

  void EncodeAudio (const float *pcm, std::size_t num_samples,
                    std::vector<float> &out_embedding) override;

  void EncodeImage (const std::uint8_t *data, int width, int height,
                    int channels, std::vector<float> &out_embedding) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext
