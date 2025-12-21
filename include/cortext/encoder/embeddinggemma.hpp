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

/// @brief EmbeddingGemma text encoder backed by ONNX Runtime.
///
/// This encoder depends on ONNX Runtime and the SentencePiece tokenizer from
/// the LiteRT-LM bundle. When unavailable, all methods throw std::runtime_error.
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
