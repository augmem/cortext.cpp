#pragma once

#include "cortext/encoder/encoder.hpp"
#include <memory>
#include <string>
#include <vector>

namespace cortext
{

/// @brief ImageBind encoder backed by ONNX Runtime sessions (host-only).
///
/// This encoder requires ONNX Runtime and filesystem access for loading
/// BPE vocabulary and model files. It is NOT available in WASM builds.
/// When CORTEXT_ENABLE_IMAGEBIND_ORT is disabled, all methods throw
/// std::runtime_error.
class ImageBindEncoder : public Encoder
{
public:
  explicit ImageBindEncoder (std::string models_dir);
  ~ImageBindEncoder () override;

  void EncodeText (const std::string &text,
                   std::vector<float> &out_embedding) override;

  void EncodeAudio (const float *pcm, std::size_t num_samples,
                    std::vector<float> &out_embedding) override;

  void EncodeImage (const std::uint8_t *data, int width, int height,
                    int channels, std::vector<float> &out_embedding) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  static constexpr int kDim = 256;
};

} // namespace cortext
