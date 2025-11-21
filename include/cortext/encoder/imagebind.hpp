#pragma once

#include "cortext/encoder/encoder.hpp"
#include <string>
#include <vector>

namespace cortext
{

/// @brief ImageBind-oriented encoder stub. Methods are no-ops for now.
class ImageBindEncoder : public Encoder
{
public:
  explicit ImageBindEncoder (std::string models_dir);

  void EncodeText (const std::string &text,
                   std::vector<float> &out_embedding) override;

  void EncodeAudio (const float *pcm, std::size_t num_samples,
                    std::vector<float> &out_embedding) override;

  void EncodeImage (const std::uint8_t *data, int width, int height,
                    int channels, std::vector<float> &out_embedding) override;

private:
  std::string models_dir_;
  static constexpr int kDim = 256;
  static void FillZeros (std::vector<float> &out, int dim);
};

} // namespace cortext
