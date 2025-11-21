#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cortext
{

/// @brief Interface for tri-modal encoding generation.
class Encoder
{
public:
  virtual ~Encoder () = default;

  virtual void EncodeText (const std::string &text,
                           std::vector<float> &out_embedding)
      = 0;

  virtual void EncodeAudio (const float *pcm, std::size_t num_samples,
                            std::vector<float> &out_embedding)
      = 0;

  virtual void EncodeImage (const std::uint8_t *data, int width, int height,
                            int channels, std::vector<float> &out_embedding)
      = 0;
};

} // namespace cortext
