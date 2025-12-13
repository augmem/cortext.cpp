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

  /// @brief Encodes text input into an embedding vector.
  /// @param text Input text string (UTF-8 encoded).
  /// @param out_embedding Output embedding vector (cleared and resized by implementation).
  ///
  /// Throws std::runtime_error on encoding failure (e.g., model not available).
  /// The output embedding dimension is implementation-defined but typically fixed.
  virtual void EncodeText (const std::string &text,
                           std::vector<float> &out_embedding)
      = 0;

  /// @brief Encodes audio PCM samples into an embedding vector.
  /// @param pcm Pointer to PCM float32 samples (must be non-null).
  /// @param num_samples Number of PCM samples.
  /// @param out_embedding Output embedding vector (cleared and resized by implementation).
  ///
  /// Throws std::runtime_error on encoding failure (e.g., unsupported sample rate).
  /// Expected format is 16kHz mono float32 PCM for ImageBind-based encoders.
  virtual void EncodeAudio (const float *pcm, std::size_t num_samples,
                            std::vector<float> &out_embedding)
      = 0;

  /// @brief Encodes image data into an embedding vector.
  /// @param data Pointer to raw image pixels (must be non-null).
  /// @param width Image width in pixels (must be > 0).
  /// @param height Image height in pixels (must be > 0).
  /// @param channels Number of color channels (3 for RGB, 4 for RGBA).
  /// @param out_embedding Output embedding vector (cleared and resized by implementation).
  ///
  /// Throws std::runtime_error on encoding failure (e.g., invalid dimensions).
  /// Image data is expected in row-major order: height × width × channels.
  virtual void EncodeImage (const std::uint8_t *data, int width, int height,
                            int channels, std::vector<float> &out_embedding)
      = 0;
};

} // namespace cortext
