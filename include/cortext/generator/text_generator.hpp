#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cortext
{

/// @brief Configuration for text generation.
struct GenerationConfig
{
  int max_new_tokens = 256;
  float temperature = 0.7f;
  float top_p = 0.9f;
  int top_k = 50;
  bool do_sample = true;
};

/// @brief Multimodal content item for generation input.
///
/// Matches Signal struct's modality representation for seamless integration
/// with memory blobs from consolidation clusters.
struct GenerationInput
{
  std::string modality; ///< "text" | "audio" | "image"

  // For text modality
  std::string text;

  // For image modality (raw pixels, row-major RGB(A))
  std::vector<std::uint8_t> image_data;
  int width = 0;
  int height = 0;
  int channels = 0; ///< 3=RGB, 4=RGBA

  // For audio modality (PCM float32)
  std::vector<float> audio_pcm;
  int sample_rate = 16000;

  /// @brief Create a text input.
  static GenerationInput
  Text (const std::string &t)
  {
    GenerationInput input;
    input.modality = "text";
    input.text = t;
    return input;
  }

  /// @brief Create an image input from raw pixel data.
  static GenerationInput
  Image (const std::uint8_t *data, int w, int h, int c)
  {
    GenerationInput input;
    input.modality = "image";
    input.image_data.assign (data, data + static_cast<std::size_t> (w * h * c));
    input.width = w;
    input.height = h;
    input.channels = c;
    return input;
  }

  /// @brief Create an audio input from PCM samples.
  static GenerationInput
  Audio (const float *pcm, std::size_t samples, int sr = 16000)
  {
    GenerationInput input;
    input.modality = "audio";
    input.audio_pcm.assign (pcm, pcm + samples);
    input.sample_rate = sr;
    return input;
  }
};

/// @brief Interface for text generation with optional multimodal input.
///
/// Implementations provide autoregressive text generation capabilities,
/// supporting both text-only and multimodal (text/image/audio) inputs.
class TextGenerator
{
public:
  virtual ~TextGenerator () = default;

  /// @brief Generate text from a text-only prompt.
  /// @param prompt Input text prompt.
  /// @param config Generation parameters.
  /// @return Generated text completion (excluding prompt).
  ///
  /// Throws std::runtime_error if generation fails or model unavailable.
  virtual std::string Generate (const std::string &prompt,
                                const GenerationConfig &config = {})
      = 0;

  /// @brief Generate text from multimodal inputs.
  /// @param inputs Sequence of interleaved text/image/audio inputs.
  /// @param config Generation parameters.
  /// @return Generated text completion.
  ///
  /// For gemma-3n-e2b:
  /// - Text is tokenized and embedded via embed_tokens
  /// - Images are encoded to 256 tokens via vision_encoder
  /// - Audio is encoded at ~6.25 tokens/second via audio_encoder
  ///
  /// Throws std::runtime_error if generation fails or model unavailable.
  virtual std::string
  GenerateMultimodal (const std::vector<GenerationInput> &inputs,
                      const GenerationConfig &config = {})
      = 0;

  /// @brief Check if the generator is available (models loaded).
  /// @return True if generation can be performed.
  virtual bool IsAvailable () const = 0;
};

} // namespace cortext
