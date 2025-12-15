#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cortext
{

/// @brief Vision encoder output.
struct VisionEncoderOutput
{
  std::vector<float> features;       ///< Image features [N, hidden_dim]
  std::vector<std::size_t> shape;    ///< Output shape
  std::size_t num_image_tokens = 0;  ///< Number of tokens representing image
};

/// @brief Audio encoder output.
struct AudioEncoderOutput
{
  std::vector<float> features;       ///< Audio features [N, hidden_dim]
  std::vector<std::size_t> shape;    ///< Output shape
  std::size_t num_audio_tokens = 0;  ///< Number of tokens representing audio
};

/// @brief Vision encoder for gemma-3n.
///
/// Processes images and produces embeddings that replace image placeholder
/// tokens in the decoder input. Uses the vision_encoder ONNX model.
///
/// Image preprocessing:
/// - Resize to 256/512/768 (shortest side)
/// - Center crop to square
/// - Normalize with ImageNet statistics
class GemmaVisionEncoder
{
public:
  /// @brief Initialize vision encoder.
  /// @param models_dir Path to gemma-3n model directory containing
  ///   onnx/vision_encoder_quantized.onnx (or vision_encoder.onnx)
  explicit GemmaVisionEncoder (const std::string &models_dir);

  ~GemmaVisionEncoder ();

  GemmaVisionEncoder (const GemmaVisionEncoder &) = delete;
  GemmaVisionEncoder &operator= (const GemmaVisionEncoder &) = delete;
  GemmaVisionEncoder (GemmaVisionEncoder &&) noexcept;
  GemmaVisionEncoder &operator= (GemmaVisionEncoder &&) noexcept;

  /// @brief Encode an image to embedding features.
  /// @param data Raw pixel data (RGB or RGBA, row-major).
  /// @param width Image width in pixels.
  /// @param height Image height in pixels.
  /// @param channels Number of channels (3=RGB, 4=RGBA).
  /// @return Vision encoder output with features.
  ///
  /// The output features are meant to replace image placeholder tokens
  /// in the input embeddings before running the decoder.
  VisionEncoderOutput Encode (const std::uint8_t *data, int width, int height,
                              int channels);

  /// @brief Check if vision encoder is available.
  bool IsAvailable () const;

  /// @brief Get hidden dimension of output features.
  std::size_t HiddenDim () const;

  /// @brief Get image token ID (placeholder in input_ids).
  int64_t ImageTokenId () const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  /// @brief Preprocess image to model input format.
  /// @param data Raw pixel data.
  /// @param width Image width.
  /// @param height Image height.
  /// @param channels Number of channels.
  /// @return Preprocessed pixel values [1, C, H, W] as float32.
  std::vector<float> Preprocess (const std::uint8_t *data, int width,
                                 int height, int channels);
};

/// @brief Audio encoder for gemma-3n.
///
/// Processes audio and produces embeddings that replace audio placeholder
/// tokens in the decoder input. Uses the audio_encoder ONNX model.
///
/// Audio preprocessing:
/// - Resample to 16kHz mono
/// - Compute mel-spectrogram (128 bins, 25ms window, 10ms hop)
/// - Normalize with model-specific statistics
class GemmaAudioEncoder
{
public:
  /// @brief Initialize audio encoder.
  /// @param models_dir Path to gemma-3n model directory containing
  ///   onnx/audio_encoder_q4.onnx (or audio_encoder.onnx)
  explicit GemmaAudioEncoder (const std::string &models_dir);

  ~GemmaAudioEncoder ();

  GemmaAudioEncoder (const GemmaAudioEncoder &) = delete;
  GemmaAudioEncoder &operator= (const GemmaAudioEncoder &) = delete;
  GemmaAudioEncoder (GemmaAudioEncoder &&) noexcept;
  GemmaAudioEncoder &operator= (GemmaAudioEncoder &&) noexcept;

  /// @brief Encode audio to embedding features.
  /// @param pcm Audio samples as PCM float32 (range [-1, 1]).
  /// @param num_samples Number of samples.
  /// @param sample_rate Sample rate in Hz.
  /// @return Audio encoder output with features.
  ///
  /// The output features are meant to replace audio placeholder tokens
  /// in the input embeddings. Produces ~6.25 tokens per second of audio.
  AudioEncoderOutput Encode (const float *pcm, std::size_t num_samples,
                             int sample_rate);

  /// @brief Check if audio encoder is available.
  bool IsAvailable () const;

  /// @brief Get hidden dimension of output features.
  std::size_t HiddenDim () const;

  /// @brief Get audio token ID (placeholder in input_ids).
  int64_t AudioTokenId () const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  /// @brief Compute mel-spectrogram from audio.
  /// @param pcm Audio samples.
  /// @param num_samples Number of samples.
  /// @param sample_rate Sample rate.
  /// @return Mel-spectrogram features and mask.
  std::pair<std::vector<float>, std::vector<float> >
  ComputeMelSpectrogram (const float *pcm, std::size_t num_samples,
                         int sample_rate);
};

/// @brief Combined multimodal encoder for gemma-3n.
///
/// Handles both vision and audio encoding, providing a unified interface
/// for processing multimodal inputs. Replaces placeholder tokens in the
/// input embeddings with encoded features.
class MultimodalEncoder
{
public:
  /// @brief Initialize multimodal encoder.
  /// @param models_dir Path to gemma-3n model directory.
  explicit MultimodalEncoder (const std::string &models_dir);

  ~MultimodalEncoder ();

  MultimodalEncoder (const MultimodalEncoder &) = delete;
  MultimodalEncoder &operator= (const MultimodalEncoder &) = delete;
  MultimodalEncoder (MultimodalEncoder &&) noexcept;
  MultimodalEncoder &operator= (MultimodalEncoder &&) noexcept;

  /// @brief Replace image placeholder tokens in embeddings.
  /// @param embeddings Input embeddings [batch, seq_len, hidden_dim].
  /// @param input_ids Token IDs [batch, seq_len].
  /// @param image_data Raw image pixel data.
  /// @param width Image width.
  /// @param height Image height.
  /// @param channels Number of channels.
  /// @return Updated embeddings with image features.
  std::vector<float>
  InjectImageFeatures (const std::vector<float> &embeddings,
                       const std::vector<std::size_t> &embed_shape,
                       const std::vector<int64_t> &input_ids,
                       const std::uint8_t *image_data, int width, int height,
                       int channels);

  /// @brief Replace audio placeholder tokens in embeddings.
  /// @param embeddings Input embeddings [batch, seq_len, hidden_dim].
  /// @param input_ids Token IDs [batch, seq_len].
  /// @param pcm Audio samples.
  /// @param num_samples Number of samples.
  /// @param sample_rate Sample rate.
  /// @return Updated embeddings with audio features.
  std::vector<float>
  InjectAudioFeatures (const std::vector<float> &embeddings,
                       const std::vector<std::size_t> &embed_shape,
                       const std::vector<int64_t> &input_ids, const float *pcm,
                       std::size_t num_samples, int sample_rate);

  /// @brief Check if vision encoder is available.
  bool HasVision () const;

  /// @brief Check if audio encoder is available.
  bool HasAudio () const;

  /// @brief Get image token ID.
  int64_t ImageTokenId () const;

  /// @brief Get audio token ID.
  int64_t AudioTokenId () const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext
