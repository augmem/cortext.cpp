#pragma once

#include "cortext/generator/text_generator.hpp"

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace cortext
{

/// @brief Gemma-3n text generator with schema-constrained JSON support.
///
/// Complete port of json_decoder.py ConstrainedJSONGenerator including:
/// - Autoregressive text generation with KV cache
/// - Schema-constrained JSON generation with streaming parser
/// - Temperature, top-k, and top-p sampling strategies
/// - Multimodal input support (text, image, audio)
///
/// Uses ONNX Runtime for inference with optional IOBinding acceleration.
/// Requires CORTEXT_ENABLE_GEMMA_ORT to be defined for actual inference.
class GemmaTextGenerator : public TextGenerator
{
public:
  /// @brief Initialize generator with model directory.
  /// @param models_dir Path to gemma-3n model directory containing:
  ///   - tokenizer.json (HuggingFace tokenizer)
  ///   - onnx/decoder_model_merged_q4.onnx
  ///   - onnx/embed_tokens_quantized.onnx
  ///   - onnx/vision_encoder.onnx (optional)
  ///   - onnx/audio_encoder.onnx (optional)
  explicit GemmaTextGenerator (const std::string &models_dir);

  ~GemmaTextGenerator () override;

  GemmaTextGenerator (const GemmaTextGenerator &) = delete;
  GemmaTextGenerator &operator= (const GemmaTextGenerator &) = delete;
  GemmaTextGenerator (GemmaTextGenerator &&) noexcept;
  GemmaTextGenerator &operator= (GemmaTextGenerator &&) noexcept;

  // --- TextGenerator interface ---

  /// @brief Generate text from a text-only prompt.
  /// @param prompt Input text prompt.
  /// @param config Generation parameters.
  /// @return Generated text completion (excluding prompt).
  std::string Generate (const std::string &prompt,
                        const GenerationConfig &config = {}) override;

  /// @brief Generate text from multimodal inputs.
  /// @param inputs Sequence of interleaved text/image/audio inputs.
  /// @param config Generation parameters.
  /// @return Generated text completion.
  std::string GenerateMultimodal (const std::vector<GenerationInput> &inputs,
                                  const GenerationConfig &config = {}) override;

  /// @brief Check if the generator is available (models loaded).
  bool IsAvailable () const override;

  // --- Schema-constrained JSON generation ---

  /// @brief Generate JSON conforming to a schema from text prompt.
  /// @param prompt Input text prompt.
  /// @param schema JSON Schema defining expected structure.
  /// @param max_tokens Maximum tokens to generate (usually stops early).
  /// @param temperature Sampling temperature (lower = more deterministic).
  /// @param top_k Optional top-k filtering (0 or negative to disable).
  /// @param top_p Optional nucleus sampling threshold (0 or negative to
  /// disable).
  /// @return Parsed JSON object matching schema.
  ///
  /// Generation stops early when schema requirements are satisfied.
  /// Throws std::runtime_error if generation fails or produces invalid JSON.
  nlohmann::json GenerateJSON (const std::string &prompt,
                               const nlohmann::json &schema,
                               int max_tokens = 200, float temperature = 0.3f,
                               int top_k = 0, float top_p = 0.0f);

  /// @brief Generate JSON from multimodal inputs.
  /// @param inputs Sequence of interleaved text/image/audio inputs.
  /// @param schema JSON Schema defining expected structure.
  /// @param max_tokens Maximum tokens to generate.
  /// @param temperature Sampling temperature.
  /// @param top_k Optional top-k filtering.
  /// @param top_p Optional nucleus sampling.
  /// @return Parsed JSON object matching schema.
  nlohmann::json GenerateJSONMultimodal (
      const std::vector<GenerationInput> &inputs, const nlohmann::json &schema,
      int max_tokens = 200, float temperature = 0.3f, int top_k = 0,
      float top_p = 0.0f);

  // --- Accessors ---

  /// @brief Get number of tokens generated in last call.
  /// @return Token count from last generation.
  std::size_t LastTokenCount () const;

  /// @brief Get EOS token ID.
  int64_t EosTokenId () const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext
