#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cortext
{

/// @brief Result from decoder forward pass.
struct DecoderOutput
{
  /// @brief Logits for next token prediction [batch, seq_len, vocab_size].
  std::vector<float> logits;
  std::vector<std::size_t> logits_shape; // [B, S, V]

  /// @brief Present key-value cache from this pass.
  /// Maps "present_key_values.{layer}.key/value" -> tensor data.
  std::map<std::string, std::vector<float> > present_kv;
  std::vector<std::size_t> kv_shape; // [B, H, S, D]
};

/// @brief Runs the Gemma decoder model with ONNX Runtime.
///
/// Uses IOBinding for efficient repeated inference during autoregressive
/// generation. Handles input/output binding for embeddings, position IDs,
/// and KV cache.
///
/// Requires CORTEXT_ENABLE_GEMMA_ORT to be defined for actual implementation.
///
/// Usage:
/// @code
/// DecoderRunner runner("models/gemma-3n/onnx");
/// auto output = runner.Run(embeds, per_layer, position_ids, past_kv);
/// // output.logits contains next-token logits
/// // output.present_kv contains updated KV cache
/// @endcode
class DecoderRunner
{
public:
  /// @brief Construct decoder runner from model directory.
  /// @param models_dir Directory containing decoder_model_merged_q4.onnx.
  /// @throws std::runtime_error if model cannot be loaded.
  explicit DecoderRunner (const std::string &models_dir);

  ~DecoderRunner ();

  DecoderRunner (const DecoderRunner &) = delete;
  DecoderRunner &operator= (const DecoderRunner &) = delete;
  DecoderRunner (DecoderRunner &&) noexcept;
  DecoderRunner &operator= (DecoderRunner &&) noexcept;

  /// @brief Run decoder forward pass.
  /// @param inputs_embeds Input embeddings [batch, seq_len, hidden_dim].
  /// @param per_layer_inputs Per-layer inputs (typically zeros).
  /// @param position_ids Position IDs [batch, seq_len].
  /// @param past_kv Past key-value cache from previous pass.
  /// @return Decoder output with logits and present KV cache.
  DecoderOutput Run (const std::vector<float> &inputs_embeds,
                     const std::vector<std::size_t> &embeds_shape,
                     const std::vector<float> &per_layer_inputs,
                     const std::vector<std::size_t> &per_layer_shape,
                     const std::vector<int64_t> &position_ids,
                     const std::vector<std::size_t> &position_shape,
                     const std::map<std::string, std::vector<float> > &past_kv,
                     const std::vector<std::size_t> &past_kv_shape);

  /// @brief Check if decoder is available (model loaded successfully).
  bool IsAvailable () const;

  /// @brief Get output names from the model.
  const std::vector<std::string> &OutputNames () const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// @brief Result from embed forward pass.
struct EmbedOutput
{
  /// @brief Input embeddings [batch, seq_len, hidden_dim].
  std::vector<float> inputs_embeds;
  std::vector<std::size_t> embeds_shape;

  /// @brief Per-layer inputs [batch, seq_len, num_layers, head_dim].
  std::vector<float> per_layer_inputs;
  std::vector<std::size_t> per_layer_shape;
};

/// @brief Runs the embedding model (embed_tokens) for token-to-embedding.
///
/// Converts token IDs to embeddings and per-layer inputs for decoder input.
/// Provides optimized methods for prefill (variable-length) and step
/// (single-token with buffer reuse) operations.
class EmbedRunner
{
public:
  /// @brief Construct embed runner from model directory.
  /// @param models_dir Directory containing embed_tokens_int8.onnx.
  explicit EmbedRunner (const std::string &models_dir);

  ~EmbedRunner ();

  EmbedRunner (const EmbedRunner &) = delete;
  EmbedRunner &operator= (const EmbedRunner &) = delete;
  EmbedRunner (EmbedRunner &&) noexcept;
  EmbedRunner &operator= (EmbedRunner &&) noexcept;

  /// @brief Embed token IDs to hidden states and per-layer inputs.
  /// @param token_ids Token IDs [batch, seq_len].
  /// @param shape Shape of token_ids.
  /// @return EmbedOutput with embeddings and per_layer_inputs.
  /// @deprecated Use EmbedPrefill or EmbedStep for better performance.
  EmbedOutput Embed (const std::vector<int64_t> &token_ids,
                     const std::vector<std::size_t> &shape);

  /// @brief Embed variable-length token sequence (prefill phase).
  /// @param token_ids Token IDs [batch, seq_len].
  /// @param shape Shape of token_ids.
  /// @return EmbedOutput with embeddings and per_layer_inputs.
  EmbedOutput EmbedPrefill (const std::vector<int64_t> &token_ids,
                            const std::vector<std::size_t> &shape);

  /// @brief Embed single token with buffer reuse (decode step).
  /// Uses preallocated buffers for zero-copy output on single-token decoding.
  /// @param token_id Single token ID to embed.
  /// @return EmbedOutput with embeddings and per_layer_inputs (views into
  ///         internal buffers - data is valid until next EmbedStep call).
  EmbedOutput EmbedStep (int64_t token_id);

  /// @brief Get hidden dimension of embeddings.
  std::size_t HiddenDim () const;

  /// @brief Check if embed model is available.
  bool IsAvailable () const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext
