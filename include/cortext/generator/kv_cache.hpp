#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace cortext
{

/// @brief Specification for KV cache dimensions.
///
/// For gemma-3n-E2B:
/// - num_layers = 30
/// - num_kv_heads = 2 (grouped-query attention)
/// - head_dim = 256
/// - batch_size = 1 (typical for inference)
struct KVSpec
{
  int num_layers = 30;
  int num_kv_heads = 2;
  int head_dim = 256;
  int batch_size = 1;
};

/// @brief KV cache manager for autoregressive generation.
///
/// Preallocates contiguous per-layer key/value buffers [B, H, T, D] and
/// provides views for binding as ONNX model inputs, plus methods to
/// initialize from prefill and append tokens during decoding.
///
/// Key names follow ONNX format: "past_key_values.{layer}.key/value"
///
/// Usage:
/// @code
/// KVSpec spec{.num_layers = 30, .num_kv_heads = 2, .head_dim = 256};
/// KVCache cache(spec, /*capacity=*/512);
///
/// // After prefill
/// cache.InitFromPresent(prefill_outputs);
///
/// // During decode loop
/// auto past = cache.PastInputs();
/// // ... run model with past ...
/// cache.AppendFromPresent(decode_outputs);
/// @endcode
class KVCache
{
public:
  /// @brief Construct KV cache with given spec and initial capacity.
  /// @param spec Model dimensions.
  /// @param capacity Initial sequence length capacity (grows if needed).
  KVCache (KVSpec spec, std::size_t capacity);

  ~KVCache () = default;

  KVCache (const KVCache &) = delete;
  KVCache &operator= (const KVCache &) = delete;
  KVCache (KVCache &&) noexcept = default;
  KVCache &operator= (KVCache &&) noexcept = default;

  /// @brief Get past KV tensors for model input binding.
  /// @return Map of "past_key_values.{layer}.key/value" -> [B, H, length, D].
  ///
  /// Returns views (slices) into the cache up to current length.
  /// Tensors are in row-major order (C contiguous).
  std::map<std::string, std::vector<float> > PastInputs () const;

  /// @brief Initialize cache from prefill model outputs.
  /// @param present Map of "present_key_values.{layer}.key/value" tensors.
  ///
  /// Copies entire time dimension from present tensors into cache.
  /// Sets length to the sequence length from the tensors.
  void InitFromPresent (
      const std::map<std::string, std::vector<float> > &present,
      const std::vector<std::size_t> &shape);

  /// @brief Append single token's KV from decode model outputs.
  /// @param present Map of "present_key_values.{layer}.key/value" tensors.
  ///
  /// Takes the last time position from present tensors and appends to cache.
  /// Increments length by 1.
  void AppendFromPresent (
      const std::map<std::string, std::vector<float> > &present);

  /// @brief Current sequence length in cache.
  std::size_t Length () const { return length_; }

  /// @brief Current allocated capacity.
  std::size_t Capacity () const { return capacity_; }

  /// @brief Ensure cache can hold at least `needed` tokens.
  /// @param needed Required capacity.
  ///
  /// If current capacity is insufficient, reallocates with 1.5x growth factor.
  void EnsureCapacity (std::size_t needed);

  /// @brief Reset cache to empty state (length = 0).
  void Clear ();

  /// @brief Get model specification.
  const KVSpec &Spec () const { return spec_; }

private:
  KVSpec spec_;
  std::size_t capacity_;
  std::size_t length_ = 0;

  // Per-layer key/value buffers: [B, H, capacity, D] in row-major order
  std::vector<std::vector<float> > keys_;   // [num_layers]
  std::vector<std::vector<float> > values_; // [num_layers]

  /// @brief Get index into flat buffer for [b, h, t, d].
  std::size_t FlatIndex (int b, int h, std::size_t t, int d) const;

  /// @brief Copy slice from source to destination.
  void CopySlice (const float *src, std::size_t src_t, float *dst,
                  std::size_t dst_t, std::size_t count);
};

} // namespace cortext
