#pragma once

#include "audio_loader.hpp"
#include "benchmark_config.hpp"
#include "metrics.hpp"

#include <memory>
#include <string>

namespace benchmark
{

/// @brief Benchmark runner for LiteRT-LM backend.
///
/// Uses Google's LiteRT-LM to run inference with Gemma 3n or other
/// LiteRT-LM compatible models. Measures tokens per second for text
/// and audio inputs.
///
/// When BENCHMARK_ENABLE_LITERT is not defined, all methods throw
/// std::runtime_error.
class LiteRTRunner
{
public:
  /// @brief Construct a LiteRT-LM runner.
  /// @param model_path Path to LiteRT-LM model (.litertlm file or directory)
  explicit LiteRTRunner (const std::string &model_path);

  ~LiteRTRunner ();

  LiteRTRunner (const LiteRTRunner &) = delete;
  LiteRTRunner &operator= (const LiteRTRunner &) = delete;

  LiteRTRunner (LiteRTRunner &&) noexcept;
  LiteRTRunner &operator= (LiteRTRunner &&) noexcept;

  /// @brief Check if the runner is available (model loaded).
  bool IsAvailable () const;

  /// @brief Get the model name.
  std::string GetModelName () const;

  /// @brief Run benchmark with text input.
  /// @param text Input text prompt
  /// @param max_tokens Maximum tokens to generate
  /// @return Metrics from this run
  RunMetrics RunTextInference (const std::string &text, int max_tokens);

  /// @brief Run benchmark with audio input.
  /// @param audio Audio data (must be 16kHz mono float32)
  /// @param max_tokens Maximum tokens to generate
  /// @return Metrics from this run
  RunMetrics RunAudioInference (const AudioData &audio, int max_tokens);

  /// @brief Run full benchmark suite.
  /// @param config Benchmark configuration
  /// @param audio_data Optional audio data (required if config.input_type ==
  /// Audio)
  /// @return Aggregated benchmark results
  BenchmarkResults Run (const BenchmarkConfig &config,
                        const AudioData *audio_data = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// @brief Check if LiteRT-LM backend is available (compiled with support).
bool IsLiteRTAvailable ();

} // namespace benchmark
