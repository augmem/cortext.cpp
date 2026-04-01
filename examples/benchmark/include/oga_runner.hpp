#pragma once

#include "audio_loader.hpp"
#include "benchmark_config.hpp"
#include "metrics.hpp"

#include <memory>
#include <string>

namespace benchmark
{

/// @brief Benchmark runner for ONNX Runtime GenAI (OGA) backend.
///
/// Uses OGA to run inference with Phi-4 or other OGA-compatible models.
/// Measures tokens per second for text and audio inputs.
class OgaRunner
{
public:
  /// @brief Construct an OGA runner.
  /// @param model_path Path to OGA model directory
  explicit OgaRunner (const std::string &model_path);

  ~OgaRunner ();

  OgaRunner (const OgaRunner &) = delete;
  OgaRunner &operator= (const OgaRunner &) = delete;

  OgaRunner (OgaRunner &&) noexcept;
  OgaRunner &operator= (OgaRunner &&) noexcept;

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

/// @brief Check if OGA backend is available (compiled with support).
bool IsOgaAvailable ();

} // namespace benchmark
