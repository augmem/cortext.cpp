#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace benchmark
{

/// @brief Metrics from a single benchmark run.
struct RunMetrics
{
  // Timing
  double load_time_ms = 0.0;    ///< Model loading time
  double prefill_time_ms = 0.0; ///< Time to process input (first token latency)
  double decode_time_ms = 0.0;  ///< Time for token generation
  double total_time_ms = 0.0;   ///< Total inference time

  // Token counts
  int input_tokens = 0;  ///< Number of input tokens
  int output_tokens = 0; ///< Number of generated tokens

  // Memory (if available)
  size_t peak_memory_bytes = 0;  ///< Peak memory usage
  size_t model_memory_bytes = 0; ///< Memory used by model

  /// @brief Calculate tokens per second for decode phase.
  /// TPS = output_tokens / (decode_time_ms / 1000.0)
  double
  DecodeTPS () const
  {
    if (decode_time_ms <= 0.0 || output_tokens <= 0)
      return 0.0;
    return output_tokens / (decode_time_ms / 1000.0);
  }

  /// @brief Calculate tokens per second for total inference.
  double
  TotalTPS () const
  {
    if (total_time_ms <= 0.0 || output_tokens <= 0)
      return 0.0;
    return output_tokens / (total_time_ms / 1000.0);
  }

  /// @brief Time to first token (TTFT).
  double
  TimeToFirstTokenMs () const
  {
    return prefill_time_ms;
  }

  /// @brief Average time per token (decode only).
  double
  AvgTokenTimeMs () const
  {
    if (output_tokens <= 0)
      return 0.0;
    return decode_time_ms / output_tokens;
  }
};

/// @brief Aggregated benchmark results across multiple runs.
struct BenchmarkResults
{
  std::string backend_name;
  std::string model_name;
  std::string input_type; ///< "text" or "audio"

  std::vector<RunMetrics> runs;

  /// @brief Mean decode tokens per second.
  double
  MeanDecodeTPS () const
  {
    if (runs.empty ())
      return 0.0;
    double sum = 0.0;
    for (const auto &run : runs)
      sum += run.DecodeTPS ();
    return sum / static_cast<double> (runs.size ());
  }

  /// @brief Minimum decode tokens per second.
  double
  MinDecodeTPS () const
  {
    if (runs.empty ())
      return 0.0;
    double min_val = runs[0].DecodeTPS ();
    for (const auto &run : runs)
      min_val = std::min (min_val, run.DecodeTPS ());
    return min_val;
  }

  /// @brief Maximum decode tokens per second.
  double
  MaxDecodeTPS () const
  {
    if (runs.empty ())
      return 0.0;
    double max_val = runs[0].DecodeTPS ();
    for (const auto &run : runs)
      max_val = std::max (max_val, run.DecodeTPS ());
    return max_val;
  }

  /// @brief Standard deviation of decode TPS.
  double
  StdDevDecodeTPS () const
  {
    if (runs.size () < 2)
      return 0.0;
    double mean = MeanDecodeTPS ();
    double variance = 0.0;
    for (const auto &run : runs)
      {
        double diff = run.DecodeTPS () - mean;
        variance += diff * diff;
      }
    return std::sqrt (variance / static_cast<double> (runs.size () - 1));
  }

  /// @brief Mean time to first token (TTFT).
  double
  MeanTTFT () const
  {
    if (runs.empty ())
      return 0.0;
    double sum = 0.0;
    for (const auto &run : runs)
      sum += run.TimeToFirstTokenMs ();
    return sum / static_cast<double> (runs.size ());
  }

  /// @brief 95th percentile time to first token.
  double
  P95TTFT () const
  {
    if (runs.empty ())
      return 0.0;
    std::vector<double> ttfts;
    ttfts.reserve (runs.size ());
    for (const auto &run : runs)
      ttfts.push_back (run.TimeToFirstTokenMs ());
    std::sort (ttfts.begin (), ttfts.end ());
    size_t p95_idx = static_cast<size_t> (
        static_cast<double> (runs.size ()) * 0.95);
    return ttfts[std::min (p95_idx, ttfts.size () - 1)];
  }

  /// @brief Mean total inference time.
  double
  MeanTotalTimeMs () const
  {
    if (runs.empty ())
      return 0.0;
    double sum = 0.0;
    for (const auto &run : runs)
      sum += run.total_time_ms;
    return sum / static_cast<double> (runs.size ());
  }

  /// @brief Mean peak memory in megabytes.
  double
  MeanPeakMemoryMB () const
  {
    if (runs.empty ())
      return 0.0;
    double sum = 0.0;
    for (const auto &run : runs)
      sum += static_cast<double> (run.peak_memory_bytes) / (1024.0 * 1024.0);
    return sum / static_cast<double> (runs.size ());
  }
};

/// @brief High-precision timer for benchmarking.
class BenchmarkTimer
{
public:
  void
  Start ()
  {
    start_ = Clock::now ();
  }

  void
  Stop ()
  {
    end_ = Clock::now ();
  }

  double
  ElapsedMs () const
  {
    return std::chrono::duration<double, std::milli> (end_ - start_).count ();
  }

  double
  ElapsedSeconds () const
  {
    return std::chrono::duration<double> (end_ - start_).count ();
  }

private:
  using Clock = std::chrono::high_resolution_clock;
  Clock::time_point start_;
  Clock::time_point end_;
};

/// @brief RAII timer that records elapsed time to a variable.
class ScopedTimer
{
public:
  explicit ScopedTimer (double &out_ms) : out_ms_ (out_ms)
  {
    start_ = std::chrono::high_resolution_clock::now ();
  }

  ~ScopedTimer ()
  {
    auto end = std::chrono::high_resolution_clock::now ();
    out_ms_ = std::chrono::duration<double, std::milli> (end - start_).count ();
  }

  ScopedTimer (const ScopedTimer &) = delete;
  ScopedTimer &operator= (const ScopedTimer &) = delete;

private:
  double &out_ms_;
  std::chrono::high_resolution_clock::time_point start_;
};

} // namespace benchmark
