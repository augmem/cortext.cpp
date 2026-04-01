#pragma once

#include "stream_usage.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>

namespace chat {

enum class UsageAccuracy {
  Missing,
  Estimated,
  Exact,
  Mixed,
};

struct UsageAccumulator {
  std::int64_t prompt_tokens = 0;
  std::int64_t completion_tokens = 0;
  std::int64_t total_tokens = 0;
  bool has_exact = false;
  bool has_estimate = false;
};

inline std::int64_t EstimateTokenCountFromChars(std::size_t chars) {
  if (chars == 0) {
    return 0;
  }
  return std::max<std::int64_t>(
      1, static_cast<std::int64_t>(std::ceil(static_cast<double>(chars) / 4.0)));
}

inline void AccumulateUsage(UsageAccumulator& accumulator,
                            const std::optional<StreamingUsage>& exact_usage,
                            std::size_t prompt_chars,
                            std::size_t completion_chars) {
  if (exact_usage.has_value()) {
    accumulator.prompt_tokens += exact_usage->prompt_tokens;
    accumulator.completion_tokens += exact_usage->completion_tokens;
    accumulator.total_tokens += exact_usage->total_tokens;
    accumulator.has_exact = true;
    return;
  }

  const std::int64_t prompt_tokens = EstimateTokenCountFromChars(prompt_chars);
  const std::int64_t completion_tokens
      = EstimateTokenCountFromChars(completion_chars);
  accumulator.prompt_tokens += prompt_tokens;
  accumulator.completion_tokens += completion_tokens;
  accumulator.total_tokens += prompt_tokens + completion_tokens;
  accumulator.has_estimate = true;
}

inline UsageAccuracy GetUsageAccuracy(const UsageAccumulator& accumulator) {
  if (accumulator.has_exact && accumulator.has_estimate) {
    return UsageAccuracy::Mixed;
  }
  if (accumulator.has_exact) {
    return UsageAccuracy::Exact;
  }
  if (accumulator.has_estimate) {
    return UsageAccuracy::Estimated;
  }
  return UsageAccuracy::Missing;
}

inline const char* UsageAccuracyLabel(UsageAccuracy accuracy) {
  switch (accuracy) {
    case UsageAccuracy::Estimated:
      return "estimated";
    case UsageAccuracy::Exact:
      return "exact";
    case UsageAccuracy::Mixed:
      return "mixed";
    case UsageAccuracy::Missing:
    default:
      return "missing";
  }
}

inline double ComputeInterruptRate(int probe_count, int interrupt_count) {
  if (probe_count <= 0) {
    return 0.0;
  }
  return static_cast<double>(interrupt_count)
         / static_cast<double>(probe_count);
}

struct ResponseMetricsSample {
  std::uint64_t timestamp_ms = 0;
  StreamingUsage usage;
  UsageAccuracy usage_accuracy = UsageAccuracy::Missing;
  double response_wall_ms = 0.0;
  double phase1_total_ms = 0.0;
  double phase3_total_ms = 0.0;
  double retrieval_latency_ms = 0.0;
  double processing_latency_ms = 0.0;
  double interrupt_rate = 0.0;
  int restart_count = 0;
  int probe_count = 0;
  int interrupt_count = 0;
  bool had_stream_error = false;
  std::string error_message;
};

struct ConsolidationMetricsSample {
  std::uint64_t timestamp_ms = 0;
  double duration_ms = 0.0;
  bool completed = false;
  bool cancelled = false;
  bool had_error = false;
  int working_memory_size = 0;
  std::string error_message;
};

struct MetricsState {
  mutable std::mutex mu;
  std::deque<ResponseMetricsSample> response_history;
  std::deque<ConsolidationMetricsSample> consolidation_history;
  std::optional<ResponseMetricsSample> latest_response;
  std::optional<ConsolidationMetricsSample> latest_consolidation;
  std::size_t total_responses = 0;
  std::size_t total_consolidations = 0;
};

}  // namespace chat
