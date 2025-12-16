#pragma once

#include "include/metrics.hpp"
#include <string>

namespace benchmark
{

/// @brief Print single backend benchmark results.
void PrintResults (const BenchmarkResults &results);

/// @brief Print comparison table for two backends.
void PrintComparisonTable (const BenchmarkResults &oga_results,
                           const BenchmarkResults &litert_results);

/// @brief Format results as JSON string.
std::string FormatAsJson (const BenchmarkResults &results);

/// @brief Format comparison as JSON string.
std::string FormatComparisonAsJson (const BenchmarkResults &oga_results,
                                     const BenchmarkResults &litert_results);

} // namespace benchmark
