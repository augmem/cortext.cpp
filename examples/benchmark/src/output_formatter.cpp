#include "include/output_formatter.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace benchmark
{

namespace
{

void
PrintRow (const std::string &metric, double oga_val, double litert_val,
          const std::string &unit, bool higher_better = true)
{
  std::string winner;
  double diff_pct = 0.0;

  if (oga_val > 0.0 && litert_val > 0.0)
    {
      winner = (higher_better ? (oga_val > litert_val) : (oga_val < litert_val))
                   ? "OGA"
                   : "LiteRT";
      diff_pct = std::abs (oga_val - litert_val)
                 / std::max (oga_val, litert_val) * 100.0;
    }
  else
    {
      winner = "N/A";
    }

  std::cout << std::fixed << std::setprecision (2);
  std::cout << std::left << std::setw (26) << metric << std::right
            << std::setw (12) << oga_val << " " << std::left << std::setw (6)
            << unit << std::right << std::setw (12) << litert_val << " "
            << std::left << std::setw (6) << unit << std::right << std::setw (8)
            << winner;

  if (diff_pct > 0.0)
    {
      std::cout << " (" << std::setprecision (1) << diff_pct << "%)";
    }
  std::cout << "\n";
}

void
PrintSeparator (int width = 80)
{
  std::cout << std::string (width, '-') << "\n";
}

void
PrintHeader (const std::string &title, int width = 80)
{
  std::cout << std::string (width, '=') << "\n";
  int padding = (width - static_cast<int> (title.length ())) / 2;
  std::cout << std::string (padding, ' ') << title << "\n";
  std::cout << std::string (width, '=') << "\n";
}

} // namespace

void
PrintResults (const BenchmarkResults &results)
{
  std::cout << "\n";
  PrintHeader ("BENCHMARK RESULTS: " + results.backend_name);
  std::cout << "\n";

  std::cout << "Model:      " << results.model_name << "\n";
  std::cout << "Input:      " << results.input_type << "\n";
  std::cout << "Iterations: " << results.runs.size () << "\n";
  std::cout << "\n";

  std::cout << std::fixed << std::setprecision (2);

  // Performance metrics
  std::cout << "Performance:\n";
  std::cout << "  Decode TPS (mean):   " << results.MeanDecodeTPS ()
            << " tok/s\n";
  std::cout << "  Decode TPS (min):    " << results.MinDecodeTPS ()
            << " tok/s\n";
  std::cout << "  Decode TPS (max):    " << results.MaxDecodeTPS ()
            << " tok/s\n";
  std::cout << "  Decode TPS (stddev): " << results.StdDevDecodeTPS ()
            << " tok/s\n";
  std::cout << "\n";

  // Latency metrics
  std::cout << "Latency:\n";
  std::cout << "  Time to First Token: " << results.MeanTTFT () << " ms\n";
  std::cout << "  TTFT P95:            " << results.P95TTFT () << " ms\n";
  std::cout << "  Total Time (mean):   " << results.MeanTotalTimeMs ()
            << " ms\n";
  std::cout << "\n";

  // Memory metrics
  if (results.MeanPeakMemoryMB () > 0.0)
    {
      std::cout << "Memory:\n";
      std::cout << "  Peak Memory:         " << results.MeanPeakMemoryMB ()
                << " MB\n";
      std::cout << "\n";
    }

  // Individual runs
  std::cout << "Individual Runs:\n";
  for (size_t i = 0; i < results.runs.size (); ++i)
    {
      const auto &run = results.runs[i];
      std::cout << "  Run " << (i + 1) << ": " << run.output_tokens
                << " tokens, " << run.DecodeTPS () << " tok/s, "
                << run.total_time_ms << " ms\n";
    }

  std::cout << std::string (80, '=') << "\n";
}

void
PrintComparisonTable (const BenchmarkResults &oga,
                      const BenchmarkResults &litert)
{
  std::cout << "\n";
  PrintHeader ("BENCHMARK COMPARISON RESULTS");
  std::cout << "\n";

  // Column headers
  std::cout << std::left << std::setw (26) << "Metric" << std::right
            << std::setw (18) << "OGA" << std::setw (18) << "LiteRT-LM"
            << std::setw (16) << "Winner" << "\n";
  PrintSeparator ();

  // Performance metrics
  std::cout << "\nPerformance:\n";
  PrintRow ("  Decode TPS (mean)", oga.MeanDecodeTPS (), litert.MeanDecodeTPS (),
            "tok/s", true);
  PrintRow ("  Decode TPS (min)", oga.MinDecodeTPS (), litert.MinDecodeTPS (),
            "tok/s", true);
  PrintRow ("  Decode TPS (max)", oga.MaxDecodeTPS (), litert.MaxDecodeTPS (),
            "tok/s", true);

  // Latency metrics
  std::cout << "\nLatency:\n";
  PrintRow ("  Time to First Token", oga.MeanTTFT (), litert.MeanTTFT (), "ms",
            false);
  PrintRow ("  TTFT P95", oga.P95TTFT (), litert.P95TTFT (), "ms", false);
  PrintRow ("  Total Time (mean)", oga.MeanTotalTimeMs (),
            litert.MeanTotalTimeMs (), "ms", false);

  // Memory metrics
  if (oga.MeanPeakMemoryMB () > 0.0 || litert.MeanPeakMemoryMB () > 0.0)
    {
      std::cout << "\nMemory:\n";
      PrintRow ("  Peak Memory", oga.MeanPeakMemoryMB (),
                litert.MeanPeakMemoryMB (), "MB", false);
    }

  std::cout << "\n";
  PrintSeparator ();

  // Summary
  double oga_tps = oga.MeanDecodeTPS ();
  double litert_tps = litert.MeanDecodeTPS ();

  if (oga_tps > 0.0 && litert_tps > 0.0)
    {
      double speedup = std::max (oga_tps, litert_tps)
                       / std::min (oga_tps, litert_tps);
      std::string faster = oga_tps > litert_tps ? "OGA" : "LiteRT-LM";

      std::cout << "\nSummary: " << faster << " is " << std::fixed
                << std::setprecision (2) << speedup
                << "x faster in decode TPS\n";
    }

  std::cout << std::string (80, '=') << "\n";
}

std::string
FormatAsJson (const BenchmarkResults &results)
{
  std::ostringstream ss;
  ss << std::fixed << std::setprecision (4);

  ss << "{\n";
  ss << "  \"backend\": \"" << results.backend_name << "\",\n";
  ss << "  \"model\": \"" << results.model_name << "\",\n";
  ss << "  \"input_type\": \"" << results.input_type << "\",\n";
  ss << "  \"metrics\": {\n";
  ss << "    \"decode_tps\": {\n";
  ss << "      \"mean\": " << results.MeanDecodeTPS () << ",\n";
  ss << "      \"min\": " << results.MinDecodeTPS () << ",\n";
  ss << "      \"max\": " << results.MaxDecodeTPS () << ",\n";
  ss << "      \"stddev\": " << results.StdDevDecodeTPS () << "\n";
  ss << "    },\n";
  ss << "    \"ttft_ms\": {\n";
  ss << "      \"mean\": " << results.MeanTTFT () << ",\n";
  ss << "      \"p95\": " << results.P95TTFT () << "\n";
  ss << "    },\n";
  ss << "    \"total_time_ms\": " << results.MeanTotalTimeMs () << ",\n";
  ss << "    \"peak_memory_mb\": " << results.MeanPeakMemoryMB () << "\n";
  ss << "  },\n";
  ss << "  \"runs\": [\n";

  for (size_t i = 0; i < results.runs.size (); ++i)
    {
      const auto &run = results.runs[i];
      ss << "    {\n";
      ss << "      \"decode_tps\": " << run.DecodeTPS () << ",\n";
      ss << "      \"decode_time_ms\": " << run.decode_time_ms << ",\n";
      ss << "      \"prefill_time_ms\": " << run.prefill_time_ms << ",\n";
      ss << "      \"output_tokens\": " << run.output_tokens << "\n";
      ss << "    }";
      if (i < results.runs.size () - 1)
        ss << ",";
      ss << "\n";
    }

  ss << "  ]\n";
  ss << "}";

  return ss.str ();
}

std::string
FormatComparisonAsJson (const BenchmarkResults &oga,
                         const BenchmarkResults &litert)
{
  std::ostringstream ss;
  ss << "{\n";
  ss << "  \"oga\": " << FormatAsJson (oga) << ",\n";
  ss << "  \"litert\": " << FormatAsJson (litert) << "\n";
  ss << "}";
  return ss.str ();
}

} // namespace benchmark
