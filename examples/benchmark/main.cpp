#include "cortext/extractor/phi4_extractor.hpp"
#include "cortext/summarizer/phi4_summarizer.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::milliseconds;

namespace
{

std::string
FindModelsDir ()
{
  std::vector<std::string> paths = {
    "models/phi4-mm-cpu",
    "../models/phi4-mm-cpu",
    "../../models/phi4-mm-cpu",
    "../../../models/phi4-mm-cpu",
  };

  for (const auto &path : paths)
    {
      if (std::filesystem::exists (path))
        {
          return path;
        }
    }
  return "";
}

struct BenchmarkResult
{
  std::string name;
  std::vector<double> times_ms;
  int tokens_generated = 0;

  double
  Mean () const
  {
    if (times_ms.empty ())
      return 0.0;
    return std::accumulate (times_ms.begin (), times_ms.end (), 0.0)
           / times_ms.size ();
  }

  double
  Min () const
  {
    if (times_ms.empty ())
      return 0.0;
    return *std::min_element (times_ms.begin (), times_ms.end ());
  }

  double
  Max () const
  {
    if (times_ms.empty ())
      return 0.0;
    return *std::max_element (times_ms.begin (), times_ms.end ());
  }
};

void
PrintResult (const BenchmarkResult &result)
{
  std::cout << std::fixed << std::setprecision (2);
  std::cout << "  " << result.name << ":\n";
  std::cout << "    Mean: " << result.Mean () << " ms\n";
  std::cout << "    Min:  " << result.Min () << " ms\n";
  std::cout << "    Max:  " << result.Max () << " ms\n";
  std::cout << "\n";
}

} // namespace

int
main (int argc, char *argv[])
{
  std::cout << "=== Cortext Phi-4 Benchmark ===\n" << std::flush;

  // Parse arguments
  int num_iterations = 3;

  for (int i = 1; i < argc; ++i)
    {
      std::string arg = argv[i];
      if (arg == "--iterations" && i + 1 < argc)
        {
          num_iterations = std::stoi (argv[++i]);
        }
      else if (arg == "--help")
        {
          std::cout << "Usage: " << argv[0] << " [options]\n";
          std::cout << "  --iterations N   Number of iterations (default: 3)\n";
          return 0;
        }
    }

  std::cout << "Configuration: iterations=" << num_iterations << "\n"
            << std::flush;

  // Find models
  auto models_dir = FindModelsDir ();
  if (models_dir.empty ())
    {
      std::cerr << "Error: Could not find models directory (phi4-mm-cpu)\n";
      std::cerr << "Download with: huggingface-cli download "
                   "lokinfey/Phi-4-Multimodal-ONNX-INT4-CPU --local-dir "
                   "models/phi4-mm-cpu\n";
      return 1;
    }
  std::cout << "Models: " << models_dir << "\n" << std::flush;

  // Check if OGA is enabled
#if defined(CORTEXT_DISABLE_OGA)
  std::cerr << "Error: OGA disabled. Rebuild without CORTEXT_DISABLE_OGA\n";
  return 1;
#endif

  // Benchmark: Model loading
  std::cout << "\n--- Loading Models ---\n" << std::flush;

  std::cout << "Loading Phi4Extractor..." << std::flush;
  BenchmarkResult extractor_load;
  extractor_load.name = "Phi4Extractor construction";

  std::unique_ptr<cortext::Phi4Extractor> extractor;
  for (int i = 0; i < num_iterations; ++i)
    {
      std::cout << " " << (i + 1) << std::flush;
      auto start = Clock::now ();
      extractor = std::make_unique<cortext::Phi4Extractor> (models_dir);
      auto end = Clock::now ();
      extractor_load.times_ms.push_back (
          std::chrono::duration<double, std::milli> (end - start).count ());
    }
  std::cout << " done\n" << std::flush;
  PrintResult (extractor_load);

  std::cout << "Loading Phi4Summarizer..." << std::flush;
  BenchmarkResult summarizer_load;
  summarizer_load.name = "Phi4Summarizer construction";

  std::unique_ptr<cortext::Phi4Summarizer> summarizer;
  for (int i = 0; i < num_iterations; ++i)
    {
      std::cout << " " << (i + 1) << std::flush;
      auto start = Clock::now ();
      summarizer = std::make_unique<cortext::Phi4Summarizer> (models_dir);
      auto end = Clock::now ();
      summarizer_load.times_ms.push_back (
          std::chrono::duration<double, std::milli> (end - start).count ());
    }
  std::cout << " done\n" << std::flush;
  PrintResult (summarizer_load);

  if (!extractor || !extractor->IsAvailable ())
    {
      std::cerr << "Error: Extractor not available\n";
      return 1;
    }
  if (!summarizer || !summarizer->IsAvailable ())
    {
      std::cerr << "Error: Summarizer not available\n";
      return 1;
    }
  std::cout << "Both components available: true\n" << std::flush;

  // Benchmark: Text extraction
  std::cout << "\n--- Text Extraction ---\n" << std::flush;
  BenchmarkResult extract_result;
  extract_result.name = "ExtractFromText";

  nlohmann::json extraction_schema = nlohmann::json::parse (R"({
    "type": "object",
    "properties": {
      "entities": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "type": {"type": "string"}
          }
        }
      },
      "relations": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {"type": "string"}
          }
        }
      }
    }
  })");

  std::string test_text
      = "Apple Inc. was founded by Steve Jobs in Cupertino. "
        "The company created the iPhone and employs Tim Cook as CEO.";

  for (int i = 0; i < num_iterations; ++i)
    {
      std::cout << "  Run " << (i + 1) << "..." << std::flush;
      auto start = Clock::now ();
      auto result = extractor->ExtractFromText (test_text, extraction_schema);
      auto end = Clock::now ();
      auto elapsed
          = std::chrono::duration<double, std::milli> (end - start).count ();
      extract_result.times_ms.push_back (elapsed);
      std::cout << " " << elapsed << "ms\n" << std::flush;

      if (i == 0)
        {
          std::cout << "  Entities found: " << result.entities.size () << "\n";
          std::cout << "  Relations found: " << result.relations.size ()
                    << "\n";
        }
    }
  PrintResult (extract_result);

  // Benchmark: Text summarization
  std::cout << "\n--- Text Summarization ---\n" << std::flush;
  BenchmarkResult summarize_result;
  summarize_result.name = "SummarizeTexts";

  std::vector<std::string> texts = {
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Climate change is affecting ecosystems worldwide.",
  };

  for (int i = 0; i < num_iterations; ++i)
    {
      std::cout << "  Run " << (i + 1) << "..." << std::flush;
      auto start = Clock::now ();
      auto result = summarizer->SummarizeTexts (texts);
      auto end = Clock::now ();
      auto elapsed
          = std::chrono::duration<double, std::milli> (end - start).count ();
      summarize_result.times_ms.push_back (elapsed);
      std::cout << " " << elapsed << "ms\n" << std::flush;

      if (i == 0)
        {
          std::cout << "  Summary: \""
                    << result.substr (0, std::min (size_t (80), result.size ()))
                    << "...\"\n";
        }
    }
  PrintResult (summarize_result);

  // Summary
  std::cout << "=== Summary ===\n";
  std::cout << std::fixed << std::setprecision (2);
  std::cout << "Extractor load: " << extractor_load.Mean () << " ms\n";
  std::cout << "Summarizer load: " << summarizer_load.Mean () << " ms\n";
  std::cout << "Text extraction: " << extract_result.Mean () << " ms\n";
  std::cout << "Text summarization: " << summarize_result.Mean () << " ms\n";

  return 0;
}
