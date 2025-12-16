#include "cortext/generator/gemma_text_generator.hpp"
#include "cortext/generator/gemma_tokenizer.hpp"

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
    "models/gemma-3n",
    "../models/gemma-3n",
    "../../models/gemma-3n",
    "../../../models/gemma-3n",
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

  double
  TokensPerSecond () const
  {
    if (tokens_generated == 0 || Mean () == 0.0)
      return 0.0;
    return (tokens_generated * 1000.0) / Mean ();
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
  if (result.tokens_generated > 0)
    {
      std::cout << "    Tokens: " << result.tokens_generated << "\n";
      std::cout << "    Throughput: " << result.TokensPerSecond ()
                << " tokens/sec\n";
    }
  std::cout << "\n";
}

} // namespace

int
main (int argc, char *argv[])
{
  std::cout << "=== Cortext Generator Benchmark ===\n" << std::flush;

  // Parse arguments
  int num_iterations = 3;
  int max_tokens = 20;

  for (int i = 1; i < argc; ++i)
    {
      std::string arg = argv[i];
      if (arg == "--iterations" && i + 1 < argc)
        {
          num_iterations = std::stoi (argv[++i]);
        }
      else if (arg == "--max-tokens" && i + 1 < argc)
        {
          max_tokens = std::stoi (argv[++i]);
        }
      else if (arg == "--help")
        {
          std::cout << "Usage: " << argv[0] << " [options]\n";
          std::cout << "  --iterations N   Number of iterations (default: 3)\n";
          std::cout << "  --max-tokens N   Max tokens to generate (default: "
                       "20)\n";
          return 0;
        }
    }

  std::cout << "Configuration: iterations=" << num_iterations
            << ", max_tokens=" << max_tokens << "\n"
            << std::flush;

  // Find models
  auto models_dir = FindModelsDir ();
  if (models_dir.empty ())
    {
      std::cerr << "Error: Could not find models directory\n";
      return 1;
    }
  std::cout << "Models: " << models_dir << "\n" << std::flush;

  // Benchmark: Model loading
  std::cout << "Loading model..." << std::flush;
  BenchmarkResult load_result;
  load_result.name = "GemmaTextGenerator construction";

  std::unique_ptr<cortext::GemmaTextGenerator> gen;
  for (int i = 0; i < num_iterations; ++i)
    {
      std::cout << " " << (i + 1) << std::flush;
      auto start = Clock::now ();
      gen = std::make_unique<cortext::GemmaTextGenerator> (models_dir);
      auto end = Clock::now ();
      load_result.times_ms.push_back (
          std::chrono::duration<double, std::milli> (end - start).count ());
    }
  std::cout << " done\n" << std::flush;
  PrintResult (load_result);

  if (!gen || !gen->IsAvailable ())
    {
      std::cerr << "Error: Generator not available (ORT not enabled?)\n";
      return 1;
    }
  std::cout << "Generator available: true\n" << std::flush;

  // Quick profiling: separate embed vs decode time
  std::cout << "\n--- Profiling Embed vs Decode ---\n" << std::flush;
  {
    // Get tokenizer to encode a short prompt
    auto tokens = gen->EosTokenId (); // Just to verify gen works
    std::cout << "EOS token ID: " << tokens << "\n" << std::flush;
  }

  // Benchmark: Text generation (greedy)
  std::cout << "--- Text Generation (Greedy) ---\n" << std::flush;
  BenchmarkResult greedy_result;
  greedy_result.name = "Generate (greedy, " + std::to_string (max_tokens)
                       + " tokens)";

  cortext::GenerationConfig greedy_config;
  greedy_config.max_new_tokens = max_tokens;
  greedy_config.do_sample = false;

  for (int i = 0; i < num_iterations; ++i)
    {
      std::cout << "  Run " << (i + 1) << "..." << std::flush;
      auto start = Clock::now ();
      auto output = gen->Generate ("The capital of France is", greedy_config);
      auto end = Clock::now ();
      auto elapsed
          = std::chrono::duration<double, std::milli> (end - start).count ();
      greedy_result.times_ms.push_back (elapsed);
      greedy_result.tokens_generated = gen->LastTokenCount ();
      std::cout << " " << elapsed << "ms (" << gen->LastTokenCount ()
                << " tokens)\n"
                << std::flush;

      if (i == 0)
        {
          std::cout << "  Output: \"" << output.substr (0, 80) << "...\"\n"
                    << std::flush;
        }
    }
  PrintResult (greedy_result);

  // Benchmark: Text generation (sampling)
  std::cout << "--- Text Generation (Sampling) ---\n";
  BenchmarkResult sample_result;
  sample_result.name = "Generate (sampling, " + std::to_string (max_tokens)
                       + " tokens)";

  cortext::GenerationConfig sample_config;
  sample_config.max_new_tokens = max_tokens;
  sample_config.temperature = 0.7f;
  sample_config.top_k = 40;
  sample_config.do_sample = true;

  for (int i = 0; i < num_iterations; ++i)
    {
      auto start = Clock::now ();
      auto output = gen->Generate ("Once upon a time", sample_config);
      auto end = Clock::now ();
      sample_result.times_ms.push_back (
          std::chrono::duration<double, std::milli> (end - start).count ());
      sample_result.tokens_generated = gen->LastTokenCount ();

      if (i == 0)
        {
          std::cout << "  Sample output: \"" << output.substr (0, 100)
                    << "...\"\n\n";
        }
    }
  PrintResult (sample_result);

  // Benchmark: JSON generation
  std::cout << "--- JSON Generation ---\n";
  BenchmarkResult json_result;
  json_result.name
      = "GenerateJSON (schema, " + std::to_string (max_tokens) + " tokens)";

  nlohmann::json schema
      = { { "type", "object" },
          { "properties",
            { { "answer", { { "type", "string" } } },
              { "confidence", { { "type", "number" } } } } },
          { "required", { "answer" } } };

  for (int i = 0; i < num_iterations; ++i)
    {
      auto start = Clock::now ();
      auto output = gen->GenerateJSON ("What is 2+2? Answer in JSON:", schema,
                                       max_tokens, 0.3f);
      auto end = Clock::now ();
      json_result.times_ms.push_back (
          std::chrono::duration<double, std::milli> (end - start).count ());
      json_result.tokens_generated = gen->LastTokenCount ();

      if (i == 0)
        {
          std::cout << "  Sample output: " << output.dump () << "\n\n";
        }
    }
  PrintResult (json_result);

  // Summary
  std::cout << "=== Summary ===\n";
  std::cout << std::fixed << std::setprecision (2);
  std::cout << "Greedy generation: " << greedy_result.TokensPerSecond ()
            << " tokens/sec\n";
  std::cout << "Sampling generation: " << sample_result.TokensPerSecond ()
            << " tokens/sec\n";
  std::cout << "JSON generation: " << json_result.TokensPerSecond ()
            << " tokens/sec\n";

  return 0;
}
