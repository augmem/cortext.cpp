#include "include/audio_loader.hpp"
#include "include/benchmark_config.hpp"
#include "include/litert_runner.hpp"
#include "include/metrics.hpp"
#include "include/oga_runner.hpp"
#include "include/output_formatter.hpp"

#include <filesystem>
#include <iostream>
#include <memory>

using namespace benchmark;

namespace
{

std::string
FindModelsDir (const std::string &model_name)
{
  std::vector<std::string> paths = {
    "models/" + model_name,
    "../models/" + model_name,
    "../../models/" + model_name,
    "../../../models/" + model_name,
  };

  for (const auto &path : paths)
    {
      if (std::filesystem::exists (path))
        return path;
    }
  return "";
}

void
PrintConfiguration (const BenchmarkConfig &config)
{
  std::cout << "\nConfiguration:\n";
  std::cout << "  Backend:    "
            << (config.compare_backends ? "Both (comparison mode)"
                                        : BackendToString (config.backend))
            << "\n";
  std::cout << "  Input:      " << InputTypeToString (config.input_type)
            << "\n";
  std::cout << "  Iterations: " << config.iterations << "\n";
  std::cout << "  Warmup:     " << config.warmup_runs << "\n";
  std::cout << "  Max tokens: " << config.max_tokens << "\n";

  if (!config.model_path.empty ())
    std::cout << "  Model:      " << config.model_path << "\n";
  if (!config.oga_model_path.empty ())
    std::cout << "  OGA Model:  " << config.oga_model_path << "\n";
  if (!config.litert_model_path.empty ())
    std::cout << "  LiteRT Model: " << config.litert_model_path << "\n";

  if (config.input_type == InputType::Audio)
    std::cout << "  Audio file: " << config.audio_file << "\n";
  else
    std::cout << "  Text:       \""
              << config.text_input.substr (
                     0, std::min (size_t (50), config.text_input.size ()))
              << "...\"\n";

  std::cout << "\n";
}

} // namespace

int
main (int argc, char *argv[])
{
  std::cout << "=== Cortext Multi-Backend Benchmark ===\n";
  std::cout << "Comparing LiteRT-LM vs ONNX Runtime GenAI\n";

  // Parse arguments
  auto config_result = ParseArguments (argc, argv);
  if (std::holds_alternative<std::string> (config_result))
    {
      std::cerr << "Error: " << std::get<std::string> (config_result) << "\n\n";
      PrintUsage (argv[0]);
      return 1;
    }

  auto config = std::get<BenchmarkConfig> (config_result);

  // Auto-detect model paths if not specified
  if (config.model_path.empty () && config.oga_model_path.empty ())
    {
      config.oga_model_path = FindModelsDir ("phi4-mm-cpu");
      if (config.oga_model_path.empty ())
        {
          std::cerr << "Warning: Could not find phi4-mm-cpu model.\n";
          std::cerr << "Download with: huggingface-cli download "
                       "lokinfey/Phi-4-Multimodal-ONNX-INT4-CPU --local-dir "
                       "models/phi4-mm-cpu\n";
        }
    }

  if (config.model_path.empty () && config.litert_model_path.empty ())
    {
      config.litert_model_path = FindModelsDir ("gemma3n-e2b-litert");
      if (config.litert_model_path.empty ())
        {
          std::cerr << "Warning: Could not find gemma3n-e2b-litert model.\n";
          std::cerr << "Download with: huggingface-cli download "
                       "google/gemma-3n-E2B-it-litert-lm --local-dir "
                       "models/gemma3n-e2b-litert\n";
        }
    }

  PrintConfiguration (config);

  // Load audio if needed
  std::optional<AudioData> audio_data;
  if (config.input_type == InputType::Audio)
    {
      std::cout << "Loading audio file: " << config.audio_file << "\n";
      audio_data = LoadAudio (config.audio_file);
      if (!audio_data)
        {
          std::cerr << "Error: Failed to load audio file: " << config.audio_file
                    << "\n";
          return 1;
        }

      std::cout << "  Sample rate: " << audio_data->sample_rate << " Hz\n";
      std::cout << "  Channels:    " << audio_data->num_channels << "\n";
      std::cout << "  Duration:    " << audio_data->duration_seconds << " s\n";

      // Normalize to 16kHz mono
      if (audio_data->sample_rate != 16000 || audio_data->num_channels != 1)
        {
          std::cout << "  Normalizing to 16kHz mono...\n";
          *audio_data = NormalizeTo16kMono (*audio_data);
        }
      std::cout << "\n";
    }

  const AudioData *audio_ptr
      = audio_data.has_value () ? &audio_data.value () : nullptr;

  try
    {
      if (config.compare_backends)
        {
          // Run both backends for comparison
          BenchmarkResults oga_results;
          BenchmarkResults litert_results;

          // OGA backend
          std::string oga_path = config.GetModelPath (Backend::OnnxRuntimeGenAI);
          if (!oga_path.empty () && IsOgaAvailable ())
            {
              std::cout << "--- Running OGA Backend ---\n";
              std::cout << "Loading model: " << oga_path << "\n";

              OgaRunner oga_runner (oga_path);
              if (oga_runner.IsAvailable ())
                {
                  oga_results = oga_runner.Run (config, audio_ptr);
                }
              else
                {
                  std::cerr << "OGA: Model not available\n";
                }
            }
          else
            {
              std::cerr << "OGA: Backend not available or no model path\n";
            }

          std::cout << "\n";

          // LiteRT backend
          std::string litert_path = config.GetModelPath (Backend::LiteRTLM);
          if (!litert_path.empty () && IsLiteRTAvailable ())
            {
              std::cout << "--- Running LiteRT-LM Backend ---\n";
              std::cout << "Loading model: " << litert_path << "\n";

              LiteRTRunner litert_runner (litert_path);
              if (litert_runner.IsAvailable ())
                {
                  litert_results = litert_runner.Run (config, audio_ptr);
                }
              else
                {
                  std::cerr << "LiteRT: Model not available\n";
                }
            }
          else
            {
              std::cerr
                  << "LiteRT: Backend not available or no model path\n";
            }

          // Print comparison
          if (!oga_results.runs.empty () || !litert_results.runs.empty ())
            {
              if (config.json_output)
                {
                  std::cout << FormatComparisonAsJson (oga_results,
                                                        litert_results)
                            << "\n";
                }
              else
                {
                  PrintComparisonTable (oga_results, litert_results);
                }
            }
          else
            {
              std::cerr << "\nNo benchmark results to compare.\n";
              return 1;
            }
        }
      else
        {
          // Single backend run
          BenchmarkResults results;
          std::string model_path = config.GetModelPath (config.backend);

          if (model_path.empty ())
            {
              std::cerr << "Error: No model path for "
                        << BackendToString (config.backend) << " backend\n";
              return 1;
            }

          if (config.backend == Backend::OnnxRuntimeGenAI)
            {
              if (!IsOgaAvailable ())
                {
                  std::cerr << "Error: OGA backend not available. "
                            << "Rebuild without CORTEXT_DISABLE_OGA\n";
                  return 1;
                }

              std::cout << "--- Running OGA Backend ---\n";
              std::cout << "Loading model: " << model_path << "\n";

              OgaRunner runner (model_path);
              if (!runner.IsAvailable ())
                {
                  std::cerr << "Error: OGA model not available\n";
                  return 1;
                }

              results = runner.Run (config, audio_ptr);
            }
          else // LiteRTLM
            {
              if (!IsLiteRTAvailable ())
                {
                  std::cerr << "Error: LiteRT-LM backend not available. "
                            << "Rebuild with BENCHMARK_ENABLE_LITERT\n";
                  return 1;
                }

              std::cout << "--- Running LiteRT-LM Backend ---\n";
              std::cout << "Loading model: " << model_path << "\n";

              LiteRTRunner runner (model_path);
              if (!runner.IsAvailable ())
                {
                  std::cerr << "Error: LiteRT-LM model not available\n";
                  return 1;
                }

              results = runner.Run (config, audio_ptr);
            }

          // Print results
          if (config.json_output)
            std::cout << FormatAsJson (results) << "\n";
          else
            PrintResults (results);
        }
    }
  catch (const std::exception &e)
    {
      std::cerr << "Benchmark failed: " << e.what () << "\n";
      return 1;
    }

  return 0;
}
