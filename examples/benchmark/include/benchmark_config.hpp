#pragma once

#include <iostream>
#include <string>
#include <variant>
#include <vector>

namespace benchmark
{

/// @brief Inference backend selection.
enum class Backend
{
  OnnxRuntimeGenAI, ///< ONNX Runtime GenAI (OGA) - Phi-4
  LiteRTLM          ///< Google LiteRT-LM - Gemma 3n
};

/// @brief Input type for benchmarking.
enum class InputType
{
  Text,
  Audio
};

/// @brief Benchmark configuration from CLI arguments.
struct BenchmarkConfig
{
  // Backend selection
  Backend backend = Backend::OnnxRuntimeGenAI;

  // Model paths
  std::string model_path;        ///< Primary model path
  std::string oga_model_path;    ///< OGA-specific model path
  std::string litert_model_path; ///< LiteRT-specific model path

  // Input configuration
  InputType input_type = InputType::Text;
  std::string text_input; ///< Text prompt for benchmarking
  std::string audio_file; ///< Path to audio file (.wav, .pcm)

  // Benchmark parameters
  int iterations = 5;    ///< Number of benchmark iterations
  int warmup_runs = 2;   ///< Number of warmup runs (discarded)
  int max_tokens = 256;  ///< Maximum tokens to generate

  // Comparison mode
  bool compare_backends = false; ///< Run both backends and compare

  // Output options
  bool verbose = false;       ///< Verbose output
  bool json_output = false;   ///< Output results as JSON
  bool report_memory = true;  ///< Report memory usage

  /// @brief Validate configuration.
  bool
  IsValid () const
  {
    if (model_path.empty () && oga_model_path.empty ()
        && litert_model_path.empty ())
      return false;
    if (input_type == InputType::Audio && audio_file.empty ())
      return false;
    if (iterations <= 0)
      return false;
    if (warmup_runs < 0)
      return false;
    if (max_tokens <= 0)
      return false;
    return true;
  }

  /// @brief Get validation error message.
  std::string
  ValidationError () const
  {
    if (model_path.empty () && oga_model_path.empty ()
        && litert_model_path.empty ())
      return "No model path specified. Use --model, --oga-model, or "
             "--litert-model";
    if (input_type == InputType::Audio && audio_file.empty ())
      return "Audio input selected but no --audio-file specified";
    if (iterations <= 0)
      return "--iterations must be positive";
    if (warmup_runs < 0)
      return "--warmup cannot be negative";
    if (max_tokens <= 0)
      return "--max-tokens must be positive";
    return "";
  }

  /// @brief Get effective model path for given backend.
  std::string
  GetModelPath (Backend b) const
  {
    if (b == Backend::OnnxRuntimeGenAI && !oga_model_path.empty ())
      return oga_model_path;
    if (b == Backend::LiteRTLM && !litert_model_path.empty ())
      return litert_model_path;
    return model_path;
  }
};

/// @brief Print usage information.
inline void
PrintUsage (const char *program_name)
{
  std::cout
      << "Usage: " << program_name << " [options]\n\n"
      << "Benchmark LiteRT-LM vs ONNX Runtime GenAI for LLM inference.\n\n"
      << "Backend Selection:\n"
      << "  --backend <oga|litert>      Select inference backend (default: "
         "oga)\n"
      << "  --compare                   Compare both backends side-by-side\n\n"
      << "Model Paths:\n"
      << "  --model <path>              Model directory (used for both "
         "backends)\n"
      << "  --oga-model <path>          OGA-specific model path (Phi-4)\n"
      << "  --litert-model <path>       LiteRT-specific model path (Gemma "
         "3n)\n\n"
      << "Input Options:\n"
      << "  --text <prompt>             Text input for benchmark\n"
      << "  --audio-file <path>         Audio file (.wav, .pcm)\n\n"
      << "Benchmark Parameters:\n"
      << "  --iterations <N>            Benchmark iterations (default: 5)\n"
      << "  --warmup <N>                Warmup runs (default: 2)\n"
      << "  --max-tokens <N>            Max tokens to generate (default: "
         "256)\n\n"
      << "Output Options:\n"
      << "  --verbose, -v               Verbose output\n"
      << "  --json                      Output results as JSON\n"
      << "  --no-memory                 Skip memory reporting\n"
      << "  --help, -h                  Show this help\n\n"
      << "Examples:\n"
      << "  " << program_name
      << " --model models/phi4-mm-cpu --text \"Hello world\"\n"
      << "  " << program_name
      << " --compare --oga-model models/phi4-mm-cpu --litert-model "
         "models/gemma3n-e2b-litert\n"
      << "  " << program_name
      << " --backend litert --model models/gemma3n-e2b-litert --audio-file "
         "test.wav\n";
}

/// @brief Parse command-line arguments.
/// @return Config on success, error message on failure
inline std::variant<BenchmarkConfig, std::string>
ParseArguments (int argc, char *argv[])
{
  BenchmarkConfig config;

  // Default text input
  config.text_input
      = "The quick brown fox jumps over the lazy dog. This is a test "
        "prompt for benchmarking language model performance.";

  for (int i = 1; i < argc; ++i)
    {
      std::string arg = argv[i];

      if (arg == "--help" || arg == "-h")
        {
          PrintUsage (argv[0]);
          std::exit (0);
        }
      else if (arg == "--backend" && i + 1 < argc)
        {
          std::string backend = argv[++i];
          if (backend == "oga" || backend == "onnxruntime-genai")
            config.backend = Backend::OnnxRuntimeGenAI;
          else if (backend == "litert" || backend == "litert-lm")
            config.backend = Backend::LiteRTLM;
          else
            return "Invalid backend: " + backend
                   + ". Use 'oga' or 'litert'.";
        }
      else if (arg == "--model" && i + 1 < argc)
        {
          config.model_path = argv[++i];
        }
      else if (arg == "--oga-model" && i + 1 < argc)
        {
          config.oga_model_path = argv[++i];
        }
      else if (arg == "--litert-model" && i + 1 < argc)
        {
          config.litert_model_path = argv[++i];
        }
      else if (arg == "--audio-file" && i + 1 < argc)
        {
          config.audio_file = argv[++i];
          config.input_type = InputType::Audio;
        }
      else if (arg == "--text" && i + 1 < argc)
        {
          config.text_input = argv[++i];
          config.input_type = InputType::Text;
        }
      else if (arg == "--iterations" && i + 1 < argc)
        {
          try
            {
              config.iterations = std::stoi (argv[++i]);
            }
          catch (...)
            {
              return "Invalid --iterations value";
            }
        }
      else if (arg == "--warmup" && i + 1 < argc)
        {
          try
            {
              config.warmup_runs = std::stoi (argv[++i]);
            }
          catch (...)
            {
              return "Invalid --warmup value";
            }
        }
      else if (arg == "--max-tokens" && i + 1 < argc)
        {
          try
            {
              config.max_tokens = std::stoi (argv[++i]);
            }
          catch (...)
            {
              return "Invalid --max-tokens value";
            }
        }
      else if (arg == "--compare")
        {
          config.compare_backends = true;
        }
      else if (arg == "--verbose" || arg == "-v")
        {
          config.verbose = true;
        }
      else if (arg == "--json")
        {
          config.json_output = true;
        }
      else if (arg == "--no-memory")
        {
          config.report_memory = false;
        }
      else if (arg[0] == '-')
        {
          return "Unknown option: " + arg;
        }
    }

  if (!config.IsValid ())
    return config.ValidationError ();

  return config;
}

/// @brief Convert Backend enum to string.
inline std::string
BackendToString (Backend b)
{
  switch (b)
    {
    case Backend::OnnxRuntimeGenAI:
      return "OnnxRuntime-GenAI";
    case Backend::LiteRTLM:
      return "LiteRT-LM";
    }
  return "Unknown";
}

/// @brief Convert InputType enum to string.
inline std::string
InputTypeToString (InputType t)
{
  switch (t)
    {
    case InputType::Text:
      return "text";
    case InputType::Audio:
      return "audio";
    }
  return "unknown";
}

} // namespace benchmark
