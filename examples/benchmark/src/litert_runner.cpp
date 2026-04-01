#include "include/litert_runner.hpp"
#include "include/gemma3n_audio.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>

#if defined(BENCHMARK_ENABLE_LITERT)
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#endif

namespace benchmark
{

#if defined(BENCHMARK_ENABLE_LITERT)

// Use fully qualified names to avoid conflict with benchmark::Backend
using LiteRtBackend = litert::lm::Backend;
using litert::lm::CpuConfig;
using litert::lm::Engine;
using litert::lm::EngineSettings;
using litert::lm::InputAudio;
using litert::lm::InputText;
using litert::lm::ModelAssets;
using litert::lm::SessionConfig;

struct LiteRTRunner::Impl
{
  std::unique_ptr<Engine> engine;
  std::string model_path;
  bool available = false;

  explicit Impl (const std::string &path) : model_path (path)
  {
    try
      {
        // Load model assets
        auto model_assets = ModelAssets::Create (path);
        if (!model_assets.ok ())
          {
            std::cerr << "LiteRTRunner: Failed to load model assets: "
                      << model_assets.status () << "\n";
            return;
          }

        // Create engine settings with CPU backend
        auto settings_result = EngineSettings::CreateDefault (
            *std::move (model_assets), LiteRtBackend::CPU, std::nullopt, LiteRtBackend::CPU);
        if (!settings_result.ok ())
          {
            std::cerr << "LiteRTRunner: Failed to create settings: "
                      << settings_result.status () << "\n";
            return;
          }

        // Configure single-threaded execution
        auto &executor_settings
            = settings_result->GetMutableMainExecutorSettings ();
        auto cpu_config_result
            = executor_settings.MutableBackendConfig<CpuConfig> ();
        if (cpu_config_result.ok ())
          {
            CpuConfig cpu_config = *cpu_config_result;
            cpu_config.number_of_threads = 1; // Single-threaded for benchmarking
            executor_settings.SetBackendConfig (cpu_config);
          }

        // Enable benchmarking
        settings_result->GetMutableBenchmarkParams ();

        // Create engine
        auto engine_result = Engine::CreateEngine (*settings_result);
        if (!engine_result.ok ())
          {
            std::cerr << "LiteRTRunner: Failed to create engine: "
                      << engine_result.status () << "\n";
            return;
          }

        engine = *std::move (engine_result);
        available = true;
        std::cerr << "LiteRTRunner: Model loaded successfully (C++ API)\n";
      }
    catch (const std::exception &e)
      {
        std::cerr << "LiteRTRunner: Exception: " << e.what () << "\n";
        available = false;
      }
  }

  RunMetrics
  RunText (const std::string &text, int max_tokens)
  {
    if (!available || !engine)
      throw std::runtime_error ("LiteRTRunner: Model not available");

    RunMetrics metrics;
    BenchmarkTimer total_timer, prefill_timer, decode_timer;

    total_timer.Start ();

    // Create session
    auto session_result
        = engine->CreateSession (SessionConfig::CreateDefault ());
    if (!session_result.ok ())
      {
        throw std::runtime_error ("LiteRTRunner: Failed to create session: "
                                  + session_result.status ().ToString ());
      }
    auto &session = *session_result;

    // Prefill with text input
    prefill_timer.Start ();
    std::vector<std::variant<InputText, litert::lm::InputImage, InputAudio,
                             litert::lm::InputAudioEnd>>
        inputs;
    inputs.emplace_back (InputText (text));
    auto prefill_status = session->RunPrefill (inputs);
    prefill_timer.Stop ();
    metrics.prefill_time_ms = prefill_timer.ElapsedMs ();

    if (!prefill_status.ok ())
      {
        throw std::runtime_error ("LiteRTRunner: Prefill failed: "
                                  + prefill_status.ToString ());
      }

    // Decode loop - check TaskState to detect completion
    decode_timer.Start ();
    metrics.output_tokens = 0;
    bool done = false;
    while (!done && metrics.output_tokens < max_tokens)
      {
        auto decode_result = session->RunDecode ();
        if (!decode_result.ok ())
          break;
        metrics.output_tokens++;
        // Check if generation is complete
        auto task_state = decode_result->GetTaskState ();
        done = litert::lm::IsTaskEndState (task_state);
      }
    decode_timer.Stop ();
    metrics.decode_time_ms = decode_timer.ElapsedMs ();

    total_timer.Stop ();
    metrics.total_time_ms = total_timer.ElapsedMs ();

    return metrics;
  }

  RunMetrics
  RunAudio (const AudioData &audio, int max_tokens)
  {
    if (!available || !engine)
      throw std::runtime_error ("LiteRTRunner: Model not available");

    if (audio.sample_rate != 16000 || audio.num_channels != 1)
      throw std::runtime_error (
          "LiteRTRunner: Audio must be 16kHz mono. Got sample_rate="
          + std::to_string (audio.sample_rate)
          + ", channels=" + std::to_string (audio.num_channels));

    RunMetrics metrics;
    BenchmarkTimer total_timer, prefill_timer, decode_timer;

    total_timer.Start ();

    // Create session with audio modality enabled
    SessionConfig session_config = SessionConfig::CreateDefault ();
    session_config.SetAudioModalityEnabled (true);

    auto session_result = engine->CreateSession (session_config);
    if (!session_result.ok ())
      {
        throw std::runtime_error ("LiteRTRunner: Failed to create session: "
                                  + session_result.status ().ToString ());
      }
    auto &session = *session_result;

    // Compute mel spectrogram features from raw PCM
    std::vector<float> mel_features;
    int64_t num_frames;
    ComputeMelSpectrogram (audio.samples.data (), audio.samples.size (),
                           mel_features, num_frames);

    // Create audio input with mel spectrogram data
    std::string mel_data (reinterpret_cast<const char *> (mel_features.data ()),
                          mel_features.size () * sizeof (float));

    // Prefill with audio + text prompt
    prefill_timer.Start ();
    std::vector<std::variant<InputText, litert::lm::InputImage, InputAudio,
                             litert::lm::InputAudioEnd>>
        inputs;
    inputs.emplace_back (InputAudio (std::move (mel_data)));
    inputs.emplace_back (InputText ("Describe what you hear."));
    auto prefill_status = session->RunPrefill (inputs);
    prefill_timer.Stop ();
    metrics.prefill_time_ms = prefill_timer.ElapsedMs ();

    if (!prefill_status.ok ())
      {
        throw std::runtime_error ("LiteRTRunner: Prefill failed: "
                                  + prefill_status.ToString ());
      }

    // Decode loop - check TaskState to detect completion
    decode_timer.Start ();
    metrics.output_tokens = 0;
    bool done = false;
    while (!done && metrics.output_tokens < max_tokens)
      {
        auto decode_result = session->RunDecode ();
        if (!decode_result.ok ())
          break;
        metrics.output_tokens++;
        // Check if generation is complete
        auto task_state = decode_result->GetTaskState ();
        done = litert::lm::IsTaskEndState (task_state);
      }
    decode_timer.Stop ();
    metrics.decode_time_ms = decode_timer.ElapsedMs ();

    total_timer.Stop ();
    metrics.total_time_ms = total_timer.ElapsedMs ();

    return metrics;
  }
};

LiteRTRunner::LiteRTRunner (const std::string &model_path)
    : impl_ (std::make_unique<Impl> (model_path))
{
}

LiteRTRunner::~LiteRTRunner () = default;

LiteRTRunner::LiteRTRunner (LiteRTRunner &&) noexcept = default;
LiteRTRunner &LiteRTRunner::operator= (LiteRTRunner &&) noexcept = default;

bool
LiteRTRunner::IsAvailable () const
{
  return impl_ && impl_->available;
}

std::string
LiteRTRunner::GetModelName () const
{
  if (!impl_)
    return "";
  std::string path = impl_->model_path;
  size_t last_slash = path.find_last_of ("/\\");
  if (last_slash != std::string::npos)
    return path.substr (last_slash + 1);
  return path;
}

RunMetrics
LiteRTRunner::RunTextInference (const std::string &text, int max_tokens)
{
  return impl_->RunText (text, max_tokens);
}

RunMetrics
LiteRTRunner::RunAudioInference (const AudioData &audio, int max_tokens)
{
  return impl_->RunAudio (audio, max_tokens);
}

BenchmarkResults
LiteRTRunner::Run (const BenchmarkConfig &config, const AudioData *audio_data)
{
  BenchmarkResults results;
  results.backend_name = "LiteRT-LM";
  results.model_name = GetModelName ();
  results.input_type = InputTypeToString (config.input_type);

  if (!IsAvailable ())
    {
      std::cerr
          << "LiteRTRunner: Model not available. Returning empty results.\n";
      return results;
    }

  // Warmup runs
  std::cout << "LiteRT: Performing " << config.warmup_runs
            << " warmup runs..." << std::flush;

  for (int i = 0; i < config.warmup_runs; ++i)
    {
      std::cout << " " << (i + 1) << std::flush;

      if (config.input_type == InputType::Audio && audio_data)
        RunAudioInference (*audio_data, config.max_tokens);
      else
        RunTextInference (config.text_input, config.max_tokens);
    }
  std::cout << " done.\n" << std::flush;

  // Benchmark runs
  std::cout << "LiteRT: Running " << config.iterations
            << " benchmark iterations...\n"
            << std::flush;

  results.runs.reserve (config.iterations);
  for (int i = 0; i < config.iterations; ++i)
    {
      std::cout << "  [" << (i + 1) << "/" << config.iterations << "] "
                << std::flush;

      RunMetrics metrics;
      if (config.input_type == InputType::Audio && audio_data)
        metrics = RunAudioInference (*audio_data, config.max_tokens);
      else
        metrics = RunTextInference (config.text_input, config.max_tokens);

      results.runs.push_back (metrics);

      std::cout << metrics.output_tokens << " tokens, " << std::fixed
                << std::setprecision (2) << metrics.DecodeTPS () << " tok/s\n"
                << std::flush;
    }

  return results;
}

bool
IsLiteRTAvailable ()
{
  return true; // Compiled with support, but model may not be available
}

#else // BENCHMARK_ENABLE_LITERT not defined

struct LiteRTRunner::Impl
{
  std::string model_path;
};

LiteRTRunner::LiteRTRunner (const std::string &model_path)
    : impl_ (std::make_unique<Impl> ())
{
  impl_->model_path = model_path;
}

LiteRTRunner::~LiteRTRunner () = default;

LiteRTRunner::LiteRTRunner (LiteRTRunner &&) noexcept = default;
LiteRTRunner &LiteRTRunner::operator= (LiteRTRunner &&) noexcept = default;

bool
LiteRTRunner::IsAvailable () const
{
  return false;
}

std::string
LiteRTRunner::GetModelName () const
{
  if (!impl_)
    return "";
  std::string path = impl_->model_path;
  size_t last_slash = path.find_last_of ("/\\");
  if (last_slash != std::string::npos)
    return path.substr (last_slash + 1);
  return path;
}

RunMetrics
LiteRTRunner::RunTextInference (const std::string & /*text*/,
                                int /*max_tokens*/)
{
  throw std::runtime_error (
      "LiteRTRunner: LiteRT-LM disabled. Rebuild with BENCHMARK_ENABLE_LITERT");
}

RunMetrics
LiteRTRunner::RunAudioInference (const AudioData & /*audio*/,
                                 int /*max_tokens*/)
{
  throw std::runtime_error (
      "LiteRTRunner: LiteRT-LM disabled. Rebuild with BENCHMARK_ENABLE_LITERT");
}

BenchmarkResults
LiteRTRunner::Run (const BenchmarkConfig &config,
                   const AudioData * /*audio_data*/)
{
  BenchmarkResults results;
  results.backend_name = "LiteRT-LM";
  results.model_name = GetModelName ();
  results.input_type = InputTypeToString (config.input_type);

  std::cerr << "LiteRTRunner: LiteRT-LM support not compiled. "
            << "Rebuild with BENCHMARK_ENABLE_LITERT.\n";

  return results;
}

bool
IsLiteRTAvailable ()
{
  return false;
}

#endif // BENCHMARK_ENABLE_LITERT

} // namespace benchmark
