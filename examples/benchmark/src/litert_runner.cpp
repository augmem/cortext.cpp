#include "include/litert_runner.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>

#if defined(BENCHMARK_ENABLE_LITERT)
// LiteRT-LM headers would go here when available
// #include "runtime/engine/engine.h"
// #include "runtime/engine/model_assets.h"
// #include "runtime/session/session.h"
#endif

namespace benchmark
{

#if defined(BENCHMARK_ENABLE_LITERT)

struct LiteRTRunner::Impl
{
  // LiteRT-LM engine and session state would go here
  // std::unique_ptr<litert::lm::Engine> engine;
  std::string model_path;
  bool available = false;

  explicit Impl (const std::string &path) : model_path (path)
  {
    // TODO: Implement LiteRT-LM model loading when SDK is available
    //
    // Expected implementation:
    // try {
    //   auto model_assets = litert::lm::ModelAssets::Create(path);
    //   auto engine_settings = litert::lm::EngineSettings::CreateDefault(
    //       *model_assets, litert::lm::Backend::CPU);
    //   auto engine_result = litert::lm::Engine::CreateEngine(engine_settings);
    //   if (engine_result.ok()) {
    //     engine = std::move(*engine_result);
    //     available = true;
    //   }
    // } catch (const std::exception& e) {
    //   std::cerr << "LiteRTRunner: Failed to load model: " << e.what() <<
    //   "\n"; available = false;
    // }

    std::cerr << "LiteRTRunner: LiteRT-LM integration pending. Model path: "
              << path << "\n";
    available = false;
  }

  RunMetrics
  RunText (const std::string &text, int max_tokens)
  {
    if (!available)
      throw std::runtime_error ("LiteRTRunner: Model not available");

    RunMetrics metrics;
    BenchmarkTimer total_timer, prefill_timer, decode_timer;

    total_timer.Start ();

    // TODO: Implement LiteRT-LM text inference
    //
    // Expected implementation:
    // auto session = engine->CreateSession(session_config);
    //
    // prefill_timer.Start();
    // session->RunPrefill({litert::lm::InputText(text)});
    // metrics.output_tokens = 1;
    // prefill_timer.Stop();
    // metrics.prefill_time_ms = prefill_timer.ElapsedMs();
    //
    // decode_timer.Start();
    // while (!session->IsDone() && metrics.output_tokens < max_tokens) {
    //   auto token = session->RunDecode();
    //   metrics.output_tokens++;
    // }
    // decode_timer.Stop();
    // metrics.decode_time_ms = decode_timer.ElapsedMs();

    total_timer.Stop ();
    metrics.total_time_ms = total_timer.ElapsedMs ();

    return metrics;
  }

  RunMetrics
  RunAudio (const AudioData &audio, int max_tokens)
  {
    if (!available)
      throw std::runtime_error ("LiteRTRunner: Model not available");

    if (audio.sample_rate != 16000 || audio.num_channels != 1)
      throw std::runtime_error (
          "LiteRTRunner: Audio must be 16kHz mono. Got sample_rate="
          + std::to_string (audio.sample_rate)
          + ", channels=" + std::to_string (audio.num_channels));

    RunMetrics metrics;
    BenchmarkTimer total_timer, prefill_timer, decode_timer;

    total_timer.Start ();

    // TODO: Implement LiteRT-LM audio inference
    //
    // Expected implementation using Conversation API:
    // auto conversation = engine->CreateConversation();
    //
    // std::vector<litert::lm::ContentPart> parts;
    // parts.push_back({.type = "audio", .audio_data = audio.samples});
    // parts.push_back({.type = "text", .text = "Describe what you hear."});
    //
    // prefill_timer.Start();
    // auto response = conversation->SendMessage(parts);
    // prefill_timer.Stop();
    // metrics.prefill_time_ms = prefill_timer.ElapsedMs();
    //
    // metrics.output_tokens = response.token_count;
    // decode_timer.Start();
    // decode_timer.Stop();
    // metrics.decode_time_ms = decode_timer.ElapsedMs();

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
  if (config.verbose)
    std::cout << "LiteRT: Performing " << config.warmup_runs
              << " warmup runs...\n";

  for (int i = 0; i < config.warmup_runs; ++i)
    {
      if (config.verbose)
        std::cout << "  Warmup " << (i + 1) << "/" << config.warmup_runs
                  << "\n";

      if (config.input_type == InputType::Audio && audio_data)
        RunAudioInference (*audio_data, config.max_tokens);
      else
        RunTextInference (config.text_input, config.max_tokens);
    }

  // Benchmark runs
  if (config.verbose)
    std::cout << "LiteRT: Running " << config.iterations
              << " benchmark iterations...\n";

  results.runs.reserve (config.iterations);
  for (int i = 0; i < config.iterations; ++i)
    {
      if (config.verbose)
        std::cout << "  Iteration " << (i + 1) << "/" << config.iterations;

      RunMetrics metrics;
      if (config.input_type == InputType::Audio && audio_data)
        metrics = RunAudioInference (*audio_data, config.max_tokens);
      else
        metrics = RunTextInference (config.text_input, config.max_tokens);

      results.runs.push_back (metrics);

      if (config.verbose)
        std::cout << " - " << metrics.output_tokens << " tokens, "
                  << std::fixed << std::setprecision (2) << metrics.DecodeTPS ()
                  << " tok/s\n";
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
