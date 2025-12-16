#include "include/oga_runner.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>

#if !defined(CORTEXT_DISABLE_OGA)
#include <ort_genai.h>
#endif

namespace benchmark
{

#if !defined(CORTEXT_DISABLE_OGA)

struct OgaRunner::Impl
{
  std::unique_ptr<OgaModel> model;
  std::unique_ptr<OgaTokenizer> tokenizer;
  std::unique_ptr<OgaMultiModalProcessor> processor;
  std::string model_path;
  bool available = false;

  explicit Impl (const std::string &path) : model_path (path)
  {
    try
      {
        model = OgaModel::Create (path.c_str ());
        tokenizer = OgaTokenizer::Create (*model);
        processor = OgaMultiModalProcessor::Create (*model);
        available = true;
      }
    catch (const std::exception &e)
      {
        std::cerr << "OgaRunner: Failed to load model: " << e.what () << "\n";
        available = false;
      }
  }

  RunMetrics
  RunText (const std::string &text, int max_tokens)
  {
    if (!available)
      throw std::runtime_error ("OgaRunner: Model not available");

    RunMetrics metrics;
    BenchmarkTimer total_timer, prefill_timer, decode_timer;

    total_timer.Start ();

    // Create generator params
    auto params = OgaGeneratorParams::Create (*model);
    params->SetSearchOption ("max_length", max_tokens);
    params->SetSearchOption ("temperature", 0.7);

    // Tokenize input
    auto sequences = OgaSequences::Create ();
    tokenizer->Encode (text.c_str (), *sequences);
    metrics.input_tokens = static_cast<int> (sequences->SequenceCount (0));

    // Prefill phase
    prefill_timer.Start ();
    auto generator = OgaGenerator::Create (*model, *params);
    generator->AppendTokenSequences (*sequences);

    // Generate first token (end of prefill)
    if (!generator->IsDone ())
      {
        generator->GenerateNextToken ();
        metrics.output_tokens = 1;
      }
    prefill_timer.Stop ();
    metrics.prefill_time_ms = prefill_timer.ElapsedMs ();

    // Decode phase (remaining tokens)
    decode_timer.Start ();
    while (!generator->IsDone ())
      {
        generator->GenerateNextToken ();
        metrics.output_tokens++;
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
    if (!available)
      throw std::runtime_error ("OgaRunner: Model not available");

    // Ensure audio is 16kHz mono
    if (audio.sample_rate != 16000 || audio.num_channels != 1)
      throw std::runtime_error (
          "OgaRunner: Audio must be 16kHz mono. Got sample_rate="
          + std::to_string (audio.sample_rate)
          + ", channels=" + std::to_string (audio.num_channels));

    RunMetrics metrics;
    BenchmarkTimer total_timer, prefill_timer, decode_timer;

    total_timer.Start ();

    // Create generator params
    auto params = OgaGeneratorParams::Create (*model);
    params->SetSearchOption ("max_length", max_tokens);
    params->SetSearchOption ("temperature", 0.7);

    // Load audio data
    const void *audio_data[] = { audio.samples.data () };
    size_t audio_sizes[] = { audio.samples.size () * sizeof (float) };
    auto audios = OgaAudios::Load (audio_data, audio_sizes, 1);

    // Build prompt with audio
    std::string prompt
        = "<|audio|>Please describe what you hear.<|end|><|assistant|>";

    // Prefill phase (includes audio processing)
    prefill_timer.Start ();
    auto inputs = processor->ProcessAudios (prompt.c_str (), audios.get ());
    auto generator = OgaGenerator::Create (*model, *params);
    generator->SetInputs (*inputs);

    // Generate first token (end of prefill)
    if (!generator->IsDone ())
      {
        generator->GenerateNextToken ();
        metrics.output_tokens = 1;
      }
    prefill_timer.Stop ();
    metrics.prefill_time_ms = prefill_timer.ElapsedMs ();

    // Decode phase (remaining tokens)
    decode_timer.Start ();
    while (!generator->IsDone ())
      {
        generator->GenerateNextToken ();
        metrics.output_tokens++;
      }
    decode_timer.Stop ();
    metrics.decode_time_ms = decode_timer.ElapsedMs ();

    total_timer.Stop ();
    metrics.total_time_ms = total_timer.ElapsedMs ();

    return metrics;
  }
};

OgaRunner::OgaRunner (const std::string &model_path)
    : impl_ (std::make_unique<Impl> (model_path))
{
}

OgaRunner::~OgaRunner () = default;

OgaRunner::OgaRunner (OgaRunner &&) noexcept = default;
OgaRunner &OgaRunner::operator= (OgaRunner &&) noexcept = default;

bool
OgaRunner::IsAvailable () const
{
  return impl_ && impl_->available;
}

std::string
OgaRunner::GetModelName () const
{
  if (!impl_)
    return "";
  // Extract model name from path
  std::string path = impl_->model_path;
  size_t last_slash = path.find_last_of ("/\\");
  if (last_slash != std::string::npos)
    return path.substr (last_slash + 1);
  return path;
}

RunMetrics
OgaRunner::RunTextInference (const std::string &text, int max_tokens)
{
  return impl_->RunText (text, max_tokens);
}

RunMetrics
OgaRunner::RunAudioInference (const AudioData &audio, int max_tokens)
{
  return impl_->RunAudio (audio, max_tokens);
}

BenchmarkResults
OgaRunner::Run (const BenchmarkConfig &config, const AudioData *audio_data)
{
  BenchmarkResults results;
  results.backend_name = "OnnxRuntime-GenAI";
  results.model_name = GetModelName ();
  results.input_type = InputTypeToString (config.input_type);

  // Warmup runs
  if (config.verbose)
    std::cout << "OGA: Performing " << config.warmup_runs
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
    std::cout << "OGA: Running " << config.iterations
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
IsOgaAvailable ()
{
  return true;
}

#else // CORTEXT_DISABLE_OGA

struct OgaRunner::Impl
{
};

OgaRunner::OgaRunner (const std::string & /*model_path*/)
    : impl_ (std::make_unique<Impl> ())
{
}

OgaRunner::~OgaRunner () = default;

OgaRunner::OgaRunner (OgaRunner &&) noexcept = default;
OgaRunner &OgaRunner::operator= (OgaRunner &&) noexcept = default;

bool
OgaRunner::IsAvailable () const
{
  return false;
}

std::string
OgaRunner::GetModelName () const
{
  return "";
}

RunMetrics
OgaRunner::RunTextInference (const std::string & /*text*/, int /*max_tokens*/)
{
  throw std::runtime_error (
      "OgaRunner: OGA disabled. Rebuild without CORTEXT_DISABLE_OGA");
}

RunMetrics
OgaRunner::RunAudioInference (const AudioData & /*audio*/, int /*max_tokens*/)
{
  throw std::runtime_error (
      "OgaRunner: OGA disabled. Rebuild without CORTEXT_DISABLE_OGA");
}

BenchmarkResults
OgaRunner::Run (const BenchmarkConfig & /*config*/,
                const AudioData * /*audio_data*/)
{
  throw std::runtime_error (
      "OgaRunner: OGA disabled. Rebuild without CORTEXT_DISABLE_OGA");
}

bool
IsOgaAvailable ()
{
  return false;
}

#endif // CORTEXT_DISABLE_OGA

} // namespace benchmark
