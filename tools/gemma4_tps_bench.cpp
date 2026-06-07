#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;

double
SecondsSince (Clock::time_point start)
{
  return std::chrono::duration<double> (Clock::now () - start).count ();
}

double
Mean (const std::vector<double> &values)
{
  if (values.empty ())
    {
      return 0.0;
    }
  return std::accumulate (values.begin (), values.end (), 0.0)
         / static_cast<double> (values.size ());
}

litert::lm::Backend
ParseBackend (const std::string &backend)
{
  if (backend == "gpu")
    {
      return litert::lm::Backend::GPU;
    }
  return litert::lm::Backend::CPU;
}
} // namespace

int
main (int argc, char **argv)
{
  const std::string model_path
      = argc > 1 ? argv[1]
                 : "models/gemma4-e2b-litert/gemma-4-E2B-it.litertlm";
  const std::string backend_name = argc > 2 ? argv[2] : "cpu";
  const int max_output_tokens = argc > 3 ? std::atoi (argv[3]) : 128;
  const int iterations = argc > 4 ? std::atoi (argv[4]) : 5;

  const std::string prompt
      = "Write exactly 120 words about why local on-device memory systems need "
        "fast small language models. Use plain prose and do not use bullets.";

  const auto load_start = Clock::now ();
  auto model_assets = litert::lm::ModelAssets::Create (model_path);
  if (!model_assets.ok ())
    {
      std::cerr << "ModelAssets::Create failed: "
                << model_assets.status ().ToString () << "\n";
      return 1;
    }

  auto settings = litert::lm::EngineSettings::CreateDefault (
      *std::move (model_assets), ParseBackend (backend_name));
  if (!settings.ok ())
    {
      std::cerr << "EngineSettings::CreateDefault failed: "
                << settings.status ().ToString () << "\n";
      return 1;
    }

  settings->GetMutableMainExecutorSettings ().SetMaxNumTokens (4096);
  auto &bench_params = settings->GetMutableBenchmarkParams ();
  bench_params.set_num_decode_tokens (max_output_tokens);
  bench_params.set_num_prefill_tokens (0);

  auto engine_result
      = litert::lm::EngineFactory::CreateDefault (*std::move (settings));
  if (!engine_result.ok ())
    {
      std::cerr << "EngineFactory::CreateDefault failed: "
                << engine_result.status ().ToString () << "\n";
      return 1;
    }
  std::unique_ptr<litert::lm::Engine> engine = std::move (*engine_result);
  const double load_seconds = SecondsSince (load_start);

  auto prompt_tokens
      = const_cast<litert::lm::Tokenizer &> (engine->GetTokenizer ())
            .TextToTokenIds (prompt);
  const size_t prompt_token_count = prompt_tokens.ok () ? prompt_tokens->size () : 0;

  std::vector<double> prefill_seconds;
  std::vector<double> decode_seconds;
  std::vector<double> wall_seconds;
  std::vector<double> output_tokens;
  std::vector<double> benchmark_decode_tps;
  std::vector<double> benchmark_prefill_tps;

  for (int i = 0; i < iterations; ++i)
    {
      litert::lm::SessionConfig session_config
          = litert::lm::SessionConfig::CreateDefault ();
      session_config.SetMaxOutputTokens (max_output_tokens);
      session_config.SetSamplerBackend (litert::lm::Backend::CPU);
      auto &sampler = session_config.GetMutableSamplerParams ();
      sampler.set_type (litert::lm::proto::SamplerParameters::TOP_P);
      sampler.set_k (1);
      sampler.set_p (1.0f);
      sampler.set_temperature (1.0f);
      sampler.set_seed (42);

      auto session_result = engine->CreateSession (session_config);
      if (!session_result.ok ())
        {
          std::cerr << "CreateSession failed: "
                    << session_result.status ().ToString () << "\n";
          return 1;
        }
      std::unique_ptr<litert::lm::Engine::Session> session
          = std::move (*session_result);

      std::vector<litert::lm::InputData> inputs;
      inputs.emplace_back (litert::lm::InputText (prompt));

      const auto turn_start = Clock::now ();
      const auto prefill_start = Clock::now ();
      auto prefill_status = session->RunPrefill (inputs);
      const double prefill_s = SecondsSince (prefill_start);
      if (!prefill_status.ok ())
        {
          std::cerr << "RunPrefill failed: " << prefill_status.ToString ()
                    << "\n";
          return 1;
        }

      auto decode_config = litert::lm::DecodeConfig::CreateDefault ();
      decode_config.SetMaxOutputTokens (max_output_tokens);

      const auto decode_start = Clock::now ();
      auto response = session->RunDecode (decode_config);
      const double decode_s = SecondsSince (decode_start);
      const double wall_s = SecondsSince (turn_start);
      if (!response.ok ())
        {
          std::cerr << "RunDecode failed: " << response.status ().ToString ()
                    << "\n";
          return 1;
        }

      std::string response_text;
      if (!response->GetTexts ().empty ())
        {
          response_text = response->GetTexts ().front ();
        }
      auto response_tokens
          = const_cast<litert::lm::Tokenizer &> (engine->GetTokenizer ())
                .TextToTokenIds (response_text);
      const size_t token_count
          = response_tokens.ok () ? response_tokens->size () : 0;

      auto bench = session->GetBenchmarkInfo ();
      double bench_decode = 0.0;
      double bench_prefill = 0.0;
      if (bench.ok ())
        {
          const auto decode_turns = bench->GetTotalDecodeTurns ();
          const auto prefill_turns = bench->GetTotalPrefillTurns ();
          if (decode_turns > 0)
            {
              bench_decode = bench->GetDecodeTokensPerSec (
                  static_cast<int> (decode_turns - 1));
            }
          if (prefill_turns > 0)
            {
              bench_prefill = bench->GetPrefillTokensPerSec (
                  static_cast<int> (prefill_turns - 1));
            }
        }

      prefill_seconds.push_back (prefill_s);
      decode_seconds.push_back (decode_s);
      wall_seconds.push_back (wall_s);
      output_tokens.push_back (static_cast<double> (token_count));
      benchmark_decode_tps.push_back (bench_decode);
      benchmark_prefill_tps.push_back (bench_prefill);

      std::cout << "iter=" << i << " output_tokens=" << token_count
                << " prefill_s=" << prefill_s << " decode_s=" << decode_s
                << " measured_decode_tok_s="
                << (decode_s > 0.0
                        ? static_cast<double> (token_count) / decode_s
                        : 0.0)
                << " litert_decode_tok_s=" << bench_decode
                << " litert_prefill_tok_s=" << bench_prefill << "\n";
    }

  std::cout << "summary"
            << " model=" << model_path << " backend=" << backend_name
            << " prompt_tokens=" << prompt_token_count
            << " max_output_tokens=" << max_output_tokens
            << " iterations=" << iterations
            << " load_s=" << load_seconds
            << " mean_output_tokens=" << Mean (output_tokens)
            << " mean_prefill_s=" << Mean (prefill_seconds)
            << " mean_decode_s=" << Mean (decode_seconds)
            << " mean_wall_s=" << Mean (wall_seconds)
            << " mean_measured_decode_tok_s="
            << (Mean (decode_seconds) > 0.0
                    ? Mean (output_tokens) / Mean (decode_seconds)
                    : 0.0)
            << " mean_litert_decode_tok_s=" << Mean (benchmark_decode_tps)
            << " mean_litert_prefill_tok_s=" << Mean (benchmark_prefill_tps)
            << "\n";

  return 0;
}
