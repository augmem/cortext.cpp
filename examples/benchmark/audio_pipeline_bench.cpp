#include "cortext/audio/sherpa_onnx.hpp"
#include "cortext/encoder/embeddinggemma.hpp"
#include "cortext/encoder/imagebind.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct Options
{
  std::string input_path;
  std::string output_path;
  std::string config_path;
  std::string mode = "both"; // gemma | imagebind | both
  std::size_t limit = 0;
};

void
PrintUsage (const char *argv0)
{
  std::cout
      << "Usage: " << argv0
      << " --input <jsonl> --config <json> --output <jsonl> [--mode gemma|imagebind|both] [--limit N]\n";
}

std::optional<std::string>
NextArg (int &i, int argc, char *argv[])
{
  if (i + 1 >= argc)
    return std::nullopt;
  ++i;
  return std::string (argv[i]);
}

Options
ParseArgs (int argc, char *argv[])
{
  Options opts;
  for (int i = 1; i < argc; ++i)
    {
      std::string arg = argv[i];
      if (arg == "--input")
        {
          auto val = NextArg (i, argc, argv);
          if (!val)
            throw std::runtime_error ("--input requires a value");
          opts.input_path = *val;
        }
      else if (arg == "--output")
        {
          auto val = NextArg (i, argc, argv);
          if (!val)
            throw std::runtime_error ("--output requires a value");
          opts.output_path = *val;
        }
      else if (arg == "--config")
        {
          auto val = NextArg (i, argc, argv);
          if (!val)
            throw std::runtime_error ("--config requires a value");
          opts.config_path = *val;
        }
      else if (arg == "--mode")
        {
          auto val = NextArg (i, argc, argv);
          if (!val)
            throw std::runtime_error ("--mode requires a value");
          opts.mode = *val;
        }
      else if (arg == "--limit")
        {
          auto val = NextArg (i, argc, argv);
          if (!val)
            throw std::runtime_error ("--limit requires a value");
          opts.limit = static_cast<std::size_t> (std::stoul (*val));
        }
      else if (arg == "--help" || arg == "-h")
        {
          PrintUsage (argv[0]);
          std::exit (0);
        }
    }
  if (opts.input_path.empty () || opts.output_path.empty ()
      || opts.config_path.empty ())
    {
      throw std::runtime_error ("Missing required arguments");
    }
  return opts;
}

nlohmann::json
LoadJson (const std::string &path)
{
  std::ifstream in (path);
  if (!in)
    throw std::runtime_error ("Failed to open config: " + path);
  nlohmann::json j;
  in >> j;
  return j;
}

cortext::SherpaOnnxTtsConfig
ParseTtsConfig (const nlohmann::json &j)
{
  if (!j.contains ("tts"))
    throw std::runtime_error ("Config missing 'tts' section");
  const auto &t = j.at ("tts");
  cortext::SherpaOnnxTtsConfig cfg;
  cfg.model_type = t.value ("model_type", "");
  cfg.model = t.value ("model", "");
  cfg.encoder = t.value ("encoder", "");
  cfg.decoder = t.value ("decoder", "");
  cfg.tokens = t.value ("tokens", "");
  cfg.lexicon = t.value ("lexicon", "");
  cfg.data_dir = t.value ("data_dir", "");
  cfg.voices = t.value ("voices", "");
  cfg.vocoder = t.value ("vocoder", "");
  cfg.language = t.value ("language", "");
  cfg.num_threads = t.value ("num_threads", 1);
  cfg.provider = t.value ("provider", "cpu");
  cfg.speed = t.value ("speed", 1.0f);
  return cfg;
}

cortext::SherpaOnnxAsrConfig
ParseAsrConfig (const nlohmann::json &j)
{
  if (!j.contains ("asr"))
    throw std::runtime_error ("Config missing 'asr' section");
  const auto &a = j.at ("asr");
  cortext::SherpaOnnxAsrConfig cfg;
  cfg.model_type = a.value ("model_type", "");
  cfg.model = a.value ("model", "");
  cfg.encoder = a.value ("encoder", "");
  cfg.decoder = a.value ("decoder", "");
  cfg.joiner = a.value ("joiner", "");
  cfg.tokens = a.value ("tokens", "");
  cfg.bpe_vocab = a.value ("bpe_vocab", "");
  cfg.language = a.value ("language", "");
  cfg.sample_rate = a.value ("sample_rate", 16000);
  cfg.num_threads = a.value ("num_threads", 1);
  cfg.provider = a.value ("provider", "cpu");
  return cfg;
}

cortext::EmbeddingGemmaConfig
ParseGemmaConfig (const nlohmann::json &j)
{
  if (!j.contains ("gemma"))
    throw std::runtime_error ("Config missing 'gemma' section");
  const auto &g = j.at ("gemma");
  cortext::EmbeddingGemmaConfig cfg;
  cfg.model_path = g.value ("model", "");
  cfg.tokenizer_path = g.value ("tokenizer", "");
  cfg.max_length = g.value ("max_length", 256);
  cfg.num_threads = g.value ("num_threads", 1);
  return cfg;
}

std::string
ParseImageBindDir (const nlohmann::json &j)
{
  if (!j.contains ("imagebind"))
    throw std::runtime_error ("Config missing 'imagebind' section");
  const auto &ib = j.at ("imagebind");
  return ib.value ("models_dir", "");
}

std::vector<float>
ResampleLinear (const std::vector<float> &input, int32_t in_rate,
                int32_t out_rate)
{
  if (in_rate == out_rate || input.empty ())
    return input;
  const double ratio = static_cast<double> (out_rate)
                       / static_cast<double> (in_rate);
  const std::size_t out_len
      = static_cast<std::size_t> (input.size () * ratio);
  std::vector<float> out;
  out.resize (out_len);
  for (std::size_t i = 0; i < out_len; ++i)
    {
      const double src_pos = static_cast<double> (i) / ratio;
      const std::size_t idx = static_cast<std::size_t> (src_pos);
      const double frac = src_pos - static_cast<double> (idx);
      const float a = input[idx];
      const float b = input[std::min (idx + 1, input.size () - 1)];
      out[i] = static_cast<float> (a + (b - a) * frac);
    }
  return out;
}

struct RunningStat
{
  std::size_t count = 0;
  double total_ms = 0.0;

  void Add (double ms)
  {
    total_ms += ms;
    ++count;
  }

  double Mean () const
  {
    return count == 0 ? 0.0 : total_ms / static_cast<double> (count);
  }
};

} // namespace

int
main (int argc, char *argv[])
{
  try
    {
      auto opts = ParseArgs (argc, argv);
      const auto config = LoadJson (opts.config_path);

      const bool run_gemma = opts.mode == "gemma" || opts.mode == "both";
      const bool run_imagebind
          = opts.mode == "imagebind" || opts.mode == "both";

      const auto tts_config = ParseTtsConfig (config);
      cortext::SherpaOnnxOfflineTts tts (tts_config);
      std::unique_ptr<cortext::SherpaOnnxOfflineAsr> asr;
      std::unique_ptr<cortext::EmbeddingGemmaEncoder> gemma;
      cortext::SherpaOnnxAsrConfig asr_config;
      cortext::EmbeddingGemmaConfig gemma_config;
      std::unique_ptr<cortext::ImageBindEncoder> imagebind;

      if (run_gemma)
        {
          asr_config = ParseAsrConfig (config);
          gemma_config = ParseGemmaConfig (config);
          asr = std::make_unique<cortext::SherpaOnnxOfflineAsr> (asr_config);
          gemma = std::make_unique<cortext::EmbeddingGemmaEncoder> (gemma_config);
        }
      if (run_imagebind)
        {
          const auto ib_dir = ParseImageBindDir (config);
          if (ib_dir.empty ())
            throw std::runtime_error ("imagebind.models_dir is required");
          imagebind = std::make_unique<cortext::ImageBindEncoder> (ib_dir);
        }

      std::ifstream in (opts.input_path);
      if (!in)
        throw std::runtime_error ("Failed to open input: " + opts.input_path);
      std::ofstream out (opts.output_path);
      if (!out)
        throw std::runtime_error ("Failed to open output: " + opts.output_path);

      RunningStat tts_ms;
      RunningStat asr_ms;
      RunningStat gemma_ms;
      RunningStat imagebind_ms;

      std::string line;
      std::size_t processed = 0;
      while (std::getline (in, line))
        {
          if (line.empty ())
            continue;
          const auto row = nlohmann::json::parse (line);
          const std::string id = row.value ("id", "");
          const std::string text = row.value ("text", "");
          if (text.empty ())
            continue;

          auto t0 = std::chrono::steady_clock::now ();
          auto audio = tts.Synthesize (text);
          auto t1 = std::chrono::steady_clock::now ();
          const double tts_item_ms
              = std::chrono::duration<double, std::milli> (t1 - t0).count ();
          tts_ms.Add (tts_item_ms);

          nlohmann::json out_row;
          out_row["id"] = id;
          out_row["text"] = text;
          out_row["tts_ms"] = tts_item_ms;

          if (run_gemma)
            {
              auto resampled = ResampleLinear (audio.samples,
                                              audio.sample_rate,
                                              static_cast<int32_t> (asr_config.sample_rate));
              auto t_asr0 = std::chrono::steady_clock::now ();
              const std::string transcript
                  = asr->Transcribe (resampled.data (), resampled.size (),
                                     asr_config.sample_rate);
              auto t_asr1 = std::chrono::steady_clock::now ();
              const double asr_item_ms
                  = std::chrono::duration<double, std::milli> (t_asr1 - t_asr0).count ();
              asr_ms.Add (asr_item_ms);

              std::vector<float> emb;
              auto t_g0 = std::chrono::steady_clock::now ();
              gemma->EncodeText (transcript, emb);
              auto t_g1 = std::chrono::steady_clock::now ();
              const double gemma_item_ms
                  = std::chrono::duration<double, std::milli> (t_g1 - t_g0).count ();
              gemma_ms.Add (gemma_item_ms);

              out_row["asr_text"] = transcript;
              out_row["asr_ms"] = asr_item_ms;
              out_row["gemma_ms"] = gemma_item_ms;
              out_row["embedding_gemma"] = emb;
            }

          if (run_imagebind)
            {
              auto resampled = ResampleLinear (audio.samples,
                                              audio.sample_rate,
                                              16000);
              std::vector<float> emb;
              auto t_ib0 = std::chrono::steady_clock::now ();
              imagebind->EncodeAudio (resampled.data (), resampled.size (), emb);
              auto t_ib1 = std::chrono::steady_clock::now ();
              const double imagebind_item_ms
                  = std::chrono::duration<double, std::milli> (t_ib1 - t_ib0).count ();
              imagebind_ms.Add (imagebind_item_ms);

              out_row["imagebind_ms"] = imagebind_item_ms;
              out_row["embedding_imagebind"] = emb;
            }

          out << out_row.dump () << "\n";
          ++processed;
          if (opts.limit > 0 && processed >= opts.limit)
            break;
        }

      std::cout << "Processed " << processed << " items\n";
      std::cout << "TTS mean ms: " << tts_ms.Mean () << "\n";
      if (run_gemma)
        {
          std::cout << "ASR mean ms: " << asr_ms.Mean () << "\n";
          std::cout << "Gemma mean ms: " << gemma_ms.Mean () << "\n";
        }
      if (run_imagebind)
        {
          std::cout << "ImageBind mean ms: " << imagebind_ms.Mean () << "\n";
        }
    }
  catch (const std::exception &e)
    {
      std::cerr << "Error: " << e.what () << "\n";
      PrintUsage (argv[0]);
      return 1;
    }
  return 0;
}
