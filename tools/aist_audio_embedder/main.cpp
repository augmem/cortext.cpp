#include "encoder/text_encoder_factory.hpp"
#include "include/audio_loader.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

std::string
TimestampUtc ()
{
  using clock = std::chrono::system_clock;
  const auto now = clock::now ();
  const std::time_t t = clock::to_time_t (now);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s (&tm, &t);
#else
  gmtime_r (&t, &tm);
#endif
  char buf[64];
  std::strftime (buf, sizeof (buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return std::string (buf);
}

std::string
TrimLine (std::string s)
{
  auto is_space = [] (unsigned char c) { return std::isspace (c) != 0; };
  while (!s.empty () && is_space (static_cast<unsigned char> (s.front ())))
    {
      s.erase (s.begin ());
    }
  while (!s.empty () && is_space (static_cast<unsigned char> (s.back ())))
    {
      s.pop_back ();
    }
  return s;
}

} // namespace

int
main (int argc, char **argv)
{
  fs::path input_list;
  fs::path models_dir = "models";
  fs::path out_path;

  for (int i = 1; i < argc; ++i)
    {
      std::string arg = argv[i];
      auto take = [&arg] (const std::string &prefix)
      -> std::optional<std::string> {
        if (arg.rfind (prefix, 0) == 0)
          {
            return arg.substr (prefix.size ());
          }
        return std::nullopt;
      };
      if (auto v = take ("--input-list="))
        {
          input_list = *v;
        }
      else if (auto v = take ("--models="))
        {
          models_dir = *v;
        }
      else if (auto v = take ("--out="))
        {
          out_path = *v;
        }
    }

  if (input_list.empty () || out_path.empty ())
    {
      std::cerr << "Usage: cortext_aist_audio_embedder --input-list=FILE "
                   "--out=FILE [--models=DIR]\n";
      return 1;
    }
  if (!fs::exists (input_list))
    {
      std::cerr << "Input list not found: " << input_list << "\n";
      return 1;
    }

  auto selection = cortext::internal::CreatePreferredTextEncoder (
      models_dir.string ());
  auto &encoder = *selection.encoder;

  std::ifstream in (input_list);
  if (!in.is_open ())
    {
      std::cerr << "Failed to open input list: " << input_list << "\n";
      return 1;
    }
  fs::create_directories (out_path.parent_path ());
  std::ofstream out (out_path);
  if (!out.is_open ())
    {
      std::cerr << "Failed to open output: " << out_path << "\n";
      return 1;
    }

  int written = 0;
  int failed = 0;
  std::string line;
  while (std::getline (in, line))
    {
      const std::string path = TrimLine (line);
      if (path.empty ())
        {
          continue;
        }
      auto audio = benchmark::LoadAudio (path);
      if (!audio)
        {
          ++failed;
          continue;
        }
      auto normalized = benchmark::NormalizeTo16kMono (*audio);
      std::vector<float> embedding;
      try
        {
          encoder.EncodeAudio (normalized.samples.data (),
                               normalized.samples.size (), embedding);
        }
      catch (const std::exception &)
        {
          ++failed;
          continue;
        }
      if (embedding.empty ())
        {
          ++failed;
          continue;
        }
      nlohmann::json row;
      row["path"] = path;
      row["embedding"] = embedding;
      row["duration_seconds"] = normalized.duration_seconds;
      out << row.dump () << "\n";
      ++written;
    }

  nlohmann::json meta;
  meta["generated_by"] = "cortext/tools/aist_audio_embedder";
  meta["timestamp"] = TimestampUtc ();
  meta["input_list"] = input_list.string ();
  meta["count"] = written;
  meta["failed"] = failed;
  meta["backend"] = selection.backend_name;
  meta["resolved_path"] = selection.resolved_path.string ();
  meta["sample_rate"] = 16000;
  meta["channels"] = 1;
  if (written > 0)
    {
      std::ifstream verify_in (out_path);
      std::string verify_line;
      if (std::getline (verify_in, verify_line))
        {
          auto payload = nlohmann::json::parse (verify_line, nullptr, false);
          if (!payload.is_discarded () && payload.contains ("embedding")
              && payload["embedding"].is_array ())
            {
              meta["embedding_dim"] = payload["embedding"].size ();
            }
        }
    }
  else
    {
      meta["embedding_dim"] = 0;
    }

  const fs::path meta_out = out_path.parent_path () / "metadata.json";
  std::ofstream meta_file (meta_out);
  meta_file << meta.dump (2) << "\n";

  std::cout << "Wrote " << written << " audio embeddings to " << out_path
            << "\n";
  std::cout << "Failed: " << failed << "\n";
  std::cout << "Metadata: " << meta_out << "\n";
  std::cout << "Backend: " << selection.backend_name << " ("
            << selection.resolved_path.string () << ")\n";
  return failed == 0 ? 0 : 2;
}
