#include "encoder/text_encoder_factory.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cctype>
#include <ctime>
#include <cstdlib>
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
    s.erase (s.begin ());
  while (!s.empty () && is_space (static_cast<unsigned char> (s.back ())))
    s.pop_back ();
  return s;
}

} // namespace

int
main (int argc, char **argv)
{
  fs::path input_path;
  fs::path models_dir = "models";
  fs::path out_path;
  std::optional<std::string> backend_override;

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
      if (auto v = take ("--input="))
        {
          input_path = *v;
        }
      else if (auto v = take ("--models="))
        {
          models_dir = *v;
        }
      else if (auto v = take ("--out="))
        {
          out_path = *v;
        }
      else if (auto v = take ("--backend="))
        {
          backend_override = *v;
        }
    }

  if (input_path.empty () || out_path.empty ())
    {
      std::cerr << "Usage: cortext_text_embedder --input=FILE --out=FILE "
                   "[--models=DIR] [--backend=llama.cpp|litert|onnx]\n";
      return 1;
    }
  if (!fs::exists (input_path))
    {
      std::cerr << "Input file not found: " << input_path << "\n";
      return 1;
    }

  if (backend_override.has_value ())
    {
#if defined(_WIN32)
      _putenv_s ("CORTEXT_EMBEDDINGGEMMA_BACKEND", backend_override->c_str ());
#else
      setenv ("CORTEXT_EMBEDDINGGEMMA_BACKEND", backend_override->c_str (), 1);
#endif
    }

  auto selection = cortext::internal::CreatePreferredTextEncoder (
      models_dir.string ());
  auto &encoder = *selection.encoder;

  std::ifstream in (input_path);
  if (!in.is_open ())
    {
      std::cerr << "Failed to open input: " << input_path << "\n";
      return 1;
    }
  fs::create_directories (out_path.parent_path ());
  std::ofstream out (out_path);
  if (!out.is_open ())
    {
      std::cerr << "Failed to open output: " << out_path << "\n";
      return 1;
    }

  std::string line;
  int written = 0;
  while (std::getline (in, line))
    {
      std::string text = TrimLine (line);
      if (text.empty ())
        {
          continue;
        }
      std::vector<float> embedding;
      encoder.EncodeText (text, embedding);
      if (embedding.empty ())
        {
          continue;
        }
      nlohmann::json row;
      row["text"] = text;
      row["embedding"] = embedding;
      out << row.dump () << "\n";
      written++;
    }

  nlohmann::json meta;
  meta["generated_by"] = "cortext/tools/text_embedder";
  meta["timestamp"] = TimestampUtc ();
  meta["input"] = input_path.string ();
  meta["count"] = written;
  meta["backend"] = selection.backend_name;
  meta["resolved_path"] = selection.resolved_path.string ();

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

  std::cout << "Wrote " << written << " embeddings to " << out_path << "\n";
  std::cout << "Metadata: " << meta_out << "\n";
  std::cout << "Backend: " << selection.backend_name << " ("
            << selection.resolved_path.string () << ")\n";
  return 0;
}
