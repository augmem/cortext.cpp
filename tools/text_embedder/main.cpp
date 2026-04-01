#include <cortext/encoder/imagebind.hpp>

#include <nlohmann/json.hpp>

#include <chrono>
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

fs::path
ResolveImageBindDir (const fs::path &models_dir)
{
  const fs::path direct = models_dir;
  if (fs::exists (direct / "text_encoder.onnx")
      || fs::exists (direct / "text_encoder_int8.onnx"))
    {
      return direct;
    }
  const fs::path nested = models_dir / "imagebind";
  if (fs::exists (nested / "text_encoder.onnx")
      || fs::exists (nested / "text_encoder_int8.onnx"))
    {
      return nested;
    }
  throw std::runtime_error ("ImageBind text encoder not found in models dir");
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
    }

  if (input_path.empty () || out_path.empty ())
    {
      std::cerr << "Usage: cortext_text_embedder --input=FILE --out=FILE "
                   "[--models=DIR]\n";
      return 1;
    }
  if (!fs::exists (input_path))
    {
      std::cerr << "Input file not found: " << input_path << "\n";
      return 1;
    }

  const fs::path imagebind_dir = ResolveImageBindDir (models_dir);
  cortext::ImageBindEncoder encoder (imagebind_dir.string ());

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
  meta["embedding_dim"] = 256;
  meta["input"] = input_path.string ();
  meta["count"] = written;

  const fs::path meta_out = out_path.parent_path () / "metadata.json";
  std::ofstream meta_file (meta_out);
  meta_file << meta.dump (2) << "\n";

  std::cout << "Wrote " << written << " embeddings to " << out_path << "\n";
  std::cout << "Metadata: " << meta_out << "\n";
  return 0;
}
