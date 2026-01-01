#include <cortext/encoder/imagebind.hpp>
#include <cortext/operations/label_utils.hpp>

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
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

} // namespace

int
main (int argc, char **argv)
{
  fs::path labels_path = "data/label_bank/hf_labels.txt";
  fs::path models_dir = "models";
  fs::path out_dir = "data/label_bank";

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
      if (auto v = take ("--labels="))
        {
          labels_path = *v;
        }
      else if (auto v = take ("--models="))
        {
          models_dir = *v;
        }
      else if (auto v = take ("--out-dir="))
        {
          out_dir = *v;
        }
    }

  if (!fs::exists (labels_path))
    {
      std::cerr << "Label list not found: " << labels_path << "\n";
      return 1;
    }

  const fs::path imagebind_dir = ResolveImageBindDir (models_dir);
  cortext::ImageBindEncoder encoder (imagebind_dir.string ());

  std::ifstream in (labels_path);
  if (!in.is_open ())
    {
      std::cerr << "Failed to open labels file: " << labels_path << "\n";
      return 1;
    }

  std::unordered_map<std::string, std::string> labels;
  std::string line;
  while (std::getline (in, line))
    {
      const std::string trimmed = cortext::operations::TrimLabel (line);
      if (trimmed.empty () || trimmed[0] == '#')
        {
          continue;
        }
      const std::string key
          = cortext::operations::NormalizeLabelKey (trimmed);
      if (key.empty ())
        {
          continue;
        }
      if (!labels.count (key))
        {
          labels.emplace (key, trimmed);
        }
    }

  if (labels.empty ())
    {
      std::cerr << "No labels loaded from " << labels_path << "\n";
      return 1;
    }

  fs::create_directories (out_dir);
  const fs::path labels_out = out_dir / "labels.jsonl";
  std::ofstream out (labels_out);
  if (!out.is_open ())
    {
      std::cerr << "Failed to write " << labels_out << "\n";
      return 1;
    }

  int written = 0;
  for (const auto &pair : labels)
    {
      const std::string &key = pair.first;
      const std::string &label = pair.second;
      std::vector<float> embedding;
      encoder.EncodeText (label, embedding);
      if (embedding.empty ())
        {
          continue;
        }
      nlohmann::json row;
      row["label"] = label;
      row["key"] = key;
      row["embedding"] = embedding;
      out << row.dump () << "\n";
      written++;
    }

  nlohmann::json meta;
  meta["generated_by"] = "cortext/tools/label_bank_generator";
  meta["timestamp"] = TimestampUtc ();
  meta["embedding_dim"] = 256;
  meta["labels_file"] = labels_out.filename ().string ();
  meta["source"] = labels_path.string ();
  meta["count"] = written;

  const fs::path meta_out = out_dir / "metadata.json";
  std::ofstream meta_file (meta_out);
  meta_file << meta.dump (2) << "\n";

  std::cout << "Wrote " << written << " label embeddings to "
            << labels_out << "\n";
  std::cout << "Metadata: " << meta_out << "\n";
  return 0;
}
