#include <cortext/models/embedding_model_pin.hpp>
#include <cortext/operations/label_utils.hpp>

#include "encoder/text_encoder_factory.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
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

  auto encoder_selection
      = cortext::internal::CreatePreferredTextEncoder (models_dir.string ());
  auto &encoder = *encoder_selection.encoder;

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
      // Match the runtime's label space (process_extraction_results.cpp
      // EncodeLabelEmbedding): Matryoshka-truncate to the 256-d working dim
      // and L2-renormalize. Full-dim encoders (AIST-87M emits 1280) would
      // otherwise produce rows the sqlite builder and runtime reject.
      constexpr size_t kLabelBankDim = 256;
      if (embedding.size () > kLabelBankDim)
        {
          embedding.resize (kLabelBankDim);
          double norm_sq = 0.0;
          for (float value : embedding)
            {
              norm_sq += static_cast<double> (value) * value;
            }
          const double norm = std::sqrt (norm_sq);
          if (norm > 1e-12 && std::isfinite (norm))
            {
              const float inv_norm = static_cast<float> (1.0 / norm);
              for (float &value : embedding)
                {
                  value *= inv_norm;
                }
            }
        }
      else if (embedding.size () != kLabelBankDim)
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
  meta["embedding_model_pin"] = cortext::models::ComputeEmbeddingModelPin (
      encoder_selection.backend_name, encoder_selection.resolved_path, 256);
  meta["labels_file"] = labels_out.filename ().string ();
  meta["source"] = labels_path.string ();
  meta["encoder_backend"] = encoder_selection.backend_name;
  meta["encoder_model"] = encoder_selection.resolved_path.string ();
  meta["count"] = written;

  const fs::path meta_out = out_dir / "metadata.json";
  std::ofstream meta_file (meta_out);
  meta_file << meta.dump (2) << "\n";

  std::cout << "Wrote " << written << " label embeddings to "
            << labels_out << "\n";
  std::cout << "Metadata: " << meta_out << "\n";
  return 0;
}
