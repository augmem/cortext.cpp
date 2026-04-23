#pragma once

#include "../../../src/encoder/text_encoder_factory.hpp"

#include <Eigen/Dense>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cortext::benchmark
{

class EmbeddingGemmaBenchEncoder final : public Encoder
{
public:
  explicit EmbeddingGemmaBenchEncoder (std::string models_dir = "models")
      : models_dir_ (std::move (models_dir))
  {
    auto selection = internal::CreatePreferredTextEncoder (models_dir_);
    backend_name_ = selection.backend_name;
    resolved_model_path_ = selection.resolved_path;
    encoder_ = std::move (selection.encoder);
    if (!encoder_)
      {
        throw std::runtime_error (
            "Preferred benchmark text encoder could not be constructed under "
            + models_dir_);
      }
  }

  void
  EncodeText (const std::string &text,
              std::vector<float> &out_embedding) override
  {
    const auto &cached = CacheLookup (text);
    out_embedding = cached;
  }

  void
  EncodeAudio (const float * /*pcm*/, std::size_t /*num_samples*/,
               std::vector<float> & /*out_embedding*/) override
  {
    throw std::runtime_error (
        "EmbeddingGemma benchmark encoder does not support audio inputs");
  }

  void
  EncodeImage (const std::uint8_t * /*data*/, int /*width*/, int /*height*/,
               int /*channels*/,
               std::vector<float> & /*out_embedding*/) override
  {
    throw std::runtime_error (
        "EmbeddingGemma benchmark encoder does not support image inputs");
  }

  Eigen::VectorXf
  EncodeTextEigen (const std::string &text)
  {
    const auto &cached = CacheLookup (text);
    Eigen::VectorXf out (static_cast<Eigen::Index> (cached.size ()));
    for (std::size_t i = 0; i < cached.size (); ++i)
      {
        out (static_cast<Eigen::Index> (i)) = cached[i];
      }
    return out;
  }

  int
  Dimension ()
  {
    return static_cast<int> (CacheLookup ("benchmark dimension probe").size ());
  }

  const std::string &
  backend_name () const
  {
    return backend_name_;
  }

  const std::filesystem::path &
  resolved_model_path () const
  {
    return resolved_model_path_;
  }

private:
  const std::vector<float> &
  CacheLookup (const std::string &text)
  {
    auto it = cache_.find (text);
    if (it != cache_.end ())
      {
        return it->second;
      }

    std::vector<float> embedding;
    encoder_->EncodeText (text, embedding);
    if (embedding.empty ())
      {
        throw std::runtime_error (
            "Benchmark text encoder produced an empty embedding");
      }

    auto inserted = cache_.emplace (text, std::move (embedding));
    return inserted.first->second;
  }

  std::string models_dir_;
  std::string backend_name_;
  std::filesystem::path resolved_model_path_;
  std::unique_ptr<Encoder> encoder_;
  std::unordered_map<std::string, std::vector<float>> cache_;
};

inline std::string
ParseModelsDirArg (int argc, char **argv,
                   const std::string &default_models_dir = "models")
{
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg (argv[i]);
      constexpr const char *prefix = "--models=";
      if (arg.rfind (prefix, 0) == 0)
        {
          return arg.substr (std::char_traits<char>::length (prefix));
        }
    }
  return default_models_dir;
}

} // namespace cortext::benchmark
