#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cortext/models/aist_gguf_encoder.hpp>

#include <cmath>
#include <filesystem>
#include <optional>
#include <vector>

namespace
{

std::filesystem::path
RepoRoot ()
{
  std::filesystem::path p = std::filesystem::current_path ();
  for (int i = 0; i < 6; ++i)
    {
      if (std::filesystem::exists (p / "CMakeLists.txt")
          && std::filesystem::exists (p / "models"))
        {
          return p;
        }
      if (!p.has_parent_path ())
        {
          break;
        }
      p = p.parent_path ();
    }
  return std::filesystem::current_path ().parent_path ();
}

std::optional<std::filesystem::path>
AistModelPath ()
{
  return cortext::ResolveAistGgufModelPath (RepoRoot () / "models");
}

double
Norm (const std::vector<float> &v)
{
  double sum = 0.0;
  for (float value : v)
    {
      sum += static_cast<double> (value) * value;
    }
  return std::sqrt (sum);
}

} // namespace

TEST_CASE ("AIST Matryoshka truncation normalizes output",
           "[aist][gguf][unit]")
{
  std::vector<float> values{ 3.0F, 4.0F, 0.0F, 10.0F };
  auto truncated = cortext::TruncateAistMatryoshka (values, 2);
  REQUIRE (truncated.size () == 2);
  const double norm = std::sqrt (static_cast<double> (truncated[0])
                                     * truncated[0]
                                 + static_cast<double> (truncated[1])
                                       * truncated[1]);
  CHECK (std::abs (norm - 1.0) < 1.0e-5);
}

TEST_CASE ("AIST GGUF metadata exposes triembed contract",
           "[aist][gguf][integration]")
{
  const auto model = AistModelPath ();
  if (!model)
    {
      SKIP ("AIST GGUF model not present");
    }

  const auto info = cortext::InspectAistGgufModel (*model);
  CHECK (info.architecture == "triembed");
  CHECK (info.n_embd == 1280);
  CHECK (info.n_embd_out == 1280);
  CHECK (info.has_semantic_vector);
  CHECK (info.runtime_available);
  CHECK (info.runtime_status == "triembed_embedding_runtime_available");
}

TEST_CASE ("AIST GGUF executes native text image and audio kernels",
           "[aist][gguf][integration]")
{
  const auto model = AistModelPath ();
  if (!model)
    {
      SKIP ("AIST GGUF model not present");
    }

  cortext::AistGgufConfig config;
  config.model_path = model->string ();
  config.context_length = 64;
  cortext::AistGgufEncoder encoder (config);
  CHECK (encoder.IsLoaded ());
  REQUIRE (encoder.IsRuntimeAvailable ());

  std::vector<float> text_embedding;
  encoder.EncodeText ("Jared has a huge in-ground pool.", text_embedding);
  REQUIRE (text_embedding.size () == 1280);
  for (float value : text_embedding)
    {
      REQUIRE (std::isfinite (value));
    }
  CHECK (std::abs (Norm (text_embedding) - 1.0) < 1.0e-3);

  constexpr int image_width = 32;
  constexpr int image_height = 24;
  constexpr int image_channels = 3;
  std::vector<std::uint8_t> image (
      static_cast<std::size_t> (image_width * image_height * image_channels),
      0);
  for (int y = 0; y < image_height; ++y)
    {
      for (int x = 0; x < image_width; ++x)
        {
          const std::size_t offset =
              static_cast<std::size_t> ((y * image_width + x)
                                        * image_channels);
          image[offset + 0] = static_cast<std::uint8_t> ((x * 255)
                                                        / image_width);
          image[offset + 1] = static_cast<std::uint8_t> ((y * 255)
                                                        / image_height);
          image[offset + 2] = 127;
        }
    }

  std::vector<float> image_embedding;
  encoder.EncodeImage (image.data (), image_width, image_height,
                       image_channels, image_embedding);
  REQUIRE (image_embedding.size () == 1280);
  CHECK (std::abs (Norm (image_embedding) - 1.0) < 1.0e-3);

  constexpr std::size_t audio_samples = 16000;
  std::vector<float> pcm (audio_samples, 0.0F);
  for (std::size_t i = 0; i < pcm.size (); ++i)
    {
      const double t = static_cast<double> (i) / 16000.0;
      pcm[i] = static_cast<float> (0.15 * std::sin (2.0 * 3.141592653589793
                                                    * 440.0 * t));
    }

  std::vector<float> audio_embedding;
  encoder.EncodeAudio (pcm.data (), pcm.size (), audio_embedding);
  REQUIRE (audio_embedding.size () == 1280);
  CHECK (std::abs (Norm (audio_embedding) - 1.0) < 1.0e-3);
}
