// Compiled only when models/AIST-87M-GGUF is present (see CMakeLists.txt).
// Once built, every assertion is unconditional: resolution failure is a
// test failure, not a skip.

#include <catch2/catch_test_macros.hpp>

#include <cortext/models/aist_gguf_encoder.hpp>

#include "../src/models/aist_semantic_cache_internal.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
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

std::filesystem::path
AistModelPath ()
{
  auto resolved = cortext::ResolveAistGgufModelPath (RepoRoot () / "models");
  REQUIRE (resolved.has_value ());
  return *resolved;
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

TEST_CASE ("AIST GGUF metadata exposes triembed contract",
           "[aist][gguf][integration]")
{
  const auto info = cortext::InspectAistGgufModel (AistModelPath ());
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
  cortext::AistGgufConfig config;
  config.model_path = AistModelPath ().string ();
  config.context_length = 64;
  cortext::AistGgufEncoder encoder (config);
  CHECK (encoder.IsLoaded ());
  REQUIRE (encoder.IsRuntimeAvailable ());
  REQUIRE (encoder.UsesKernelOps ());
  CHECK (encoder.KernelOpsGranularity () != "none");

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

TEST_CASE ("AIST GGUF text graph memory stays bounded across lengths",
           "[aist][gguf][integration]")
{
  cortext::AistGgufConfig config;
  config.model_path = AistModelPath ().string ();
  config.context_length = 512;
  cortext::AistGgufEncoder encoder (config);
  REQUIRE (encoder.UsesKernelOps ());

  // Each text tokenizes to a distinct sequence length, as a real ingest
  // stream does. The longest goes first so it sizes the activation buffer.
  std::vector<std::string> texts;
  for (int words = 160; words >= 1; words -= 3)
    {
      std::string text;
      for (int w = 0; w < words; ++w)
        {
          text += (w % 2 == 0) ? "memory " : "garden ";
        }
      texts.push_back (text);
    }

  std::vector<float> embedding;
  encoder.EncodeText (texts.front (), embedding);
  const std::size_t after_longest
      = encoder.DebugRuntimeStats ().text_graph_buffer_bytes;
  REQUIRE (after_longest > 0);
  for (const std::string &text : texts)
    {
      encoder.EncodeText (text, embedding);
    }
  CHECK (encoder.DebugRuntimeStats ().text_graph_buffer_bytes
         == after_longest);

  // Shared activation memory must not leak state between lengths.
  std::vector<float> fresh;
  cortext::AistGgufEncoder fresh_encoder (config);
  fresh_encoder.EncodeText (texts[texts.size () / 2], fresh);
  encoder.EncodeText (texts.back () + "pool", embedding);
  encoder.EncodeText (texts[texts.size () / 2] + " ", embedding);
  REQUIRE (embedding.size () == fresh.size ());
  double max_abs_diff = 0.0;
  for (std::size_t i = 0; i < fresh.size (); ++i)
    {
      max_abs_diff = std::max (
          max_abs_diff,
          static_cast<double> (std::abs (embedding[i] - fresh[i])));
    }
  CHECK (max_abs_diff < 1.0e-5);
}

TEST_CASE ("AIST GGUF semantic cache stays within its byte budget",
           "[aist][gguf][integration]")
{
  cortext::AistGgufConfig config;
  config.model_path = AistModelPath ().string ();
  config.context_length = 64;
  cortext::AistGgufEncoder encoder (config);

  const std::string probe = "probe text that is encoded first";
  std::vector<float> embedding;
  encoder.EncodeText (probe, embedding);
  encoder.EncodeText (probe, embedding);
  REQUIRE (encoder.DebugRuntimeStats ().semantic_cache_hits == 1);

  // Fill past the budget with distinct texts; each entry costs at least its
  // embedding plus bookkeeping, so the probe must age out.
  using Cache = cortext::internal::AistSemanticCache;
  const std::size_t per_entry = Cache::EntryBytes (probe, embedding);
  const std::size_t fill = Cache::kDefaultCapacityBytes / per_entry + 2;
  for (std::size_t i = 0; i < fill; ++i)
    {
      encoder.EncodeText ("filler " + std::to_string (i), embedding);
    }
  const std::vector<float> before = embedding;
  encoder.EncodeText (probe, embedding);
  const auto stats = encoder.DebugRuntimeStats ();
  CHECK (stats.semantic_cache_hits == 1);
  CHECK (stats.semantic_cache_misses == fill + 2);
  CHECK (stats.semantic_requests == fill + 3);

  // The most recent text is still a hit.
  encoder.EncodeText ("filler " + std::to_string (fill - 1), embedding);
  CHECK (encoder.DebugRuntimeStats ().semantic_cache_hits == 2);
  CHECK (embedding == before);
}

TEST_CASE ("AIST native image tower output is pinned",
           "[aist][gguf][integration]")
{
  // Values recorded from the scalar single-threaded native image path; the
  // vectorized, row-parallel kernels must reproduce them.
  cortext::AistGgufConfig config;
  config.model_path = AistModelPath ().string ();
  cortext::AistGgufEncoder encoder (config);
  REQUIRE (encoder.IsRuntimeAvailable ());
  std::vector<std::uint8_t> pixels (768 * 512 * 3);
  for (std::size_t i = 0; i < pixels.size (); ++i)
    pixels[i] = static_cast<std::uint8_t> ((i * 2654435761u) >> 24);
  std::vector<float> v;
  encoder.EncodeImage (pixels.data (), 768, 512, 3, v);
  REQUIRE (v.size () == 1280);
  double sum = 0.0;
  for (float x : v)
    sum += x;
  CHECK (std::abs (v[0] - 0.0423770249) < 1.0e-6);
  CHECK (std::abs (v[1] - 0.0298241284) < 1.0e-6);
  CHECK (std::abs (sum - -0.047651531) < 1.0e-5);
}

TEST_CASE ("AIST position features match the training mirror",
           "[aist][gguf][integration][late_interaction]")
{
  // Values from tools/reranker/aist_torch.py (PyTorch mirror the
  // late-interaction reranker trains on).
  cortext::AistGgufConfig config;
  config.model_path = AistModelPath ().string ();
  cortext::AistGgufEncoder encoder (config);
  REQUIRE (encoder.IsRuntimeAvailable ());

  std::vector<float> plain;
  std::vector<float> with_positions;
  cortext::EncoderPositions text;
  const std::string sentence = "The playlist is called Summer Vibes.";
  encoder.EncodeText (sentence, plain);
  encoder.EncodeTextPositions (sentence, with_positions, text);
  // positions come from the full ggml text graph, not the Eigen fallback
  INFO ("full graph error: " << encoder.FullTextGraphError ());
  CHECK (encoder.FullTextGraphError ().empty ());
  REQUIRE (with_positions.size () == plain.size ());
  for (std::size_t i = 0; i < plain.size (); ++i)
    CHECK (std::abs (with_positions[i] - plain[i]) < 1.0e-5);
  REQUIRE (text.tower == "text");
  REQUIRE (text.count == 11);
  REQUIRE (text.dim == 384);
  const float token1[4] = { -0.0316F, -0.0425F, -0.0064F, -0.0158F };
  const float last[4] = { -0.0064F, -0.0123F, -0.0212F, 0.0067F };
  for (int c = 0; c < 4; ++c)
    {
      CHECK (std::abs (text.values[static_cast<std::size_t> (384 + c)] - token1[c]) < 1.0e-3);
      CHECK (std::abs (text.values[static_cast<std::size_t> (10 * 384 + c)] - last[c]) < 1.0e-3);
    }

  std::vector<std::uint8_t> pixels (768 * 512 * 3);
  for (std::size_t i = 0; i < pixels.size (); ++i)
    pixels[i] = static_cast<std::uint8_t> ((i * 2654435761u) >> 24);
  cortext::EncoderPositions image;
  std::vector<float> v;
  encoder.EncodeImagePositions (pixels.data (), 768, 512, 3, v, image);
  REQUIRE (v.size () == 1280);
  CHECK (std::abs (v[0] - 0.0423770249) < 1.0e-6);
  REQUIRE (image.tower == "image");
  REQUIRE (image.count == 144);
  REQUIRE (image.dim == 960);
  double total = 0.0;
  double patch70 = 0.0;
  for (std::size_t i = 0; i < image.values.size (); ++i)
    {
      total += image.values[i];
      if (i / 960 == 70)
        patch70 += image.values[i];
    }
  CHECK (std::abs (total - 91.2225) < 2.0e-2);
  CHECK (std::abs (patch70 - 0.052535) < 1.0e-3);
}
