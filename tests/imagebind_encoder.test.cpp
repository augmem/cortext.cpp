#include <catch2/catch_test_macros.hpp>
#include <cortext/encoder/imagebind.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace
{
double
L2Norm (const std::vector<float> &v)
{
  double s2 = 0.0;
  for (float x : v)
    s2 += static_cast<double> (x) * static_cast<double> (x);
  return std::sqrt (s2);
}

bool
AllFinite (const std::vector<float> &v)
{
  for (float x : v)
    {
      if (!std::isfinite (x))
        return false;
    }
  return true;
}

bool
ModelsPresent ()
{
  namespace fs = std::filesystem;
  const fs::path md ("models/imagebind");
  const bool has_text = fs::exists (md / "text_encoder_int8.onnx")
                        || fs::exists (md / "text_encoder.onnx");
  const bool has_audio = fs::exists (md / "audio_encoder_int8.onnx")
                         || fs::exists (md / "audio_encoder.onnx");
  const bool has_vision = fs::exists (md / "vision_encoder_int8.onnx")
                          || fs::exists (md / "vision_encoder.onnx");
  // BPE may be found in models/ or in poc/ (encoder searches for it).
  const bool has_bpe = fs::exists (md / "bpe" / "bpe_simple_vocab_16e6.txt.gz")
                       || fs::exists (md / "bpe_simple_vocab_16e6.txt.gz")
                       || fs::exists ("poc/ImageBind/imagebind/bpe/"
                                      "bpe_simple_vocab_16e6.txt.gz");
  return has_text && has_audio && has_vision && has_bpe;
}
} // namespace

TEST_CASE ("ImageBindEncoder produces embeddings for all three modalities",
           "[encoder][imagebind]")
{
#if !defined(CORTEXT_ENABLE_IMAGEBIND_ORT)
  SUCCEED ("ONNX Runtime disabled; skipping ImageBind encoder test.");
#else
  if (!ModelsPresent ())
    {
      SUCCEED ("ImageBind models/tokenizer assets not present; skipping.");
      return;
    }
  constexpr double kPi = 3.14159265358979323846;
  cortext::ImageBindEncoder enc ("models/imagebind");

  SECTION ("text")
  {
    std::vector<float> e1;
    std::vector<float> e2;
    enc.EncodeText ("a dog", e1);
    enc.EncodeText ("a dog", e2);
    REQUIRE (e1.size () == 256);
    REQUIRE (e2.size () == 256);
    REQUIRE (AllFinite (e1));
    REQUIRE (L2Norm (e1) > 0.0);
    REQUIRE (e1 == e2);
  }

  SECTION ("audio")
  {
    // 2 seconds of a 440Hz sine wave at 16kHz.
    constexpr int sr = 16000;
    constexpr int n = 2 * sr;
    std::vector<float> pcm;
    pcm.resize (n);
    for (int i = 0; i < n; ++i)
      {
        const double t = static_cast<double> (i) / static_cast<double> (sr);
        pcm[static_cast<std::size_t> (i)]
            = static_cast<float> (0.2 * std::sin (2.0 * kPi * 440.0 * t));
      }

    std::vector<float> e1;
    std::vector<float> e2;
    enc.EncodeAudio (pcm.data (), pcm.size (), e1);
    enc.EncodeAudio (pcm.data (), pcm.size (), e2);
    REQUIRE (e1.size () == 256);
    REQUIRE (e2.size () == 256);
    REQUIRE (AllFinite (e1));
    REQUIRE (L2Norm (e1) > 0.0);
    REQUIRE (e1 == e2);
  }

  SECTION ("image")
  {
    constexpr int w = 224;
    constexpr int h = 224;
    constexpr int c = 3;
    std::vector<std::uint8_t> rgb;
    rgb.resize (static_cast<std::size_t> (w * h * c));
    for (int y = 0; y < h; ++y)
      {
        for (int x = 0; x < w; ++x)
          {
            const int idx = (y * w + x) * c;
            rgb[static_cast<std::size_t> (idx + 0)]
                = static_cast<std::uint8_t> (x & 0xFF);
            rgb[static_cast<std::size_t> (idx + 1)]
                = static_cast<std::uint8_t> (y & 0xFF);
            rgb[static_cast<std::size_t> (idx + 2)]
                = static_cast<std::uint8_t> ((x ^ y) & 0xFF);
          }
      }

    std::vector<float> e1;
    std::vector<float> e2;
    enc.EncodeImage (rgb.data (), w, h, c, e1);
    enc.EncodeImage (rgb.data (), w, h, c, e2);
    REQUIRE (e1.size () == 256);
    REQUIRE (e2.size () == 256);
    REQUIRE (AllFinite (e1));
    REQUIRE (L2Norm (e1) > 0.0);
    REQUIRE (e1 == e2);
  }
#endif
}

