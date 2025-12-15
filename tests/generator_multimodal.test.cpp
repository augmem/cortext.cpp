#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cortext/generator/multimodal_encoder.hpp"
#include "cortext/generator/text_generator.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace
{

std::string
GetModelsDir ()
{
  std::vector<std::string> paths = {
    "models/gemma-3n",
    "../models/gemma-3n",
    "../../models/gemma-3n",
  };

  for (const auto &path : paths)
    {
      if (std::filesystem::exists (path))
        {
          return path;
        }
    }

  return "";
}

} // namespace

TEST_CASE ("GemmaVisionEncoder construction", "[generator][multimodal]")
{
  auto models_dir = GetModelsDir ();
  if (models_dir.empty ())
    {
      SKIP ("Models directory not found");
    }

  SECTION ("Constructs without throwing")
  {
    REQUIRE_NOTHROW (cortext::GemmaVisionEncoder (models_dir));
  }

  SECTION ("Reports availability correctly")
  {
    cortext::GemmaVisionEncoder encoder (models_dir);

#if defined(CORTEXT_ENABLE_GEMMA_ORT)
    // May be available if ONNX file exists
    // Just check the API works
    bool available = encoder.IsAvailable ();
    (void)available;
#else
    CHECK_FALSE (encoder.IsAvailable ());
#endif
  }

  SECTION ("Returns image token ID")
  {
    cortext::GemmaVisionEncoder encoder (models_dir);
    auto token_id = encoder.ImageTokenId ();
    CHECK (token_id > 0);
  }
}

TEST_CASE ("GemmaAudioEncoder construction", "[generator][multimodal]")
{
  auto models_dir = GetModelsDir ();
  if (models_dir.empty ())
    {
      SKIP ("Models directory not found");
    }

  SECTION ("Constructs without throwing")
  {
    REQUIRE_NOTHROW (cortext::GemmaAudioEncoder (models_dir));
  }

  SECTION ("Reports availability correctly")
  {
    cortext::GemmaAudioEncoder encoder (models_dir);

#if defined(CORTEXT_ENABLE_GEMMA_ORT)
    bool available = encoder.IsAvailable ();
    (void)available;
#else
    CHECK_FALSE (encoder.IsAvailable ());
#endif
  }

  SECTION ("Returns audio token ID")
  {
    cortext::GemmaAudioEncoder encoder (models_dir);
    auto token_id = encoder.AudioTokenId ();
    CHECK (token_id > 0);
  }
}

TEST_CASE ("MultimodalEncoder construction", "[generator][multimodal]")
{
  auto models_dir = GetModelsDir ();
  if (models_dir.empty ())
    {
      SKIP ("Models directory not found");
    }

  SECTION ("Constructs with valid models directory")
  {
    REQUIRE_NOTHROW (cortext::MultimodalEncoder (models_dir));
  }

  SECTION ("Reports vision/audio availability")
  {
    cortext::MultimodalEncoder encoder (models_dir);

#if defined(CORTEXT_ENABLE_GEMMA_ORT)
    // API should work regardless of actual availability
    bool has_vision = encoder.HasVision ();
    bool has_audio = encoder.HasAudio ();
    (void)has_vision;
    (void)has_audio;
#else
    CHECK_FALSE (encoder.HasVision ());
    CHECK_FALSE (encoder.HasAudio ());
#endif
  }

  SECTION ("Returns token IDs")
  {
    cortext::MultimodalEncoder encoder (models_dir);
    CHECK (encoder.ImageTokenId () > 0);
    CHECK (encoder.AudioTokenId () > 0);
  }
}

#if defined(CORTEXT_ENABLE_GEMMA_ORT)
TEST_CASE ("GemmaVisionEncoder encoding", "[generator][multimodal][integration]")
{
  auto models_dir = GetModelsDir ();
  if (models_dir.empty ())
    {
      SKIP ("Models directory not found");
    }

  cortext::GemmaVisionEncoder encoder (models_dir);

  if (!encoder.IsAvailable ())
    {
      SKIP ("Vision encoder not available");
    }

  SECTION ("Encodes RGB image")
  {
    // Create a simple 64x64 RGB test image (solid red)
    std::vector<std::uint8_t> image (64 * 64 * 3, 0);
    for (std::size_t i = 0; i < image.size (); i += 3)
      {
        image[i] = 255; // R
      }

    auto output = encoder.Encode (image.data (), 64, 64, 3);

    CHECK (!output.features.empty ());
    CHECK (output.num_image_tokens > 0);
    CHECK (output.shape.size () >= 2);
  }

  SECTION ("Encodes RGBA image")
  {
    // Create a 32x32 RGBA test image
    std::vector<std::uint8_t> image (32 * 32 * 4, 128);

    auto output = encoder.Encode (image.data (), 32, 32, 4);

    CHECK (!output.features.empty ());
    CHECK (output.num_image_tokens > 0);
  }

  SECTION ("Hidden dim matches output")
  {
    std::vector<std::uint8_t> image (64 * 64 * 3, 128);

    auto output = encoder.Encode (image.data (), 64, 64, 3);

    if (output.shape.size () >= 3)
      {
        CHECK (output.shape[2] == encoder.HiddenDim ());
      }
  }
}

TEST_CASE ("GemmaAudioEncoder encoding", "[generator][multimodal][integration]")
{
  auto models_dir = GetModelsDir ();
  if (models_dir.empty ())
    {
      SKIP ("Models directory not found");
    }

  cortext::GemmaAudioEncoder encoder (models_dir);

  if (!encoder.IsAvailable ())
    {
      SKIP ("Audio encoder not available");
    }

  SECTION ("Encodes audio samples")
  {
    // Create 1 second of silence at 16kHz
    std::vector<float> audio (16000, 0.0f);

    auto output = encoder.Encode (audio.data (), audio.size (), 16000);

    CHECK (!output.features.empty ());
    CHECK (output.num_audio_tokens > 0);
    CHECK (output.shape.size () >= 2);
  }

  SECTION ("Audio tokens scale with duration")
  {
    // 0.5 seconds
    std::vector<float> audio_short (8000, 0.0f);
    auto output_short
        = encoder.Encode (audio_short.data (), audio_short.size (), 16000);

    // 2 seconds
    std::vector<float> audio_long (32000, 0.0f);
    auto output_long
        = encoder.Encode (audio_long.data (), audio_long.size (), 16000);

    // Longer audio should produce more tokens
    CHECK (output_long.num_audio_tokens > output_short.num_audio_tokens);
  }

  SECTION ("Handles different sample rates")
  {
    // 48kHz audio (will be resampled to 16kHz)
    std::vector<float> audio (48000, 0.0f); // 1 second at 48kHz

    auto output = encoder.Encode (audio.data (), audio.size (), 48000);

    CHECK (!output.features.empty ());
    CHECK (output.num_audio_tokens > 0);
  }
}

TEST_CASE ("MultimodalEncoder feature injection",
           "[generator][multimodal][integration]")
{
  auto models_dir = GetModelsDir ();
  if (models_dir.empty ())
    {
      SKIP ("Models directory not found");
    }

  cortext::MultimodalEncoder encoder (models_dir);

  SECTION ("InjectImageFeatures modifies embeddings")
  {
    if (!encoder.HasVision ())
      {
        SKIP ("Vision encoder not available");
      }

    // Mock embeddings with image token placeholder
    std::size_t hidden_dim = 2048;
    std::size_t seq_len = 10;
    std::vector<float> embeddings (seq_len * hidden_dim, 1.0f);
    std::vector<std::size_t> shape = { 1, seq_len, hidden_dim };

    // Input IDs with image token at position 5
    std::vector<int64_t> input_ids (seq_len, 100);
    input_ids[5] = encoder.ImageTokenId ();

    // Create test image
    std::vector<std::uint8_t> image (64 * 64 * 3, 128);

    auto result = encoder.InjectImageFeatures (embeddings, shape, input_ids,
                                               image.data (), 64, 64, 3);

    CHECK (result.size () == embeddings.size ());

    // Check that position 5 was modified (should differ from 1.0f)
    bool modified = false;
    for (std::size_t i = 5 * hidden_dim; i < 6 * hidden_dim; ++i)
      {
        if (result[i] != 1.0f)
          {
            modified = true;
            break;
          }
      }
    CHECK (modified);
  }

  SECTION ("InjectAudioFeatures modifies embeddings")
  {
    if (!encoder.HasAudio ())
      {
        SKIP ("Audio encoder not available");
      }

    std::size_t hidden_dim = 2048;
    std::size_t seq_len = 10;
    std::vector<float> embeddings (seq_len * hidden_dim, 1.0f);
    std::vector<std::size_t> shape = { 1, seq_len, hidden_dim };

    std::vector<int64_t> input_ids (seq_len, 100);
    input_ids[3] = encoder.AudioTokenId ();

    std::vector<float> audio (8000, 0.0f); // 0.5 seconds

    auto result = encoder.InjectAudioFeatures (embeddings, shape, input_ids,
                                               audio.data (), audio.size (),
                                               16000);

    CHECK (result.size () == embeddings.size ());
  }
}
#endif

TEST_CASE ("Multimodal preprocessing unit tests", "[generator][multimodal]")
{
  // These tests verify preprocessing logic without requiring ORT

  SECTION ("Image dimensions preserved in GenerationInput")
  {
    std::vector<std::uint8_t> pixels (100 * 50 * 3, 0);
    auto input = cortext::GenerationInput::Image (pixels.data (), 100, 50, 3);

    CHECK (input.width == 100);
    CHECK (input.height == 50);
    CHECK (input.channels == 3);
    CHECK (input.image_data.size () == 100 * 50 * 3);
  }

  SECTION ("Audio samples preserved in GenerationInput")
  {
    std::vector<float> samples = { 0.1f, -0.2f, 0.3f, -0.4f };
    auto input = cortext::GenerationInput::Audio (samples.data (),
                                                  samples.size (), 44100);

    CHECK (input.sample_rate == 44100);
    CHECK (input.audio_pcm.size () == 4);
    CHECK (input.audio_pcm[0] == Catch::Approx (0.1f));
    CHECK (input.audio_pcm[1] == Catch::Approx (-0.2f));
  }
}
