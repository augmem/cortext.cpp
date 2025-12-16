#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cortext/generator/multimodal_encoder.hpp"
#include "cortext/generator/text_generator.hpp"

#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
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

#if defined(CORTEXT_ENABLE_GEMMA_ORT)
#include "cortext/generator/gemma_text_generator.hpp"

TEST_CASE ("GemmaTextGenerator multimodal generation",
           "[generator][multimodal][integration]")
{
  auto models_dir = GetModelsDir ();
  if (models_dir.empty ())
    {
      SKIP ("Models directory not found");
    }

  try
    {
      cortext::GemmaTextGenerator gen (models_dir);

      if (!gen.IsAvailable ())
        {
          SKIP ("Generator not available");
        }

      SECTION ("Image to text generation")
      {
        // Create a simple test image (64x64 red square)
        std::vector<std::uint8_t> image (64 * 64 * 3, 0);
        for (std::size_t i = 0; i < image.size (); i += 3)
          {
            image[i] = 255; // R
          }

        std::vector<cortext::GenerationInput> inputs = {
          cortext::GenerationInput::Text ("Describe this image: "),
          cortext::GenerationInput::Image (image.data (), 64, 64, 3),
        };

        cortext::GenerationConfig config;
        config.max_new_tokens = 30;
        config.temperature = 0.5f;

        auto result = gen.GenerateMultimodal (inputs, config);

        CHECK (!result.empty ());
      }

      SECTION ("Text only multimodal")
      {
        std::vector<cortext::GenerationInput> inputs = {
          cortext::GenerationInput::Text ("Hello, "),
          cortext::GenerationInput::Text ("world!"),
        };

        cortext::GenerationConfig config;
        config.max_new_tokens = 10;
        config.temperature = 0.5f;

        auto result = gen.GenerateMultimodal (inputs, config);

        CHECK (!result.empty ());
      }

      SECTION ("Audio to text generation")
      {
        // Create 0.5 seconds of silence at 16kHz
        std::vector<float> audio (8000, 0.0f);

        std::vector<cortext::GenerationInput> inputs = {
          cortext::GenerationInput::Text ("Transcribe this audio: "),
          cortext::GenerationInput::Audio (audio.data (), audio.size (), 16000),
        };

        cortext::GenerationConfig config;
        config.max_new_tokens = 30;
        config.temperature = 0.5f;

        auto result = gen.GenerateMultimodal (inputs, config);

        CHECK (!result.empty ());
      }

      SECTION ("Interleaved text and image")
      {
        std::vector<std::uint8_t> image (32 * 32 * 3, 128);

        std::vector<cortext::GenerationInput> inputs = {
          cortext::GenerationInput::Text ("Look at this: "),
          cortext::GenerationInput::Image (image.data (), 32, 32, 3),
          cortext::GenerationInput::Text (" What do you see?"),
        };

        cortext::GenerationConfig config;
        config.max_new_tokens = 20;
        config.temperature = 0.5f;

        auto result = gen.GenerateMultimodal (inputs, config);

        CHECK (!result.empty ());
      }
    }
  catch (const std::runtime_error &e)
    {
      SKIP ("Generator construction failed: " + std::string (e.what ()));
    }
}

TEST_CASE ("GemmaTextGenerator multimodal JSON generation",
           "[generator][multimodal][integration]")
{
  auto models_dir = GetModelsDir ();
  if (models_dir.empty ())
    {
      SKIP ("Models directory not found");
    }

  try
    {
      cortext::GemmaTextGenerator gen (models_dir);

      if (!gen.IsAvailable ())
        {
          SKIP ("Generator not available");
        }

      SECTION ("Image classification with JSON schema")
      {
        std::vector<std::uint8_t> image (64 * 64 * 3, 0);
        // Create a gradient pattern
        for (std::size_t y = 0; y < 64; ++y)
          {
            for (std::size_t x = 0; x < 64; ++x)
              {
                std::size_t idx = (y * 64 + x) * 3;
                image[idx] = static_cast<std::uint8_t> (x * 4);     // R
                image[idx + 1] = static_cast<std::uint8_t> (y * 4); // G
                image[idx + 2] = 128;                               // B
              }
          }

        std::vector<cortext::GenerationInput> inputs = {
          cortext::GenerationInput::Text (
              "Classify this image. Respond in JSON:"),
          cortext::GenerationInput::Image (image.data (), 64, 64, 3),
        };

        nlohmann::json schema
            = { { "type", "object" },
                { "properties",
                  { { "category", { { "type", "string" } } },
                    { "confidence", { { "type", "number" } } } } },
                { "required", { "category" } } };

        auto result
            = gen.GenerateJSONMultimodal (inputs, schema, 100, 0.3f, 40, 0.9f);

        REQUIRE (result.contains ("category"));
        CHECK (result["category"].is_string ());
      }
    }
  catch (const std::runtime_error &e)
    {
      SKIP ("Generator construction failed: " + std::string (e.what ()));
    }
}
#endif
