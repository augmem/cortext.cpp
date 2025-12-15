#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cortext/generator/gemma_text_generator.hpp"
#include "cortext/generator/text_generator.hpp"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>

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

TEST_CASE ("GemmaTextGenerator construction", "[generator][text_generator]")
{
  auto models_dir = GetModelsDir ();

  SECTION ("Throws on missing models directory")
  {
    REQUIRE_THROWS_AS (
        cortext::GemmaTextGenerator ("/nonexistent/models/path"),
        std::runtime_error);
  }

  if (models_dir.empty ())
    {
      WARN ("Models directory not found, skipping remaining tests");
      return;
    }

  SECTION ("Constructs with valid models directory")
  {
    // May succeed or throw depending on whether tokenizer.json exists
    // and ORT is enabled
    try
      {
        cortext::GemmaTextGenerator gen (models_dir);
        // If it succeeds, check availability
        // (will be false if ORT not enabled)
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
        // With ORT, should be available if ONNX files exist
#else
        CHECK_FALSE (gen.IsAvailable ());
#endif
      }
    catch (const std::runtime_error &)
      {
        // Expected if tokenizer or models missing
      }
  }
}

TEST_CASE ("GemmaTextGenerator availability", "[generator][text_generator]")
{
  auto models_dir = GetModelsDir ();
  if (models_dir.empty ())
    {
      SKIP ("Models directory not found");
    }

  try
    {
      cortext::GemmaTextGenerator gen (models_dir);

      SECTION ("IsAvailable reflects ORT status")
      {
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
        // Available if ONNX files exist
        // This test just checks the API works
        bool available = gen.IsAvailable ();
        (void)available;
#else
        CHECK_FALSE (gen.IsAvailable ());
#endif
      }

      SECTION ("EosTokenId returns valid token")
      {
        auto eos = gen.EosTokenId ();
        CHECK (eos >= 0);
      }
    }
  catch (const std::runtime_error &)
    {
      SKIP ("Generator construction failed");
    }
}

TEST_CASE ("GenerationConfig defaults", "[generator][text_generator]")
{
  cortext::GenerationConfig config;

  SECTION ("Has sensible defaults")
  {
    CHECK (config.max_new_tokens == 256);
    CHECK (config.temperature == Catch::Approx (0.7f));
    CHECK (config.top_p == Catch::Approx (0.9f));
    CHECK (config.top_k == 50);
    CHECK (config.do_sample == true);
  }
}

TEST_CASE ("GenerationInput factory methods", "[generator][text_generator]")
{
  SECTION ("Text input")
  {
    auto input = cortext::GenerationInput::Text ("Hello, world!");
    CHECK (input.modality == "text");
    CHECK (input.text == "Hello, world!");
    CHECK (input.image_data.empty ());
    CHECK (input.audio_pcm.empty ());
  }

  SECTION ("Image input")
  {
    std::vector<std::uint8_t> pixels (64 * 64 * 3, 128);
    auto input = cortext::GenerationInput::Image (pixels.data (), 64, 64, 3);

    CHECK (input.modality == "image");
    CHECK (input.width == 64);
    CHECK (input.height == 64);
    CHECK (input.channels == 3);
    CHECK (input.image_data.size () == 64 * 64 * 3);
    CHECK (input.text.empty ());
    CHECK (input.audio_pcm.empty ());
  }

  SECTION ("Audio input")
  {
    std::vector<float> samples (16000, 0.5f); // 1 second at 16kHz
    auto input
        = cortext::GenerationInput::Audio (samples.data (), samples.size (),
                                           16000);

    CHECK (input.modality == "audio");
    CHECK (input.sample_rate == 16000);
    CHECK (input.audio_pcm.size () == 16000);
    CHECK (input.text.empty ());
    CHECK (input.image_data.empty ());
  }
}

#if defined(CORTEXT_ENABLE_GEMMA_ORT)
TEST_CASE ("GemmaTextGenerator JSON generation",
           "[generator][text_generator][integration]")
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

      SECTION ("Simple JSON schema")
      {
        nlohmann::json schema
            = { { "type", "object" },
                { "properties",
                  { { "answer", { { "type", "string" } } } } },
                { "required", { "answer" } } };

        auto result = gen.GenerateJSON ("What is 2+2? Answer in JSON format:",
                                        schema, 50, 0.3f);

        REQUIRE (result.contains ("answer"));
      }

      SECTION ("Enum constraint")
      {
        nlohmann::json schema = { { "type", "object" },
                                  { "properties",
                                    { { "sentiment",
                                        { { "type", "string" },
                                          { "enum",
                                            { "positive", "negative",
                                              "neutral" } } } } } },
                                  { "required", { "sentiment" } } };

        auto result = gen.GenerateJSON (
            "What is the sentiment of 'I love this!'? Respond with JSON:",
            schema, 50, 0.3f);

        REQUIRE (result.contains ("sentiment"));
        std::string sentiment = result["sentiment"];
        CHECK ((sentiment == "positive" || sentiment == "negative"
                || sentiment == "neutral"));
      }

      SECTION ("LastTokenCount tracks generation length")
      {
        nlohmann::json schema
            = { { "type", "object" },
                { "properties", { { "x", { { "type", "integer" } } } } },
                { "required", { "x" } } };

        gen.GenerateJSON ("Return x=1:", schema, 50, 0.3f);

        CHECK (gen.LastTokenCount () > 0);
        CHECK (gen.LastTokenCount () <= 50);
      }
    }
  catch (const std::runtime_error &e)
    {
      SKIP ("Generator construction failed: " + std::string (e.what ()));
    }
}

TEST_CASE ("GemmaTextGenerator text generation",
           "[generator][text_generator][integration]")
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

      SECTION ("Basic text generation")
      {
        cortext::GenerationConfig config;
        config.max_new_tokens = 20;
        config.temperature = 0.5f;

        auto result = gen.Generate ("Once upon a time", config);

        CHECK (!result.empty ());
      }

      SECTION ("Greedy generation")
      {
        cortext::GenerationConfig config;
        config.max_new_tokens = 10;
        config.do_sample = false;

        auto result = gen.Generate ("The capital of France is", config);

        CHECK (!result.empty ());
      }
    }
  catch (const std::runtime_error &e)
    {
      SKIP ("Generator construction failed: " + std::string (e.what ()));
    }
}
#endif
