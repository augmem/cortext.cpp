#include <catch2/catch_test_macros.hpp>

#include "cortext/extractor/gemma_extractor.hpp"

#include <filesystem>

TEST_CASE ("GemmaExtractor behavior", "[extractor][gemma]")
{
  SECTION ("GIVEN a non-existent model path")
  {
    cortext::GemmaExtractor extractor ("models/nonexistent-gemma-model");

    SECTION ("IsAvailable returns false")
    {
      // Model loading fails gracefully, IsAvailable should be false
      REQUIRE (extractor.IsAvailable () == false);
    }

    SECTION ("ExtractFromText throws runtime_error")
    {
      nlohmann::json schema = { { "type", "object" } };
      REQUIRE_THROWS_AS (extractor.ExtractFromText ("test", schema),
                         std::runtime_error);
    }

    SECTION ("ExtractFromAudio throws runtime_error")
    {
      std::vector<float> pcm (16000, 0.0f); // 1 second of silence
      nlohmann::json schema = { { "type", "object" } };
      REQUIRE_THROWS_AS (
          extractor.ExtractFromAudio (pcm.data (), pcm.size (), schema),
          std::runtime_error);
    }
  }

  SECTION ("GIVEN a valid Gemma model path")
  {
    const std::string model_path
        = "models/gemma3n-e2b-litert/gemma-3n-E2B-it-int4.litertlm";

    if (!std::filesystem::exists (model_path))
      {
        SUCCEED ("Skipping - model not found at " << model_path);
        return;
      }

    cortext::GemmaExtractor extractor (model_path);

    SECTION ("IsAvailable returns true when model loads successfully")
    {
      // Note: This may still be false if LiteRT-LM fails to load the model
      // for other reasons (incompatible format, etc.)
      CHECK (extractor.IsAvailable () == true);
    }
  }
}
