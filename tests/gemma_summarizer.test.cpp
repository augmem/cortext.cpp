#include <catch2/catch_test_macros.hpp>

#include "cortext/summarizer/gemma_summarizer.hpp"

#include <filesystem>

TEST_CASE ("GemmaSummarizer behavior", "[summarizer][gemma]")
{
  SECTION ("GIVEN a non-existent model path")
  {
    cortext::GemmaSummarizer summarizer ("models/nonexistent-gemma-model");

    SECTION ("IsAvailable returns false")
    {
      // Model loading fails gracefully, IsAvailable should be false
      REQUIRE (summarizer.IsAvailable () == false);
    }

    SECTION ("SummarizeTexts throws runtime_error")
    {
      std::vector<std::string> texts = { "Test text 1", "Test text 2" };
      REQUIRE_THROWS_AS (summarizer.SummarizeTexts (texts), std::runtime_error);
    }

    SECTION ("SummarizeAudio throws runtime_error")
    {
      std::vector<float> pcm (16000, 0.0f); // 1 second of silence
      REQUIRE_THROWS_AS (summarizer.SummarizeAudio (pcm.data (), pcm.size ()),
                         std::runtime_error);
    }

    SECTION ("SummarizeAudioSegments throws runtime_error")
    {
      std::vector<float> pcm (16000, 0.0f);
      std::vector<cortext::AudioSegment> segments
          = { { pcm.data (), pcm.size () } };
      REQUIRE_THROWS_AS (summarizer.SummarizeAudioSegments (segments),
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

    cortext::GemmaSummarizer summarizer (model_path);

    SECTION ("IsAvailable returns true when model loads successfully")
    {
      // Note: This may still be false if LiteRT-LM fails to load the model
      // for other reasons (incompatible format, etc.)
      CHECK (summarizer.IsAvailable () == true);
    }
  }
}
