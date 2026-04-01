#include <catch2/catch_test_macros.hpp>
#include <cortext/summarizer/phi4_summarizer.hpp>
#include <stdexcept>
#include <vector>

using namespace cortext;

TEST_CASE ("Phi4Summarizer behavior", "[summarizer][phi4]")
{
  SECTION ("GIVEN OGA is disabled")
  {
    // When CORTEXT_DISABLE_OGA is defined, Phi4Summarizer throws on use.
    Phi4Summarizer summarizer ("models/phi4-mm-cpu");

    SECTION ("IsAvailable returns false")
    {
#if !defined(CORTEXT_DISABLE_OGA)
      // If OGA is enabled but model not present, should return false
      // (model loading would fail in constructor)
      SUCCEED ("Skipping - OGA enabled, behavior depends on model presence");
#else
      REQUIRE (summarizer.IsAvailable () == false);
#endif
    }

    SECTION ("SummarizeTexts throws when OGA disabled")
    {
#if !defined(CORTEXT_DISABLE_OGA)
      SUCCEED ("Skipping - OGA enabled");
#else
      std::vector<std::string> texts = { "text1", "text2" };
      REQUIRE_THROWS_AS (summarizer.SummarizeTexts (texts), std::runtime_error);
#endif
    }

    SECTION ("SummarizeAudio throws when OGA disabled")
    {
#if !defined(CORTEXT_DISABLE_OGA)
      SUCCEED ("Skipping - OGA enabled");
#else
      float pcm[160] = { 0 };
      REQUIRE_THROWS_AS (summarizer.SummarizeAudio (pcm, 160),
                         std::runtime_error);
#endif
    }

    SECTION ("SummarizeAudioSegments throws when OGA disabled")
    {
#if !defined(CORTEXT_DISABLE_OGA)
      SUCCEED ("Skipping - OGA enabled");
#else
      float pcm[160] = { 0 };
      std::vector<AudioSegment> segments = { { pcm, 160 } };
      REQUIRE_THROWS_AS (summarizer.SummarizeAudioSegments (segments),
                         std::runtime_error);
#endif
    }
  }
}

TEST_CASE ("Phi4Summarizer move semantics", "[summarizer][phi4]")
{
  SECTION ("Move constructor works")
  {
    Phi4Summarizer summarizer1 ("models/phi4-mm-cpu");
    Phi4Summarizer summarizer2 (std::move (summarizer1));

    // summarizer2 should be in valid state
#if !defined(CORTEXT_DISABLE_OGA)
    SUCCEED ("Move completed");
#else
    REQUIRE (summarizer2.IsAvailable () == false);
#endif
  }

  SECTION ("Move assignment works")
  {
    Phi4Summarizer summarizer1 ("models/phi4-mm-cpu");
    Phi4Summarizer summarizer2 ("models/phi4-mm-cpu");

    summarizer2 = std::move (summarizer1);

#if !defined(CORTEXT_DISABLE_OGA)
    SUCCEED ("Move assignment completed");
#else
    REQUIRE (summarizer2.IsAvailable () == false);
#endif
  }
}

TEST_CASE ("Phi4Summarizer empty inputs", "[summarizer][phi4]")
{
  SECTION ("SummarizeTexts with empty vector")
  {
#if !defined(CORTEXT_DISABLE_OGA)
    SUCCEED ("Skipping - OGA enabled");
#else
    Phi4Summarizer summarizer ("models/phi4-mm-cpu");
    std::vector<std::string> empty_texts;
    // Empty input should throw (OGA disabled)
    REQUIRE_THROWS_AS (summarizer.SummarizeTexts (empty_texts),
                       std::runtime_error);
#endif
  }
}
