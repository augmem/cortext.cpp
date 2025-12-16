#include <catch2/catch_test_macros.hpp>
#include <cortext/extractor/phi4_extractor.hpp>
#include <nlohmann/json.hpp>
#include <stdexcept>

using namespace cortext;

TEST_CASE ("Phi4Extractor behavior", "[extractor][phi4]")
{
  SECTION ("GIVEN OGA is disabled")
  {
    // When CORTEXT_DISABLE_OGA is defined, Phi4Extractor throws on use.
    Phi4Extractor extractor ("models/phi4-mm-cpu");

    SECTION ("IsAvailable returns false")
    {
#if !defined(CORTEXT_DISABLE_OGA)
      // If OGA is enabled but model not present, should return false
      // (model loading would fail in constructor)
      // Skip this test as model availability depends on filesystem
      SUCCEED ("Skipping - OGA enabled, behavior depends on model presence");
#else
      REQUIRE (extractor.IsAvailable () == false);
#endif
    }

    SECTION ("ExtractFromText throws when OGA disabled")
    {
#if !defined(CORTEXT_DISABLE_OGA)
      SUCCEED ("Skipping - OGA enabled");
#else
      nlohmann::json schema = { { "type", "object" } };
      REQUIRE_THROWS_AS (extractor.ExtractFromText ("test text", schema),
                         std::runtime_error);
#endif
    }

    SECTION ("ExtractFromAudio throws when OGA disabled")
    {
#if !defined(CORTEXT_DISABLE_OGA)
      SUCCEED ("Skipping - OGA enabled");
#else
      nlohmann::json schema = { { "type", "object" } };
      float pcm[160] = { 0 };
      REQUIRE_THROWS_AS (extractor.ExtractFromAudio (pcm, 160, schema),
                         std::runtime_error);
#endif
    }
  }
}

TEST_CASE ("Phi4Extractor move semantics", "[extractor][phi4]")
{
  SECTION ("Move constructor works")
  {
    Phi4Extractor extractor1 ("models/phi4-mm-cpu");
    Phi4Extractor extractor2 (std::move (extractor1));

    // extractor2 should be in valid state
#if !defined(CORTEXT_DISABLE_OGA)
    SUCCEED ("Move completed");
#else
    REQUIRE (extractor2.IsAvailable () == false);
#endif
  }

  SECTION ("Move assignment works")
  {
    Phi4Extractor extractor1 ("models/phi4-mm-cpu");
    Phi4Extractor extractor2 ("models/phi4-mm-cpu");

    extractor2 = std::move (extractor1);

#if !defined(CORTEXT_DISABLE_OGA)
    SUCCEED ("Move assignment completed");
#else
    REQUIRE (extractor2.IsAvailable () == false);
#endif
  }
}
