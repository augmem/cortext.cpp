#include <catch2/catch_test_macros.hpp>
#include <cortext/extractor/phi4_extractor.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>

using namespace cortext;

namespace
{
std::string
Trim (std::string value)
{
  auto start = value.begin ();
  auto end = value.end ();
  while (start != end && std::isspace (static_cast<unsigned char> (*start)))
    {
      ++start;
    }
  while (end != start)
    {
      auto prev = end;
      --prev;
      if (!std::isspace (static_cast<unsigned char> (*prev)))
        break;
      end = prev;
    }
  return std::string (start, end);
}

std::string
FindModelPath (const std::string &relative_path)
{
  std::filesystem::path probe = std::filesystem::current_path ();
  for (int i = 0; i < 6; ++i)
    {
      const auto candidate = probe / relative_path;
      if (std::filesystem::exists (candidate))
        {
          return candidate.string ();
        }
      if (!probe.has_parent_path ())
        {
          break;
        }
      probe = probe.parent_path ();
    }
  return relative_path;
}

nlohmann::json
BuildLabelSchema ()
{
  nlohmann::json labels = {
    { "type", "array" },
    { "items", { { "type", "string" } } },
    { "minItems", 1 }
  };

  nlohmann::json relation_props = {
    { "subject", { { "type", "string" } } },
    { "predicate", { { "type", "string" } } },
    { "object", { { "type", "string" } } },
    { "confidence", { { "type", "number" } } }
  };

  nlohmann::json relation_items = {
    { "type", "object" },
    { "properties", relation_props },
    { "required", nlohmann::json::array ({ "subject", "predicate", "object" }) }
  };

  nlohmann::json relations = {
    { "type", "array" },
    { "items", relation_items }
  };

  nlohmann::json schema = {
    { "type", "object" },
    { "properties", { { "labels", labels }, { "relations", relations } } },
    { "required", nlohmann::json::array ({ "labels" }) }
  };
  return schema;
}
} // namespace

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

TEST_CASE ("Phi4Extractor label extraction (integration)",
           "[extractor][phi4][integration]")
{
#if defined(CORTEXT_DISABLE_OGA)
  SUCCEED ("Skipping - OGA disabled");
  return;
#else
  const std::string model_path = FindModelPath ("models/phi4-mm-cpu");
  if (!std::filesystem::exists (model_path))
    {
      SUCCEED ("Skipping - model not found at " << model_path);
      return;
    }

  Phi4Extractor extractor (model_path);
  if (!extractor.IsAvailable ())
    {
      SUCCEED ("Skipping - extractor unavailable for " << model_path);
      return;
    }

  auto schema = BuildLabelSchema ();
  auto result = extractor.ExtractFromText (
      "Alice works at Acme Corp in Seattle.", schema);

  REQUIRE_FALSE (result.labels.empty ());
  for (const auto &label : result.labels)
    {
      CHECK_FALSE (Trim (label.label).empty ());
    }
  for (const auto &relation : result.relations)
    {
      CHECK_FALSE (Trim (relation.subject).empty ());
      CHECK_FALSE (Trim (relation.predicate).empty ());
      CHECK_FALSE (Trim (relation.object).empty ());
    }
#endif
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
