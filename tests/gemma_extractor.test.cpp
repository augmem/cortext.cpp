#include <catch2/catch_test_macros.hpp>

#include "cortext/extractor/gemma_extractor.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>

namespace
{
std::string
ToLower (std::string value)
{
  std::transform (value.begin (), value.end (), value.begin (),
                  [] (unsigned char c) { return static_cast<char> (std::tolower (c)); });
  return value;
}

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
BuildLabelSchema (bool require_labels)
{
  nlohmann::json labels = {
    { "type", "array" },
    { "items", { { "type", "string" } } }
  };
  if (require_labels)
    {
      labels["minItems"] = 1;
    }

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
    { "properties", { { "labels", labels }, { "relations", relations } } }
  };
  if (require_labels)
    {
      schema["required"] = nlohmann::json::array ({ "labels" });
    }
  return schema;
}
} // namespace

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
    const std::string model_path = FindModelPath (
        "models/gemma3n-e2b-litert/gemma-3n-E2B-it-int4.litertlm");

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

TEST_CASE ("GemmaExtractor label extraction (integration)",
           "[extractor][gemma][integration]")
{
  const std::string model_path = FindModelPath (
      "models/gemma3n-e2b-litert/gemma-3n-E2B-it-int4.litertlm");

  if (!std::filesystem::exists (model_path))
    {
      SUCCEED ("Skipping - model not found at " << model_path);
      return;
    }

  cortext::GemmaExtractor extractor (model_path);

  if (!extractor.IsAvailable ())
    {
      SUCCEED ("Skipping - extractor unavailable for " << model_path);
      return;
    }

  nlohmann::json schema = BuildLabelSchema (true);
  auto result = extractor.ExtractFromText (
      "Alice works at Acme Corp in Seattle.", schema);

  REQUIRE_FALSE (result.labels.empty ());

  bool has_named_label = false;
  bool has_expected_label = false;
  for (const auto &label : result.labels)
    {
      if (!Trim (label.label).empty ())
        {
          has_named_label = true;
          const auto lower_name = ToLower (label.label);
          if (lower_name.find ("alice") != std::string::npos
              || lower_name.find ("acme") != std::string::npos
              || lower_name.find ("seattle") != std::string::npos)
            {
              has_expected_label = true;
            }
        }
    }

  CHECK (has_named_label);
  CHECK (has_expected_label);

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
}
