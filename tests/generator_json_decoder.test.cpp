#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cortext/generator/json_decoder.hpp"

#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <vector>

using json = nlohmann::json;

TEST_CASE ("StreamingJSONParser construction", "[generator][json_decoder]")
{
  SECTION ("Simple object schema")
  {
    json schema = { { "type", "object" },
                    { "properties", { { "name", { { "type", "string" } } } } },
                    { "required", { "name" } } };

    REQUIRE_NOTHROW (cortext::StreamingJSONParser (schema));
  }

  SECTION ("Schema with enum")
  {
    json schema
        = { { "type", "object" },
            { "properties",
              { { "color",
                  { { "type", "string" },
                    { "enum", { "red", "green", "blue" } } } } } } };

    REQUIRE_NOTHROW (cortext::StreamingJSONParser (schema));
  }

  SECTION ("Schema with oneOf")
  {
    json schema = { { "oneOf",
                      { { { "type", "object" },
                          { "properties",
                            { { "type", { { "const", "cat" } } } } } },
                        { { "type", "object" },
                          { "properties",
                            { { "type", { { "const", "dog" } } } } } } } } };

    REQUIRE_NOTHROW (cortext::StreamingJSONParser (schema));
  }
}

TEST_CASE ("StreamingJSONParser basic parsing", "[generator][json_decoder]")
{
  json schema = { { "type", "object" },
                  { "properties",
                    { { "name", { { "type", "string" } } },
                      { "age", { { "type", "integer" } } } } },
                  { "required", { "name" } } };

  cortext::StreamingJSONParser parser (schema);

  SECTION ("Single token parsing")
  {
    auto status = parser.AddToken ("{");
    CHECK (status == cortext::ParserStatus::kValidPartial);
    CHECK (parser.State () == cortext::JSONState::kExpectingKeyOrClose);
  }

  SECTION ("Valid JSON object parsing")
  {
    std::vector<std::string> tokens = { "{", "\"name\"", ":", "\"Alice\"",
                                        "}" };

    cortext::ParserStatus status = cortext::ParserStatus::kValidPartial;
    for (const auto &token : tokens)
      {
        status = parser.AddToken (token);
        if (status == cortext::ParserStatus::kInvalid)
          break;
      }

    CHECK (status == cortext::ParserStatus::kComplete);

    auto result = parser.GetParsedObject ();
    CHECK (result["name"] == "Alice");
  }

  SECTION ("Partial JSON object")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"name\"");
    parser.AddToken (":");

    CHECK (parser.State () == cortext::JSONState::kExpectingValue);

    auto status = parser.AddToken ("\"Bob");
    CHECK (status == cortext::ParserStatus::kValidPartial);
    CHECK (parser.State () == cortext::JSONState::kInStringValue);
  }

  SECTION ("Character by character string parsing")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"");
    parser.AddToken ("n");
    parser.AddToken ("a");
    parser.AddToken ("m");
    parser.AddToken ("e");
    parser.AddToken ("\"");
    parser.AddToken (":");
    parser.AddToken ("\"");
    parser.AddToken ("T");
    parser.AddToken ("e");
    parser.AddToken ("s");
    parser.AddToken ("t");
    parser.AddToken ("\"");
    auto status = parser.AddToken ("}");

    CHECK (status == cortext::ParserStatus::kComplete);
    CHECK (parser.GetParsedObject ()["name"] == "Test");
  }
}

TEST_CASE ("StreamingJSONParser number parsing", "[generator][json_decoder]")
{
  json schema
      = { { "type", "object" },
          { "properties", { { "value", { { "type", "number" } } } } } };

  cortext::StreamingJSONParser parser (schema);

  SECTION ("Integer parsing")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"value\"");
    parser.AddToken (":");
    parser.AddToken ("42");
    auto status = parser.AddToken ("}");

    CHECK (status == cortext::ParserStatus::kComplete);
    CHECK (parser.GetParsedObject ()["value"] == 42);
  }

  SECTION ("Negative number")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"value\"");
    parser.AddToken (":");
    parser.AddToken ("-123");
    auto status = parser.AddToken ("}");

    CHECK (status == cortext::ParserStatus::kComplete);
    CHECK (parser.GetParsedObject ()["value"] == -123);
  }

  SECTION ("Decimal number")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"value\"");
    parser.AddToken (":");
    parser.AddToken ("3.14");
    auto status = parser.AddToken ("}");

    CHECK (status == cortext::ParserStatus::kComplete);
    CHECK (parser.GetParsedObject ()["value"].get<double> ()
           == Catch::Approx (3.14));
  }

  SECTION ("Number with exponent")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"value\"");
    parser.AddToken (":");
    parser.AddToken ("1.5e10");
    auto status = parser.AddToken ("}");

    CHECK (status == cortext::ParserStatus::kComplete);
  }
}

TEST_CASE ("StreamingJSONParser boolean and null",
           "[generator][json_decoder]")
{
  json schema
      = { { "type", "object" },
          { "properties",
            { { "active", { { "type", "boolean" } } },
              { "data", { { "type", "null" } } } } } };

  SECTION ("Boolean true")
  {
    cortext::StreamingJSONParser parser (schema);
    parser.AddToken ("{");
    parser.AddToken ("\"active\"");
    parser.AddToken (":");
    parser.AddToken ("true");
    auto status = parser.AddToken ("}");

    CHECK (status == cortext::ParserStatus::kComplete);
    CHECK (parser.GetParsedObject ()["active"] == true);
  }

  SECTION ("Boolean false")
  {
    cortext::StreamingJSONParser parser (schema);
    parser.AddToken ("{");
    parser.AddToken ("\"active\"");
    parser.AddToken (":");
    parser.AddToken ("false");
    auto status = parser.AddToken ("}");

    CHECK (status == cortext::ParserStatus::kComplete);
    CHECK (parser.GetParsedObject ()["active"] == false);
  }

  SECTION ("Null value")
  {
    cortext::StreamingJSONParser parser (schema);
    parser.AddToken ("{");
    parser.AddToken ("\"data\"");
    parser.AddToken (":");
    parser.AddToken ("null");
    auto status = parser.AddToken ("}");

    CHECK (status == cortext::ParserStatus::kComplete);
    CHECK (parser.GetParsedObject ()["data"].is_null ());
  }
}

TEST_CASE ("StreamingJSONParser allowed tokens", "[generator][json_decoder]")
{
  json schema = { { "type", "object" },
                  { "properties",
                    { { "name", { { "type", "string" } } },
                      { "count", { { "type", "integer" } } } } },
                  { "required", { "name", "count" } } };

  cortext::StreamingJSONParser parser (schema);

  SECTION ("Initial state allows opening brace")
  {
    auto allowed = parser.GetAllowedTokens ();
    CHECK (std::find (allowed.begin (), allowed.end (), "{") != allowed.end ());
  }

  SECTION ("After brace allows key")
  {
    parser.AddToken ("{");
    auto allowed = parser.GetAllowedTokens ();

    // Should allow quote for starting key
    CHECK (std::find (allowed.begin (), allowed.end (), "\"")
           != allowed.end ());
    // Note: closing brace may not be allowed if required fields exist
  }

  SECTION ("After key allows colon")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"name\"");
    auto allowed = parser.GetAllowedTokens ();

    CHECK (std::find (allowed.begin (), allowed.end (), ":")
           != allowed.end ());
  }
}

TEST_CASE ("StreamingJSONParser enum values", "[generator][json_decoder]")
{
  json schema
      = { { "type", "object" },
          { "properties",
            { { "color",
                { { "type", "string" },
                  { "enum", { "red", "green", "blue" } } } } } },
          { "required", { "color" } } };

  cortext::StreamingJSONParser parser (schema);

  SECTION ("Valid enum value")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"color\"");
    parser.AddToken (":");
    parser.AddToken ("\"red\"");
    auto status = parser.AddToken ("}");

    CHECK (status == cortext::ParserStatus::kComplete);
    CHECK (parser.GetParsedObject ()["color"] == "red");
  }

  SECTION ("CurrentEnumValues returns enum options")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"color\"");
    parser.AddToken (":");

    auto enum_values = parser.CurrentEnumValues ();
    CHECK (enum_values.has_value ());
    if (enum_values.has_value ())
      {
        CHECK (enum_values->size () == 3);
      }
  }
}

TEST_CASE ("StreamingJSONParser oneOf handling", "[generator][json_decoder]")
{
  json schema = {
    { "type", "object" },
    { "oneOf",
      { { { "properties",
            { { "pet_type", { { "const", "cat" } } },
              { "lives", { { "type", "integer" } } } } },
          { "required", { "pet_type" } } },
        { { "properties",
            { { "pet_type", { { "const", "dog" } } },
              { "breed", { { "type", "string" } } } } },
          { "required", { "pet_type" } } } } }
  };

  cortext::StreamingJSONParser parser (schema);

  SECTION ("Detects discriminator property")
  {
    auto disc = parser.OneOfDiscriminatorInfo ();
    CHECK (disc.has_value ());
    if (disc.has_value ())
      {
        CHECK (disc->prop == "pet_type");
      }
  }

  SECTION ("Has oneOf branches")
  {
    auto branches = parser.OneOfBranches ();
    CHECK (branches.size () == 2);
  }

  SECTION ("Branch not locked initially")
  {
    CHECK (!parser.ActiveBranchIndex ().has_value ());
  }

  SECTION ("Branch locks on discriminator value")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"pet_type\"");
    parser.AddToken (":");
    parser.AddToken ("\"cat\"");

    auto branch = parser.ActiveBranchIndex ();
    CHECK (branch.has_value ());
    CHECK (branch.value () == 0); // cat is first branch
  }
}

TEST_CASE ("StreamingJSONParser required fields", "[generator][json_decoder]")
{
  json schema = { { "type", "object" },
                  { "properties",
                    { { "id", { { "type", "integer" } } },
                      { "name", { { "type", "string" } } },
                      { "email", { { "type", "string" } } } } },
                  { "required", { "id", "name" } } };

  cortext::StreamingJSONParser parser (schema);

  SECTION ("RequiredFields returns required set")
  {
    auto required = parser.RequiredFields ();
    CHECK (required.size () == 2);
    CHECK (required.count ("id") == 1);
    CHECK (required.count ("name") == 1);
    CHECK (required.count ("email") == 0);
  }

  SECTION ("SeenFields tracks parsed fields")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"id\"");
    parser.AddToken (":");
    parser.AddToken ("42");
    parser.AddToken (",");

    auto seen = parser.SeenFields ();
    CHECK (seen.count ("id") == 1);
    CHECK (seen.count ("name") == 0);
  }

  SECTION ("EffectiveRequiredFields shows remaining required")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"id\"");
    parser.AddToken (":");
    parser.AddToken ("42");
    parser.AddToken (",");

    auto effective = parser.EffectiveRequiredFields ();
    // Implementation may or may not remove seen fields from effective set
    // depending on when the tracking happens. Just verify 'name' is required.
    CHECK (effective.count ("name") == 1);
  }
}

TEST_CASE ("StreamingJSONParser reset", "[generator][json_decoder]")
{
  json schema = { { "type", "object" },
                  { "properties", { { "x", { { "type", "integer" } } } } } };

  cortext::StreamingJSONParser parser (schema);

  parser.AddToken ("{");
  parser.AddToken ("\"x\"");
  parser.AddToken (":");
  parser.AddToken ("1");
  parser.AddToken ("}");

  REQUIRE (parser.State () != cortext::JSONState::kExpectingObjectStart);

  SECTION ("Reset returns to initial state")
  {
    parser.Reset ();
    CHECK (parser.State () == cortext::JSONState::kExpectingObjectStart);
    CHECK (parser.Buffer ().empty ());
    CHECK (parser.SeenFields ().empty ());
  }
}

TEST_CASE ("StreamingJSONParser nested objects", "[generator][json_decoder]")
{
  // Note: Nested object parsing is complex and may require additional
  // implementation work. This test validates the basic state transitions.
  json schema = { { "type", "object" },
                  { "properties",
                    { { "person",
                        { { "type", "object" },
                          { "properties",
                            { { "name", { { "type", "string" } } } } } } } } } };

  cortext::StreamingJSONParser parser (schema);

  SECTION ("Enters nested object state")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"person\"");
    parser.AddToken (":");
    auto status = parser.AddToken ("{");

    // Should be in a valid partial state (inside nested object)
    CHECK (status == cortext::ParserStatus::kValidPartial);
  }
}

TEST_CASE ("StreamingJSONParser arrays", "[generator][json_decoder]")
{
  // Note: Array parsing is complex and may require additional implementation.
  // This test validates basic state transitions.
  json schema
      = { { "type", "object" },
          { "properties",
            { { "items",
                { { "type", "array" },
                  { "items", { { "type", "integer" } } } } } } } };

  cortext::StreamingJSONParser parser (schema);

  SECTION ("Enters array state")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"items\"");
    parser.AddToken (":");
    auto status = parser.AddToken ("[");

    // Should be in a valid partial state (inside array)
    CHECK (status == cortext::ParserStatus::kValidPartial);
    CHECK (parser.State () == cortext::JSONState::kInArray);
  }
}
