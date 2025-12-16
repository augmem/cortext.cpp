#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cortext/generator/constrained_sampler.hpp"
#include "cortext/generator/gemma_tokenizer.hpp"
#include "cortext/generator/json_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

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

std::string
FindTokenizerPath ()
{
  auto models_dir = GetModelsDir ();
  if (models_dir.empty ())
    return "";

  auto tokenizer_path = models_dir + "/tokenizer.json";
  if (std::filesystem::exists (tokenizer_path))
    return tokenizer_path;

  return "";
}

} // namespace

TEST_CASE ("SchemaConstrainedSampler construction",
           "[generator][constrained_sampler]")
{
  auto tokenizer_path = FindTokenizerPath ();
  if (tokenizer_path.empty ())
    {
      SKIP ("Tokenizer not found");
    }

  cortext::GemmaTokenizer tokenizer (tokenizer_path);

  SECTION ("Constructs with simple object schema")
  {
    json schema = { { "type", "object" },
                    { "properties", { { "name", { { "type", "string" } } } } },
                    { "required", { "name" } } };

    cortext::StreamingJSONParser parser (schema);
    REQUIRE_NOTHROW (cortext::SchemaConstrainedSampler (tokenizer, parser));
  }

  SECTION ("Constructs with enum schema")
  {
    json schema
        = { { "type", "object" },
            { "properties",
              { { "color",
                  { { "type", "string" },
                    { "enum", { "red", "green", "blue" } } } } } },
            { "required", { "color" } } };

    cortext::StreamingJSONParser parser (schema);
    REQUIRE_NOTHROW (cortext::SchemaConstrainedSampler (tokenizer, parser));
  }

  SECTION ("Constructs with oneOf schema")
  {
    json schema = {
      { "type", "object" },
      { "oneOf",
        { { { "properties",
              { { "type", { { "const", "cat" } } },
                { "lives", { { "type", "integer" } } } } },
            { "required", { "type" } } },
          { { "properties",
              { { "type", { { "const", "dog" } } },
                { "breed", { { "type", "string" } } } } },
            { "required", { "type" } } } } }
    };

    cortext::StreamingJSONParser parser (schema);
    REQUIRE_NOTHROW (cortext::SchemaConstrainedSampler (tokenizer, parser));
  }
}

TEST_CASE ("SchemaConstrainedSampler vocab size",
           "[generator][constrained_sampler]")
{
  auto tokenizer_path = FindTokenizerPath ();
  if (tokenizer_path.empty ())
    {
      SKIP ("Tokenizer not found");
    }

  cortext::GemmaTokenizer tokenizer (tokenizer_path);

  json schema = { { "type", "object" },
                  { "properties", { { "x", { { "type", "integer" } } } } } };

  cortext::StreamingJSONParser parser (schema);
  cortext::SchemaConstrainedSampler sampler (tokenizer, parser);

  SECTION ("VocabSize matches tokenizer")
  {
    CHECK (sampler.VocabSize () == tokenizer.VocabSize ());
    CHECK (sampler.VocabSize () > 100000); // Gemma has large vocab
  }
}

TEST_CASE ("SchemaConstrainedSampler token ID lookup",
           "[generator][constrained_sampler]")
{
  auto tokenizer_path = FindTokenizerPath ();
  if (tokenizer_path.empty ())
    {
      SKIP ("Tokenizer not found");
    }

  cortext::GemmaTokenizer tokenizer (tokenizer_path);

  json schema = { { "type", "object" },
                  { "properties", { { "x", { { "type", "integer" } } } } } };

  cortext::StreamingJSONParser parser (schema);
  cortext::SchemaConstrainedSampler sampler (tokenizer, parser);

  SECTION ("GetTokenIds returns valid IDs for JSON tokens")
  {
    auto brace_ids = sampler.GetTokenIds ("{");
    CHECK (!brace_ids.empty ());

    auto quote_ids = sampler.GetTokenIds ("\"");
    CHECK (!quote_ids.empty ());

    auto colon_ids = sampler.GetTokenIds (":");
    CHECK (!colon_ids.empty ());
  }

  SECTION ("GetTokenIds returns valid IDs for text")
  {
    auto hello_ids = sampler.GetTokenIds ("hello");
    CHECK (!hello_ids.empty ());
  }
}

TEST_CASE ("SchemaConstrainedSampler logit masking",
           "[generator][constrained_sampler]")
{
  auto tokenizer_path = FindTokenizerPath ();
  if (tokenizer_path.empty ())
    {
      SKIP ("Tokenizer not found");
    }

  cortext::GemmaTokenizer tokenizer (tokenizer_path);

  json schema = { { "type", "object" },
                  { "properties", { { "name", { { "type", "string" } } } } },
                  { "required", { "name" } } };

  cortext::StreamingJSONParser parser (schema);
  cortext::SchemaConstrainedSampler sampler (tokenizer, parser);

  SECTION ("Initial state constrains to opening brace")
  {
    std::vector<float> logits (sampler.VocabSize (), 0.0f);
    auto constrained = sampler.ConstrainLogits (logits);

    CHECK (constrained.size () == logits.size ());

    // Count how many tokens are not masked
    int allowed_count = 0;
    for (float logit : constrained)
      {
        if (logit > -std::numeric_limits<float>::infinity ())
          {
            allowed_count++;
          }
      }

    // Should only allow opening brace tokens
    CHECK (allowed_count > 0);
    CHECK (allowed_count < 100); // Should be very constrained
  }

  SECTION ("After opening brace allows keys")
  {
    parser.AddToken ("{");

    std::vector<float> logits (sampler.VocabSize (), 0.0f);
    auto constrained = sampler.ConstrainLogits (logits);

    // Count allowed tokens
    int allowed_count = 0;
    for (float logit : constrained)
      {
        if (logit > -std::numeric_limits<float>::infinity ())
          {
            allowed_count++;
          }
      }

    CHECK (allowed_count > 0);
  }

  SECTION ("After key allows colon")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"name\"");

    std::vector<float> logits (sampler.VocabSize (), 0.0f);
    auto constrained = sampler.ConstrainLogits (logits);

    // Find colon token IDs
    auto colon_ids = sampler.GetTokenIds (":");

    // At least one colon variant should be allowed
    bool colon_allowed = false;
    for (auto id : colon_ids)
      {
        if (id < static_cast<int64_t> (constrained.size ())
            && constrained[id] > -std::numeric_limits<float>::infinity ())
          {
            colon_allowed = true;
            break;
          }
      }

    CHECK (colon_allowed);
  }
}

TEST_CASE ("SchemaConstrainedSampler enum constraining",
           "[generator][constrained_sampler]")
{
  auto tokenizer_path = FindTokenizerPath ();
  if (tokenizer_path.empty ())
    {
      SKIP ("Tokenizer not found");
    }

  cortext::GemmaTokenizer tokenizer (tokenizer_path);

  json schema
      = { { "type", "object" },
          { "properties",
            { { "color",
                { { "type", "string" },
                  { "enum", { "red", "green", "blue" } } } } } },
          { "required", { "color" } } };

  cortext::StreamingJSONParser parser (schema);
  cortext::SchemaConstrainedSampler sampler (tokenizer, parser);

  SECTION ("After key with enum, logits are constrained")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"color\"");
    parser.AddToken (":");
    parser.AddToken ("\"");

    // Now should be constrained to enum prefixes
    std::vector<float> logits (sampler.VocabSize (), 0.0f);
    auto constrained = sampler.ConstrainLogits (logits);

    // Count how many tokens are allowed (not -inf)
    int allowed_count = 0;
    for (float logit : constrained)
      {
        if (logit > -std::numeric_limits<float>::infinity ())
          {
            allowed_count++;
          }
      }

    // Should be constrained (fewer than all tokens allowed)
    CHECK (allowed_count > 0);
    CHECK (allowed_count < static_cast<int> (sampler.VocabSize ()));
  }
}

TEST_CASE ("SchemaConstrainedSampler parser state integration",
           "[generator][constrained_sampler][integration]")
{
  auto tokenizer_path = FindTokenizerPath ();
  if (tokenizer_path.empty ())
    {
      SKIP ("Tokenizer not found");
    }

  cortext::GemmaTokenizer tokenizer (tokenizer_path);

  json schema = { { "type", "object" },
                  { "properties",
                    { { "count", { { "type", "integer" } } },
                      { "name", { { "type", "string" } } } } },
                  { "required", { "count" } } };

  cortext::StreamingJSONParser parser (schema);
  cortext::SchemaConstrainedSampler sampler (tokenizer, parser);

  SECTION ("Full JSON parsing flow with constraint checks")
  {
    // Parse: {"count": 42}
    parser.AddToken ("{");
    CHECK (parser.State () == cortext::JSONState::kExpectingKeyOrClose);

    parser.AddToken ("\"count\"");
    CHECK (parser.State () == cortext::JSONState::kExpectingColon);

    parser.AddToken (":");
    CHECK (parser.State () == cortext::JSONState::kExpectingValue);

    // At this point, expecting an integer value
    std::vector<float> logits (sampler.VocabSize (), 0.0f);
    auto constrained = sampler.ConstrainLogits (logits);

    // Digits should be allowed
    auto digit_ids = sampler.GetTokenIds ("4");
    bool digit_allowed = false;
    for (auto id : digit_ids)
      {
        if (id < static_cast<int64_t> (constrained.size ())
            && constrained[id] > -std::numeric_limits<float>::infinity ())
          {
            digit_allowed = true;
            break;
          }
      }

    CHECK (digit_allowed);

    parser.AddToken ("42");
    auto status = parser.AddToken ("}");

    CHECK (status == cortext::ParserStatus::kComplete);
  }
}

TEST_CASE ("SchemaConstrainedSampler with boolean schema",
           "[generator][constrained_sampler]")
{
  auto tokenizer_path = FindTokenizerPath ();
  if (tokenizer_path.empty ())
    {
      SKIP ("Tokenizer not found");
    }

  cortext::GemmaTokenizer tokenizer (tokenizer_path);

  json schema
      = { { "type", "object" },
          { "properties", { { "active", { { "type", "boolean" } } } } },
          { "required", { "active" } } };

  cortext::StreamingJSONParser parser (schema);
  cortext::SchemaConstrainedSampler sampler (tokenizer, parser);

  SECTION ("Boolean value position allows true/false")
  {
    parser.AddToken ("{");
    parser.AddToken ("\"active\"");
    parser.AddToken (":");

    std::vector<float> logits (sampler.VocabSize (), 0.0f);
    auto constrained = sampler.ConstrainLogits (logits);

    // 't' (start of true) should be allowed
    auto t_ids = sampler.GetTokenIds ("t");
    bool t_allowed = false;
    for (auto id : t_ids)
      {
        if (id < static_cast<int64_t> (constrained.size ())
            && constrained[id] > -std::numeric_limits<float>::infinity ())
          {
            t_allowed = true;
            break;
          }
      }

    // 'f' (start of false) should be allowed
    auto f_ids = sampler.GetTokenIds ("f");
    bool f_allowed = false;
    for (auto id : f_ids)
      {
        if (id < static_cast<int64_t> (constrained.size ())
            && constrained[id] > -std::numeric_limits<float>::infinity ())
          {
            f_allowed = true;
            break;
          }
      }

    CHECK ((t_allowed || f_allowed));
  }
}
