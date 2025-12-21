#include <catch2/catch_test_macros.hpp>

#include "cortext/extractor/json_schema_constraint.hpp"

#include "absl/status/status.h"

#include <memory>
#include <string>

namespace
{
using TokenIds = litert::lm::TokenIds;

class FakeTokenizer : public litert::lm::Tokenizer
{
public:
  litert::lm::TokenizerType
  GetTokenizerType () const override
  {
    return litert::lm::TokenizerType::kUnspecified;
  }

  absl::StatusOr<TokenIds>
  TextToTokenIds (absl::string_view text) override
  {
    TokenIds ids;
    ids.reserve (text.size ());
    for (unsigned char c : text)
      ids.push_back (static_cast<int> (c));
    return ids;
  }

  absl::StatusOr<int>
  TokenToId (absl::string_view token) override
  {
    if (token.size () != 1)
      return absl::NotFoundError ("fake tokenizer only supports 1-byte tokens");
    return static_cast<int> (static_cast<unsigned char> (token[0]));
  }

  absl::StatusOr<std::string>
  TokenIdsToText (const TokenIds &token_ids) override
  {
    std::string out;
    out.reserve (token_ids.size ());
    for (int id : token_ids)
      out.push_back (static_cast<char> (id));
    return out;
  }
};

nlohmann::json
BuildLabelSchema ()
{
  nlohmann::json labels = {
    { "type", "array" },
    { "items", { { "type", "string" } } },
    { "minItems", 1 }
  };

  nlohmann::json schema = {
    { "type", "object" },
    { "properties", { { "labels", labels } } },
    { "required", nlohmann::json::array ({ "labels" }) }
  };

  return schema;
}

nlohmann::json
BuildRelationsSchema ()
{
  nlohmann::json labels = {
    { "type", "array" },
    { "items", { { "type", "string" } } },
    { "minItems", 1 }
  };

  nlohmann::json relation_props = {
    { "subject", { { "type", "string" } } },
    { "predicate", { { "type", "string" } } },
    { "object", { { "type", "string" } } }
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
    { "required", nlohmann::json::array ({ "labels", "relations" }) }
  };

  return schema;
}

bool
IsAllowed (const cortext::extractor::JsonSchemaConstraint &constraint,
           const std::unique_ptr<litert::lm::Constraint::State> &state, char c)
{
  auto bitmap_or = constraint.ComputeBitmap (*state);
  REQUIRE (bitmap_or.ok ());
  auto bitmap = std::move (*bitmap_or);
  return bitmap->Get (static_cast<unsigned char> (c));
}

void
Advance (const cortext::extractor::JsonSchemaConstraint &constraint,
         std::unique_ptr<litert::lm::Constraint::State> &state, char c)
{
  auto next_or = constraint.ComputeNext (*state,
                                         static_cast<unsigned char> (c));
  REQUIRE (next_or.ok ());
  state = std::move (*next_or);
}

void
AdvanceString (const cortext::extractor::JsonSchemaConstraint &constraint,
               std::unique_ptr<litert::lm::Constraint::State> &state,
               const std::string &text)
{
  for (char c : text)
    {
      REQUIRE (IsAllowed (constraint, state, c));
      Advance (constraint, state, c);
    }
}
} // namespace

TEST_CASE ("JsonSchemaConstraint enforces required labels key",
           "[extractor][constraint]")
{
  FakeTokenizer tokenizer;
  nlohmann::json schema = BuildLabelSchema ();
  cortext::extractor::JsonSchemaConstraint constraint (schema, tokenizer, 256);

  auto state = constraint.Start ();

  REQUIRE (IsAllowed (constraint, state, '{'));
  REQUIRE_FALSE (IsAllowed (constraint, state, '}'));

  Advance (constraint, state, '{');

  REQUIRE (IsAllowed (constraint, state, '"'));
  REQUIRE_FALSE (IsAllowed (constraint, state, '}'));

  Advance (constraint, state, '"');

  REQUIRE (IsAllowed (constraint, state, 'l'));
  REQUIRE_FALSE (IsAllowed (constraint, state, 'r'));

  AdvanceString (constraint, state, "labels");

  REQUIRE (IsAllowed (constraint, state, '"'));
  Advance (constraint, state, '"');

  REQUIRE (IsAllowed (constraint, state, ':'));
  Advance (constraint, state, ':');

  REQUIRE (IsAllowed (constraint, state, '['));
  REQUIRE_FALSE (IsAllowed (constraint, state, '"'));

  Advance (constraint, state, '[');

  REQUIRE (IsAllowed (constraint, state, '"'));
  REQUIRE_FALSE (IsAllowed (constraint, state, ']'));

  Advance (constraint, state, '"');
  AdvanceString (constraint, state, "Acme");
  Advance (constraint, state, '"');

  REQUIRE (IsAllowed (constraint, state, ']'));
  Advance (constraint, state, ']');

  REQUIRE (IsAllowed (constraint, state, '}'));
  Advance (constraint, state, '}');

  REQUIRE (constraint.IsEnded (*state));
}

TEST_CASE ("JsonSchemaConstraint enforces relation required fields",
           "[extractor][constraint]")
{
  FakeTokenizer tokenizer;
  nlohmann::json schema = BuildRelationsSchema ();
  cortext::extractor::JsonSchemaConstraint constraint (schema, tokenizer, 256);

  auto state = constraint.Start ();

  Advance (constraint, state, '{');
  Advance (constraint, state, '"');
  AdvanceString (constraint, state, "labels");
  Advance (constraint, state, '"');
  Advance (constraint, state, ':');
  Advance (constraint, state, '[');
  Advance (constraint, state, '"');
  AdvanceString (constraint, state, "Acme");
  Advance (constraint, state, '"');
  Advance (constraint, state, ']');
  Advance (constraint, state, ',');
  Advance (constraint, state, '"');
  AdvanceString (constraint, state, "relations");
  Advance (constraint, state, '"');
  Advance (constraint, state, ':');
  Advance (constraint, state, '[');
  Advance (constraint, state, '{');

  REQUIRE_FALSE (IsAllowed (constraint, state, '}'));

  Advance (constraint, state, '"');
  AdvanceString (constraint, state, "subject");
  Advance (constraint, state, '"');
  Advance (constraint, state, ':');
  Advance (constraint, state, '"');
  AdvanceString (constraint, state, "Alice");
  Advance (constraint, state, '"');

  REQUIRE_FALSE (IsAllowed (constraint, state, '}'));

  Advance (constraint, state, ',');
  Advance (constraint, state, '"');
  AdvanceString (constraint, state, "predicate");
  Advance (constraint, state, '"');
  Advance (constraint, state, ':');
  Advance (constraint, state, '"');
  AdvanceString (constraint, state, "works at");
  Advance (constraint, state, '"');

  REQUIRE_FALSE (IsAllowed (constraint, state, '}'));

  Advance (constraint, state, ',');
  Advance (constraint, state, '"');
  AdvanceString (constraint, state, "object");
  Advance (constraint, state, '"');
  Advance (constraint, state, ':');
  Advance (constraint, state, '"');
  AdvanceString (constraint, state, "Acme");
  Advance (constraint, state, '"');

  REQUIRE (IsAllowed (constraint, state, '}'));
  Advance (constraint, state, '}');
  REQUIRE (IsAllowed (constraint, state, ']'));
  Advance (constraint, state, ']');
  REQUIRE (IsAllowed (constraint, state, '}'));
  Advance (constraint, state, '}');
  REQUIRE (constraint.IsEnded (*state));
}
