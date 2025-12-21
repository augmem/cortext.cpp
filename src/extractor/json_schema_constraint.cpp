#include "cortext/extractor/json_schema_constraint.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include "absl/status/status.h"

#include "runtime/components/constrained_decoding/bitmap.h"

namespace cortext::extractor
{

namespace
{
class AllowedTokenBitmap : public litert::lm::Bitmap
{
public:
  explicit AllowedTokenBitmap (std::unordered_set<int> ids)
      : allowed_ids_ (std::move (ids))
  {
  }

  bool
  Get (int index) const override
  {
    return allowed_ids_.find (index) != allowed_ids_.end ();
  }

private:
  std::unordered_set<int> allowed_ids_;
};

bool
StartsWith (const std::string &str, const std::string &prefix)
{
  return str.size () >= prefix.size ()
         && str.compare (0, prefix.size (), prefix) == 0;
}
} // namespace

JsonSchemaConstraint::TokenVocabulary::TokenVocabulary (
    litert::lm::Tokenizer &tokenizer_in)
    : tokenizer (tokenizer_in)
{
  auto add_union = [this] (const std::string &key,
                           std::vector<std::string> variants) {
    json_tokens.emplace (key, UnionTokenIds (variants));
  };

  add_union ("{", { "{", " {" });
  add_union ("}", { "}", " }" });
  add_union ("[", { "[", " [" });
  add_union ("]", { "]", " ]" });
  add_union (":", { ":", " :" });
  add_union (",", { ",", " ," });
  add_union ("\"", { "\"", " \"" });

  json_tokens["t"] = TokenIdsForText ("t");
  json_tokens["r"] = TokenIdsForText ("r");
  json_tokens["u"] = TokenIdsForText ("u");
  json_tokens["e"] = TokenIdsForText ("e");
  json_tokens["f"] = TokenIdsForText ("f");
  json_tokens["a"] = TokenIdsForText ("a");
  json_tokens["l"] = TokenIdsForText ("l");
  json_tokens["s"] = TokenIdsForText ("s");
  add_union ("true", { "true", " true" });
  add_union ("false", { "false", " false" });

  json_tokens["n"] = TokenIdsForText ("n");
  add_union ("null", { "null", " null" });

  for (int i = 0; i < 10; ++i)
    {
      std::string digit = std::to_string (i);
      add_union (digit, { digit, " " + digit });
    }
  add_union ("-", { "-", " -" });
  add_union (".", { ".", " ." });
  add_union ("E", { "E", " E" });
  add_union ("e", { "e", " e" });
  add_union ("+", { "+", " +" });
}

std::vector<int>
JsonSchemaConstraint::TokenVocabulary::TokenIdsForText (
    const std::string &text) const
{
  auto ids_or = tokenizer.TextToTokenIds (text);
  if (!ids_or.ok ())
    {
      return {};
    }
  return *ids_or;
}

std::vector<int>
JsonSchemaConstraint::TokenVocabulary::UnionTokenIds (
    const std::vector<std::string> &texts) const
{
  std::unordered_set<int> seen;
  std::vector<int> out;
  for (const auto &text : texts)
    {
      auto ids = TokenIdsForText (text);
      for (int id : ids)
        {
          if (seen.insert (id).second)
            {
              out.push_back (id);
            }
        }
    }
  return out;
}

JsonSchemaConstraint::JsonSchemaConstraint (nlohmann::json schema,
                                            litert::lm::Tokenizer &tokenizer,
                                            int vocabulary_size)
    : schema_ (std::move (schema)),
      tokenizer_ (tokenizer),
      vocabulary_size_ (vocabulary_size),
      vocab_ (tokenizer)
{
}

std::unique_ptr<litert::lm::Constraint::State>
JsonSchemaConstraint::Start () const
{
  return std::make_unique<JsonState> ();
}

bool
JsonSchemaConstraint::IsEnded (const State &state) const
{
  const auto &json_state = static_cast<const JsonState &> (state);
  ParserStatus status = ParserStatus::kValidPartial;
  auto parser = BuildParser (json_state.decoded_text, status);
  return parser.has_value () && status == ParserStatus::kComplete;
}

int
JsonSchemaConstraint::GetVocabularySize () const
{
  return vocabulary_size_;
}

absl::StatusOr<std::unique_ptr<litert::lm::Constraint::State>>
JsonSchemaConstraint::ComputeNext (const State &state, int token) const
{
  const auto &prev = static_cast<const JsonState &> (state);
  auto next = std::make_unique<JsonState> ();
  next->token_ids = prev.token_ids;
  next->token_ids.push_back (token);
  next->decoded_text = prev.decoded_text;

  auto decoded_or = tokenizer_.TokenIdsToText (next->token_ids);
  if (decoded_or.ok ())
    {
      next->decoded_text = *decoded_or;
    }
  else if (!litert::lm::Tokenizer::IsIncompleteBpeSequence (decoded_or))
    {
      return decoded_or.status ();
    }

  return next;
}

absl::StatusOr<std::unique_ptr<litert::lm::Bitmap>>
JsonSchemaConstraint::ComputeBitmap (const State &state) const
{
  const auto &json_state = static_cast<const JsonState &> (state);
  ParserStatus status = ParserStatus::kValidPartial;
  auto parser = BuildParser (json_state.decoded_text, status);
  if (!parser.has_value () || status == ParserStatus::kInvalid)
    {
      return std::make_unique<litert::lm::AllAllowedBitmap> ();
    }

  auto allowed = BuildAllowedTokens (*parser);
  if (allowed.allow_all)
    {
      return std::make_unique<litert::lm::AllAllowedBitmap> ();
    }
  return std::make_unique<AllowedTokenBitmap> (std::move (allowed.ids));
}

std::optional<StreamingJSONParser>
JsonSchemaConstraint::BuildParser (const std::string &buffer,
                                   ParserStatus &status) const
{
  try
    {
      StreamingJSONParser parser (schema_);
      if (!buffer.empty ())
        {
          status = parser.AddToken (buffer);
        }
      return parser;
    }
  catch (const std::exception &)
    {
      status = ParserStatus::kInvalid;
      return std::nullopt;
    }
}

JsonSchemaConstraint::AllowedTokens
JsonSchemaConstraint::BuildAllowedTokens (const StreamingJSONParser &parser) const
{
  AllowedTokens out;
  const auto allowed_tokens = parser.GetAllowedTokens ();
  for (const auto &token : allowed_tokens)
    {
      if (token == "ANY")
        {
          out.allow_all = true;
          return out;
        }
    }

  // Strict numeric gating at value start.
  if (parser.State () == JSONState::kExpectingValue)
    {
      const auto &vt = parser.CurrentValueType ();
      if (vt && (*vt == "number" || *vt == "integer"))
        {
          for (int d = 0; d < 10; ++d)
            {
              std::string digit = std::to_string (d);
              auto it = vocab_.json_tokens.find (digit);
              if (it != vocab_.json_tokens.end ())
                {
                  out.ids.insert (it->second.begin (), it->second.end ());
                }
            }
          for (const auto &t : { "-", ".", "e", "E" })
            {
              auto it = vocab_.json_tokens.find (t);
              if (it != vocab_.json_tokens.end ())
                {
                  out.ids.insert (it->second.begin (), it->second.end ());
                }
            }
          return out;
        }
    }

  // Enum string handling in string values.
  if (parser.State () == JSONState::kInStringValue
      && parser.CurrentEnumValues ()
      && parser.CurrentEnumType () == "string")
    {
      const std::string &prefix = parser.PartialValue ();
      for (const auto &item : allowed_tokens)
        {
          auto it = vocab_.json_tokens.find (item);
          if (it != vocab_.json_tokens.end ())
            {
              out.ids.insert (it->second.begin (), it->second.end ());
              continue;
            }
          if (StartsWith (item, prefix))
            {
              std::string remaining = item.substr (prefix.size ());
              if (remaining.empty ())
                continue;
              auto ids1 = vocab_.TokenIdsForText (remaining);
              auto ids2 = vocab_.TokenIdsForText (" " + remaining);
              if (!ids1.empty ())
                out.ids.insert (ids1.front ());
              if (!ids2.empty ())
                out.ids.insert (ids2.front ());
            }
        }
      return out;
    }

  if (parser.State () == JSONState::kExpectingKeyString)
    {
      const std::string &prefix = parser.PartialKey ();
      for (const auto &item : allowed_tokens)
        {
          auto it = vocab_.json_tokens.find (item);
          if (it != vocab_.json_tokens.end ())
            {
              out.ids.insert (it->second.begin (), it->second.end ());
              continue;
            }
          if (StartsWith (item, prefix))
            {
              std::string remaining = item.substr (prefix.size ());
              if (remaining.empty ())
                continue;
              auto ids1 = vocab_.TokenIdsForText (remaining);
              if (!ids1.empty ())
                out.ids.insert (ids1.front ());
            }
        }
      return out;
    }

  for (const auto &item : allowed_tokens)
    {
      auto it = vocab_.json_tokens.find (item);
      if (it != vocab_.json_tokens.end ())
        {
          out.ids.insert (it->second.begin (), it->second.end ());
        }
      else
        {
          auto ids = vocab_.TokenIdsForText (item);
          out.ids.insert (ids.begin (), ids.end ());
        }
    }

  if (!out.ids.empty ())
    {
      return out;
    }

  // Conservative fallback if nothing allowed.
  const JSONState st = parser.State ();
  if (st == JSONState::kExpectingObjectStart)
    {
      auto it = vocab_.json_tokens.find ("{");
      if (it != vocab_.json_tokens.end ())
        out.ids.insert (it->second.begin (), it->second.end ());
    }
  else if (st == JSONState::kExpectingKeyOrClose
           || st == JSONState::kExpectingKeyString)
    {
      auto it = vocab_.json_tokens.find ("\"");
      if (it != vocab_.json_tokens.end ())
        out.ids.insert (it->second.begin (), it->second.end ());
    }
  else if (st == JSONState::kExpectingColon)
    {
      auto it = vocab_.json_tokens.find (":");
      if (it != vocab_.json_tokens.end ())
        out.ids.insert (it->second.begin (), it->second.end ());
    }
  else if (st == JSONState::kExpectingValue)
    {
      const auto &vt = parser.CurrentValueType ();
      if (vt)
        {
          if (*vt == "number" || *vt == "integer")
            {
              for (const auto &t : { "0", "1", "2", "3", "4", "5", "6", "7",
                                     "8", "9", "-", ".", "e", "E" })
                {
                  auto it = vocab_.json_tokens.find (t);
                  if (it != vocab_.json_tokens.end ())
                    out.ids.insert (it->second.begin (), it->second.end ());
                }
            }
          else if (*vt == "string")
            {
              auto it = vocab_.json_tokens.find ("\"");
              if (it != vocab_.json_tokens.end ())
                out.ids.insert (it->second.begin (), it->second.end ());
            }
          else if (*vt == "boolean")
            {
              for (const auto &t : { "t", "f" })
                {
                  auto it = vocab_.json_tokens.find (t);
                  if (it != vocab_.json_tokens.end ())
                    out.ids.insert (it->second.begin (), it->second.end ());
                }
            }
          else if (*vt == "null")
            {
              auto it = vocab_.json_tokens.find ("n");
              if (it != vocab_.json_tokens.end ())
                out.ids.insert (it->second.begin (), it->second.end ());
            }
          else if (*vt == "array")
            {
              auto it = vocab_.json_tokens.find ("[");
              if (it != vocab_.json_tokens.end ())
                out.ids.insert (it->second.begin (), it->second.end ());
            }
          else if (*vt == "object")
            {
              auto it = vocab_.json_tokens.find ("{");
              if (it != vocab_.json_tokens.end ())
                out.ids.insert (it->second.begin (), it->second.end ());
            }
        }
      else
        {
          for (const auto &t :
               { "\"", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-",
                 "t", "f", "n", "{", "[" })
            {
              auto it = vocab_.json_tokens.find (t);
              if (it != vocab_.json_tokens.end ())
                out.ids.insert (it->second.begin (), it->second.end ());
            }
        }
    }
  else if (st == JSONState::kExpectingCommaOrClose)
    {
      for (const auto &t : { ",", "}" })
        {
          auto it = vocab_.json_tokens.find (t);
          if (it != vocab_.json_tokens.end ())
            out.ids.insert (it->second.begin (), it->second.end ());
        }
    }
  else if (st == JSONState::kInNumberValue)
    {
      for (const auto &t :
           { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", ".", "e",
             "E" })
        {
          auto it = vocab_.json_tokens.find (t);
          if (it != vocab_.json_tokens.end ())
            out.ids.insert (it->second.begin (), it->second.end ());
        }
    }
  else if (st == JSONState::kInBooleanValue)
    {
      for (const auto &t : { "t", "f" })
        {
          auto it = vocab_.json_tokens.find (t);
          if (it != vocab_.json_tokens.end ())
            out.ids.insert (it->second.begin (), it->second.end ());
        }
    }
  else if (st == JSONState::kInNullValue)
    {
      auto it = vocab_.json_tokens.find ("n");
      if (it != vocab_.json_tokens.end ())
        out.ids.insert (it->second.begin (), it->second.end ());
    }

  return out;
}

} // namespace cortext::extractor
