#include "cortext/generator/constrained_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace cortext
{

namespace
{

/// @brief Check if a string starts with a prefix.
bool
StartsWith (const std::string &str, const std::string &prefix)
{
  return str.size () >= prefix.size ()
         && str.compare (0, prefix.size (), prefix) == 0;
}

} // namespace

struct SchemaConstrainedSampler::Impl
{
  const GemmaTokenizer &tokenizer;
  StreamingJSONParser &parser;
  std::size_t vocab_size;

  // Pre-computed token IDs for JSON structural elements
  std::unordered_map<std::string, std::vector<int64_t>> json_tokens;

  Impl (const GemmaTokenizer &tok, StreamingJSONParser &p)
      : tokenizer (tok), parser (p), vocab_size (tok.VocabSize ())
  {
    BuildTokenVocabulary ();
  }

  std::vector<int64_t>
  GetTokenIds (const std::string &text) const
  {
    auto ids = tokenizer.Encode (text, false);
    // Filter out BOS token (ID 2)
    ids.erase (std::remove (ids.begin (), ids.end (), tokenizer.BosTokenId ()),
               ids.end ());
    return ids;
  }

  std::vector<int64_t>
  UnionTokenIds (const std::vector<std::string> &texts) const
  {
    std::unordered_set<int64_t> seen;
    std::vector<int64_t> out;
    for (const auto &t : texts)
      {
        auto ids = GetTokenIds (t);
        for (int64_t id : ids)
          {
            if (seen.find (id) == seen.end ())
              {
                seen.insert (id);
                out.push_back (id);
              }
          }
      }
    return out;
  }

  void
  BuildTokenVocabulary ()
  {
    // Structural tokens (include leading-space variants)
    json_tokens["{"] = UnionTokenIds ({ "{", " {" });
    json_tokens["}"] = UnionTokenIds ({ "}", " }" });
    json_tokens["["] = UnionTokenIds ({ "[", " [" });
    json_tokens["]"] = UnionTokenIds ({ "]", " ]" });
    json_tokens[":"] = UnionTokenIds ({ ":", " :" });
    json_tokens[","] = UnionTokenIds ({ ",", " ," });
    json_tokens["\""] = UnionTokenIds ({ "\"", " \"" });

    // Boolean letters
    json_tokens["t"] = GetTokenIds ("t");
    json_tokens["r"] = GetTokenIds ("r");
    json_tokens["u"] = GetTokenIds ("u");
    json_tokens["e"] = GetTokenIds ("e");
    json_tokens["f"] = GetTokenIds ("f");
    json_tokens["a"] = GetTokenIds ("a");
    json_tokens["l"] = GetTokenIds ("l");
    json_tokens["s"] = GetTokenIds ("s");
    json_tokens["true"] = UnionTokenIds ({ "true", " true" });
    json_tokens["false"] = UnionTokenIds ({ "false", " false" });

    // Null
    json_tokens["n"] = GetTokenIds ("n");
    json_tokens["null"] = UnionTokenIds ({ "null", " null" });

    // Numbers
    for (int i = 0; i < 10; ++i)
      {
        std::string digit = std::to_string (i);
        json_tokens[digit] = UnionTokenIds ({ digit, " " + digit });
      }
    json_tokens["-"] = UnionTokenIds ({ "-", " -" });
    json_tokens["."] = UnionTokenIds ({ ".", " ." });
    json_tokens["E"] = UnionTokenIds ({ "E", " E" });
    json_tokens["+"] = UnionTokenIds ({ "+", " +" });
  }

  std::vector<float>
  ConstrainLogits (const std::vector<float> &logits) const
  {
    std::vector<float> result = logits;
    const float neg_inf = -std::numeric_limits<float>::infinity ();

    auto allowed_tokens = parser.GetAllowedTokens ();
    std::unordered_set<int64_t> key_boost_ids;

    // Strict numeric gating at value start
    if (parser.State () == JSONState::kExpectingValue)
      {
        const auto &vt = parser.CurrentValueType ();
        if (vt && (*vt == "number" || *vt == "integer"))
          {
            std::unordered_set<int64_t> numeric_ids;
            for (int d = 0; d < 10; ++d)
              {
                std::string digit = std::to_string (d);
                if (json_tokens.count (digit))
                  {
                    for (int64_t id : json_tokens.at (digit))
                      numeric_ids.insert (id);
                  }
              }
            if (json_tokens.count ("-"))
              {
                for (int64_t id : json_tokens.at ("-"))
                  numeric_ids.insert (id);
              }
            if (*vt == "number")
              {
                for (const auto &t : { ".", "e", "E" })
                  {
                    if (json_tokens.count (t))
                      {
                        for (int64_t id : json_tokens.at (t))
                          numeric_ids.insert (id);
                      }
                  }
              }

            if (!numeric_ids.empty ())
              {
                std::fill (result.begin (), result.end (), neg_inf);
                for (int64_t id : numeric_ids)
                  {
                    if (id >= 0
                        && static_cast<std::size_t> (id) < result.size ())
                      result[id] = logits[id];
                  }
                // Boost digits on first char
                if (parser.PartialValue ().empty ())
                  {
                    for (int d = 0; d < 10; ++d)
                      {
                        std::string digit = std::to_string (d);
                        if (json_tokens.count (digit))
                          {
                            for (int64_t id : json_tokens.at (digit))
                              {
                                if (id >= 0
                                    && static_cast<std::size_t> (id)
                                           < result.size ())
                                  result[id] += 4.0f;
                              }
                          }
                      }
                  }
                return result;
              }
          }
      }

    // Special case: if in string value with "ANY", allow all but boost quote
    bool any_allowed = false;
    for (const auto &t : allowed_tokens)
      {
        if (t == "ANY")
          {
            any_allowed = true;
            break;
          }
      }
    if (any_allowed)
      {
        if (json_tokens.count ("\""))
          {
            for (int64_t qid : json_tokens.at ("\""))
              {
                if (qid >= 0 && static_cast<std::size_t> (qid) < result.size ())
                  result[qid] += 10.0f;
              }
          }
        return result;
      }

    // Convert allowed tokens/strings to token IDs
    std::unordered_set<int64_t> valid_token_ids;

    // Handle oneOf discriminator value at EXPECTING_VALUE
    if (parser.State () == JSONState::kExpectingValue
        && parser.OneOfDiscriminatorInfo ()
        && !parser.ActiveBranchIndex ()
        && parser.CurrentKey ()
               == parser.OneOfDiscriminatorInfo ()->prop)
      {
        const auto &disc_vals
            = parser.OneOfDiscriminatorInfo ()->values_by_branch;
        bool has_string = false;
        for (const auto &dv : disc_vals)
          {
            if (dv.is_string ())
              {
                has_string = true;
              }
            else if (dv.is_number ())
              {
                for (const auto &t :
                     { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-",
                       ".", "e", "E" })
                  {
                    if (json_tokens.count (t))
                      for (int64_t id : json_tokens.at (t))
                        valid_token_ids.insert (id);
                  }
              }
            else if (dv.is_boolean ())
              {
                for (const auto &t : { "t", "f" })
                  {
                    if (json_tokens.count (t))
                      for (int64_t id : json_tokens.at (t))
                        valid_token_ids.insert (id);
                  }
              }
            else if (dv.is_null ())
              {
                if (json_tokens.count ("n"))
                  for (int64_t id : json_tokens.at ("n"))
                    valid_token_ids.insert (id);
              }
          }
        if (has_string && json_tokens.count ("\""))
          {
            for (int64_t id : json_tokens.at ("\""))
              valid_token_ids.insert (id);
          }
        if (!valid_token_ids.empty ())
          {
            std::fill (result.begin (), result.end (), neg_inf);
            for (int64_t id : valid_token_ids)
              {
                if (id >= 0 && static_cast<std::size_t> (id) < result.size ())
                  result[id] = logits[id];
              }
            return result;
          }
      }

    // Enum string handling in IN_STRING_VALUE
    if (parser.State () == JSONState::kInStringValue
        && parser.CurrentEnumValues ()
        && parser.CurrentEnumType () == "string")
      {
        const std::string &prefix = parser.PartialValue ();
        for (const auto &item : allowed_tokens)
          {
            if (json_tokens.count (item))
              {
                for (int64_t id : json_tokens.at (item))
                  valid_token_ids.insert (id);
              }
            else
              {
                // Candidate enum value - allow only first token of remaining
                if (StartsWith (item, prefix))
                  {
                    std::string remaining = item.substr (prefix.size ());
                    if (remaining.empty ())
                      continue;
                    auto ids1 = GetTokenIds (remaining);
                    auto ids2 = GetTokenIds (" " + remaining);
                    if (!ids1.empty ())
                      valid_token_ids.insert (ids1[0]);
                    if (!ids2.empty ())
                      valid_token_ids.insert (ids2[0]);
                  }
              }
          }
      }
    // Field name handling in EXPECTING_KEY_STRING
    else if (parser.State () == JSONState::kExpectingKeyString)
      {
        const std::string &current_prefix = parser.PartialKey ();
        for (const auto &item : allowed_tokens)
          {
            if (json_tokens.count (item))
              {
                for (int64_t id : json_tokens.at (item))
                  valid_token_ids.insert (id);
              }
            else
              {
                // Field name - allow first token of remaining
                if (StartsWith (item, current_prefix))
                  {
                    std::string remaining = item.substr (current_prefix.size ());
                    if (remaining.empty ())
                      continue;
                    auto ids1 = GetTokenIds (remaining);
                    if (!ids1.empty ())
                      valid_token_ids.insert (ids1[0]);
                  }
              }
          }
        // Boost preferred fields pre-lock
        if (!parser.ActiveBranchIndex ()
            && !parser.OneOfDiscriminatorInfo ())
          {
            const auto &preferred = parser.PreferredFieldsPrelock ();
            for (std::size_t i = 0; i < std::min (preferred.size (), std::size_t (8)); ++i)
              {
                const auto &name = preferred[i];
                if (StartsWith (name, current_prefix))
                  {
                    std::string remaining = name.substr (current_prefix.size ());
                    if (remaining.empty ())
                      continue;
                    auto ids1 = GetTokenIds (remaining);
                    auto ids2 = GetTokenIds (" " + remaining);
                    if (!ids1.empty ())
                      key_boost_ids.insert (ids1[0]);
                    if (!ids2.empty ())
                      key_boost_ids.insert (ids2[0]);
                  }
              }
          }
      }
    else
      {
        // Normal handling
        for (const auto &item : allowed_tokens)
          {
            if (json_tokens.count (item))
              {
                for (int64_t id : json_tokens.at (item))
                  valid_token_ids.insert (id);
              }
            else
              {
                auto ids = GetTokenIds (item);
                for (int64_t id : ids)
                  valid_token_ids.insert (id);
              }
          }
      }

    // Fallback handling if nothing allowed
    if (valid_token_ids.empty ())
      {
        std::unordered_set<int64_t> fallback_ids;
        JSONState st = parser.State ();

        if (st == JSONState::kExpectingObjectStart)
          {
            if (json_tokens.count ("{"))
              for (int64_t id : json_tokens.at ("{"))
                fallback_ids.insert (id);
          }
        else if (st == JSONState::kExpectingKeyOrClose)
          {
            if (json_tokens.count ("\""))
              for (int64_t id : json_tokens.at ("\""))
                fallback_ids.insert (id);
          }
        else if (st == JSONState::kExpectingKeyString)
          {
            if (json_tokens.count ("\""))
              for (int64_t id : json_tokens.at ("\""))
                fallback_ids.insert (id);
          }
        else if (st == JSONState::kExpectingColon)
          {
            if (json_tokens.count (":"))
              for (int64_t id : json_tokens.at (":"))
                fallback_ids.insert (id);
          }
        else if (st == JSONState::kExpectingValue)
          {
            const auto &vt = parser.CurrentValueType ();
            if (vt)
              {
                if (*vt == "number" || *vt == "integer")
                  {
                    for (const auto &t :
                         { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                           "-", ".", "e", "E" })
                      {
                        if (json_tokens.count (t))
                          for (int64_t id : json_tokens.at (t))
                            fallback_ids.insert (id);
                      }
                  }
                else if (*vt == "string")
                  {
                    if (json_tokens.count ("\""))
                      for (int64_t id : json_tokens.at ("\""))
                        fallback_ids.insert (id);
                  }
                else if (*vt == "boolean")
                  {
                    for (const auto &t : { "t", "f" })
                      {
                        if (json_tokens.count (t))
                          for (int64_t id : json_tokens.at (t))
                            fallback_ids.insert (id);
                      }
                  }
                else if (*vt == "null")
                  {
                    if (json_tokens.count ("n"))
                      for (int64_t id : json_tokens.at ("n"))
                        fallback_ids.insert (id);
                  }
                else if (*vt == "array")
                  {
                    if (json_tokens.count ("["))
                      for (int64_t id : json_tokens.at ("["))
                        fallback_ids.insert (id);
                  }
                else if (*vt == "object")
                  {
                    if (json_tokens.count ("{"))
                      for (int64_t id : json_tokens.at ("{"))
                        fallback_ids.insert (id);
                  }
              }
            else
              {
                // Conservative fallback
                for (const auto &t :
                     { "\"", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                       "-", "t", "f", "n", "{", "[" })
                  {
                    if (json_tokens.count (t))
                      for (int64_t id : json_tokens.at (t))
                        fallback_ids.insert (id);
                  }
              }
          }
        else if (st == JSONState::kExpectingCommaOrClose)
          {
            if (json_tokens.count (","))
              for (int64_t id : json_tokens.at (","))
                fallback_ids.insert (id);
            if (json_tokens.count ("}"))
              for (int64_t id : json_tokens.at ("}"))
                fallback_ids.insert (id);
          }
        else if (st == JSONState::kInStringValue)
          {
            // Let model continue
            return result;
          }
        else if (st == JSONState::kInNumberValue)
          {
            for (const auto &t :
                 { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", ".",
                   "e", "E" })
              {
                if (json_tokens.count (t))
                  for (int64_t id : json_tokens.at (t))
                    fallback_ids.insert (id);
              }
          }
        else if (st == JSONState::kInBooleanValue)
          {
            for (const auto &t : { "t", "f" })
              {
                if (json_tokens.count (t))
                  for (int64_t id : json_tokens.at (t))
                    fallback_ids.insert (id);
              }
          }
        else if (st == JSONState::kInNullValue)
          {
            if (json_tokens.count ("n"))
              for (int64_t id : json_tokens.at ("n"))
                fallback_ids.insert (id);
          }

        if (!fallback_ids.empty ())
          {
            std::fill (result.begin (), result.end (), neg_inf);
            for (int64_t id : fallback_ids)
              {
                if (id >= 0 && static_cast<std::size_t> (id) < result.size ())
                  result[id] = logits[id];
              }
            return result;
          }

        // No fallback, allow all
        return result;
      }

    // Create mask
    std::fill (result.begin (), result.end (), neg_inf);
    for (int64_t id : valid_token_ids)
      {
        if (id >= 0 && static_cast<std::size_t> (id) < result.size ())
          result[id] = logits[id];
      }

    // Boost terminator tokens
    std::vector<int64_t> boost_tokens;
    if (parser.State () == JSONState::kInNumberValue)
      {
        if (json_tokens.count (","))
          boost_tokens.insert (boost_tokens.end (), json_tokens.at (",").begin (),
                               json_tokens.at (",").end ());
        if (json_tokens.count ("}"))
          boost_tokens.insert (boost_tokens.end (), json_tokens.at ("}").begin (),
                               json_tokens.at ("}").end ());
      }
    else if (parser.State () == JSONState::kInStringValue
             && parser.CurrentEnumValues ()
             && parser.CurrentEnumType () == "string")
      {
        for (const auto &t : allowed_tokens)
          {
            if (t == "\"" && json_tokens.count ("\""))
              {
                boost_tokens.insert (boost_tokens.end (),
                                     json_tokens.at ("\"").begin (),
                                     json_tokens.at ("\"").end ());
              }
          }
      }
    else if (parser.State () == JSONState::kExpectingValue)
      {
        const auto &vt = parser.CurrentValueType ();
        if (vt && (*vt == "number" || *vt == "integer"))
          {
            for (int d = 0; d < 10; ++d)
              {
                std::string digit = std::to_string (d);
                if (json_tokens.count (digit))
                  boost_tokens.insert (boost_tokens.end (),
                                       json_tokens.at (digit).begin (),
                                       json_tokens.at (digit).end ());
              }
            if (json_tokens.count ("-"))
              boost_tokens.insert (boost_tokens.end (),
                                   json_tokens.at ("-").begin (),
                                   json_tokens.at ("-").end ());
          }
      }

    for (int64_t tid : boost_tokens)
      {
        if (valid_token_ids.count (tid))
          {
            if (tid >= 0 && static_cast<std::size_t> (tid) < result.size ())
              result[tid] += 10.0f;
          }
      }

    // Apply key boosts
    for (int64_t tid : key_boost_ids)
      {
        if (valid_token_ids.count (tid))
          {
            if (tid >= 0 && static_cast<std::size_t> (tid) < result.size ())
              result[tid] += 3.0f;
          }
      }

    return result;
  }
};

SchemaConstrainedSampler::SchemaConstrainedSampler (
    const GemmaTokenizer &tokenizer, StreamingJSONParser &parser)
    : impl_ (std::make_unique<Impl> (tokenizer, parser))
{
}

SchemaConstrainedSampler::~SchemaConstrainedSampler () = default;
SchemaConstrainedSampler::SchemaConstrainedSampler (
    SchemaConstrainedSampler &&) noexcept
    = default;
SchemaConstrainedSampler &
SchemaConstrainedSampler::operator= (SchemaConstrainedSampler &&) noexcept
    = default;

std::vector<float>
SchemaConstrainedSampler::ConstrainLogits (
    const std::vector<float> &logits) const
{
  return impl_->ConstrainLogits (logits);
}

std::vector<int64_t>
SchemaConstrainedSampler::GetTokenIds (const std::string &text) const
{
  return impl_->GetTokenIds (text);
}

std::size_t
SchemaConstrainedSampler::VocabSize () const
{
  return impl_->vocab_size;
}

} // namespace cortext
