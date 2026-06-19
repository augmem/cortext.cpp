#include "cortext/internal/cancellation.hpp"
#include "cortext/operations/process_extraction_results.hpp"

#include "../store/facts.hpp"
#include "evidence_confidence.hpp"
#include "cortext/store/store.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/encoder/encoder.hpp"
#include "cortext/extractor/extractor.hpp"
#include "cortext/extractor/gemma_extractor.hpp"
#include "cortext/operations/extraction.hpp"
#include "cortext/operations/label_utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <any>
#include <chrono>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <cstring>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <Eigen/Dense>

namespace cortext::operations
{

namespace
{

/// @brief Add a write instruction to the transaction.
void
AddWrite (Transaction &tx, const std::string &q,
          const std::vector<std::any> &p = {})
{
  tx.Execute (q, p);
}

/// @brief Default JSON schema for extraction.
const nlohmann::json kExtractionSchema = nlohmann::json::parse (R"({
  "type": "object",
  "properties": {
    "labels": {
      "type": "array",
      "minItems": 0,
      "items": {"type": "string"}
    },
    "relations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "subject": {"type": "string"},
          "predicate": {
            "type": "string",
            "enum": [
              "co_occurs",
              "implies",
              "contradicts",
              "reinforces",
              "causes",
              "similar_to"
            ]
          },
          "object": {"type": "string"},
          "confidence": {"type": "number"}
        },
        "required": ["subject", "predicate", "object"]
      }
    },
    "facts": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "subject": {"type": "string"},
          "predicate": {"type": "string"},
          "object": {"type": "string"},
          "confidence": {"type": "number"},
          "valid_start_ts": {"type": "integer"},
          "valid_end_ts": {"type": "integer"}
        },
        "required": ["subject", "predicate", "object"]
      }
    }
  },
  "required": ["labels", "relations", "facts"]
})");

bool
UseLegacyLabelFrequencyGate ()
{
  const char *value = std::getenv ("CORTEXT_ABLATE_LEGACY_LABEL_GATE");
  return value != nullptr && std::string (value) == "1";
}

bool
FactWritesDisabled ()
{
  const char *value = std::getenv ("CORTEXT_DISABLE_FACTS");
  if (value == nullptr)
    {
      return false;
    }
  std::string text (value);
  std::transform (text.begin (), text.end (), text.begin (),
                  [] (unsigned char c) {
                    return static_cast<char> (std::tolower (c));
                  });
  return text == "1" || text == "true" || text == "yes" || text == "on";
}

constexpr int kEmbeddingDim = 256;

std::optional<Eigen::VectorXf>
LoadEmbedding (Transaction &tx, long long embedding_id,
               int expected_dim = kEmbeddingDim)
{
  auto rows = tx.Execute (
      "SELECT embedding FROM embeddings WHERE embedding_id = ?",
      { embedding_id });
  if (rows.empty ())
    {
      return std::nullopt;
    }

  auto it = rows[0].find ("embedding");
  if (it == rows[0].end ())
    {
      return std::nullopt;
    }

  Eigen::VectorXf out;
  if (!core::DecodeFloatBlob (it->second, expected_dim, out))
    {
      return std::nullopt;
    }
  return out;
}

std::optional<std::vector<float>>
LoadEmbeddingVector (Transaction &tx, long long embedding_id,
                     int expected_dim = kEmbeddingDim)
{
  auto emb = LoadEmbedding (tx, embedding_id, expected_dim);
  if (!emb.has_value ())
    {
      return std::nullopt;
    }
  std::vector<float> out (static_cast<size_t> (emb->size ()));
  Eigen::Map<Eigen::VectorXf> out_vec (out.data (),
                                       static_cast<int> (out.size ()));
  out_vec = *emb;
  return out;
}

std::optional<std::vector<float>>
LoadAttachedLabelBankEmbedding (Transaction &tx, const std::string &label_key,
                                int expected_dim = kEmbeddingDim)
{
  if (label_key.empty ())
    {
      return std::nullopt;
    }
  try
    {
      auto rows = tx.Execute (
          "SELECT embedding FROM cortext_label_bank.label_bank_vec "
          "WHERE key = ? LIMIT 1",
          { label_key });
      if (rows.empty ())
        {
          return std::nullopt;
        }
      auto it = rows[0].find ("embedding");
      if (it == rows[0].end ())
        {
          return std::nullopt;
        }
      Eigen::VectorXf emb;
      if (!core::DecodeFloatBlob (it->second, expected_dim, emb))
        {
          return std::nullopt;
        }
      std::vector<float> out (static_cast<size_t> (emb.size ()));
      Eigen::Map<Eigen::VectorXf> out_vec (out.data (),
                                           static_cast<int> (out.size ()));
      out_vec = emb;
      return out;
    }
  catch (...)
    {
      return std::nullopt;
    }
}

double
ComputeLabelSalience (const std::vector<float> *label_embedding,
                      const std::optional<Eigen::VectorXf> &summary_embedding,
                      double focus, double sensitivity, double stability)
{
  if (!summary_embedding.has_value () || summary_embedding->size () == 0
      || label_embedding == nullptr || label_embedding->empty ())
    {
      return core::LabelSalienceFallback (focus, sensitivity, stability);
    }
  if (static_cast<int> (label_embedding->size ()) != summary_embedding->size ())
    {
      return core::LabelSalienceFallback (focus, sensitivity, stability);
    }

  Eigen::Map<const Eigen::VectorXf> label_vec (
      label_embedding->data (),
      static_cast<int> (label_embedding->size ()));
  const double cos = core::CosineSimilarity (label_vec, *summary_embedding);
  return core::Map01 (cos);
}

std::optional<std::vector<float>>
EncodeLabelEmbedding (const std::string &label, Encoder &encoder)
{
  std::vector<float> embedding;
  try
    {
      encoder.EncodeText (label, embedding);
    }
  catch (const std::exception &)
    {
      return std::nullopt;
    }
  if (embedding.empty () || embedding.size () != kEmbeddingDim)
    {
      if (embedding.size () > kEmbeddingDim)
        {
          embedding.resize (kEmbeddingDim);
          double norm_sq = 0.0;
          for (float value : embedding)
            {
              norm_sq += static_cast<double> (value) * value;
            }
          const double norm = std::sqrt (norm_sq);
          if (norm > 1e-12 && std::isfinite (norm))
            {
              const float inv_norm = static_cast<float> (1.0 / norm);
              for (float &value : embedding)
                {
                  value *= inv_norm;
                }
            }
        }
      else
        {
          return std::nullopt;
        }
    }
  return embedding;
}

std::string
NormalizePredicate (const std::string &raw)
{
  std::string norm;
  norm.reserve (raw.size ());
  for (unsigned char c : raw)
    {
      if (std::isalnum (c))
        {
          norm.push_back (static_cast<char> (std::tolower (c)));
        }
      else if (c == '-' || c == ' ' || c == '_')
        {
          norm.push_back ('_');
        }
    }
  // Collapse repeated underscores.
  norm.erase (std::unique (norm.begin (), norm.end (),
                           [] (char a, char b) {
                             return a == '_' && b == '_';
                           }),
              norm.end ());
  if (!norm.empty () && norm.front () == '_')
    {
      norm.erase (norm.begin ());
    }
  if (!norm.empty () && norm.back () == '_')
    {
      norm.pop_back ();
    }
  return norm;
}

std::string
MapEdgeType (const std::string &predicate)
{
  const std::string norm = NormalizePredicate (predicate);
  if (norm == "co_occurs" || norm == "co_occurs_with" || norm == "cooccurs"
      || norm == "co_occur" || norm == "related_to"
      || norm == "associated_with" || norm == "associated"
      || norm == "mentions" || norm == "mention" || norm == "about"
      || norm == "involves" || norm == "includes" || norm == "has")
    {
      return "co_occurs";
    }
  if (norm == "implies" || norm == "implication" || norm == "imply"
      || norm == "suggests" || norm == "indicates")
    {
      return "implies";
    }
  if (norm == "contradicts" || norm == "contradiction" || norm == "contradict")
    {
      return "contradicts";
    }
  if (norm == "reinforces" || norm == "reinforce" || norm == "supports"
      || norm == "confirms" || norm == "supports_relation")
    {
      return "reinforces";
    }
  if (norm == "causes" || norm == "cause" || norm == "leads_to"
      || norm == "results_in")
    {
      return "causes";
    }
  if (norm == "similar_to" || norm == "similar" || norm == "similarto"
      || norm == "is_similar_to")
    {
      return "similar_to";
    }

  // If the model emitted a relation object but the predicate is malformed
  // punctuation or otherwise outside the enum, keep the weakest relation type.
  // Endpoint admission below still requires durable, grounded labels.
  return "co_occurs";
}

std::optional<long long>
OptionalI64 (std::uint64_t value)
{
  return static_cast<long long> (value);
}

std::any
NullableAny (const std::optional<long long> &value)
{
  if (value.has_value ())
    {
      return std::any (*value);
    }
  return std::any ();
}

long long
ExtractInt64Field (const std::map<std::string, std::any> &row,
                   const char *field)
{
  auto it = row.find (field);
  if (it == row.end () || !it->second.has_value ())
    {
      return 0;
    }
  if (it->second.type () == typeid (long long))
    {
      return std::any_cast<long long> (it->second);
    }
  if (it->second.type () == typeid (int))
    {
      return std::any_cast<int> (it->second);
    }
  return 0;
}

double
RequiredSupersessionConfidence (double existing_confidence, double sensitivity,
                                double stability)
{
  const double margin = core::FactSupersessionConfidenceMargin (
      sensitivity, stability);
  return core::Clamp (
      existing_confidence
          + margin * (1.0 - core::Clamp (existing_confidence, 0.0, 1.0)),
      0.0, 1.0);
}

double
ExtractDoubleField (const std::map<std::string, std::any> &row,
                    const char *field, double fallback = 0.0)
{
  auto it = row.find (field);
  if (it == row.end () || !it->second.has_value ())
    {
      return fallback;
    }
  if (it->second.type () == typeid (double))
    {
      return std::any_cast<double> (it->second);
    }
  if (it->second.type () == typeid (float))
    {
      return static_cast<double> (std::any_cast<float> (it->second));
    }
  if (it->second.type () == typeid (int))
    {
      return static_cast<double> (std::any_cast<int> (it->second));
    }
  if (it->second.type () == typeid (long long))
    {
      return static_cast<double> (std::any_cast<long long> (it->second));
    }
  return fallback;
}

std::string
ExtractStringField (const std::map<std::string, std::any> &row,
                    const char *field)
{
  auto it = row.find (field);
  if (it == row.end () || !it->second.has_value ())
    {
      return {};
    }
  if (it->second.type () == typeid (std::string))
    {
      return std::any_cast<std::string> (it->second);
    }
  if (it->second.type () == typeid (const char *))
    {
      const char *value = std::any_cast<const char *> (it->second);
      return value ? std::string (value) : std::string ();
    }
  return {};
}

std::string
CanonicalEvidenceText (const std::string &text)
{
  std::string out;
  out.reserve (text.size ());
  bool previous_space = false;
  for (unsigned char c : text)
    {
      if (std::isalnum (c) != 0)
        {
          out.push_back (static_cast<char> (std::tolower (c)));
          previous_space = false;
        }
      else if (!out.empty () && !previous_space)
        {
          out.push_back (' ');
          previous_space = true;
        }
    }
  if (!out.empty () && out.back () == ' ')
    {
      out.pop_back ();
    }
  return out;
}

std::vector<std::string>
SplitCanonicalTokens (const std::string &tokens)
{
  std::vector<std::string> out;
  std::string current;
  for (char c : tokens)
    {
      if (std::isspace (static_cast<unsigned char> (c)) != 0)
        {
          if (!current.empty ())
            {
              out.push_back (current);
              current.clear ();
            }
          continue;
        }
      current.push_back (c);
    }
  if (!current.empty ())
    {
      out.push_back (current);
    }
  return out;
}

bool
TokensContainPhrase (const std::string &haystack_tokens,
                     const std::string &needle_tokens)
{
  if (haystack_tokens.empty () || needle_tokens.empty ())
    {
      return false;
    }
  const std::string haystack = " " + haystack_tokens + " ";
  const std::string needle = " " + needle_tokens + " ";
  return haystack.find (needle) != std::string::npos;
}

int
SharedTokenCount (const std::string &a_tokens, const std::string &b_tokens)
{
  const auto a = SplitCanonicalTokens (a_tokens);
  const auto b = SplitCanonicalTokens (b_tokens);
  int shared = 0;
  for (const auto &left : a)
    {
      if (std::find (b.begin (), b.end (), left) != b.end ())
        {
          ++shared;
        }
    }
  return shared;
}

bool
StrongEndpointAliasMatch (const std::string &endpoint_tokens,
                          const std::string &candidate_tokens,
                          int min_shared_tokens)
{
  if (endpoint_tokens.empty () || candidate_tokens.empty ())
    {
      return false;
    }
  if (endpoint_tokens == candidate_tokens)
    {
      return true;
    }
  if (TokensContainPhrase (candidate_tokens, endpoint_tokens)
      || TokensContainPhrase (endpoint_tokens, candidate_tokens))
    {
      return true;
    }

  const auto endpoint_parts = SplitCanonicalTokens (endpoint_tokens);
  const auto candidate_parts = SplitCanonicalTokens (candidate_tokens);
  const int shared = SharedTokenCount (endpoint_tokens, candidate_tokens);
  if (shared <= 0)
    {
      return false;
    }

  const int shorter = static_cast<int> (
      std::min (endpoint_parts.size (), candidate_parts.size ()));
  return shorter > 0 && shared == shorter
         && shared >= std::max (1, min_shared_tokens);
}

bool
LabelAppearsInEvidence (const std::string &label_key,
                        const std::string &evidence_text)
{
  if (evidence_text.empty ())
    {
      return true;
    }

  const std::string label = CanonicalLabelTokenKey (label_key);
  if (label.empty ())
    {
      return false;
    }

  const std::string haystack = " " + CanonicalEvidenceText (evidence_text) + " ";
  const std::string needle = " " + label + " ";
  if (haystack.find (needle) != std::string::npos)
    {
      return true;
    }

  const auto label_parts = SplitCanonicalTokens (label);
  if (label_parts.size () < 2)
    {
      return false;
    }
  for (const auto &part : label_parts)
    {
      if (haystack.find (" " + part + " ") == std::string::npos)
        {
          return false;
        }
    }
  return true;
}

long long
CountLabelKeysAppearingInEvidence (
    const std::unordered_set<std::string> &label_keys,
    const std::string &evidence_text)
{
  if (evidence_text.empty ())
    {
      return 0;
    }

  long long count = 0;
  for (const auto &label_key : label_keys)
    {
      if (LabelAppearsInEvidence (label_key, evidence_text))
        {
          ++count;
        }
    }
  return count;
}

bool
EnvBool (const char *name, bool fallback = false)
{
  const char *value = std::getenv (name);
  if (value == nullptr)
    {
      return fallback;
    }
  std::string text (value);
  std::transform (text.begin (), text.end (), text.begin (),
                  [] (unsigned char c) {
                    return static_cast<char> (std::tolower (c));
                  });
  return text == "1" || text == "true" || text == "yes" || text == "on";
}

evidence::RevisionOptions
EvidenceConfidenceOptions (double focus, double sensitivity, double stability)
{
  return evidence::OptionsForKnobs (focus, sensitivity, stability);
}

std::vector<evidence::EvidenceStamp>
BuildFactEvidenceStamps (const std::vector<long long> &evidence_memory_ids)
{
  std::vector<evidence::EvidenceStamp> stamps;
  stamps.reserve (evidence_memory_ids.size ());
  for (size_t i = 0; i < evidence_memory_ids.size (); ++i)
    {
      const long long evidence_memory_id = evidence_memory_ids[i];
      if (evidence_memory_id <= 0)
        {
          continue;
        }
      stamps.push_back (
          { evidence_memory_id, i == 0 ? "summary" : "episodic" });
    }
  return stamps;
}

std::vector<evidence::EvidenceStamp>
LoadFactEvidenceStamps (Transaction &tx, long long fact_id)
{
  std::vector<evidence::EvidenceStamp> stamps;
  if (fact_id <= 0)
    {
      return stamps;
    }
  auto rows = tx.Execute (
      "SELECT source_memory_id, evidence_type FROM fact_evidence "
      "WHERE fact_id = ?",
      { fact_id });
  stamps.reserve (rows.size ());
  for (const auto &row : rows)
    {
      const long long source_memory_id
          = ExtractInt64Field (row, "source_memory_id");
      if (source_memory_id <= 0)
        {
          continue;
        }
      stamps.push_back (
          { source_memory_id, ExtractStringField (row, "evidence_type") });
    }
  return stamps;
}

bool
IsSourceSpanStopword (const std::string &token)
{
  static const std::unordered_set<std::string_view> kStopwords = {
    "a", "an", "and", "are", "as", "assistant", "at", "be", "been",
    "being", "but", "by", "can", "could", "did", "do", "does", "doing",
    "for", "from", "get", "go", "got", "had", "has", "have", "having",
    "her", "him", "his", "i", "in", "is", "it", "its", "just", "like",
    "make", "maybe", "me", "my", "of", "oh", "on", "or", "our", "she",
    "so", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "to", "user", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "with", "would", "you",
    "your", "agree", "amazon", "backs", "big", "brand", "cute", "days",
    "delivery", "good", "how", "https", "image", "img", "including",
    "line", "long", "not", "off", "once", "sorry", "way", "wild"
  };
  return kStopwords.find (token) != kStopwords.end ();
}

int
CanonicalTokenCount (const std::string &canonical)
{
  int count = 0;
  bool in_token = false;
  for (unsigned char c : canonical)
    {
      if (std::isalnum (c) != 0)
        {
          if (!in_token)
            {
              ++count;
              in_token = true;
            }
        }
      else
        {
          in_token = false;
        }
    }
  return count;
}

struct SourceSpanToken
{
  std::string surface;
  std::string canonical;
  bool capitalized = false;
  bool boundary_before = false;
};

bool
IsStrongSourceSpanLabel (const std::string &label,
                         const std::string &label_key)
{
  if (!IsDurableLabelCandidate (label, label_key))
    {
      return false;
    }
  const std::string canonical = CanonicalLabelTokenKey (label_key);
  if (CanonicalTokenCount (canonical) >= 2)
    {
      return true;
    }
  for (unsigned char c : label)
    {
      if (std::isalpha (c) != 0)
        {
          return std::isupper (c) != 0;
        }
    }
  return false;
}

bool
IsContextualSourceSpanLabel (const std::vector<SourceSpanToken> &tokens,
                             size_t start, int width,
                             const core::STMLTMSourceSpanPolicy &policy)
{
  if (width < policy.contextual_min_width)
    {
      return false;
    }
  int content_count = 0;
  bool has_capitalized = false;
  bool has_eventish_token = false;
  static const std::unordered_set<std::string_view> kEventishTokens = {
    "appointment", "birthday",  "bridge",    "class",     "conversation",
    "crash",       "dinner",    "errand",    "flight",    "game",
    "hospital",    "meeting",   "memory",    "message",   "party",
    "park",        "place",     "plaza",     "project",   "restaurant",
    "school",      "situation", "story",     "trip",      "visit"
  };
  for (int offset = 0; offset < width; ++offset)
    {
      const auto &token = tokens[start + static_cast<size_t> (offset)];
      if (offset > 0 && token.boundary_before)
        {
          return false;
        }
      if (IsSourceSpanStopword (token.canonical))
        {
          continue;
        }
      if (token.canonical.size () < 3)
        {
          return false;
        }
      ++content_count;
      has_capitalized = has_capitalized || token.capitalized;
      has_eventish_token
          = has_eventish_token
            || kEventishTokens.find (token.canonical) != kEventishTokens.end ();
    }
  return content_count >= policy.contextual_min_content_tokens
         && (has_capitalized || has_eventish_token);
}

bool
IsSourceSpanActionToken (std::string_view token)
{
  static const std::unordered_set<std::string_view> kActionTokens = {
    "ate",    "bought", "brought", "called", "crashed", "found",
    "gave",   "give",   "gives",   "love",   "loved",   "loves",
    "met",    "named",  "paid",    "picked", "saw",     "sent",
    "trained", "visited"
  };
  return kActionTokens.find (token) != kActionTokens.end ();
}

std::string
BuildTrimmedSourceSpanPhrase (const std::vector<SourceSpanToken> &tokens,
                              size_t start, int width)
{
  size_t begin = start;
  size_t end = start + static_cast<size_t> (width);
  while (begin < end && IsSourceSpanStopword (tokens[begin].canonical))
    {
      ++begin;
    }
  while (end > begin && IsSourceSpanStopword (tokens[end - 1].canonical))
    {
      --end;
    }
  std::string phrase;
  for (size_t i = begin; i < end; ++i)
    {
      if (!phrase.empty ())
        {
          phrase += " ";
        }
      phrase += tokens[i].surface;
    }
  return phrase;
}

std::string
BuildEventSourceSpanPhrase (const std::vector<SourceSpanToken> &tokens,
                            std::optional<size_t> subject_index,
                            size_t verb_index, size_t object_start,
                            size_t object_end)
{
  std::string phrase;
  auto append = [&] (const std::string &surface) {
    if (!phrase.empty ())
      {
        phrase += " ";
      }
    phrase += surface;
  };

  if (subject_index.has_value ())
    {
      append (tokens[*subject_index].surface);
    }
  append (tokens[verb_index].surface);
  for (size_t i = object_start; i < object_end; ++i)
    {
      append (tokens[i].surface);
    }
  return phrase;
}

bool
OnlySourceSpanStopwordsBetween (const std::vector<SourceSpanToken> &tokens,
                                size_t start, size_t end)
{
  for (size_t i = start; i < end && i < tokens.size (); ++i)
    {
      if (!IsSourceSpanStopword (tokens[i].canonical))
        {
          return false;
        }
    }
  return true;
}

std::vector<SourceSpanToken>
TokenizeSourceSpans (const std::string &raw_text)
{
  const std::string text = NormalizeApostrophes (raw_text);
  std::vector<SourceSpanToken> tokens;
  std::string current;
  bool next_boundary_before = true;
  auto flush = [&] {
    if (current.empty ())
      {
        return;
      }
    SourceSpanToken token;
    token.surface = current;
    token.canonical = CanonicalLabelTokenKey (NormalizeLabelKey (current));
    token.capitalized = std::isupper (
        static_cast<unsigned char> (current.front ())) != 0;
    token.boundary_before = next_boundary_before;
    if (!token.canonical.empty ())
      {
        tokens.push_back (std::move (token));
      }
    current.clear ();
    next_boundary_before = false;
  };

  for (unsigned char c : text)
    {
      if (std::isalnum (c) != 0 || c == '\'' || c == '-')
        {
          current.push_back (static_cast<char> (c));
        }
      else
        {
          flush ();
          if (c == '.' || c == '!' || c == '?' || c == '\n' || c == '\r')
            {
              next_boundary_before = true;
            }
        }
    }
  flush ();
  return tokens;
}

/// Genericity signal derived from the retrieval space instead of token
/// lists: a candidate label is conversational filler when its similarity
/// profile against the existing label bank is HIGH and FLAT (it sits in the
/// dense conversational region, near everything, peaked at nothing).
/// Measured on AIST-256 against a six-label bank (see
/// stm_labeling_integration.test.cpp, which keeps these measurements
/// honest): filler phrases sit at mean 0.688-0.701 with peak-mean
/// 0.064-0.079; on-topic content peaks 0.24-0.31 above its mean; novel
/// topics sit at mean 0.47-0.61. The mean floor (0.65) sits in the
/// 0.611-0.688 gap separating novel topics from filler, and the peak
/// threshold (0.15) sits in the 0.111-0.244 gap separating everything
/// non-content from content.
struct LabelBankContrast
{
  bool available = false;
  double mean_similarity = 0.0;
  double max_similarity = 0.0;
};

constexpr int kContrastMinBankSize = 4;
constexpr double kContrastGenericMeanFloor = 0.65;
constexpr double kContrastMinPeak = 0.15;

/// bank_size_limit caps the scan to summary-cache entries that existed
/// before this consolidation pass began: labels admitted moments ago for
/// the same summary are siblings of the candidate, and measuring contrast
/// against them makes every cohesive scene's later labels look generic
/// (high-flat against their own scene-mates).
LabelBankContrast
ComputeLabelBankContrast (const std::vector<float> &embedding,
                          const ProcessorContext &p_ctx,
                          size_t bank_size_limit)
{
  LabelBankContrast out;
  if (embedding.empty ())
    {
      return out;
    }
  const Eigen::Map<const Eigen::VectorXf> v (
      embedding.data (), static_cast<Eigen::Index> (embedding.size ()));
  const float v_norm = v.norm ();
  if (v_norm <= 1e-9f)
    {
      return out;
    }
  double sum = 0.0;
  double max_sim = -1.0;
  int count = 0;
  const size_t scan_count
      = std::min (bank_size_limit, p_ctx.summary_cache.size ());
  for (size_t entry_idx = 0; entry_idx < scan_count; ++entry_idx)
    {
      const auto &entry = p_ctx.summary_cache[entry_idx];
      if (!entry.is_label || entry.embedding_norm <= 1e-9f
          || entry.embedding.size () != v.size ())
        {
          continue;
        }
      const double sim = static_cast<double> (entry.embedding.dot (v))
                         / (static_cast<double> (entry.embedding_norm)
                            * static_cast<double> (v_norm));
      sum += sim;
      max_sim = std::max (max_sim, sim);
      ++count;
    }
  if (count < kContrastMinBankSize)
    {
      return out;
    }
  out.available = true;
  out.mean_similarity = sum / static_cast<double> (count);
  out.max_similarity = max_sim;
  return out;
}

bool
IsGenericByLabelBankContrast (const LabelBankContrast &contrast)
{
  return contrast.available
         && contrast.mean_similarity >= kContrastGenericMeanFloor
         && (contrast.max_similarity - contrast.mean_similarity)
                < kContrastMinPeak;
}

size_t
CountContrastBankLabels (const ProcessorContext &p_ctx,
                         size_t bank_size_limit)
{
  size_t count = 0;
  const size_t scan_count
      = std::min (bank_size_limit, p_ctx.summary_cache.size ());
  for (size_t entry_idx = 0; entry_idx < scan_count; ++entry_idx)
    {
      const auto &entry = p_ctx.summary_cache[entry_idx];
      if (entry.is_label && entry.embedding_norm > 1e-9f)
        {
          ++count;
        }
    }
  return count;
}


std::vector<std::string>
BuildSourceSpanCandidates (const std::string &evidence_text,
                           int max_candidates,
                           const core::STMLTMSourceSpanPolicy &policy)
{
  if (max_candidates <= 0 || evidence_text.empty ())
    {
      return {};
    }

  const auto tokens = TokenizeSourceSpans (evidence_text);
  std::vector<std::string> candidates;
  std::unordered_set<std::string> seen;
  std::unordered_set<std::string> window_keys;
  auto add_candidate_impl = [&] (const std::string &candidate,
                                 bool from_window) -> bool {
    if (static_cast<int> (candidates.size ()) >= max_candidates)
      {
        return false;
      }
    const std::string label = TrimLabel (candidate);
    const std::string key = NormalizeLabelKey (label);
    if (key.empty () || seen.find (key) != seen.end ())
      {
        return false;
      }
    if (!IsStrongSourceSpanLabel (label, key)
        || !LabelAppearsInEvidence (key, evidence_text))
      {
        return false;
      }
    seen.insert (key);
    if (from_window)
      {
        window_keys.insert (key);
      }
    candidates.push_back (label);
    return true;
  };
  auto add_candidate = [&] (const std::string &candidate) -> bool {
    return add_candidate_impl (candidate, false);
  };
  auto add_window_candidate = [&] (const std::string &candidate) -> bool {
    return add_candidate_impl (candidate, true);
  };

  // The sliding-window generators below emit every (start, width)
  // combination, which turns one event into a chain of nested variants
  // ("Maya gave", "Maya gave a pitch", ...). Windows iterate widest-first;
  // once a window is admitted its token range is covered and any window
  // nested inside an already-covered range is redundant. The event,
  // proper-noun, and action generators are exempt: their shorter anchors
  // ("taco bell", "gave money") are intentional multi-granularity labels.
  std::vector<bool> covered;
  auto range_covered = [&] (size_t start, int width) {
    if (covered.size () < static_cast<size_t> (start) + width)
      {
        return false;
      }
    for (int j = 0; j < width; ++j)
      {
        if (!covered[start + static_cast<size_t> (j)])
          {
            return false;
          }
      }
    return true;
  };
  auto mark_covered = [&] (size_t start, int width) {
    if (covered.size () < static_cast<size_t> (start) + width)
      {
        covered.resize (start + static_cast<size_t> (width), false);
      }
    for (int j = 0; j < width; ++j)
      {
        covered[start + static_cast<size_t> (j)] = true;
      }
  };

  for (size_t verb_index = 0; verb_index < tokens.size (); ++verb_index)
    {
      const auto &verb = tokens[verb_index];
      if (!IsSourceSpanActionToken (verb.canonical))
        {
          continue;
        }
      if (verb_index + 1 >= tokens.size ())
        {
          continue;
        }
      size_t object_start = verb_index + 1;
      while (object_start < tokens.size ()
             && !tokens[object_start].boundary_before
             && IsSourceSpanStopword (tokens[object_start].canonical))
        {
          ++object_start;
        }
      if (object_start >= tokens.size ()
          || tokens[object_start].boundary_before
          || tokens[object_start].canonical.size () < 4
          || IsSourceSpanStopword (tokens[object_start].canonical))
        {
          continue;
        }

      size_t object_end = object_start;
      int object_tokens = 0;
      while (object_end < tokens.size ()
             && object_tokens < policy.action_object_max_tokens)
        {
          const auto &object_token = tokens[object_end];
          if (object_end > object_start && object_token.boundary_before)
            {
              break;
            }
          if (object_token.canonical.size () < 4
              || IsSourceSpanStopword (object_token.canonical))
            {
              break;
            }
          ++object_end;
          ++object_tokens;
        }
      if (object_tokens == 0)
        {
          continue;
        }

      if (tokens[object_start].capitalized)
        {
          continue;
        }

      std::optional<size_t> subject_index;
      if (verb_index > 0)
        {
          size_t subject_search = verb_index;
          while (subject_search > 0)
            {
              --subject_search;
              const auto &subject = tokens[subject_search];
              if (subject.capitalized
                  && !IsSourceSpanStopword (subject.canonical)
                  && subject.canonical.size () >= 3
                  && OnlySourceSpanStopwordsBetween (
                      tokens, subject_search + 1, verb_index))
                {
                  subject_index = subject_search;
                  break;
                }
              if (subject.boundary_before && subject_search + 1 < verb_index)
                {
                  break;
                }
              if (verb_index - subject_search
                  >= static_cast<size_t> (policy.subject_search_max_gap))
                {
                  break;
                }
            }
        }
      if (subject_index.has_value ())
        {
          add_candidate (BuildEventSourceSpanPhrase (
              tokens, subject_index, verb_index, object_start, object_end));
        }
      add_candidate (BuildEventSourceSpanPhrase (
          tokens, std::nullopt, verb_index, object_start, object_end));
    }

  for (size_t i = 0; i < tokens.size (); ++i)
    {
      if (!tokens[i].capitalized
          || IsSourceSpanStopword (tokens[i].canonical))
        {
          continue;
        }
      std::string phrase = tokens[i].surface;
      size_t j = i + 1;
      int parts = 1;
	      while (j < tokens.size () && parts < policy.proper_noun_max_parts
	             && tokens[j].capitalized
	             && !tokens[j].boundary_before
	             && !IsSourceSpanStopword (tokens[j].canonical))
	        {
	          phrase += " " + tokens[j].surface;
          ++j;
          ++parts;
	        }
	      add_candidate (phrase);
	      if (parts > 1)
	        {
	          i = j - 1;
	        }
	    }

	  for (int width = policy.contextual_max_width;
	       width >= policy.contextual_min_width; --width)
    {
      for (size_t i = 0; i + static_cast<size_t> (width) <= tokens.size ();
           ++i)
        {
          if (range_covered (i, width))
            {
              continue;
            }
	          if (!IsContextualSourceSpanLabel (tokens, i, width, policy))
            {
              continue;
            }
          if (add_window_candidate (BuildTrimmedSourceSpanPhrase (tokens, i, width)))
            {
              mark_covered (i, width);
            }
        }
    }

	  for (int width = policy.phrase_max_width;
	       width >= policy.phrase_min_width; --width)
    {
      for (size_t i = 0; i + static_cast<size_t> (width) <= tokens.size ();
           ++i)
        {
          if (range_covered (i, width))
            {
              continue;
            }
          std::string phrase;
          bool usable = true;
          for (int j = 0; j < width; ++j)
            {
	              const auto &token = tokens[i + static_cast<size_t> (j)];
	              if (j > 0 && token.boundary_before)
	                {
	                  usable = false;
	                  break;
	                }
	              if (token.canonical.size () < 4
                  || IsSourceSpanStopword (token.canonical))
                {
                  usable = false;
                  break;
                }
              if (!phrase.empty ())
                {
                  phrase += " ";
                }
              phrase += token.surface;
            }
          if (usable && add_window_candidate (phrase))
            {
              mark_covered (i, width);
            }
        }
    }

  for (const auto &token : tokens)
    {
	      if (static_cast<int> (token.canonical.size ())
	              >= policy.singleton_min_chars
          && !IsSourceSpanStopword (token.canonical))
        {
          add_candidate (token.surface);
        }
    }

  // Sliding windows emit nested variants of one span ("Maya gave a pitch",
  // "Maya gave a pitch this weekend"); a window label whose tokens are a
  // subset of any other candidate's adds no information. Event, action,
  // and proper-noun anchors are exempt: their shorter forms are deliberate
  // multi-granularity labels.
  auto token_set = [] (const std::string &label) {
    std::unordered_set<std::string> tokens_out;
    const std::string canonical
        = CanonicalLabelTokenKey (NormalizeLabelKey (label));
    std::string token;
    for (unsigned char c : canonical)
      {
        if (std::isalnum (c) != 0)
          {
            token.push_back (static_cast<char> (c));
          }
        else if (!token.empty ())
          {
            tokens_out.insert (token);
            token.clear ();
          }
      }
    if (!token.empty ())
      {
        tokens_out.insert (token);
      }
    return tokens_out;
  };
  std::vector<std::string> kept;
  kept.reserve (candidates.size ());
  for (size_t i = 0; i < candidates.size (); ++i)
    {
      const std::string key_i = NormalizeLabelKey (candidates[i]);
      if (window_keys.find (key_i) == window_keys.end ())
        {
          kept.push_back (candidates[i]);
          continue;
        }
      const auto tokens_i = token_set (candidates[i]);
      bool subsumed = false;
      for (size_t j = 0; j < candidates.size () && !subsumed; ++j)
        {
          if (i == j)
            {
              continue;
            }
          const auto tokens_j = token_set (candidates[j]);
          if (tokens_i.size () >= tokens_j.size ())
            {
              continue;
            }
          subsumed = std::all_of (
              tokens_i.begin (), tokens_i.end (),
              [&] (const std::string &t) { return tokens_j.count (t) > 0; });
        }
      if (!subsumed)
        {
          kept.push_back (candidates[i]);
        }
    }
  return kept;
}

std::optional<ExtractedFact>
BuildFactFromDurableEventLabel (const std::string &label, double focus,
                                double sensitivity, double stability)
{
  const auto tokens = TokenizeSourceSpans (label);
  if (tokens.size () < 3 || !tokens.front ().capitalized)
    {
      return std::nullopt;
    }

  static const std::unordered_set<std::string_view> kActionVerbs = {
    "ate",     "bought", "brought", "called",  "found", "gave",
    "give",    "gives",  "love",    "loved",   "loves", "met",
    "named",   "paid",   "picked",  "saw",     "sent",  "trained",
    "visited"
  };
  static const std::unordered_set<std::string_view> kObjectSkipTokens = {
    "a", "an", "and", "me", "my", "the"
  };
  static const std::unordered_set<std::string_view> kObjectBoundaryTokens = {
    "at", "for", "in", "of", "on", "to", "with"
  };

  size_t verb_index = tokens.size ();
  for (size_t i = 1; i < tokens.size (); ++i)
    {
      if (kActionVerbs.find (tokens[i].canonical) != kActionVerbs.end ())
        {
          verb_index = i;
          break;
        }
    }
  if (verb_index == tokens.size () || verb_index + 1 >= tokens.size ())
    {
      return std::nullopt;
    }

  std::string object;
  int object_tokens = 0;
  for (size_t i = verb_index + 1; i < tokens.size () && object_tokens < 4; ++i)
    {
      if (kObjectBoundaryTokens.find (tokens[i].canonical)
          != kObjectBoundaryTokens.end ())
        {
          if (object_tokens > 0)
            {
              break;
            }
          continue;
        }
      if (kObjectSkipTokens.find (tokens[i].canonical)
          != kObjectSkipTokens.end ())
        {
          continue;
        }
      if (tokens[i].canonical.size () < 3)
        {
          continue;
        }
      if (!object.empty ())
        {
          object += " ";
        }
      object += tokens[i].surface;
      ++object_tokens;
    }
  if (object.empty ())
    {
      return std::nullopt;
    }

  ExtractedFact fact;
  fact.subject = tokens.front ().surface;
  fact.predicate = tokens[verb_index].canonical;
  fact.object = object;
  fact.confidence = core::FactDerivedEventConfidence (focus, sensitivity,
                                                      stability);
  return fact;
}

bool
StmLtmAuditEnabled ()
{
  return EnvBool ("CORTEXT_STM_LTM_AUDIT", false);
}

std::string
AuditJoinStrings (const std::vector<std::string> &values)
{
  std::string out;
  for (const auto &value : values)
    {
      if (!out.empty ())
        {
          out += "\n";
        }
      out += value;
    }
  return out;
}

void
EnsureStmLtmAuditTable (Transaction &tx)
{
  AddWrite (
      tx,
      "CREATE TABLE IF NOT EXISTS stm_ltm_relabel_audit ("
      "  summary_id TEXT PRIMARY KEY,"
      "  created_at INTEGER,"
      "  cluster_size INTEGER,"
      "  source_memory_count INTEGER,"
      "  source_text_count INTEGER,"
      "  source_blob_count INTEGER,"
      "  source_memory_ids TEXT,"
      "  stm_graph_count INTEGER,"
      "  stm_item_count INTEGER,"
      "  stm_label_edge_count INTEGER,"
      "  current_label_count INTEGER,"
      "  current_labels TEXT,"
      "  refined_label_count INTEGER DEFAULT 0,"
      "  refined_labels TEXT DEFAULT '',"
      "  kept_label_count INTEGER DEFAULT 0,"
      "  added_label_count INTEGER DEFAULT 0,"
      "  removed_label_count INTEGER DEFAULT 0,"
      "  removed_labels TEXT DEFAULT '',"
      "  current_labels_in_selected_evidence INTEGER DEFAULT 0,"
      "  current_labels_in_full_source INTEGER DEFAULT 0,"
      "  removed_labels_in_selected_evidence INTEGER DEFAULT 0,"
      "  removed_labels_in_full_source INTEGER DEFAULT 0,"
      "  refined_labels_in_selected_evidence INTEGER DEFAULT 0,"
      "  refined_labels_in_full_source INTEGER DEFAULT 0,"
      "  extraction_label_candidate_count INTEGER DEFAULT 0,"
      "  extraction_relation_candidate_count INTEGER DEFAULT 0,"
      "  source_span_candidate_count INTEGER DEFAULT 0,"
      "  label_candidates_rejected_non_durable INTEGER DEFAULT 0,"
      "  label_candidates_rejected_ungrounded INTEGER DEFAULT 0,"
      "  label_candidates_rejected_low_contrast INTEGER DEFAULT 0,"
      "  label_candidates_rejected_duplicate INTEGER DEFAULT 0,"
      "  label_candidates_rejected_legacy_gate INTEGER DEFAULT 0,"
      "  labels_inserted_from_extractor INTEGER DEFAULT 0,"
      "  labels_inserted_from_current_floor INTEGER DEFAULT 0,"
      "  labels_inserted_from_source_span_floor INTEGER DEFAULT 0,"
      "  labels_inserted_from_relation_endpoint INTEGER DEFAULT 0,"
      "  has_label_edges_after INTEGER DEFAULT 0,"
      "  derived_from_edges INTEGER DEFAULT 0,"
      "  durable_ltm_nodes_with_source INTEGER DEFAULT 0,"
      "  durable_ltm_nodes_missing_source INTEGER DEFAULT 0,"
      "  durable_ltm_source_link_pairs INTEGER DEFAULT 0,"
      "  relation_count INTEGER DEFAULT 0,"
      "  relation_edges_created INTEGER DEFAULT 0,"
      "  label_cooccurrence_edges_created INTEGER DEFAULT 0,"
      "  relation_edges_skipped_non_durable_endpoint INTEGER DEFAULT 0,"
      "  relation_edges_skipped_missing_endpoint INTEGER DEFAULT 0,"
      "  relation_edges_skipped_unsupported_predicate INTEGER DEFAULT 0,"
      "  relation_endpoint_direct_hits INTEGER DEFAULT 0,"
      "  relation_endpoint_repair_hits INTEGER DEFAULT 0,"
      "  relation_endpoint_created_labels INTEGER DEFAULT 0,"
      "  relation_endpoint_relation_backed_labels INTEGER DEFAULT 0,"
      "  relation_endpoint_rejected_count INTEGER DEFAULT 0,"
      "  relation_endpoint_rejected_non_durable INTEGER DEFAULT 0,"
      "  relation_endpoint_rejected_ungrounded INTEGER DEFAULT 0,"
      "  fact_assertions_touched INTEGER DEFAULT 0,"
      "  source_memories_with_content INTEGER DEFAULT 0"
      ")");

  auto has_column = [&tx] (const std::string &column) {
    auto rows = tx.Execute ("PRAGMA table_info(stm_ltm_relabel_audit)", {});
    for (const auto &row : rows)
      {
        auto it = row.find ("name");
        if (it != row.end () && it->second.type () == typeid (std::string)
            && std::any_cast<std::string> (it->second) == column)
          {
            return true;
          }
      }
    return false;
  };
  auto add_column_if_missing = [&tx, &has_column] (
                                   const std::string &column,
                                   const std::string &definition) {
    if (!has_column (column))
      {
        AddWrite (tx,
                  "ALTER TABLE stm_ltm_relabel_audit ADD COLUMN "
                      + definition,
                  {});
      }
  };
  add_column_if_missing (
      "refined_label_count", "refined_label_count INTEGER DEFAULT 0");
  add_column_if_missing ("refined_labels", "refined_labels TEXT DEFAULT ''");
  add_column_if_missing ("kept_label_count",
                         "kept_label_count INTEGER DEFAULT 0");
  add_column_if_missing ("added_label_count",
                         "added_label_count INTEGER DEFAULT 0");
  add_column_if_missing ("removed_label_count",
                         "removed_label_count INTEGER DEFAULT 0");
  add_column_if_missing ("removed_labels", "removed_labels TEXT DEFAULT ''");
  add_column_if_missing ("has_label_edges_after",
                         "has_label_edges_after INTEGER DEFAULT 0");
  add_column_if_missing ("derived_from_edges",
                         "derived_from_edges INTEGER DEFAULT 0");
  add_column_if_missing (
      "durable_ltm_nodes_with_source",
      "durable_ltm_nodes_with_source INTEGER DEFAULT 0");
  add_column_if_missing (
      "durable_ltm_nodes_missing_source",
      "durable_ltm_nodes_missing_source INTEGER DEFAULT 0");
  add_column_if_missing (
      "durable_ltm_source_link_pairs",
      "durable_ltm_source_link_pairs INTEGER DEFAULT 0");
  add_column_if_missing ("relation_count",
                         "relation_count INTEGER DEFAULT 0");
  add_column_if_missing ("relation_edges_created",
                         "relation_edges_created INTEGER DEFAULT 0");
  add_column_if_missing (
      "label_cooccurrence_edges_created",
      "label_cooccurrence_edges_created INTEGER DEFAULT 0");
  add_column_if_missing (
      "relation_edges_skipped_non_durable_endpoint",
      "relation_edges_skipped_non_durable_endpoint INTEGER DEFAULT 0");
  add_column_if_missing (
      "relation_edges_skipped_missing_endpoint",
      "relation_edges_skipped_missing_endpoint INTEGER DEFAULT 0");
  add_column_if_missing (
      "relation_edges_skipped_unsupported_predicate",
      "relation_edges_skipped_unsupported_predicate INTEGER DEFAULT 0");
  add_column_if_missing ("relation_endpoint_direct_hits",
                         "relation_endpoint_direct_hits INTEGER DEFAULT 0");
  add_column_if_missing ("relation_endpoint_repair_hits",
                         "relation_endpoint_repair_hits INTEGER DEFAULT 0");
  add_column_if_missing (
      "relation_endpoint_created_labels",
      "relation_endpoint_created_labels INTEGER DEFAULT 0");
  add_column_if_missing (
      "relation_endpoint_relation_backed_labels",
      "relation_endpoint_relation_backed_labels INTEGER DEFAULT 0");
  add_column_if_missing (
      "relation_endpoint_rejected_count",
      "relation_endpoint_rejected_count INTEGER DEFAULT 0");
  add_column_if_missing (
      "relation_endpoint_rejected_non_durable",
      "relation_endpoint_rejected_non_durable INTEGER DEFAULT 0");
  add_column_if_missing (
      "relation_endpoint_rejected_ungrounded",
      "relation_endpoint_rejected_ungrounded INTEGER DEFAULT 0");
  add_column_if_missing (
      "current_labels_in_selected_evidence",
      "current_labels_in_selected_evidence INTEGER DEFAULT 0");
  add_column_if_missing (
      "current_labels_in_full_source",
      "current_labels_in_full_source INTEGER DEFAULT 0");
  add_column_if_missing (
      "removed_labels_in_selected_evidence",
      "removed_labels_in_selected_evidence INTEGER DEFAULT 0");
  add_column_if_missing (
      "removed_labels_in_full_source",
      "removed_labels_in_full_source INTEGER DEFAULT 0");
  add_column_if_missing (
      "refined_labels_in_selected_evidence",
      "refined_labels_in_selected_evidence INTEGER DEFAULT 0");
  add_column_if_missing (
      "refined_labels_in_full_source",
      "refined_labels_in_full_source INTEGER DEFAULT 0");
  add_column_if_missing (
      "extraction_label_candidate_count",
      "extraction_label_candidate_count INTEGER DEFAULT 0");
  add_column_if_missing (
      "extraction_relation_candidate_count",
      "extraction_relation_candidate_count INTEGER DEFAULT 0");
  add_column_if_missing (
      "source_span_candidate_count",
      "source_span_candidate_count INTEGER DEFAULT 0");
  add_column_if_missing (
      "label_candidates_rejected_non_durable",
      "label_candidates_rejected_non_durable INTEGER DEFAULT 0");
  add_column_if_missing (
      "label_candidates_rejected_ungrounded",
      "label_candidates_rejected_ungrounded INTEGER DEFAULT 0");
  add_column_if_missing (
      "label_candidates_rejected_low_contrast",
      "label_candidates_rejected_low_contrast INTEGER DEFAULT 0");
  add_column_if_missing (
      "label_candidates_rejected_duplicate",
      "label_candidates_rejected_duplicate INTEGER DEFAULT 0");
  add_column_if_missing (
      "label_candidates_rejected_legacy_gate",
      "label_candidates_rejected_legacy_gate INTEGER DEFAULT 0");
  add_column_if_missing (
      "labels_inserted_from_extractor",
      "labels_inserted_from_extractor INTEGER DEFAULT 0");
  add_column_if_missing (
      "labels_inserted_from_current_floor",
      "labels_inserted_from_current_floor INTEGER DEFAULT 0");
  add_column_if_missing (
      "labels_inserted_from_source_span_floor",
      "labels_inserted_from_source_span_floor INTEGER DEFAULT 0");
  add_column_if_missing (
      "labels_inserted_from_relation_endpoint",
      "labels_inserted_from_relation_endpoint INTEGER DEFAULT 0");
  add_column_if_missing ("fact_assertions_touched",
                         "fact_assertions_touched INTEGER DEFAULT 0");
  add_column_if_missing (
      "source_memories_with_content",
      "source_memories_with_content INTEGER DEFAULT 0");
}

std::unordered_set<std::string>
LoadAttachedLabelKeys (Transaction &tx, long long summary_memory_id)
{
  std::unordered_set<std::string> keys;
  if (summary_memory_id <= 0)
    {
      return keys;
    }
  auto rows = tx.Execute (
      "SELECT l.source_id FROM associations a "
      "JOIN memories l ON l.memory_id = a.target_memory_id "
      "WHERE a.source_memory_id = ? AND a.edge_type = 'has_label' "
      "  AND l.kind = 'LABEL'",
      { summary_memory_id });
  keys.reserve (rows.size ());
  for (const auto &row : rows)
    {
      auto it = row.find ("source_id");
      if (it != row.end () && it->second.type () == typeid (std::string))
        {
          const std::string key = NormalizeLabelKey (
              std::any_cast<std::string> (it->second));
          if (!key.empty ())
            {
              keys.insert (key);
            }
        }
    }
  return keys;
}

long long
CountRowsForMemory (Transaction &tx, const std::string &sql,
                    long long memory_id)
{
  auto rows = tx.Execute (sql, { memory_id });
  if (rows.empty () || rows[0].count ("c") == 0)
    {
      return 0;
    }
  return ExtractInt64Field (rows[0], "c");
}

void
AuditProcessedRelabelResult (
    Transaction &tx, const std::string &summary_id, long long summary_memory_id,
    const std::vector<std::string> &refined_labels,
    const std::vector<std::string> &removed_labels, long long kept_count,
    long long added_count, long long relation_count,
    long long relation_edges_created,
    long long label_cooccurrence_edges_created,
    long long relation_edges_skipped_non_durable_endpoint,
    long long relation_edges_skipped_missing_endpoint,
    long long relation_edges_skipped_unsupported_predicate,
    long long relation_endpoint_direct_hits,
    long long relation_endpoint_repair_hits,
    long long relation_endpoint_created_labels,
    long long relation_endpoint_relation_backed_labels,
    long long relation_endpoint_rejected_count,
    long long relation_endpoint_rejected_non_durable,
    long long relation_endpoint_rejected_ungrounded,
    long long current_labels_in_selected_evidence,
    long long current_labels_in_full_source,
    long long removed_labels_in_selected_evidence,
    long long removed_labels_in_full_source,
    long long refined_labels_in_selected_evidence,
    long long refined_labels_in_full_source,
    long long extraction_label_candidate_count,
    long long extraction_relation_candidate_count,
    long long source_span_candidate_count,
    long long label_candidates_rejected_non_durable,
    long long label_candidates_rejected_ungrounded,
    long long label_candidates_rejected_low_contrast,
    long long label_candidates_rejected_duplicate,
    long long label_candidates_rejected_legacy_gate,
    long long labels_inserted_from_extractor,
    long long labels_inserted_from_current_floor,
    long long labels_inserted_from_source_span_floor,
    long long labels_inserted_from_relation_endpoint,
    long long facts_touched)
{
  if (!StmLtmAuditEnabled () || summary_id.empty () || summary_memory_id <= 0)
    {
      return;
    }

  EnsureStmLtmAuditTable (tx);
  const long long has_label_edges_after = CountRowsForMemory (
      tx,
      "SELECT COUNT(*) AS c FROM associations "
      "WHERE source_memory_id = ? AND edge_type = 'has_label'",
      summary_memory_id);
  const long long derived_from_edges = CountRowsForMemory (
      tx,
      "SELECT COUNT(*) AS c FROM associations "
      "WHERE source_memory_id = ? AND edge_type = 'derived_from'",
      summary_memory_id);
  const long long source_memories_with_content = CountRowsForMemory (
      tx,
      "SELECT COUNT(DISTINCT m.memory_id) AS c "
      "FROM associations a "
      "JOIN memories m ON m.memory_id = a.target_memory_id "
      "LEFT JOIN signals s ON s.memory_id = m.memory_id "
      "WHERE a.source_memory_id = ? AND a.edge_type = 'derived_from' "
      "  AND (m.blob_id IS NOT NULL OR s.blob_id IS NOT NULL)",
      summary_memory_id);
  const long long durable_ltm_nodes_with_source = CountRowsForMemory (
      tx,
      "SELECT COUNT(DISTINCT label.target_memory_id) AS c "
      "FROM associations label "
      "WHERE label.source_memory_id = ? "
      "  AND label.edge_type = 'has_label' "
      "  AND EXISTS ("
      "    SELECT 1 FROM associations src "
      "    WHERE src.source_memory_id = label.source_memory_id "
      "      AND src.edge_type = 'derived_from' "
      "      AND src.target_memory_id > 0"
      "  )",
      summary_memory_id);
  const long long durable_ltm_nodes_missing_source
      = std::max<long long> (0,
                             has_label_edges_after
                                 - durable_ltm_nodes_with_source);
  const long long durable_ltm_source_link_pairs = CountRowsForMemory (
      tx,
      "SELECT COUNT(*) AS c "
      "FROM associations label "
      "JOIN associations src "
      "  ON src.source_memory_id = label.source_memory_id "
      " AND src.edge_type = 'derived_from' "
      "WHERE label.source_memory_id = ? "
      "  AND label.edge_type = 'has_label' "
      "  AND src.target_memory_id > 0",
      summary_memory_id);

  AddWrite (
      tx,
      "UPDATE stm_ltm_relabel_audit SET "
      "refined_label_count = ?, refined_labels = ?, "
      "kept_label_count = ?, added_label_count = ?, "
      "removed_label_count = ?, removed_labels = ?, "
      "current_labels_in_selected_evidence = ?, "
      "current_labels_in_full_source = ?, "
      "removed_labels_in_selected_evidence = ?, "
      "removed_labels_in_full_source = ?, "
      "refined_labels_in_selected_evidence = ?, "
      "refined_labels_in_full_source = ?, "
      "extraction_label_candidate_count = ?, "
      "extraction_relation_candidate_count = ?, "
      "source_span_candidate_count = ?, "
      "label_candidates_rejected_non_durable = ?, "
      "label_candidates_rejected_ungrounded = ?, "
      "label_candidates_rejected_low_contrast = ?, "
      "label_candidates_rejected_duplicate = ?, "
      "label_candidates_rejected_legacy_gate = ?, "
      "labels_inserted_from_extractor = ?, "
      "labels_inserted_from_current_floor = ?, "
      "labels_inserted_from_source_span_floor = ?, "
      "labels_inserted_from_relation_endpoint = ?, "
      "has_label_edges_after = ?, derived_from_edges = ?, "
      "durable_ltm_nodes_with_source = ?, "
      "durable_ltm_nodes_missing_source = ?, "
      "durable_ltm_source_link_pairs = ?, "
      "relation_count = ?, relation_edges_created = ?, "
      "label_cooccurrence_edges_created = ?, "
      "relation_edges_skipped_non_durable_endpoint = ?, "
      "relation_edges_skipped_missing_endpoint = ?, "
      "relation_edges_skipped_unsupported_predicate = ?, "
      "relation_endpoint_direct_hits = ?, "
      "relation_endpoint_repair_hits = ?, "
      "relation_endpoint_created_labels = ?, "
      "relation_endpoint_relation_backed_labels = ?, "
      "relation_endpoint_rejected_count = ?, "
      "relation_endpoint_rejected_non_durable = ?, "
      "relation_endpoint_rejected_ungrounded = ?, "
      "fact_assertions_touched = ?, "
      "source_memories_with_content = ? "
      "WHERE summary_id = ?",
      { static_cast<long long> (refined_labels.size ()),
        AuditJoinStrings (refined_labels), kept_count, added_count,
        static_cast<long long> (removed_labels.size ()),
        AuditJoinStrings (removed_labels),
        current_labels_in_selected_evidence, current_labels_in_full_source,
        removed_labels_in_selected_evidence, removed_labels_in_full_source,
        refined_labels_in_selected_evidence, refined_labels_in_full_source,
        extraction_label_candidate_count, extraction_relation_candidate_count,
        source_span_candidate_count,
        label_candidates_rejected_non_durable,
        label_candidates_rejected_ungrounded,
        label_candidates_rejected_low_contrast,
        label_candidates_rejected_duplicate,
        label_candidates_rejected_legacy_gate,
        labels_inserted_from_extractor,
        labels_inserted_from_current_floor,
        labels_inserted_from_source_span_floor,
        labels_inserted_from_relation_endpoint,
        has_label_edges_after, derived_from_edges,
        durable_ltm_nodes_with_source, durable_ltm_nodes_missing_source,
        durable_ltm_source_link_pairs, relation_count,
        relation_edges_created, label_cooccurrence_edges_created,
        relation_edges_skipped_non_durable_endpoint,
        relation_edges_skipped_missing_endpoint,
        relation_edges_skipped_unsupported_predicate,
        relation_endpoint_direct_hits, relation_endpoint_repair_hits,
        relation_endpoint_created_labels,
        relation_endpoint_relation_backed_labels,
        relation_endpoint_rejected_count,
        relation_endpoint_rejected_non_durable,
        relation_endpoint_rejected_ungrounded, facts_touched,
        source_memories_with_content, summary_id });
}

std::string
FormatCurrentLabelsForPrompt (const std::vector<std::string> &labels)
{
  if (labels.empty ())
    {
      return "none";
    }
  std::string out;
  for (size_t i = 0; i < labels.size (); ++i)
    {
      if (i > 0)
        {
          out += ", ";
        }
      out += labels[i];
    }
  return out;
}

std::string
BuildTextLabelRefinementEvidence (
	    const std::string &combined_text,
	    const std::vector<std::string> &current_labels,
	    double F, double S, double T)
{
  const int min_words = core::STMLTMLabelPromptMinWords (F, S, T);
  const int max_words = std::max (
      min_words, core::STMLTMLabelPromptMaxWords (F, S, T));
  const int min_labels = core::STMLTMDurableMinLabels (F, S, T);
  const int max_labels = core::STMLTMDurableMaxLabels (F, S, T);
  return "Refine labels for one memory graph association. Keep correct "
         "current labels, remove unsupported or generic labels, and add "
         "missing concrete labels. Return the final replacement labels. "
         "Use durable memory anchors: named people, places, organizations, "
         "pets, specific objects, and short event phrases. Prefer "
         + std::to_string (min_words) + "-" + std::to_string (max_words)
         + " word noun/event phrases unless the label is a proper name or "
           "concrete object. When evidence contains enough anchors, return "
         + std::to_string (min_labels) + "-" + std::to_string (max_labels)
         + " labels. Every important word in a label must appear verbatim in "
           "the evidence; do not paraphrase, infer, summarize, or invent "
           "labels. Do not return pronouns, chat roles, filler words, helper "
           "verbs, modal words, status phrases, or generic labels such as "
           "that, this, thing, get, go, make, might, idea, food, stuff, "
           "almost done, user, assistant. "
           "Relations, when present, must use one of these predicates: "
         "co_occurs, implies, contradicts, reinforces, causes, similar_to. "
         "Every relation subject and object must exactly match one final "
         "label; omit unclear or unsupported relations. "
         "Current labels: "
         + FormatCurrentLabelsForPrompt (current_labels)
         + "\nEvidence:\n" + combined_text;
}

std::optional<std::vector<float>>
DecodeFloat32Blob (const std::vector<unsigned char> &bytes)
{
  if (bytes.empty () || bytes.size () % sizeof (float) != 0)
    {
      return std::nullopt;
    }
  std::vector<float> out (bytes.size () / sizeof (float));
  std::memcpy (out.data (), bytes.data (), bytes.size ());
  return out;
}

std::string
LoadDerivedEvidenceText (Transaction &tx, long long association_memory_id)
{
  if (association_memory_id <= 0)
    {
      return {};
    }

  auto rows = tx.Execute (
      "SELECT m.blob_id FROM associations a "
      "JOIN memories m ON m.memory_id = a.target_memory_id "
      "WHERE a.source_memory_id = ? AND a.edge_type = 'derived_from' "
      "AND m.blob_id IS NOT NULL",
      { association_memory_id });

  std::string text;
  for (const auto &row : rows)
    {
      auto it = row.find ("blob_id");
      if (it == row.end ())
        {
          continue;
        }
      const auto blob_id = store::BlobFromAny (it->second);
      if (blob_id.empty ())
        {
          continue;
        }
      auto payload_rows
          = tx.Execute ("SELECT objstore_get(?1) AS payload", { blob_id });
      if (payload_rows.empty () || payload_rows[0].count ("payload") == 0)
        {
          continue;
        }
      const auto payload = store::BlobFromAny (payload_rows[0].at ("payload"));
      if (payload.empty ())
        {
          continue;
        }
      if (!text.empty ())
        {
          text += "\n---\n";
        }
      text.append (reinterpret_cast<const char *> (payload.data ()),
                   payload.size ());
    }
  return text;
}

} // namespace

void
ProcessExtractionResults::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const double label_cooccurrence_edge_weight
      = core::LabelCooccurrenceEdgeWeight (cfg.focus, cfg.sensitivity,
                                           cfg.stability);

  // Get extractor (may be null if OGA disabled)
  Extractor *extractor = context.GetExtractor ();

  // Process extraction requests using in-process extractor if available
  const auto &requests = context.GetExtractionRequests ();
  std::vector<Extractor::BatchTextItem> extraction_batch_items;
  std::unordered_map<std::string, std::string> request_evidence_text;
  std::unordered_map<std::string, bool> request_replaces_labels;
  std::unordered_map<std::string, bool> request_has_blob_evidence;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      request_current_label_keys;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      request_source_span_label_keys;
  std::unordered_map<std::string, long long> request_source_span_counts;
  std::unordered_map<std::string, uint64_t> request_created_at;
  std::unordered_map<std::string,
                     std::vector<std::pair<std::string, std::string>>>
      request_current_labels_by_key;
  if (extractor && extractor->IsAvailable () && !requests.empty ())
    {
      for (const auto &req : requests)
        {
          request_created_at[req.summary_id] = req.created_at;
          internal::ThrowIfStopRequested ();
          try
            {
              // Extract from raw episodic source texts (not the compressed
              // summary). The summary serves as a retrieval node but
              // extraction works on the original evidence to avoid lossy
              // compression hiding extractable facts.
              std::string combined_text;
              if (!req.source_texts.empty ())
                {
                  for (const auto &txt : req.source_texts)
                    {
                      if (!combined_text.empty ())
                        combined_text += "\n---\n";
                      combined_text += txt;
                    }
                }
              else
                {
                  combined_text = req.summary_text;
                }
              request_evidence_text[req.summary_id] = combined_text;
              request_replaces_labels[req.summary_id] = true;
              request_has_blob_evidence[req.summary_id]
                  = std::any_of (
                      req.source_blobs.begin (), req.source_blobs.end (),
                      [] (const ExtractionSourceBlob &blob) {
                        return blob.modality == "audio"
                               || blob.modality == "image";
                      });
              auto &current_keys = request_current_label_keys[req.summary_id];
              auto &current_labels_by_key
                  = request_current_labels_by_key[req.summary_id];
              std::vector<std::string> refinement_current_labels
                  = req.current_labels;
	              const int source_span_candidate_limit
	                  = core::STMLTMSourceSpanCandidateLimit (cfg.focus,
	                                                          cfg.sensitivity,
	                                                          cfg.stability);
              const auto source_span_policy
                  = core::STMLTMSourceSpanCandidatePolicy (
                      cfg.focus, cfg.sensitivity, cfg.stability);
	              const auto source_span_candidates = BuildSourceSpanCandidates (
	                  combined_text, source_span_candidate_limit,
                      source_span_policy);
              auto &source_span_keys
                  = request_source_span_label_keys[req.summary_id];
              request_source_span_counts[req.summary_id]
                  = static_cast<long long> (source_span_candidates.size ());
              refinement_current_labels.insert (
                  refinement_current_labels.end (),
                  source_span_candidates.begin (),
                  source_span_candidates.end ());
              for (const auto &label : refinement_current_labels)
                {
                  const std::string key = NormalizeLabelKey (label);
                  if (!key.empty ())
                    {
                      if (current_keys.insert (key).second)
                        {
                          current_labels_by_key.emplace_back (key, label);
                        }
                      if (std::find (source_span_candidates.begin (),
                                     source_span_candidates.end (), label)
                          != source_span_candidates.end ())
                        {
                          source_span_keys.insert (key);
                        }
                    }
                }

              operations::ExtractionResult result;
              bool extracted = false;
              const auto extraction_call_start
                  = std::chrono::steady_clock::now ();
              if (auto *gemma = dynamic_cast<GemmaExtractor *> (extractor))
                {
                  for (const auto &source_blob : req.source_blobs)
                    {
                      if (source_blob.modality == "audio")
                        {
                          auto pcm = DecodeFloat32Blob (source_blob.bytes);
                          if (!pcm.has_value () || pcm->empty ())
                            {
                              continue;
                            }
                          try
                            {
                              result = gemma->RefineLabelsFromAudio (
                                  pcm->data (), pcm->size (),
                                  refinement_current_labels,
                                  kExtractionSchema);
                              extracted = true;
                              break;
                            }
                          catch (const std::exception &)
                            {
                              continue;
                            }
                        }
                      if (source_blob.modality == "image")
                        {
                          try
                            {
                              result = gemma->RefineLabelsFromImage (
                                  source_blob.bytes,
                                  refinement_current_labels,
                                  kExtractionSchema);
                              extracted = true;
                              break;
                            }
                          catch (const std::exception &)
                            {
                              continue;
                            }
                        }
                    }
                  if (!extracted && !combined_text.empty ())
                    {
                      result = gemma->RefineLabelsFromText (
                          combined_text, refinement_current_labels,
                          kExtractionSchema);
                      extracted = true;
                    }
                }
              if (!extracted)
                {
                  const std::string extraction_text
                      = refinement_current_labels.empty ()
	                            ? combined_text
	                            : BuildTextLabelRefinementEvidence (
	                                  combined_text, refinement_current_labels,
	                                  cfg.focus, cfg.sensitivity,
	                                  cfg.stability);
                  extraction_batch_items.push_back (
                      { req.summary_id, extraction_text });
                  continue;
                }
              const auto extraction_call_end
                  = std::chrono::steady_clock::now ();
              context.AddOperationTiming (
                  "ProcessExtractionResults.extractor_call",
                  std::chrono::duration<double, std::milli> (
                      extraction_call_end - extraction_call_start)
                      .count ());
              result.summary_id = req.summary_id;

              // Add to pending results for processing
              p_ctx.pending_extraction_results.push_back (std::move (result));
            }
          catch (const std::exception &e)
            {
              telemetry::LogWarn (
                  "cortext.extraction_failed",
                  { telemetry::Attribute::String ("summary_id",
                                                  req.summary_id),
                    telemetry::Attribute::Int64 ("cluster_size",
                                                 req.cluster_size),
                    telemetry::Attribute::String ("error", e.what ()) });
              // Skip failed extractions
            }
        }
      if (!extraction_batch_items.empty ())
        {
          const auto extraction_batch_start
              = std::chrono::steady_clock::now ();
          try
            {
              auto results = extractor->ExtractBatchFromTexts (
                  extraction_batch_items, kExtractionSchema);
              const auto extraction_batch_end
                  = std::chrono::steady_clock::now ();
              const double batch_ms
                  = std::chrono::duration<double, std::milli> (
                        extraction_batch_end - extraction_batch_start)
                        .count ();
              context.AddOperationTiming (
                  "ProcessExtractionResults.extractor_batch_call", batch_ms);
              context.AddOperationTiming (
                  "ProcessExtractionResults.extractor_call", batch_ms);
              for (std::size_t i = 0; i < results.size ()
                                      && i < extraction_batch_items.size ();
                   ++i)
                {
                  results[i].summary_id = extraction_batch_items[i].id;
                  p_ctx.pending_extraction_results.push_back (
                      std::move (results[i]));
                }
            }
          catch (const std::exception &e)
            {
              telemetry::LogWarn (
                  "cortext.extraction_batch_failed",
                  { telemetry::Attribute::Int64 (
                        "request_count",
                        static_cast<std::int64_t> (
                            extraction_batch_items.size ())),
                    telemetry::Attribute::String ("error", e.what ()) });
              const auto extraction_batch_end
                  = std::chrono::steady_clock::now ();
              context.AddOperationTiming (
                  "ProcessExtractionResults.extractor_batch_failed_ms",
                  std::chrono::duration<double, std::milli> (
                      extraction_batch_end - extraction_batch_start)
                      .count ());
              for (const auto &item : extraction_batch_items)
                {
                  const auto extraction_call_start
                      = std::chrono::steady_clock::now ();
                  try
                    {
                      auto result = extractor->ExtractFromText (
                          item.text, kExtractionSchema);
                      const auto extraction_call_end
                          = std::chrono::steady_clock::now ();
                      context.AddOperationTiming (
                          "ProcessExtractionResults.extractor_call",
                          std::chrono::duration<double, std::milli> (
                              extraction_call_end - extraction_call_start)
                              .count ());
                      result.summary_id = item.id;
                      p_ctx.pending_extraction_results.push_back (
                          std::move (result));
                    }
                  catch (const std::exception &single_error)
                    {
                      telemetry::LogWarn (
                          "cortext.extraction_failed",
                          { telemetry::Attribute::String ("summary_id",
                                                          item.id),
                            telemetry::Attribute::String (
                                "error", single_error.what ()) });
                    }
                }
            }
        }
    }
  else if (!requests.empty ())
    {
      // Extractor not available - invoke callback if set
      auto *callback = context.GetExtractionCallback ();
      if (callback && *callback)
        {
          (*callback) (requests);
        }
    }

  // Process pending extraction results (from extractor or external callback)
  if (p_ctx.pending_extraction_results.empty ())
    {
      return;
    }

  int total_results = static_cast<int> (p_ctx.pending_extraction_results.size ());
  int total_labels = 0;
  int total_relations = 0;
  int total_facts = 0;
  std::vector<long long> touched_fact_ids;
  for (const auto &pending : p_ctx.pending_extraction_results)
    {
      total_labels += static_cast<int> (pending.labels.size ());
      total_relations += static_cast<int> (pending.relations.size ());
      total_facts += static_cast<int> (pending.facts.size ());
    }

  const uint64_t now_ts = context.GetSignal ().timestamp;
  Encoder *lifecycle_encoder = context.GetConfig ().encoder;
  const bool use_legacy_label_gate = UseLegacyLabelFrequencyGate ();
  const int label_threshold
      = use_legacy_label_gate
            ? core::LabelFrequencyThreshold (cfg.stability)
            : 0;
  std::unordered_map<std::string, int> label_counts;
  if (use_legacy_label_gate)
    {
      for (const auto &pending : p_ctx.pending_extraction_results)
        {
          for (const auto &label_entry : pending.labels)
            {
              const std::string label_key
                  = NormalizeLabelKey (label_entry.label);
              if (label_key.empty ())
                {
                  continue;
                }
              label_counts[label_key] += 1;
            }
        }
    }
  struct ExistingLabel
  {
    long long memory_id = 0;
    long long embedding_id = 0;
    bool loaded = false;
  };

      auto &label_cache = p_ctx.label_embedding_cache;
      auto upsert_summary_cache = [&](long long memory_id,
                                      long long embedding_id,
                                      const std::vector<float> *embedding) {
        if (!embedding || embedding->empty ())
          {
            return;
          }
        Eigen::VectorXf vec (
            static_cast<Eigen::Index> (embedding->size ()));
        for (size_t i = 0; i < embedding->size (); ++i)
          {
            vec (static_cast<Eigen::Index> (i)) = (*embedding)[i];
          }
        p_ctx.UpsertSummaryCache (memory_id, embedding_id, vec, false, true);
      };
  std::unordered_map<std::string, ExistingLabel> existing_labels;
  auto get_existing = [&](const std::string &label_key) -> ExistingLabel & {
    auto it = existing_labels.find (label_key);
    if (it != existing_labels.end ())
      {
        return it->second;
      }
    ExistingLabel entry;
    auto rows = tx.Execute (
        "SELECT memory_id, embedding_id FROM memories "
        "WHERE source_id = ? AND kind = 'LABEL'",
        { label_key });
    if (!rows.empty ())
      {
        const auto &row = rows[0];
        auto mem_it = row.find ("memory_id");
        if (mem_it != row.end ())
          {
            if (mem_it->second.type () == typeid (long long))
              {
                entry.memory_id = std::any_cast<long long> (mem_it->second);
              }
            else if (mem_it->second.type () == typeid (int))
              {
                entry.memory_id = std::any_cast<int> (mem_it->second);
              }
          }
        auto emb_it = row.find ("embedding_id");
        if (emb_it != row.end ())
          {
            if (emb_it->second.type () == typeid (long long))
              {
                entry.embedding_id
                    = std::any_cast<long long> (emb_it->second);
              }
            else if (emb_it->second.type () == typeid (int))
              {
                entry.embedding_id = std::any_cast<int> (emb_it->second);
              }
          }
      }
    entry.loaded = true;
    auto [ins_it, _] = existing_labels.emplace (label_key, entry);
    return ins_it->second;
  };

  uint64_t fact_maintenance_ts = 0;
  for (const auto &result : p_ctx.pending_extraction_results)
    {
      // Find summary memory for has_label edges.
      long long summary_memory_id = 0;
      long long summary_embedding_id = 0;
      long long summary_start_ts = 0;
      auto summary_rows = tx.Execute (
          "SELECT memory_id, embedding_id, start_ts FROM memories "
          "WHERE source_id = ? AND kind = 'ASSOCIATION'",
          { result.summary_id });
      if (!summary_rows.empty ())
        {
          const auto &row = summary_rows[0];
          summary_memory_id = ExtractInt64Field (row, "memory_id");
          summary_embedding_id = ExtractInt64Field (row, "embedding_id");
          summary_start_ts = ExtractInt64Field (row, "start_ts");
        }
      auto request_created_it = request_created_at.find (result.summary_id);
      const uint64_t result_ts
          = request_created_it != request_created_at.end ()
                    && request_created_it->second > 0
                ? request_created_it->second
                : now_ts;
      fact_maintenance_ts = std::max (fact_maintenance_ts, result_ts);

      const auto summary_embedding
          = summary_embedding_id > 0
                ? LoadEmbedding (tx, summary_embedding_id)
                : std::optional<Eigen::VectorXf> ();
      const auto evidence_it = request_evidence_text.find (result.summary_id);
      const std::string evidence_text
          = evidence_it != request_evidence_text.end ()
                ? evidence_it->second
                : LoadDerivedEvidenceText (tx, summary_memory_id);
      std::string full_source_evidence_text;
      if (StmLtmAuditEnabled ())
        {
          full_source_evidence_text = LoadDerivedEvidenceText (
              tx, summary_memory_id);
          if (full_source_evidence_text.empty ())
            {
              full_source_evidence_text = evidence_text;
            }
        }
      const auto has_blob_it = request_has_blob_evidence.find (
          result.summary_id);
      const bool has_blob_evidence
          = has_blob_it != request_has_blob_evidence.end ()
                && has_blob_it->second;
      Encoder *encoder = context.GetConfig ().encoder;
      if (!encoder)
        {
          throw std::runtime_error (
              "ProcessExtractionResults requires a non-null Encoder");
        }
      const auto previous_label_keys = LoadAttachedLabelKeys (
          tx, summary_memory_id);

      // Track label -> memory_id for relation linking
      std::unordered_map<std::string, long long> label_memory_ids;
      bool inserted_any_label = false;
      std::vector<std::string> refined_label_texts;
      long long relation_count = 0;
      long long relation_edges_created = 0;
      long long label_cooccurrence_edges_created = 0;
      long long relation_edges_skipped_non_durable_endpoint = 0;
      long long relation_edges_skipped_missing_endpoint = 0;
      long long relation_edges_skipped_unsupported_predicate = 0;
      long long relation_endpoint_direct_hits = 0;
      long long relation_endpoint_repair_hits = 0;
      long long relation_endpoint_created_labels = 0;
      long long relation_endpoint_relation_backed_labels = 0;
      long long relation_endpoint_rejected_count = 0;
      long long relation_endpoint_rejected_non_durable = 0;
      long long relation_endpoint_rejected_ungrounded = 0;
      long long label_candidates_rejected_non_durable = 0;
      long long label_candidates_rejected_ungrounded = 0;
      long long label_candidates_rejected_low_contrast = 0;
      long long label_candidates_rejected_duplicate = 0;
      long long label_candidates_rejected_legacy_gate = 0;
      long long labels_inserted_from_extractor = 0;
      long long labels_inserted_from_current_floor = 0;
      long long labels_inserted_from_source_span_floor = 0;
      long long labels_inserted_from_relation_endpoint = 0;
      long long facts_touched_for_result = 0;
      struct FallbackLabel
      {
        std::string label;
        std::string label_key;
        double salience = 0.0;
        std::vector<float> embedding;
        bool has_embedding = false;
      };
      std::optional<FallbackLabel> fallback_label;
      std::unordered_set<std::string> inserted_label_keys;

      auto insert_label = [&](const std::string &label,
                              const std::string &label_key,
                              double salience,
                              const std::vector<float> *label_embedding,
                              ExistingLabel &existing) -> bool {
        if (label.empty () || label_key.empty ())
          {
            return false;
          }

        // Use label text as source_id for uniqueness.
        std::string source_id = label_key;

        long long label_memory_id = existing.memory_id;
        long long existing_embedding_id = existing.embedding_id;
        const std::vector<float> *embedding_for_store
            = (label_embedding != nullptr
               && label_embedding->size () == kEmbeddingDim)
                  ? label_embedding
                  : nullptr;

        if (label_memory_id > 0)
          {
            // Update salience if higher
            AddWrite (tx,
                      "UPDATE memories SET s_max = MAX(s_max, ?) "
                      "WHERE memory_id = ?",
                      { salience, label_memory_id });
          }
        else
          {
            long long embedding_id = 0;
            if (embedding_for_store != nullptr && !embedding_for_store->empty ())
              {
                AddWrite (tx,
                          "INSERT INTO embeddings (embedding, created_at) "
                          "VALUES (?, ?)",
                          { *embedding_for_store,
                            static_cast<long long> (result_ts) });
                auto emb_rows
                    = tx.Execute ("SELECT last_insert_rowid() AS id", {});
                if (!emb_rows.empty () && emb_rows[0].count ("id"))
                  {
                    auto emb_val = emb_rows[0].at ("id");
                    if (emb_val.type () == typeid (long long))
                      {
                        embedding_id
                            = std::any_cast<long long> (emb_val);
                      }
                    else if (emb_val.type () == typeid (int))
                      {
                        embedding_id = std::any_cast<int> (emb_val);
                      }
                  }
              }
            // Insert new label as MEMORIES row
            AddWrite (tx,
                      "INSERT INTO memories "
                      "(embedding_id, source_id, kind, label, start_ts, "
                      "s_max, created_at) "
                      "VALUES (?, ?, 'LABEL', ?, ?, ?, ?)",
                      { embedding_id > 0 ? std::any (embedding_id) : std::any (),
                        source_id, label,
                        static_cast<long long> (result_ts), salience,
                        static_cast<long long> (result_ts) });

            // Get the new memory_id
            auto id_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
            if (!id_rows.empty () && id_rows[0].count ("id"))
              {
                auto val = id_rows[0].at ("id");
                if (val.type () == typeid (long long))
                  {
                    label_memory_id = std::any_cast<long long> (val);
                  }
                else if (val.type () == typeid (int))
                  {
                    label_memory_id = std::any_cast<int> (val);
                  }
              }
            existing.memory_id = label_memory_id;
            existing.embedding_id = embedding_id;
            existing_embedding_id = embedding_id;
          }

        if (label_memory_id > 0 && existing_embedding_id == 0
            && embedding_for_store != nullptr && !embedding_for_store->empty ())
          {
            AddWrite (tx,
                      "INSERT INTO embeddings (embedding, created_at) "
                      "VALUES (?, ?)",
                      { *embedding_for_store,
                        static_cast<long long> (result_ts) });
            auto emb_rows
                = tx.Execute ("SELECT last_insert_rowid() AS id", {});
            long long embedding_id = 0;
            if (!emb_rows.empty () && emb_rows[0].count ("id"))
              {
                auto emb_val = emb_rows[0].at ("id");
                if (emb_val.type () == typeid (long long))
                  {
                    embedding_id = std::any_cast<long long> (emb_val);
                  }
                else if (emb_val.type () == typeid (int))
                  {
                    embedding_id = std::any_cast<int> (emb_val);
                  }
              }
            if (embedding_id > 0)
              {
                AddWrite (tx,
                          "UPDATE memories SET embedding_id = ? "
                          "WHERE memory_id = ?",
                          { embedding_id, label_memory_id });
                existing.embedding_id = embedding_id;
                upsert_summary_cache (label_memory_id, embedding_id,
                                      embedding_for_store);
              }
          }

        // Attach label to summary (has_label) and track for relation linking
        if (label_memory_id > 0)
          {
            label_memory_ids[label_key] = label_memory_id;
            if (existing.embedding_id > 0)
              {
                upsert_summary_cache (label_memory_id, existing.embedding_id,
                                      embedding_for_store);
              }
            if (summary_memory_id > 0)
              {
                const double weight01 = core::Clamp (salience, 0.0, 1.0);
                AddWrite (tx,
                          "INSERT OR REPLACE INTO associations "
                          "(source_memory_id, target_memory_id, edge_type, weight) "
                          "VALUES (?, ?, 'has_label', ?)",
                          { summary_memory_id, label_memory_id, weight01 });
              }
            return true;
          }
        return false;
      };

      auto resolve_label_embedding =
          [&] (const std::string &label,
               const std::string &label_key,
               ExistingLabel &existing) -> const std::vector<float> * {
        auto cached_it = label_cache.find (label_key);
        if (cached_it != label_cache.end ())
          {
            return &cached_it->second;
          }
        if (existing.embedding_id > 0)
          {
            auto loaded = LoadEmbeddingVector (tx, existing.embedding_id);
            if (loaded.has_value ())
              {
                auto [it, _] = label_cache.emplace (label_key,
                                                    std::move (*loaded));
                return &it->second;
              }
          }
        auto static_loaded = LoadAttachedLabelBankEmbedding (tx, label_key);
        if (static_loaded.has_value ())
          {
            auto [it, _] = label_cache.emplace (label_key,
                                                std::move (*static_loaded));
            return &it->second;
          }
        auto encoded = EncodeLabelEmbedding (label, *encoder);
        if (encoded.has_value ())
          {
            auto [it, _] = label_cache.emplace (label_key,
                                                std::move (*encoded));
            return &it->second;
          }
        return nullptr;
      };

      // Contrast is measured against the bank as it existed before this
      // result's admissions (see ComputeLabelBankContrast).
      const size_t label_bank_snapshot = p_ctx.summary_cache.size ();
      // The optional admission paths (floor fill, relation-endpoint
      // creation) only run once the contrast gate can vet candidates.
      // Before the bank matures, forcing unvetted labels seeds it with
      // cold-start junk that the rest of the run then measures against;
      // fewer labels is strictly better than unvetted ones. The primary
      // extractor path still admits so the bank can seed at all.
      const bool contrast_gate_available
          = CountContrastBankLabels (p_ctx, label_bank_snapshot)
            >= static_cast<size_t> (kContrastMinBankSize);
      // A single-token label whose token already appears inside an admitted
      // multi-token label for this summary is a redundant fragment ("Cart"
      // next to "cart situation"); standalone single-token labels ("Dog")
      // are fine. Multi-token labels are admitted first so the check sees
      // the full batch.
      std::unordered_set<std::string> admitted_multi_tokens;
      auto note_admitted_label_tokens = [&] (const std::string &label_key) {
        const std::string canonical = CanonicalLabelTokenKey (label_key);
        if (CanonicalTokenCount (canonical) < 2)
          {
            return;
          }
        std::string token;
        for (unsigned char c : canonical)
          {
            if (std::isalnum (c) != 0)
              {
                token.push_back (static_cast<char> (c));
              }
            else if (!token.empty ())
              {
                admitted_multi_tokens.insert (token);
                token.clear ();
              }
          }
        if (!token.empty ())
          {
            admitted_multi_tokens.insert (token);
          }
      };
      auto single_token_subsumed = [&] (const std::string &label_key) {
        const std::string canonical = CanonicalLabelTokenKey (label_key);
        if (CanonicalTokenCount (canonical) != 1)
          {
            return false;
          }
        return admitted_multi_tokens.find (canonical)
               != admitted_multi_tokens.end ();
      };

      // 1. Insert labels into MEMORIES (kind='LABEL'). Multi-token labels
      // go first so single-token fragments can be checked against them.
      std::vector<const operations::ExtractedLabel *> ordered_labels;
      ordered_labels.reserve (result.labels.size ());
      for (const auto &label_entry : result.labels)
        {
          if (CanonicalTokenCount (CanonicalLabelTokenKey (
                  NormalizeLabelKey (label_entry.label))) >= 2)
            {
              ordered_labels.push_back (&label_entry);
            }
        }
      for (const auto &label_entry : result.labels)
        {
          if (CanonicalTokenCount (CanonicalLabelTokenKey (
                  NormalizeLabelKey (label_entry.label))) < 2)
            {
              ordered_labels.push_back (&label_entry);
            }
        }
      for (const auto *label_entry_ptr : ordered_labels)
        {
          const auto &label_entry = *label_entry_ptr;
          const std::string label = TrimLabel (label_entry.label);
          const std::string label_key = NormalizeLabelKey (label_entry.label);
          if (!IsDurableLabelCandidate (label, label_key))
            {
              ++label_candidates_rejected_non_durable;
              continue;
            }
          if (!LabelAppearsInEvidence (label_key, evidence_text))
            {
              ++label_candidates_rejected_ungrounded;
              continue;
            }
          if (single_token_subsumed (label_key))
            {
              ++label_candidates_rejected_duplicate;
              continue;
            }
          if (!inserted_label_keys.insert (label_key).second)
            {
              ++label_candidates_rejected_duplicate;
              continue;
            }
          auto &existing = get_existing (label_key);
          const std::vector<float> *label_embedding_ptr
              = resolve_label_embedding (label, label_key, existing);
          if (label_embedding_ptr != nullptr)
            {
              const auto contrast
                  = ComputeLabelBankContrast (*label_embedding_ptr, p_ctx,
                                              label_bank_snapshot);
              if (IsGenericByLabelBankContrast (contrast))
                {
                  ++label_candidates_rejected_low_contrast;
                  inserted_label_keys.erase (label_key);
                  telemetry::LogDebug (
                      "cortext.label_admission",
                      { telemetry::Attribute::String ("label", label),
                        telemetry::Attribute::String ("decision",
                                                      "rejected_low_contrast"),
                        telemetry::Attribute::Double (
                            "mean_similarity", contrast.mean_similarity),
                        telemetry::Attribute::Double (
                            "max_similarity", contrast.max_similarity) });
                  continue;
                }
            }
          const double salience
              = ComputeLabelSalience (label_embedding_ptr, summary_embedding,
                                      cfg.focus, cfg.sensitivity,
                                      cfg.stability);
          if (use_legacy_label_gate && label_threshold > 1
              && label_counts[label_key] < label_threshold)
            {
              ++label_candidates_rejected_legacy_gate;
              if (!fallback_label || salience > fallback_label->salience)
                {
                  FallbackLabel candidate;
                  candidate.label = label;
                  candidate.label_key = label_key;
                  candidate.salience = salience;
                  if (label_embedding_ptr && !label_embedding_ptr->empty ())
                    {
                      candidate.embedding = *label_embedding_ptr;
                      candidate.has_embedding = true;
                    }
                  fallback_label = std::move (candidate);
                }
              continue;
            }

          if (insert_label (label, label_key, salience, label_embedding_ptr,
                            existing))
            {
              inserted_any_label = true;
              ++labels_inserted_from_extractor;
              refined_label_texts.push_back (label);
              note_admitted_label_tokens (label_key);
            }
        }

      if (use_legacy_label_gate && !inserted_any_label
          && fallback_label.has_value ())
        {
          const std::vector<float> *fallback_embedding
              = fallback_label->has_embedding
                    ? &fallback_label->embedding
                    : nullptr;
          if (insert_label (fallback_label->label,
                            fallback_label->label_key,
                            fallback_label->salience,
                            fallback_embedding,
                            get_existing (fallback_label->label_key)))
            {
              inserted_label_keys.insert (fallback_label->label_key);
              ++labels_inserted_from_extractor;
              refined_label_texts.push_back (fallback_label->label);
            }
        }

      const auto replace_it = request_replaces_labels.find (result.summary_id);
	      if (summary_memory_id > 0 && replace_it != request_replaces_labels.end ()
	          && replace_it->second)
	        {
		          const int min_durable_labels
		              = core::STMLTMDurableMinLabels (
		                  cfg.focus, cfg.sensitivity, cfg.stability);
		          const int max_durable_labels
		              = core::STMLTMDurableMaxLabels (
		                  cfg.focus, cfg.sensitivity, cfg.stability);
	          auto current_label_texts_it = request_current_labels_by_key.find (
	              result.summary_id);
	          const auto source_span_keys_it = request_source_span_label_keys.find (
	              result.summary_id);
	          auto admit_current_or_source_span_label =
	              [&] (const std::string &label_key,
	                   const std::string &original_label) {
	            if (!contrast_gate_available)
	              {
	                telemetry::LogDebug (
	                    "cortext.label_admission",
	                    { telemetry::Attribute::String ("label",
	                                                    original_label),
	                      telemetry::Attribute::String (
	                          "decision", "rejected_cold_start_floor") });
	                return false;
	              }
	            if (inserted_label_keys.find (label_key)
	                != inserted_label_keys.end ())
	              {
	                return false;
	              }
	            const std::string label = TrimLabel (original_label);
	            if (!IsDurableLabelCandidate (label, label_key)
	                || (!has_blob_evidence
	                    && !LabelAppearsInEvidence (label_key, evidence_text))
	                || single_token_subsumed (label_key))
	              {
	                return false;
	              }
	            auto &existing = get_existing (label_key);
	            const std::vector<float> *label_embedding_ptr
	                = resolve_label_embedding (label, label_key, existing);
	            if (label_embedding_ptr != nullptr)
	              {
	                const auto contrast
	                    = ComputeLabelBankContrast (*label_embedding_ptr, p_ctx,
	                                          label_bank_snapshot);
	                if (IsGenericByLabelBankContrast (contrast))
	                  {
	                    ++label_candidates_rejected_low_contrast;
	                    telemetry::LogDebug (
	                        "cortext.label_admission",
	                        { telemetry::Attribute::String ("label", label),
	                          telemetry::Attribute::String (
	                              "decision", "rejected_low_contrast"),
	                          telemetry::Attribute::Double (
	                              "mean_similarity", contrast.mean_similarity),
	                          telemetry::Attribute::Double (
	                              "max_similarity", contrast.max_similarity) });
	                    return false;
	                  }
	              }
	            const double salience = ComputeLabelSalience (
	                label_embedding_ptr, summary_embedding, cfg.focus,
	                cfg.sensitivity, cfg.stability);
	            if (!insert_label (label, label_key, salience,
	                               label_embedding_ptr, existing))
	              {
	                return false;
	              }
	            inserted_any_label = true;
	            inserted_label_keys.insert (label_key);
	            if (source_span_keys_it != request_source_span_label_keys.end ()
	                && source_span_keys_it->second.find (label_key)
	                       != source_span_keys_it->second.end ())
	              {
	                ++labels_inserted_from_source_span_floor;
	              }
	            else
	              {
	                ++labels_inserted_from_current_floor;
	              }
	            refined_label_texts.push_back (label);
	            return true;
	          };
	          if (min_durable_labels > 0
	              && current_label_texts_it != request_current_labels_by_key.end ())
	            {
	              auto try_floor_labels = [&] (bool require_source_span) {
	                for (const auto &[label_key, original_label] :
	                     current_label_texts_it->second)
	                  {
	                    if (static_cast<int> (inserted_label_keys.size ())
	                        >= min_durable_labels)
	                      {
	                        break;
	                      }
	                    const bool is_source_span
	                        = source_span_keys_it
	                              != request_source_span_label_keys.end ()
	                          && source_span_keys_it->second.find (label_key)
	                                 != source_span_keys_it->second.end ();
	                    if (require_source_span != is_source_span)
	                      {
	                        continue;
	                      }
	                    (void)admit_current_or_source_span_label (
	                        label_key, original_label);
	                  }
	              };
	              try_floor_labels (true);
	              if (static_cast<int> (inserted_label_keys.size ())
	                  < min_durable_labels)
	                {
	                  try_floor_labels (false);
	                }
	            }
	          if (max_durable_labels > min_durable_labels
	              && current_label_texts_it != request_current_labels_by_key.end ()
	              && source_span_keys_it != request_source_span_label_keys.end ())
	            {
	              for (const auto &[label_key, original_label] :
	                   current_label_texts_it->second)
	                {
	                  if (static_cast<int> (inserted_label_keys.size ())
	                      >= max_durable_labels)
	                    {
	                      break;
	                    }
	                  if (source_span_keys_it->second.find (label_key)
	                      == source_span_keys_it->second.end ())
	                    {
	                      continue;
	                    }
	                  (void)admit_current_or_source_span_label (
	                      label_key, original_label);
	                }
	            }

	        }

      const int relation_endpoint_alias_min_shared
          = core::STMLTMRelationEndpointAliasMinSharedTokens (
              cfg.focus, cfg.sensitivity, cfg.stability);
	      auto ensure_relation_endpoint_label =
	          [&] (const std::string &endpoint,
	               double relation_confidence) -> long long {
        const std::string label = TrimLabel (endpoint);
        const std::string label_key = NormalizeLabelKey (endpoint);
        auto existing_it = label_memory_ids.find (label_key);
        if (existing_it != label_memory_ids.end ())
          {
            return existing_it->second;
          }
        const std::string endpoint_tokens = CanonicalLabelTokenKey (label_key);
        if (!endpoint_tokens.empty ())
          {
            long long repaired_id = 0;
            size_t repaired_token_chars = 0;
            for (const auto &[candidate_key, candidate_id] : label_memory_ids)
              {
                const std::string candidate_tokens
                    = CanonicalLabelTokenKey (candidate_key);
	                if (!StrongEndpointAliasMatch (
	                        endpoint_tokens, candidate_tokens,
	                        relation_endpoint_alias_min_shared))
                  {
                    continue;
                  }
                if (candidate_tokens.size () > repaired_token_chars)
                  {
                    repaired_id = candidate_id;
                    repaired_token_chars = candidate_tokens.size ();
                  }
              }
            if (repaired_id > 0)
              {
                ++relation_endpoint_repair_hits;
                return repaired_id;
              }
          }

        const auto current_label_texts_it = request_current_labels_by_key.find (
            result.summary_id);
        if (!endpoint_tokens.empty ()
            && current_label_texts_it != request_current_labels_by_key.end ())
          {
            std::string alias_key;
            std::string alias_label;
            size_t alias_token_chars = 0;
            for (const auto &[candidate_key, candidate_label] :
                 current_label_texts_it->second)
              {
                const std::string candidate_tokens
                    = CanonicalLabelTokenKey (candidate_key);
	                if (!StrongEndpointAliasMatch (
	                        endpoint_tokens, candidate_tokens,
	                        relation_endpoint_alias_min_shared))
                  {
                    continue;
                  }
                if (candidate_tokens.size () > alias_token_chars)
                  {
                    alias_key = candidate_key;
                    alias_label = candidate_label;
                    alias_token_chars = candidate_tokens.size ();
                  }
              }
            if (!alias_key.empty ())
              {
                auto admitted_it = label_memory_ids.find (alias_key);
                if (admitted_it != label_memory_ids.end ())
                  {
                    ++relation_endpoint_repair_hits;
                    return admitted_it->second;
                  }
                const std::string durable_alias_label = TrimLabel (alias_label);
                if (IsDurableLabelCandidate (durable_alias_label, alias_key))
                  {
                    auto &existing = get_existing (alias_key);
                    const std::vector<float> *label_embedding_ptr
                        = resolve_label_embedding (durable_alias_label,
                                                   alias_key, existing);
                    const double salience = ComputeLabelSalience (
                        label_embedding_ptr, summary_embedding, cfg.focus,
                        cfg.sensitivity, cfg.stability);
                    if (insert_label (durable_alias_label, alias_key,
                                      salience, label_embedding_ptr, existing))
                      {
                        inserted_any_label = true;
                        ++labels_inserted_from_relation_endpoint;
                        if (inserted_label_keys.insert (alias_key).second)
                          {
                            refined_label_texts.push_back (
                                durable_alias_label);
                          }
                        auto inserted_it = label_memory_ids.find (alias_key);
                        if (inserted_it != label_memory_ids.end ())
                          {
                            ++relation_endpoint_repair_hits;
                            return inserted_it->second;
                          }
                      }
                  }
              }
          }

	        const bool relation_backed_endpoint
	            = core::Clamp (relation_confidence, 0.0, 1.0)
	              >= core::STMLTMRelationEndpointMinConfidence (
	                  cfg.focus, cfg.sensitivity, cfg.stability);
        const bool endpoint_appears_in_evidence = LabelAppearsInEvidence (
            label_key, evidence_text);
        if (!IsDurableLabelCandidate (label, label_key))
          {
            ++relation_endpoint_rejected_count;
            ++relation_endpoint_rejected_non_durable;
            return 0;
          }
        if (!has_blob_evidence && !endpoint_appears_in_evidence
            && !relation_backed_endpoint)
          {
            ++relation_endpoint_rejected_count;
            ++relation_endpoint_rejected_ungrounded;
            return 0;
          }
        // Creating a label to back a relation endpoint is optional work;
        // it waits for a mature bank and passes the same contrast gate as
        // every other admission path.
        if (!contrast_gate_available)
          {
            ++relation_endpoint_rejected_count;
            telemetry::LogDebug (
                "cortext.label_admission",
                { telemetry::Attribute::String ("label", label),
                  telemetry::Attribute::String (
                      "decision", "rejected_cold_start_endpoint") });
            return 0;
          }

        auto &existing = get_existing (label_key);
        const std::vector<float> *label_embedding_ptr = nullptr;
        auto cached_it = label_cache.find (label_key);
        if (cached_it != label_cache.end ())
          {
            label_embedding_ptr = &cached_it->second;
          }
        else if (existing.embedding_id > 0)
          {
            auto loaded = LoadEmbeddingVector (tx, existing.embedding_id);
            if (loaded.has_value ())
              {
                auto [it, _] = label_cache.emplace (label_key,
                                                    std::move (*loaded));
                label_embedding_ptr = &it->second;
              }
          }
        if (!label_embedding_ptr)
          {
            auto static_loaded = LoadAttachedLabelBankEmbedding (tx, label_key);
            if (static_loaded.has_value ())
              {
                auto [it, _] = label_cache.emplace (
                    label_key, std::move (*static_loaded));
                label_embedding_ptr = &it->second;
              }
          }
        if (!label_embedding_ptr)
          {
            auto encoded = EncodeLabelEmbedding (label, *encoder);
            if (encoded.has_value ())
              {
                auto [it, _] = label_cache.emplace (label_key,
                                                    std::move (*encoded));
                label_embedding_ptr = &it->second;
              }
          }

        if (label_embedding_ptr != nullptr)
          {
            const auto contrast
                = ComputeLabelBankContrast (*label_embedding_ptr, p_ctx,
                                            label_bank_snapshot);
            if (IsGenericByLabelBankContrast (contrast))
              {
                ++relation_endpoint_rejected_count;
                ++label_candidates_rejected_low_contrast;
                telemetry::LogDebug (
                    "cortext.label_admission",
                    { telemetry::Attribute::String ("label", label),
                      telemetry::Attribute::String (
                          "decision", "rejected_low_contrast_endpoint"),
                      telemetry::Attribute::Double (
                          "mean_similarity", contrast.mean_similarity),
                      telemetry::Attribute::Double (
                          "max_similarity", contrast.max_similarity) });
                return 0;
              }
          }

        const double salience
            = ComputeLabelSalience (label_embedding_ptr, summary_embedding,
                                    cfg.focus, cfg.sensitivity,
                                    cfg.stability);
        if (insert_label (label, label_key, salience, label_embedding_ptr,
                          existing))
          {
            inserted_any_label = true;
            ++labels_inserted_from_relation_endpoint;
            if (inserted_label_keys.insert (label_key).second)
              {
                refined_label_texts.push_back (label);
              }
            auto inserted_it = label_memory_ids.find (label_key);
            if (inserted_it != label_memory_ids.end ())
              {
                ++relation_endpoint_created_labels;
                if (!has_blob_evidence && !endpoint_appears_in_evidence
                    && relation_backed_endpoint)
                  {
                    ++relation_endpoint_relation_backed_labels;
                  }
                return inserted_it->second;
              }
          }
        ++relation_endpoint_rejected_count;
        return 0;
      };

      std::unordered_set<std::string> removed_label_key_set;
      if (summary_memory_id > 0 && replace_it != request_replaces_labels.end ()
          && replace_it->second)
        {
          auto label_edge_rows = tx.Execute (
              "SELECT l.memory_id, l.source_id FROM associations a "
              "JOIN memories l ON l.memory_id = a.target_memory_id "
              "WHERE a.source_memory_id = ? AND a.edge_type = 'has_label' "
              "  AND l.kind = 'LABEL'",
              { summary_memory_id });
          for (const auto &row : label_edge_rows)
            {
              const long long label_memory_id
                  = ExtractInt64Field (row, "memory_id");
              std::string label_key;
              auto key_it = row.find ("source_id");
              if (key_it != row.end ()
                  && key_it->second.type () == typeid (std::string))
                {
                  label_key = NormalizeLabelKey (
                      std::any_cast<std::string> (key_it->second));
                }
              if (label_memory_id > 0
                  && (label_key.empty ()
                      || inserted_label_keys.find (label_key)
                             == inserted_label_keys.end ()))
                {
                  AddWrite (tx,
                            "DELETE FROM associations "
                            "WHERE source_memory_id = ? "
                            "  AND target_memory_id = ? "
                            "  AND edge_type = 'has_label'",
                            { summary_memory_id, label_memory_id });
                  if (!label_key.empty ())
                    {
                      removed_label_key_set.insert (label_key);
                    }
                }
            }
        }

      // 2. Insert relations into ASSOCIATIONS.
      for (const auto &relation : result.relations)
        {
          ++relation_count;
          const std::string edge_type = MapEdgeType (relation.predicate);
          if (edge_type.empty ())
            {
              ++relation_edges_skipped_unsupported_predicate;
              continue;
            }

	          const std::string subject_label = TrimLabel (relation.subject);
	          const std::string subject_key = NormalizeLabelKey (relation.subject);
	          const std::string object_label = TrimLabel (relation.object);
	          const std::string object_key = NormalizeLabelKey (relation.object);
	          const bool subject_durable = IsDurableLabelCandidate (
	              subject_label, subject_key);
	          const bool object_durable = IsDurableLabelCandidate (
	              object_label, object_key);

	          // Look up subject and object memory_ids
	          long long subject_id = 0;
          long long object_id = 0;

          auto it_subj = label_memory_ids.find (subject_key);
          if (it_subj != label_memory_ids.end ())
            {
              subject_id = it_subj->second;
              ++relation_endpoint_direct_hits;
            }

          auto it_obj = label_memory_ids.find (object_key);
          if (it_obj != label_memory_ids.end ())
            {
              object_id = it_obj->second;
              ++relation_endpoint_direct_hits;
            }
          if (subject_id <= 0)
            {
              subject_id = ensure_relation_endpoint_label (
                  relation.subject, relation.confidence);
            }
          if (object_id <= 0)
            {
              object_id = ensure_relation_endpoint_label (
                  relation.object, relation.confidence);
            }

          // Only create association if both labels exist and edge is supported
	          if (subject_id > 0 && object_id > 0)
	            {
              const double weight01
                  = core::Clamp (relation.confidence, 0.0, 1.0);
              AddWrite (tx,
                        "INSERT OR REPLACE INTO associations "
                        "(source_memory_id, target_memory_id, edge_type, weight) "
                        "VALUES (?, ?, ?, ?)",
                        { subject_id, object_id, edge_type, weight01 });
	              ++relation_edges_created;
	            }
	          else
	            {
	              if (!subject_durable || !object_durable)
	                {
	                  ++relation_edges_skipped_non_durable_endpoint;
	                }
	              else
	                {
	                  ++relation_edges_skipped_missing_endpoint;
	                }
	            }
	        }

      const int cooccurrence_label_limit
          = core::STMLTMLabelCooccurrenceMaxLabels (
              cfg.focus, cfg.sensitivity, cfg.stability);
      if (cooccurrence_label_limit > 1 && inserted_label_keys.size () >= 2)
        {
          std::vector<std::pair<std::string, long long>> admitted_labels;
          admitted_labels.reserve (inserted_label_keys.size ());
          for (const auto &label_key : inserted_label_keys)
            {
              auto id_it = label_memory_ids.find (label_key);
              if (id_it != label_memory_ids.end () && id_it->second > 0)
                {
                  admitted_labels.emplace_back (label_key, id_it->second);
                }
            }
          std::sort (admitted_labels.begin (), admitted_labels.end (),
                     [] (const auto &lhs, const auto &rhs) {
                       return lhs.first < rhs.first;
                     });
          if (static_cast<int> (admitted_labels.size ())
              > cooccurrence_label_limit)
            {
              admitted_labels.resize (
                  static_cast<size_t> (cooccurrence_label_limit));
            }

          for (size_t i = 0; i < admitted_labels.size (); ++i)
            {
              for (size_t j = i + 1; j < admitted_labels.size (); ++j)
                {
                  const long long lhs_id = admitted_labels[i].second;
                  const long long rhs_id = admitted_labels[j].second;
                  if (lhs_id <= 0 || rhs_id <= 0 || lhs_id == rhs_id)
                    {
                      continue;
                    }
                  auto existing_edge_rows = tx.Execute (
                      "SELECT COUNT(*) AS c FROM associations "
                      "WHERE edge_type = 'co_occurs' "
                      "  AND ((source_memory_id = ? AND target_memory_id = ?) "
                      "       OR (source_memory_id = ? AND target_memory_id = ?))",
                      { lhs_id, rhs_id, rhs_id, lhs_id });
                  if (!existing_edge_rows.empty ()
                      && ExtractInt64Field (existing_edge_rows[0], "c") > 0)
                    {
                      continue;
                    }
                  AddWrite (tx,
                            "INSERT INTO associations "
                            "(source_memory_id, target_memory_id, edge_type, weight) "
                            "VALUES (?, ?, 'co_occurs', ?)",
                            { lhs_id, rhs_id,
                              label_cooccurrence_edge_weight });
                  ++relation_edges_created;
                  ++label_cooccurrence_edges_created;
                }
            }
        }

      const auto current_it = request_current_label_keys.find (
          result.summary_id);
      const auto &baseline_label_keys
          = current_it != request_current_label_keys.end ()
                ? current_it->second
                : previous_label_keys;
      for (const auto &label_key : baseline_label_keys)
        {
          if (inserted_label_keys.find (label_key) == inserted_label_keys.end ())
            {
              removed_label_key_set.insert (label_key);
            }
        }

      long long kept_label_count = 0;
      long long added_label_count = 0;
      for (const auto &label_key : inserted_label_keys)
        {
          if (baseline_label_keys.find (label_key) != baseline_label_keys.end ())
            {
              ++kept_label_count;
            }
          else
            {
              ++added_label_count;
            }
        }
      std::vector<std::string> removed_label_keys (
          removed_label_key_set.begin (), removed_label_key_set.end ());
      std::sort (removed_label_keys.begin (), removed_label_keys.end ());

      const long long current_labels_in_selected_evidence
          = CountLabelKeysAppearingInEvidence (baseline_label_keys,
                                               evidence_text);
      const long long current_labels_in_full_source
          = CountLabelKeysAppearingInEvidence (baseline_label_keys,
                                               full_source_evidence_text);
      const long long removed_labels_in_selected_evidence
          = CountLabelKeysAppearingInEvidence (removed_label_key_set,
                                               evidence_text);
      const long long removed_labels_in_full_source
          = CountLabelKeysAppearingInEvidence (removed_label_key_set,
                                               full_source_evidence_text);
      const long long refined_labels_in_selected_evidence
          = CountLabelKeysAppearingInEvidence (inserted_label_keys,
                                               evidence_text);
      const long long refined_labels_in_full_source
          = CountLabelKeysAppearingInEvidence (inserted_label_keys,
                                               full_source_evidence_text);

      // 3. Insert fact assertions and evidence.
      std::vector<long long> evidence_memory_ids;
      if (summary_memory_id > 0)
        {
          evidence_memory_ids.push_back (summary_memory_id);
          auto evidence_rows = tx.Execute (
              "SELECT target_memory_id FROM associations "
              "WHERE source_memory_id = ? AND edge_type = 'derived_from'",
              { summary_memory_id });
          for (const auto &row : evidence_rows)
            {
              const long long source_memory_id
                  = ExtractInt64Field (row, "target_memory_id");
              if (source_memory_id > 0)
                {
                  evidence_memory_ids.push_back (source_memory_id);
                }
            }
        }

      std::vector<ExtractedFact> facts_to_process;
      if (!FactWritesDisabled ())
        {
          facts_to_process = result.facts;
          for (const auto &label : refined_label_texts)
            {
              auto event_fact = BuildFactFromDurableEventLabel (
                  label, cfg.focus, cfg.sensitivity, cfg.stability);
              if (!event_fact.has_value ())
                {
                  continue;
                }
              const std::string event_subject
                  = store::NormalizeFactTerm (event_fact->subject);
              const std::string event_predicate
                  = store::NormalizeFactPredicate (event_fact->predicate);
              const std::string event_object
                  = store::NormalizeFactTerm (event_fact->object);
              const bool duplicate = std::any_of (
                  facts_to_process.begin (), facts_to_process.end (),
                  [&] (const ExtractedFact &existing) {
                    return store::NormalizeFactTerm (existing.subject)
                               == event_subject
                           && store::NormalizeFactPredicate (existing.predicate)
                                  == event_predicate
                           && store::NormalizeFactTerm (existing.object)
                                  == event_object;
                  });
              if (!duplicate)
                {
                  facts_to_process.push_back (std::move (*event_fact));
                }
	            }
	        }

      // Fact admission is derived, never list-based: a durable assertion
      // needs a subject that references something (not pure closed-class
      // grammar), a predicate that is an actual assertion (the six relation
      // predicates are mis-slotted relations, already handled by the
      // relation path), and endpoints that are not the same phrase.
      static const std::unordered_set<std::string_view>
          kRelationOnlyPredicates = { "co_occurs",   "implies", "contradicts",
                                      "reinforces",  "causes",  "similar_to" };
      auto fact_token_set = [] (const std::string &text) {
        std::unordered_set<std::string> tokens;
        const std::string canonical
            = CanonicalLabelTokenKey (NormalizeLabelKey (text));
        std::string token;
        for (unsigned char c : canonical)
          {
            if (std::isalnum (c) != 0)
              {
                token.push_back (static_cast<char> (c));
              }
            else if (!token.empty ())
              {
                tokens.insert (token);
                token.clear ();
              }
          }
        if (!token.empty ())
          {
            tokens.insert (token);
          }
        return tokens;
      };
      auto reject_fact = [] (const std::string &subject,
                             const std::string &predicate,
                             const std::string &object,
                             const char *reason) {
        telemetry::LogDebug (
            "cortext.fact_admission",
            { telemetry::Attribute::String ("subject", subject),
              telemetry::Attribute::String ("predicate", predicate),
              telemetry::Attribute::String ("object", object),
              telemetry::Attribute::String ("decision", reason) });
      };
      const bool evidence_confidence_enabled
          = EnvBool ("CORTEXT_ENABLE_EVIDENCE_CONFIDENCE");
      const auto evidence_confidence_options = EvidenceConfidenceOptions (
          cfg.focus, cfg.sensitivity, cfg.stability);

      for (const auto &fact : facts_to_process)
        {
          const std::string canonical_subject
              = store::NormalizeFactTerm (fact.subject);
          const std::string canonical_predicate
              = store::NormalizeFactPredicate (fact.predicate);
          const std::string canonical_object
              = store::NormalizeFactTerm (fact.object);
          if (canonical_subject.empty () || canonical_predicate.empty ()
              || canonical_object.empty () || summary_memory_id <= 0)
            {
              continue;
            }
          if (kRelationOnlyPredicates.find (canonical_predicate)
              != kRelationOnlyPredicates.end ())
            {
              reject_fact (fact.subject, fact.predicate, fact.object,
                           "rejected_relation_predicate");
              continue;
            }
          if (IsAllClosedClassTokens (
                  CanonicalLabelTokenKey (NormalizeLabelKey (fact.subject))))
            {
              reject_fact (fact.subject, fact.predicate, fact.object,
                           "rejected_closed_class_subject");
              continue;
            }
          const auto subject_tokens = fact_token_set (fact.subject);
          const auto object_tokens = fact_token_set (fact.object);
          const bool subject_covers_object = std::all_of (
              object_tokens.begin (), object_tokens.end (),
              [&] (const std::string &t) {
                return subject_tokens.count (t) > 0;
              });
          const bool object_covers_subject = std::all_of (
              subject_tokens.begin (), subject_tokens.end (),
              [&] (const std::string &t) {
                return object_tokens.count (t) > 0;
              });
          if (subject_covers_object || object_covers_subject)
            {
              reject_fact (fact.subject, fact.predicate, fact.object,
                           "rejected_self_loop");
              continue;
            }

          const std::optional<long long> valid_start_ts
              = fact.valid_start_ts.has_value ()
                    ? OptionalI64 (*fact.valid_start_ts)
                    : (summary_start_ts > 0
                           ? std::optional<long long> (summary_start_ts)
                           : std::nullopt);
          const std::optional<long long> valid_end_ts
              = fact.valid_end_ts.has_value ()
                    ? OptionalI64 (*fact.valid_end_ts)
                    : std::nullopt;
	          const bool explicit_valid_start = fact.valid_start_ts.has_value ();
	          const bool explicit_valid_end = fact.valid_end_ts.has_value ();
	          const double confidence = core::Clamp (fact.confidence, 0.0, 1.0);
	          const auto incoming_evidence_stamps
	              = evidence_confidence_enabled
	                    ? BuildFactEvidenceStamps (evidence_memory_ids)
	                    : std::vector<evidence::EvidenceStamp> ();

	          long long fact_id = 0;
          const auto duplicate_rows
              = (!explicit_valid_start && !explicit_valid_end)
                    ? tx.Execute (
                          "SELECT fact_id, confidence FROM fact_assertions "
                          "WHERE canonical_subject = ? "
                          "  AND canonical_predicate = ? "
                          "  AND canonical_object = ? "
                          "  AND (superseded_at_ts IS NULL OR superseded_at_ts > ?) "
                          "ORDER BY recorded_at_ts DESC LIMIT 1",
                          { canonical_subject, canonical_predicate,
                            canonical_object,
                            static_cast<long long> (result_ts) })
                    : tx.Execute (
                          "SELECT fact_id, confidence FROM fact_assertions "
                          "WHERE canonical_subject = ? "
                          "  AND canonical_predicate = ? "
                          "  AND canonical_object = ? "
                          "  AND ((valid_start_ts IS NULL AND ? IS NULL) OR valid_start_ts = ?) "
                          "  AND ((valid_end_ts IS NULL AND ? IS NULL) OR valid_end_ts = ?) "
                          "  AND (superseded_at_ts IS NULL OR superseded_at_ts > ?) "
                          "ORDER BY recorded_at_ts DESC LIMIT 1",
                          { canonical_subject, canonical_predicate,
                            canonical_object,
                            NullableAny (valid_start_ts),
                            NullableAny (valid_start_ts),
                            NullableAny (valid_end_ts),
                            NullableAny (valid_end_ts),
                            static_cast<long long> (result_ts) });
	          if (!duplicate_rows.empty ())
	            {
	              fact_id = ExtractInt64Field (duplicate_rows[0], "fact_id");
	              const double existing_confidence
	                  = ExtractDoubleField (duplicate_rows[0], "confidence",
	                                        confidence);
	              double merged_confidence
	                  = std::max (confidence,
	                              existing_confidence);
	              if (evidence_confidence_enabled && fact_id > 0)
	                {
	                  const auto existing_evidence_stamps
	                      = LoadFactEvidenceStamps (tx, fact_id);
	                  const auto existing_truth = evidence::MakeTruth (
	                      1.0, existing_confidence, existing_evidence_stamps, 0.0,
	                      static_cast<long long> (result_ts),
	                      evidence_confidence_options.horizon);
	                  const auto incoming_truth = evidence::MakeTruth (
	                      1.0, confidence, incoming_evidence_stamps, 0.0,
	                      static_cast<long long> (result_ts),
	                      evidence_confidence_options.horizon);
	                  const auto revision = evidence::Revise (
	                      existing_truth, incoming_truth, false,
	                      evidence_confidence_options);
	                  merged_confidence = revision.truth.confidence;
	                  telemetry::LogDebug (
	                      "cortext.fact_evidence_confidence",
	                      { telemetry::Attribute::String (
	                            "decision",
	                            evidence::ToString (revision.decision)),
	                        telemetry::Attribute::Int64 ("fact_id", fact_id),
	                        telemetry::Attribute::Double (
	                            "old_confidence",
	                            revision.old_confidence),
	                        telemetry::Attribute::Double (
	                            "incoming_confidence",
	                            revision.incoming_confidence),
	                        telemetry::Attribute::Double (
	                            "revised_confidence",
	                            revision.revised_confidence),
	                        telemetry::Attribute::Double (
	                            "evidence_weight",
	                            revision.truth.evidence_weight),
	                        telemetry::Attribute::Int64 (
	                            "independent_stamps",
	                            revision.independent_stamp_count),
	                        telemetry::Attribute::Int64 (
	                            "overlapping_stamps",
	                            revision.overlapping_stamp_count),
	                        telemetry::Attribute::Double (
	                            "contradiction_mass",
	                            revision.truth.contradiction_mass) });
	                }
	              if (fact_id > 0)
	                {
                  if (!explicit_valid_start && valid_start_ts.has_value ())
                    {
                      AddWrite (
                          tx,
                          "UPDATE fact_assertions "
                          "SET confidence = ?, "
                          "    confirmation_count = confirmation_count + 1, "
                          "    compressed_support_count = compressed_support_count + 1, "
                          "    last_confirmation_ts = ?, "
                          "    lifecycle_state = 'active', "
                          "    archived_at = NULL, "
                          "    valid_start_ts = CASE "
                          "      WHEN valid_start_ts IS NULL OR valid_start_ts > ? THEN ? "
                          "      ELSE valid_start_ts END "
                          "WHERE fact_id = ?",
                          { merged_confidence,
                            static_cast<long long> (result_ts),
                            *valid_start_ts,
                            *valid_start_ts,
                            fact_id });
                    }
                  else
                    {
                      AddWrite (tx,
                                "UPDATE fact_assertions "
                                "SET confidence = ?, "
                                "    confirmation_count = confirmation_count + 1, "
                                "    compressed_support_count = compressed_support_count + 1, "
                                "    last_confirmation_ts = ?, "
                                "    lifecycle_state = 'active', "
                                "    archived_at = NULL "
                                "WHERE fact_id = ?",
                                { merged_confidence,
                                  static_cast<long long> (result_ts), fact_id });
                    }
                }
            }
          else
            {
              auto conflicting_rows = tx.Execute (
                  "SELECT fact_id, confidence FROM fact_assertions "
                  "WHERE canonical_subject = ? "
                  "  AND canonical_predicate = ? "
                  "  AND canonical_object <> ? "
                  "  AND (superseded_at_ts IS NULL OR superseded_at_ts > ?) "
                  "  AND (valid_end_ts IS NULL OR valid_end_ts > ?)",
                  { canonical_subject, canonical_predicate, canonical_object,
                    static_cast<long long> (result_ts),
                    NullableAny (valid_start_ts) });

	              double strongest_conflicting_confidence = 0.0;
	              for (const auto &row : conflicting_rows)
                {
                  strongest_conflicting_confidence = std::max (
                      strongest_conflicting_confidence,
	                      ExtractDoubleField (row, "confidence", 0.0));
	                }
	              double confidence_for_write = confidence;
	              if (evidence_confidence_enabled
	                  && strongest_conflicting_confidence > 0.0)
	                {
	                  confidence_for_write
	                      = evidence::DampenConfidenceForContradiction (
	                          confidence, strongest_conflicting_confidence,
	                          evidence_confidence_options);
	                  telemetry::LogDebug (
	                      "cortext.fact_evidence_confidence",
	                      { telemetry::Attribute::String (
	                            "decision",
	                            evidence::ToString (
	                                evidence::RevisionDecision::
	                                    DampenedConflict)),
	                        telemetry::Attribute::Int64 ("fact_id", 0),
	                        telemetry::Attribute::Double ("old_confidence",
	                                                      0.0),
	                        telemetry::Attribute::Double (
	                            "incoming_confidence", confidence),
	                        telemetry::Attribute::Double (
	                            "revised_confidence", confidence_for_write),
	                        telemetry::Attribute::Double ("evidence_weight",
	                                                      0.0),
	                        telemetry::Attribute::Int64 (
	                            "independent_stamps",
	                            static_cast<int64_t> (
	                                incoming_evidence_stamps.size ())),
	                        telemetry::Attribute::Int64 ("overlapping_stamps",
	                                                     0),
	                        telemetry::Attribute::Double (
	                            "contradiction_mass",
	                            strongest_conflicting_confidence) });
	                }
	              const bool accept_conflicting_update
	                  = conflicting_rows.empty ()
	                    || confidence
                           >= RequiredSupersessionConfidence (
                               strongest_conflicting_confidence,
                               cfg.sensitivity, cfg.stability);

              for (const auto &row : conflicting_rows)
                {
                  const long long conflicting_fact_id
                      = ExtractInt64Field (row, "fact_id");
                  if (conflicting_fact_id <= 0)
                    {
                      continue;
                    }
                  if (!accept_conflicting_update)
                    {
                      continue;
                    }
                  AddWrite (
                      tx,
                      "UPDATE fact_assertions "
                      "SET valid_end_ts = CASE "
                      "      WHEN ? IS NULL THEN valid_end_ts "
                      "      WHEN valid_end_ts IS NULL OR valid_end_ts > ? THEN ? "
                      "      ELSE valid_end_ts END, "
                      "    superseded_at_ts = ?, "
                      "    challenge_count = challenge_count + 1, "
                      "    last_challenge_ts = ? "
                      "WHERE fact_id = ?",
                      { NullableAny (valid_start_ts), NullableAny (valid_start_ts),
                        NullableAny (valid_start_ts),
                        static_cast<long long> (result_ts),
                        static_cast<long long> (result_ts), conflicting_fact_id });
                  touched_fact_ids.push_back (conflicting_fact_id);
                  store::RefreshFactCache (tx, encoder, conflicting_fact_id,
                                           result_ts);
                }

              AddWrite (tx,
                        "INSERT INTO fact_assertions "
                        "(subject, predicate, object, canonical_subject, "
                        " canonical_predicate, canonical_object, valid_start_ts, "
                        " valid_end_ts, recorded_at_ts, superseded_at_ts, "
                        " confidence, summary_memory_id, created_at, "
                        " support_mass, source_diversity, contradiction_mass, "
                        " confirmation_count, challenge_count, "
                        " compressed_support_count, last_confirmation_ts, "
                        " last_challenge_ts, severity_class, lifecycle_state, "
                        " archived_at, last_maintenance_ts) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                        "        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        { fact.subject, fact.predicate, fact.object,
                          canonical_subject, canonical_predicate, canonical_object,
                          NullableAny (valid_start_ts),
                          NullableAny (valid_end_ts),
                          static_cast<long long> (result_ts),
                          accept_conflicting_update ? std::any ()
                                                    : std::any (
                                                          static_cast<long long> (
                                                              result_ts)),
	                          confidence_for_write, summary_memory_id,
                          static_cast<long long> (result_ts),
                          0.0,
                          0LL,
                          strongest_conflicting_confidence,
                          1LL,
                          0LL,
                          0LL,
                          static_cast<long long> (result_ts),
                          std::any (),
                          store::PredicateSeverityClass (canonical_predicate),
                          accept_conflicting_update
                              ? std::any (std::string ("active"))
                              : std::any (std::string ("archived")),
                          accept_conflicting_update ? std::any ()
                                                    : std::any (
                                                          static_cast<long long> (
                                                              result_ts)),
                          0LL });
              auto fact_rows
                  = tx.Execute ("SELECT last_insert_rowid() AS id", {});
              if (!fact_rows.empty () && fact_rows[0].count ("id"))
                {
                  fact_id = ExtractInt64Field (fact_rows[0], "id");
                }
            }

          if (fact_id <= 0)
            {
              continue;
            }
          touched_fact_ids.push_back (fact_id);
          ++facts_touched_for_result;

          for (size_t i = 0; i < evidence_memory_ids.size (); ++i)
            {
              const long long evidence_memory_id = evidence_memory_ids[i];
              if (evidence_memory_id <= 0)
                {
                  continue;
                }
	              const std::string evidence_type
	                  = i == 0 ? "summary" : "episodic";
	              const double support_weight
	                  = core::FactEvidenceWriteSupportWeight (
	                      cfg.focus, cfg.sensitivity, cfg.stability,
	                      evidence_type.c_str ());
              AddWrite (tx,
                        "INSERT OR IGNORE INTO fact_evidence "
                        "(fact_id, source_memory_id, evidence_type, support_weight) "
                        "VALUES (?, ?, ?, ?)",
                        { fact_id, evidence_memory_id, evidence_type,
                          support_weight });
            }

          store::RefreshFactCache (tx, encoder, fact_id, result_ts);
        }
      AuditProcessedRelabelResult (
          tx, result.summary_id, summary_memory_id, refined_label_texts,
          removed_label_keys, kept_label_count, added_label_count,
          relation_count, relation_edges_created,
          label_cooccurrence_edges_created,
          relation_edges_skipped_non_durable_endpoint,
          relation_edges_skipped_missing_endpoint,
          relation_edges_skipped_unsupported_predicate,
          relation_endpoint_direct_hits, relation_endpoint_repair_hits,
          relation_endpoint_created_labels,
          relation_endpoint_relation_backed_labels,
          relation_endpoint_rejected_count,
          relation_endpoint_rejected_non_durable,
          relation_endpoint_rejected_ungrounded,
          current_labels_in_selected_evidence,
          current_labels_in_full_source,
          removed_labels_in_selected_evidence,
          removed_labels_in_full_source,
          refined_labels_in_selected_evidence,
          refined_labels_in_full_source,
          static_cast<long long> (result.labels.size ()),
          static_cast<long long> (result.relations.size ()),
          request_source_span_counts[result.summary_id],
          label_candidates_rejected_non_durable,
          label_candidates_rejected_ungrounded,
          label_candidates_rejected_low_contrast,
          label_candidates_rejected_duplicate,
          label_candidates_rejected_legacy_gate,
          labels_inserted_from_extractor,
          labels_inserted_from_current_floor,
          labels_inserted_from_source_span_floor,
          labels_inserted_from_relation_endpoint,
          facts_touched_for_result);
    }

  if (!touched_fact_ids.empty ())
    {
      auto lifecycle_options = store::GetFactLifecycleOptions ();
      lifecycle_options.focus = cfg.focus;
      lifecycle_options.sensitivity = cfg.sensitivity;
      lifecycle_options.stability = cfg.stability;
      store::ScopedFactLifecycleOptions lifecycle_options_guard (
          lifecycle_options);
      store::MaintainFactLifecycle (
          tx, lifecycle_encoder,
          fact_maintenance_ts > 0 ? fact_maintenance_ts : now_ts,
          touched_fact_ids);
    }

  // 4. Clear pending results.
  p_ctx.pending_extraction_results.clear ();

  telemetry::LogInfo ("cortext.process_extraction_results", {
    telemetry::Attribute::Int64 ("results_processed", total_results),
    telemetry::Attribute::Int64 ("labels_seen", total_labels),
    telemetry::Attribute::Int64 ("relations_seen", total_relations),
    telemetry::Attribute::Int64 ("facts_seen", total_facts),
    telemetry::Attribute::Int64 ("label_threshold", label_threshold)
  });
}

} // namespace cortext::operations
