#pragma once

#include <cctype>
#include <string_view>
#include <string>
#include <unordered_set>

namespace cortext::operations
{

// Smart single quotes (U+2018/U+2019) are apostrophes in chat corpora; the
// byte-wise tokenizers below would otherwise split "aren’t" into "aren" + "t"
// and emit truncated label surfaces.
inline std::string
NormalizeApostrophes (const std::string &text)
{
  std::string out;
  out.reserve (text.size ());
  for (size_t i = 0; i < text.size (); ++i)
    {
      if (i + 2 < text.size ()
          && static_cast<unsigned char> (text[i]) == 0xE2
          && static_cast<unsigned char> (text[i + 1]) == 0x80
          && (static_cast<unsigned char> (text[i + 2]) == 0x98
              || static_cast<unsigned char> (text[i + 2]) == 0x99))
        {
          out.push_back ('\'');
          i += 2;
        }
      else
        {
          out.push_back (text[i]);
        }
    }
  return out;
}

inline std::string
TrimLabel (const std::string &raw_label)
{
  const std::string label = NormalizeApostrophes (raw_label);
  auto start = label.begin ();
  auto end = label.end ();
  while (start != end && std::isspace (static_cast<unsigned char> (*start)))
    {
      ++start;
    }
  while (end != start)
    {
      auto prev = end;
      --prev;
      if (!std::isspace (static_cast<unsigned char> (*prev)))
        {
          break;
        }
      end = prev;
    }
  return std::string (start, end);
}

inline std::string
NormalizeLabelKey (const std::string &label)
{
  std::string trimmed = TrimLabel (label);
  std::string out;
  out.reserve (trimmed.size ());
  for (unsigned char c : trimmed)
    {
      out.push_back (static_cast<char> (std::tolower (c)));
    }
  return out;
}

inline std::string
CanonicalLabelTokenKey (const std::string &label_key)
{
  std::string out;
  out.reserve (label_key.size ());
  bool previous_space = false;
  for (unsigned char c : label_key)
    {
      if (std::isalnum (c) != 0)
        {
          out.push_back (static_cast<char> (c));
          previous_space = false;
        }
      else if (std::isspace (c) != 0 || c == '-' || c == '_'
               || c == '\'')
        {
          if (!out.empty () && !previous_space)
            {
              out.push_back (' ');
              previous_space = true;
            }
        }
    }
  if (!out.empty () && out.back () == ' ')
    {
      out.pop_back ();
    }
  return out;
}

inline bool
IsDurableLabelCandidate (const std::string &label,
                         const std::string &label_key)
{
  const std::string trimmed = TrimLabel (label);
  if (trimmed.empty () || label_key.empty ())
    {
      return false;
    }

  const std::string canonical = CanonicalLabelTokenKey (label_key);
  if (canonical.empty ())
    {
      return false;
    }

  // Closed-class language structure only: function words, pronouns,
  // auxiliaries, interjections, and conversational acknowledgments, plus
  // chat-transcript role artifacts. This set is frozen - it encodes grammar,
  // not corpus tuning. Content-word filtering is the job of corpus-derived
  // signals (label-bank similarity contrast, document frequency), never of
  // this list. See docs: a phrase is rejected here only when EVERY token is
  // closed-class.
  static const std::unordered_set<std::string_view> kClosedClassTokens = {
    // Articles, conjunctions, prepositions, particles
    "a",     "about", "after",  "again",  "all",   "almost", "also",
    "an",    "and",   "any",    "around", "as",    "at",     "back",
    "basically", "before", "but", "by",   "down",  "for",    "from",
    "if",    "in",    "into",   "just",   "kind",  "like",   "maybe",
    "not",   "of",    "off",    "on",     "once",  "only",   "or",
    "out",   "over",  "probably", "so",   "some",  "sort",   "still",
    "than",  "that",  "the",    "then",   "there", "these",  "this",
    "those", "through", "to",   "too",    "under", "up",     "very",
    "well",  "when",  "where",  "which",  "while", "who",    "why",
    "with",  "would",
    // Pronouns and possessives (incl. indefinite pronouns)
    "anybody", "anyone", "anything", "everybody", "everyone", "everything",
    "he",    "her",   "hers",   "him",    "his",   "i",      "it",
    "its",   "me",    "my",     "nobody", "none",  "nothing", "our",
    "ours",  "she",   "somebody", "someone", "something", "their",
    "theirs", "them", "they",   "us",     "we",    "what",   "you",
    "your",  "yours",
    // Auxiliaries, light verbs, and desiderative semi-modals
    "am",    "are",   "be",     "been",   "being", "came",   "can",
    "come",  "could", "did",    "do",     "does",  "doing",  "done",
    "get",   "gets",
    "getting", "go",  "goes",   "going",  "got",   "had",    "has",
    "have",  "having", "is",    "make",   "makes", "making", "might",
    "must",  "need",  "needed", "needing", "needs", "shall", "should",
    "turn",  "turned", "turning", "want", "wanted", "wanting", "wants",
    "was",   "were",  "will",
    // Degree adverbs
    "quite", "rather", "really",
    // Interjections, acknowledgments, discourse markers
    "ah",    "aha",   "alright", "bye",   "cool",  "er",     "erm",
    "fine",  "good",  "great",  "hello",  "hey",   "hi",     "hmm",
    "huh",   "know",  "mhm",    "mm",     "nah",   "nice",   "no",
    "nope",  "oh",    "ok",     "okay",   "oops",  "please", "right",
    "sorry", "sounds", "sure",  "thank",  "thanks", "uh",    "um",
    "umm",   "welcome", "wow",  "ya",     "yeah",  "yep",    "yes",
    // Chat-transcript role artifacts
    "assistant", "system", "user"
  };

  int phrase_token_count = 0;
  int weak_phrase_token_count = 0;
  std::string phrase_token;
  auto finish_phrase_token = [&] {
    if (phrase_token.empty ())
      {
        return;
      }
    ++phrase_token_count;
    if (kClosedClassTokens.find (phrase_token) != kClosedClassTokens.end ())
      {
        ++weak_phrase_token_count;
      }
    phrase_token.clear ();
  };
  for (unsigned char c : canonical)
    {
      if (std::isalnum (c) != 0)
        {
          phrase_token.push_back (static_cast<char> (c));
        }
      else
        {
          finish_phrase_token ();
        }
    }
  finish_phrase_token ();
  // Single closed-class tokens are as empty as all-closed-class phrases.
  if (phrase_token_count >= 1
      && phrase_token_count == weak_phrase_token_count)
    {
      return false;
    }

  int token_count = 0;
  bool in_token = false;
  int alpha_count = 0;
  int digit_count = 0;
  bool has_upper = false;
  bool saw_alpha = false;
  bool first_alpha_upper = false;
  bool has_space = false;
  bool has_connector = false;
  bool first_nonspace_digit = false;
  bool saw_nonspace = false;
  for (unsigned char c : trimmed)
    {
      if (std::isspace (c) == 0 && !saw_nonspace)
        {
          first_nonspace_digit = std::isdigit (c) != 0;
          saw_nonspace = true;
        }
      has_space = has_space || std::isspace (c) != 0;
      has_connector = has_connector || c == '-' || c == '_';
      if (std::isalnum (c) != 0)
        {
          if (!in_token)
            {
              ++token_count;
              in_token = true;
            }
          if (std::isalpha (c) != 0)
            {
              if (!saw_alpha)
                {
                  first_alpha_upper = std::isupper (c) != 0;
                  saw_alpha = true;
                }
              ++alpha_count;
              has_upper = has_upper || std::isupper (c) != 0;
            }
          else if (std::isdigit (c) != 0)
            {
              ++digit_count;
            }
        }
      else
        {
          in_token = false;
        }
    }

  if (alpha_count == 0)
    {
      return false;
    }
  if (token_count == 1 && alpha_count <= 2 && !has_connector)
    {
      bool all_upper = true;
      for (unsigned char c : trimmed)
        {
          if (std::isalpha (c) != 0 && std::isupper (c) == 0)
            {
              all_upper = false;
              break;
            }
        }
      if (!all_upper)
        {
          return false;
        }
    }
  if (token_count <= 1 && digit_count > 0 && canonical.size () >= 6)
    {
      return false;
    }
  if (first_nonspace_digit && token_count <= 2 && !has_upper)
    {
      return false;
    }
  if (token_count <= 1 && (!has_upper || !first_alpha_upper)
      && digit_count == 0)
    {
      return false;
    }
  if (!has_space && has_connector && !first_alpha_upper)
    {
      return false;
    }
  if (!has_space && has_connector && digit_count > 0 && canonical.size () >= 6)
    {
      return false;
    }

  return true;
}

} // namespace cortext::operations
