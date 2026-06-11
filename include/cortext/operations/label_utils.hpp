#pragma once

#include <cctype>
#include <string_view>
#include <string>
#include <unordered_set>

namespace cortext::operations
{

inline std::string
TrimLabel (const std::string &label)
{
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

  static const std::unordered_set<std::string_view> kFillerLabels = {
	    "a",         "about",     "ah",        "aha",       "alright",
	    "also",      "an",        "and",       "anyway",    "are",
	    "as",        "assistant", "at",        "basically", "be",
	    "almost",    "almost done", "been",    "being",     "big cart",
	    "all",       "but",
	    "by",        "can",       "could",     "cool",      "did",
	    "do",        "does",
	    "doing",     "done",      "equate",    "er",        "erm",       "for",
	    "food",      "from",      "get",       "gets",      "getting",   "go",
	    "goes",      "going",     "got",       "had",       "has",
	    "good",      "good luck", "have",      "having",    "he",        "her",       "hers",
	    "him",       "his",       "hmm",       "huh",       "i",
	    "idea",      "ideas",     "image",     "in",        "is",        "it",
	    "its",       "just",
	    "k",         "kind of",   "know",      "like",      "make",
	    "makes",     "making",    "maybe",     "me",        "might",
	    "mm",        "mm hmm",    "mhm",       "my",        "no",        "nope",
	    "of",        "off brand", "oh",
	    "ok",        "okay",      "on",        "once",      "or",        "our",
	    "ours",      "probably",  "right",     "she",       "sorry",     "so",
	    "sort of",   "stuff",     "sure",      "system",    "that",
	    "the",       "their",
	    "theirs",    "them",      "then",      "there",     "these",
	    "they",      "thing",     "things",    "this",      "those",
	    "to",        "thanks",    "thank you", "uh",        "uh huh",
	    "um",        "umm",       "up",        "us",        "user",
	    "we",        "well",      "what",      "when",      "where",
	    "which",     "who",       "why",       "with",      "would",
	    "ya",        "ya know",   "yeah",      "yep",       "you",
	    "your",      "yours",     "you know",
	    "agree",     "almost",    "back",      "back row",  "backs",     "bathroom",
	    "cage",      "cart",      "come back", "cute",      "damn",      "damn passed",
	    "how",       "including", "lighting",  "long",      "long line",
	    "light",     "metal",     "not",       "not ideal", "shade",     "wall",
	    "way",       "wild"
	  };

  if (kFillerLabels.find (canonical) != kFillerLabels.end ())
    {
      return false;
    }

  // A phrase composed entirely of these tokens carries no durable content.
  // This is the compositional complement of kFillerLabels above: the filler
  // list can only reject exact phrases it has already seen ("okay" and
  // "thanks" were listed, "okay thanks" sailed through), while this rule
  // rejects any combination of weak and conversational-acknowledgment
  // tokens.
  static const std::unordered_set<std::string_view> kWeakPhraseTokens = {
    "a",       "about",   "all",     "almost", "and",    "around",
    "as",      "at",      "back",    "be",     "been",   "being",
    "can",     "come",    "could",   "did",    "do",     "does",
    "doing",   "done",    "for",     "from",   "get",    "gets",
    "getting", "go",      "goes",    "going",  "good",   "got",
    "guess",   "had",     "has",     "have",   "having", "how",
    "if",      "in",      "is",      "it",     "just",   "kind",
    "like",    "make",    "makes",   "making", "maybe",  "might",
    "not",     "of",      "on",      "once",   "or",     "probably",
    "right",   "so",      "sort",    "that",   "the",    "then",
    "there",   "thing",   "things",  "this",   "to",     "turn",
    "turning", "up",      "was",     "way",    "well",   "what",
    "when",    "where",   "which",   "who",    "why",    "with",
    "would",
    // Conversational acknowledgments and discourse filler.
    "ah",      "aha",     "alright", "bye",    "cool",   "fine",
    "great",   "hello",   "hey",     "hi",     "hmm",    "huh",
    "know",    "mhm",     "mm",      "nah",    "nice",   "no",
    "nope",    "oh",      "ok",      "okay",   "oops",   "please",
    "sorry",   "sounds",  "sure",    "thank",  "thanks", "uh",
    "um",      "umm",     "welcome", "wow",    "ya",     "yeah",
    "yep",     "yes",     "you",     "your"
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
    if (kWeakPhraseTokens.find (phrase_token) != kWeakPhraseTokens.end ())
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
  if (phrase_token_count >= 2
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
