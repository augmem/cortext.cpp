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
  if (TrimLabel (label).empty () || label_key.empty ())
    {
      return false;
    }

  const std::string canonical = CanonicalLabelTokenKey (label_key);
  if (canonical.empty ())
    {
      return false;
    }

  static const std::unordered_set<std::string_view> kFillerLabels = {
    "ah",        "aha",   "alright", "anyway",   "assistant",
    "basically", "cool",  "er",      "erm",      "hmm",
    "huh",       "just",  "k",       "kind of",  "know",
    "like",      "mm",    "mm hmm",  "mhm",      "nope",
    "oh",        "ok",    "okay",    "right",    "sort of",
    "sure",      "system", "thanks", "thank you", "uh",
    "uh huh",    "um",    "umm",     "user",     "well",
    "ya",        "ya know",
    "yeah",      "yep",   "you know"
  };

  return kFillerLabels.find (canonical) == kFillerLabels.end ();
}

} // namespace cortext::operations
