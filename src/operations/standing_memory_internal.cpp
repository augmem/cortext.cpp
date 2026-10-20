#include "standing_memory_internal.hpp"

#include "../store/memory_text_index_internal.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace cortext::operations::standing_memory_internal
{

namespace
{

// Clause-initial markers, lowercase. Each must end at a word boundary.
constexpr std::string_view kMarkers[] = {
  "always",         "never",          "from now on",   "going forward",
  "in the future",  "in future",      "whenever i",    "every time i",
  "do not ever",    "don't ever",     "i prefer",      "i'd prefer",
  "i would prefer", "my preference is",
};

constexpr std::string_view kPlease = "please";

bool
IsWordByte (unsigned char c)
{
  return std::isalnum (c) != 0 || c == '\'' || c >= 0x80;
}

// Case-insensitive prefix match of word at text[pos], ending at a word
// boundary.
bool
MatchesWordAt (std::string_view text, std::size_t pos, std::string_view word)
{
  if (pos + word.size () > text.size ())
    return false;
  for (std::size_t i = 0; i < word.size (); ++i)
    if (std::tolower (static_cast<unsigned char> (text[pos + i])) != word[i])
      return false;
  const std::size_t end = pos + word.size ();
  return end == text.size ()
         || !IsWordByte (static_cast<unsigned char> (text[end]));
}

std::size_t
SkipSpaces (std::string_view text, std::size_t pos)
{
  while (pos < text.size ()
         && std::isspace (static_cast<unsigned char> (text[pos])) != 0)
    ++pos;
  return pos;
}

bool
ClauseOpensWithMarker (std::string_view text, std::size_t pos)
{
  pos = SkipSpaces (text, pos);
  if (MatchesWordAt (text, pos, kPlease))
    {
      pos = SkipSpaces (text, pos + kPlease.size ());
      if (pos < text.size () && text[pos] == ',')
        pos = SkipSpaces (text, pos + 1);
    }
  for (const auto marker : kMarkers)
    if (MatchesWordAt (text, pos, marker))
      return true;
  return false;
}

} // namespace

bool
IsStandingDirective (std::string_view text)
{
  if (text.empty () || text.size () > kMaxTextBytes)
    return false;
  if (ClauseOpensWithMarker (text, 0))
    return true;
  for (std::size_t i = 0; i < text.size (); ++i)
    {
      const char c = text[i];
      if ((c == ':' || c == '.' || c == '!' || c == '?' || c == ';'
           || c == '\n')
          && ClauseOpensWithMarker (text, i + 1))
        return true;
    }
  return false;
}

namespace
{

std::vector<std::string>
Stems (std::string_view text)
{
  std::vector<std::string> out;
  std::unordered_set<std::string> seen;
  for (auto word : store::memory_text_index_internal::QueryWords (
           text, static_cast<std::size_t> (-1)))
    {
      if (word.size () > kStemBytes)
        word.resize (kStemBytes);
      if (seen.insert (word).second)
        out.push_back (std::move (word));
    }
  return out;
}

} // namespace

std::string
TermsOf (std::string_view text)
{
  std::string out;
  for (const auto &stem : Stems (text))
    {
      if (!out.empty ())
        out.push_back (' ');
      out += stem;
    }
  return out;
}

std::vector<std::pair<long long, double> >
RankByTerms (std::string_view query_text,
             const std::vector<std::pair<long long, std::string> > &members)
{
  std::vector<std::pair<long long, double> > ranked;
  const auto query = Stems (query_text);
  if (query.empty () || members.empty ())
    return ranked;
  std::vector<std::unordered_set<std::string> > docs;
  docs.reserve (members.size ());
  std::unordered_map<std::string, std::size_t> df;
  double total_length = 0.0;
  for (const auto &[memory_id, terms] : members)
    {
      (void)memory_id;
      std::unordered_set<std::string> doc;
      std::istringstream in (terms);
      for (std::string term; in >> term;)
        doc.insert (term);
      for (const auto &term : doc)
        ++df[term];
      total_length += static_cast<double> (doc.size ());
      docs.push_back (std::move (doc));
    }
  const double n = static_cast<double> (members.size ());
  const double average = std::max (1.0, total_length / n);
  constexpr double k1 = 1.2;
  constexpr double b = 0.75;
  for (std::size_t i = 0; i < members.size (); ++i)
    {
      double score = 0.0;
      const double length = static_cast<double> (docs[i].size ());
      for (const auto &term : query)
        {
          if (docs[i].count (term) == 0)
            continue;
          const double f = static_cast<double> (df[term]);
          const double idf = std::log (1.0 + (n - f + 0.5) / (f + 0.5));
          score += idf * (k1 + 1.0) / (1.0 + k1 * (1.0 - b + b * length / average));
        }
      if (score > 0.0)
        ranked.emplace_back (members[i].first, score);
    }
  std::sort (ranked.begin (), ranked.end (), [] (const auto &x, const auto &y) {
    if (x.second != y.second)
      return x.second > y.second;
    return x.first > y.first;
  });
  return ranked;
}

} // namespace cortext::operations::standing_memory_internal
