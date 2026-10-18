#include "memory_text_index_internal.hpp"

#include "../operations/standing_memory_internal.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iterator>
#include <map>
#include <string>
#include <string_view>
#include <any>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <utility>
#include <vector>

namespace cortext::store::memory_text_index_internal
{

namespace
{

constexpr std::string_view kStopwords[] = {
  "a",       "about",   "above",   "after",    "again",   "against", "all",
  "am",      "an",      "and",     "any",      "are",     "as",      "at",
  "be",      "because", "been",    "before",   "being",   "below",   "between",
  "both",    "but",     "by",      "can",      "could",   "did",     "do",
  "does",    "doing",   "down",    "during",   "each",    "few",     "for",
  "from",    "further", "had",     "has",      "have",    "having",  "he",
  "her",     "here",    "hers",    "herself",  "him",     "himself", "his",
  "how",     "i",       "if",      "in",       "into",    "is",      "it",
  "its",     "itself",  "just",    "me",       "more",    "most",    "my",
  "myself",  "no",      "nor",     "not",      "now",     "of",      "off",
  "on",      "once",    "only",    "or",       "other",   "our",     "ours",
  "out",     "over",    "own",     "same",     "she",     "should",  "so",
  "some",    "such",    "than",    "that",     "the",     "their",   "theirs",
  "them",    "then",    "there",   "these",    "they",    "this",    "those",
  "through", "to",      "too",     "under",    "until",   "up",      "very",
  "was",     "we",      "were",    "what",     "when",    "where",   "which",
  "while",   "who",     "whom",    "why",      "will",    "with",    "would",
  "you",     "your",    "yours",   "yourself", "ll",      "re",      "ve",
  "don",     "s",       "t",       "d",        "m",
};

bool
IsStopword (const std::string &lower)
{
  static const std::unordered_set<std::string_view> set (
      std::begin (kStopwords), std::end (kStopwords));
  return set.count (lower) != 0;
}

// ASCII letters/digits and every non-ASCII byte form words; FTS5's
// unicode61 tokenizer splits and folds the quoted word further.
bool
IsWordByte (unsigned char c)
{
  return c >= 0x80 || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
         || (c >= '0' && c <= '9');
}

} // namespace

std::vector<std::string>
QueryWords (std::string_view text, std::size_t max_terms)
{
  std::vector<std::string> words;
  std::unordered_set<std::string> seen;
  std::size_t i = 0;
  while (i < text.size () && words.size () < max_terms)
    {
      while (i < text.size ()
             && !IsWordByte (static_cast<unsigned char> (text[i])))
        ++i;
      const std::size_t start = i;
      while (i < text.size ()
             && IsWordByte (static_cast<unsigned char> (text[i])))
        ++i;
      if (i == start)
        continue;
      std::string lower (text.substr (start, i - start));
      std::transform (lower.begin (), lower.end (), lower.begin (),
                      [] (unsigned char c) {
                        return static_cast<char> (
                            c < 0x80 ? std::tolower (c) : c);
                      });
      if (IsStopword (lower) || !seen.insert (lower).second)
        continue;
      words.push_back (std::move (lower));
    }
  return words;
}

std::string
BuildMatchQuery (const std::vector<std::string> &words)
{
  std::string query;
  for (const auto &word : words)
    {
      if (!query.empty ())
        query += " OR ";
      query += '"';
      query += word; // word bytes never include '"'
      query += '"';
    }
  return query;
}

namespace
{

// Drops words whose index term is too common, using the connection-local
// helper tables created with the schema (see ApplyMigrations). Without them
// the words are kept unfiltered.
std::vector<std::string>
DropCommonWords (Transaction &tx, std::vector<std::string> words,
                 std::size_t max_df)
{
  if (words.empty ())
    return words;
  try
    {
      tx.Execute ("INSERT INTO memory_query_terms(memory_query_terms) "
                  "VALUES ('delete-all')",
                  {});
      for (std::size_t k = 0; k < words.size (); ++k)
        tx.Execute ("INSERT INTO memory_query_terms(rowid, text) "
                    "VALUES (?, ?)",
                    { static_cast<long long> (k + 1), words[k] });
      // word index -> index terms of that word (unicode61 may split one).
      std::vector<std::vector<std::string>> terms (words.size ());
      std::vector<std::any> params;
      std::string placeholders;
      for (const auto &row : tx.Execute (
               "SELECT doc, term FROM memory_query_term_stems", {}))
        {
          const auto doc = AnyToLongLong (row.at ("doc")).value_or (0);
          if (doc <= 0 || doc > static_cast<long long> (words.size ()))
            continue;
          auto term = std::any_cast<std::string> (row.at ("term"));
          terms[static_cast<std::size_t> (doc - 1)].push_back (term);
          if (!placeholders.empty ())
            placeholders += ", ";
          placeholders += "?";
          params.emplace_back (std::move (term));
        }
      std::unordered_map<std::string, long long> df;
      if (!params.empty ())
        for (const auto &row : tx.Execute (
                 "SELECT term, doc FROM memory_text_vocab WHERE term IN ("
                     + placeholders + ")",
                 params))
          df[std::any_cast<std::string> (row.at ("term"))]
              = AnyToLongLong (row.at ("doc")).value_or (0);
      std::vector<std::string> kept;
      for (std::size_t k = 0; k < words.size (); ++k)
        {
          bool useful = false;
          for (const auto &term : terms[k])
            {
              const auto it = df.find (term);
              const long long n = it == df.end () ? 0 : it->second;
              useful = useful
                       || (n > 0 && n <= static_cast<long long> (max_df));
            }
          if (useful)
            kept.push_back (std::move (words[k]));
        }
      return kept;
    }
  catch (const std::exception &e)
    {
      telemetry::LogDebug (
          "cortext.memory_text_fts.term_filter_unavailable",
          { telemetry::Attribute::String ("error", e.what ()) });
      return words;
    }
}

} // namespace

namespace
{

std::string
MatchFor (Transaction &tx, std::string_view query_text, std::size_t documents,
          const SearchParams &params)
{
  const auto max_df = std::max (
      params.max_df_floor,
      static_cast<std::size_t> (params.max_df_ratio
                                * static_cast<double> (documents)));
  return BuildMatchQuery (
      DropCommonWords (tx, QueryWords (query_text, params.max_terms), max_df));
}

std::vector<Hit>
HitsOf (const std::vector<std::map<std::string, std::any> > &rows)
{
  std::vector<Hit> hits;
  hits.reserve (rows.size ());
  for (const auto &row : rows)
    {
      const auto memory_id
          = AnyToLongLong (row.at ("memory_id")).value_or (0);
      if (memory_id > 0)
        hits.push_back ({ memory_id, AnyToDouble (row.at ("score"), 0.0) });
    }
  return hits;
}

} // namespace

std::vector<Hit>
Search (Transaction &tx, std::string_view query_text, std::size_t documents,
        const SearchParams &params)
{
  if (params.limit == 0)
    return {};
  const std::string match = MatchFor (tx, query_text, documents, params);
  if (match.empty ())
    return {};
  return HitsOf (tx.Execute (
      "SELECT rowid AS memory_id, rank AS score FROM memory_text_fts "
      "WHERE memory_text_fts MATCH ? ORDER BY rank LIMIT ?",
      { match, static_cast<long long> (params.limit) }));
}

void
IndexMemoryText (Transaction &tx, long long memory_id, std::string_view text)
{
  if (memory_id <= 0 || text.empty ())
    return;
  tx.Execute ("INSERT INTO memory_text_fts(rowid, text) VALUES (?, ?)",
              { memory_id, std::string (text) });
}

void
RunPendingBackfill (Store &store, ObjectStore *object_store)
{
  namespace standing = operations::standing_memory_internal;
  bool fts_pending = false;
  bool standing_pending = false;
  for (const auto &row : store.Execute (
           "SELECT name FROM cortext_pending_backfills "
           "WHERE name IN ('memory_text_fts', 'standing_memories')"))
    {
      const auto &value = row.at ("name");
      const std::string name = value.type () == typeid (std::string)
                                   ? std::any_cast<std::string> (value)
                                   : std::string ();
      fts_pending = fts_pending || name == "memory_text_fts";
      standing_pending = standing_pending || name == "standing_memories";
    }
  if (!fts_pending && !standing_pending)
    return;
  const auto started = std::chrono::steady_clock::now ();
  auto tx = store.Begin ();
  // An in-database object store reads through the SQL transaction itself.
  std::unique_ptr<ObjectTransaction> object_tx;
  if (object_store && dynamic_cast<SqlObjectStore *> (object_store) == nullptr)
    object_tx = object_store->Begin ();
  std::size_t indexed = 0;
  // start_ts, id, terms
  std::vector<std::tuple<long long, long long, std::string> > directives;
  try
    {
      const auto rows = tx->Execute (
          "SELECT m.memory_id, m.start_ts, m.blob_id FROM memories m "
          "WHERE m.kind = 'LONG_TERM' AND m.modality = 'text' "
          "  AND m.blob_id IS NOT NULL",
          {});
      for (const auto &row : rows)
        {
          const auto memory_id
              = AnyToLongLong (row.at ("memory_id")).value_or (0);
          const auto blob_id = BlobFromAny (row.at ("blob_id"));
          if (memory_id <= 0 || blob_id.empty ())
            continue;
          const auto payload = GetObject (object_tx.get (), *tx, blob_id);
          if (!payload || payload->empty ())
            continue;
          const std::string_view text (
              reinterpret_cast<const char *> (payload->data ()),
              payload->size ());
          if (fts_pending)
            {
              IndexMemoryText (*tx, memory_id, text);
              ++indexed;
            }
          if (standing_pending && standing::IsStandingDirective (text))
            directives.emplace_back (
                AnyToLongLong (row.at ("start_ts")).value_or (0), memory_id,
                standing::TermsOf (text));
        }
      // The newest members fill the bounded set.
      std::sort (directives.rbegin (), directives.rend ());
      if (directives.size () > standing::kCapacity)
        directives.resize (standing::kCapacity);
      for (const auto &[start_ts, memory_id, terms] : directives)
        tx->Execute ("INSERT OR IGNORE INTO standing_memories "
                     "(memory_id, start_ts, terms) VALUES (?, ?, ?)",
                     { memory_id, start_ts, terms });
      tx->Execute ("DELETE FROM cortext_pending_backfills "
                   "WHERE name IN ('memory_text_fts', 'standing_memories')",
                   {});
      if (object_tx)
        object_tx->Commit ();
      tx->Commit ();
    }
  catch (const std::exception &e)
    {
      if (object_tx)
        object_tx->Rollback ();
      tx->Rollback ();
      telemetry::LogWarn (
          "cortext.memory_text_fts.backfill_failed",
          { telemetry::Attribute::String ("error", e.what ()) });
      return;
    }
  telemetry::LogDebug (
      "cortext.memory_text_fts.backfilled",
      { telemetry::Attribute::Int64 ("memories",
                                     static_cast<std::int64_t> (indexed)),
        telemetry::Attribute::Int64 (
            "standing", static_cast<std::int64_t> (directives.size ())),
        telemetry::Attribute::Double (
            "ms", std::chrono::duration<double, std::milli> (
                      std::chrono::steady_clock::now () - started)
                      .count ()) });
}

} // namespace cortext::store::memory_text_index_internal
