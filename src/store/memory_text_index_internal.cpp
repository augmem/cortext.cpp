#include "memory_text_index_internal.hpp"

#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iterator>
#include <string_view>
#include <unordered_set>

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

std::string
BuildMatchQuery (std::string_view text, std::size_t max_terms)
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
      std::string word (text.substr (start, i - start));
      std::string lower = word;
      std::transform (lower.begin (), lower.end (), lower.begin (),
                      [] (unsigned char c) {
                        return static_cast<char> (
                            c < 0x80 ? std::tolower (c) : c);
                      });
      if (IsStopword (lower) || !seen.insert (lower).second)
        continue;
      words.push_back (std::move (lower));
    }
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

std::vector<Hit>
Search (Transaction &tx, std::string_view query_text, std::size_t limit,
        std::size_t max_terms)
{
  std::vector<Hit> hits;
  const std::string match = BuildMatchQuery (query_text, max_terms);
  if (match.empty () || limit == 0)
    return hits;
  const auto rows = tx.Execute (
      "SELECT rowid AS memory_id, rank AS score FROM memory_text_fts "
      "WHERE memory_text_fts MATCH ? ORDER BY rank LIMIT ?",
      { match, static_cast<long long> (limit) });
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
  const auto pending = store.Execute (
      "SELECT 1 FROM cortext_pending_backfills "
      "WHERE name = 'memory_text_fts' LIMIT 1");
  if (pending.empty ())
    return;
  const auto started = std::chrono::steady_clock::now ();
  auto tx = store.Begin ();
  // An in-database object store reads through the SQL transaction itself.
  std::unique_ptr<ObjectTransaction> object_tx;
  if (object_store && dynamic_cast<SqlObjectStore *> (object_store) == nullptr)
    object_tx = object_store->Begin ();
  std::size_t indexed = 0;
  try
    {
      const auto rows = tx->Execute (
          "SELECT m.memory_id, m.blob_id FROM memories m "
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
          IndexMemoryText (
              *tx, memory_id,
              std::string_view (
                  reinterpret_cast<const char *> (payload->data ()),
                  payload->size ()));
          ++indexed;
        }
      tx->Execute ("DELETE FROM cortext_pending_backfills "
                   "WHERE name = 'memory_text_fts'",
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
        telemetry::Attribute::Double (
            "ms", std::chrono::duration<double, std::milli> (
                      std::chrono::steady_clock::now () - started)
                      .count ()) });
}

} // namespace cortext::store::memory_text_index_internal
