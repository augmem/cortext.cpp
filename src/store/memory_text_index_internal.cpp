#include "memory_text_index_internal.hpp"

#include "../operations/standing_memory_internal.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include <sqlite3.h>

#include "../../third_party/sqlite/ext/fts5/fts5.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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
      // Index terms of each scratch row (rowid = 1-based slot).
      const auto stems_of = [&tx] (const std::vector<std::string> &texts) {
        tx.Execute ("INSERT INTO memory_query_terms(memory_query_terms) "
                    "VALUES ('delete-all')",
                    {});
        for (std::size_t k = 0; k < texts.size (); ++k)
          tx.Execute ("INSERT INTO memory_query_terms(rowid, text) "
                      "VALUES (?, ?)",
                      { static_cast<long long> (k + 1), texts[k] });
        std::vector<std::vector<std::string>> stems (texts.size ());
        for (const auto &row : tx.Execute (
                 "SELECT doc, term FROM memory_query_term_stems", {}))
          {
            const auto doc = AnyToLongLong (row.at ("doc")).value_or (0);
            if (doc <= 0 || doc > static_cast<long long> (texts.size ()))
              continue;
            stems[static_cast<std::size_t> (doc - 1)].push_back (
                std::any_cast<std::string> (row.at ("term")));
          }
        return stems;
      };
      // word index -> index terms of that word (unicode61 may split one).
      const auto terms = stems_of (words);
      std::vector<std::string> distinct;
      std::unordered_set<std::string> seen;
      for (const auto &word_terms : terms)
        for (const auto &term : word_terms)
          if (seen.insert (term).second)
            distinct.push_back (term);
      // Only "df <= max_df" matters, so a term whose quoted form tokenizes
      // back to itself is counted through its posting list, stopping after
      // max_df + 1 documents: the cost stays bounded by max_df however
      // common the term is. A term the tokenizer would rewrite again (porter
      // is not idempotent on every stem) keeps the vocabulary's full count.
      const auto restemmed = stems_of (distinct);
      std::unordered_map<std::string, long long> df;
      std::vector<std::any> params;
      std::string placeholders;
      for (std::size_t j = 0; j < distinct.size (); ++j)
        {
          const auto &term = distinct[j];
          if (restemmed[j].size () == 1 && restemmed[j][0] == term)
            {
              const auto rows = tx.Execute (
                  "SELECT count(*) AS n FROM (SELECT rowid FROM "
                  "memory_text_fts WHERE memory_text_fts MATCH ? LIMIT ?)",
                  { '"' + term + '"', static_cast<long long> (max_df) + 1 });
              df[term] = rows.empty ()
                             ? 0
                             : AnyToLongLong (rows[0].at ("n")).value_or (0);
              continue;
            }
          if (!placeholders.empty ())
            placeholders += ", ";
          placeholders += "?";
          params.emplace_back (term);
        }
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

} // namespace

namespace
{

// Top-k BM25 without ranking every match. FTS5's "ORDER BY rank" scores
// each matching row with bm25() (one docsize b-tree read per row) and sorts
// them all, so its cost grows with the store. The collector below computes
// the same score, term for term in bm25()'s order, keeps the best `limit`
// rows in a heap, and skips the document-length read (and the score) of a
// row whose bound with the shortest possible document already falls below
// the current k-th score. Ties keep the lower rowid first, as FTS5's stable
// rank sort over its rowid-ordered scan does.
struct RankedRow
{
  double score = 0.0; // bm25() is -score
  long long rowid = 0;
};

bool
RanksBefore (const RankedRow &a, const RankedRow &b)
{
  return a.score != b.score ? a.score > b.score : a.rowid < b.rowid;
}

struct TopKCollector
{
  std::size_t limit = 0;
  std::vector<RankedRow> heap; // heap.front() is the worst kept row
  std::size_t scored = 0;
  std::size_t matched = 0;
};

thread_local TopKCollector *active_collector = nullptr;

// fts5Bm25GetData's per-query values (fts5_aux.c), computed the same way.
struct Bm25Data
{
  double avgdl = 0.0;
  std::vector<double> idf;
  std::vector<double> freq;
};

int
CountHit (const Fts5ExtensionApi *, Fts5Context *, void *count)
{
  ++*static_cast<sqlite3_int64 *> (count);
  return SQLITE_OK;
}

Bm25Data *
QueryData (const Fts5ExtensionApi *api, Fts5Context *fts, int *rc)
{
  if (auto *data = static_cast<Bm25Data *> (api->xGetAuxdata (fts, 0)))
    return data;
  auto data = std::make_unique<Bm25Data> ();
  const int phrases = api->xPhraseCount (fts);
  data->idf.assign (static_cast<std::size_t> (phrases), 0.0);
  data->freq.assign (static_cast<std::size_t> (phrases), 0.0);
  sqlite3_int64 rows = 0;
  sqlite3_int64 tokens = 0;
  *rc = api->xRowCount (fts, &rows);
  if (*rc == SQLITE_OK)
    *rc = api->xColumnTotalSize (fts, -1, &tokens);
  if (*rc == SQLITE_OK)
    data->avgdl = (double)tokens / (double)rows;
  for (int i = 0; *rc == SQLITE_OK && i < phrases; ++i)
    {
      sqlite3_int64 hits = 0;
      *rc = api->xQueryPhrase (fts, i, &hits, CountHit);
      if (*rc == SQLITE_OK)
        {
          double idf = log ((rows - hits + 0.5) / (hits + 0.5));
          if (idf <= 0.0)
            idf = 1e-6;
          data->idf[static_cast<std::size_t> (i)] = idf;
        }
    }
  if (*rc != SQLITE_OK)
    return nullptr;
  auto *raw = data.release ();
  *rc = api->xSetAuxdata (fts, raw, [] (void *p) {
    delete static_cast<Bm25Data *> (p);
  });
  return *rc == SQLITE_OK ? raw : nullptr;
}

void
CollectTopK (const Fts5ExtensionApi *api, Fts5Context *fts,
             sqlite3_context *ctx, int, sqlite3_value **)
{
  TopKCollector *collector = active_collector;
  if (collector == nullptr || collector->limit == 0)
    {
      sqlite3_result_error (ctx, "cortext_bm25_topk outside Search", -1);
      return;
    }
  int rc = SQLITE_OK;
  Bm25Data *data = QueryData (api, fts, &rc);
  if (data == nullptr)
    {
      sqlite3_result_error_code (ctx, rc);
      return;
    }
  const double k1 = 1.2;
  const double b = 0.75;
  std::fill (data->freq.begin (), data->freq.end (), 0.0);
  int instances = 0;
  rc = api->xInstCount (fts, &instances);
  for (int i = 0; rc == SQLITE_OK && i < instances; ++i)
    {
      int phrase = 0;
      int column = 0;
      int offset = 0;
      rc = api->xInst (fts, i, &phrase, &column, &offset);
      if (rc == SQLITE_OK)
        data->freq[static_cast<std::size_t> (phrase)] += 1.0;
    }
  if (rc != SQLITE_OK)
    {
      sqlite3_result_error_code (ctx, rc);
      return;
    }
  ++collector->matched;
  const std::size_t phrases = data->idf.size ();
  if (collector->heap.size () >= collector->limit)
    {
      // D >= 0, so each term is at most its value at D = 0. The margin
      // covers rounding: only rows clearly below the k-th score are skipped.
      double bound = 0.0;
      for (std::size_t i = 0; i < phrases; ++i)
        bound += data->idf[i]
                 * ((data->freq[i] * (k1 + 1.0))
                    / (data->freq[i] + k1 * (1 - b)));
      if (bound * (1.0 + 1e-9) < collector->heap.front ().score)
        {
          sqlite3_result_null (ctx);
          return;
        }
    }
  int tokens = 0;
  rc = api->xColumnSize (fts, -1, &tokens);
  if (rc != SQLITE_OK)
    {
      sqlite3_result_error_code (ctx, rc);
      return;
    }
  const double D = (double)tokens;
  double score = 0.0;
  for (std::size_t i = 0; i < phrases; ++i)
    score += data->idf[i]
             * ((data->freq[i] * (k1 + 1.0))
                / (data->freq[i] + k1 * (1 - b + b * D / data->avgdl)));
  ++collector->scored;
  const RankedRow row{ score, api->xRowid (fts) };
  auto &heap = collector->heap;
  if (heap.size () < collector->limit)
    {
      heap.push_back (row);
      std::push_heap (heap.begin (), heap.end (), RanksBefore);
    }
  else if (RanksBefore (row, heap.front ()))
    {
      std::pop_heap (heap.begin (), heap.end (), RanksBefore);
      heap.back () = row;
      std::push_heap (heap.begin (), heap.end (), RanksBefore);
    }
  sqlite3_result_null (ctx);
}

} // namespace

bool
RegisterRankingFunction (sqlite3 *db)
{
  fts5_api *api = nullptr;
  sqlite3_stmt *stmt = nullptr;
  if (sqlite3_prepare_v2 (db, "SELECT fts5(?1)", -1, &stmt, nullptr)
      != SQLITE_OK)
    {
      sqlite3_finalize (stmt);
      return false;
    }
  sqlite3_bind_pointer (stmt, 1, static_cast<void *> (&api), "fts5_api_ptr",
                        nullptr);
  (void)sqlite3_step (stmt);
  sqlite3_finalize (stmt);
  return api != nullptr
         && api->xCreateFunction (api, "cortext_bm25_topk", nullptr,
                                  CollectTopK, nullptr)
                == SQLITE_OK;
}

std::vector<Hit>
Search (Transaction &tx, std::string_view query_text, std::size_t documents,
        const SearchParams &params)
{
  if (params.limit == 0)
    return {};
  const std::string match = MatchFor (tx, query_text, documents, params);
  if (match.empty ())
    return {};
  TopKCollector collector;
  collector.limit = params.limit;
  struct Scope
  {
    TopKCollector *previous;
    ~Scope () { active_collector = previous; }
  } scope{ active_collector };
  active_collector = &collector;
  // One result row: the subquery's LIMIT keeps it from being flattened
  // into the aggregate, where FTS5 auxiliary functions cannot run.
  (void)tx.Execute ("SELECT count(x) AS n FROM (SELECT "
                    "cortext_bm25_topk(memory_text_fts) AS x FROM "
                    "memory_text_fts WHERE memory_text_fts MATCH ? LIMIT -1)",
                    { match });
  auto &heap = collector.heap;
  std::sort_heap (heap.begin (), heap.end (), RanksBefore);
  std::vector<Hit> hits;
  hits.reserve (heap.size ());
  for (const auto &row : heap)
    hits.push_back ({ row.rowid, -row.score });
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
  namespace standing = operations::standing_memory_internal;
  bool fts_pending = false;
  bool standing_pending = false;
  for (const auto &row : store.Execute (
           "SELECT name FROM cortext_pending_backfills "
           "WHERE name IN ('memory_text_fts', 'standing_signals')"))
    {
      const auto &value = row.at ("name");
      const std::string name = value.type () == typeid (std::string)
                                   ? std::any_cast<std::string> (value)
                                   : std::string ();
      fts_pending = fts_pending || name == "memory_text_fts";
      standing_pending = standing_pending || name == "standing_signals";
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
  // start_ts, signal id, terms
  std::vector<std::tuple<long long, long long, std::string> > directives;
  try
    {
      if (fts_pending)
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
        }
      if (standing_pending)
        {
          // The detector bounds directive length, so only short signal
          // payloads are read.
          const auto rows = tx->Execute (
              "SELECT s.signal_id, s.timestamp, s.blob_id FROM signals s "
              "JOIN memories m ON m.memory_id = s.memory_id "
              "WHERE m.kind = 'LONG_TERM' AND s.modality = 'text' "
              "  AND s.blob_id IS NOT NULL",
              {});
          for (const auto &row : rows)
            {
              const auto signal_id
                  = AnyToLongLong (row.at ("signal_id")).value_or (0);
              const auto blob_id = BlobFromAny (row.at ("blob_id"));
              if (signal_id <= 0 || blob_id.empty ())
                continue;
              const auto payload = GetObject (object_tx.get (), *tx, blob_id);
              if (!payload || payload->empty ())
                continue;
              const std::string_view text (
                  reinterpret_cast<const char *> (payload->data ()),
                  payload->size ());
              if (standing::IsStandingDirective (text))
                directives.emplace_back (
                    AnyToLongLong (row.at ("timestamp")).value_or (0),
                    signal_id, standing::TermsOf (text));
            }
        }
      // The newest members fill the bounded set.
      std::sort (directives.rbegin (), directives.rend ());
      if (directives.size () > standing::kCapacity)
        directives.resize (standing::kCapacity);
      for (const auto &[start_ts, signal_id, terms] : directives)
        tx->Execute ("INSERT OR IGNORE INTO standing_signals "
                     "(signal_id, start_ts, terms) VALUES (?, ?, ?)",
                     { signal_id, start_ts, terms });
      tx->Execute ("DELETE FROM cortext_pending_backfills "
                   "WHERE name IN ('memory_text_fts', 'standing_signals')",
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
