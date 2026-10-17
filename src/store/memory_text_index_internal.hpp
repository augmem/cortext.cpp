#pragma once

// Keyword search over LONG_TERM memory text, backed by the SQLite FTS5 table
// memory_text_fts (schema migration 49). The table is contentless: memory
// text stays in the object store, the FTS5 index holds only the inverted
// lists, keyed by rowid = memory_id. Rows are written in the same savepoint
// as the memory itself, so they commit and roll back with it.

#include "cortext/store/object_store.hpp"
#include "cortext/store/store.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace cortext::store::memory_text_index_internal
{

struct Hit
{
  long long memory_id = 0;
  // FTS5 bm25(): lower is a better match.
  double bm25 = 0.0;
};

/// Builds an FTS5 MATCH expression that ORs the query's words (each one a
/// quoted string, so query text never parses as FTS5 syntax). English
/// stopwords are dropped; at most max_terms words are kept. Returns an empty
/// string when no searchable word remains.
std::string BuildMatchQuery (std::string_view text, std::size_t max_terms);

/// Best-first FTS5 hits for the query text.
std::vector<Hit> Search (Transaction &tx, std::string_view query_text,
                         std::size_t limit, std::size_t max_terms);

/// Indexes a stored memory's text (call inside the memory's savepoint).
void IndexMemoryText (Transaction &tx, long long memory_id,
                      std::string_view text);

/// Indexes LONG_TERM text memories written before migration 49 created the
/// table, once: migration 49 queues the backfill and this clears it.
void RunPendingBackfill (Store &store, ObjectStore *object_store);

} // namespace cortext::store::memory_text_index_internal
