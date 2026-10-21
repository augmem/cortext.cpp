// tests/schema_baseline.test.cpp
//
// Fresh stores get the baseline schema directly; stores created earlier
// upgrade through the legacy migrations. These tests prove both paths reach
// the same schema: sqlite_schema (SQL compared with whitespace normalized),
// every table's columns, indexes and foreign keys, every index's columns,
// every row of every ordinary table (seed, config and virtual-table shadow
// rows), and the version pragmas.
#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <algorithm>
#include <any>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace cortext;

namespace cortext::store
{
std::set<int64_t> DebugGetAppliedMigrationIdsForTest (Store &store);
void DebugApplyCoreMigrationsThroughForTest (Store &store, int64_t maximum_id);
void DebugApplyLegacyMigrationChainThroughForTest (Store &store,
                                                   int64_t maximum_id);
std::vector<std::string> DebugBaselineStatementsForTest ();
} // namespace cortext::store

namespace
{

constexpr int64_t kBaselineVersion = 58;
// Every migration: the legacy chain, then any added after the baseline.
constexpr int64_t kAllMigrations = std::numeric_limits<int64_t>::max ();

// Collapses whitespace runs to one space and drops whitespace next to
// punctuation and at the ends. Nothing else about the SQL is changed.
std::string
NormalizeWhitespace (const std::string &sql)
{
  std::string collapsed;
  bool in_space = false;
  for (const char c : sql)
    {
      if (std::isspace (static_cast<unsigned char> (c)))
        {
          in_space = true;
          continue;
        }
      if (in_space && !collapsed.empty ())
        collapsed.push_back (' ');
      in_space = false;
      collapsed.push_back (c);
    }
  std::string out;
  for (std::size_t i = 0; i < collapsed.size (); ++i)
    {
      const char c = collapsed[i];
      if (c == ' ')
        {
          const char prev = out.empty () ? '\0' : out.back ();
          const char next
              = i + 1 < collapsed.size () ? collapsed[i + 1] : '\0';
          auto punct = [] (char p) {
            return p == '(' || p == ')' || p == ',';
          };
          if (punct (prev) || punct (next))
            continue;
        }
      out.push_back (c);
    }
  return out;
}

std::string
ValueText (const std::any &value)
{
  if (!value.has_value () || value.type () == typeid (std::nullptr_t))
    return "NULL";
  if (value.type () == typeid (long long))
    return "i:" + std::to_string (std::any_cast<long long> (value));
  if (value.type () == typeid (int))
    return "i:" + std::to_string (std::any_cast<int> (value));
  if (value.type () == typeid (double))
    {
      std::ostringstream os;
      os.precision (17);
      os << "r:" << std::any_cast<double> (value);
      return os.str ();
    }
  if (value.type () == typeid (std::string))
    return "t:'" + std::any_cast<std::string> (value) + "'";
  if (value.type () == typeid (std::vector<char>))
    {
      const auto &blob = std::any_cast<const std::vector<char> &> (value);
      std::string hex = "x:";
      static const char *digits = "0123456789abcdef";
      for (const char b : blob)
        {
          const auto u = static_cast<unsigned char> (b);
          hex.push_back (digits[u >> 4]);
          hex.push_back (digits[u & 15]);
        }
      return hex;
    }
  return std::string ("?:") + value.type ().name ();
}

std::string
RowText (const std::map<std::string, std::any> &row,
         const std::set<std::string> &skip = {})
{
  std::string out;
  for (const auto &[name, value] : row)
    {
      if (skip.count (name))
        continue;
      out += name + "=" + ValueText (value) + " ";
    }
  return out;
}

std::string
Quote (const std::string &identifier)
{
  std::string out = "\"";
  for (const char c : identifier)
    {
      if (c == '"')
        out += "\"\"";
      else
        out += c;
    }
  return out + "\"";
}

std::string
Text (const std::map<std::string, std::any> &row, const std::string &name)
{
  const auto it = row.find (name);
  if (it == row.end () || it->second.type () != typeid (std::string))
    return {};
  return std::any_cast<std::string> (it->second);
}

// One store's schema, section by section, as comparable lines.
struct SchemaSnapshot
{
  std::vector<std::string> objects;      // sqlite_schema minus rootpage
  std::vector<std::string> columns;      // PRAGMA table_xinfo
  std::vector<std::string> indexes;      // PRAGMA index_list
  std::vector<std::string> index_columns; // PRAGMA index_xinfo
  std::vector<std::string> foreign_keys; // PRAGMA foreign_key_list
  std::vector<std::string> rows;         // every ordinary table's rows
  std::vector<std::string> pragmas;      // user_version, application_id
};

SchemaSnapshot
Snapshot (Store &store, bool include_rows = true)
{
  SchemaSnapshot snap;
  const auto objects = store.Execute (
      "SELECT type, name, tbl_name, sql FROM sqlite_schema "
      "ORDER BY type, name");
  std::vector<std::string> tables;
  std::set<std::string> index_names;
  std::set<std::string> virtual_tables;
  for (const auto &row : objects)
    {
      const auto type = Text (row, "type");
      const auto name = Text (row, "name");
      const auto sql = Text (row, "sql");
      snap.objects.push_back (type + " " + name + " on "
                              + Text (row, "tbl_name") + ": "
                              + NormalizeWhitespace (sql));
      if (type == "table")
        {
          tables.push_back (name);
          if (sql.rfind ("CREATE VIRTUAL TABLE", 0) == 0)
            virtual_tables.insert (name);
        }
      if (type == "index")
        index_names.insert (name);
    }
  for (const auto &table : tables)
    {
      for (const auto &row :
           store.Execute ("PRAGMA table_xinfo(" + Quote (table) + ")"))
        snap.columns.push_back (table + ": " + RowText (row));
      for (const auto &row :
           store.Execute ("PRAGMA index_list(" + Quote (table) + ")"))
        {
          snap.indexes.push_back (table + ": " + RowText (row, { "seq" }));
          index_names.insert (Text (row, "name"));
        }
      for (const auto &row :
           store.Execute ("PRAGMA foreign_key_list(" + Quote (table) + ")"))
        snap.foreign_keys.push_back (table + ": " + RowText (row));
    }
  std::sort (snap.indexes.begin (), snap.indexes.end ());
  for (const auto &index : index_names)
    for (const auto &row :
         store.Execute ("PRAGMA index_xinfo(" + Quote (index) + ")"))
      snap.index_columns.push_back (index + ": " + RowText (row));
  if (include_rows)
    for (const auto &table : tables)
      {
        if (virtual_tables.count (table))
          continue;
        // The ledger's apply time is the only per-run value.
        const std::set<std::string> skip
            = table == "cortext_schema_migrations"
                  ? std::set<std::string>{ "applied_at" }
                  : std::set<std::string>{};
        std::vector<std::string> table_rows;
        for (const auto &row :
             store.Execute ("SELECT * FROM " + Quote (table)))
          table_rows.push_back (table + ": " + RowText (row, skip));
        std::sort (table_rows.begin (), table_rows.end ());
        snap.rows.insert (snap.rows.end (), table_rows.begin (),
                          table_rows.end ());
      }
  for (const char *pragma : { "user_version", "application_id" })
    for (const auto &row :
         store.Execute (std::string ("PRAGMA ") + pragma))
      snap.pragmas.push_back (std::string (pragma) + ": " + RowText (row));
  return snap;
}

// Requires two sections to match, naming every line only one side has.
void
RequireSameSection (const std::string &section,
                    const std::vector<std::string> &expected,
                    const std::vector<std::string> &actual)
{
  if (expected != actual)
    {
      const std::multiset<std::string> e (expected.begin (), expected.end ());
      const std::multiset<std::string> a (actual.begin (), actual.end ());
      std::vector<std::string> only_expected;
      std::vector<std::string> only_actual;
      std::set_difference (e.begin (), e.end (), a.begin (), a.end (),
                           std::back_inserter (only_expected));
      std::set_difference (a.begin (), a.end (), e.begin (), e.end (),
                           std::back_inserter (only_actual));
      for (const auto &line : only_expected)
        UNSCOPED_INFO (section << " only in expected: " << line);
      for (const auto &line : only_actual)
        UNSCOPED_INFO (section << " only in actual: " << line);
      if (only_expected.empty () && only_actual.empty ())
        UNSCOPED_INFO (section << " differs in order only");
    }
  CHECK (expected == actual);
}

void
RequireSameSchema (const SchemaSnapshot &expected,
                   const SchemaSnapshot &actual)
{
  RequireSameSection ("sqlite_schema", expected.objects, actual.objects);
  RequireSameSection ("table_xinfo", expected.columns, actual.columns);
  RequireSameSection ("index_list", expected.indexes, actual.indexes);
  RequireSameSection ("index_xinfo", expected.index_columns,
                      actual.index_columns);
  RequireSameSection ("foreign_key_list", expected.foreign_keys,
                      actual.foreign_keys);
  RequireSameSection ("rows", expected.rows, actual.rows);
  RequireSameSection ("pragmas", expected.pragmas, actual.pragmas);
}

std::unique_ptr<SQLiteStore>
BaselineStore ()
{
  auto store = SQLiteStore::Create (":memory:");
  cortext::store::ApplyMigrations (*store);
  return store;
}

std::unique_ptr<SQLiteStore>
ChainStore (int64_t maximum_id)
{
  auto store = SQLiteStore::Create (":memory:");
  cortext::store::DebugApplyLegacyMigrationChainThroughForTest (*store,
                                                                maximum_id);
  return store;
}

// Records every statement and answers each with no rows, so the
// migration pass sees an empty store.
class RecordingStore : public Store
{
public:
  std::vector<std::string> queries;

  std::vector<std::map<std::string, std::any> >
  Execute (const std::string &query,
           const std::vector<std::any> &params = {}) override
  {
    (void)params;
    queries.push_back (query);
    return {};
  }
  std::unique_ptr<Transaction>
  Begin () override
  {
    throw StoreError ("migrations should not call Store::Begin");
  }
  void Commit () override {}
  void Rollback () override {}
  void Close () override {}
};

std::size_t
CountRows (Store &store, const std::string &sql)
{
  return store.Execute (sql).size ();
}

} // namespace

TEST_CASE ("Fresh stores get the baseline, not the legacy chain",
           "[schema][migration][baseline]")
{
  auto store = BaselineStore ();
  const auto ids = cortext::store::DebugGetAppliedMigrationIdsForTest (*store);
  REQUIRE_FALSE (ids.empty ());
  REQUIRE (ids.count (kBaselineVersion) == 1);
  REQUIRE (ids.count (0) == 1);
  // The chain would create and drop these; the baseline never has them.
  REQUIRE (store->Execute ("SELECT 1 FROM sqlite_schema WHERE name IN ("
                           "'soft_anchors', 'rif_recovery_clock', "
                           "'standing_memories', 'meta_learning_coeffs')")
               .empty ());
  // A fresh store has nothing to backfill.
  REQUIRE (CountRows (*store, "SELECT name FROM cortext_pending_backfills")
           == 0);

  // The empty-store pass runs the baseline statements and no legacy
  // create-then-drop DDL.
  RecordingStore recording;
  cortext::store::ApplyMigrations (recording);
  const auto baseline = cortext::store::DebugBaselineStatementsForTest ();
  for (const auto &sql : baseline)
    REQUIRE (std::count (recording.queries.begin (), recording.queries.end (),
                         sql)
             == 1);
  for (const auto &sql : recording.queries)
    {
      INFO (sql);
      REQUIRE (sql.rfind ("ALTER TABLE", 0) == std::string::npos);
      REQUIRE (sql.rfind ("DROP ", 0) == std::string::npos);
      REQUIRE (sql.rfind ("DELETE ", 0) == std::string::npos);
    }
}

TEST_CASE ("Baseline schema equals the legacy chain 0..58 exactly",
           "[schema][migration][baseline][equivalence]")
{
  auto baseline = BaselineStore ();
  auto chain = ChainStore (kAllMigrations);
  const auto expected = Snapshot (*chain);
  const auto actual = Snapshot (*baseline);
  REQUIRE_FALSE (expected.objects.empty ());
  REQUIRE_FALSE (expected.columns.empty ());
  REQUIRE_FALSE (expected.index_columns.empty ());
  RequireSameSchema (expected, actual);
}

TEST_CASE ("Baseline schema equals the legacy chain on file stores",
           "[schema][migration][baseline][equivalence][file]")
{
  const auto dir = std::filesystem::temp_directory_path ()
                   / ("cortext_schema_baseline_"
                      + std::to_string (std::chrono::steady_clock::now ()
                                            .time_since_epoch ()
                                            .count ()));
  std::filesystem::create_directories (dir);
  {
    auto baseline = SQLiteStore::Create ((dir / "baseline.db").string ());
    cortext::store::ApplyMigrations (*baseline);
    auto chain = SQLiteStore::Create ((dir / "chain.db").string ());
    cortext::store::DebugApplyLegacyMigrationChainThroughForTest (
        *chain, kAllMigrations);
    RequireSameSchema (Snapshot (*chain), Snapshot (*baseline));
  }
  std::filesystem::remove_all (dir);
}

TEST_CASE ("Schema comparison catches a changed table, column, index or "
           "default",
           "[schema][migration][baseline][equivalence]")
{
  const auto reference = Snapshot (*BaselineStore ());

  SECTION ("extra column")
  {
    auto store = BaselineStore ();
    store->Execute ("ALTER TABLE episodes ADD COLUMN extra INTEGER");
    const auto changed = Snapshot (*store);
    REQUIRE (changed.objects != reference.objects);
    REQUIRE (changed.columns != reference.columns);
  }
  SECTION ("missing index")
  {
    auto store = BaselineStore ();
    store->Execute ("DROP INDEX idx_memories_ltm_end");
    const auto changed = Snapshot (*store);
    REQUIRE (changed.objects != reference.objects);
    REQUIRE (changed.indexes != reference.indexes);
    REQUIRE (changed.index_columns != reference.index_columns);
  }
  SECTION ("missing table")
  {
    auto store = BaselineStore ();
    store->Execute ("DROP TABLE cortext_pending_backfills");
    const auto changed = Snapshot (*store);
    REQUIRE (changed.objects != reference.objects);
    REQUIRE (changed.columns != reference.columns);
  }
  SECTION ("changed default")
  {
    auto store = BaselineStore ();
    store->Execute ("DROP TABLE standing_signals");
    store->Execute ("CREATE TABLE standing_signals ("
                    "  signal_id INTEGER PRIMARY KEY,"
                    "  start_ts INTEGER NOT NULL,"
                    "  terms TEXT NOT NULL DEFAULT 'x')");
    const auto changed = Snapshot (*store);
    REQUIRE (changed.objects != reference.objects);
    REQUIRE (changed.columns != reference.columns);
  }
  SECTION ("seed row")
  {
    auto store = BaselineStore ();
    store->Execute ("INSERT INTO cortext_pending_backfills(name) "
                    "VALUES ('memory_text_fts')");
    REQUIRE (Snapshot (*store).rows != reference.rows);
  }
  SECTION ("version pragma")
  {
    auto store = BaselineStore ();
    store->Execute ("PRAGMA user_version = 7");
    REQUIRE (Snapshot (*store).pragmas != reference.pragmas);
  }
}

TEST_CASE ("Whitespace normalization changes only whitespace",
           "[schema][migration][baseline]")
{
  REQUIRE (NormalizeWhitespace ("CREATE TABLE t (\n  a INTEGER ,  b TEXT\n)")
           == "CREATE TABLE t(a INTEGER,b TEXT)");
  REQUIRE (NormalizeWhitespace ("DEFAULT 'a  b'")
           != NormalizeWhitespace ("DEFAULT 'a b c'"));
  REQUIRE (NormalizeWhitespace ("a REAL DEFAULT 0.0")
           != NormalizeWhitespace ("a REAL DEFAULT 1.0"));
}

TEST_CASE ("Historical stores upgrade to the baseline schema",
           "[schema][migration][baseline][upgrade]")
{
  auto baseline = BaselineStore ();
  const auto reference = Snapshot (*baseline);
  const auto reference_ids
      = cortext::store::DebugGetAppliedMigrationIdsForTest (*baseline);
  for (const int64_t version : { 0, 11, 28, 34, 48, 53, 55, 57 })
    {
      INFO ("store created at version " << version);
      auto store = ChainStore (version);
      const auto before
          = cortext::store::DebugGetAppliedMigrationIdsForTest (*store);
      REQUIRE (*before.rbegin () == version);

      cortext::store::ApplyMigrations (*store);

      const auto after
          = cortext::store::DebugGetAppliedMigrationIdsForTest (*store);
      REQUIRE (after == reference_ids);
      RequireSameSchema (reference, Snapshot (*store));
    }
}

TEST_CASE ("Upgraded stores holding memories queue their backfills",
           "[schema][migration][baseline][upgrade][backfill]")
{
  auto store = ChainStore (48);
  std::vector<float> embedding (256, 0.0f);
  embedding[0] = 1.0f;
  store->Execute ("INSERT INTO embeddings (embedding_id, embedding, "
                  "created_at) VALUES (1, ?, 1000)",
                  { embedding });
  store->Execute ("INSERT INTO memories (memory_id, embedding_id, source_id, "
                  "kind, start_ts, created_at) "
                  "VALUES (1, 1, 'opaque/source', 'LONG_TERM', 1000, 1000)");

  cortext::store::ApplyMigrations (*store);

  std::set<std::string> pending;
  for (const auto &row :
       store->Execute ("SELECT name FROM cortext_pending_backfills"))
    pending.insert (Text (row, "name"));
  REQUIRE (pending
           == std::set<std::string>{ "memory_text_fts", "standing_signals" });
  RequireSameSchema (Snapshot (*BaselineStore (), false),
                     Snapshot (*store, false));
}

TEST_CASE ("Applying migrations twice changes nothing",
           "[schema][migration][baseline]")
{
  auto store = BaselineStore ();
  const auto first = Snapshot (*store);
  cortext::store::ApplyMigrations (*store);
  RequireSameSchema (first, Snapshot (*store));
}

TEST_CASE ("Schema creation timing", "[.][schema-timing]")
{
  using clock = std::chrono::steady_clock;
  constexpr int kRuns = 50;
  auto time_ms = [&] (auto &&make) {
    std::vector<double> ms;
    for (int i = 0; i < kRuns; ++i)
      {
        const auto start = clock::now ();
        make (i);
        ms.push_back (std::chrono::duration<double, std::milli> (
                          clock::now () - start)
                          .count ());
      }
    std::sort (ms.begin (), ms.end ());
    return ms[ms.size () / 2];
  };
  const auto dir = std::filesystem::temp_directory_path ()
                   / "cortext_schema_timing";
  std::filesystem::remove_all (dir);
  std::filesystem::create_directories (dir);
  auto memory = [] (bool chain) {
    return [chain] (int) {
      auto store = SQLiteStore::Create (":memory:");
      if (chain)
        cortext::store::DebugApplyLegacyMigrationChainThroughForTest (
            *store, kBaselineVersion);
      else
        cortext::store::ApplyMigrations (*store);
    };
  };
  auto file = [&dir] (bool chain) {
    return [chain, &dir] (int i) {
      const auto path
          = dir / ((chain ? "chain_" : "baseline_") + std::to_string (i)
                   + ".db");
      auto store = SQLiteStore::Create (path.string ());
      if (chain)
        cortext::store::DebugApplyLegacyMigrationChainThroughForTest (
            *store, kBaselineVersion);
      else
        cortext::store::ApplyMigrations (*store);
    };
  };
  std::printf ("schema-timing median of %d runs (store open + schema)\n",
               kRuns);
  std::printf ("  :memory: chain    %.3f ms\n", time_ms (memory (true)));
  std::printf ("  :memory: baseline %.3f ms\n", time_ms (memory (false)));
  std::printf ("  file     chain    %.3f ms\n", time_ms (file (true)));
  std::printf ("  file     baseline %.3f ms\n", time_ms (file (false)));
  std::filesystem::remove_all (dir);
}
