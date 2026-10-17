#include "../src/store/memory_text_index_internal.hpp"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <cortext/cortext.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

namespace fts = cortext::store::memory_text_index_internal;

namespace
{

class TempDbPath
{
public:
  TempDbPath ()
  {
    const auto stamp = std::chrono::high_resolution_clock::now ()
                           .time_since_epoch ()
                           .count ();
    path_ = (std::filesystem::temp_directory_path ()
             / ("cortext_fts_test_" + std::to_string (stamp) + ".db"))
                .string ();
  }
  ~TempDbPath ()
  {
    std::error_code ec;
    for (const char *suffix : { "", "-wal", "-shm", "-journal" })
      std::filesystem::remove (path_ + suffix, ec);
  }
  const std::string &
  path () const
  {
    return path_;
  }

private:
  std::string path_;
};

std::shared_ptr<cortext::Store>
OpenStore (const std::string &path)
{
  return std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (path.c_str ()));
}

std::vector<long long>
SearchIds (cortext::Store &store, const std::string &query)
{
  auto tx = store.Begin ();
  std::vector<long long> ids;
  for (const auto &hit : fts::Search (*tx, query, 10, 16))
    ids.push_back (hit.memory_id);
  tx->Commit ();
  return ids;
}

void
Index (cortext::Store &store, long long memory_id, const std::string &text)
{
  auto tx = store.Begin ();
  fts::IndexMemoryText (*tx, memory_id, text);
  tx->Commit ();
}

std::string
TextOf (const cortext::Cortext::Context::Memory &memory)
{
  std::string text;
  for (const auto &blob : memory.content)
    text.append (reinterpret_cast<const char *> (blob.data ()), blob.size ());
  return text;
}

bool
Retrieved (const cortext::Cortext::Context &context, const std::string &needle)
{
  for (const auto &memory : context.retrieved_memory)
    if (TextOf (memory).find (needle) != std::string::npos)
      return true;
  return false;
}

double
OperationValue (const cortext::Cortext::Context &context, const char *key)
{
  const auto it = context.output.operation_ms.find (key);
  return it == context.output.operation_ms.end () ? 0.0 : it->second;
}

const std::vector<std::string> kDistractors = {
  "We talked about the weather and how rainy the spring has been.",
  "I made pasta for dinner with a tomato and basil sauce.",
  "The team meeting moved to Thursday afternoon this week.",
  "My sister recommended a podcast about urban gardening.",
  "I need to renew my library card before the end of the month.",
  "The bike tire went flat again on the way to work.",
  "We watched a documentary about deep sea creatures last night.",
  "The new coffee shop downtown has excellent pastries.",
  "I started reading a mystery novel set in Edinburgh.",
  "The printer at the office keeps jamming on double sided pages.",
};

} // namespace

TEST_CASE ("memory text match query quotes words and drops stopwords",
           "[fts]")
{
  REQUIRE (fts::BuildMatchQuery ("What is the Garage door code?", 16)
           == "\"garage\" OR \"door\" OR \"code\"");
  // FTS5 syntax in user text is inert inside quoted strings.
  REQUIRE (fts::BuildMatchQuery ("NEAR(a b) OR x* -y \"z\"", 16)
           == "\"near\" OR \"b\" OR \"x\" OR \"y\" OR \"z\"");
  REQUIRE (fts::BuildMatchQuery ("door Door DOOR", 16) == "\"door\"");
  REQUIRE (fts::BuildMatchQuery ("one two three", 2)
           == "\"one\" OR \"two\"");
  REQUIRE (fts::BuildMatchQuery ("?! -- the and", 16).empty ());
  // Non-ASCII words pass through for the FTS5 tokenizer to fold.
  REQUIRE (fts::BuildMatchQuery ("Café Москва", 16)
           == "\"café\" OR \"Москва\"");
}

TEST_CASE ("memory text index ranks, stems and folds unicode", "[fts]")
{
  TempDbPath db;
  auto store = OpenStore (db.path ());
  cortext::store::ApplyMigrations (*store);
  Index (*store, 1, "We booked trips to Zanzibar in June.");
  Index (*store, 2, "The trip to the lake was in June.");
  Index (*store, 3, "June was warm and the lake was calm.");
  Index (*store, 4, "Lunch at the CAFÉ near МОСКВА station; 東京 next.");

  auto ids = SearchIds (*store, "When is the Zanzibar trip?");
  REQUIRE (ids.size () == 2);
  REQUIRE (ids.front () == 1); // rare term ranks first; "trips" stems
  REQUIRE (SearchIds (*store, "cafe") == std::vector<long long>{ 4 });
  REQUIRE (SearchIds (*store, "москва") == std::vector<long long>{ 4 });
  REQUIRE (SearchIds (*store, "東京") == std::vector<long long>{ 4 });
  REQUIRE (SearchIds (*store, "groceries").empty ());
  REQUIRE (SearchIds (*store, "— …").empty ());
}

TEST_CASE ("memory text index rolls back and deletes with its memory",
           "[fts]")
{
  TempDbPath db;
  auto store = OpenStore (db.path ());
  cortext::store::ApplyMigrations (*store);
  {
    auto tx = store->Begin ();
    fts::IndexMemoryText (*tx, 7, "quillfeather locker 4417");
    tx->Rollback ();
  }
  REQUIRE (SearchIds (*store, "quillfeather").empty ());

  store->Execute ("INSERT INTO memories(memory_id, source_id, kind, "
                  "start_ts, created_at) VALUES (8, 's', 'LONG_TERM', 1, 1)");
  store->Execute ("INSERT INTO memories(memory_id, source_id, kind, "
                  "start_ts, created_at) VALUES (9, 's', 'WORKING', 1, 1)");
  Index (*store, 8, "brightwater ferry pier");
  REQUIRE (SearchIds (*store, "brightwater") == std::vector<long long>{ 8 });
  // Deleting an unindexed memory leaves the index intact.
  store->Execute ("DELETE FROM memories WHERE memory_id = 9");
  REQUIRE (SearchIds (*store, "brightwater") == std::vector<long long>{ 8 });
  store->Execute ("DELETE FROM memories WHERE memory_id = 8");
  REQUIRE (SearchIds (*store, "brightwater").empty ());
}

TEST_CASE ("engine seeds queries from the memory text index",
           "[fts][cortext][aist]")
{
  TempDbPath db;
  cortext::Cortext::Config cfg;
  auto store = OpenStore (db.path ());
  auto ctx = cortext::Cortext::Create (cfg, store);
  REQUIRE (ctx != nullptr);

  std::uint64_t ts = 1000;
  for (const auto &text : kDistractors)
    ctx->ProcessTextAt (text, "chat/main", ts += 1000,
                        cortext::Retention::Durable);
  auto stored = ctx->ProcessTextAt (
      "My locker combination at the Quillfeather gym is 4417.", "chat/main",
      ts += 1000, cortext::Retention::Durable);
  REQUIRE (stored.output.stored_memory_id.has_value ());
  // Indexed at write time, in the memory's own transaction.
  REQUIRE (SearchIds (*store, "Quillfeather")
           == std::vector<long long>{ *stored.output.stored_memory_id });

  auto first = ctx->ProcessTextAt ("Quillfeather locker", "query/main",
                                   ts += 1000,
                                   cortext::Retention::Ephemeral);
  REQUIRE (OperationValue (first, "GraphRetrieve.lexical_hits") >= 1.0);
  REQUIRE (Retrieved (first, "4417"));
  // The read-only query wrote nothing to the index.
  REQUIRE (SearchIds (*store, "Quillfeather locker").size () == 1);

  ctx->ProcessTextAt ("The Brightwater ferry leaves pier 9 at 7:40.",
                      "chat/main", ts += 1000, cortext::Retention::Durable);
  auto second = ctx->ProcessTextAt ("Brightwater ferry pier", "query/main",
                                    ts += 1000,
                                    cortext::Retention::Ephemeral);
  REQUIRE (OperationValue (second, "GraphRetrieve.lexical_hits") >= 1.0);
  REQUIRE (Retrieved (second, "pier 9"));
}

TEST_CASE ("keyword-seeded stale facts are recalled with their correction",
           "[fts][cortext][belief_revision][aist]")
{
  TempDbPath db;
  cortext::Cortext::Config cfg;
  auto store = OpenStore (db.path ());
  auto ctx = cortext::Cortext::Create (cfg, store);
  std::uint64_t ts = 1000;
  for (const auto &text : kDistractors)
    ctx->ProcessTextAt (text, "chat/main", ts += 1000,
                        cortext::Retention::Durable);
  ctx->ProcessTextAt ("Pemberly's dentist appointment is March 3 at 10am.",
                      "chat/main", ts += 1000, cortext::Retention::Durable);
  ctx->ProcessTextAt ("Pemberly's March 3 dentist appointment is canceled; "
                      "it moved to March 5 at 2pm.",
                      "chat/main", ts += 1000, cortext::Retention::Durable);
  auto recalled = ctx->ProcessTextAt ("When is Pemberly's dentist "
                                      "appointment?",
                                      "query/main", ts += 1000,
                                      cortext::Retention::Ephemeral);
  REQUIRE (OperationValue (recalled, "GraphRetrieve.lexical_hits") >= 2.0);
  REQUIRE (Retrieved (recalled, "March 5 at 2pm"));
}

TEST_CASE ("migration backfills the memory text index of an existing store",
           "[fts][cortext][migration][aist]")
{
  TempDbPath db;
  cortext::Cortext::Config cfg;
  std::uint64_t ts = 1000;
  {
    auto store = OpenStore (db.path ());
    auto ctx = cortext::Cortext::Create (cfg, store);
    for (const auto &text : kDistractors)
      ctx->ProcessTextAt (text, "chat/main", ts += 1000,
                          cortext::Retention::Durable);
    ctx->ProcessTextAt ("The Marrowind cabin wifi password is tulip88.",
                        "chat/main", ts += 1000,
                        cortext::Retention::Durable);
  }
  {
    // Rewind the store to before migration 49.
    auto store = OpenStore (db.path ());
    store->Execute ("DROP TRIGGER memories_text_fts_delete");
    store->Execute ("DROP TABLE memory_text_fts");
    store->Execute ("DROP TABLE cortext_pending_backfills");
    store->Execute ("DELETE FROM cortext_schema_migrations WHERE id = 49");
  }
  auto store = OpenStore (db.path ());
  auto ctx = cortext::Cortext::Create (cfg, store);
  REQUIRE (store->Execute ("SELECT 1 FROM cortext_pending_backfills").empty ());
  REQUIRE (SearchIds (*store, "Marrowind").size () == 1);
  REQUIRE (SearchIds (*store, "pasta basil").size () == 1);
  auto recalled = ctx->ProcessTextAt ("Marrowind wifi password", "query/main",
                                      ts += 1000,
                                      cortext::Retention::Ephemeral);
  REQUIRE (OperationValue (recalled, "GraphRetrieve.lexical_hits") >= 1.0);
  REQUIRE (Retrieved (recalled, "tulip88"));
}
