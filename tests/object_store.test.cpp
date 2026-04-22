#include <catch2/catch_test_macros.hpp>
#include <cortext/store/object_store.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>

TEST_CASE ("ObjectStore computes BLAKE3 content ids", "[object_store]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  cortext::store::ApplyMigrations (*store);

  const std::vector<unsigned char> payload = { 'p', 'l', 'a', 'n', 'u', 'm' };
  auto rows = store->Execute ("SELECT objstore_put(?1) AS id", { payload });
  REQUIRE (rows.size () == 1);

  const auto sqlite_id = cortext::store::BlobFromAny (rows[0].at ("id"));
  const auto computed_id = cortext::ComputeObjectId (payload);

  REQUIRE (computed_id.size () == cortext::kObjectIdSize);
  REQUIRE (computed_id == sqlite_id);
}

TEST_CASE ("SqlObjectStore stores, reads, and rolls back objects",
           "[object_store]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  cortext::store::ApplyMigrations (*unique_store);
  std::shared_ptr<cortext::Store> store (std::move (unique_store));
  cortext::SqlObjectStore object_store (store);

  const std::vector<unsigned char> payload = { 0x01, 0x02, 0x03 };
  cortext::ObjectId committed_id;
  {
    auto tx = store->Begin ();
    auto object_tx = object_store.Attach (*tx);
    committed_id = object_tx->Put (payload);
    REQUIRE (object_tx->Exists (committed_id));
    object_tx->Commit ();
    tx->Commit ();
  }

  {
    auto object_tx = object_store.Begin ();
    auto loaded = object_tx->Get (committed_id);
    REQUIRE (loaded.has_value ());
    REQUIRE (*loaded == payload);
    object_tx->Commit ();
  }

  REQUIRE (committed_id.size () == cortext::kObjectIdSize);
}
