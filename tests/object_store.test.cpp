#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/store/object_store.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>
#include <string>

namespace
{

class MissingPutTransaction final : public cortext::Transaction
{
public:
  std::unique_ptr<cortext::Transaction>
  Begin () override
  {
    return std::make_unique<MissingPutTransaction> ();
  }

  std::vector<std::map<std::string, std::any>>
  Execute (const std::string &query,
           const std::vector<std::any> & /*params*/ = {}) override
  {
    if (query.find ("objstore_put") != std::string::npos)
      {
        ++put_calls;
        return {};
      }
    if (query.find ("objstore_exists") != std::string::npos)
      {
        ++exists_calls;
        return { { { "exists_flag", 0LL } } };
      }
    return {};
  }

  void
  Commit () override
  {
  }

  void
  Rollback () override
  {
  }

  int put_calls = 0;
  int exists_calls = 0;
};

} // namespace

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

  const std::vector<unsigned char> rollback_payload = { 'r', 'o', 'l', 'l' };
  cortext::ObjectId rolled_back_id;
  {
    auto tx = store->Begin ();
    auto object_tx = object_store.Attach (*tx);
    rolled_back_id = object_tx->Put (rollback_payload);
    REQUIRE (object_tx->Exists (rolled_back_id));
    object_tx->Rollback ();
    tx->Rollback ();
  }

  {
    auto object_tx = object_store.Begin ();
    auto loaded = object_tx->Get (rolled_back_id);
    REQUIRE_FALSE (loaded.has_value ());
    object_tx->Commit ();
  }
}

TEST_CASE ("SqlObjectStore respects nested savepoint rollback",
           "[object_store]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  cortext::store::ApplyMigrations (*unique_store);
  std::shared_ptr<cortext::Store> store (std::move (unique_store));
  cortext::SqlObjectStore object_store (store);

  const std::vector<unsigned char> payload = { 's', 'a', 'v', 'e' };
  cortext::ObjectId rolled_back_id;
  {
    auto tx = store->Begin ();
    auto object_tx = object_store.Attach (*tx);
    auto nested = object_tx->Begin ();
    rolled_back_id = nested->Put (payload);
    REQUIRE (nested->Exists (rolled_back_id));
    nested->Rollback ();
    REQUIRE_FALSE (object_tx->Exists (rolled_back_id));
    object_tx->Commit ();
    tx->Commit ();
  }

  auto read_tx = object_store.Begin ();
  const auto loaded = read_tx->Get (rolled_back_id);
  REQUIRE_FALSE (loaded.has_value ());
  read_tx->Commit ();
}

TEST_CASE ("SqlObjectStore duplicate puts are idempotent", "[object_store]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  cortext::store::ApplyMigrations (*unique_store);
  std::shared_ptr<cortext::Store> store (std::move (unique_store));
  cortext::SqlObjectStore object_store (store);

  const std::vector<unsigned char> payload = { 'd', 'u', 'p' };

  auto tx = store->Begin ();
  auto object_tx = object_store.Attach (*tx);
  const auto first_id = object_tx->Put (payload);
  const auto second_id = object_tx->Put (payload);

  REQUIRE (first_id == second_id);
  REQUIRE (object_tx->Exists (first_id));

  object_tx->Commit ();
  tx->Commit ();

  auto read_tx = object_store.Begin ();
  const auto loaded = read_tx->Get (first_id);
  REQUIRE (loaded.has_value ());
  REQUIRE (*loaded == payload);
  read_tx->Commit ();
}

TEST_CASE ("SqlObjectStore rejects operations after transaction finishes",
           "[object_store]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  cortext::store::ApplyMigrations (*unique_store);
  std::shared_ptr<cortext::Store> store (std::move (unique_store));
  cortext::SqlObjectStore object_store (store);

  const std::vector<unsigned char> payload = { 'd', 'o', 'n', 'e' };
  auto object_tx = object_store.Begin ();
  const auto object_id = object_tx->Put (payload);
  object_tx->Commit ();

  REQUIRE_THROWS_AS (object_tx->Begin (),
                     cortext::TransactionAlreadyFinishedError);
  REQUIRE_THROWS_AS (object_tx->Put (payload),
                     cortext::TransactionAlreadyFinishedError);
  REQUIRE_THROWS_AS (object_tx->Get (object_id),
                     cortext::TransactionAlreadyFinishedError);
  REQUIRE_THROWS_AS (object_tx->Exists (object_id),
                     cortext::TransactionAlreadyFinishedError);
  REQUIRE_THROWS_AS (object_tx->Delete (object_id),
                     cortext::TransactionAlreadyFinishedError);

  auto rolled_back_tx = object_store.Begin ();
  rolled_back_tx->Rollback ();
  REQUIRE_THROWS_AS (rolled_back_tx->Put (payload),
                     cortext::TransactionAlreadyFinishedError);
}

TEST_CASE ("SqlObjectStore attached extension transaction stays bound to DB tx",
           "[object_store]")
{
  cortext::testing::ScopedEnvVar backend ("CORTEXT_OBJSTORE_BACKEND",
                                          "extension");
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  cortext::store::ApplyMigrations (*unique_store);
  std::shared_ptr<cortext::Store> store (std::move (unique_store));
  cortext::SqlObjectStore object_store (store);

  const std::vector<unsigned char> payload = { 'b', 'o', 'u', 'n', 'd' };
  {
    auto tx = store->Begin ();
    auto object_tx = object_store.Attach (*tx);
    auto competing_savepoint = tx->Begin ();
    REQUIRE_THROWS_AS (object_tx->Begin (),
                       cortext::TransactionNotCurrentError);
    REQUIRE_THROWS_AS (object_tx->Put (payload),
                       cortext::TransactionNotCurrentError);
    competing_savepoint->Rollback ();
    object_tx->Rollback ();
    tx->Rollback ();
  }

  {
    auto tx = store->Begin ();
    auto object_tx = object_store.Attach (*tx);
    tx->Commit ();
    REQUIRE_THROWS_AS (object_tx->Put (payload),
                       cortext::TransactionAlreadyFinishedError);
  }
}

TEST_CASE ("SqlObjectStore extension put fails without persistence confirmation",
           "[object_store]")
{
  cortext::testing::ScopedEnvVar backend ("CORTEXT_OBJSTORE_BACKEND",
                                          "extension");
  MissingPutTransaction tx;
  const std::vector<unsigned char> payload = { 'm', 'i', 's', 's' };

  REQUIRE_THROWS_AS (cortext::PutObject (nullptr, tx, payload),
                     cortext::StoreError);
  REQUIRE (tx.put_calls == 1);
  REQUIRE (tx.exists_calls == 1);
}
