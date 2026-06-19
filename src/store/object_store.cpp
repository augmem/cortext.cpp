#include "cortext/store/object_store.hpp"

#include "cortext/store/utils.hpp"

#include <algorithm>
#include <any>
#include <cctype>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(CORTEXT_EMBED_OBJSTORE)
#include <objstore/blake3.h>
#endif

namespace cortext
{

namespace
{

constexpr std::size_t kObjstoreRowidPrefixSize = 8;

std::string
LowerEnv (const char *name)
{
  const char *value = std::getenv (name);
  if (!value || value[0] == '\0')
    {
      return {};
    }
  std::string out (value);
  std::transform (out.begin (), out.end (), out.begin (),
                  [] (unsigned char c) {
                    return static_cast<char> (std::tolower (c));
                  });
  return out;
}

bool
UseDirectSQLiteObjstore ()
{
  const std::string backend = LowerEnv ("CORTEXT_OBJSTORE_BACKEND");
  return backend.empty () || backend == "sqlite" || backend == "db";
}

std::vector<unsigned char>
RowidPrefix (const ObjectId &id)
{
  const std::size_t n = std::min (kObjstoreRowidPrefixSize, id.size ());
  return std::vector<unsigned char> (id.begin (), id.begin () + n);
}

std::string
ObjectIdKey (const ObjectId &id)
{
  if (id.empty ())
    {
      return {};
    }
  return std::string (reinterpret_cast<const char *> (id.data ()),
                      id.size ());
}

std::optional<std::vector<unsigned char>>
DirectSQLiteGet (Transaction &tx, const ObjectId &id)
{
  auto rows = tx.Execute ("SELECT data FROM objstore_data WHERE id = ?",
                          { id });
  if (rows.empty () || rows[0].count ("data") == 0)
    {
      return std::nullopt;
    }
  return store::BlobFromAny (rows[0].at ("data"));
}

bool
DirectSQLiteExists (Transaction &tx, const ObjectId &id)
{
  auto rows = tx.Execute ("SELECT 1 AS present FROM objstore_data "
                          "WHERE id = ? LIMIT 1",
                          { id });
  return !rows.empty ();
}

ObjectId
DirectSQLitePut (Transaction &tx, const std::vector<unsigned char> &data)
{
  ObjectId id = ComputeObjectId (data);
  auto existing = DirectSQLiteGet (tx, id);
  if (existing)
    {
      if (*existing != data)
        {
          throw StoreError ("objstore content id collision");
        }
      return id;
    }

  tx.Execute ("INSERT INTO objstore_data(id, data) VALUES (?, ?)",
              { id, data });
  tx.Execute ("INSERT OR IGNORE INTO objstore_rowidx(rowid_prefix, id) "
              "VALUES (?, ?)",
              { RowidPrefix (id), id });
  return id;
}

bool
DirectSQLiteDelete (Transaction &tx, const ObjectId &id)
{
  if (!DirectSQLiteExists (tx, id))
    {
      return false;
    }
  tx.Execute ("DELETE FROM objstore_data WHERE id = ?", { id });
  tx.Execute ("DELETE FROM objstore_rowidx WHERE rowid_prefix = ? AND id = ?",
              { RowidPrefix (id), id });
  return true;
}

class SqlObjectTransaction final : public ObjectTransaction
{
public:
  explicit SqlObjectTransaction (Transaction *tx,
                                 bool direct_sqlite = UseDirectSQLiteObjstore ())
      : tx_ (tx), direct_sqlite_ (direct_sqlite)
  {
  }

  explicit SqlObjectTransaction (
      std::unique_ptr<Transaction> owned_tx,
      bool direct_sqlite = UseDirectSQLiteObjstore ())
      : tx_ (owned_tx.get ()), owned_tx_ (std::move (owned_tx)),
        direct_sqlite_ (direct_sqlite)
  {
  }

  ~SqlObjectTransaction () override
  {
    if (owned_tx_ && !finished_)
      {
        try
          {
            owned_tx_->Rollback ();
          }
        catch (...)
          {
          }
      }
  }

  std::unique_ptr<ObjectTransaction> Begin () override
  {
    if (!tx_)
      {
        throw StoreError ("Cannot begin object transaction without DB tx");
      }
    return std::make_unique<SqlObjectTransaction> (tx_->Begin (),
                                                   direct_sqlite_);
  }

  ObjectId Put (const std::vector<unsigned char> &data) override
  {
    if (!tx_)
      {
        throw StoreError ("Cannot put object without DB tx");
      }
    if (direct_sqlite_)
      {
        return DirectSQLitePut (*tx_, data);
      }
    ObjectId id = ComputeObjectId (data);
    const std::string id_key = ObjectIdKey (id);
    if (put_ids_.find (id_key) != put_ids_.end ())
      {
        return id;
      }
    auto rows = tx_->Execute ("SELECT objstore_put(?1) AS id", { data });
    if (!rows.empty () && rows[0].count ("id") != 0)
      {
        auto stored_id = store::BlobFromAny (rows[0].at ("id"));
        if (!stored_id.empty ())
          {
            put_ids_.insert (ObjectIdKey (stored_id));
            return stored_id;
          }
      }
    if (Exists (id))
      {
        put_ids_.insert (id_key);
        return id;
      }
    put_ids_.insert (id_key);
    return id;
  }

  std::optional<std::vector<unsigned char>>
  Get (const ObjectId &id) override
  {
    if (!tx_)
      {
        throw StoreError ("Cannot get object without DB tx");
      }
    if (direct_sqlite_)
      {
        return DirectSQLiteGet (*tx_, id);
      }
    auto rows = tx_->Execute ("SELECT objstore_get(?1) AS data", { id });
    if (rows.empty () || rows[0].count ("data") == 0)
      {
        return std::nullopt;
      }
    return store::BlobFromAny (rows[0].at ("data"));
  }

  bool Exists (const ObjectId &id) override
  {
    if (!tx_)
      {
        throw StoreError ("Cannot check object without DB tx");
      }
    if (direct_sqlite_)
      {
        return DirectSQLiteExists (*tx_, id);
      }
    try
      {
        auto rows = tx_->Execute ("SELECT objstore_exists(?1) AS exists_flag",
                                  { id });
        if (rows.empty () || rows[0].count ("exists_flag") == 0)
          {
            return false;
          }
        return store::AnyToLongLong (rows[0].at ("exists_flag")).value_or (0)
               != 0;
      }
    catch (const StoreError &)
      {
        return false;
      }
  }

  bool Delete (const ObjectId &id) override
  {
    if (!tx_)
      {
        throw StoreError ("Cannot delete object without DB tx");
      }
    if (direct_sqlite_)
      {
        return DirectSQLiteDelete (*tx_, id);
      }
    auto rows = tx_->Execute ("SELECT objstore_delete(?1) AS deleted",
                              { id });
    if (rows.empty () || rows[0].count ("deleted") == 0)
      {
        return false;
      }
    return store::AnyToLongLong (rows[0].at ("deleted")).value_or (0) != 0;
  }

  void Commit () override
  {
    if (finished_)
      {
        throw TransactionAlreadyFinishedError (
            "Object transaction is already finished");
      }
    if (owned_tx_)
      {
        owned_tx_->Commit ();
      }
    finished_ = true;
  }

  void Rollback () override
  {
    if (finished_)
      {
        throw TransactionAlreadyFinishedError (
            "Object transaction is already finished");
      }
    if (owned_tx_)
      {
        owned_tx_->Rollback ();
      }
    finished_ = true;
  }

private:
  Transaction *tx_ = nullptr;
  std::unique_ptr<Transaction> owned_tx_;
  std::set<std::string> put_ids_;
  bool direct_sqlite_ = false;
  bool finished_ = false;
};

} // namespace

ObjectId
ComputeObjectId (const std::vector<unsigned char> &data)
{
  ObjectId id (kObjectIdSize);
#if defined(CORTEXT_EMBED_OBJSTORE)
  objstore_id raw_id{};
  objstore_blake3_hash_blob (
      data.empty () ? nullptr : reinterpret_cast<const uint8_t *> (data.data ()),
      data.size (), &raw_id);
  id.assign (raw_id.bytes, raw_id.bytes + kObjectIdSize);
  return id;
#else
  (void)data;
  throw StoreError (
      "BLAKE3 object ids require CORTEXT_EMBED_OBJSTORE to be enabled");
#endif
}

SqlObjectStore::SqlObjectStore (std::shared_ptr<Store> store)
    : store_ (std::move (store))
{
  if (!store_)
    {
      throw StoreError ("SqlObjectStore requires a non-null Store");
    }
}

std::unique_ptr<ObjectTransaction>
SqlObjectStore::Begin ()
{
  return std::make_unique<SqlObjectTransaction> (store_->Begin ());
}

void
SqlObjectStore::Close ()
{
}

std::unique_ptr<ObjectTransaction>
SqlObjectStore::Attach (Transaction &tx)
{
  return std::make_unique<SqlObjectTransaction> (&tx);
}

ObjectId
PutObject (ObjectTransaction *object_tx, Transaction &tx,
           const std::vector<unsigned char> &data)
{
  if (object_tx)
    {
      return object_tx->Put (data);
    }
  SqlObjectTransaction fallback (&tx);
  return fallback.Put (data);
}

std::optional<std::vector<unsigned char>>
GetObject (ObjectTransaction *object_tx, Transaction &tx, const ObjectId &id)
{
  if (object_tx)
    {
      return object_tx->Get (id);
    }
  SqlObjectTransaction fallback (&tx);
  return fallback.Get (id);
}

} // namespace cortext
