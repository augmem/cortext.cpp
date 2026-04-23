#include "cortext/store/object_store.hpp"

#include "cortext/store/utils.hpp"

#include <any>
#include <set>
#include <stdexcept>
#include <utility>

#if defined(CORTEXT_EMBED_OBJSTORE)
#include <objstore/blake3.h>
#endif

namespace cortext
{

namespace
{

class SqlObjectTransaction final : public ObjectTransaction
{
public:
  explicit SqlObjectTransaction (Transaction *tx) : tx_ (tx) {}

  explicit SqlObjectTransaction (std::unique_ptr<Transaction> owned_tx)
      : tx_ (owned_tx.get ()), owned_tx_ (std::move (owned_tx))
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
    return std::make_unique<SqlObjectTransaction> (tx_->Begin ());
  }

  ObjectId Put (const std::vector<unsigned char> &data) override
  {
    if (!tx_)
      {
        throw StoreError ("Cannot put object without DB tx");
      }
    ObjectId id = ComputeObjectId (data);
    if (put_ids_.find (id) != put_ids_.end ())
      {
        return id;
      }
    if (Exists (id))
      {
        put_ids_.insert (id);
        return id;
      }
    auto rows = tx_->Execute ("SELECT objstore_put(?1) AS id", { data });
    if (!rows.empty () && rows[0].count ("id") != 0)
      {
        auto stored_id = store::BlobFromAny (rows[0].at ("id"));
        if (!stored_id.empty ())
          {
            put_ids_.insert (stored_id);
            return stored_id;
          }
      }
    put_ids_.insert (id);
    return id;
  }

  std::optional<std::vector<unsigned char>>
  Get (const ObjectId &id) override
  {
    if (!tx_)
      {
        throw StoreError ("Cannot get object without DB tx");
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
  std::set<ObjectId> put_ids_;
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
