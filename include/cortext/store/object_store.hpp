#pragma once

#include "cortext/store/store.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace cortext
{

using ObjectId = std::vector<unsigned char>;

inline constexpr std::size_t kObjectIdSize = 32;

/// @brief Compute the content-addressed object id for a payload.
ObjectId ComputeObjectId (const std::vector<unsigned char> &data);

/// @brief Transactional content-addressed object storage.
class ObjectTransaction
{
public:
  virtual ~ObjectTransaction () = default;

  /// @brief Begin a nested object transaction.
  virtual std::unique_ptr<ObjectTransaction> Begin () = 0;

  /// @brief Store bytes idempotently and return their content-addressed id.
  virtual ObjectId Put (const std::vector<unsigned char> &data) = 0;

  /// @brief Read bytes by object id. Empty optional means not found.
  virtual std::optional<std::vector<unsigned char>>
  Get (const ObjectId &id) = 0;

  /// @brief Check whether an object id exists.
  virtual bool Exists (const ObjectId &id) = 0;

  /// @brief Delete an object id if it exists.
  virtual bool Delete (const ObjectId &id) = 0;

  /// @brief Commit the current object transaction.
  virtual void Commit () = 0;

  /// @brief Rollback the current object transaction.
  virtual void Rollback () = 0;

protected:
  ObjectTransaction () = default;
};

/// @brief Content-addressed object storage provider.
class ObjectStore
{
public:
  virtual ~ObjectStore () = default;

  /// @brief Begin a new object transaction.
  virtual std::unique_ptr<ObjectTransaction> Begin () = 0;

  /// @brief Close the provider.
  virtual void Close () = 0;

protected:
  ObjectStore () = default;
};

/// @brief sqlite-objstore adapter over an existing SQL store.
class SqlObjectStore final : public ObjectStore
{
public:
  explicit SqlObjectStore (std::shared_ptr<Store> store);

  std::unique_ptr<ObjectTransaction> Begin () override;
  void Close () override;

  /// @brief Create an object transaction attached to an existing DB tx.
  std::unique_ptr<ObjectTransaction> Attach (Transaction &tx);

private:
  std::shared_ptr<Store> store_;
};

/// @brief Store bytes through an object transaction when present, otherwise
/// fall back to sqlite-objstore SQL on the supplied DB transaction.
ObjectId PutObject (ObjectTransaction *object_tx, Transaction &tx,
                    const std::vector<unsigned char> &data);

/// @brief Load bytes through an object transaction when present, otherwise
/// fall back to sqlite-objstore SQL on the supplied DB transaction.
std::optional<std::vector<unsigned char>>
GetObject (ObjectTransaction *object_tx, Transaction &tx, const ObjectId &id);

} // namespace cortext
