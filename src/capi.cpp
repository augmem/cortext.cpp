/// @file
/// @brief C API implementation for Cortext.
///
/// Return codes convention:
///   0 = success
///   1 = invalid parameters (NULL pointer, invalid handle)
///   2 = internal error (exception caught during processing)
#include "cortext/capi.h"
#include "cortext/cortext.hpp"
#include "cortext/store/object_store.hpp"
#include "cortext/store/sqlite_store.hpp"
#include "cortext/store/store.hpp"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef CORTEXT_VERSION
#define CORTEXT_VERSION "0.0.0-dev"
#endif

namespace
{
thread_local std::string g_last_error;

void
clear_last_error ()
{
  g_last_error.clear ();
}

void
set_last_error (std::string message)
{
  g_last_error = std::move (message);
}

cortext::Cortext::Config
default_config ()
{
  return cortext::Cortext::Config{};
}

cortext::Cortext::Config
config_from_c (const cortext_config *cfg)
{
  auto cpp_cfg = default_config ();
  if (!cfg)
    {
      return cpp_cfg;
    }

  cpp_cfg.focus = cfg->focus;
  cpp_cfg.sensitivity = cfg->sensitivity;
  cpp_cfg.stability = cfg->stability;
  cpp_cfg.affect_interrupt = cfg->affect_interrupt != 0;
  cpp_cfg.affect_retrieval = cfg->affect_retrieval != 0;
  cpp_cfg.reinforcement_enabled = cfg->reinforcement_enabled != 0;
  cpp_cfg.procedural_enabled = cfg->procedural_enabled != 0;
  cpp_cfg.sequential_edges_enabled = cfg->sequential_edges_enabled != 0;
  if (cfg->struct_size
      >= offsetof (cortext_config, signal_filter_audio_enabled)
             + sizeof (cfg->signal_filter_audio_enabled))
    {
      cpp_cfg.signal_filter_audio_enabled
          = cfg->signal_filter_audio_enabled != 0;
    }
  if (cfg->struct_size
      >= offsetof (cortext_config, signal_filter_image_enabled)
             + sizeof (cfg->signal_filter_image_enabled))
    {
      cpp_cfg.signal_filter_image_enabled
          = cfg->signal_filter_image_enabled != 0;
    }
  if (cfg->struct_size
      >= offsetof (cortext_config, signal_filter_text_enabled)
             + sizeof (cfg->signal_filter_text_enabled))
    {
      cpp_cfg.signal_filter_text_enabled
          = cfg->signal_filter_text_enabled != 0;
    }
  return cpp_cfg;
}

cortext::Cortext::Media
media_from_c (const cortext_media *media)
{
  if (!media || media->size == 0)
    {
      return {};
    }
  return cortext::Cortext::Media{
    media->data, media->size, media->mimetype ? media->mimetype : ""
  };
}

bool
valid_media_or_set_error (const cortext_media *media)
{
  if (!media || media->size == 0)
    {
      return true;
    }
  if (!media->data)
    {
      set_last_error ("media data must be non-NULL when media size is non-zero");
      return false;
    }
  if (!media->mimetype || media->mimetype[0] == '\0')
    {
      set_last_error ("media mimetype must be non-NULL and non-empty when media data is provided");
      return false;
    }
  return true;
}

bool
include_embedding_from_options (const cortext_process_json_options *options)
{
  if (!options)
    {
      return true;
    }
  if (options->struct_size
      < offsetof (cortext_process_json_options, include_embedding)
            + sizeof (options->include_embedding))
    {
      return true;
    }
  return options->include_embedding != 0;
}

cortext::Retention
retention_from_options (const cortext_process_json_options *options)
{
  // Omitted / undersized options => Natural (false/false forces).
  if (!options)
    {
      return cortext::Retention::Natural;
    }
  // The prior two-field struct has the same sizeof as the initial three-field
  // layout on common 64-bit ABIs because retention occupied tail padding.
  // Require the expanded layout before reading retention so old padding is
  // always interpreted as the Natural default.
  if (options->struct_size < sizeof (cortext_process_json_options))
    {
      return cortext::Retention::Natural;
    }
  switch (options->retention)
    {
    case CORTEXT_RETENTION_DURABLE:
      return cortext::Retention::Durable;
    case CORTEXT_RETENTION_BOUNDARY:
      return cortext::Retention::Boundary;
    case CORTEXT_RETENTION_EPHEMERAL:
      return cortext::Retention::Ephemeral;
    case CORTEXT_RETENTION_NATURAL:
    default:
      return cortext::Retention::Natural;
    }
}

std::any
value_to_any (const cortext_db_value &value)
{
  switch (value.type)
    {
    case CORTEXT_DB_VALUE_INT64:
      return static_cast<long long> (value.int64_value);
    case CORTEXT_DB_VALUE_DOUBLE:
      return value.double_value;
    case CORTEXT_DB_VALUE_TEXT:
      return std::string (value.text_value ? value.text_value : "");
    case CORTEXT_DB_VALUE_BLOB:
      {
        const auto *begin = value.blob_data;
        if (!begin || value.blob_size == 0)
          {
            return std::vector<unsigned char>{};
          }
        return std::vector<unsigned char> (begin, begin + value.blob_size);
      }
    case CORTEXT_DB_VALUE_NULL:
    default:
      return nullptr;
    }
}

struct CallbackParam
{
  cortext_db_value value{};
  std::string text;
  std::vector<unsigned char> blob;
};

std::vector<cortext_db_value>
params_to_values (const std::vector<std::any> &params,
                  std::vector<CallbackParam> &storage)
{
  storage.clear ();
  storage.reserve (params.size ());

  for (const auto &param : params)
    {
      CallbackParam entry;
      if (param.type () == typeid (int))
        {
          entry.value.type = CORTEXT_DB_VALUE_INT64;
          entry.value.int64_value = std::any_cast<int> (param);
        }
      else if (param.type () == typeid (long))
        {
          entry.value.type = CORTEXT_DB_VALUE_INT64;
          entry.value.int64_value = std::any_cast<long> (param);
        }
      else if (param.type () == typeid (long long))
        {
          entry.value.type = CORTEXT_DB_VALUE_INT64;
          entry.value.int64_value = std::any_cast<long long> (param);
        }
      else if (param.type () == typeid (double))
        {
          entry.value.type = CORTEXT_DB_VALUE_DOUBLE;
          entry.value.double_value = std::any_cast<double> (param);
        }
      else if (param.type () == typeid (float))
        {
          entry.value.type = CORTEXT_DB_VALUE_DOUBLE;
          entry.value.double_value
              = static_cast<double> (std::any_cast<float> (param));
        }
      else if (param.type () == typeid (std::string))
        {
          entry.text = std::any_cast<std::string> (param);
          entry.value.type = CORTEXT_DB_VALUE_TEXT;
        }
      else if (param.type () == typeid (const char *))
        {
          entry.text = std::any_cast<const char *> (param);
          entry.value.type = CORTEXT_DB_VALUE_TEXT;
        }
      else if (param.type () == typeid (std::vector<char>))
        {
          const auto &bytes = std::any_cast<const std::vector<char> &> (param);
          entry.blob.assign (bytes.begin (), bytes.end ());
          entry.value.type = CORTEXT_DB_VALUE_BLOB;
          entry.value.blob_size = entry.blob.size ();
        }
      else if (param.type () == typeid (std::vector<unsigned char>))
        {
          entry.blob
              = std::any_cast<const std::vector<unsigned char> &> (param);
          entry.value.type = CORTEXT_DB_VALUE_BLOB;
          entry.value.blob_size = entry.blob.size ();
        }
      else if (param.type () == typeid (std::vector<float>))
        {
          const auto &floats
              = std::any_cast<const std::vector<float> &> (param);
          if (!floats.empty ())
            {
              const auto *raw
                  = reinterpret_cast<const unsigned char *> (floats.data ());
              entry.blob.assign (raw, raw + floats.size () * sizeof (float));
            }
          entry.value.type = CORTEXT_DB_VALUE_BLOB;
          entry.value.blob_size = entry.blob.size ();
        }
      else
        {
          entry.value.type = CORTEXT_DB_VALUE_NULL;
        }
      storage.push_back (std::move (entry));
    }

  for (auto &entry : storage)
    {
      if (entry.value.type == CORTEXT_DB_VALUE_TEXT)
        {
          entry.value.text_value = entry.text.c_str ();
        }
      else if (entry.value.type == CORTEXT_DB_VALUE_BLOB)
        {
          entry.value.blob_data = entry.blob.empty () ? nullptr
                                                      : entry.blob.data ();
        }
    }

  std::vector<cortext_db_value> values;
  values.reserve (storage.size ());
  for (const auto &entry : storage)
    {
      values.push_back (entry.value);
    }
  return values;
}

class CallbackStore;

class CallbackTransaction final : public cortext::Transaction
{
public:
  CallbackTransaction (CallbackStore *store, void *transaction) noexcept
      : store_ (store), transaction_ (transaction)
  {
  }

  ~CallbackTransaction () override;

  std::unique_ptr<cortext::Transaction> Begin () override;

  std::vector<std::map<std::string, std::any> >
  Execute (const std::string &query,
           const std::vector<std::any> &params = {}) override;

  void Commit () override;
  void Rollback () override;

private:
  void EnsureActive () const;

  CallbackStore *store_ = nullptr;
  void *transaction_ = nullptr;
  bool finished_ = false;
};

class CallbackStore final : public cortext::Store
{
public:
  CallbackStore (cortext_db_callbacks callbacks, void *user_data)
      : callbacks_ (callbacks), user_data_ (user_data)
  {
    if (!callbacks_.execute || !callbacks_.begin || !callbacks_.commit
        || !callbacks_.rollback || !callbacks_.release_transaction
        || !callbacks_.release_result)
      {
        throw cortext::StoreError ("DB callbacks must all be non-null");
      }
  }

  std::vector<std::map<std::string, std::any> >
  Execute (const std::string &query,
           const std::vector<std::any> &params = {}) override
  {
    return ExecuteInTransaction (CurrentTransaction (), query, params);
  }

  std::unique_ptr<cortext::Transaction> Begin () override
  {
    return BeginNested (CurrentTransaction ());
  }

  void Commit () override
  {
    throw cortext::NoActiveTransactionError (
        "Cannot commit external callback store without a transaction");
  }

  void Rollback () override
  {
    throw cortext::NoActiveTransactionError (
        "Cannot rollback external callback store without a transaction");
  }

  void Close () override {}

  std::unique_ptr<cortext::Transaction> BeginNested (void *parent)
  {
    if (parent)
      {
        EnsureCurrent (parent);
      }
    else if (!transaction_stack_.empty ())
      {
        throw cortext::TransactionNotCurrentError (
            "External callback store already has an active transaction");
      }

    void *transaction = nullptr;
    const char *error = nullptr;
    const int rc
        = callbacks_.begin (user_data_, parent, &transaction, &error);
    if (rc != 0 || !transaction)
      {
        ThrowCallbackError ("begin transaction failed", error);
      }
    transaction_stack_.push_back (transaction);
    return std::make_unique<CallbackTransaction> (this, transaction);
  }

  void EnsureCurrent (void *transaction) const
  {
    if (!transaction || transaction_stack_.empty ()
        || transaction_stack_.back () != transaction)
      {
        throw cortext::TransactionNotCurrentError (
            "External callback transaction is not current");
      }
  }

  void *CurrentTransaction () const
  {
    return transaction_stack_.empty () ? nullptr : transaction_stack_.back ();
  }

  std::vector<std::map<std::string, std::any> >
  ExecuteInTransaction (void *transaction, const std::string &query,
                        const std::vector<std::any> &params)
  {
    std::vector<CallbackParam> param_storage;
    const auto param_values = params_to_values (params, param_storage);

    cortext_db_result result{};
    const char *error = nullptr;
    const int rc = callbacks_.execute (
        user_data_, transaction, query.c_str (),
        param_values.empty () ? nullptr : param_values.data (),
        param_values.size (), &result, &error);
    if (rc != 0)
      {
        callbacks_.release_result (user_data_, &result);
        ThrowCallbackError ("execute failed", error);
      }

    std::vector<std::map<std::string, std::any> > rows;
    rows.reserve (result.row_count);
    for (size_t i = 0; i < result.row_count; ++i)
      {
        std::map<std::string, std::any> row;
        const cortext_db_row &source_row = result.rows[i];
        for (size_t j = 0; j < source_row.cell_count; ++j)
          {
            const cortext_db_cell &cell = source_row.cells[j];
            if (cell.column_name)
              {
                row.emplace (cell.column_name, value_to_any (cell.value));
              }
          }
        rows.push_back (std::move (row));
      }

    callbacks_.release_result (user_data_, &result);
    return rows;
  }

  void CommitTransaction (void *transaction)
  {
    EnsureCurrent (transaction);
    const char *error = nullptr;
    const int rc = callbacks_.commit (user_data_, transaction, &error);
    if (rc != 0)
      {
        ThrowCallbackError ("commit failed", error);
      }
    transaction_stack_.pop_back ();
  }

  void RollbackTransaction (void *transaction)
  {
    EnsureCurrent (transaction);
    const char *error = nullptr;
    const int rc = callbacks_.rollback (user_data_, transaction, &error);
    if (rc != 0)
      {
        ThrowCallbackError ("rollback failed", error);
      }
    transaction_stack_.pop_back ();
  }

  void ReleaseTransaction (void *transaction) noexcept
  {
    callbacks_.release_transaction (user_data_, transaction);
  }

  void RollbackTransactionStackFrom (void *transaction) noexcept
  {
    auto it = std::find (transaction_stack_.begin (),
                         transaction_stack_.end (), transaction);
    if (it == transaction_stack_.end ())
      {
        return;
      }

    const auto first_index = static_cast<std::size_t> (
        std::distance (transaction_stack_.begin (), it));
    for (std::size_t index = transaction_stack_.size (); index > first_index;
         --index)
      {
        try
          {
            const char *error = nullptr;
            (void)callbacks_.rollback (
                user_data_, transaction_stack_[index - 1], &error);
          }
        catch (...)
          {
          }
      }
    transaction_stack_.erase (it, transaction_stack_.end ());
  }

private:
  [[noreturn]] void ThrowCallbackError (const char *prefix,
                                        const char *error) const
  {
    std::string message = prefix;
    if (error && error[0] != '\0')
      {
        message += ": ";
        message += error;
      }
    throw cortext::StoreError (message);
  }

  cortext_db_callbacks callbacks_{};
  void *user_data_ = nullptr;
  std::vector<void *> transaction_stack_;
};

CallbackTransaction::~CallbackTransaction ()
{
  if (store_ && transaction_)
    {
      if (!finished_)
        {
          store_->RollbackTransactionStackFrom (transaction_);
        }
      store_->ReleaseTransaction (transaction_);
    }
}

std::unique_ptr<cortext::Transaction>
CallbackTransaction::Begin ()
{
  EnsureActive ();
  store_->EnsureCurrent (transaction_);
  return store_->BeginNested (transaction_);
}

std::vector<std::map<std::string, std::any> >
CallbackTransaction::Execute (const std::string &query,
                              const std::vector<std::any> &params)
{
  EnsureActive ();
  store_->EnsureCurrent (transaction_);
  return store_->ExecuteInTransaction (transaction_, query, params);
}

void
CallbackTransaction::EnsureActive () const
{
  if (finished_)
    {
      throw cortext::TransactionAlreadyFinishedError (
          "External callback transaction is already finished");
    }
}

void
CallbackTransaction::Commit ()
{
  EnsureActive ();
  store_->CommitTransaction (transaction_);
  finished_ = true;
}

void
CallbackTransaction::Rollback ()
{
  EnsureActive ();
  store_->RollbackTransaction (transaction_);
  finished_ = true;
}

class CallbackObjectStore;

class CallbackObjectTransaction final : public cortext::ObjectTransaction
{
public:
  CallbackObjectTransaction (CallbackObjectStore *store, void *transaction)
      : store_ (store), transaction_ (transaction)
  {
  }

  ~CallbackObjectTransaction () override;

  std::unique_ptr<cortext::ObjectTransaction> Begin () override;
  cortext::ObjectId Put (const std::vector<unsigned char> &data) override;
  std::optional<std::vector<unsigned char>>
  Get (const cortext::ObjectId &id) override;
  bool Exists (const cortext::ObjectId &id) override;
  bool Delete (const cortext::ObjectId &id) override;
  void Commit () override;
  void Rollback () override;

private:
  void EnsureActive () const;

  CallbackObjectStore *store_ = nullptr;
  void *transaction_ = nullptr;
  bool finished_ = false;
};

class CallbackObjectStore final : public cortext::ObjectStore
{
public:
  CallbackObjectStore (cortext_object_callbacks callbacks, void *user_data)
      : callbacks_ (callbacks), user_data_ (user_data)
  {
    if (!callbacks_.begin || !callbacks_.put || !callbacks_.get
        || !callbacks_.exists || !callbacks_.delete_object
        || !callbacks_.commit || !callbacks_.rollback
        || !callbacks_.release_transaction || !callbacks_.release_blob)
      {
        throw cortext::StoreError ("Object callbacks must all be non-null");
      }
  }

  std::unique_ptr<cortext::ObjectTransaction> Begin () override
  {
    return BeginNested (CurrentTransaction ());
  }

  void Close () override {}

  std::unique_ptr<cortext::ObjectTransaction> BeginNested (void *parent)
  {
    if (parent)
      {
        EnsureCurrent (parent);
      }
    else if (!transaction_stack_.empty ())
      {
        throw cortext::TransactionNotCurrentError (
            "External object store already has an active transaction");
      }

    void *transaction = nullptr;
    const char *error = nullptr;
    const int rc
        = callbacks_.begin (user_data_, parent, &transaction, &error);
    if (rc != 0 || !transaction)
      {
        ThrowCallbackError ("object begin failed", error);
      }
    transaction_stack_.push_back (transaction);
    return std::make_unique<CallbackObjectTransaction> (this, transaction);
  }

  void EnsureCurrent (void *transaction) const
  {
    if (!transaction || transaction_stack_.empty ()
        || transaction_stack_.back () != transaction)
      {
        throw cortext::TransactionNotCurrentError (
            "External object transaction is not current");
      }
  }

  void *CurrentTransaction () const
  {
    return transaction_stack_.empty () ? nullptr : transaction_stack_.back ();
  }

  void PutObject (void *transaction, const cortext::ObjectId &id,
                  const std::vector<unsigned char> &data)
  {
    EnsureCurrent (transaction);
    const char *error = nullptr;
    const int rc = callbacks_.put (
        user_data_, transaction, id.empty () ? nullptr : id.data (),
        id.size (), data.empty () ? nullptr : data.data (), data.size (),
        &error);
    if (rc != 0)
      {
        ThrowCallbackError ("object put failed", error);
      }
  }

  std::optional<std::vector<unsigned char>>
  GetObject (void *transaction, const cortext::ObjectId &id)
  {
    EnsureCurrent (transaction);
    const uint8_t *data = nullptr;
    size_t data_size = 0;
    const char *error = nullptr;
    const int rc = callbacks_.get (
        user_data_, transaction, id.empty () ? nullptr : id.data (), id.size (),
        &data, &data_size, &error);
    if (rc != 0)
      {
        ThrowCallbackError ("object get failed", error);
      }
    if (!data && data_size == 0)
      {
        return std::nullopt;
      }
    std::vector<unsigned char> out;
    if (data && data_size > 0)
      {
        out.assign (data, data + data_size);
      }
    callbacks_.release_blob (user_data_, data, data_size);
    return out;
  }

  bool ExistsObject (void *transaction, const cortext::ObjectId &id)
  {
    EnsureCurrent (transaction);
    int exists = 0;
    const char *error = nullptr;
    const int rc = callbacks_.exists (
        user_data_, transaction, id.empty () ? nullptr : id.data (), id.size (),
        &exists, &error);
    if (rc != 0)
      {
        ThrowCallbackError ("object exists failed", error);
      }
    return exists != 0;
  }

  bool DeleteObject (void *transaction, const cortext::ObjectId &id)
  {
    EnsureCurrent (transaction);
    int deleted = 0;
    const char *error = nullptr;
    const int rc = callbacks_.delete_object (
        user_data_, transaction, id.empty () ? nullptr : id.data (), id.size (),
        &deleted, &error);
    if (rc != 0)
      {
        ThrowCallbackError ("object delete failed", error);
      }
    return deleted != 0;
  }

  void CommitTransaction (void *transaction)
  {
    EnsureCurrent (transaction);
    const char *error = nullptr;
    const int rc = callbacks_.commit (user_data_, transaction, &error);
    if (rc != 0)
      {
        ThrowCallbackError ("object commit failed", error);
      }
    transaction_stack_.pop_back ();
  }

  void RollbackTransaction (void *transaction)
  {
    EnsureCurrent (transaction);
    const char *error = nullptr;
    const int rc = callbacks_.rollback (user_data_, transaction, &error);
    if (rc != 0)
      {
        ThrowCallbackError ("object rollback failed", error);
      }
    transaction_stack_.pop_back ();
  }

  void ReleaseTransaction (void *transaction) noexcept
  {
    callbacks_.release_transaction (user_data_, transaction);
  }

  void RollbackTransactionStackFrom (void *transaction) noexcept
  {
    auto it = std::find (transaction_stack_.begin (),
                         transaction_stack_.end (), transaction);
    if (it == transaction_stack_.end ())
      {
        return;
      }

    const auto first_index = static_cast<std::size_t> (
        std::distance (transaction_stack_.begin (), it));
    for (std::size_t index = transaction_stack_.size (); index > first_index;
         --index)
      {
        try
          {
            const char *error = nullptr;
            (void)callbacks_.rollback (
                user_data_, transaction_stack_[index - 1], &error);
          }
        catch (...)
          {
          }
      }
    transaction_stack_.erase (it, transaction_stack_.end ());
  }

private:
  [[noreturn]] void ThrowCallbackError (const char *prefix,
                                        const char *error) const
  {
    std::string message = prefix;
    if (error && error[0] != '\0')
      {
        message += ": ";
        message += error;
      }
    throw cortext::StoreError (message);
  }

  cortext_object_callbacks callbacks_{};
  void *user_data_ = nullptr;
  std::vector<void *> transaction_stack_;
};

CallbackObjectTransaction::~CallbackObjectTransaction ()
{
  if (store_ && transaction_)
    {
      if (!finished_)
        {
          store_->RollbackTransactionStackFrom (transaction_);
        }
      store_->ReleaseTransaction (transaction_);
    }
}

void
CallbackObjectTransaction::EnsureActive () const
{
  if (finished_)
    {
      throw cortext::TransactionAlreadyFinishedError (
          "External object transaction is already finished");
    }
}

std::unique_ptr<cortext::ObjectTransaction>
CallbackObjectTransaction::Begin ()
{
  EnsureActive ();
  return store_->BeginNested (transaction_);
}

cortext::ObjectId
CallbackObjectTransaction::Put (const std::vector<unsigned char> &data)
{
  EnsureActive ();
  auto id = cortext::ComputeObjectId (data);
  store_->PutObject (transaction_, id, data);
  return id;
}

std::optional<std::vector<unsigned char>>
CallbackObjectTransaction::Get (const cortext::ObjectId &id)
{
  EnsureActive ();
  return store_->GetObject (transaction_, id);
}

bool
CallbackObjectTransaction::Exists (const cortext::ObjectId &id)
{
  EnsureActive ();
  return store_->ExistsObject (transaction_, id);
}

bool
CallbackObjectTransaction::Delete (const cortext::ObjectId &id)
{
  EnsureActive ();
  return store_->DeleteObject (transaction_, id);
}

void
CallbackObjectTransaction::Commit ()
{
  EnsureActive ();
  store_->CommitTransaction (transaction_);
  finished_ = true;
}

void
CallbackObjectTransaction::Rollback ()
{
  EnsureActive ();
  store_->RollbackTransaction (transaction_);
  finished_ = true;
}

char *
copy_string_result (const std::string &value)
{
  auto *buf = static_cast<char *> (std::malloc (value.size () + 1));
  if (!buf)
    {
      set_last_error ("out of memory while allocating string result");
      return nullptr;
    }
  std::memcpy (buf, value.c_str (), value.size () + 1);
  return buf;
}

std::string
encode_base64 (const std::vector<unsigned char> &data)
{
  static constexpr char kAlphabet[]
      = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  std::string out;
  out.reserve (((data.size () + 2) / 3) * 4);

  for (size_t i = 0; i < data.size (); i += 3)
    {
      const unsigned int octet_a = data[i];
      const unsigned int octet_b = (i + 1 < data.size ()) ? data[i + 1] : 0;
      const unsigned int octet_c = (i + 2 < data.size ()) ? data[i + 2] : 0;
      const unsigned int triple = (octet_a << 16) | (octet_b << 8) | octet_c;

      out.push_back (kAlphabet[(triple >> 18) & 0x3f]);
      out.push_back (kAlphabet[(triple >> 12) & 0x3f]);
      out.push_back ((i + 1 < data.size ()) ? kAlphabet[(triple >> 6) & 0x3f]
                                            : '=');
      out.push_back ((i + 2 < data.size ()) ? kAlphabet[triple & 0x3f] : '=');
    }

  return out;
}

nlohmann::json
memory_to_json (const cortext::Cortext::Context::Memory &memory)
{
  nlohmann::json content = nlohmann::json::array ();
  for (const auto &blob : memory.content)
    {
      content.push_back ({
          { "size_bytes", blob.size () },
          { "base64", encode_base64 (blob) },
      });
    }

  nlohmann::json soft_anchors = nlohmann::json::array ();
  for (const auto &anchor : memory.soft_anchors)
    {
      soft_anchors.push_back ({
          { "id", anchor.id },
          { "strength", anchor.strength },
          { "likelihood", anchor.likelihood },
          { "label", anchor.label },
          { "tier", anchor.tier },
          { "score", anchor.score },
          { "margin", anchor.margin },
          { "entropy", anchor.entropy },
      });
    }

  return {
    { "source_id", memory.source_id },
    { "id", memory.id },
    { "timestamp", memory.timestamp },
    { "retrieved_count", memory.retrieved_count },
    { "used_count", memory.used_count },
    { "modality", memory.modality },
    { "mimetype", memory.mimetype },
    { "content", std::move (content) },
    { "relevance", memory.relevance },
    { "mismatch", memory.mismatch },
    { "surprise", memory.surprise },
    { "rarity", memory.rarity },
    { "drift", memory.drift },
    { "contradiction", memory.contradiction },
    { "utility", memory.utility },
    { "periphery", memory.periphery },
    { "coverage", memory.coverage },
    { "salience", memory.salience },
    { "valence", memory.valence },
    { "arousal", memory.arousal },
    { "composite_score", memory.composite_score },
    { "threshold_t", memory.threshold_t },
    { "soft_anchors", std::move (soft_anchors) },
  };
}

nlohmann::json
context_to_json (const cortext::Cortext::Context &ctx,
                 bool include_embedding)
{
  nlohmann::json working_memory = nlohmann::json::array ();
  for (const auto &memory : ctx.working_memory)
    {
      working_memory.push_back (memory_to_json (memory));
    }

  nlohmann::json retrieved_memory = nlohmann::json::array ();
  for (const auto &memory : ctx.retrieved_memory)
    {
      retrieved_memory.push_back (memory_to_json (memory));
    }

  nlohmann::json metrics = nlohmann::json::object ();
  for (const auto &[metric_id, value] : ctx.output.metrics)
    {
      metrics[std::to_string (metric_id)] = value;
    }

  nlohmann::json operation_ms = nlohmann::json::object ();
  for (const auto &[name, value] : ctx.output.operation_ms)
    {
      operation_ms[name] = value;
    }

  nlohmann::json output = {
    { "effective_focus", ctx.output.effective_focus },
    { "coherence", ctx.output.coherence },
    { "emotion_intensity", ctx.output.emotion_intensity },
    { "valence", ctx.output.valence },
    { "arousal", ctx.output.arousal },
    { "metrics", std::move (metrics) },
    { "operation_ms", std::move (operation_ms) },
    { "composite_score",
      ctx.output.composite_score.has_value ()
          ? nlohmann::json (*ctx.output.composite_score)
          : nlohmann::json (nullptr) },
    { "threshold",
      ctx.output.threshold.has_value () ? nlohmann::json (*ctx.output.threshold)
                                        : nlohmann::json (nullptr) },
    { "decision",
      ctx.output.decision.has_value () ? nlohmann::json (*ctx.output.decision)
                                       : nlohmann::json (nullptr) },
    { "stored_embedding_id",
      ctx.output.stored_embedding_id.has_value ()
          ? nlohmann::json (*ctx.output.stored_embedding_id)
          : nlohmann::json (nullptr) },
    { "stored_memory_id",
      ctx.output.stored_memory_id.has_value ()
          ? nlohmann::json (*ctx.output.stored_memory_id)
          : nlohmann::json (nullptr) },
    { "stored_signal_id",
      ctx.output.stored_signal_id.has_value ()
          ? nlohmann::json (*ctx.output.stored_signal_id)
          : nlohmann::json (nullptr) },
    { "signal_filter_evaluated", ctx.output.signal_filter_evaluated },
    { "signal_filter_accepted", ctx.output.signal_filter_accepted },
    { "signal_filter_modality", ctx.output.signal_filter_modality },
    { "signal_filter_reason", ctx.output.signal_filter_reason },
    { "signal_filter_score", ctx.output.signal_filter_score },
    { "signal_filter_threshold", ctx.output.signal_filter_threshold },
    { "signal_filter_quiet_seconds",
      ctx.output.signal_filter_quiet_seconds },
    { "soft_anchor_enabled", ctx.output.soft_anchor_enabled },
    { "soft_anchor_state_count", ctx.output.soft_anchor_state_count },
    { "soft_anchor_link_count", ctx.output.soft_anchor_link_count },
    { "soft_anchor_create_count", ctx.output.soft_anchor_create_count },
    { "soft_anchor_update_count", ctx.output.soft_anchor_update_count },
    { "soft_anchor_none_count", ctx.output.soft_anchor_none_count },
    { "soft_anchor_last_update_us",
      ctx.output.soft_anchor_last_update_us },
    { "soft_anchor_mean_update_us",
      ctx.output.soft_anchor_mean_update_us },
  };

  nlohmann::json result = {
    { "working_memory", std::move (working_memory) },
    { "retrieved_memory", std::move (retrieved_memory) },
    { "should_interrupt", ctx.should_interrupt },
    { "interrupt_aborted", ctx.interrupt_aborted },
    { "at_boundary", ctx.at_boundary },
    { "consolidation_recommended", ctx.consolidation_recommended },
    { "consolidation_required", ctx.consolidation_required },
    { "interrupt_gate_has_candidates", ctx.interrupt_gate_has_candidates },
    { "interrupt_gate_blocked_no_store", ctx.interrupt_gate_blocked_no_store },
    { "interrupt_gate_rel_pass", ctx.interrupt_gate_rel_pass },
    { "interrupt_gate_novelty_pass", ctx.interrupt_gate_novelty_pass },
    { "interrupt_gate_mu_pass", ctx.interrupt_gate_mu_pass },
    { "interrupt_gate_novelty_mu_pass", ctx.interrupt_gate_novelty_mu_pass },
    { "interrupt_gate_dup_pass", ctx.interrupt_gate_dup_pass },
    { "interrupt_gate_boundary_mu_pass", ctx.interrupt_gate_boundary_mu_pass },
    { "interrupt_gate_rel_star", ctx.interrupt_gate_rel_star },
    { "interrupt_gate_retrieval_thresh", ctx.interrupt_gate_retrieval_thresh },
    { "interrupt_gate_boundary_mult_eff", ctx.interrupt_gate_boundary_mult_eff },
    { "interrupt_gate_affect_drive", ctx.interrupt_gate_affect_drive },
    { "boundary_score",
      ctx.boundary_score.has_value () ? nlohmann::json (*ctx.boundary_score)
                                      : nlohmann::json (nullptr) },
    { "boundary_type",
      ctx.boundary_type.has_value () ? nlohmann::json (*ctx.boundary_type)
                                     : nlohmann::json (nullptr) },
    { "output", std::move (output) },
    { "encode_ms", ctx.encode_ms },
    { "process_ms", ctx.process_ms },
    { "hydrate_ms", ctx.hydrate_ms },
    { "total_ms", ctx.total_ms },
  };

  if (include_embedding)
    {
      result["embedding"] = ctx.embedding;
      result["embedding_dimension"] = ctx.embedding.size ();
    }

  return result;
}

char *
context_result_to_json (const cortext::Cortext::Context &ctx,
                        bool include_embedding = true)
{
  clear_last_error ();
  // Memory text can carry invalid UTF-8 from raw sources; replace with
  // U+FFFD rather than throwing across the C boundary.
  return copy_string_result (context_to_json (ctx, include_embedding).dump (
      -1, ' ', false, nlohmann::json::error_handler_t::replace));
}

char *
embedding_result_to_json (const std::vector<float> &embedding)
{
  clear_last_error ();
  return copy_string_result (
      nlohmann::json {
          { "embedding", embedding },
          { "dimension", embedding.size () },
      }
          .dump ());
}

template <typename Fn>
int
invoke_status_only (Fn &&fn)
{
  try
    {
      clear_last_error ();
      fn ();
      return 0;
    }
  catch (const std::exception &ex)
    {
      set_last_error (ex.what ());
      return 2;
    }
  catch (...)
    {
      set_last_error ("internal error");
      return 2;
    }
}

template <typename Fn>
char *
invoke_json (Fn &&fn, bool include_embedding = true)
{
  try
    {
      return context_result_to_json (fn (), include_embedding);
    }
  catch (const std::exception &ex)
    {
      set_last_error (ex.what ());
      return nullptr;
    }
  catch (...)
    {
      set_last_error ("internal error");
      return nullptr;
    }
}

template <typename Fn>
char *
invoke_embedding_json (Fn &&fn)
{
  try
    {
      return embedding_result_to_json (fn ());
    }
  catch (const std::exception &ex)
    {
      set_last_error (ex.what ());
      return nullptr;
    }
  catch (...)
    {
      set_last_error ("internal error");
      return nullptr;
    }
}
} // namespace

extern "C"
{

  static cortext::Cortext *
  cast_handle (cortext_handle h)
  {
    return reinterpret_cast<cortext::Cortext *> (h);
  }

  cortext_handle
  cortext_create (double focus, double sensitivity, double stability,
                  const char *db_path)
  {
    if (!db_path)
      {
        set_last_error ("db_path must not be NULL");
        return nullptr;
      }

    cortext_config cfg{};
    cortext_config_init (&cfg);
    cfg.focus = focus;
    cfg.sensitivity = sensitivity;
    cfg.stability = stability;
    return cortext_create_with_config (&cfg, db_path);
  }

  void
  cortext_config_init (cortext_config *cfg)
  {
    if (!cfg)
      {
        set_last_error ("config pointer must not be NULL");
        return;
      }

    // Callers built against an older (smaller) cortext_config may set
    // struct_size before calling init; writes are clipped to that size so
    // the library never scribbles past the caller's allocation. A zero or
    // implausible struct_size selects the current full layout.
    std::size_t available = sizeof (cortext_config);
    if (cfg->struct_size >= offsetof (cortext_config, focus)
        && cfg->struct_size <= sizeof (cortext_config))
      available = cfg->struct_size;
    const auto field_fits = [&] (std::size_t offset, std::size_t size) {
      return offset + size <= available;
    };

    const auto defaults = default_config ();
    cfg->struct_size = available;
    cfg->focus = defaults.focus;
    cfg->sensitivity = defaults.sensitivity;
    cfg->stability = defaults.stability;
    cfg->affect_interrupt = defaults.affect_interrupt ? 1 : 0;
    cfg->affect_retrieval = defaults.affect_retrieval ? 1 : 0;
    cfg->reinforcement_enabled = defaults.reinforcement_enabled ? 1 : 0;
    cfg->procedural_enabled = defaults.procedural_enabled ? 1 : 0;
    cfg->sequential_edges_enabled = defaults.sequential_edges_enabled ? 1 : 0;
    if (field_fits (offsetof (cortext_config, signal_filter_audio_enabled),
                    sizeof (cfg->signal_filter_audio_enabled)))
      cfg->signal_filter_audio_enabled
          = defaults.signal_filter_audio_enabled ? 1 : 0;
    if (field_fits (offsetof (cortext_config, signal_filter_image_enabled),
                    sizeof (cfg->signal_filter_image_enabled)))
      cfg->signal_filter_image_enabled
          = defaults.signal_filter_image_enabled ? 1 : 0;
    if (field_fits (offsetof (cortext_config, signal_filter_text_enabled),
                    sizeof (cfg->signal_filter_text_enabled)))
      cfg->signal_filter_text_enabled
          = defaults.signal_filter_text_enabled ? 1 : 0;
    clear_last_error ();
  }

  void
  cortext_process_json_options_init (cortext_process_json_options *options)
  {
    if (!options)
      {
        return;
      }
    // A caller compiled against the previous struct only allocated the
    // prefix through include_embedding. It must set struct_size to opt into
    // initialization of later fields; a zero size stays ABI-safe by using
    // that legacy prefix.
    constexpr std::size_t kLegacyOptionsSize
        = offsetof (cortext_process_json_options, retention);
    const std::size_t available
        = options->struct_size == sizeof (*options) ? sizeof (*options)
                                                  : kLegacyOptionsSize;
    options->struct_size = available;
    options->include_embedding = 1;
    if (available == sizeof (*options))
      {
        options->retention = CORTEXT_RETENTION_NATURAL;
        options->reserved = 0;
      }
  }

  cortext_handle
  cortext_create_with_config (const cortext_config *cfg, const char *db_path)
  {
    if (!db_path)
      {
        set_last_error ("db_path must not be NULL");
        return nullptr;
      }

    try
      {
        auto inst = cortext::Cortext::Create (
            config_from_c (cfg), std::string (db_path));
        clear_last_error ();
        return reinterpret_cast<cortext_handle> (inst.release ());
      }
    catch (const std::exception &ex)
      {
        set_last_error (ex.what ());
        return nullptr;
      }
    catch (...)
      {
        set_last_error ("internal error");
        return nullptr;
      }
  }

  cortext_handle
  cortext_create_with_store_callbacks (
      const cortext_config *cfg, const cortext_db_callbacks *callbacks,
      void *user_data)
  {
    if (!callbacks)
      {
        set_last_error ("callbacks must not be NULL");
        return nullptr;
      }

    if (callbacks->struct_size < sizeof (cortext_db_callbacks))
      {
        set_last_error ("callbacks struct_size is invalid");
        return nullptr;
      }

    try
      {
        auto store = std::make_shared<CallbackStore> (*callbacks, user_data);
        auto inst = cortext::Cortext::Create (
            config_from_c (cfg), std::move (store));
        clear_last_error ();
        return reinterpret_cast<cortext_handle> (inst.release ());
      }
    catch (const std::exception &ex)
      {
        set_last_error (ex.what ());
        return nullptr;
      }
    catch (...)
      {
        set_last_error ("internal error");
        return nullptr;
      }
  }

  cortext_handle
  cortext_create_with_store_and_object_callbacks (
      const cortext_config *cfg, const cortext_db_callbacks *db_callbacks,
      void *db_user_data, const cortext_object_callbacks *object_callbacks,
      void *object_user_data)
  {
    if (!db_callbacks)
      {
        set_last_error ("db_callbacks must not be NULL");
        return nullptr;
      }
    if (!object_callbacks)
      {
        set_last_error ("object_callbacks must not be NULL");
        return nullptr;
      }
    if (db_callbacks->struct_size < sizeof (cortext_db_callbacks))
      {
        set_last_error ("db_callbacks struct_size is invalid");
        return nullptr;
      }
    if (object_callbacks->struct_size < sizeof (cortext_object_callbacks))
      {
        set_last_error ("object_callbacks struct_size is invalid");
        return nullptr;
      }

    try
      {
        auto store
            = std::make_shared<CallbackStore> (*db_callbacks, db_user_data);
        auto object_store = std::make_shared<CallbackObjectStore> (
            *object_callbacks, object_user_data);
        auto inst = cortext::Cortext::Create (
            config_from_c (cfg), std::move (store), std::move (object_store),
            std::shared_ptr<cortext::Clock>{});
        clear_last_error ();
        return reinterpret_cast<cortext_handle> (inst.release ());
      }
    catch (const std::exception &ex)
      {
        set_last_error (ex.what ());
        return nullptr;
      }
    catch (...)
      {
        set_last_error ("internal error");
        return nullptr;
      }
  }

  cortext_handle
  cortext_create_with_config_and_object_callbacks (
      const cortext_config *cfg, const char *db_path,
      const cortext_object_callbacks *object_callbacks, void *object_user_data)
  {
    if (!db_path)
      {
        set_last_error ("db_path must not be NULL");
        return nullptr;
      }
    if (!object_callbacks)
      {
        set_last_error ("object_callbacks must not be NULL");
        return nullptr;
      }
    if (object_callbacks->struct_size < sizeof (cortext_object_callbacks))
      {
        set_last_error ("object_callbacks struct_size is invalid");
        return nullptr;
      }

    try
      {
        auto object_store = std::make_shared<CallbackObjectStore> (
            *object_callbacks, object_user_data);
        auto inst = cortext::Cortext::Create (
            config_from_c (cfg), std::string (db_path),
            std::move (object_store));
        clear_last_error ();
        return reinterpret_cast<cortext_handle> (inst.release ());
      }
    catch (const std::exception &ex)
      {
        set_last_error (ex.what ());
        return nullptr;
      }
    catch (...)
      {
        set_last_error ("internal error");
        return nullptr;
      }
  }

  void
  cortext_free (cortext_handle h)
  {
    if (!h)
      {
        return;
      }
    auto *p = cast_handle (h);
    delete p;
  }

  int
  cortext_process_text (cortext_handle h, const char *text,
                        const char *source_id)
  {
    auto *p = cast_handle (h);
    if (!p || !text || !source_id)
      {
        set_last_error ("handle, text, and source_id must all be non-NULL");
        return 1;
      }

    return invoke_status_only (
        [&] { (void)p->ProcessText (std::string (text), std::string (source_id),
                                    cortext::Retention::Durable); });
  }

  int
  cortext_process_audio (cortext_handle h, const float *pcm,
                         size_t num_samples, const char *source_id)
  {
    auto *p = cast_handle (h);
    if (!p || !pcm || !source_id)
      {
        set_last_error ("handle, pcm, and source_id must all be non-NULL");
        return 1;
      }

    return invoke_status_only (
        [&] { (void)p->ProcessAudio (pcm, num_samples, std::string (source_id),
                                     cortext::Retention::Durable); });
  }

  int
  cortext_process_audio_with_media (cortext_handle h, const float *pcm,
                                    size_t num_samples,
                                    const char *source_id,
                                    const cortext_media *media)
  {
    auto *p = cast_handle (h);
    if (!p || !pcm || !source_id)
      {
        set_last_error ("handle, pcm, and source_id must all be non-NULL");
        return 1;
      }
    if (!valid_media_or_set_error (media))
      {
        return 1;
      }

    return invoke_status_only ([&] {
      (void)p->ProcessAudio (pcm, num_samples, std::string (source_id),
                             media_from_c (media), cortext::Retention::Durable);
    });
  }

  int
  cortext_process_image (cortext_handle h, const uint8_t *data, int width,
                         int height, int channels, const char *source_id)
  {
    auto *p = cast_handle (h);
    if (!p || !data || !source_id)
      {
        set_last_error ("handle, data, and source_id must all be non-NULL");
        return 1;
      }

    return invoke_status_only ([&] {
      (void)p->ProcessImage (data, width, height, channels,
                             std::string (source_id), cortext::Retention::Durable);
    });
  }

  int
  cortext_process_image_with_media (cortext_handle h, const uint8_t *data,
                                    int width, int height, int channels,
                                    const char *source_id,
                                    const cortext_media *media)
  {
    auto *p = cast_handle (h);
    if (!p || !data || !source_id)
      {
        set_last_error ("handle, data, and source_id must all be non-NULL");
        return 1;
      }
    if (!valid_media_or_set_error (media))
      {
        return 1;
      }

    return invoke_status_only ([&] {
      (void)p->ProcessImage (data, width, height, channels,
                             std::string (source_id), media_from_c (media),
                             cortext::Retention::Durable);
    });
  }

  char *
  cortext_embed_text_json (cortext_handle h, const char *text)
  {
    auto *p = cast_handle (h);
    if (!p || !text)
      {
        set_last_error ("handle and text must both be non-NULL");
        return nullptr;
      }

    return invoke_embedding_json (
        [&] { return p->EmbedText (std::string (text)); });
  }

  char *
  cortext_embed_audio_json (cortext_handle h, const float *pcm,
                            size_t num_samples)
  {
    auto *p = cast_handle (h);
    if (!p || !pcm)
      {
        set_last_error ("handle and pcm must both be non-NULL");
        return nullptr;
      }

    return invoke_embedding_json (
        [&] { return p->EmbedAudio (pcm, num_samples); });
  }

  char *
  cortext_embed_image_json (cortext_handle h, const uint8_t *data, int width,
                            int height, int channels)
  {
    auto *p = cast_handle (h);
    if (!p || !data)
      {
        set_last_error ("handle and data must both be non-NULL");
        return nullptr;
      }

    return invoke_embedding_json (
        [&] { return p->EmbedImage (data, width, height, channels); });
  }

  int
  cortext_consolidate (cortext_handle h)
  {
    auto *p = cast_handle (h);
    if (!p)
      {
        set_last_error ("handle must not be NULL");
        return 1;
      }

    return invoke_status_only ([&] { (void)p->Consolidate (); });
  }

  int
  cortext_flush (cortext_handle h)
  {
    auto *p = cast_handle (h);
    if (!p)
      {
        set_last_error ("handle must not be NULL");
        return 1;
      }

    return invoke_status_only ([&] { p->Flush (); });
  }

  int
  cortext_reset (cortext_handle h)
  {
    auto *p = cast_handle (h);
    if (!p)
      {
        set_last_error ("handle must not be NULL");
        return 1;
      }

    return invoke_status_only ([&] { p->Reset (); });
  }

  const char *
  cortext_version (void)
  {
    return CORTEXT_VERSION;
  }

  const char *
  cortext_last_error (void)
  {
    return g_last_error.empty () ? nullptr : g_last_error.c_str ();
  }

  void
  cortext_string_free (char *value)
  {
    std::free (value);
  }

  char *
  cortext_process_text_json (cortext_handle h, const char *text,
                             const char *source_id)
  {
    return cortext_process_text_json_with_options (h, text, source_id,
                                                   nullptr);
  }

  char *
  cortext_process_text_json_with_options (
      cortext_handle h, const char *text, const char *source_id,
      const cortext_process_json_options *options)
  {
    auto *p = cast_handle (h);
    if (!p || !text || !source_id)
      {
        set_last_error ("handle, text, and source_id must all be non-NULL");
        return nullptr;
      }

    const auto retention = retention_from_options (options);
    return invoke_json (
        [&] {
          return p->ProcessText (std::string (text), std::string (source_id),
                                 retention);
        },
        include_embedding_from_options (options));
  }

  char *
  cortext_process_audio_json (cortext_handle h, const float *pcm,
                              size_t num_samples, const char *source_id)
  {
    return cortext_process_audio_json_with_options (h, pcm, num_samples,
                                                    source_id, nullptr);
  }

  char *
  cortext_process_audio_json_with_options (
      cortext_handle h, const float *pcm, size_t num_samples,
      const char *source_id, const cortext_process_json_options *options)
  {
    auto *p = cast_handle (h);
    if (!p || !pcm || !source_id)
      {
        set_last_error ("handle, pcm, and source_id must all be non-NULL");
        return nullptr;
      }

    const auto retention = retention_from_options (options);
    return invoke_json (
        [&] {
          return p->ProcessAudio (pcm, num_samples, std::string (source_id),
                                  retention);
        },
        include_embedding_from_options (options));
  }

  char *
  cortext_process_audio_with_media_json (cortext_handle h, const float *pcm,
                                         size_t num_samples,
                                         const char *source_id,
                                         const cortext_media *media)
  {
    return cortext_process_audio_with_media_json_with_options (
        h, pcm, num_samples, source_id, media, nullptr);
  }

  char *
  cortext_process_audio_with_media_json_with_options (
      cortext_handle h, const float *pcm, size_t num_samples,
      const char *source_id, const cortext_media *media,
      const cortext_process_json_options *options)
  {
    auto *p = cast_handle (h);
    if (!p || !pcm || !source_id)
      {
        set_last_error ("handle, pcm, and source_id must all be non-NULL");
        return nullptr;
      }
    if (!valid_media_or_set_error (media))
      {
        return nullptr;
      }

    const auto retention = retention_from_options (options);
    return invoke_json ([&] {
      return p->ProcessAudio (pcm, num_samples, std::string (source_id),
                              media_from_c (media), retention);
    },
                        include_embedding_from_options (options));
  }

  char *
  cortext_process_image_json (cortext_handle h, const uint8_t *data, int width,
                              int height, int channels,
                              const char *source_id)
  {
    return cortext_process_image_json_with_options (
        h, data, width, height, channels, source_id, nullptr);
  }

  char *
  cortext_process_image_json_with_options (
      cortext_handle h, const uint8_t *data, int width, int height,
      int channels, const char *source_id,
      const cortext_process_json_options *options)
  {
    auto *p = cast_handle (h);
    if (!p || !data || !source_id)
      {
        set_last_error ("handle, data, and source_id must all be non-NULL");
        return nullptr;
      }

    const auto retention = retention_from_options (options);
    return invoke_json ([&] {
      return p->ProcessImage (data, width, height, channels,
                              std::string (source_id), retention);
    },
                        include_embedding_from_options (options));
  }

  char *
  cortext_process_image_with_media_json (cortext_handle h,
                                         const uint8_t *data, int width,
                                         int height, int channels,
                                         const char *source_id,
                                         const cortext_media *media)
  {
    return cortext_process_image_with_media_json_with_options (
        h, data, width, height, channels, source_id, media, nullptr);
  }

  char *
  cortext_process_image_with_media_json_with_options (
      cortext_handle h, const uint8_t *data, int width, int height,
      int channels, const char *source_id, const cortext_media *media,
      const cortext_process_json_options *options)
  {
    auto *p = cast_handle (h);
    if (!p || !data || !source_id)
      {
        set_last_error ("handle, data, and source_id must all be non-NULL");
        return nullptr;
      }
    if (!valid_media_or_set_error (media))
      {
        return nullptr;
      }

    const auto retention = retention_from_options (options);
    return invoke_json ([&] {
      return p->ProcessImage (data, width, height, channels,
                              std::string (source_id), media_from_c (media),
                              retention);
    },
                        include_embedding_from_options (options));
  }

  char *
  cortext_consolidate_json (cortext_handle h)
  {
    auto *p = cast_handle (h);
    if (!p)
      {
        set_last_error ("handle must not be NULL");
        return nullptr;
      }

    return invoke_json ([&] { return p->Consolidate (); });
  }

} // extern "C"
