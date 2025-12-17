#include "cortext/operations/memory_storage.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/signal.hpp"
#include "cortext/store/schema_helpers.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <cstring>
#include <string>

namespace cortext::operations
{

void
MemoryStorage::Execute (OperationContext &context, Transaction &tx) const
{
  // Check write gate decision - if rejected, discard entirely
  if (!context.GetWriteDecision ())
    {
      telemetry::AddCounter ("cortext.memory_storage.rejected_total", 1);
      return;
    }

  const auto &signal = context.GetSignal ();

  // Check if there's a payload to store
  if (!signal.payload || signal.payload->empty ())
    {
      telemetry::AddCounter ("cortext.memory_storage.no_payload_total", 1);
      return;
    }

  Store *store = context.GetStore ();
  if (!store)
    {
      telemetry::AddCounter ("cortext.memory_storage.no_store_total", 1);
      return;
    }

  // Use nested transaction for atomicity - all writes succeed or none
  auto savepoint = tx.Begin ();

  try
    {
      using store::AnyToLongLong;
      using store::BlobFromAny;
      using store::EigenToFloatVec;

      // Convert embedding from Eigen to vector<float> for SQL storage
      const std::vector<float> emb_vec = EigenToFloatVec (signal.embedding);

      // Convert embedding to char blob for sqlite-vec storage
      std::vector<char> emb_blob (sizeof (float) * emb_vec.size ());
      std::memcpy (emb_blob.data (), emb_vec.data (), emb_blob.size ());

      // 1. Store payload in objstore
      auto blob_rows = savepoint->Execute ("SELECT objstore_put(?1) AS id",
                                           { *signal.payload });
      if (blob_rows.empty () || blob_rows[0].count ("id") == 0)
        {
          savepoint->Rollback ();
          telemetry::AddCounter (
              "cortext.memory_storage.objstore_error_total", 1);
          return;
        }
      const auto blob_id = BlobFromAny (blob_rows[0].at ("id"));
      if (blob_id.empty ())
        {
          savepoint->Rollback ();
          telemetry::AddCounter (
              "cortext.memory_storage.objstore_error_total", 1);
          return;
        }

      // 2. Insert embedding with initial metadata (unified embeddings table)
      // Note: sqlite-vec doesn't support DEFAULT values, so we must set all fields
      const std::string insert_sql = std::string ("INSERT INTO embeddings (")
                                     + store::kEmbeddingsInsertColumns + ") VALUES ("
                                     + store::kEmbeddingsMemoryDefaults + ")";
      savepoint->Execute (insert_sql,
                          { emb_vec, static_cast<long long> (signal.timestamp) });
      auto id_rows
          = savepoint->Execute ("SELECT last_insert_rowid() AS id", {});
      if (id_rows.empty () || id_rows[0].count ("id") == 0)
        {
          savepoint->Rollback ();
          telemetry::AddCounter (
              "cortext.memory_storage.embedding_error_total", 1);
          return;
        }
      const auto id_opt = AnyToLongLong (id_rows[0].at ("id"));
      if (!id_opt)
        {
          savepoint->Rollback ();
          telemetry::AddCounter (
              "cortext.memory_storage.embedding_error_total", 1);
          return;
        }
      const long long embedding_id = *id_opt;

      // 3. Insert memories (metadata)
      savepoint->Execute (
          "INSERT INTO memories (embedding_id, modality, mime, source_id, "
          "timestamp, width, height, channels, sample_rate, num_samples, "
          "blob_id) "
          "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
          { embedding_id, signal.modality, signal.mimetype, signal.source_id,
            static_cast<long long> (signal.timestamp),
            static_cast<long long> (signal.width),
            static_cast<long long> (signal.height),
            static_cast<long long> (signal.channels),
            static_cast<long long> (signal.sample_rate),
            static_cast<long long> (signal.num_samples), blob_id });

      // Commit the savepoint
      savepoint->Commit ();

      // Set stored_embedding_id in context for output
      context.SetStoredEmbeddingId (embedding_id);
      telemetry::AddCounter ("cortext.memory_storage.stored_total", 1);
    }
  catch (const std::exception &e)
    {
      savepoint->Rollback ();
      telemetry::AddCounter ("cortext.memory_storage.error_total", 1);
      telemetry::LogError (
          "MemoryStorage failed",
          { telemetry::Attribute::String ("error", e.what ()) });
    }
}

} // namespace cortext::operations
