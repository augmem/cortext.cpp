#include "cortext/operations/memory_storage.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/signal.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <cstring>

namespace cortext::operations
{

void
MemoryStorage::Execute (OperationContext &context) const
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

  // Use savepoint for atomicity - all writes succeed or none
  auto transaction = store->Begin ();

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
      auto blob_rows = transaction->Execute ("SELECT objstore_put(?1) AS id",
                                             { *signal.payload });
      if (blob_rows.empty () || blob_rows[0].count ("id") == 0)
        {
          transaction->Rollback ();
          telemetry::AddCounter (
              "cortext.memory_storage.objstore_error_total", 1);
          return;
        }
      const auto blob_id = BlobFromAny (blob_rows[0].at ("id"));
      if (blob_id.empty ())
        {
          transaction->Rollback ();
          telemetry::AddCounter (
              "cortext.memory_storage.objstore_error_total", 1);
          return;
        }

      // 2. Insert embedding
      transaction->Execute ("INSERT INTO embeddings (embedding) VALUES (?)",
                            { emb_blob });
      auto id_rows
          = transaction->Execute ("SELECT last_insert_rowid() AS id", {});
      if (id_rows.empty () || id_rows[0].count ("id") == 0)
        {
          transaction->Rollback ();
          telemetry::AddCounter (
              "cortext.memory_storage.embedding_error_total", 1);
          return;
        }
      const auto id_opt = AnyToLongLong (id_rows[0].at ("id"));
      if (!id_opt)
        {
          transaction->Rollback ();
          telemetry::AddCounter (
              "cortext.memory_storage.embedding_error_total", 1);
          return;
        }
      const long long embedding_id = *id_opt;

      // 2b. Insert into vec_embeddings for KNN search
      transaction->Execute (
          "INSERT INTO vec_embeddings (embedding_id, embedding) VALUES (?, ?)",
          { embedding_id, emb_vec });

      // 3. Insert memory_index
      transaction->Execute (
          "INSERT INTO memory_index (embedding_id, modality, mime, source_id, "
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

      // 4. Insert memory_feedback
      transaction->Execute (
          "INSERT INTO memory_feedback (embedding_id, strength) "
          "VALUES (?, 1.0)",
          { embedding_id });

      // Commit the savepoint
      transaction->Commit ();

      // Set stored_embedding_id in context for output
      context.SetStoredEmbeddingId (embedding_id);
      telemetry::AddCounter ("cortext.memory_storage.stored_total", 1);
    }
  catch (const std::exception &e)
    {
      transaction->Rollback ();
      telemetry::AddCounter ("cortext.memory_storage.error_total", 1);
      telemetry::LogError (
          "MemoryStorage failed",
          { telemetry::Attribute::String ("error", e.what ()) });
    }
}

} // namespace cortext::operations
