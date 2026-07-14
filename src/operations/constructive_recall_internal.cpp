#include "constructive_recall_internal.hpp"
#include "historical_surface_search_cache_internal.hpp"

#include "cortext/core/utils.hpp"
#include "cortext/store/utils.hpp"
#include "../experimental_env.hpp"

#include <any>
#include <algorithm>
#include <chrono>
#include <map>
#include <sstream>
#include <string>

namespace cortext::operations::constructive_recall
{

template <typename Executor>
std::optional<ReconstructionRecord>
LoadLatestReconstructionImpl (Executor &executor, long long memory_id);

bool
CurrentSurfaceWritesDisabled ()
{
  return internal::experimental_env::Flag (
      "CORTEXT_DISABLE_CURRENT_MEMORY_SURFACE_WRITES");
}

namespace
{
template <typename Executor>
std::vector<std::map<std::string, std::any> >
Exec (Executor &executor, const std::string &query,
      const std::vector<std::any> &params = {})
{
  return executor.Execute (query, params);
}

long long
AnyToInt64 (const std::any &value)
{
  if (!value.has_value ())
    {
      return 0;
    }
  if (value.type () == typeid (long long))
    {
      return std::any_cast<long long> (value);
    }
  if (value.type () == typeid (int))
    {
      return static_cast<long long> (std::any_cast<int> (value));
    }
  return 0;
}

int
ResolveHistoryLimit (ReconstructionUpdatePolicy policy)
{
  if (const auto env
      = internal::experimental_env::Int ("CORTEXT_RECONSTRUCTION_HISTORY_LIMIT"))
    {
      if (*env > 0)
        {
          return *env;
        }
    }
  return policy.history_limit;
}

int
ResolvePruneBatchLimit (ReconstructionUpdatePolicy policy,
                        int history_limit)
{
  if (const auto env = internal::experimental_env::Int (
          "CORTEXT_RECONSTRUCTION_PRUNE_BATCH_LIMIT"))
    {
      if (*env > 0)
        {
          return *env;
        }
    }
  if (policy.prune_batch_limit > 0)
    {
      return policy.prune_batch_limit;
    }
  return history_limit;
}

long long
ResolveMinUpdateIntervalMs (ReconstructionUpdatePolicy policy)
{
  if (const auto env = internal::experimental_env::Int64 (
          "CORTEXT_RECONSTRUCTION_MIN_UPDATE_MS"))
    {
      return *env;
    }
  return policy.min_update_interval_ms;
}

std::string
Placeholders (size_t count)
{
  std::string out;
  for (size_t i = 0; i < count; ++i)
    {
      if (i > 0)
        {
          out += ",";
        }
      out += "?";
    }
  return out;
}

std::vector<float>
ToFloatVector (const Eigen::VectorXf &v)
{
  std::vector<float> out;
  out.resize (static_cast<size_t> (v.size ()));
  for (int i = 0; i < v.size (); ++i)
    {
      out[static_cast<size_t> (i)] = v[i];
    }
  return out;
}

double
ElapsedMs (std::chrono::steady_clock::time_point start,
           std::chrono::steady_clock::time_point end)
{
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli> > (
             end - start)
      .count ();
}

long long
ExtractInsertedId (const std::vector<std::map<std::string, std::any>> &rows)
{
  if (rows.empty ())
    {
      return 0;
    }
  auto it = rows[0].find ("id");
  if (it == rows[0].end () || !it->second.has_value ())
    {
      return 0;
    }
  if (it->second.type () == typeid (long long))
    {
      return std::any_cast<long long> (it->second);
    }
  if (it->second.type () == typeid (int))
    {
      return static_cast<long long> (std::any_cast<int> (it->second));
    }
  return 0;
}

template <typename Executor>
bool
IsCurrentSurfaceMemoryKind (Executor &executor, long long memory_id)
{
  auto rows = Exec (executor, "SELECT kind FROM memories WHERE memory_id = ?",
                    { memory_id });
  if (rows.empty ())
    {
      return false;
    }
  auto it = rows[0].find ("kind");
  if (it == rows[0].end () || !it->second.has_value ()
      || it->second.type () != typeid (std::string))
    {
      return false;
    }
  const auto kind = std::any_cast<std::string> (it->second);
  return kind != "WORKING" && kind != "LABEL";
}

template <typename Executor>
bool
ShouldSkipReconstructionUpdateImpl (Executor &executor, long long memory_id,
                                    long long created_at,
                                    std::string_view trigger,
                                    ReconstructionUpdatePolicy policy)
{
  if (memory_id <= 0 || trigger == "initial")
    {
      return false;
    }
  const long long min_interval_ms = ResolveMinUpdateIntervalMs (policy);
  if (min_interval_ms <= 0)
    {
      return false;
    }

  long long latest_created_at = 0;
  auto current_rows = Exec (
      executor,
      "SELECT created_at FROM current_memory_embeddings WHERE memory_id = ?",
      { memory_id });
  if (!current_rows.empty ())
    {
      const auto it = current_rows[0].find ("created_at");
      if (it != current_rows[0].end ())
        {
          latest_created_at = AnyToInt64 (it->second);
        }
    }
  const auto latest = LoadLatestReconstructionImpl (executor, memory_id);
  if (latest.has_value ())
    {
      latest_created_at = std::max (latest_created_at, latest->created_at);
    }
  if (latest_created_at <= 0 || created_at <= 0)
    {
      return false;
    }
  if (created_at <= latest_created_at)
    {
      return true;
    }
  return (created_at - latest_created_at) < min_interval_ms;
}

template <typename Executor>
void
PruneReconstructionHistory (Executor &executor, long long memory_id,
                            ReconstructionUpdatePolicy policy,
                            ProcessorContext *p_ctx = nullptr)
{
  const int history_limit = ResolveHistoryLimit (policy);
  if (memory_id <= 0 || history_limit <= 0)
    {
      return;
    }
  const int batch_limit = std::max (
      0, ResolvePruneBatchLimit (policy, history_limit));
  if (batch_limit <= 0)
    {
      return;
    }

  auto rows = Exec (
      executor,
      "SELECT reconstruction_id, embedding_id "
      "FROM memory_reconstructions "
      "WHERE memory_id = ? "
      "ORDER BY reconstruction_id DESC "
      "LIMIT ? OFFSET ?",
      { memory_id, static_cast<long long> (batch_limit),
        static_cast<long long> (history_limit) });
  if (rows.empty ())
    {
      return;
    }

  std::vector<std::any> reconstruction_ids;
  std::vector<std::any> embedding_ids;
  reconstruction_ids.reserve (rows.size ());
  embedding_ids.reserve (rows.size ());
  std::vector<long long> seen_embeddings;
  seen_embeddings.reserve (rows.size ());
  for (const auto &row : rows)
    {
      const long long reconstruction_id = AnyToInt64 (
          row.at ("reconstruction_id"));
      const long long embedding_id = AnyToInt64 (row.at ("embedding_id"));
      if (reconstruction_id > 0)
        {
          reconstruction_ids.push_back (reconstruction_id);
        }
      if (embedding_id > 0
          && std::find (seen_embeddings.begin (), seen_embeddings.end (),
                        embedding_id)
                 == seen_embeddings.end ())
        {
          seen_embeddings.push_back (embedding_id);
          embedding_ids.push_back (embedding_id);
        }
    }
  if (reconstruction_ids.empty ())
    {
      return;
    }

  Exec (executor,
        std::string ("DELETE FROM memory_reconstructions "
                     "WHERE reconstruction_id IN (")
            + Placeholders (reconstruction_ids.size ()) + ")",
        reconstruction_ids);

  if (!policy.delete_orphan_embeddings || embedding_ids.empty ())
    {
      return;
    }
  Exec (executor,
        std::string ("DELETE FROM embeddings "
                     "WHERE embedding_id IN (")
            + Placeholders (embedding_ids.size ())
            + ") "
              "  AND NOT EXISTS ("
              "    SELECT 1 FROM memories m "
              "    WHERE m.embedding_id = embeddings.embedding_id"
              "  ) "
              "  AND NOT EXISTS ("
              "    SELECT 1 FROM memory_reconstructions mr "
              "    WHERE mr.embedding_id = embeddings.embedding_id"
              "  )",
        embedding_ids);
  if (p_ctx != nullptr)
    {
      auto retained_rows = Exec (
          executor,
          std::string ("SELECT embedding_id FROM embeddings "
                       "WHERE embedding_id IN (")
              + Placeholders (embedding_ids.size ()) + ")",
          embedding_ids);
      std::vector<long long> retained_ids;
      retained_ids.reserve (retained_rows.size ());
      for (const auto &row : retained_rows)
        {
          const auto it = row.find ("embedding_id");
          if (it != row.end ())
            {
              retained_ids.push_back (AnyToInt64 (it->second));
            }
        }
      for (const long long embedding_id : seen_embeddings)
        {
          if (std::find (retained_ids.begin (), retained_ids.end (),
                        embedding_id)
              == retained_ids.end ())
            {
              historical_surface_search_cache_internal::RemoveEmbedding (
                  *p_ctx, embedding_id);
            }
        }
    }
}

template <typename Executor>
void
StoreCurrentEmbeddingSurface (Executor &executor, long long memory_id,
                              long long embedding_id,
                              const std::any &embedding,
                              long long created_at)
{
  if (memory_id <= 0 || embedding_id <= 0 || !embedding.has_value ())
    {
      return;
    }
  if (CurrentSurfaceWritesDisabled ())
    {
      return;
    }
  if (!IsCurrentSurfaceMemoryKind (executor, memory_id))
    {
      Exec (executor,
            "DELETE FROM current_memory_embeddings WHERE memory_id = ?",
            { memory_id });
      return;
    }

  Exec (executor,
        "DELETE FROM current_memory_embeddings WHERE memory_id = ?",
        { memory_id });
  Exec (
      executor,
      "INSERT INTO current_memory_embeddings("
      "memory_id, embedding, embedding_id, created_at"
      ") VALUES (?, ?, ?, ?)",
      { memory_id, embedding, embedding_id, created_at });
}

template <typename Executor>
void
UpdateCurrentEmbeddingSurface (Executor &executor, long long memory_id,
                               long long embedding_id, long long created_at)
{
  if (memory_id <= 0 || embedding_id <= 0)
    {
      return;
    }
  if (CurrentSurfaceWritesDisabled ())
    {
      return;
    }

  auto rows = Exec (
      executor,
      "SELECT embedding FROM embeddings WHERE embedding_id = ?",
      { embedding_id });
  if (rows.empty ())
    {
      return;
    }
  auto it = rows[0].find ("embedding");
  if (it == rows[0].end () || !it->second.has_value ())
    {
      return;
    }

  const auto embedding = store::BlobFromAny (it->second);
  if (embedding.empty ())
    {
      return;
    }

  StoreCurrentEmbeddingSurface (executor, memory_id, embedding_id,
                                std::any (embedding), created_at);
}
} // namespace

bool
Disabled ()
{
  return internal::experimental_env::Flag (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL");
}

template <typename Executor>
std::optional<ReconstructionRecord>
LoadLatestReconstructionImpl (Executor &executor, long long memory_id)
{
  if (memory_id <= 0 || Disabled ())
    {
      return std::nullopt;
    }

  auto rows = Exec (
      executor,
      "SELECT reconstruction_id, memory_id, embedding_id, blob_id, "
      "       uncertainty, created_at, trigger "
      "FROM memory_reconstructions "
      "WHERE memory_id = ? "
      "ORDER BY reconstruction_id DESC "
      "LIMIT 1",
      { memory_id });
  if (rows.empty ())
    {
      return std::nullopt;
    }

  ReconstructionRecord record;
  const auto &row = rows[0];
  if (auto it = row.find ("reconstruction_id");
      it != row.end () && it->second.type () == typeid (long long))
    {
      record.reconstruction_id = std::any_cast<long long> (it->second);
    }
  if (auto it = row.find ("memory_id");
      it != row.end () && it->second.type () == typeid (long long))
    {
      record.memory_id = std::any_cast<long long> (it->second);
    }
  if (auto it = row.find ("embedding_id");
      it != row.end () && it->second.type () == typeid (long long))
    {
      record.embedding_id = std::any_cast<long long> (it->second);
    }
  if (auto it = row.find ("blob_id");
      it != row.end () && it->second.has_value ())
    {
      record.blob_id = store::BlobFromAny (it->second);
    }
  if (auto it = row.find ("uncertainty");
      it != row.end () && it->second.has_value ())
    {
      if (it->second.type () == typeid (double))
        {
          record.uncertainty = std::any_cast<double> (it->second);
        }
      else if (it->second.type () == typeid (float))
        {
          record.uncertainty = static_cast<double> (std::any_cast<float> (it->second));
        }
    }
  if (auto it = row.find ("created_at");
      it != row.end () && it->second.type () == typeid (long long))
    {
      record.created_at = std::any_cast<long long> (it->second);
    }
  if (auto it = row.find ("trigger");
      it != row.end () && it->second.type () == typeid (std::string))
    {
      record.trigger = std::any_cast<std::string> (it->second);
    }
  if (record.reconstruction_id <= 0 || record.embedding_id <= 0)
    {
      return std::nullopt;
    }
  return record;
}

template <typename Executor>
std::vector<unsigned char>
LoadCurrentBlobIdImpl (Executor &executor, long long memory_id)
{
  if (memory_id <= 0)
    {
      return {};
    }
  if (const auto latest = LoadLatestReconstructionImpl (executor, memory_id);
      latest.has_value () && !latest->blob_id.empty ())
    {
      return latest->blob_id;
    }

  auto rows = Exec (executor, "SELECT blob_id FROM memories WHERE memory_id = ?",
                    { memory_id });
  if (rows.empty ())
    {
      return {};
    }
  auto it = rows[0].find ("blob_id");
  if (it == rows[0].end () || !it->second.has_value ())
    {
      return {};
    }
  return store::BlobFromAny (it->second);
}

template <typename Executor>
std::optional<Eigen::VectorXf>
LoadCurrentEmbeddingImpl (Executor &executor, long long memory_id,
                          long long base_embedding_id, int expected_dim)
{
  if (expected_dim <= 0)
    {
      return std::nullopt;
    }

  const auto latest = LoadLatestReconstructionImpl (executor, memory_id);

  if (!Disabled ())
    {
      auto current_rows = Exec (
          executor,
          "SELECT embedding, embedding_id, created_at "
          "FROM current_memory_embeddings WHERE memory_id = ?",
          { memory_id });
      if (!current_rows.empty ())
        {
          const long long current_embedding_id = AnyToInt64 (
              current_rows[0].at ("embedding_id"));
          const bool current_surface_fresh
              = !latest.has_value ()
                || current_embedding_id == latest->embedding_id;
          auto it = current_rows[0].find ("embedding");
          if (current_surface_fresh && it != current_rows[0].end ())
            {
              Eigen::VectorXf current;
              if (core::DecodeFloatBlob (it->second, expected_dim, current))
                {
                  return current;
                }
            }
        }
    }

  long long embedding_id = base_embedding_id;
  if (latest.has_value ())
    {
      embedding_id = latest->embedding_id;
    }
  if (embedding_id <= 0)
    {
      return std::nullopt;
    }

  auto rows = Exec (executor, "SELECT embedding FROM embeddings WHERE embedding_id = ?",
                    { embedding_id });
  if (rows.empty ())
    {
      return std::nullopt;
    }
  auto it = rows[0].find ("embedding");
  if (it == rows[0].end ())
    {
      return std::nullopt;
    }

  Eigen::VectorXf out;
  if (!core::DecodeFloatBlob (it->second, expected_dim, out))
    {
      return std::nullopt;
    }
  return out;
}

std::optional<ReconstructionRecord>
LoadLatestReconstruction (Store *store, long long memory_id)
{
  if (!store)
    {
      return std::nullopt;
    }
  return LoadLatestReconstructionImpl (*store, memory_id);
}

std::optional<ReconstructionRecord>
LoadLatestReconstruction (Transaction &tx, long long memory_id)
{
  return LoadLatestReconstructionImpl (tx, memory_id);
}

bool
ShouldSkipReconstructionUpdate (Transaction &tx, long long memory_id,
                                long long created_at,
                                std::string_view trigger,
                                ReconstructionUpdatePolicy policy)
{
  return ShouldSkipReconstructionUpdateImpl (tx, memory_id, created_at,
                                             trigger, policy);
}

std::vector<unsigned char>
LoadCurrentBlobId (Store *store, long long memory_id)
{
  if (!store)
    {
      return {};
    }
  return LoadCurrentBlobIdImpl (*store, memory_id);
}

std::vector<unsigned char>
LoadCurrentBlobId (Transaction &tx, long long memory_id)
{
  return LoadCurrentBlobIdImpl (tx, memory_id);
}

std::optional<Eigen::VectorXf>
LoadCurrentEmbedding (Store *store, long long memory_id, long long base_embedding_id,
                      int expected_dim)
{
  if (!store)
    {
      return std::nullopt;
    }
  return LoadCurrentEmbeddingImpl (*store, memory_id, base_embedding_id,
                                   expected_dim);
}

std::optional<Eigen::VectorXf>
LoadCurrentEmbedding (Transaction &tx, long long memory_id,
                      long long base_embedding_id, int expected_dim)
{
  return LoadCurrentEmbeddingImpl (tx, memory_id, base_embedding_id,
                                   expected_dim);
}

long long
AppendReconstructionWithEmbeddingId (Transaction &tx, long long memory_id,
                                     long long embedding_id,
                                     const std::vector<unsigned char> &blob_id,
                                     long long created_at, double uncertainty,
                                     std::string_view trigger,
                                     double source_confidence,
                                     double context_similarity,
                                     ReconstructionUpdatePolicy policy)
{
  if (memory_id <= 0 || embedding_id <= 0)
    {
      return 0;
    }
  if (ShouldSkipReconstructionUpdateImpl (tx, memory_id, created_at, trigger,
                                          policy))
    {
      return 0;
    }

  tx.Execute (
      "INSERT INTO memory_reconstructions("
      "memory_id, embedding_id, blob_id, created_at, uncertainty, trigger, "
      "source_confidence, context_similarity"
      ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
      { memory_id, embedding_id,
        blob_id.empty () ? std::any () : std::any (blob_id), created_at,
        uncertainty, std::string (trigger), source_confidence,
        context_similarity });
  const long long reconstruction_id
      = ExtractInsertedId (tx.Execute ("SELECT last_insert_rowid() AS id", {}));

  if (policy.update_current_surface)
    {
      UpdateCurrentEmbeddingSurface (tx, memory_id, embedding_id, created_at);
    }
  PruneReconstructionHistory (tx, memory_id, policy);

  return reconstruction_id;
}

long long
AppendReconstructionWithEmbedding (Transaction &tx, long long memory_id,
                                   const Eigen::VectorXf &embedding,
                                   const std::vector<unsigned char> &blob_id,
                                   long long created_at, double uncertainty,
                                   std::string_view trigger,
                                   double source_confidence,
                                   double context_similarity,
                                   ReconstructionUpdatePolicy policy,
                                   ProcessorContext *p_ctx)
{
  if (memory_id <= 0 || embedding.size () == 0)
    {
      return 0;
    }
  if (ShouldSkipReconstructionUpdateImpl (tx, memory_id, created_at, trigger,
                                          policy))
    {
      return 0;
    }

  return AppendReconstructionWithEmbeddingUnchecked (
      tx, memory_id, embedding, blob_id, created_at, uncertainty, trigger,
      source_confidence, context_similarity, policy, nullptr, p_ctx);
}

long long
AppendReconstructionWithEmbeddingUnchecked (
    Transaction &tx, long long memory_id, const Eigen::VectorXf &embedding,
    const std::vector<unsigned char> &blob_id, long long created_at,
    double uncertainty, std::string_view trigger, double source_confidence,
    double context_similarity, ReconstructionUpdatePolicy policy)
{
  return AppendReconstructionWithEmbeddingUnchecked (
      tx, memory_id, embedding, blob_id, created_at, uncertainty, trigger,
      source_confidence, context_similarity, policy, nullptr, nullptr);
}

long long
AppendReconstructionWithEmbeddingUnchecked (
    Transaction &tx, long long memory_id, const Eigen::VectorXf &embedding,
    const std::vector<unsigned char> &blob_id, long long created_at,
    double uncertainty, std::string_view trigger, double source_confidence,
    double context_similarity, ReconstructionUpdatePolicy policy,
    ReconstructionAppendTimings *timings, ProcessorContext *p_ctx)
{
  if (memory_id <= 0 || embedding.size () == 0)
    {
      return 0;
    }

  const auto embedding_values = ToFloatVector (embedding);
  auto t_start = std::chrono::steady_clock::now ();
  tx.Execute (
      "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
      { embedding_values, created_at });
  const long long embedding_id
      = ExtractInsertedId (tx.Execute ("SELECT last_insert_rowid() AS id", {}));
  if (timings)
    {
      timings->insert_embedding_ms += ElapsedMs (
          t_start, std::chrono::steady_clock::now ());
    }
  if (embedding_id <= 0)
    {
      return 0;
    }
  if (p_ctx != nullptr)
    {
      historical_surface_search_cache_internal::Append (
          *p_ctx,
          { embedding_id, 0, 0, std::string (), std::string (), embedding });
    }

  t_start = std::chrono::steady_clock::now ();
  tx.Execute (
      "INSERT INTO memory_reconstructions("
      "memory_id, embedding_id, blob_id, created_at, uncertainty, trigger, "
      "source_confidence, context_similarity"
      ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
      { memory_id, embedding_id,
        blob_id.empty () ? std::any () : std::any (blob_id), created_at,
        uncertainty, std::string (trigger), source_confidence,
        context_similarity });
  const long long reconstruction_id
      = ExtractInsertedId (tx.Execute ("SELECT last_insert_rowid() AS id", {}));
  if (timings)
    {
      timings->insert_reconstruction_ms += ElapsedMs (
          t_start, std::chrono::steady_clock::now ());
    }

  if (policy.update_current_surface)
    {
      t_start = std::chrono::steady_clock::now ();
      StoreCurrentEmbeddingSurface (tx, memory_id, embedding_id,
                                    std::any (embedding_values), created_at);
      if (timings)
        {
          timings->current_surface_ms += ElapsedMs (
              t_start, std::chrono::steady_clock::now ());
        }
    }
  t_start = std::chrono::steady_clock::now ();
  PruneReconstructionHistory (tx, memory_id, policy, p_ctx);
  if (timings)
    {
      timings->prune_ms += ElapsedMs (
          t_start, std::chrono::steady_clock::now ());
    }

  return reconstruction_id;
}

} // namespace cortext::operations::constructive_recall
