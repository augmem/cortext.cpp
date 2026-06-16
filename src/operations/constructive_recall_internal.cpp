#include "constructive_recall_internal.hpp"

#include "cortext/core/utils.hpp"
#include "cortext/store/utils.hpp"

#include <any>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>

namespace cortext::operations::constructive_recall
{

namespace
{
template <typename Executor>
std::vector<std::map<std::string, std::any> >
Exec (Executor &executor, const std::string &query,
      const std::vector<std::any> &params = {})
{
  return executor.Execute (query, params);
}

bool
EnvFlag (const char *name)
{
  const char *value = std::getenv (name);
  if (!value)
    {
      return false;
    }
  std::string s (value);
  std::transform (s.begin (), s.end (), s.begin (),
                  [] (unsigned char c) { return static_cast<char> (std::tolower (c)); });
  return s == "1" || s == "true" || s == "yes" || s == "on";
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
  return EnvFlag ("CORTEXT_DISABLE_CONSTRUCTIVE_RECALL");
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
  long long embedding_id = base_embedding_id;
  if (const auto latest = LoadLatestReconstructionImpl (executor, memory_id);
      latest.has_value ())
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
                                     double context_similarity)
{
  if (memory_id <= 0 || embedding_id <= 0)
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

  UpdateCurrentEmbeddingSurface (tx, memory_id, embedding_id, created_at);

  return ExtractInsertedId (
      tx.Execute ("SELECT last_insert_rowid() AS id", {}));
}

long long
AppendReconstructionWithEmbedding (Transaction &tx, long long memory_id,
                                   const Eigen::VectorXf &embedding,
                                   const std::vector<unsigned char> &blob_id,
                                   long long created_at, double uncertainty,
                                   std::string_view trigger,
                                   double source_confidence,
                                   double context_similarity)
{
  if (memory_id <= 0 || embedding.size () == 0)
    {
      return 0;
    }

  const auto embedding_values = ToFloatVector (embedding);
  tx.Execute (
      "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
      { embedding_values, created_at });
  const long long embedding_id
      = ExtractInsertedId (tx.Execute ("SELECT last_insert_rowid() AS id", {}));
  if (embedding_id <= 0)
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

  StoreCurrentEmbeddingSurface (tx, memory_id, embedding_id,
                                std::any (embedding_values), created_at);

  return reconstruction_id;
}

} // namespace cortext::operations::constructive_recall
