#pragma once

#include "cortext/store/schema.hpp"
#include "cortext/store/store.hpp"
#include "cortext/processor.hpp"
#include "cortext/encoder/encoder.hpp"
#include "cortext/models/aist_gguf_encoder.hpp"
#include "cortext/models/embedding_model_pin.hpp"
#include <Eigen/Dense>
#include <atomic>
#include <any>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace cortext::testing
{

class ScopedEnvVar
{
public:
  explicit ScopedEnvVar (const char *name) : name_ (name)
  {
    const char *existing = std::getenv (name);
    if (existing != nullptr)
      {
        had_value_ = true;
        old_value_ = existing;
      }
    unsetenv (name_);
  }

  ScopedEnvVar (const char *name, const std::string &value)
      : ScopedEnvVar (name)
  {
    setenv (name_, value.c_str (), 1);
  }

  ~ScopedEnvVar ()
  {
    if (had_value_)
      {
        setenv (name_, old_value_.c_str (), 1);
      }
    else
      {
        unsetenv (name_);
      }
  }

private:
  const char *name_;
  bool had_value_ = false;
  std::string old_value_;
};

class TestEncoder : public Encoder
{
public:
  void
  EncodeText (const std::string & /*text*/,
              std::vector<float> &out_embedding) override
  {
    out_embedding.assign (256, 0.0f);
    out_embedding[0] = 1.0f;
  }

  void
  EncodeAudio (const float * /*pcm*/, std::size_t /*num_samples*/,
               std::vector<float> &out_embedding) override
  {
    out_embedding.assign (256, 0.0f);
    out_embedding[0] = 1.0f;
  }

  void
  EncodeImage (const std::uint8_t * /*data*/, int /*width*/, int /*height*/,
               int /*channels*/, std::vector<float> &out_embedding) override
  {
    out_embedding.assign (256, 0.0f);
    out_embedding[0] = 1.0f;
  }
};

inline Encoder &
GetTestEncoder ()
{
  static TestEncoder encoder;
  return encoder;
}

inline void
RequireEncoder (SignalProcessor::Config &cfg)
{
  cfg.encoder = &GetTestEncoder ();
}

// ============================================================================
// Data extraction helpers
// ============================================================================

/// @brief Extract long long value from a query result row.
inline long long
GetInt64 (const std::map<std::string, std::any> &row, const std::string &key)
{
  auto it = row.find (key);
  if (it == row.end ())
    return 0;
  if (it->second.type () == typeid (long long))
    return std::any_cast<long long> (it->second);
  if (it->second.type () == typeid (int))
    return static_cast<long long> (std::any_cast<int> (it->second));
  return 0;
}

/// @brief Extract double value from a query result row.
inline double
GetDouble (const std::map<std::string, std::any> &row, const std::string &key)
{
  auto it = row.find (key);
  if (it == row.end ())
    return 0.0;
  if (it->second.type () == typeid (double))
    return std::any_cast<double> (it->second);
  if (it->second.type () == typeid (float))
    return static_cast<double> (std::any_cast<float> (it->second));
  return 0.0;
}

// ============================================================================
// v2 Schema seeding helpers
// ============================================================================

/// @brief Get current timestamp in milliseconds.
inline long long
NowMs ()
{
  return std::chrono::duration_cast<std::chrono::milliseconds> (
             std::chrono::system_clock::now ().time_since_epoch ())
      .count ();
}

inline std::filesystem::path
UniqueTempPath (std::string_view prefix, std::string_view suffix)
{
  static std::atomic<unsigned long long> counter{ 0 };
#if defined(_WIN32)
  const auto pid = static_cast<unsigned long long> (_getpid ());
#else
  const auto pid = static_cast<unsigned long long> (getpid ());
#endif
  const auto tick = static_cast<unsigned long long> (
      std::chrono::high_resolution_clock::now ().time_since_epoch ().count ());
  const auto seq = counter.fetch_add (1, std::memory_order_relaxed);
  return std::filesystem::temp_directory_path ()
         / (std::string (prefix) + std::to_string (pid) + "_"
            + std::to_string (tick) + "_" + std::to_string (seq)
            + std::string (suffix));
}

/// @brief Seed an embedding into the v2 embeddings table (minimal vec0 table).
/// @param store The store to seed into.
/// @param id The embedding_id.
/// @param emb The embedding vector.
/// @param created_at Optional timestamp (defaults to current time).
inline void
SeedEmbeddingV2 (Store &store, long long id, const std::vector<float> &emb,
                 long long created_at = 0)
{
  if (created_at == 0)
    created_at = NowMs ();
  store.Execute (
      "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { id, emb, created_at });
}

/// @brief Seed an embedding into the v2 embeddings table from Eigen vector.
inline void
SeedEmbeddingV2 (Store &store, long long id, const Eigen::VectorXf &emb,
                 long long created_at = 0)
{
  std::vector<float> vec (emb.data (), emb.data () + emb.size ());
  SeedEmbeddingV2 (store, id, vec, created_at);
}

/// @brief Seed or refresh the bounded current-memory vector surface.
inline void
SeedCurrentMemoryEmbeddingV2 (Store &store, long long memory_id,
                              long long embedding_id)
{
  try
    {
      store.Execute ("DELETE FROM current_memory_embeddings "
                     "WHERE memory_id = ?",
                     { memory_id });
      store.Execute (
          "INSERT INTO current_memory_embeddings("
          "memory_id, embedding, embedding_id, created_at"
          ") "
          "SELECT ?, embedding, embedding_id, created_at "
          "FROM embeddings WHERE embedding_id = ?",
          { memory_id, embedding_id });
    }
  catch (...)
    {
    }
}

/// @brief Seed a memory into the v2 memories table with comprehensive metadata.
/// @param store The store to seed into.
/// @param memory_id The memory_id (primary key).
/// @param embedding_id The embedding_id (foreign key to embeddings).
/// @param source_id The source identifier.
/// @param kind Memory kind: 'LONG_TERM', 'WORKING', 'ASSOCIATION', 'LABEL'.
/// @param strength Memory strength (default 1.0).
/// @param start_ts Start timestamp in milliseconds.
inline void
SeedMemoryV2 (Store &store, long long memory_id, long long embedding_id,
              const std::string &source_id, const std::string &kind = "LONG_TERM",
              double strength = 1.0, long long start_ts = 0)
{
  if (start_ts == 0)
    start_ts = NowMs ();
  store.Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, created_at) "
      "VALUES(?, ?, ?, ?, ?, 1, 'text', 0.5, 0.5, ?, ?)",
      { memory_id, embedding_id, source_id, kind, start_ts, strength, start_ts });
  SeedCurrentMemoryEmbeddingV2 (store, memory_id, embedding_id);
}

/// @brief Seed a memory with extended metadata fields.
inline void
SeedMemoryV2Extended (Store &store, long long memory_id, long long embedding_id,
                      const std::string &source_id, double strength = 1.0,
                      double redundancy = 0.0, double connectivity = 0.0,
                      double stability = 1.0, long long start_ts = 0)
{
  if (start_ts == 0)
    start_ts = NowMs ();
  store.Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, redundancy, connectivity, stability, created_at) "
      "VALUES(?, ?, ?, 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, ?, ?, ?, ?, ?)",
      { memory_id, embedding_id, source_id, start_ts, strength, redundancy,
        connectivity, stability, start_ts });
  SeedCurrentMemoryEmbeddingV2 (store, memory_id, embedding_id);
}

/// @brief Seed a signal into the v2 signals table with inline metrics.
/// @param store The store to seed into.
/// @param signal_id The signal_id (primary key).
/// @param embedding_id The embedding_id (foreign key to embeddings).
/// @param source_id The source identifier.
/// @param timestamp Signal timestamp in milliseconds.
/// @param score Optional composite score.
inline void
SeedSignalV2 (Store &store, long long signal_id, long long embedding_id,
              const std::string &source_id, long long timestamp,
              double score = 0.0)
{
  store.Execute (
      "INSERT OR REPLACE INTO signals("
      "signal_id, embedding_id, source_id, timestamp, modality, score, created_at) "
      "VALUES(?, ?, ?, ?, 'text', ?, ?)",
      { signal_id, embedding_id, source_id, timestamp, score, timestamp });
}

/// @brief Stamp the embedding-model pin the engine will expect.
///
/// Fixtures that seed embeddings directly (bypassing the engine) must record
/// the active encoder's fingerprint, exactly as Cortext::Create does;
/// otherwise the pin check correctly refuses to start on the seeded store.
inline void
StampEmbeddingModelPin (Store &store)
{
  namespace fs = std::filesystem;
  const fs::path models_dir
      = fs::path (__FILE__).parent_path ().parent_path () / "models";
  const auto aist = cortext::ResolveAistGgufModelPath (models_dir);
  if (!aist)
    {
      return; // No model on this machine; engine creation fails earlier.
    }
  const std::string pin
      = cortext::models::ComputeEmbeddingModelPin ("AIST-87M-GGUF", *aist,
                                                   256);
  store.Execute ("CREATE TABLE IF NOT EXISTS cortext_embedding_model_pin ("
                 "  id INTEGER PRIMARY KEY CHECK (id = 1),"
                 "  fingerprint TEXT NOT NULL"
                 ")",
                 {});
  store.Execute ("INSERT OR REPLACE INTO cortext_embedding_model_pin "
                 "(id, fingerprint) VALUES (1, ?)",
                 { pin });
}

/// @brief Initialize the core schema on a store using ApplyMigrations.
/// This ensures tests use the same schema as production code.
inline void
InitializeCoreSchema (Store &store)
{
  cortext::store::ApplyMigrations (store);
  StampEmbeddingModelPin (store);
}

/// @brief A no-op transaction for tests that don't need actual database writes.
class NullTransaction : public Transaction
{
public:
  std::unique_ptr<Transaction>
  Begin () override
  {
    return std::make_unique<NullTransaction> ();
  }

  std::vector<std::map<std::string, std::any>>
  Execute (const std::string & /*query*/,
           const std::vector<std::any> & /*params*/ = {}) override
  {
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
};

/// @brief Get a global NullTransaction instance for tests.
inline Transaction &
GetNullTransaction ()
{
  static NullTransaction null_tx;
  return null_tx;
}

/// @brief Seed a mature label-contrast bank into the processor context.
///
/// The optional label-admission paths (floor fill, relation-endpoint
/// creation) decline until the contrast bank holds kContrastMinBankSize
/// labels. Fixtures exercising them seed this minimal mature bank; its
/// embeddings sit on the highest axes so fixture labels (low-index
/// one-hot by convention) measure mean ~0 against the bank and are never
/// rejected as generic.
inline void
SeedMatureLabelContrastBank (ProcessorContext &p_ctx, int dim = 256,
                             int count = 4)
{
  for (int i = 0; i < count; ++i)
    {
      Eigen::VectorXf v = Eigen::VectorXf::Zero (dim);
      v (dim - 1 - i) = 1.0f;
      const long long id = 900000 + i;
      p_ctx.UpsertAssociationCache (id, id, v, false);
    }
}

} // namespace cortext::testing
