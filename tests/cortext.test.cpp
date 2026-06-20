#include "test_helpers.hpp"
#include <any>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include <cortext/clock.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/capi.h>
#include <cortext/cortext.hpp>
#include <cortext/internal/replay_ingress.hpp>
#include <cortext/store/sqlite_store.hpp>

namespace
{
std::string
MakeTempDbPath ()
{
  const auto temp_dir = std::filesystem::temp_directory_path ();
  const auto stamp = std::chrono::high_resolution_clock::now ()
                         .time_since_epoch ()
                         .count ();
  auto path
      = temp_dir / ("cortext_objstore_test_" + std::to_string (stamp) + ".db");
  return path.string ();
}

void
CleanupTempDb (const std::string &path)
{
  std::error_code ec;
  std::filesystem::remove (path, ec);
  std::filesystem::remove (path + "-wal", ec);
  std::filesystem::remove (path + "-shm", ec);
}

std::vector<unsigned char>
BlobFromAny (const std::any &value)
{
  if (value.type () == typeid (std::vector<unsigned char>))
    {
      return std::any_cast<const std::vector<unsigned char> &> (value);
    }
  if (value.type () == typeid (std::vector<char>))
    {
      const auto &blob = std::any_cast<const std::vector<char> &> (value);
      return std::vector<unsigned char> (blob.begin (), blob.end ());
    }
  return {};
}

std::string
TextFromMemory (const cortext::Cortext::Context::Memory &memory)
{
  std::string text;
  for (const auto &blob : memory.content)
    {
      text.append (reinterpret_cast<const char *> (blob.data ()),
                   blob.size ());
    }
  return text;
}

long long
AnyToLongLong (const std::any &value)
{
  if (value.type () == typeid (long long))
    return std::any_cast<long long> (value);
  if (value.type () == typeid (int))
    return static_cast<long long> (std::any_cast<int> (value));
  return 0;
}

class ScopedTempDb
{
public:
  ScopedTempDb () : path_ (MakeTempDbPath ()) {}
  ~ScopedTempDb () { CleanupTempDb (path_); }

  const std::string &
  path () const
  {
    return path_;
  }

private:
  std::string path_;
};

std::vector<unsigned char>
SampleWavBytes ()
{
  const unsigned char kData[] = {
    0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56,
    0x45, 0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00,
    0x01, 0x00, 0x22, 0x56, 0x00, 0x00, 0x44, 0xac, 0x00, 0x00, 0x02,
    0x00, 0x10, 0x00, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00,
  };
  return std::vector<unsigned char> (std::begin (kData), std::end (kData));
}

std::vector<unsigned char>
SampleMp4Bytes ()
{
  const unsigned char kData[] = {
    0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70, 0x69, 0x73,
    0x6f, 0x6d, 0x00, 0x00, 0x02, 0x00, 0x69, 0x73, 0x6f, 0x6d,
    0x69, 0x73, 0x6f, 0x32, 0x61, 0x76, 0x63, 0x31,
  };
  return std::vector<unsigned char> (std::begin (kData), std::end (kData));
}

std::string
RepoModelsDir ()
{
  namespace fs = std::filesystem;
  const fs::path root = fs::path (__FILE__).parent_path ().parent_path ();
  return (root / "models").string ();
}
} // namespace

TEST_CASE ("Cortext C++ stub can be created and used", "[cortext][stub]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();
  const std::string models_dir = RepoModelsDir ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path, models_dir));
  REQUIRE (ctx != nullptr);

  REQUIRE_NOTHROW ([&] {
    auto out_text = ctx->ProcessText ("hello world", "test");
    (void)out_text.should_interrupt; // Algorithm-dependent, not asserted

    auto out_cons = ctx->Consolidate ();
    (void)out_cons.should_interrupt;
    ctx->Flush ();
  }());

  // DB assertion: open the same DB and execute a trivial query successfully.
  auto uniq = cortext::SQLiteStore::Create (db_path.c_str ());
  auto store = std::shared_ptr<cortext::Store> (std::move (uniq));
  auto rows = store->Execute ("SELECT 1 AS ok");
  REQUIRE (rows.size () == 1);
}

TEST_CASE ("Cortext embed-only API returns vectors without storing signals",
           "[cortext][embed]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();
  const std::string models_dir = RepoModelsDir ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path, models_dir));
  REQUIRE (ctx != nullptr);

  std::vector<float> embedding;
  REQUIRE_NOTHROW (embedding = ctx->EmbedText ("embed only text"));
  REQUIRE_FALSE (embedding.empty ());

  auto store = std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (db_path.c_str ()));
  const auto rows = store->Execute ("SELECT COUNT(*) AS n FROM signals");
  REQUIRE (rows.size () == 1);
  REQUIRE (AnyToLongLong (rows[0].at ("n")) == 0);
}

TEST_CASE ("internal replay ingress preserves media event timestamps",
           "[cortext][replay][media][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  cfg.signal_filter_audio_enabled = false;
  const std::string &db_path = temp_db.path ();
  const std::string models_dir = RepoModelsDir ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path, models_dir));
  REQUIRE (ctx != nullptr);

  const std::uint64_t replay_ts = 1573184762000ULL;
  std::vector<float> pcm (16000, 0.01f);

  REQUIRE_NOTHROW (
      cortext::internal::ReplayIngress::ProcessAudioAt (
          *ctx, pcm.data (), pcm.size (), "stream/reply", replay_ts));
  ctx->Flush ();

  auto unique_store = cortext::SQLiteStore::Create (db_path.c_str ());
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  auto rows = store->Execute (
      "SELECT timestamp, source_id, modality FROM signals "
      "WHERE modality = 'audio' ORDER BY signal_id DESC LIMIT 1");
  REQUIRE (rows.size () == 1);
  REQUIRE (AnyToLongLong (rows[0].at ("timestamp"))
           == static_cast<long long> (replay_ts));
  REQUIRE (std::any_cast<std::string> (rows[0].at ("source_id"))
           == "stream/reply");
  REQUIRE (std::any_cast<std::string> (rows[0].at ("modality")) == "audio");
}

TEST_CASE ("timestamped replay persists working memory source timestamps",
           "[cortext][replay][working_memory]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();
  const std::string models_dir = RepoModelsDir ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path, models_dir));
  REQUIRE (ctx != nullptr);

  const std::uint64_t replay_ts = 1573184762000ULL;
  REQUIRE_NOTHROW (
      ctx->ProcessTextAt ("timestamped working memory replay", "stream/main",
                          replay_ts));
  ctx->Flush ();

  auto unique_store = cortext::SQLiteStore::Create (db_path.c_str ());
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  auto rows = store->Execute (
      "SELECT start_ts, created_at, last_access, kind FROM memories "
      "WHERE kind = 'WORKING' ORDER BY memory_id DESC LIMIT 1");
  REQUIRE (rows.size () == 1);
  REQUIRE (AnyToLongLong (rows[0].at ("start_ts"))
           == static_cast<long long> (replay_ts));
  REQUIRE (AnyToLongLong (rows[0].at ("created_at"))
           == static_cast<long long> (replay_ts));
  REQUIRE (AnyToLongLong (rows[0].at ("last_access"))
           == static_cast<long long> (replay_ts));
  REQUIRE (std::any_cast<std::string> (rows[0].at ("kind")) == "WORKING");
}

TEST_CASE ("internal replay text ingress can skip context hydration",
           "[cortext][replay][hydration]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();
  const std::string models_dir = RepoModelsDir ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path, models_dir));
  REQUIRE (ctx != nullptr);

  const std::uint64_t replay_ts = 1573184762000ULL;
  auto ingest = cortext::internal::ReplayIngress::ProcessTextAt (
      *ctx, "non hydrated replay text still persists memory", "stream/main",
      replay_ts, cortext::Retention::Durable, false);
  REQUIRE (ingest.working_memory.empty ());
  REQUIRE (ingest.retrieved_memory.empty ());
  REQUIRE (ingest.output.stored_memory_id.has_value ());
  REQUIRE (ingest.output.stored_signal_id.has_value ());
  ctx->Flush ();

  auto unique_store = cortext::SQLiteStore::Create (db_path.c_str ());
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  auto rows = store->Execute (
      "SELECT timestamp, signal_id FROM signals "
      "WHERE memory_id = ? ORDER BY signal_id DESC LIMIT 1",
      { *ingest.output.stored_memory_id });
  REQUIRE (rows.size () == 1);
  REQUIRE (AnyToLongLong (rows[0].at ("timestamp"))
           == static_cast<long long> (replay_ts));
  REQUIRE (AnyToLongLong (rows[0].at ("signal_id"))
           == ingest.output.stored_signal_id);
  auto wm_rows = store->Execute (
      "SELECT start_ts, kind FROM memories "
      "WHERE kind = 'WORKING' ORDER BY memory_id DESC LIMIT 1");
  REQUIRE (wm_rows.size () == 1);
  REQUIRE (AnyToLongLong (wm_rows[0].at ("start_ts"))
           == static_cast<long long> (replay_ts));
  REQUIRE (std::any_cast<std::string> (wm_rows[0].at ("kind")) == "WORKING");

  auto probe = cortext::internal::ReplayIngress::ProcessTextAt (
      *ctx, "hydrated replay probe sees existing working memory",
      "stream/main", replay_ts + 1000ULL, cortext::Retention::Ephemeral,
      true);
  REQUIRE_FALSE (probe.working_memory.empty ());
  REQUIRE_FALSE (probe.output.stored_memory_id.has_value ());
  REQUIRE_FALSE (probe.output.stored_signal_id.has_value ());
}

TEST_CASE ("working memory persistence does not rewrite clean slot embeddings",
           "[cortext][replay][working_memory][performance]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();
  const std::string models_dir = RepoModelsDir ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path, models_dir));
  REQUIRE (ctx != nullptr);

  const std::uint64_t replay_ts = 1573184762000ULL;
  for (int i = 0; i < 6; ++i)
    {
      const auto ts = replay_ts + static_cast<std::uint64_t> (i) * 1000ULL;
      REQUIRE_NOTHROW (
          cortext::internal::ReplayIngress::ProcessTextAt (
              *ctx,
              "bounded replay working memory persistence message "
                  + std::to_string (i),
              "stream/main", ts, cortext::Retention::Durable, false));
    }
  ctx->Flush ();

  auto unique_store = cortext::SQLiteStore::Create (db_path.c_str ());
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  auto rows = store->Execute (
      "SELECT COUNT(*) AS c FROM embeddings e "
      "WHERE NOT EXISTS ("
      "  SELECT 1 FROM memories m WHERE m.embedding_id = e.embedding_id"
      ") AND NOT EXISTS ("
      "  SELECT 1 FROM signals s WHERE s.embedding_id = e.embedding_id"
      ")");
  REQUIRE (rows.size () == 1);
  auto wm_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM memories WHERE kind = 'WORKING'");
  REQUIRE (wm_rows.size () == 1);
  REQUIRE (AnyToLongLong (rows[0].at ("c"))
           <= AnyToLongLong (wm_rows[0].at ("c")));
}

TEST_CASE ("replay clock override preserves working memory on reopen",
           "[cortext][replay][clock][working_memory]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();
  const std::string models_dir = RepoModelsDir ();
  const std::uint64_t replay_ts = 1573184762000ULL;

  {
    auto clock = std::make_shared<cortext::FixedClock> (replay_ts);
    auto ctx = cortext::Cortext::Create (cfg, db_path, models_dir, clock);
    REQUIRE (ctx != nullptr);
    REQUIRE_NOTHROW (ctx->ProcessTextAt (
        "timestamped working memory replay survives reopen", "stream/main",
        replay_ts));
    ctx->Flush ();
  }

  {
    auto clock = std::make_shared<cortext::FixedClock> (replay_ts + 1000ULL);
    auto reopened = cortext::Cortext::Create (cfg, db_path, models_dir,
                                              clock);
    REQUIRE (reopened != nullptr);
    auto probe = reopened->ProcessTextAt (
        "follow up for the replay working memory", "stream/main",
        replay_ts + 2000ULL, cortext::Retention::Ephemeral);
    REQUIRE_FALSE (probe.working_memory.empty ());
  }
}

TEST_CASE ("internal replay ingress preserves consolidation event timestamps",
           "[cortext][replay][consolidation]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();
  const std::string models_dir = RepoModelsDir ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path, models_dir));
  REQUIRE (ctx != nullptr);

  const std::uint64_t source_ts = 1573184762000ULL;
  for (int i = 0; i < 24; ++i)
    {
      REQUIRE_NOTHROW (ctx->ProcessTextAt (
          "shared replay consolidation topic package pickup dinner logistics",
          "stream/main", source_ts + static_cast<std::uint64_t> (i) * 1000ULL));
    }
  ctx->Flush ();

  const std::uint64_t consolidation_ts = source_ts + 86400000ULL;
  REQUIRE_NOTHROW (
      cortext::internal::ReplayIngress::ConsolidateAt (*ctx, consolidation_ts));
  ctx->Flush ();

  auto unique_store = cortext::SQLiteStore::Create (db_path.c_str ());
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  auto rows = store->Execute (
      "SELECT start_ts, created_at, source_id, kind FROM memories "
      "WHERE kind = 'ASSOCIATION' AND source_id LIKE 'association_%' "
      "ORDER BY memory_id DESC LIMIT 1");
  REQUIRE (rows.size () == 1);
  REQUIRE (AnyToLongLong (rows[0].at ("start_ts"))
           == static_cast<long long> (consolidation_ts));
  REQUIRE (AnyToLongLong (rows[0].at ("created_at"))
           == static_cast<long long> (consolidation_ts));
  REQUIRE (std::any_cast<std::string> (rows[0].at ("kind"))
           == "ASSOCIATION");
}

TEST_CASE ("Cortext::Create succeeds even when models dir is missing",
           "[cortext][models]")
{
  cortext::Cortext::Config cfg;
  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (
      ctx = cortext::Cortext::Create (cfg, ":memory:", "models/does-not-exist"));
  REQUIRE (ctx != nullptr);
}

TEST_CASE ("Cortext C ABI stubs return success", "[cortext][capi][stub]")
{
  const std::string models_dir = RepoModelsDir ();
  cortext_handle h = cortext_create_with_models (
      0.5, 0.5, 0.5, ":memory:", models_dir.c_str ());
  REQUIRE (h != nullptr);

  const int text_rc = cortext_process_text (h, "hello", "test");
  if (text_rc == 0)
    {
      const float pcm[2] = { 0.0f, 0.0f };
      const int audio_rc = cortext_process_audio (h, pcm, 2, "test");

      const std::uint8_t px[3] = { 0, 0, 0 };
      const int image_rc = cortext_process_image (h, px, 1, 1, 3, "test");

      REQUIRE (cortext_consolidate (h) == 0);
      REQUIRE (cortext_flush (h) == 0);

      REQUIRE ((audio_rc == 0 || audio_rc == 2));
      REQUIRE ((image_rc == 0 || image_rc == 2));
    }
  else
    {
      // Without ORT (or without models), C API should return non-zero (but not
      // crash).
      REQUIRE (text_rc != 0);
    }

  cortext_free (h);
}

TEST_CASE ("Cortext C ABI rejects invalid store callbacks",
           "[cortext][capi][store]")
{
  cortext_config cfg{};
  cortext_config_init (&cfg);
  REQUIRE (cortext_create_with_store_callbacks (
               &cfg, nullptr, nullptr, "models/does-not-exist")
           == nullptr);

  cortext_db_callbacks callbacks{};
  callbacks.struct_size = sizeof (callbacks);
  REQUIRE (cortext_create_with_store_callbacks (
               &cfg, &callbacks, nullptr, "models/does-not-exist")
           == nullptr);
}

TEST_CASE ("Cortext hydrates sqlite-objstore payloads",
           "[cortext][objstore][hydration]")
{
  ScopedTempDb temp_db;
  const auto &db_path = temp_db.path ();
  auto store = cortext::SQLiteStore::Create (db_path);

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  struct Sample
  {
    long long embedding_id;
    std::string modality;
    std::string mime;
    std::vector<unsigned char> payload;
  };
  const std::vector<Sample> samples = {
    { 42, "audio", cortext::Cortext::MakeAudioMimePcmF32 (),
      SampleWavBytes () },
    { 43, "video", "video/mp4", SampleMp4Bytes () },
  };

  // Create 256D embedding for vec0 compatibility
  constexpr int kEmbeddingDim = 256;
  std::vector<float> embedding (kEmbeddingDim, 0.0f);
  embedding[0] = 0.0f;
  embedding[1] = 1.0f;
  embedding[2] = 2.0f;
  embedding[3] = 3.0f;

  std::unordered_map<long long, std::vector<unsigned char>> expected_payloads;
  std::unordered_map<long long, std::string> expected_mimes;

  for (const auto &sample : samples)
    {
      auto blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                       { sample.payload });
      REQUIRE (blob_rows.size () == 1);
      const auto blob_id = BlobFromAny (blob_rows[0].at ("id"));
      REQUIRE (!blob_id.empty ());

      // v2: Insert into embeddings (minimal vec0 table)
      store->Execute (
          "INSERT INTO embeddings (embedding_id, embedding, created_at) "
          "VALUES (?, ?, ?)",
          { sample.embedding_id, embedding, 0LL });

      // v2: Insert into memories table with comprehensive metadata
      store->Execute (
          "INSERT INTO memories (memory_id, embedding_id, source_id, kind, blob_id, "
          "start_ts, end_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
          "VALUES (?, ?, 'test', 'LONG_TERM', ?, 0, 1000, 1, ?, 0.5, 0.5, 1.0, 0)",
          { sample.embedding_id, sample.embedding_id, blob_id, sample.modality });

      // v2: Insert into signals table for signal-level mimetype
      // memory_id is needed for LoadSignalBlobs to find the blobs
      store->Execute ("INSERT INTO signals (memory_id, embedding_id, source_id, timestamp, modality, "
                      "mime, blob_id, serial_position, created_at)"
                      " VALUES (?, ?, 'test', ?, ?, ?, ?, 0, ?)",
                      {
                          sample.embedding_id, // memory_id matches embedding_id in this test
                          sample.embedding_id,
                          500LL,
                          sample.modality,
                          sample.mime,
                          blob_id,
                          500LL,
                      });
      expected_payloads[sample.embedding_id] = sample.payload;
      expected_mimes[sample.embedding_id] = sample.mime;
    }

  cortext::Cortext::Config cfg;
  auto ctx = cortext::Cortext::Create (cfg, db_path, RepoModelsDir ());
  REQUIRE (ctx != nullptr);

  std::vector<long long> candidate_ids;
  for (const auto &sample : samples)
    {
      candidate_ids.push_back (sample.embedding_id);
    }
  auto hydrated = ctx->DebugHydrateForTest (candidate_ids, {});
  REQUIRE (hydrated.retrieved_memory.size () == samples.size ());
  for (const auto &memory : hydrated.retrieved_memory)
    {
      REQUIRE (expected_payloads.count (memory.id) == 1);
      REQUIRE (expected_mimes.count (memory.id) == 1);
      // v2: content is now a vector of blobs (one per signal)
      REQUIRE (memory.content.size () == 1);
      REQUIRE (memory.content[0] == expected_payloads.at (memory.id));
      REQUIRE (memory.mimetype == expected_mimes.at (memory.id));
    }

  {
    cortext::testing::ScopedEnvVar disable_source_blobs (
        "CORTEXT_DISABLE_SOURCE_BLOBS", "1");
    auto stripped = ctx->DebugHydrateForTest (candidate_ids, {});
    REQUIRE (stripped.retrieved_memory.size () == samples.size ());
    for (const auto &memory : stripped.retrieved_memory)
      {
        REQUIRE (expected_payloads.count (memory.id) == 1);
        REQUIRE (expected_mimes.count (memory.id) == 1);
        REQUIRE (memory.content.empty ());
        REQUIRE (memory.mimetype == expected_mimes.at (memory.id));
      }
  }
}

TEST_CASE ("Cortext filters mixed signal blobs to the memory surface",
           "[cortext][hydration]")
{
  ScopedTempDb temp_db;
  const auto &db_path = temp_db.path ();
  auto store = cortext::SQLiteStore::Create (db_path);
  cortext::testing::InitializeCoreSchema (*store);

  constexpr int kEmbeddingDim = 256;
  std::vector<float> embedding (kEmbeddingDim, 0.0f);
  embedding[0] = 1.0f;

  const std::string source_text = "Slow delivery on amazon";
  const std::vector<unsigned char> text_payload (source_text.begin (),
                                                 source_text.end ());
  const std::vector<unsigned char> image_payload = {
    0x00, 0x01, 0x02, 0x03, 0xff, 0x00, 0x10, 0x20,
  };

  auto text_blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                        { text_payload });
  REQUIRE (text_blob_rows.size () == 1);
  const auto text_blob_id = BlobFromAny (text_blob_rows[0].at ("id"));
  REQUIRE (!text_blob_id.empty ());

  auto image_blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                         { image_payload });
  REQUIRE (image_blob_rows.size () == 1);
  const auto image_blob_id = BlobFromAny (image_blob_rows[0].at ("id"));
  REQUIRE (!image_blob_id.empty ());

  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?)",
      { 701LL, embedding, 0LL });

  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, "
      "start_ts, end_ts, n_signals, modality, s_max, s_avg, strength, "
      "created_at) "
      "VALUES (701, 701, 'test/source', 'LONG_TERM', 100, 200, 2, 'text', "
      "0.5, 0.5, 1.0, 0)");

  store->Execute (
      "INSERT INTO signals (memory_id, embedding_id, source_id, timestamp, "
      "modality, mime, blob_id, serial_position, created_at) "
      "VALUES (701, 701, 'test/source', 100, 'text', 'text/plain', ?, 0, "
      "100)",
      { text_blob_id });
  store->Execute (
      "INSERT INTO signals (memory_id, embedding_id, source_id, timestamp, "
      "modality, mime, blob_id, serial_position, created_at) "
      "VALUES (701, 701, 'test/source', 100, 'image', "
      "'image/raw;width=224;height=224;channels=3', ?, 1, 100)",
      { image_blob_id });

  cortext::Cortext::Config cfg;
  auto ctx = cortext::Cortext::Create (cfg, db_path, RepoModelsDir ());
  REQUIRE (ctx != nullptr);

  auto hydrated = ctx->DebugHydrateForTest ({ 701LL }, {});
  REQUIRE (hydrated.retrieved_memory.size () == 1);
  const auto &memory = hydrated.retrieved_memory[0];
  REQUIRE (memory.modality == "text");
  REQUIRE (memory.mimetype == "text/plain");
  REQUIRE (memory.content.size () == 1);
  REQUIRE (TextFromMemory (memory) == source_text);
}

TEST_CASE ("Cortext hydrates SoftAnchor metadata with retrieved memories",
           "[cortext][hydration][soft_anchor]")
{
  ScopedTempDb temp_db;
  const auto &db_path = temp_db.path ();
  auto store = cortext::SQLiteStore::Create (db_path);
  cortext::testing::InitializeCoreSchema (*store);

  constexpr int kEmbeddingDim = 256;
  std::vector<float> embedding (kEmbeddingDim, 0.0f);
  embedding[0] = 1.0f;

  const std::string source_text = "I went to Jared's house yesterday.";
  const std::vector<unsigned char> payload (source_text.begin (),
                                            source_text.end ());
  auto blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                   { payload });
  REQUIRE (blob_rows.size () == 1);
  const auto blob_id = BlobFromAny (blob_rows[0].at ("id"));
  REQUIRE (!blob_id.empty ());

  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?)",
      { 501LL, embedding, 1000LL });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, blob_id, "
      "start_ts, end_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, 'stream/main', 'LONG_TERM', ?, 1000, 1000, 1, 'text', "
      "0.5, 0.5, 1.0, 1000)",
      { 501LL, 501LL, blob_id });
  store->Execute (
      "INSERT INTO signals (memory_id, embedding_id, source_id, timestamp, "
      "modality, mime, blob_id, serial_position, created_at) "
      "VALUES (?, ?, 'stream/main', 1000, 'text', 'text/plain', ?, 0, 1000)",
      { 501LL, 501LL, blob_id });

  store->Execute (
      "INSERT INTO soft_anchor_links "
      "(memory_id, anchor_id, anchor_strength, anchor_label, memory_tier, "
      " score, margin, entropy, support_count, contradiction_count, "
      " created_step, updated_step, created_at) "
      "VALUES "
      "(501, 'anchor-low', 0.20, 'tentative', 'STM', 0.30, 0.04, 0.70, "
      " 1, 0, 1, 1, 1000),"
      "(501, 'anchor-top', 1.20, 'durable', 'WM', -0.40, 0.22, 1.80, "
      " 3, 0, 1, 3, 1000),"
      "(501, 'anchor-mid', 0.73, 'ambiguous', 'STM', 0.64, 0.02, 0.51, "
      " 2, 0, 1, 2, 1000),"
      "(501, 'anchor-third', 0.42, 'tentative', 'LTM', 0.44, 0.01, 0.62, "
      " 1, 0, 1, 2, 1000),"
      "(501, 'anchor-rejected', 0.99, 'rejected', 'WM', 0.99, 0.60, 0.10, "
      " 1, 1, 1, 4, 1000)");

  cortext::Cortext::Config cfg;
  auto ctx = cortext::Cortext::Create (cfg, db_path, RepoModelsDir ());
  REQUIRE (ctx != nullptr);

  auto hydrated = ctx->DebugHydrateForTest ({ 501LL }, {});
  REQUIRE (hydrated.retrieved_memory.size () == 1);
  const auto &memory = hydrated.retrieved_memory.front ();
  REQUIRE (TextFromMemory (memory) == source_text);
  REQUIRE (memory.soft_anchors.size () == 3);

  REQUIRE (memory.soft_anchors[0].id == "anchor-top");
  REQUIRE (memory.soft_anchors[0].strength == 1.0);
  REQUIRE (memory.soft_anchors[0].likelihood == 1.0);
  REQUIRE (memory.soft_anchors[0].label == "durable");
  REQUIRE (memory.soft_anchors[0].tier == "WM");
  REQUIRE (memory.soft_anchors[0].score == 0.0);
  REQUIRE (memory.soft_anchors[0].entropy == 1.0);

  REQUIRE (memory.soft_anchors[1].id == "anchor-mid");
  REQUIRE (memory.soft_anchors[2].id == "anchor-third");
}

TEST_CASE ("Cortext expands internal retrieval nodes to linked text memories",
           "[cortext][hydration][retrieval]")
{
  ScopedTempDb temp_db;
  const auto &db_path = temp_db.path ();
  auto store = cortext::SQLiteStore::Create (db_path);
  cortext::testing::InitializeCoreSchema (*store);

  constexpr int kEmbeddingDim = 256;
  std::vector<float> embedding (kEmbeddingDim, 0.0f);
  embedding[0] = 1.0f;

  const std::string source_text = "I am thinking about messaging Jared.";
  const std::vector<unsigned char> payload (source_text.begin (),
                                            source_text.end ());
  auto blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                   { payload });
  REQUIRE (blob_rows.size () == 1);
  const auto blob_id = BlobFromAny (blob_rows[0].at ("id"));
  REQUIRE (!blob_id.empty ());

  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?), (?, ?, ?), (?, ?, ?)",
      { 100LL, embedding, 1000LL, 200LL, embedding, 1000LL, 300LL,
        embedding, 1000LL });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, blob_id, "
      "start_ts, end_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, 'stream/main', 'LONG_TERM', ?, 1000, 1000, 1, 'text', "
      "0.5, 0.5, 1.0, 1000)",
      { 100LL, 100LL, blob_id });
  store->Execute (
      "INSERT INTO signals (memory_id, embedding_id, source_id, timestamp, "
      "modality, mime, blob_id, serial_position, created_at) "
      "VALUES (?, ?, 'stream/main', 1000, 'text', 'text/plain', ?, 0, 1000)",
      { 100LL, 100LL, blob_id });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, label, "
      "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, 'associative_cue_test', 'ASSOCIATION', "
      "'associative cue test', 1000, 1, 'text', 0.5, 0.5, 1.0, 1000), "
      "(?, ?, 'jared', 'LABEL', 'Jared', 1000, 1, 'text', 0.5, 0.5, 1.0, 1000)",
      { 200LL, 200LL, 300LL, 300LL });
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'derived_from', 1.0), (?, ?, 'reinforces', 1.0)",
      { 200LL, 100LL, 100LL, 300LL });

  cortext::Cortext::Config cfg;
  auto ctx = cortext::Cortext::Create (cfg, db_path, RepoModelsDir ());
  REQUIRE (ctx != nullptr);

  auto cue_hydrated = ctx->DebugHydrateForTest ({ 200LL }, {});
  REQUIRE (cue_hydrated.retrieved_memory.size () == 1);
  REQUIRE (cue_hydrated.retrieved_memory[0].id == 100LL);
  REQUIRE (TextFromMemory (cue_hydrated.retrieved_memory[0]) == source_text);

  auto label_hydrated = ctx->DebugHydrateForTest ({ 300LL }, {});
  REQUIRE (label_hydrated.retrieved_memory.size () == 1);
  REQUIRE (label_hydrated.retrieved_memory[0].id == 100LL);
  REQUIRE (TextFromMemory (label_hydrated.retrieved_memory[0]) == source_text);
}

TEST_CASE ("Cortext expands durable association retrieval nodes even when the cue has text",
           "[cortext][hydration][retrieval]")
{
  ScopedTempDb temp_db;
  const auto &db_path = temp_db.path ();
  auto store = cortext::SQLiteStore::Create (db_path);
  cortext::testing::InitializeCoreSchema (*store);

  constexpr int kEmbeddingDim = 256;
  std::vector<float> embedding (kEmbeddingDim, 0.0f);
  embedding[0] = 1.0f;

  const std::string source_text = "Jamie mentioned the art show downtown.";
  const std::string cue_text = "durable cue about art";
  const std::vector<unsigned char> source_payload (source_text.begin (),
                                                   source_text.end ());
  const std::vector<unsigned char> cue_payload (cue_text.begin (),
                                                cue_text.end ());
  auto source_blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                          { source_payload });
  auto cue_blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                       { cue_payload });
  REQUIRE (source_blob_rows.size () == 1);
  REQUIRE (cue_blob_rows.size () == 1);
  const auto source_blob_id = BlobFromAny (source_blob_rows[0].at ("id"));
  const auto cue_blob_id = BlobFromAny (cue_blob_rows[0].at ("id"));
  REQUIRE (!source_blob_id.empty ());
  REQUIRE (!cue_blob_id.empty ());

  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?), (?, ?, ?)",
      { 100LL, embedding, 1000LL, 200LL, embedding, 1000LL });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, blob_id, "
      "start_ts, end_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, 'stream/main', 'LONG_TERM', ?, 1000, 1000, 1, 'text', "
      "0.5, 0.5, 1.0, 1000), "
      "(?, ?, 'associative_cue_test', 'ASSOCIATION', ?, 1000, 1000, 1, "
      "'text', 0.5, 0.5, 1.0, 1000)",
      { 100LL, 100LL, source_blob_id, 200LL, 200LL, cue_blob_id });
  store->Execute (
      "INSERT INTO signals (memory_id, embedding_id, source_id, timestamp, "
      "modality, mime, blob_id, serial_position, created_at) "
      "VALUES (?, ?, 'stream/main', 1000, 'text', 'text/plain', ?, 0, 1000), "
      "(?, ?, 'associative_cue_test', 1000, 'text', 'text/plain', ?, 0, 1000)",
      { 100LL, 100LL, source_blob_id, 200LL, 200LL, cue_blob_id });
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'derived_from', 1.0)",
      { 200LL, 100LL });

  cortext::Cortext::Config cfg;
  auto ctx = cortext::Cortext::Create (cfg, db_path, RepoModelsDir ());
  REQUIRE (ctx != nullptr);

  auto cue_hydrated = ctx->DebugHydrateForTest ({ 200LL }, {});
  REQUIRE (cue_hydrated.retrieved_memory.size () == 1);
  REQUIRE (cue_hydrated.retrieved_memory[0].id == 100LL);
  REQUIRE (TextFromMemory (cue_hydrated.retrieved_memory[0]) == source_text);
}

TEST_CASE ("Cortext expands durable label retrieval nodes through association sources",
           "[cortext][hydration][retrieval]")
{
  ScopedTempDb temp_db;
  const auto &db_path = temp_db.path ();
  auto store = cortext::SQLiteStore::Create (db_path);
  cortext::testing::InitializeCoreSchema (*store);

  constexpr int kEmbeddingDim = 256;
  std::vector<float> embedding (kEmbeddingDim, 0.0f);
  embedding[0] = 1.0f;

  const std::string source_text = "Jamie talked about the gallery opening.";
  const std::vector<unsigned char> source_payload (source_text.begin (),
                                                   source_text.end ());
  auto source_blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                          { source_payload });
  REQUIRE (source_blob_rows.size () == 1);
  const auto source_blob_id = BlobFromAny (source_blob_rows[0].at ("id"));
  REQUIRE (!source_blob_id.empty ());

  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?), (?, ?, ?), (?, ?, ?)",
      { 100LL, embedding, 1000LL, 200LL, embedding, 1000LL, 300LL,
        embedding, 1000LL });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, blob_id, "
      "start_ts, end_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, 'stream/main', 'LONG_TERM', ?, 1000, 1000, 1, 'text', "
      "0.5, 0.5, 1.0, 1000)",
      { 100LL, 100LL, source_blob_id });
  store->Execute (
      "INSERT INTO signals (memory_id, embedding_id, source_id, timestamp, "
      "modality, mime, blob_id, serial_position, created_at) "
      "VALUES (?, ?, 'stream/main', 1000, 'text', 'text/plain', ?, 0, 1000)",
      { 100LL, 100LL, source_blob_id });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, label, "
      "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, 'associative_cue_test', 'ASSOCIATION', "
      "'associative cue test', 1000, 1, 'text', 0.5, 0.5, 1.0, 1000), "
      "(?, ?, 'gallery', 'LABEL', 'gallery', 1000, 1, 'text', 0.5, 0.5, "
      "1.0, 1000)",
      { 200LL, 200LL, 300LL, 300LL });
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'derived_from', 1.0), (?, ?, 'has_label', 1.0)",
      { 200LL, 100LL, 200LL, 300LL });

  cortext::Cortext::Config cfg;
  auto ctx = cortext::Cortext::Create (cfg, db_path, RepoModelsDir ());
  REQUIRE (ctx != nullptr);

  auto label_hydrated = ctx->DebugHydrateForTest ({ 300LL }, {});
  REQUIRE (label_hydrated.retrieved_memory.size () == 1);
  REQUIRE (label_hydrated.retrieved_memory[0].id == 100LL);
  REQUIRE (TextFromMemory (label_hydrated.retrieved_memory[0]) == source_text);
}

TEST_CASE ("Cortext orders durable label source hydration by query similarity",
           "[cortext][hydration][retrieval]")
{
  ScopedTempDb temp_db;
  const auto &db_path = temp_db.path ();
  auto store = cortext::SQLiteStore::Create (db_path);
  cortext::testing::InitializeCoreSchema (*store);

  constexpr int kEmbeddingDim = 256;
  std::vector<float> query_embedding (kEmbeddingDim, 0.0f);
  query_embedding[0] = 1.0f;
  std::vector<float> unrelated_embedding (kEmbeddingDim, 0.0f);
  unrelated_embedding[1] = 1.0f;

  const std::string relevant_text = "Jamie talked about noodles at Moon Plaza.";
  const std::string recent_text = "Jamie mentioned a different errand.";
  const std::vector<unsigned char> relevant_payload (relevant_text.begin (),
                                                     relevant_text.end ());
  const std::vector<unsigned char> recent_payload (recent_text.begin (),
                                                   recent_text.end ());
  auto relevant_blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                            { relevant_payload });
  auto recent_blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                          { recent_payload });
  REQUIRE (relevant_blob_rows.size () == 1);
  REQUIRE (recent_blob_rows.size () == 1);
  const auto relevant_blob_id = BlobFromAny (relevant_blob_rows[0].at ("id"));
  const auto recent_blob_id = BlobFromAny (recent_blob_rows[0].at ("id"));
  REQUIRE (!relevant_blob_id.empty ());
  REQUIRE (!recent_blob_id.empty ());

  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?), (?, ?, ?), (?, ?, ?), (?, ?, ?)",
      { 100LL, query_embedding, 1000LL, 101LL, unrelated_embedding, 2000LL,
        200LL, unrelated_embedding, 2000LL, 300LL, unrelated_embedding,
        2000LL });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, blob_id, "
      "start_ts, end_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, 'stream/main', 'LONG_TERM', ?, 1000, 1000, 1, 'text', "
      "0.5, 0.5, 1.0, 1000), "
      "(?, ?, 'stream/main', 'LONG_TERM', ?, 2000, 2000, 1, 'text', "
      "0.5, 0.5, 1.0, 2000)",
      { 100LL, 100LL, relevant_blob_id, 101LL, 101LL, recent_blob_id });
  store->Execute (
      "INSERT INTO signals (memory_id, embedding_id, source_id, timestamp, "
      "modality, mime, blob_id, serial_position, created_at) "
      "VALUES (?, ?, 'stream/main', 1000, 'text', 'text/plain', ?, 0, 1000), "
      "(?, ?, 'stream/main', 2000, 'text', 'text/plain', ?, 0, 2000)",
      { 100LL, 100LL, relevant_blob_id, 101LL, 101LL, recent_blob_id });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, label, "
      "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, 'associative_cue_test', 'ASSOCIATION', "
      "'associative cue test', 2000, 1, 'text', 0.5, 0.5, 1.0, 2000), "
      "(?, ?, 'dinner errand', 'LABEL', 'dinner errand', 2000, 1, 'text', "
      "0.5, 0.5, 1.0, 2000)",
      { 200LL, 200LL, 300LL, 300LL });
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'derived_from', 1.0), (?, ?, 'derived_from', 1.0), "
      "       (?, ?, 'has_label', 1.0)",
      { 200LL, 100LL, 200LL, 101LL, 200LL, 300LL });

  cortext::Cortext::Config cfg;
  auto ctx = cortext::Cortext::Create (cfg, db_path, RepoModelsDir ());
  REQUIRE (ctx != nullptr);

  auto label_hydrated = ctx->DebugHydrateForTest ({ 300LL }, {},
                                                  query_embedding);
  REQUIRE (label_hydrated.retrieved_memory.size () >= 2);
  REQUIRE (label_hydrated.retrieved_memory[0].id == 100LL);
  REQUIRE (TextFromMemory (label_hydrated.retrieved_memory[0])
           == relevant_text);

  {
    cortext::testing::ScopedEnvVar disabled (
        "CORTEXT_DISABLE_QUERY_AWARE_LINKED_SOURCE_HYDRATION", "1");
    auto recency_hydrated = ctx->DebugHydrateForTest ({ 300LL }, {},
                                                      query_embedding);
    REQUIRE (recency_hydrated.retrieved_memory.size () >= 2);
    REQUIRE (recency_hydrated.retrieved_memory[0].id == 101LL);
  }
}

TEST_CASE ("Cortext caps linked source hydration with knob-derived compact limit",
           "[cortext][hydration][retrieval]")
{
  ScopedTempDb temp_db;
  const auto &db_path = temp_db.path ();
  auto store = cortext::SQLiteStore::Create (db_path);
  cortext::testing::InitializeCoreSchema (*store);

  cortext::testing::ScopedEnvVar clear_query_aware_disable (
      "CORTEXT_DISABLE_QUERY_AWARE_LINKED_SOURCE_HYDRATION");

  constexpr int kEmbeddingDim = 256;
  std::vector<float> embedding (kEmbeddingDim, 0.0f);
  embedding[0] = 1.0f;

  constexpr long long kAssociationId = 200LL;
  constexpr long long kLabelId = 300LL;
  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?), (?, ?, ?)",
      { kAssociationId, embedding, 1000LL, kLabelId, embedding, 1000LL });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, label, "
      "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, 'associative_cue_test', 'ASSOCIATION', "
      "'associative cue test', 1000, 1, 'text', 0.5, 0.5, 1.0, 1000), "
      "(?, ?, 'dense label', 'LABEL', 'dense label', 1000, 1, 'text', "
      "0.5, 0.5, 1.0, 1000)",
      { kAssociationId, kAssociationId, kLabelId, kLabelId });

  std::vector<long long> source_ids;
  for (long long i = 0; i < 5; ++i)
    {
      const long long memory_id = 100LL + i;
      const long long ts = 1000LL + i;
      const std::string source_text = "linked source " + std::to_string (i);
      const std::vector<unsigned char> payload (source_text.begin (),
                                                source_text.end ());
      auto blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                       { payload });
      REQUIRE (blob_rows.size () == 1);
      const auto blob_id = BlobFromAny (blob_rows[0].at ("id"));
      REQUIRE (!blob_id.empty ());

      store->Execute (
          "INSERT INTO embeddings (embedding_id, embedding, created_at) "
          "VALUES (?, ?, ?)",
          { memory_id, embedding, ts });
      store->Execute (
          "INSERT INTO memories (memory_id, embedding_id, source_id, kind, "
          "blob_id, start_ts, end_ts, n_signals, modality, s_max, s_avg, "
          "strength, created_at) "
          "VALUES (?, ?, 'stream/main', 'LONG_TERM', ?, ?, ?, 1, 'text', "
          "0.5, 0.5, 1.0, ?)",
          { memory_id, memory_id, blob_id, ts, ts, ts });
      store->Execute (
          "INSERT INTO signals (memory_id, embedding_id, source_id, timestamp, "
          "modality, mime, blob_id, serial_position, created_at) "
          "VALUES (?, ?, 'stream/main', ?, 'text', 'text/plain', ?, 0, ?)",
          { memory_id, memory_id, ts, blob_id, ts });
      store->Execute (
          "INSERT INTO associations(source_memory_id, target_memory_id, "
          "edge_type, weight) VALUES (?, ?, 'derived_from', 1.0)",
          { kAssociationId, memory_id });
      source_ids.push_back (memory_id);
    }
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight) VALUES (?, ?, 'has_label', 1.0)",
      { kAssociationId, kLabelId });

  cortext::Cortext::Config cfg;
  auto ctx = cortext::Cortext::Create (cfg, db_path, RepoModelsDir ());
  REQUIRE (ctx != nullptr);

  const int expected_limit = std::max (
      1, cortext::core::RetrievalGraphExpandedRagCompactItemLimit (
             cfg.focus, cfg.stability));
  REQUIRE (expected_limit < static_cast<int> (source_ids.size ()));

  auto hydrated = ctx->DebugHydrateForTest ({ kLabelId }, {}, embedding);
  REQUIRE (static_cast<int> (hydrated.retrieved_memory.size ())
           == expected_limit);
  for (const auto &memory : hydrated.retrieved_memory)
    {
      REQUIRE (std::find (source_ids.begin (), source_ids.end (), memory.id)
               != source_ids.end ());
      REQUIRE (!TextFromMemory (memory).empty ());
    }
}

TEST_CASE ("Cortext hides unresolved internal retrieval nodes",
           "[cortext][hydration][retrieval]")
{
  ScopedTempDb temp_db;
  const auto &db_path = temp_db.path ();
  auto store = cortext::SQLiteStore::Create (db_path);
  cortext::testing::InitializeCoreSchema (*store);

  constexpr int kEmbeddingDim = 256;
  std::vector<float> embedding (kEmbeddingDim, 0.0f);
  embedding[0] = 1.0f;

  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?), (?, ?, ?)",
      { 200LL, embedding, 1000LL, 300LL, embedding, 1000LL });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, label, "
      "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, 'associative_cue_test', 'ASSOCIATION', "
      "'associative cue test', 1000, 1, 'text', 0.5, 0.5, 1.0, 1000), "
      "(?, ?, 'jared', 'LABEL', 'Jared', 1000, 1, 'text', 0.5, 0.5, 1.0, 1000)",
      { 200LL, 200LL, 300LL, 300LL });
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'has_label', 1.0)",
      { 200LL, 300LL });

  cortext::Cortext::Config cfg;
  auto ctx = cortext::Cortext::Create (cfg, db_path, RepoModelsDir ());
  REQUIRE (ctx != nullptr);

  auto cue_hydrated = ctx->DebugHydrateForTest ({ 200LL }, {});
  REQUIRE (cue_hydrated.retrieved_memory.empty ());

  auto label_hydrated = ctx->DebugHydrateForTest ({ 300LL }, {});
  REQUIRE (label_hydrated.retrieved_memory.empty ());
}

TEST_CASE ("C API handles NULL inputs correctly", "[cortext][capi][safety]")
{
  SECTION ("cortext_version returns a non-empty string")
  {
    const char *version = cortext_version ();
    REQUIRE (version != nullptr);
    REQUIRE (std::strlen (version) > 0);
  }

  SECTION ("cortext_free accepts NULL handle")
  {
    REQUIRE_NOTHROW (cortext_free (nullptr));
  }

  SECTION ("cortext_process_text returns 1 for NULL handle")
  {
    CHECK (cortext_process_text (nullptr, "text", "src") == 1);
    REQUIRE (std::string (cortext_last_error ())
             == "handle, text, and source_id must all be non-NULL");
  }

  SECTION ("cortext_process_text returns 1 for NULL text")
  {
    ScopedTempDb temp_db;
    const auto models_dir = RepoModelsDir ();
    auto h = cortext_create_with_models (0.5, 0.5, 0.5, temp_db.path ().c_str (),
                                         models_dir.c_str ());
    REQUIRE (h != nullptr);
    CHECK (cortext_process_text (h, nullptr, "src") == 1);
    REQUIRE (std::string (cortext_last_error ())
             == "handle, text, and source_id must all be non-NULL");
    cortext_free (h);
  }

  SECTION ("cortext_process_text returns 1 for NULL source_id")
  {
    ScopedTempDb temp_db;
    const auto models_dir = RepoModelsDir ();
    auto h = cortext_create_with_models (0.5, 0.5, 0.5, temp_db.path ().c_str (),
                                         models_dir.c_str ());
    REQUIRE (h != nullptr);
    CHECK (cortext_process_text (h, "text", nullptr) == 1);
    REQUIRE (std::string (cortext_last_error ())
             == "handle, text, and source_id must all be non-NULL");
    cortext_free (h);
  }

  SECTION ("cortext_process_audio returns 1 for NULL pcm")
  {
    ScopedTempDb temp_db;
    const auto models_dir = RepoModelsDir ();
    auto h = cortext_create_with_models (0.5, 0.5, 0.5, temp_db.path ().c_str (),
                                         models_dir.c_str ());
    REQUIRE (h != nullptr);
    CHECK (cortext_process_audio (h, nullptr, 100, "src") == 1);
    REQUIRE (std::string (cortext_last_error ())
             == "handle, pcm, and source_id must all be non-NULL");
    cortext_free (h);
  }

  SECTION ("cortext_process_image returns 1 for NULL data")
  {
    ScopedTempDb temp_db;
    const auto models_dir = RepoModelsDir ();
    auto h = cortext_create_with_models (0.5, 0.5, 0.5, temp_db.path ().c_str (),
                                         models_dir.c_str ());
    REQUIRE (h != nullptr);
    CHECK (cortext_process_image (h, nullptr, 10, 10, 3, "src") == 1);
    REQUIRE (std::string (cortext_last_error ())
             == "handle, data, and source_id must all be non-NULL");
    cortext_free (h);
  }

  SECTION ("cortext_embed_text_json reports NULL arguments")
  {
    char *json_ptr = cortext_embed_text_json (nullptr, "hello");
    REQUIRE (json_ptr == nullptr);
    REQUIRE (std::string (cortext_last_error ())
             == "handle and text must both be non-NULL");
  }

  SECTION ("cortext_embed_audio_json reports NULL arguments")
  {
    ScopedTempDb temp_db;
    const auto models_dir = RepoModelsDir ();
    auto h = cortext_create_with_models (0.5, 0.5, 0.5, temp_db.path ().c_str (),
                                         models_dir.c_str ());
    REQUIRE (h != nullptr);
    char *json_ptr = cortext_embed_audio_json (h, nullptr, 100);
    REQUIRE (json_ptr == nullptr);
    REQUIRE (std::string (cortext_last_error ())
             == "handle and pcm must both be non-NULL");
    cortext_free (h);
  }

  SECTION ("cortext_embed_image_json reports NULL arguments")
  {
    ScopedTempDb temp_db;
    const auto models_dir = RepoModelsDir ();
    auto h = cortext_create_with_models (0.5, 0.5, 0.5, temp_db.path ().c_str (),
                                         models_dir.c_str ());
    REQUIRE (h != nullptr);
    char *json_ptr = cortext_embed_image_json (h, nullptr, 10, 10, 3);
    REQUIRE (json_ptr == nullptr);
    REQUIRE (std::string (cortext_last_error ())
             == "handle and data must both be non-NULL");
    cortext_free (h);
  }

  SECTION ("cortext_consolidate returns 1 for NULL handle")
  {
    CHECK (cortext_consolidate (nullptr) == 1);
    REQUIRE (std::string (cortext_last_error ()) == "handle must not be NULL");
  }

  SECTION ("cortext_flush returns 1 for NULL handle")
  {
    CHECK (cortext_flush (nullptr) == 1);
    REQUIRE (std::string (cortext_last_error ()) == "handle must not be NULL");
  }

  SECTION ("cortext_reset returns 1 for NULL handle")
  {
    CHECK (cortext_reset (nullptr) == 1);
    REQUIRE (std::string (cortext_last_error ()) == "handle must not be NULL");
  }

  SECTION ("cortext_reset keeps handle usable")
  {
    ScopedTempDb temp_db;
    const auto models_dir = RepoModelsDir ();
    auto h = cortext_create_with_models (0.5, 0.5, 0.5, temp_db.path ().c_str (),
                                         models_dir.c_str ());
    REQUIRE (h != nullptr);

    REQUIRE (cortext_process_text (h, "before reset", "reset/source") == 0);
    REQUIRE (cortext_reset (h) == 0);
    REQUIRE (cortext_last_error () == nullptr);

    char *json_ptr
        = cortext_process_text_json (h, "after reset", "reset/source");
    REQUIRE (json_ptr != nullptr);

    auto parsed = nlohmann::json::parse (json_ptr);
    REQUIRE (parsed.contains ("output"));
    REQUIRE (parsed.at ("output").contains ("stored_signal_id"));
    REQUIRE (parsed.at ("output").at ("stored_signal_id").is_number_integer ());
    cortext_string_free (json_ptr);

    auto store = std::shared_ptr<cortext::Store> (
        cortext::SQLiteStore::Create (temp_db.path ().c_str ()));
    const auto rows = store->Execute ("SELECT COUNT(*) AS n FROM signals");
    REQUIRE (rows.size () == 1);
    REQUIRE (AnyToLongLong (rows[0].at ("n")) >= 2);

    cortext_free (h);
  }

  SECTION ("cortext_config_init populates defaults and create_with_config works")
  {
    ScopedTempDb temp_db;
    cortext_config cfg{};
    cortext_config_init (&cfg);
    REQUIRE (cfg.struct_size == sizeof (cortext_config));
    cfg.focus = 0.7;
    cfg.sensitivity = 0.4;
    cfg.stability = 0.9;

    const auto models_dir = RepoModelsDir ();
    auto h = cortext_create_with_config (&cfg, temp_db.path ().c_str (),
                                         models_dir.c_str ());
    REQUIRE (h != nullptr);
    cortext_free (h);
  }

  SECTION ("JSON C API returns parseable context")
  {
    ScopedTempDb temp_db;
    const auto models_dir = RepoModelsDir ();
    auto h = cortext_create_with_models (0.5, 0.5, 0.5, temp_db.path ().c_str (),
                                         models_dir.c_str ());
    REQUIRE (h != nullptr);

    char *json_ptr
        = cortext_process_text_json (h, "json api stores ids", "json/source");
    REQUIRE (json_ptr != nullptr);

    auto parsed = nlohmann::json::parse (json_ptr);
    REQUIRE (parsed.contains ("output"));
    REQUIRE (parsed.at ("output").contains ("stored_memory_id"));
    REQUIRE (parsed.at ("output").contains ("stored_signal_id"));
    REQUIRE (parsed.at ("output").at ("stored_memory_id").is_number_integer ());
    REQUIRE (parsed.at ("output").at ("stored_signal_id").is_number_integer ());
    cortext_string_free (json_ptr);

    json_ptr = cortext_consolidate_json (h);
    REQUIRE (json_ptr != nullptr);

    parsed = nlohmann::json::parse (json_ptr);
    REQUIRE (parsed.contains ("working_memory"));
    REQUIRE (parsed.contains ("retrieved_memory"));
    REQUIRE (parsed.contains ("output"));
    REQUIRE (parsed.at ("working_memory").is_array ());
    REQUIRE (parsed.at ("retrieved_memory").is_array ());

    cortext_string_free (json_ptr);
    cortext_free (h);
  }

  SECTION ("Embed JSON C API returns parseable embedding")
  {
    ScopedTempDb temp_db;
    const auto models_dir = RepoModelsDir ();
    auto h = cortext_create_with_models (0.5, 0.5, 0.5, temp_db.path ().c_str (),
                                         models_dir.c_str ());
    REQUIRE (h != nullptr);

    char *json_ptr = cortext_embed_text_json (h, "json api embeds text");
    REQUIRE (json_ptr != nullptr);

    auto parsed = nlohmann::json::parse (json_ptr);
    REQUIRE (parsed.contains ("embedding"));
    REQUIRE (parsed.contains ("dimension"));
    REQUIRE (parsed.at ("embedding").is_array ());
    REQUIRE (parsed.at ("dimension").is_number_unsigned ());
    REQUIRE_FALSE (parsed.at ("embedding").empty ());
    REQUIRE (parsed.at ("dimension").get<std::size_t> ()
             == parsed.at ("embedding").size ());
    cortext_string_free (json_ptr);

    auto store = std::shared_ptr<cortext::Store> (
        cortext::SQLiteStore::Create (temp_db.path ().c_str ()));
    const auto rows = store->Execute ("SELECT COUNT(*) AS n FROM signals");
    REQUIRE (rows.size () == 1);
    REQUIRE (AnyToLongLong (rows[0].at ("n")) == 0);

    cortext_free (h);
  }

  SECTION ("JSON C API reports errors through cortext_last_error")
  {
    char *json_ptr = cortext_process_text_json (nullptr, "hello", "src");
    REQUIRE (json_ptr == nullptr);
    REQUIRE (std::string (cortext_last_error ())
             == "handle, text, and source_id must all be non-NULL");
  }
}
