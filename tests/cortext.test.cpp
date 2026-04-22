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

#include <cortext/capi.h>
#include <cortext/cortext.hpp>
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

bool
ImageBindAssetsPresent (const std::string &models_dir)
{
  namespace fs = std::filesystem;
  const fs::path md (models_dir);
  const bool has_text = fs::exists (md / "text_encoder_int8.onnx")
                        || fs::exists (md / "text_encoder.onnx");
  const bool has_audio = fs::exists (md / "audio_encoder_int8.onnx")
                         || fs::exists (md / "audio_encoder.onnx");
  const bool has_vision = fs::exists (md / "vision_encoder_int8.onnx")
                          || fs::exists (md / "vision_encoder.onnx");
  const bool has_bpe = fs::exists (md / "bpe" / "bpe_simple_vocab_16e6.txt.gz")
                       || fs::exists (md / "bpe_simple_vocab_16e6.txt.gz")
                       || fs::exists ("third_party/imagebind_assets/bpe/"
                                      "bpe_simple_vocab_16e6.txt.gz");
  return has_text && has_audio && has_vision && has_bpe;
}

std::string
RepoModelsImageBindDir ()
{
  namespace fs = std::filesystem;
  const fs::path root = fs::path (__FILE__).parent_path ().parent_path ();
  return (root / "models" / "imagebind").string ();
}
} // namespace

TEST_CASE ("Cortext C++ stub can be created and used", "[cortext][stub]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();
  const std::string models_dir = RepoModelsImageBindDir ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path, models_dir));
  REQUIRE (ctx != nullptr);

  // Encoding behavior depends on build flags and local environment:
  // - If ORT is enabled AND models/tokenizer assets are present: encoding should
  //   succeed.
  // - Otherwise: encoding should fail fast by throwing (but construction should
  //   still succeed).
#if defined(CORTEXT_ENABLE_IMAGEBIND_ORT) || defined(CORTEXT_ENABLE_EMBEDDINGGEMMA)
  const bool expect_encode_ok = ImageBindAssetsPresent (models_dir);
#else
  const bool expect_encode_ok = false;
#endif

  if (expect_encode_ok)
    {
      REQUIRE_NOTHROW ([&] {
        // Smoke test: verify the default text path and consolidation work.
        // Audio/image may be unsupported when EmbeddingGemma is the preferred
        // encoder, so only text is required here.
        auto out_text = ctx->ProcessText ("hello world", "test");
        (void)out_text.should_interrupt; // Algorithm-dependent, not asserted

        auto out_cons = ctx->Consolidate ();
        (void)out_cons.should_interrupt;
        ctx->Flush ();
      }());

#if defined(CORTEXT_ENABLE_IMAGEBIND_ORT) && !defined(CORTEXT_ENABLE_EMBEDDINGGEMMA)
      REQUIRE_NOTHROW ([&] {
        const float pcm[4] = { 0.0f, 0.1f, -0.1f, 0.0f };
        auto out_audio = ctx->ProcessAudio (pcm, 4, "test");
        (void)out_audio.should_interrupt;

        const std::uint8_t px[4] = { 0, 0, 0, 0 };
        auto out_image = ctx->ProcessImage (px, 1, 1, 4, "test");
        (void)out_image.should_interrupt;
      }());
#endif
    }
  else
    {
      REQUIRE_NOTHROW (ctx->ProcessText ("hello world", "test"));
    }

  // DB assertion: open the same DB and execute a trivial query successfully.
  auto uniq = cortext::SQLiteStore::Create (db_path.c_str ());
  auto store = std::shared_ptr<cortext::Store> (std::move (uniq));
  auto rows = store->Execute ("SELECT 1 AS ok");
  REQUIRE (rows.size () == 1);
}

TEST_CASE ("Cortext::Create succeeds even when models dir is missing",
           "[cortext][models]")
{
  cortext::Cortext::Config cfg;
  std::unique_ptr<cortext::Cortext> ctx;
#if defined(CORTEXT_DISABLE_LITERT)
  REQUIRE_NOTHROW (
      ctx = cortext::Cortext::Create (cfg, ":memory:", "models/does-not-exist"));
  REQUIRE (ctx != nullptr);
  REQUIRE_THROWS (ctx->ProcessText ("hello", "test"));
#else
  REQUIRE_THROWS (
      ctx = cortext::Cortext::Create (cfg, ":memory:", "models/does-not-exist"));
#endif
}

TEST_CASE ("Cortext can initialize with caller supplied store",
           "[cortext][store]")
{
#if defined(CORTEXT_DISABLE_LITERT)
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  cortext::Cortext::Config cfg;
  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (
      ctx = cortext::Cortext::Create (cfg, store, "models/does-not-exist"));
  REQUIRE (ctx != nullptr);

  auto rows = store->Execute (
      "SELECT name FROM sqlite_master WHERE type='table' AND name='state'",
      {});
  REQUIRE (rows.size () == 1);
#else
  SKIP ("Injected-store construction without local models is covered in "
        "CORTEXT_DISABLE_LITERT builds");
#endif
}

TEST_CASE ("Cortext C ABI stubs return success", "[cortext][capi][stub]")
{
  const std::string models_dir = RepoModelsImageBindDir ();
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

#if defined(CORTEXT_ENABLE_IMAGEBIND_ORT) && !defined(CORTEXT_ENABLE_EMBEDDINGGEMMA)
      REQUIRE (audio_rc == 0);
      REQUIRE (image_rc == 0);
#else
      REQUIRE ((audio_rc == 0 || audio_rc == 2));
      REQUIRE ((image_rc == 0 || image_rc == 2));
#endif
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
  auto ctx = cortext::Cortext::Create (cfg, db_path, RepoModelsImageBindDir ());
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
    const auto models_dir = RepoModelsImageBindDir ();
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
    const auto models_dir = RepoModelsImageBindDir ();
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
    const auto models_dir = RepoModelsImageBindDir ();
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
    const auto models_dir = RepoModelsImageBindDir ();
    auto h = cortext_create_with_models (0.5, 0.5, 0.5, temp_db.path ().c_str (),
                                         models_dir.c_str ());
    REQUIRE (h != nullptr);
    CHECK (cortext_process_image (h, nullptr, 10, 10, 3, "src") == 1);
    REQUIRE (std::string (cortext_last_error ())
             == "handle, data, and source_id must all be non-NULL");
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

  SECTION ("cortext_config_init populates defaults and create_with_config works")
  {
    ScopedTempDb temp_db;
    cortext_config cfg{};
    cortext_config_init (&cfg);
    REQUIRE (cfg.struct_size == sizeof (cortext_config));
    cfg.focus = 0.7;
    cfg.sensitivity = 0.4;
    cfg.stability = 0.9;

    const auto models_dir = RepoModelsImageBindDir ();
    auto h = cortext_create_with_config (&cfg, temp_db.path ().c_str (),
                                         models_dir.c_str ());
    REQUIRE (h != nullptr);
    cortext_free (h);
  }

  SECTION ("JSON C API returns parseable context")
  {
    ScopedTempDb temp_db;
    const auto models_dir = RepoModelsImageBindDir ();
    auto h = cortext_create_with_models (0.5, 0.5, 0.5, temp_db.path ().c_str (),
                                         models_dir.c_str ());
    REQUIRE (h != nullptr);

    char *json_ptr
        = cortext_consolidate_mode_json (h, CORTEXT_CONSOLIDATE_SHALLOW);
    REQUIRE (json_ptr != nullptr);

    const auto parsed = nlohmann::json::parse (json_ptr);
    REQUIRE (parsed.contains ("working_memory"));
    REQUIRE (parsed.contains ("retrieved_memory"));
    REQUIRE (parsed.contains ("output"));
    REQUIRE (parsed.at ("working_memory").is_array ());
    REQUIRE (parsed.at ("retrieved_memory").is_array ());

    cortext_string_free (json_ptr);
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
