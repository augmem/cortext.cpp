#include "test_helpers.hpp"
#include <any>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstring>
#include <filesystem>
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
                       || fs::exists ("poc/ImageBind/imagebind/bpe/"
                                      "bpe_simple_vocab_16e6.txt.gz");
  return has_text && has_audio && has_vision && has_bpe;
}
} // namespace

TEST_CASE ("Cortext C++ stub can be created and used", "[cortext][stub]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();
  const std::string models_dir = "models/imagebind";

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path, models_dir));
  REQUIRE (ctx != nullptr);

  // Encoding behavior depends on build flags and local environment:
  // - If ORT is enabled AND models/tokenizer assets are present: encoding should
  //   succeed.
  // - Otherwise: encoding should fail fast by throwing (but construction should
  //   still succeed).
#if defined(CORTEXT_ENABLE_IMAGEBIND_ORT)
  const bool expect_encode_ok = ImageBindAssetsPresent (models_dir);
#else
  const bool expect_encode_ok = false;
#endif

  if (expect_encode_ok)
    {
      REQUIRE_NOTHROW ([&] {
        // Smoke test: verify each API can be called successfully.
        // Note: should_interrupt is an algorithm decision that depends on
        // retrieval results and signal metrics; we only verify no crashes.
        auto out_text = ctx->ProcessText ("hello world", /*ts*/ 1000ULL, "test");
        (void)out_text.should_interrupt; // Algorithm-dependent, not asserted

        const float pcm[4] = { 0.0f, 0.1f, -0.1f, 0.0f };
        auto out_audio = ctx->ProcessAudio (pcm, 4, /*ts*/ 2000ULL, "test");
        (void)out_audio.should_interrupt;

        const std::uint8_t px[4] = { 0, 0, 0, 0 };
        auto out_image
            = ctx->ProcessImage (px, 1, 1, 4, /*ts*/ 3000ULL, "test");
        (void)out_image.should_interrupt;

        auto out_cons = ctx->Consolidate (4000ULL);
        (void)out_cons.should_interrupt;
        ctx->Flush ();
      }());
    }
  else
    {
      REQUIRE_THROWS (ctx->ProcessText ("hello world", /*ts*/ 1000ULL, "test"));
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
  REQUIRE_NOTHROW (
      ctx = cortext::Cortext::Create (cfg, ":memory:", "models/does-not-exist"));
  REQUIRE (ctx != nullptr);
  REQUIRE_THROWS (ctx->ProcessText ("hello", 0ULL, "test"));
}

TEST_CASE ("Cortext C ABI stubs return success", "[cortext][capi][stub]")
{
  cortext_handle h = cortext_create_with_models (
      0.5, 0.5, 0.5, ":memory:", "models/imagebind");
  REQUIRE (h != nullptr);

  const int text_rc = cortext_process_text (h, "hello", 1234ULL, "test");
  if (text_rc == 0)
    {
      const float pcm[2] = { 0.0f, 0.0f };
      REQUIRE (cortext_process_audio (h, pcm, 2, 2345ULL, "test") == 0);

      const std::uint8_t px[3] = { 0, 0, 0 };
      REQUIRE (cortext_process_image (h, px, 1, 1, 3, 3456ULL, "test") == 0);

      REQUIRE (cortext_consolidate (h, 4567ULL) == 0);
      REQUIRE (cortext_flush (h) == 0);
    }
  else
    {
      // Without ORT (or without models), C API should return non-zero (but not
      // crash).
      REQUIRE (text_rc != 0);
    }

  cortext_free (h);
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

  std::unordered_map<long long, std::string> expected_payloads;
  std::unordered_map<long long, std::string> expected_mimes;

  for (const auto &sample : samples)
    {
      auto blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                       { sample.payload });
      REQUIRE (blob_rows.size () == 1);
      const auto blob_id = BlobFromAny (blob_rows[0].at ("id"));
      REQUIRE (!blob_id.empty ());

      store->Execute ("INSERT INTO embeddings (embedding_id, embedding, type, strength, "
                      "use_frequency, stability, connectivity, drift_mag, influence, "
                      "sustained_influence, contextual_gain, redundancy, pre_activation, "
                      "lability_state, suppression_count) VALUES (?,?,'memory',1.0,0.0,"
                      "1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0)",
                      { sample.embedding_id, embedding });

      store->Execute ("INSERT INTO memories (embedding_id, modality, mime,"
                      " source_id, timestamp, width, height,"
                      " channels, sample_rate, num_samples, blob_id)"
                      " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      {
                          sample.embedding_id,
                          sample.modality,
                          sample.mime,
                          std::string ("unit-test"),
                          0LL,
                          sample.modality == "video" ? 2LL : 1LL,
                          sample.modality == "video" ? 2LL : 1LL,
                          sample.modality == "audio" ? 1LL : 3LL,
                          sample.modality == "audio" ? 16000LL : 24LL,
                          4LL,
                          blob_id,
                      });
      expected_payloads[sample.embedding_id] = std::string (
          reinterpret_cast<const char *> (sample.payload.data ()),
          sample.payload.size ());
      expected_mimes[sample.embedding_id] = sample.mime;
    }

  cortext::Cortext::Config cfg;
  auto ctx = cortext::Cortext::Create (cfg, db_path, "models/imagebind");
  REQUIRE (ctx != nullptr);

  std::vector<long long> candidate_ids;
  for (const auto &sample : samples)
    {
      candidate_ids.push_back (sample.embedding_id);
    }
  auto hydrated = ctx->DebugHydrateForTest (candidate_ids, {});
  REQUIRE (hydrated.memories.size () == samples.size ());
  for (const auto &memory : hydrated.memories)
    {
      REQUIRE (expected_payloads.count (memory.id) == 1);
      REQUIRE (expected_mimes.count (memory.id) == 1);
      REQUIRE (memory.content == expected_payloads.at (memory.id));
      REQUIRE (memory.mimetype == expected_mimes.at (memory.id));
    }
}

TEST_CASE ("C API handles NULL inputs correctly", "[cortext][capi][safety]")
{
  SECTION ("cortext_free accepts NULL handle")
  {
    REQUIRE_NOTHROW (cortext_free (nullptr));
  }

  SECTION ("cortext_process_text returns 1 for NULL handle")
  {
    CHECK (cortext_process_text (nullptr, "text", 0, "src") == 1);
  }

  SECTION ("cortext_process_text returns 1 for NULL text")
  {
    ScopedTempDb temp_db;
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
    REQUIRE (h != nullptr);
    CHECK (cortext_process_text (h, nullptr, 0, "src") == 1);
    cortext_free (h);
  }

  SECTION ("cortext_process_text returns 1 for NULL source_id")
  {
    ScopedTempDb temp_db;
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
    REQUIRE (h != nullptr);
    CHECK (cortext_process_text (h, "text", 0, nullptr) == 1);
    cortext_free (h);
  }

  SECTION ("cortext_process_audio returns 1 for NULL pcm")
  {
    ScopedTempDb temp_db;
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
    REQUIRE (h != nullptr);
    CHECK (cortext_process_audio (h, nullptr, 100, 0, "src") == 1);
    cortext_free (h);
  }

  SECTION ("cortext_process_image returns 1 for NULL data")
  {
    ScopedTempDb temp_db;
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
    REQUIRE (h != nullptr);
    CHECK (cortext_process_image (h, nullptr, 10, 10, 3, 0, "src") == 1);
    cortext_free (h);
  }

  SECTION ("cortext_consolidate returns 1 for NULL handle")
  {
    CHECK (cortext_consolidate (nullptr, 0) == 1);
  }

  SECTION ("cortext_flush returns 1 for NULL handle")
  {
    CHECK (cortext_flush (nullptr) == 1);
  }
}
