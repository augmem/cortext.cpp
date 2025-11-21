// tests/store_extensions.test.cpp
#include <any>
#include <catch2/catch_test_macros.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <vector>

namespace
{

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

} // namespace

TEST_CASE ("Extensions: sqlite-graph and sqlite-vec are available",
           "[extensions]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");

  // sqlite-graph: create vtable and basic functions
  SECTION ("sqlite-graph virtual table and functions")
  {
    // Create virtual table
    store->Execute ("CREATE VIRTUAL TABLE graph USING graph()", {});

    // Basic function exists
    auto res = store->Execute ("SELECT graph_count_nodes() AS c", {});
    REQUIRE (res.size () == 1);
    REQUIRE (res[0].count ("c") == 1);
    // Count is 0 on empty graph
    REQUIRE (std::any_cast<long long> (res[0].at ("c")) == 0LL);

    // Minimal cypher: create and match
    store->Execute ("SELECT cypher_execute('CREATE (p:Person {name: \"A\"})')",
                    {});
    auto match = store->Execute (
        "SELECT cypher_execute('MATCH (n:Person) RETURN n') AS r", {});
    REQUIRE (match.size () == 1);
    REQUIRE (match[0].count ("r") == 1);
    // JSON result should be non-empty
    REQUIRE (std::any_cast<std::string> (match[0].at ("r")).size () > 0);
  }

  // sqlite-vec: version function present
  SECTION ("sqlite-vec version function")
  {
    auto res = store->Execute ("SELECT vec_version() AS v", {});
    REQUIRE (res.size () == 1);
    REQUIRE (res[0].count ("v") == 1);
    REQUIRE (std::any_cast<std::string> (res[0].at ("v")).size () > 0);
  }

  // sqlite-objstore: blob round-trip via scalar helpers
  SECTION ("sqlite-objstore blob round-trip")
  {
    store->Execute ("CREATE VIRTUAL TABLE objstore USING objstore()", {});
    const std::vector<unsigned char> payload = { 0x01, 0x02, 0x03, 0x04 };
    auto put_rows
        = store->Execute ("SELECT objstore_put(?1) AS id", { payload });
    REQUIRE (put_rows.size () == 1);
    auto id_it = put_rows[0].find ("id");
    REQUIRE (id_it != put_rows[0].end ());
    auto blob_id = BlobFromAny (id_it->second);
    REQUIRE (!blob_id.empty ());

    auto fetch
        = store->Execute ("SELECT objstore_get(?1) AS data", { blob_id });
    REQUIRE (fetch.size () == 1);
    auto data_it = fetch[0].find ("data");
    REQUIRE (data_it != fetch[0].end ());
    auto round_trip = BlobFromAny (data_it->second);
    REQUIRE (round_trip == payload);
  }

  SECTION ("sqlite-objstore stores WAV audio payloads")
  {
    store->Execute ("CREATE VIRTUAL TABLE objstore USING objstore()", {});
    auto wav_payload = SampleWavBytes ();
    auto put_rows
        = store->Execute ("SELECT objstore_put(?1) AS id", { wav_payload });
    REQUIRE (put_rows.size () == 1);
    const auto wav_id = BlobFromAny (put_rows[0].at ("id"));
    REQUIRE (wav_id.size () == 32);

    auto fetch
        = store->Execute ("SELECT objstore_get(?1) AS data", { wav_id });
    REQUIRE (fetch.size () == 1);
    const auto round_trip = BlobFromAny (fetch[0].at ("data"));
    REQUIRE (round_trip == wav_payload);
  }

  SECTION ("sqlite-objstore stores MP4 video payloads")
  {
    store->Execute ("CREATE VIRTUAL TABLE objstore USING objstore()", {});
    auto mp4_payload = SampleMp4Bytes ();
    auto put_rows
        = store->Execute ("SELECT objstore_put(?1) AS id", { mp4_payload });
    REQUIRE (put_rows.size () == 1);
    const auto mp4_id = BlobFromAny (put_rows[0].at ("id"));
    REQUIRE (mp4_id.size () == 32);

    auto fetch
        = store->Execute ("SELECT objstore_get(?1) AS data", { mp4_id });
    REQUIRE (fetch.size () == 1);
    const auto round_trip = BlobFromAny (fetch[0].at ("data"));
    REQUIRE (round_trip == mp4_payload);
  }
}
