#include "test_helpers.hpp"
#include "../src/capi_internal.hpp"
#include "../src/cortext_pipeline_internal.hpp"
#include "../src/operations/retrieval_trace_state.hpp"
#include "../src/operations/bounded_activation_shadow_internal.hpp"
#include "../src/operations/sparse_retrieval_knobs_internal.hpp"
#include <algorithm>
#include <array>
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cortext/clock.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/capi.h>
#include <cortext/cortext.hpp>
#include <cortext/internal/replay_ingress.hpp>
#include <cortext/signal.hpp>
#include <cortext/store/sqlite_store.hpp>

namespace
{
struct ScopedRankedCandidates
{
  void
  Set (std::vector<cortext::operations::retrieval_trace::RankedCandidate>
           candidates)
  {
    cortext::operations::retrieval_trace::SetLastRankedCandidates (candidates);
  }

  ~ScopedRankedCandidates ()
  {
    cortext::operations::retrieval_trace::ClearLastRankedCandidates ();
  }
};

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

struct DurableRetrievalState
{
  long long memories = 0;
  long long retrieved = 0;
  long long used = 0;
  long long accessed = 0;
  long long strength_nanos = 0;
  long long associations = 0;
  long long association_weight_nanos = 0;
  long long reconstructions = 0;
  long long state_signals_processed = 0;
  long long episodes = 0;

  bool
  operator== (const DurableRetrievalState &) const = default;
};

DurableRetrievalState
SnapshotDurableRetrievalState (cortext::Store &store)
{
  DurableRetrievalState out;
  const auto memories = store.Execute (
      "SELECT COUNT(*) AS memories, "
      "COALESCE(SUM(retrieved_count), 0) AS retrieved, "
      "COALESCE(SUM(used_count), 0) AS used, "
      "COALESCE(SUM(CASE WHEN last_access IS NOT NULL THEN 1 ELSE 0 END), 0) "
      "  AS accessed, "
      "COALESCE(CAST(ROUND(SUM(strength) * 1000000000.0) AS INTEGER), 0) "
      "  AS strength_nanos FROM memories");
  REQUIRE (memories.size () == 1);
  out.memories = AnyToLongLong (memories[0].at ("memories"));
  out.retrieved = AnyToLongLong (memories[0].at ("retrieved"));
  out.used = AnyToLongLong (memories[0].at ("used"));
  out.accessed = AnyToLongLong (memories[0].at ("accessed"));
  out.strength_nanos = AnyToLongLong (memories[0].at ("strength_nanos"));

  const auto associations = store.Execute (
      "SELECT COUNT(*) AS associations, "
      "COALESCE(CAST(ROUND(SUM(weight) * 1000000000.0) AS INTEGER), 0) "
      "  AS weight_nanos FROM associations");
  REQUIRE (associations.size () == 1);
  out.associations = AnyToLongLong (associations[0].at ("associations"));
  out.association_weight_nanos
      = AnyToLongLong (associations[0].at ("weight_nanos"));
  const auto reconstructions = store.Execute (
      "SELECT COUNT(*) AS reconstructions FROM memory_reconstructions");
  REQUIRE (reconstructions.size () == 1);
  out.reconstructions
      = AnyToLongLong (reconstructions[0].at ("reconstructions"));
  const auto state = store.Execute (
      "SELECT COALESCE(MAX(signals_processed), 0) AS signals_processed "
      "FROM state");
  REQUIRE (state.size () == 1);
  out.state_signals_processed
      = AnyToLongLong (state[0].at ("signals_processed"));
  const auto episodes = store.Execute ("SELECT COUNT(*) AS episodes FROM episodes");
  REQUIRE (episodes.size () == 1);
  out.episodes = AnyToLongLong (episodes[0].at ("episodes"));
  return out;
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

TEST_CASE ("Cortext context defaults to one consolidation state",
           "[cortext][consolidation][api]")
{
  cortext::Cortext::Context context;
  REQUIRE (context.consolidation_state == cortext::ConsolidationState::None);
}

TEST_CASE ("Cortext C++ stub can be created and used",
           "[cortext][stub][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path));
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
           "[cortext][embed][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path));
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

TEST_CASE ("Cortext ephemeral text query retrieves without storing input",
           "[cortext][retention][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();

  auto unique_store = cortext::SQLiteStore::Create (db_path.c_str ());
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  const std::string query = "garage door code";
  std::vector<float> query_embedding;
  {
    std::unique_ptr<cortext::Cortext> embedder;
    REQUIRE_NOTHROW (
        embedder = cortext::Cortext::Create (cfg, store));
    REQUIRE (embedder != nullptr);
    REQUIRE_NOTHROW (query_embedding = embedder->EmbedText (query));
  }
  REQUIRE (query_embedding.size () >= 256);
  std::vector<float> retrieval_embedding (query_embedding.begin (),
                                          query_embedding.begin () + 256);
  double norm_sq = 0.0;
  for (const float value : retrieval_embedding)
    {
      norm_sq += static_cast<double> (value) * value;
    }
  const double norm = std::sqrt (norm_sq);
  REQUIRE (norm > 0.0);
  for (float &value : retrieval_embedding)
    {
      value = static_cast<float> (static_cast<double> (value) / norm);
    }

  constexpr long long kMemoryId = 9001LL;
  cortext::testing::SeedEmbeddingV2 (*store, kMemoryId, retrieval_embedding,
                                     1000LL);
  cortext::testing::SeedMemoryV2 (*store, kMemoryId, kMemoryId,
                                  "seed/manual", "LONG_TERM", 1.0, 1000LL);
  const std::string source_text = "The garage door code is 8841.";
  const std::vector<unsigned char> payload (source_text.begin (),
                                           source_text.end ());
  auto blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                   { payload });
  REQUIRE (blob_rows.size () == 1);
  const auto blob_id = BlobFromAny (blob_rows[0].at ("id"));
  REQUIRE_FALSE (blob_id.empty ());
  store->Execute (
      "UPDATE memories SET blob_id = ?, modality = 'text' WHERE memory_id = ?",
      { blob_id, kMemoryId });
  store->Execute (
      "INSERT INTO signals (memory_id, embedding_id, source_id, timestamp, "
      "modality, mime, blob_id, serial_position, created_at) "
      "VALUES (?, ?, 'seed/manual', 1000, 'text', 'text/plain', ?, 0, 1000)",
      { kMemoryId, kMemoryId, blob_id });

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, store));
  REQUIRE (ctx != nullptr);

  const auto before_memories
      = store->Execute ("SELECT COUNT(*) AS n FROM memories");
  const auto before_signals
      = store->Execute ("SELECT COUNT(*) AS n FROM signals");
  const auto before_durable_state = SnapshotDurableRetrievalState (*store);
  REQUIRE (before_memories.size () == 1);
  REQUIRE (before_signals.size () == 1);

  auto recalled = ctx->ProcessTextAt (
      query, "query/public", 2000ULL, cortext::Retention::Ephemeral);

  REQUIRE (recalled.at_boundary);
  REQUIRE (recalled.boundary_score.has_value ());
  REQUIRE (*recalled.boundary_score == 1.0);
  REQUIRE_FALSE (recalled.retrieved_memory.empty ());
  bool found_seeded_memory = false;
  for (const auto &memory : recalled.retrieved_memory)
    {
      found_seeded_memory = found_seeded_memory || memory.id == kMemoryId;
    }
  REQUIRE (found_seeded_memory);
  REQUIRE_FALSE (recalled.output.stored_memory_id.has_value ());
  REQUIRE_FALSE (recalled.output.stored_signal_id.has_value ());

  const auto after_memories
      = store->Execute ("SELECT COUNT(*) AS n FROM memories");
  const auto after_signals
      = store->Execute ("SELECT COUNT(*) AS n FROM signals");
  REQUIRE (AnyToLongLong (after_memories[0].at ("n"))
           == AnyToLongLong (before_memories[0].at ("n")));
  REQUIRE (AnyToLongLong (after_signals[0].at ("n"))
           == AnyToLongLong (before_signals[0].at ("n")));
  REQUIRE (SnapshotDurableRetrievalState (*store) == before_durable_state);
}

TEST_CASE ("Cortext ephemeral public query retrieves durable API memory",
           "[cortext][retention][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();

  auto unique_store = cortext::SQLiteStore::Create (db_path.c_str ());
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, store));
  REQUIRE (ctx != nullptr);

  auto ingest = ctx->ProcessTextAt ("The garage door code is 8841.",
                                    "notes/main", 1000ULL,
                                    cortext::Retention::Durable);
  REQUIRE (ingest.output.stored_memory_id.has_value ());
  REQUIRE (ingest.output.stored_signal_id.has_value ());

  const auto before_memories
      = store->Execute ("SELECT COUNT(*) AS n FROM memories");
  const auto before_signals
      = store->Execute ("SELECT COUNT(*) AS n FROM signals");

  auto recalled = ctx->ProcessTextAt (
      "garage door code", "query/public", 2000ULL,
      cortext::Retention::Ephemeral);

  REQUIRE (recalled.at_boundary);
  REQUIRE (recalled.boundary_score.has_value ());
  REQUIRE (*recalled.boundary_score == 1.0);
  REQUIRE_FALSE (recalled.retrieved_memory.empty ());
  bool found_code = false;
  for (const auto &memory : recalled.retrieved_memory)
    {
      found_code = found_code || TextFromMemory (memory).find ("8841")
                                      != std::string::npos;
    }
  REQUIRE (found_code);
  REQUIRE_FALSE (recalled.output.stored_memory_id.has_value ());
  REQUIRE_FALSE (recalled.output.stored_signal_id.has_value ());

  const auto after_memories
      = store->Execute ("SELECT COUNT(*) AS n FROM memories");
  const auto after_signals
      = store->Execute ("SELECT COUNT(*) AS n FROM signals");
  REQUIRE (AnyToLongLong (after_memories[0].at ("n"))
           == AnyToLongLong (before_memories[0].at ("n")));
  REQUIRE (AnyToLongLong (after_signals[0].at ("n"))
           == AnyToLongLong (before_signals[0].at ("n")));
}

TEST_CASE ("Retention Natural omits force flags; Durable forces boundary+write",
           "[cortext][retention][aist]")
{
  REQUIRE (static_cast<int> (cortext::Retention::Natural) == 0);
  REQUIRE (static_cast<int> (cortext::Retention::Durable) == 1);
  REQUIRE (static_cast<int> (cortext::Retention::Boundary) == 2);
  REQUIRE (static_cast<int> (cortext::Retention::Ephemeral) == 3);

  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  auto unique_store = cortext::SQLiteStore::Create (temp_db.path ().c_str ());
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, store));
  REQUIRE (ctx != nullptr);

  // Omitted retention defaults to Natural: does not force explicit_turn.
  // boundary_score can legitimately hit 1.0 under natural algorithms, so
  // assert on boundary_type rather than score as the force diagnostic.
  const auto before_objects
      = store->Execute ("SELECT COUNT(*) AS n FROM objstore_data");
  auto natural = ctx->ProcessTextAt (
      "Streaming token about weather.", "stream/main", 1000ULL);
  if (natural.boundary_type.has_value ())
    {
      REQUIRE (*natural.boundary_type != "explicit_turn");
    }
  const auto after_objects
      = store->Execute ("SELECT COUNT(*) AS n FROM objstore_data");
  REQUIRE (AnyToLongLong (after_objects[0].at ("n"))
           == AnyToLongLong (before_objects[0].at ("n")));

  auto durable = ctx->ProcessTextAt (
      "The garage door code is 8841.", "chat/main", 2000ULL,
      cortext::Retention::Durable);
  REQUIRE (durable.at_boundary);
  REQUIRE (durable.boundary_score.has_value ());
  REQUIRE (*durable.boundary_score == 1.0);
  REQUIRE (durable.boundary_type.has_value ());
  REQUIRE (*durable.boundary_type == "explicit_turn");
  REQUIRE (durable.output.stored_memory_id.has_value ());

  auto boundary = ctx->ProcessTextAt (
      "Close the unit without force write.", "chat/main", 3000ULL,
      cortext::Retention::Boundary);
  REQUIRE (boundary.at_boundary);
  REQUIRE (boundary.boundary_score.has_value ());
  REQUIRE (*boundary.boundary_score == 1.0);
  REQUIRE (boundary.boundary_type.has_value ());
  REQUIRE (*boundary.boundary_type == "explicit_turn");

  auto ephemeral = ctx->ProcessTextAt (
      "garage door code", "query/main", 4000ULL,
      cortext::Retention::Ephemeral);
  REQUIRE (ephemeral.at_boundary);
  REQUIRE (ephemeral.boundary_type.has_value ());
  REQUIRE (*ephemeral.boundary_type == "explicit_turn");
  REQUIRE_FALSE (ephemeral.output.stored_memory_id.has_value ());
}

TEST_CASE ("Durable flushes one shared Natural accumulator unit",
           "[cortext][retention][accumulator][aist]")
{
  cortext::testing::ScopedEnvVar disable_natural_boundary (
      "CORTEXT_BOUNDARY_DISABLE_NATURAL", "1");
  cortext::testing::ScopedEnvVar disable_pressure_boundary (
      "CORTEXT_BOUNDARY_DISABLE_PRESSURE", "1");
  cortext::testing::ScopedEnvVar disable_surprisal_boundary (
      "CORTEXT_BOUNDARY_DISABLE_SURPRISAL", "1");
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  auto unique_store = cortext::SQLiteStore::Create (temp_db.path ().c_str ());
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  auto ctx = cortext::Cortext::Create (cfg, store);
  REQUIRE (ctx != nullptr);

  const auto first = ctx->ProcessTextAt (
      "A calm sentence in one pending thought.", "stream/shared", 1000ULL,
      cortext::Retention::Natural);
  const auto second = ctx->ProcessTextAt (
      "A second calm sentence in the same thought.", "stream/shared",
      2000ULL, cortext::Retention::Natural);
  REQUIRE_FALSE (first.output.stored_memory_id.has_value ());
  REQUIRE_FALSE (second.output.stored_memory_id.has_value ());
  REQUIRE (store->Execute (
               "SELECT memory_id FROM memories WHERE source_id = ?",
               { std::string ("stream/shared") })
               .empty ());

  const auto durable = ctx->ProcessTextAt (
      "The explicit turn closes that pending thought.", "stream/shared",
      3000ULL, cortext::Retention::Durable);
  REQUIRE (durable.output.stored_memory_id.has_value ());
  const auto memory_rows = store->Execute (
      "SELECT memory_id, n_signals FROM memories WHERE source_id = ? "
      "AND kind = 'LONG_TERM'",
      { std::string ("stream/shared") });
  REQUIRE (memory_rows.size () == 1);
  REQUIRE (AnyToLongLong (memory_rows[0].at ("n_signals")) == 3);
  const auto signal_rows = store->Execute (
      "SELECT timestamp, serial_position FROM signals WHERE memory_id = ? "
      "ORDER BY serial_position",
      { AnyToLongLong (memory_rows[0].at ("memory_id")) });
  REQUIRE (signal_rows.size () == 3);
  REQUIRE (AnyToLongLong (signal_rows[0].at ("timestamp")) == 1000);
  REQUIRE (AnyToLongLong (signal_rows[1].at ("timestamp")) == 2000);
  REQUIRE (AnyToLongLong (signal_rows[2].at ("timestamp")) == 3000);
}

TEST_CASE ("Flush closes every open Natural unit",
           "[cortext][retention][accumulator][flush][aist]")
{
  cortext::testing::ScopedEnvVar disable_natural_boundary (
      "CORTEXT_BOUNDARY_DISABLE_NATURAL", "1");
  cortext::testing::ScopedEnvVar disable_pressure_boundary (
      "CORTEXT_BOUNDARY_DISABLE_PRESSURE", "1");
  cortext::testing::ScopedEnvVar disable_surprisal_boundary (
      "CORTEXT_BOUNDARY_DISABLE_SURPRISAL", "1");
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  auto unique_store = cortext::SQLiteStore::Create (temp_db.path ().c_str ());
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  auto ctx = cortext::Cortext::Create (cfg, store);
  REQUIRE (ctx != nullptr);

  // Two conversations end with their units still open; Flush is the host
  // saying "commit what is pending", so neither unit may outlive it.
  (void)ctx->ProcessTextAt ("A calm sentence in one pending thought.",
                            "stream/a", 1000ULL, cortext::Retention::Natural);
  (void)ctx->ProcessTextAt ("A second calm sentence in the same thought.",
                            "stream/a", 2000ULL, cortext::Retention::Natural);
  (void)ctx->ProcessTextAt ("Another stream starts a different thought.",
                            "stream/b", 2500ULL, cortext::Retention::Natural);
  ctx->Flush ();

  // Every closed unit is stored, whatever its window score.
  const auto flushed = store->Execute (
      "SELECT source_id, n_signals FROM memories WHERE kind = 'LONG_TERM' "
      "ORDER BY source_id");
  REQUIRE (flushed.size () == 2);
  REQUIRE (std::any_cast<std::string> (flushed[0].at ("source_id"))
           == "stream/a");
  REQUIRE (AnyToLongLong (flushed[0].at ("n_signals")) == 2);
  REQUIRE (std::any_cast<std::string> (flushed[1].at ("source_id"))
           == "stream/b");
  REQUIRE (AnyToLongLong (flushed[1].at ("n_signals")) == 1);

  // A later explicit turn starts a fresh unit on each source: it carries
  // only itself, not the signals that were open before Flush.
  for (const auto *source : { "stream/a", "stream/b" })
    {
      const auto durable = ctx->ProcessTextAt (
          "An explicit turn after the flush.", source, 3000ULL,
          cortext::Retention::Durable);
      REQUIRE (durable.output.stored_memory_id.has_value ());
      const auto rows = store->Execute (
          "SELECT n_signals FROM memories WHERE memory_id = ?",
          { static_cast<long long> (*durable.output.stored_memory_id) });
      REQUIRE (rows.size () == 1);
      INFO ("source=" << source);
      REQUIRE (AnyToLongLong (rows[0].at ("n_signals")) == 1);
    }
}

TEST_CASE ("A failed Flush keeps every open Natural unit for the next Flush",
           "[cortext][retention][accumulator][flush][rollback][aist]")
{
  cortext::testing::ScopedEnvVar disable_natural_boundary (
      "CORTEXT_BOUNDARY_DISABLE_NATURAL", "1");
  cortext::testing::ScopedEnvVar disable_pressure_boundary (
      "CORTEXT_BOUNDARY_DISABLE_PRESSURE", "1");
  cortext::testing::ScopedEnvVar disable_surprisal_boundary (
      "CORTEXT_BOUNDARY_DISABLE_SURPRISAL", "1");
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  auto unique_store = cortext::SQLiteStore::Create (temp_db.path ().c_str ());
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  auto ctx = cortext::Cortext::Create (cfg, store);
  REQUIRE (ctx != nullptr);

  (void)ctx->ProcessTextAt ("A calm sentence in one pending thought.",
                            "stream/a", 1000ULL, cortext::Retention::Natural);
  (void)ctx->ProcessTextAt ("A second calm sentence in the same thought.",
                            "stream/a", 2000ULL, cortext::Retention::Natural);
  (void)ctx->ProcessTextAt ("Another stream starts a different thought.",
                            "stream/b", 2500ULL, cortext::Retention::Natural);

  // The flush transaction fails after its open units were closed (state is
  // persisted after the closes); the store rolls back, so the units must
  // still be open afterwards.
  store->Execute ("CREATE TEMP TRIGGER fail_state_persist "
                  "BEFORE INSERT ON state "
                  "BEGIN SELECT RAISE(ABORT, 'injected flush failure'); END");
  REQUIRE_THROWS (ctx->Flush ());
  store->Execute ("DROP TRIGGER fail_state_persist");
  REQUIRE (store->Execute ("SELECT 1 FROM memories WHERE kind = 'LONG_TERM'")
               .empty ());

  ctx->Flush ();
  const auto flushed = store->Execute (
      "SELECT source_id, n_signals FROM memories WHERE kind = 'LONG_TERM' "
      "ORDER BY source_id");
  REQUIRE (flushed.size () == 2);
  REQUIRE (std::any_cast<std::string> (flushed[0].at ("source_id"))
           == "stream/a");
  REQUIRE (AnyToLongLong (flushed[0].at ("n_signals")) == 2);
  REQUIRE (std::any_cast<std::string> (flushed[1].at ("source_id"))
           == "stream/b");
  REQUIRE (AnyToLongLong (flushed[1].at ("n_signals")) == 1);
}

TEST_CASE ("Cortext recalls durable corrections at default knobs",
           "[cortext][belief_revision][eval][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();

  auto unique_store = cortext::SQLiteStore::Create (db_path.c_str ());
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, store));
  REQUIRE (ctx != nullptr);

  const std::string stale_vet
      = "Bailey's vet appointment is July 12 at 8am.";
  const std::string corrected_vet
      = "Bailey's July 12 vet appointment is canceled. "
        "The appointment moved to July 14 at 9am.";
  const std::string stale_pill
      = "Bailey takes the morning pill at 8am.";
  const std::string corrected_pill
      = "The doctor changed Bailey's morning pill to 9am.";

  auto ingest_vet = ctx->ProcessTextAt (
      stale_vet, "cli/main", 1000ULL, cortext::Retention::Durable);
  auto correct_vet = ctx->ProcessTextAt (
      corrected_vet, "cli/main", 2000ULL, cortext::Retention::Durable);
  auto ingest_pill = ctx->ProcessTextAt (
      stale_pill, "cli/main", 3000ULL, cortext::Retention::Durable);
  auto correct_pill = ctx->ProcessTextAt (
      corrected_pill, "cli/main", 4000ULL, cortext::Retention::Durable);

  REQUIRE (ingest_vet.output.stored_memory_id.has_value ());
  REQUIRE (correct_vet.output.stored_memory_id.has_value ());
  REQUIRE (ingest_pill.output.stored_memory_id.has_value ());
  REQUIRE (correct_pill.output.stored_memory_id.has_value ());

  auto edge_rows = store->Execute (
      "SELECT source_memory_id, target_memory_id FROM associations "
      "WHERE edge_type = 'supersedes'");
  REQUIRE (edge_rows.size () >= 2);

  // Superseded facts are no longer excluded from seeding (engine ablation
  // 2026-10: exclusion lowered gold-turn recall), so the stale fact may rank
  // ahead of its correction. The contract is that a recall never shows the
  // stale fact without the correction that replaced it.
  auto assert_correction_recalled =
      [] (const cortext::Cortext::Context &recalled,
          const std::string &corrected) {
        bool found = false;
        for (const auto &memory : recalled.retrieved_memory)
          {
            if (TextFromMemory (memory).find (corrected) != std::string::npos)
              {
                found = true;
              }
          }
        REQUIRE (found);
      };

  auto vet_recall = ctx->ProcessTextAt (
      "When is Bailey's vet appointment?", "cli/recall", 5000ULL,
      cortext::Retention::Ephemeral);
  assert_correction_recalled (vet_recall, corrected_vet);

  auto pill_recall = ctx->ProcessTextAt (
      "When is Bailey's morning pill?", "cli/recall", 6000ULL,
      cortext::Retention::Ephemeral);
  assert_correction_recalled (pill_recall, corrected_pill);
}

TEST_CASE ("A retrieved fact brings back its newest restatement after it",
           "[cortext][retrieval][supersession]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  auto store = std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (temp_db.path ().c_str ()));
  auto ctx = cortext::Cortext::Create (cfg, store);
  const std::uint64_t hour = 60ULL * 60ULL * 1000ULL;
  const auto first = ctx->ProcessTextAt (
      "I logged 14 tasks in Jira for the sprint, aiming for 85% completion.",
      "chat/a/1", 1000ULL, cortext::Retention::Durable);
  const auto second = ctx->ProcessTextAt (
      "I now have 16 tasks in Jira for the sprint.", "chat/b/1",
      1000ULL + 5 * hour, cortext::Retention::Durable);
  const auto latest = ctx->ProcessTextAt (
      "Jira now shows 17 tasks for the sprint and the target is 88%.",
      "chat/c/1", 1000ULL + 10 * hour, cortext::Retention::Durable);
  REQUIRE (first.output.stored_memory_id.has_value ());
  REQUIRE (second.output.stored_memory_id.has_value ());
  REQUIRE (latest.output.stored_memory_id.has_value ());
  const long long first_id = *first.output.stored_memory_id;
  // Pin the edges so the test does not depend on similarity thresholds.
  store->Execute ("DELETE FROM associations WHERE edge_type = 'supersedes'");
  for (const long long source :
       { *second.output.stored_memory_id, *latest.output.stored_memory_id })
    store->Execute ("INSERT INTO associations (source_memory_id, "
                    "target_memory_id, edge_type, weight) "
                    "VALUES (?, ?, 'supersedes', 1.0)",
                    { source, first_id });

  const auto packet = ctx->DebugHydrateForTest ({ first_id }, {});
  REQUIRE (packet.retrieved_memory.size () == 1);
  const auto &memory = packet.retrieved_memory.front ();
  std::vector<std::string> blobs;
  for (const auto &blob : memory.content)
    blobs.emplace_back (blob.begin (), blob.end ());
  const auto at = [&blobs] (const std::string &needle) {
    for (std::size_t i = 0; i < blobs.size (); ++i)
      if (blobs[i].find (needle) != std::string::npos)
        return static_cast<long long> (i);
    return -1LL;
  };
  REQUIRE (at ("aiming for 85%") >= 0);
  REQUIRE (at ("17 tasks") > at ("aiming for 85%"));
  REQUIRE (at ("16 tasks") == -1); // newest restatement only
  REQUIRE (packet.output.operation_ms.at (
               "Cortext.superseding_successors_attached")
           == 1.0);
}

TEST_CASE ("internal replay ingress preserves media event timestamps",
           "[cortext][replay][media][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  cfg.signal_filter_audio_enabled = false;
  const std::string &db_path = temp_db.path ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path));
  REQUIRE (ctx != nullptr);

  const std::uint64_t replay_ts = 1573184762000ULL;
  std::vector<float> pcm (16000, 0.01f);

  REQUIRE_NOTHROW (
      cortext::internal::ReplayIngress::ProcessAudioAt (
          *ctx, pcm.data (), pcm.size (), "stream/reply", replay_ts,
          cortext::Retention::Durable));
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

TEST_CASE ("Cortext media overloads validate source media before encoding",
           "[cortext][media][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  cfg.signal_filter_audio_enabled = false;
  const std::string &db_path = temp_db.path ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path));
  REQUIRE (ctx != nullptr);

  std::vector<float> pcm (16000, 0.01f);
  const std::vector<unsigned char> wav = SampleWavBytes ();
  const cortext::Cortext::Media missing_mime{ wav.data (), wav.size (), "" };

  REQUIRE_THROWS_AS (
      ctx->ProcessAudio (pcm.data (), pcm.size (), "stream/source",
                         missing_mime),
      std::invalid_argument);

  const std::uint8_t pixel[3] = { 0, 0, 0 };
  const cortext::Cortext::Media missing_data{ nullptr, wav.size (),
                                              "image/png" };
  REQUIRE_THROWS_AS (
      ctx->ProcessImage (pixel, 1, 1, 3, "stream/source", missing_data),
      std::invalid_argument);
}

TEST_CASE ("Cortext media ingress validates processing inputs before use",
           "[cortext][media][safety][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  cfg.signal_filter_audio_enabled = false;
  const std::string &db_path = temp_db.path ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path));
  REQUIRE (ctx != nullptr);

  cortext::Cortext::Media empty_media{};
  const std::uint8_t pixel[3] = { 0, 0, 0 };
  std::vector<float> pcm (16000, 0.01f);

  SECTION ("ProcessAudio rejects NULL PCM")
  {
    REQUIRE_THROWS_AS (ctx->ProcessAudio (nullptr, 1, "stream/source"),
                       std::invalid_argument);
  }

  SECTION ("ProcessAudio media overload rejects NULL PCM")
  {
    REQUIRE_THROWS_AS (
        ctx->ProcessAudio (nullptr, 1, "stream/source", empty_media),
        std::invalid_argument);
  }

  SECTION ("ProcessImage rejects NULL data")
  {
    REQUIRE_THROWS_AS (
        ctx->ProcessImage (nullptr, 1, 1, 3, "stream/source"),
        std::invalid_argument);
  }

  SECTION ("ProcessImage rejects non-positive dimensions")
  {
    REQUIRE_THROWS_AS (
        ctx->ProcessImage (pixel, 0, 1, 3, "stream/source"),
        std::invalid_argument);
    REQUIRE_THROWS_AS (
        ctx->ProcessImage (pixel, 1, -1, 3, "stream/source"),
        std::invalid_argument);
    REQUIRE_THROWS_AS (
        ctx->ProcessImage (pixel, 1, 1, 0, "stream/source"),
        std::invalid_argument);
  }

  SECTION ("ProcessImage media overload rejects invalid dimensions")
  {
    REQUIRE_THROWS_AS (
        ctx->ProcessImage (pixel, 1, 1, -1, "stream/source", empty_media),
        std::invalid_argument);
  }

  SECTION ("replay audio ingress rejects NULL PCM")
  {
    REQUIRE_THROWS_AS (
        cortext::internal::ReplayIngress::ProcessAudioAt (
            *ctx, nullptr, 1, "stream/replay", 1573184762000ULL),
        std::invalid_argument);
  }

  SECTION ("replay image ingress rejects invalid image input")
  {
    REQUIRE_THROWS_AS (
        cortext::internal::ReplayIngress::ProcessImageAt (
            *ctx, pixel, 1, 0, 3, "stream/replay", 1573184762000ULL),
        std::invalid_argument);
  }

  SECTION ("valid inputs still pass validation")
  {
    REQUIRE_NOTHROW (ctx->ProcessAudio (pcm.data (), pcm.size (),
                                        "stream/source"));
    REQUIRE_NOTHROW (ctx->ProcessImage (pixel, 1, 1, 3, "stream/source"));
  }
}

TEST_CASE ("encoded modalities and source cardinalities share bounded work",
           "[cortext][performance][modality][source_id]"
           "[bounded_activation_shadow]")
{
  cortext::testing::ScopedEnvVar profile_work (
      "CORTEXT_PROFILE_WORK_COUNTERS", "1");
  cortext::testing::ScopedEnvVar profile_graph (
      "CORTEXT_PROFILE_GRAPH_RETRIEVAL", "1");
  cortext::testing::ScopedEnvVar profile_emotion (
      "CORTEXT_PROFILE_EMOTIONAL_CASCADE", "1");
  cortext::testing::ScopedEnvVar shadow_enabled (
      "CORTEXT_BOUNDED_ACTIVATION_SHADOW", "1");

  namespace shadow
      = cortext::operations::bounded_activation_shadow_internal;

  struct CounterSpec
  {
    const char *name;
    const char *primary;
    const char *secondary;
    double fixed_bound;
  };
  const std::array<CounterSpec, 9> kCounters { {
    { "rollback_full_cache_copy_count",
      "SignalProcessor.snapshot_full_cache_copy_count", nullptr, 0.0 },
    { "rollback_cache_entry_copy_count",
      "SignalProcessor.snapshot_cache_entry_copy_count", nullptr, 0.0 },
    { "graph_candidate_count", "GraphRetrieve.candidate_count", nullptr,
      4096.0 },
    { "graph_exact_comparison_count",
      "GraphRetrieve.family_exact_comparison_count", nullptr,
      static_cast<double> (
          cortext::operations::sparse_retrieval_knobs_internal::
              FamilyExactComparisonLimit (0.5, 0.5, 0.5)) },
    { "graph_cache_rebuild_count", "GraphRetrieve.cache_rebuild_count",
      nullptr, 1.0 },
    { "graph_rows_visited", "GraphRetrieve.rows_visited", nullptr, 4096.0 },
    { "supersession_current_candidate_count",
      "MemoryStorage.supersession_current_candidate_count", nullptr,
      4096.0 },
    { "supersession_historical_candidate_count",
      "MemoryStorage.supersession_historical_candidate_count", nullptr,
      4096.0 },
    { "supersession_rows_visited",
      "MemoryStorage.supersession_current_rows_visited",
      "MemoryStorage.supersession_historical_rows_visited", 4096.0 },
  } };
  constexpr std::array<const char *, 2> kSharedOperations {
    "cortext::operations::GraphAugmentedRetrieveCandidates",
    "cortext::operations::MemoryStorage",
  };

  struct RunResult
  {
    std::map<std::string, double> maxima;
    std::vector<shadow::Snapshot> shadow_sequence;
  };

  auto run_cardinality = [&] (int source_count,
                              cortext::Retention retention) {
    auto store = std::shared_ptr<cortext::Store> (
        cortext::SQLiteStore::Create (":memory:"));
    cortext::testing::InitializeCoreSchema (*store);
    cortext::SignalProcessor::Config cfg;
    cfg.focus = 0.45;
    cfg.sensitivity = 0.5;
    cfg.stability = 0.5;
    cortext::testing::RequireEncoder (cfg);
    const int supersession_candidate_limit
        = cortext::core::SupersessionCandidateLimit (
            cfg.focus, cfg.sensitivity, cfg.stability);
    for (int index = 0; index <= supersession_candidate_limit; ++index)
      {
        std::vector<float> embedding (256, 0.0f);
        embedding[static_cast<std::size_t> (index) % embedding.size ()]
            = 1.0f;
        const long long id = 100000LL + index;
        cortext::testing::SeedEmbeddingV2 (*store, id, embedding, 1000);
        cortext::testing::SeedMemoryV2 (
            *store, id, id, "mixed/seed", "LONG_TERM", 1.0, 1000);
      }
    cortext::SignalProcessor processor (
        cfg, store, cortext::BuildRootOperationSet (false));

    RunResult result;
    bool active_counter_seen = false;
    for (int event = 0; event < 12; ++event)
      {
        const auto source
            = "mixed/source/" + std::to_string (event % source_count);
        const auto timestamp
            = 1700000000000ULL + static_cast<std::uint64_t> (event) * 1000ULL;
        cortext::Signal signal;
        signal.embedding = Eigen::VectorXf::Zero (256);
        signal.embedding[event % 256] = 1.0f;
        signal.source_id = source;
        signal.timestamp = timestamp;
        signal.retention = retention;
        signal.payload = std::vector<unsigned char> (
            16, static_cast<unsigned char> (event + 1));
        switch (event % 3)
          {
          case 0:
            signal.modality = "text";
            signal.mimetype = "text/plain";
            break;
          case 1:
            signal.modality = "audio";
            signal.mimetype = "audio/pcm;format=f32";
            signal.sample_rate = 16000;
            signal.num_samples = 4;
            break;
          default:
            signal.modality = "image";
            signal.mimetype = "image/raw;width=1;height=1;channels=3";
            signal.width = 1;
            signal.height = 1;
            signal.channels = 3;
            break;
          }
        const auto output = processor.Process (signal);
        const auto snapshots = shadow::SnapshotsForTest ();
        REQUIRE (snapshots.size () == 1);
        const auto &snapshot = snapshots.front ();
        REQUIRE (snapshot.available);
        REQUIRE_FALSE (snapshot.disabled);
        REQUIRE_FALSE (snapshot.restart_rebuild_required);
        REQUIRE (snapshot.generation
                 == static_cast<std::uint64_t> (event + 1));
        REQUIRE (snapshot.event_index
                 == static_cast<std::uint64_t> (event + 1));
        REQUIRE (snapshot.root_count <= snapshot.parameters.roots);
        REQUIRE (snapshot.leaf_count
                 <= snapshot.parameters.leaf_capacity);
        REQUIRE (snapshot.metrics.normal_comparisons
                 <= snapshot.parameters.normal_comparison_bound);
        REQUIRE (snapshot.metrics.normal_comparisons_max
                 <= snapshot.parameters.normal_comparison_bound);
        REQUIRE (snapshot.metrics.recall_comparisons
                 <= snapshot.parameters.recall_comparison_bound);
        REQUIRE (snapshot.metrics.recall_comparisons_max
                 <= snapshot.parameters.recall_comparison_bound);
        REQUIRE (snapshot.metrics.consolidation_comparisons_max
                 <= snapshot.parameters.consolidation_comparison_bound);
        REQUIRE (snapshot.metrics.allocation_after_initialization_count == 0);
        result.shadow_sequence.push_back (snapshot);

        for (const auto *operation : kSharedOperations)
          REQUIRE (output.operation_ms.contains (operation));
        if (retention == cortext::Retention::Durable)
          REQUIRE (output.operation_ms.contains (
              "SignalProcessor.sqlite_wal_sync"));
        else
          REQUIRE_FALSE (output.operation_ms.contains (
              "SignalProcessor.sqlite_wal_sync"));
        for (std::size_t index = 0; index < kCounters.size (); ++index)
          {
            const auto &spec = kCounters[index];
            auto value = [&] (const char *key) {
              if (!key)
                return 0.0;
              const auto it = output.operation_ms.find (key);
              return it == output.operation_ms.end () ? 0.0 : it->second;
            };
            const double observed = value (spec.primary) + value (spec.secondary);
            INFO ("counter=" << spec.name << " source_count=" << source_count
                              << " event=" << event);
            REQUIRE (std::isfinite (observed));
            REQUIRE (observed >= 0.0);
            REQUIRE (observed <= spec.fixed_bound);
            result.maxima[spec.name]
                = std::max (result.maxima[spec.name], observed);
            active_counter_seen = active_counter_seen || observed > 0.0;
          }
      }

    REQUIRE (result.maxima["supersession_rows_visited"]
             > result.maxima["supersession_current_candidate_count"]
                   + result.maxima["supersession_historical_candidate_count"]);

    const auto before_maintenance = result.shadow_sequence.back ();

    cortext::Signal maintenance;
    maintenance.embedding = Eigen::VectorXf::Zero (256);
    maintenance.source_id = "mixed/maintenance";
    maintenance.timestamp = 1700000013000ULL;
    maintenance.retention = retention;
    maintenance.force_consolidation = true;
    (void)processor.Process (maintenance);
    const auto after_maintenance = shadow::SnapshotsForTest ();
    REQUIRE (after_maintenance.size () == 1);
    REQUIRE (after_maintenance.front ().generation
             == before_maintenance.generation);
    REQUIRE (after_maintenance.front ().event_index
             == before_maintenance.event_index);
    REQUIRE (after_maintenance.front ().state_digest
             == before_maintenance.state_digest);

    const auto source_rows = store->Execute (
        "SELECT COUNT(DISTINCT source_id) AS n FROM signals");
    REQUIRE (AnyToLongLong (source_rows[0].at ("n")) == source_count);
    const auto modality_rows = store->Execute (
        "SELECT DISTINCT modality FROM signals ORDER BY modality");
    std::unordered_set<std::string> modalities;
    for (const auto &row : modality_rows)
      modalities.insert (std::any_cast<std::string> (row.at ("modality")));
    REQUIRE (modalities
             == std::unordered_set<std::string> { "audio", "image", "text" });
    REQUIRE (active_counter_seen);
    return result;
  };

  std::optional<std::vector<shadow::Snapshot>> natural_two_source_sequence;
  for (const auto retention : { cortext::Retention::Natural,
                                cortext::Retention::Durable })
    {
      const auto two_sources = run_cardinality (2, retention);
      const auto four_sources = run_cardinality (4, retention);
      REQUIRE (two_sources.shadow_sequence.size ()
               == four_sources.shadow_sequence.size ());
      for (std::size_t event = 0; event < two_sources.shadow_sequence.size ();
           ++event)
        {
          const auto &two = two_sources.shadow_sequence[event];
          const auto &four = four_sources.shadow_sequence[event];
          REQUIRE (two.generation == four.generation);
          REQUIRE (two.event_index == four.event_index);
          REQUIRE (two.state_digest == four.state_digest);
          REQUIRE (two.metrics.normal_comparisons
                   == four.metrics.normal_comparisons);
          REQUIRE (two.metrics.recall_comparisons
                   == four.metrics.recall_comparisons);
          REQUIRE (two.metrics.consolidation_comparisons
                   == four.metrics.consolidation_comparisons);
        }
      if (retention == cortext::Retention::Natural)
        natural_two_source_sequence = two_sources.shadow_sequence;
      else
        {
          REQUIRE (natural_two_source_sequence.has_value ());
          REQUIRE (natural_two_source_sequence->size ()
                   == two_sources.shadow_sequence.size ());
          for (std::size_t event = 0;
               event < two_sources.shadow_sequence.size (); ++event)
            REQUIRE (natural_two_source_sequence->at (event).state_digest
                     == two_sources.shadow_sequence[event].state_digest);
        }
      for (std::size_t index = 0; index < kCounters.size (); ++index)
        {
          INFO ("counter=" << kCounters[index].name
                            << " retention="
                            << static_cast<int> (retention));
          REQUIRE (two_sources.maxima.at (kCounters[index].name)
                   <= kCounters[index].fixed_bound);
          REQUIRE (four_sources.maxima.at (kCounters[index].name)
                   <= kCounters[index].fixed_bound);
        }
    }
}

TEST_CASE ("timestamped replay persists working memory source timestamps",
           "[cortext][replay][working_memory][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path));
  REQUIRE (ctx != nullptr);

  const std::uint64_t replay_ts = 1573184762000ULL;
  REQUIRE_NOTHROW (
      ctx->ProcessTextAt ("timestamped working memory replay", "stream/main",
                          replay_ts, cortext::Retention::Durable));
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
           "[cortext][replay][hydration][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path));
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
           "[cortext][replay][working_memory][performance][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path));
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
           "[cortext][replay][clock][working_memory][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();
  const std::uint64_t replay_ts = 1573184762000ULL;

  {
    auto clock = std::make_shared<cortext::FixedClock> (replay_ts);
    auto ctx = cortext::Cortext::Create (cfg, db_path, clock);
    REQUIRE (ctx != nullptr);
    REQUIRE_NOTHROW (ctx->ProcessTextAt (
        "timestamped working memory replay survives reopen", "stream/main",
        replay_ts, cortext::Retention::Durable));
    ctx->Flush ();
  }

  {
    auto clock = std::make_shared<cortext::FixedClock> (replay_ts + 1000ULL);
    auto reopened = cortext::Cortext::Create (cfg, db_path, clock);
    REQUIRE (reopened != nullptr);
    auto probe = reopened->ProcessTextAt (
        "follow up for the replay working memory", "stream/main",
        replay_ts + 2000ULL, cortext::Retention::Ephemeral);
    REQUIRE_FALSE (probe.working_memory.empty ());
  }
}

TEST_CASE ("internal replay consolidation is maintenance that leaves memories unchanged",
           "[cortext][replay][consolidation][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  const std::string &db_path = temp_db.path ();

  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, db_path));
  REQUIRE (ctx != nullptr);

  const std::uint64_t source_ts = 1573184762000ULL;
  for (int i = 0; i < 24; ++i)
    {
      REQUIRE_NOTHROW (ctx->ProcessTextAt (
          "shared replay consolidation topic package pickup dinner logistics",
          "stream/main",
          source_ts + static_cast<std::uint64_t> (i) * 1000ULL,
          cortext::Retention::Durable));
    }
  ctx->Flush ();

  long long signal_count_before = 0;
  long long processed_count_before = 0;
  long long long_term_count_before = 0;
  {
    auto eligibility_store = cortext::SQLiteStore::Create (db_path.c_str ());
    signal_count_before = AnyToLongLong (
        eligibility_store->Execute ("SELECT COUNT(*) AS c FROM signals", {})
            [0]
            .at ("c"));
    processed_count_before = AnyToLongLong (
        eligibility_store->Execute (
            "SELECT signals_processed AS c FROM state WHERE id = 1", {})
            [0]
            .at ("c"));
    long_term_count_before = AnyToLongLong (
        eligibility_store->Execute (
            "SELECT COUNT(*) AS c FROM memories WHERE kind = 'LONG_TERM'", {})
            [0]
            .at ("c"));
  }

  const std::uint64_t consolidation_ts = source_ts + 86400000ULL;
  REQUIRE_NOTHROW (
      cortext::internal::ReplayIngress::ConsolidateAt (*ctx, consolidation_ts));
  ctx->Flush ();

  auto unique_store = cortext::SQLiteStore::Create (db_path.c_str ());
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  REQUIRE (AnyToLongLong (
               store->Execute ("SELECT COUNT(*) AS c FROM signals", {})[0]
                   .at ("c"))
           == signal_count_before);
  REQUIRE (AnyToLongLong (
               store->Execute (
                   "SELECT signals_processed AS c FROM state WHERE id = 1",
                   {})[0]
                   .at ("c"))
           == processed_count_before);
  REQUIRE (AnyToLongLong (
               store->Execute (
                   "SELECT COUNT(*) AS c FROM memories "
                   "WHERE kind = 'LONG_TERM'",
                   {})[0]
                   .at ("c"))
           == long_term_count_before);
  const long long association_count = AnyToLongLong (
      store->Execute (
          "SELECT COUNT(*) AS c FROM memories WHERE kind = 'ASSOCIATION'",
          {})[0]
          .at ("c"));
  // Clustering consolidation was removed: maintenance writes no nodes.
  REQUIRE (association_count == 0);
  const long long edge_count = AnyToLongLong (
      store->Execute ("SELECT COUNT(*) AS c FROM associations", {})[0]
          .at ("c"));
  REQUIRE_NOTHROW (
      cortext::internal::ReplayIngress::ConsolidateAt (*ctx, consolidation_ts));
  ctx->Flush ();
  REQUIRE (AnyToLongLong (
               store->Execute (
                   "SELECT COUNT(*) AS c FROM memories "
                   "WHERE kind = 'ASSOCIATION'",
                   {})[0]
                   .at ("c"))
           == association_count);
  REQUIRE (AnyToLongLong (
               store->Execute ("SELECT COUNT(*) AS c FROM associations", {})[0]
                   .at ("c"))
           == edge_count);
}

TEST_CASE ("Cortext::Create can resolve AIST from env",
           "[cortext][models][aist]")
{
  const auto model_path = std::filesystem::path (RepoModelsDir ())
                          / "AIST-87M-GGUF" / "AIST-87M_q8_0.gguf";
  cortext::testing::ScopedEnvVar model_override (
      "CORTEXT_AIST_MODEL_PATH", model_path.string ());
  cortext::Cortext::Config cfg;
  std::unique_ptr<cortext::Cortext> ctx;
  REQUIRE_NOTHROW (ctx = cortext::Cortext::Create (cfg, ":memory:"));
  REQUIRE (ctx != nullptr);
}

TEST_CASE ("Cortext C ABI stubs return success",
           "[cortext][capi][stub][aist]")
{
  cortext_handle h = cortext_create (0.5, 0.5, 0.5, ":memory:");
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
               &cfg, nullptr, nullptr)
           == nullptr);

  cortext_db_callbacks callbacks{};
  callbacks.struct_size = sizeof (callbacks);
  REQUIRE (cortext_create_with_store_callbacks (
               &cfg, &callbacks, nullptr)
           == nullptr);
}

TEST_CASE ("Cortext hydrates sqlite-objstore payloads",
           "[cortext][objstore][hydration][aist]")
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
  auto ctx = cortext::Cortext::Create (cfg, db_path);
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
#if defined(CORTEXT_EXPERIMENT_HOOKS)
        REQUIRE (memory.content.empty ());
#else
        REQUIRE (memory.content.size () == 1);
        REQUIRE (memory.content[0] == expected_payloads.at (memory.id));
#endif
        REQUIRE (memory.mimetype == expected_mimes.at (memory.id));
      }
  }
}

TEST_CASE ("Fallback hydration keeps a knob-bounded recent slice across "
           "sources and modalities",
           "[cortext][objstore][hydration][bounded][aist]")
{
  ScopedTempDb temp_db;
  const auto &db_path = temp_db.path ();
  auto store = cortext::SQLiteStore::Create (db_path);
  cortext::testing::InitializeCoreSchema (*store);

  constexpr double kFocus = 0.5;
  constexpr double kSensitivity = 0.5;
  constexpr double kStability = 0.5;
  const int limit
      = cortext::operations::sparse_retrieval_knobs_internal::
            HydrationFallbackSignalLimit (kFocus, kSensitivity, kStability);
  REQUIRE (limit == 128);

  struct Surface
  {
    long long memory_id;
    std::string source_id;
    std::string modality;
    std::string mime;
  };
  const std::array<Surface, 3> surfaces = {
    Surface{ 101, "opaque-source-7f", "text", "text/plain" },
    Surface{ 202, "device://opaque/91", "audio", "audio/pcm" },
    Surface{ 303, "urn:opaque:source:ab", "image", "image/png" },
  };

  for (std::size_t surface_index = 0; surface_index < surfaces.size ();
       ++surface_index)
    {
      const auto &surface = surfaces[surface_index];
      const int row_count = limit + 7;
      store->Execute (
          "INSERT INTO memories(memory_id, source_id, kind, start_ts, "
          "end_ts, n_signals, modality, created_at) "
          "VALUES (?, ?, 'WORKING', ?, NULL, ?, ?, ?)",
          { surface.memory_id, surface.source_id,
            static_cast<long long> (surface_index + 1),
            static_cast<long long> (row_count), surface.modality,
            static_cast<long long> (surface_index + 1) });

      for (int serial = 0; serial < row_count; ++serial)
        {
          std::vector<unsigned char> blob_id (32, 0);
          blob_id[0] = static_cast<unsigned char> (surface_index + 1);
          blob_id[1] = static_cast<unsigned char> (serial & 0xff);
          blob_id[2]
              = static_cast<unsigned char> ((serial >> 8) & 0xff);
          const std::string marker
              = std::to_string (surface.memory_id) + ":"
                + std::to_string (serial);
          const std::vector<unsigned char> payload (marker.begin (),
                                                    marker.end ());
          store->Execute (
              "INSERT INTO objstore_data(id, data) VALUES (?, ?)",
              { blob_id, payload });
          store->Execute (
              "INSERT INTO signals(memory_id, source_id, embedding_id, "
              "blob_id, timestamp, modality, mime, serial_position, "
              "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
              { surface.memory_id, surface.source_id,
                static_cast<long long> (100000 + serial), blob_id,
                static_cast<long long> (serial), surface.modality,
                surface.mime, static_cast<long long> (serial),
                static_cast<long long> (serial) });
        }
    }

  cortext::Cortext::Config cfg;
  cfg.focus = kFocus;
  cfg.sensitivity = kSensitivity;
  cfg.stability = kStability;
  auto ctx = cortext::Cortext::Create (cfg, db_path);
  REQUIRE (ctx != nullptr);
  const auto hydrated = ctx->DebugHydrateForTest ({}, {});

  REQUIRE (hydrated.working_memory.size () == surfaces.size ());
  REQUIRE (hydrated.output.operation_ms.at (
               "Cortext.fallback_hydration_signal_limit")
           == 128.0);
  REQUIRE (hydrated.output.operation_ms.at (
               "Cortext.fallback_hydration_signal_rows")
           == 384.0);
  for (std::size_t i = 0; i < surfaces.size (); ++i)
    {
      const auto &surface = surfaces[i];
      const auto &memory = hydrated.working_memory[i];
      INFO (surface.modality);
      REQUIRE (memory.id == surface.memory_id);
      REQUIRE (memory.source_id == surface.source_id);
      REQUIRE (memory.modality == surface.modality);
      REQUIRE (memory.mimetype == surface.mime);
      REQUIRE (memory.content.size () == static_cast<std::size_t> (limit));
      const std::string first (memory.content.front ().begin (),
                               memory.content.front ().end ());
      const std::string last (memory.content.back ().begin (),
                              memory.content.back ().end ());
      REQUIRE (first == std::to_string (surface.memory_id) + ":7");
      REQUIRE (last == std::to_string (surface.memory_id) + ":134");
    }
}

TEST_CASE ("Cortext filters mixed signal blobs to the memory surface",
           "[cortext][hydration][aist]")
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
  auto ctx = cortext::Cortext::Create (cfg, db_path);
  REQUIRE (ctx != nullptr);

  auto hydrated = ctx->DebugHydrateForTest ({ 701LL }, {});
  REQUIRE (hydrated.retrieved_memory.size () == 1);
  const auto &memory = hydrated.retrieved_memory[0];
  REQUIRE (memory.modality == "text");
  REQUIRE (memory.mimetype == "text/plain");
  REQUIRE (memory.content.size () == 1);
  REQUIRE (TextFromMemory (memory) == source_text);
}

TEST_CASE ("Fallback hydration applies its bound after surface filtering",
           "[cortext][objstore][hydration][bounded][aist]")
{
  ScopedTempDb temp_db;
  const auto &db_path = temp_db.path ();
  auto store = cortext::SQLiteStore::Create (db_path);
  cortext::testing::InitializeCoreSchema (*store);

  constexpr long long kMemoryId = 702;
  constexpr int kLimit = 128;
  std::vector<float> embedding (256, 0.0f);
  embedding[0] = 1.0f;
  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) "
      "VALUES (?, ?, 0)",
      { kMemoryId, embedding });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, "
      "start_ts, end_ts, n_signals, modality, strength, created_at) "
      "VALUES (?, ?, 'opaque/mixed', 'LONG_TERM', 0, 300, ?, 'text', 1, 0)",
      { kMemoryId, kMemoryId, static_cast<long long> (kLimit + 1) });

  const std::string expected = "older matching text payload";
  const std::vector<unsigned char> text_payload (expected.begin (),
                                                 expected.end ());
  const auto text_blob_rows = store->Execute (
      "SELECT objstore_put(?1) AS id", { text_payload });
  const auto text_blob_id = BlobFromAny (text_blob_rows[0].at ("id"));
  store->Execute (
      "INSERT INTO signals(memory_id, embedding_id, source_id, timestamp, "
      "modality, mime, blob_id, serial_position, created_at) "
      "VALUES (?, ?, 'opaque/mixed', 0, 'text', 'text/plain', ?, 0, 0)",
      { kMemoryId, kMemoryId, text_blob_id });

  const std::vector<unsigned char> invalid_text_payload
      = { 0, 1, 2, 3, 255 };
  const auto invalid_blob_rows = store->Execute (
      "SELECT objstore_put(?1) AS id", { invalid_text_payload });
  const auto invalid_blob_id = BlobFromAny (invalid_blob_rows[0].at ("id"));
  for (int serial = 1; serial <= kLimit; ++serial)
    {
      store->Execute (
          "INSERT INTO signals(memory_id, embedding_id, source_id, "
          "timestamp, modality, mime, blob_id, serial_position, created_at) "
          "VALUES (?, ?, 'opaque/mixed', ?, 'text', 'text/plain', ?, ?, ?)",
          { kMemoryId, kMemoryId, static_cast<long long> (serial),
            invalid_blob_id, static_cast<long long> (serial),
            static_cast<long long> (serial) });
    }

  cortext::Cortext::Config cfg;
  auto ctx = cortext::Cortext::Create (cfg, db_path);
  REQUIRE (ctx != nullptr);
  const auto hydrated = ctx->DebugHydrateForTest ({ kMemoryId }, {});
  REQUIRE (hydrated.retrieved_memory.size () == 1);
  REQUIRE (hydrated.retrieved_memory[0].content.size () == 1);
  REQUIRE (TextFromMemory (hydrated.retrieved_memory[0]) == expected);
  REQUIRE (hydrated.output.operation_ms.at (
               "Cortext.fallback_hydration_signal_rows")
           == 1.0);
}

TEST_CASE ("Cortext hides unresolved internal retrieval nodes",
           "[cortext][hydration][retrieval][aist]")
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
  auto ctx = cortext::Cortext::Create (cfg, db_path);
  REQUIRE (ctx != nullptr);

  auto cue_hydrated = ctx->DebugHydrateForTest ({ 200LL }, {});
  REQUIRE (cue_hydrated.retrieved_memory.empty ());

  auto label_hydrated = ctx->DebugHydrateForTest ({ 300LL }, {});
  REQUIRE (label_hydrated.retrieved_memory.empty ());
}

TEST_CASE ("C API handles NULL handles without a model",
           "[cortext][capi][safety]")
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

  SECTION ("cortext_embed_text_json reports NULL handle")
  {
    char *json_ptr = cortext_embed_text_json (nullptr, "hello");
    REQUIRE (json_ptr == nullptr);
    REQUIRE (std::string (cortext_last_error ())
             == "handle and text must both be non-NULL");
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

  SECTION ("JSON C API reports errors through cortext_last_error")
  {
    char *json_ptr = cortext_process_text_json (nullptr, "hello", "src");
    REQUIRE (json_ptr == nullptr);
    REQUIRE (std::string (cortext_last_error ())
             == "handle, text, and source_id must all be non-NULL");
  }
}

TEST_CASE ("C API handles NULL inputs correctly",
           "[cortext][capi][safety][aist]")
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
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
    REQUIRE (h != nullptr);
    CHECK (cortext_process_text (h, nullptr, "src") == 1);
    REQUIRE (std::string (cortext_last_error ())
             == "handle, text, and source_id must all be non-NULL");
    cortext_free (h);
  }

  SECTION ("cortext_process_text returns 1 for NULL source_id")
  {
    ScopedTempDb temp_db;
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
    REQUIRE (h != nullptr);
    CHECK (cortext_process_text (h, "text", nullptr) == 1);
    REQUIRE (std::string (cortext_last_error ())
             == "handle, text, and source_id must all be non-NULL");
    cortext_free (h);
  }

  SECTION ("cortext_process_audio returns 1 for NULL pcm")
  {
    ScopedTempDb temp_db;
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
    REQUIRE (h != nullptr);
    CHECK (cortext_process_audio (h, nullptr, 100, "src") == 1);
    REQUIRE (std::string (cortext_last_error ())
             == "handle, pcm, and source_id must all be non-NULL");
    cortext_free (h);
  }

  SECTION ("cortext_process_audio_with_media rejects invalid media")
  {
    ScopedTempDb temp_db;
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
    REQUIRE (h != nullptr);
    const float pcm[2] = { 0.0f, 0.0f };
    cortext_media media{};
    media.size = 4;
    media.mimetype = "audio/wav";
    CHECK (cortext_process_audio_with_media (h, pcm, 2, "src", &media) == 1);
    REQUIRE (std::string (cortext_last_error ())
             == "media data must be non-NULL when media size is non-zero");
    cortext_free (h);
  }

  SECTION ("cortext_process_image returns 1 for NULL data")
  {
    ScopedTempDb temp_db;
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
    REQUIRE (h != nullptr);
    CHECK (cortext_process_image (h, nullptr, 10, 10, 3, "src") == 1);
    REQUIRE (std::string (cortext_last_error ())
             == "handle, data, and source_id must all be non-NULL");
    cortext_free (h);
  }

  SECTION ("cortext_process_image_with_media rejects invalid media")
  {
    ScopedTempDb temp_db;
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
    REQUIRE (h != nullptr);
    const std::uint8_t px[3] = { 0, 0, 0 };
    const std::uint8_t media_bytes[4] = { 'R', 'I', 'F', 'F' };
    cortext_media media{};
    media.data = media_bytes;
    media.size = sizeof (media_bytes);
    CHECK (
        cortext_process_image_with_media (h, px, 1, 1, 3, "src", &media)
        == 1);
    REQUIRE (std::string (cortext_last_error ())
             == "media mimetype must be non-NULL and non-empty when media data is provided");
    cortext_free (h);
  }

  SECTION ("legacy image-with-media JSON entry point remains Durable")
  {
    ScopedTempDb temp_db;
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
    REQUIRE (h != nullptr);
    const std::uint8_t px[3] = { 12, 34, 56 };
    cortext_media media{ px, sizeof (px), "image/raw" };
    char *json_ptr = cortext_process_image_with_media_json (
        h, px, 1, 1, 3, "legacy/media", &media);
    REQUIRE (json_ptr != nullptr);
    const auto parsed = nlohmann::json::parse (json_ptr);
    REQUIRE (parsed.at ("boundary_type") == "explicit_turn");
    REQUIRE (parsed.at ("output").at ("stored_memory_id").is_number_integer ());
    cortext_string_free (json_ptr);
    cortext_free (h);
  }

  SECTION ("legacy audio-with-media JSON entry point remains Durable")
  {
    ScopedTempDb temp_db;
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
    REQUIRE (h != nullptr);
    const std::array<float, 160> pcm{};
    const std::uint8_t media_bytes[4] = { 'R', 'I', 'F', 'F' };
    cortext_media media{ media_bytes, sizeof (media_bytes), "audio/wav" };
    char *json_ptr = cortext_process_audio_with_media_json (
        h, pcm.data (), pcm.size (), "legacy/media", &media);
    REQUIRE (json_ptr != nullptr);
    const auto parsed = nlohmann::json::parse (json_ptr);
    REQUIRE (parsed.at ("boundary_type") == "explicit_turn");
    REQUIRE (parsed.at ("output").at ("stored_memory_id").is_number_integer ());
    cortext_string_free (json_ptr);
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
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
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
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
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
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
    REQUIRE (h != nullptr);

    cortext_process_json_options durable_opts{};
    durable_opts.struct_size = sizeof (durable_opts);
    cortext_process_json_options_init (&durable_opts);
    durable_opts.retention = CORTEXT_RETENTION_DURABLE;

    char *before_json = cortext_process_text_json_with_options (
        h, "before reset", "reset/source", &durable_opts);
    REQUIRE (before_json != nullptr);
    cortext_string_free (before_json);

    REQUIRE (cortext_reset (h) == 0);
    REQUIRE (cortext_last_error () == nullptr);

    char *json_ptr = cortext_process_text_json_with_options (
        h, "after reset", "reset/source", &durable_opts);
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

    auto h = cortext_create_with_config (&cfg, temp_db.path ().c_str ());
    REQUIRE (h != nullptr);
    cortext_free (h);
  }

  SECTION ("JSON C API returns parseable context")
  {
    ScopedTempDb temp_db;
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
    REQUIRE (h != nullptr);

    cortext_process_json_options durable_opts{};
    durable_opts.struct_size = sizeof (durable_opts);
    cortext_process_json_options_init (&durable_opts);
    durable_opts.retention = CORTEXT_RETENTION_DURABLE;

    char *json_ptr = cortext_process_text_json_with_options (
        h, "json api stores ids", "json/source", &durable_opts);
    REQUIRE (json_ptr != nullptr);

    auto parsed = nlohmann::json::parse (json_ptr);
    REQUIRE (parsed.contains ("output"));
    REQUIRE (parsed.at ("output").contains ("stored_memory_id"));
    REQUIRE (parsed.at ("output").contains ("stored_signal_id"));
    REQUIRE (parsed.at ("output").at ("stored_memory_id").is_number_integer ());
    REQUIRE (parsed.at ("output").at ("stored_signal_id").is_number_integer ());
    REQUIRE (parsed.at ("consolidation_state") == "none");
    REQUIRE_FALSE (parsed.contains ("consolidation_recommended"));
    REQUIRE_FALSE (parsed.contains ("consolidation_required"));
    REQUIRE (parsed.contains ("embedding"));
    REQUIRE (parsed.contains ("embedding_dimension"));
    REQUIRE (parsed.at ("embedding").is_array ());
    REQUIRE_FALSE (parsed.at ("embedding").empty ());
    REQUIRE (parsed.at ("embedding_dimension").get<std::size_t> ()
             == parsed.at ("embedding").size ());
    cortext_string_free (json_ptr);

    cortext_process_json_options options{};
    options.struct_size = sizeof (options);
    cortext_process_json_options_init (&options);
    REQUIRE (options.include_embedding == 1);
    REQUIRE (options.retention == CORTEXT_RETENTION_NATURAL);
    options.include_embedding = 0;
    options.retention = CORTEXT_RETENTION_DURABLE;
    json_ptr = cortext_process_text_json_with_options (
        h, "json api can omit embedding", "json/source", &options);
    REQUIRE (json_ptr != nullptr);
    parsed = nlohmann::json::parse (json_ptr);
    REQUIRE_FALSE (parsed.contains ("embedding"));
    REQUIRE_FALSE (parsed.contains ("embedding_dimension"));
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

  SECTION ("JSON options initializer preserves the legacy struct prefix")
  {
    struct LegacyProcessJsonOptions
    {
      std::size_t struct_size;
      int include_embedding;
    };
    struct LegacyOptionsWithGuard
    {
      LegacyProcessJsonOptions options;
      int guard = 0x1234;
    } legacy;

    legacy.options.struct_size = sizeof (legacy.options);
    cortext_process_json_options_init (
        reinterpret_cast<cortext_process_json_options *> (&legacy.options));

    REQUIRE (legacy.options.struct_size
             == offsetof (cortext_process_json_options, retention));
    REQUIRE (legacy.options.include_embedding == 1);
    REQUIRE (legacy.guard == 0x1234);
  }

  SECTION ("Embed JSON C API returns parseable embedding")
  {
    ScopedTempDb temp_db;
    auto h = cortext_create (0.5, 0.5, 0.5, temp_db.path ().c_str ());
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

TEST_CASE ("Sparse route growth recommends consolidation and recenters the "
           "route repeatedly on a long stream",
           "[cortext][consolidation][sparse-route][long]")
{
  // F = S = T = 0 gives the smallest route capacity C, so the SQLite
  // sparse route publishes and then outgrows its activation centroid
  // several times within a short stream. The caller follows the engine's
  // hint like a host would: consolidate whenever it is not None.
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  cfg.focus = 0.0;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;
  auto engine = cortext::Cortext::Create (cfg, temp_db.path ());
  REQUIRE (engine != nullptr);
  const auto route_capacity = static_cast<std::size_t> (
      cortext::operations::sparse_retrieval_knobs_internal::RouteCapacity (
          cfg.focus, cfg.sensitivity, cfg.stability));

  static const std::array<const char *, 12> kSubjects
      = { "Maya",   "the courier", "Dr. Okafor", "my landlord",
          "Priya",  "the vet",     "Tomas",      "our neighbor",
          "Lena",   "the plumber", "Grandpa Joe", "the tutor" };
  static const std::array<const char *, 12> kActions
      = { "repaired the bicycle",     "booked a flight to Lisbon",
          "planted tomatoes",         "lost the spare key",
          "baked sourdough bread",    "renewed the passport",
          "painted the fence blue",   "adopted a grey kitten",
          "sold the old piano",       "fixed the leaking sink",
          "started learning Korean",  "ran a half marathon" };
  static const std::array<const char *, 10> kWhen
      = { "on Monday",     "last winter",   "after lunch",
          "in March",      "at dawn",       "before the storm",
          "during the trip", "on my birthday", "late at night",
          "this weekend" };

  const std::size_t turns = 4 * route_capacity;
  std::size_t consolidations = 0;
  std::size_t recentered = 0;
  std::size_t first_recenter_turn = 0;
  std::uint64_t ts = 1'700'000'000'000ULL;
  for (std::size_t i = 0; i < turns; ++i)
    {
      const std::string text
          = std::string (kSubjects[i % kSubjects.size ()]) + " "
            + kActions[(i / kSubjects.size ()) % kActions.size ()] + " "
            + kWhen[(i * 7) % kWhen.size ()] + ", note "
            + std::to_string (i) + " with code " + std::to_string (i * 7919 % 10007)
            + ".";
      ts += 60'000;
      const auto context = cortext::internal::ReplayIngress::ProcessTextAt (
          *engine, text, "stream", ts, cortext::Retention::Durable, false);
      if (context.consolidation_state == cortext::ConsolidationState::None)
        continue;
      ts += 1'000;
      const auto maintenance
          = cortext::internal::ReplayIngress::ConsolidateAt (*engine, ts);
      ++consolidations;
      const auto it = maintenance.output.operation_ms.find (
          "SignalProcessor.sqlite_sparse_route_recenter_succeeded");
      if (it != maintenance.output.operation_ms.end () && it->second == 1.0)
        {
          if (recentered == 0)
            first_recenter_turn = i;
          ++recentered;
        }
    }
  INFO ("route capacity " << route_capacity << ", consolidations "
                          << consolidations << ", recentered " << recentered
                          << ", first recenter at turn "
                          << first_recenter_turn);
  // Once the route is published, every C new memories must earn a
  // recentering consolidation: at least two more after the first.
  REQUIRE (recentered >= 3);
  REQUIRE (consolidations <= turns / (route_capacity / 2));
}

namespace
{
const std::vector<std::uint8_t> kFakeJpeg = { 0xFF, 0xD8, 0xFF, 0xE0, 0x00,
                                              0x10, 'J',  'F',  'I',  'F',
                                              0x00, 0xFF, 0xD9 };

const cortext::Cortext::Context::Memory *
FindMemoryWithText (const std::vector<cortext::Cortext::Context::Memory> &mems,
                    const std::string &needle)
{
  for (const auto &memory : mems)
    {
      for (const auto &blob : memory.content)
        {
          if (std::string (blob.begin (), blob.end ()).find (needle)
              != std::string::npos)
            return &memory;
        }
    }
  return nullptr;
}

std::vector<std::string>
ContentMimes (const cortext::Cortext::Context::Memory &memory)
{
  std::vector<std::string> out;
  for (std::size_t i = 0; i < memory.content.size (); ++i)
    {
      out.push_back (i < memory.content_mimetypes.size ()
                         ? memory.content_mimetypes[i]
                         : memory.mimetype);
    }
  return out;
}
} // namespace

TEST_CASE ("Cortext multi-modality memory keeps and returns every signal",
           "[cortext][media][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  auto store = std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (temp_db.path ().c_str ()));
  auto ctx = cortext::Cortext::Create (cfg, store);

  // One turn = one source: a text signal and the photo sent with it, as
  // peers. The turn's last signal commits it.
  (void)ctx->ProcessTextAt ("The parking garage ticket says level 4 spot 27.",
                            "chat/user/turn1", 1000ULL,
                            cortext::Retention::Natural);
  std::vector<std::uint8_t> pixels (8 * 8 * 3, 128);
  cortext::Cortext::Media photo;
  photo.data = kFakeJpeg.data ();
  photo.size = kFakeJpeg.size ();
  photo.mimetype = "image/jpeg";
  auto commit = ctx->ProcessImage (pixels.data (), 8, 8, 3, "chat/user/turn1",
                                   photo, cortext::Retention::Durable);
  REQUIRE (commit.output.stored_memory_id.has_value ());
  const long long memory_id = *commit.output.stored_memory_id;
  const auto signals = store->Execute (
      "SELECT COUNT(*) AS n FROM signals WHERE memory_id = ?", { memory_id });
  REQUIRE (AnyToLongLong (signals[0].at ("n")) == 2);
  const auto rows = store->Execute (
      "SELECT modality FROM memories WHERE memory_id = ?", { memory_id });
  REQUIRE (std::any_cast<std::string> (rows[0].at ("modality")) == "mixed");
  // The text signal is keyword-indexed without making the memory "text".
  const auto fts = store->Execute (
      "SELECT rowid FROM memory_text_fts WHERE memory_text_fts MATCH 'garage'");
  REQUIRE (fts.size () == 1);

  auto recalled = ctx->ProcessTextAt ("parking garage level spot",
                                      "query/public", 2000ULL,
                                      cortext::Retention::Ephemeral);
  const auto *memory
      = FindMemoryWithText (recalled.retrieved_memory, "level 4 spot 27");
  REQUIRE (memory != nullptr);
  REQUIRE (memory->modality == "mixed");
  const auto mimes = ContentMimes (*memory);
  REQUIRE (std::count (mimes.begin (), mimes.end (), "text/plain") == 1);
  REQUIRE (std::count (mimes.begin (), mimes.end (), "image/jpeg") == 1);
  bool photo_returned = false;
  for (const auto &blob : memory->content)
    photo_returned = photo_returned
                     || blob == std::vector<unsigned char> (kFakeJpeg.begin (),
                                                            kFakeJpeg.end ());
  REQUIRE (photo_returned);

  const auto *slot
      = FindMemoryWithText (recalled.working_memory, "level 4 spot 27");
  REQUIRE (slot != nullptr);
  const auto slot_mimes = ContentMimes (*slot);
  REQUIRE (std::count (slot_mimes.begin (), slot_mimes.end (), "image/jpeg")
           == 1);
}

namespace
{
std::vector<float>
RetrievalView (std::vector<float> v)
{
  v.resize (256);
  double norm = 0.0;
  for (float x : v)
    norm += static_cast<double> (x) * x;
  for (float &x : v)
    x = static_cast<float> (x / std::sqrt (norm));
  return v;
}
} // namespace

TEST_CASE ("Cortext recalls memories across modalities in both directions",
           "[cortext][media][aist]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  cfg.signal_filter_image_enabled = false;
  auto store = std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (temp_db.path ().c_str ()));
  auto clock = std::make_shared<cortext::FixedClock> (1000);
  auto ctx = cortext::Cortext::Create (cfg, store, clock);
  const auto set_vector = [&] (long long memory_id, const std::string &modality,
                               const std::vector<float> &v) {
    store->Execute ("UPDATE memory_signal_vectors SET embedding = ? "
                    "WHERE blob_id IN (SELECT blob_id FROM signals "
                    "WHERE memory_id = ? AND modality = ?)",
                    { v, memory_id, modality });
  };

  // Photos (image memories). Their stored vectors stand in for what each
  // photo shows; the image tower itself is not under test.
  const std::vector<std::string> scenes = {
    "a bowl of tomato soup",        "a snowy mountain trail",
    "a laptop on a desk",           "a cat sleeping on a sofa",
    "a crowded subway platform",    "a plate of sushi",
    "a sunset over the ocean",      "a pile of autumn leaves",
    "a birthday cake with candles", "a city skyline at night",
    "a red bicycle leaning on a fence",
  };
  std::uint64_t ts = 1000;
  long long bicycle_photo = 0;
  for (std::size_t i = 0; i < scenes.size (); ++i)
    {
      std::vector<std::uint8_t> pixels (8 * 8 * 3,
                                        static_cast<std::uint8_t> (20 * i));
      std::vector<std::uint8_t> bytes = kFakeJpeg;
      bytes.push_back (static_cast<std::uint8_t> (i));
      cortext::Cortext::Media photo;
      photo.data = bytes.data ();
      photo.size = bytes.size ();
      photo.mimetype = "image/jpeg";
      clock->SetNowMillis (ts += 60000);
      auto ingest = ctx->ProcessImage (pixels.data (), 8, 8, 3,
                                       "camera/roll/" + std::to_string (i),
                                       photo, cortext::Retention::Durable);
      REQUIRE (ingest.output.stored_memory_id.has_value ());
      set_vector (*ingest.output.stored_memory_id, "image",
                  RetrievalView (ctx->EmbedText (scenes[i])));
      if (i + 1 == scenes.size ())
        bicycle_photo = *ingest.output.stored_memory_id;
    }
  // Notes (text memories); one is about what a query photo will show.
  std::vector<std::uint8_t> query_pixels (8 * 8 * 3, 77);
  long long matching_note = 0;
  for (int i = 0; i < 40; ++i)
    {
      const bool target = i == 39;
      auto ingest = ctx->ProcessTextAt (
          target ? "Remember to water the plants on Friday."
                 : "Weekly note " + std::to_string (i)
                       + ": the red invoice for the bicycle shop fence repair"
                         " is due next week.",
          "notes/main", ts += 60000, cortext::Retention::Durable);
      clock->SetNowMillis (ts);
      REQUIRE (ingest.output.stored_memory_id.has_value ());
      if (target)
        {
          matching_note = *ingest.output.stored_memory_id;
          set_vector (matching_note, "text",
                      RetrievalView (ctx->EmbedImage (query_pixels.data (), 8,
                                                      8, 3)));
        }
    }

  const auto contains = [] (const cortext::Cortext::Context &c, long long id) {
    return std::any_of (c.retrieved_memory.begin (), c.retrieved_memory.end (),
                        [id] (const auto &m) { return m.id == id; });
  };
  clock->SetNowMillis (ts += 60000);
  auto by_text = ctx->ProcessTextAt ("the red bicycle by the fence",
                                     "query/public", ts,
                                     cortext::Retention::Ephemeral);
  INFO ("xmodal hits "
        << by_text.output.operation_ms.at ("GraphRetrieve.cross_modal_hits")
        << " added "
        << by_text.output.operation_ms.at ("GraphRetrieve.cross_modal_added")
        << " vectors "
        << AnyToLongLong (store->Execute ("SELECT COUNT(*) AS n FROM "
                                          "memory_signal_vectors")[0]
                              .at ("n"))
        << " retrieved " << by_text.retrieved_memory.size ());
  REQUIRE (contains (by_text, bicycle_photo));
  REQUIRE (by_text.output.operation_ms.at ("GraphRetrieve.cross_modal_hits")
           >= 1.0);

  clock->SetNowMillis (ts += 60000);
  auto by_image = ctx->ProcessImage (query_pixels.data (), 8, 8, 3,
                                     "query/camera",
                                     cortext::Retention::Ephemeral);
  REQUIRE (contains (by_image, matching_note));
}

namespace
{
// Unrelated turns, each its own stream and hours apart, so a query has more
// candidates than packet anchors and none of them is anyone's neighbour.
void
AddUnrelatedTurns (cortext::Cortext &ctx, cortext::FixedClock *clock,
                   std::uint64_t &ts)
{
  const std::vector<std::string> topics = {
    "The quarterly tax filing is due at the end of April.",
    "My sister adopted a grey kitten named Pebble.",
    "The bakery on Elm Street sells sourdough on Sundays.",
    "We repainted the guest bedroom a pale green colour.",
    "The marathon route passes the old harbour bridge.",
    "Our team switched the build system to Bazel last month.",
    "The dentist moved my cleaning appointment to Friday.",
    "Grandpa's vinyl collection includes rare jazz records.",
    "The library extended its weekend opening hours.",
    "I need new brake pads for the blue hatchback.",
    "The community garden plot gets full sun until noon.",
    "Our flight to Lisbon departs from terminal two.",
    "The thermostat schedule drops to sixty at night.",
    "She is learning to play the cello with a tutor.",
  };
  for (std::size_t i = 0; i < topics.size (); ++i)
    {
      ts += 3ULL * 60ULL * 60ULL * 1000ULL;
      if (clock)
        clock->SetNowMillis (ts);
      (void)ctx.ProcessTextAt (topics[i],
                               "other/" + std::to_string (i) + "/0", ts,
                               cortext::Retention::Durable);
    }
}

std::string
PacketText (const cortext::Cortext::Context &context)
{
  std::string out;
  for (const auto &memory : context.retrieved_memory)
    for (std::size_t i = 0; i < memory.content.size (); ++i)
      {
        const auto mimes = ContentMimes (memory);
        if (mimes[i].rfind ("text/", 0) == 0 || mimes[i].empty ())
          out.append (memory.content[i].begin (), memory.content[i].end ())
              .append ("\n");
      }
  return out;
}

} // namespace

TEST_CASE ("Standing directives lead query packets whatever their rank",
           "[cortext][retrieval][standing]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  auto store = std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (temp_db.path ().c_str ()));
  auto ctx = cortext::Cortext::Create (cfg, store);
  std::uint64_t ts = 1000;
  const auto directive = ctx->ProcessTextAt (
      "Always include version numbers when I ask about software libraries.",
      "chat/rules/1", ts, cortext::Retention::Durable);
  REQUIRE (directive.output.stored_memory_id.has_value ());
  ts += 3ULL * 60ULL * 60ULL * 1000ULL;
  // A long turn that mentions "always" is not a standing directive.
  std::string explanation = "Always check the lockfile first. ";
  for (int i = 0; i < 12; ++i)
    explanation += "The resolver reads the manifest and pins transitive "
                   "dependencies to exact versions. ";
  const auto long_turn = ctx->ProcessTextAt (explanation, "chat/notes/1", ts,
                                             cortext::Retention::Durable);
  REQUIRE (long_turn.output.stored_memory_id.has_value ());
  // Turns about the query topic fill every anchor slot ahead of the
  // directive.
  for (int i = 0; i < 16; ++i)
    {
      ts += 3ULL * 60ULL * 60ULL * 1000ULL;
      (void)ctx->ProcessTextAt (
          "Which libraries does the reporting service use, part "
              + std::to_string (i) + "? It uses pandas and requests.",
          "chat/libs/" + std::to_string (i), ts, cortext::Retention::Durable);
    }
  ts += 3ULL * 60ULL * 60ULL * 1000ULL;
  AddUnrelatedTurns (*ctx, nullptr, ts);

  const auto packet = ctx->ProcessTextAt (
      "Which libraries does the reporting service use?", "query/public",
      ts + 1000, cortext::Retention::Ephemeral);
  REQUIRE_FALSE (packet.retrieved_memory.empty ());
  REQUIRE (packet.retrieved_memory.front ().id
           == *directive.output.stored_memory_id);
  REQUIRE (packet.output.operation_ms.at ("Cortext.standing_joined") == 1.0);
  for (const auto &memory : packet.retrieved_memory)
    if (memory.id == *long_turn.output.stored_memory_id)
      REQUIRE (&memory != &packet.retrieved_memory.front ());

  // Ingest does not join standing memories.
  ts += 3ULL * 60ULL * 60ULL * 1000ULL;
  const auto ingest = ctx->ProcessTextAt (
      "Which libraries does the reporting service use?", "chat/libs/x", ts,
      cortext::Retention::Durable);
  const auto joined = ingest.output.operation_ms.find ("Cortext.standing_joined");
  REQUIRE ((joined == ingest.output.operation_ms.end () || joined->second == 0.0));
}

TEST_CASE ("A directive inside a grouped memory leads query packets",
           "[cortext][retrieval][standing][retention][aist]")
{
  cortext::testing::ScopedEnvVar disable_natural_boundary (
      "CORTEXT_BOUNDARY_DISABLE_NATURAL", "1");
  cortext::testing::ScopedEnvVar disable_pressure_boundary (
      "CORTEXT_BOUNDARY_DISABLE_PRESSURE", "1");
  cortext::testing::ScopedEnvVar disable_surprisal_boundary (
      "CORTEXT_BOUNDARY_DISABLE_SURPRISAL", "1");
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  auto store = std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (temp_db.path ().c_str ()));
  auto ctx = cortext::Cortext::Create (cfg, store);
  std::uint64_t ts = 1000;
  // A Natural unit: a long turn, the directive, another long turn.
  std::string before = "The bakery oven runs hot on the left side. ";
  while (before.size () < 2000)
    before += "The bakery oven runs hot on the left side. ";
  std::string after = "The delivery van needs new tires next month. ";
  while (after.size () < 2000)
    after += "The delivery van needs new tires next month. ";
  const std::string directive
      = "Always include version numbers when I ask about software libraries.";
  for (const std::string &text : std::vector<std::string>{ before, directive, after })
    {
      (void)ctx->ProcessTextAt (text, "chat/unit", ts,
                                cortext::Retention::Natural);
      ts += 60000;
    }
  ctx->Flush ();
  const auto members = store->Execute (
      "SELECT m.n_signals AS n FROM standing_signals ss "
      "JOIN signals s ON s.signal_id = ss.signal_id "
      "JOIN memories m ON m.memory_id = s.memory_id");
  REQUIRE (members.size () == 1);
  REQUIRE (AnyToLongLong (members[0].at ("n")) >= 2);
  for (int i = 0; i < 16; ++i)
    {
      ts += 3ULL * 60ULL * 60ULL * 1000ULL;
      (void)ctx->ProcessTextAt (
          "Which libraries does the reporting service use, part "
              + std::to_string (i) + "? It uses pandas and requests.",
          "chat/libs/" + std::to_string (i), ts, cortext::Retention::Durable);
    }
  const auto packet = ctx->ProcessTextAt (
      "Which libraries does the reporting service use?", "query/public",
      ts + 1000, cortext::Retention::Ephemeral);
  REQUIRE_FALSE (packet.retrieved_memory.empty ());
  std::string front;
  for (const auto &blob : packet.retrieved_memory.front ().content)
    front.append (blob.begin (), blob.end ());
  // The directive leads, kept by query focus within the unit's budget
  // (pinned ahead of members more similar to the query), or joined alone
  // when its memory was not retrieved.
  REQUIRE (front.find (directive) != std::string::npos);
  REQUIRE (front.size () <= 3072);
  REQUIRE (packet.output.operation_ms.at ("Cortext.standing_joined") == 1.0);
}

TEST_CASE ("Stores from before the standing set fill it once on open",
           "[cortext][standing][migration]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  long long directive_id = 0;
  {
    auto ctx = cortext::Cortext::Create (cfg, temp_db.path ());
    const auto directive = ctx->ProcessTextAt (
        "From now on, answer in French.", "chat/rules/1", 1000ULL,
        cortext::Retention::Durable);
    (void)ctx->ProcessTextAt ("I always forget my keys on Mondays.",
                              "chat/notes/1", 2000ULL,
                              cortext::Retention::Durable);
    REQUIRE (directive.output.stored_memory_id.has_value ());
    directive_id = *directive.output.stored_memory_id;
    ctx->Flush ();
  }
  {
    // A store written before migration 53: no members, backfill queued.
    auto store = std::shared_ptr<cortext::Store> (
        cortext::SQLiteStore::Create (temp_db.path ().c_str ()));
    store->Execute ("DELETE FROM standing_signals");
    store->Execute ("INSERT INTO cortext_pending_backfills(name) "
                    "VALUES ('standing_signals')");
  }
  auto store = std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (temp_db.path ().c_str ()));
  auto ctx = cortext::Cortext::Create (cfg, store);
  (void)ctx->ProcessTextAt ("What language should replies use?",
                            "query/public", 5000ULL,
                            cortext::Retention::Ephemeral);
  const auto members = store->Execute (
      "SELECT s.memory_id AS memory_id FROM standing_signals ss "
      "JOIN signals s ON s.signal_id = ss.signal_id");
  REQUIRE (members.size () == 1);
  REQUIRE (AnyToLongLong (members[0].at ("memory_id")) == directive_id);
  REQUIRE (store->Execute ("SELECT 1 FROM cortext_pending_backfills "
                           "WHERE name = 'standing_signals'")
               .empty ());
}

TEST_CASE ("Query retrieval defers the adjacent turn of a higher anchor",
           "[cortext][retrieval][contiguity]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  auto store = std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (temp_db.path ().c_str ()));
  auto ctx = cortext::Cortext::Create (cfg, store);
  std::uint64_t ts = 1000;
  AddUnrelatedTurns (*ctx, nullptr, ts);
  ts += 3ULL * 60ULL * 60ULL * 1000ULL;
  // One exchange: both turns match the query, and hydration attaches each
  // to the other when only one of them is an anchor.
  const auto question = ctx->ProcessTextAt (
      "When does the zebra kite festival start at Marlow beach?",
      "chat/kites/1", ts, cortext::Retention::Durable);
  const auto reply = ctx->ProcessTextAt (
      "The zebra kite festival at Marlow beach starts Saturday at ten.",
      "chat/kites/2", ts + 60000, cortext::Retention::Durable);
  REQUIRE (question.output.stored_memory_id.has_value ());
  REQUIRE (reply.output.stored_memory_id.has_value ());
  ts += 3ULL * 60ULL * 60ULL * 1000ULL;
  AddUnrelatedTurns (*ctx, nullptr, ts);

  const auto packet = ctx->ProcessTextAt ("zebra kite festival Marlow beach",
                                          "query/public", ts + 1000,
                                          cortext::Retention::Ephemeral);
  std::size_t exchange_anchors = 0;
  for (const auto &memory : packet.retrieved_memory)
    if (memory.id == *question.output.stored_memory_id
        || memory.id == *reply.output.stored_memory_id)
      ++exchange_anchors;
  // One anchor slot for the exchange; the other turn rides along with it.
  REQUIRE (exchange_anchors == 1);
  const std::string text = PacketText (packet);
  REQUIRE (text.find ("When does the zebra kite festival") != std::string::npos);
  REQUIRE (text.find ("starts Saturday at ten") != std::string::npos);
  REQUIRE (packet.output.operation_ms.at ("GraphRetrieve.neighbours_deferred")
           >= 1.0);
}

TEST_CASE ("Attached turns keep the window that covers the query",
           "[cortext][retrieval][contiguity]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  auto store = std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (temp_db.path ().c_str ()));
  auto ctx = cortext::Cortext::Create (cfg, store);
  std::uint64_t ts = 1000;
  AddUnrelatedTurns (*ctx, nullptr, ts);
  ts += 3ULL * 60ULL * 60ULL * 1000ULL;
  (void)ctx->ProcessTextAt ("Can you book tickets for the orchid conservatory?",
                            "chat/garden/1", ts, cortext::Retention::Durable);
  std::string reply = "Sure. ";
  for (int i = 0; i < 24; ++i)
    reply += "Weekend traffic downtown is heavy, so leave early. ";
  reply += "The orchid conservatory opens at nine and tickets cost twelve.";
  REQUIRE (reply.size () > 1000);
  (void)ctx->ProcessTextAt (reply, "chat/garden/2", ts + 60000,
                            cortext::Retention::Durable);
  ts += 3ULL * 60ULL * 60ULL * 1000ULL;
  AddUnrelatedTurns (*ctx, nullptr, ts);

  const auto packet = ctx->ProcessTextAt ("orchid conservatory tickets",
                                          "query/public", ts + 1000,
                                          cortext::Retention::Ephemeral);
  const auto *anchor = FindMemoryWithText (packet.retrieved_memory,
                                           "book tickets for the orchid");
  REQUIRE (anchor != nullptr);
  bool clipped_window = false;
  for (const auto &blob : anchor->content)
    {
      const std::string s (blob.begin (), blob.end ());
      if (s.find ("opens at nine") != std::string::npos && s.size () <= 480)
        clipped_window = s.rfind ("Sure.", 0) == 0; // head kept too
    }
  REQUIRE (clipped_window);

  // Ingest-time hydration skips the windowing scan and keeps the head.
  // Short turns on the same topic fill the anchor slots ahead of the long
  // reply, so the reply comes back attached (clipped) to its question.
  for (int i = 0; i < 12; ++i)
    {
      ts += 3ULL * 60ULL * 60ULL * 1000ULL;
      (void)ctx->ProcessTextAt ("Orchid conservatory tickets note "
                                    + std::to_string (i) + ".",
                                "notes/" + std::to_string (i) + "/0", ts,
                                cortext::Retention::Durable);
    }
  ts += 3ULL * 60ULL * 60ULL * 1000ULL;
  const auto ingest = ctx->ProcessTextAt (
      "Can you book tickets for the orchid conservatory?", "chat/other/1", ts,
      cortext::Retention::Durable);
  std::size_t head_clips = 0, window_clips = 0;
  for (const auto &m : ingest.retrieved_memory)
    for (const auto &blob : m.content)
      {
        const std::string s (blob.begin (), blob.end ());
        if (s.rfind ("Sure.", 0) != 0 || s.size () > 600) // clipped only
          continue;
        if (s.find ("opens at nine") == std::string::npos)
          ++head_clips;
        else
          ++window_clips;
      }
  REQUIRE (head_clips >= 1);
  REQUIRE (window_clips == 0);
}

TEST_CASE ("A recalled multi-signal memory keeps the members the query is about",
           "[cortext][retrieval][retention][focus][aist]")
{
  cortext::testing::ScopedEnvVar disable_natural_boundary (
      "CORTEXT_BOUNDARY_DISABLE_NATURAL", "1");
  cortext::testing::ScopedEnvVar disable_pressure_boundary (
      "CORTEXT_BOUNDARY_DISABLE_PRESSURE", "1");
  cortext::testing::ScopedEnvVar disable_surprisal_boundary (
      "CORTEXT_BOUNDARY_DISABLE_SURPRISAL", "1");
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  auto store = std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (temp_db.path ().c_str ()));
  auto ctx = cortext::Cortext::Create (cfg, store);
  REQUIRE (ctx != nullptr);

  // Natural units of long turns on different subjects (the engine groups
  // them; every unit here holds more than the focus budget).
  const std::vector<std::string> topics = {
    "My sourdough starter needs feeding twice a day with rye flour. ",
    "The quarterly tax filing for the bakery is due on the fifteenth. ",
    "Our cat Miso hates the vacuum and hides under the porch. ",
    "The marathon training plan has a long run every Sunday morning. ",
    "The bathroom tiles are cracked and the plumber comes on Tuesday. ",
    "The garden needs tomato cages before the heat wave in July. ",
  };
  std::uint64_t ts = 1000;
  for (const auto &topic : topics)
    {
      std::string turn;
      while (turn.size () < 1600)
        turn += topic;
      (void)ctx->ProcessTextAt (turn, "chat/unit", ts,
                                cortext::Retention::Natural);
      ts += 60000;
    }
  ctx->Flush ();
  const auto units = store->Execute (
      "SELECT n_signals FROM memories WHERE kind = 'LONG_TERM' "
      "AND n_signals >= 2");
  REQUIRE (!units.empty ());

  const auto joined = [] (const cortext::Cortext::Context::Memory &m) {
    std::string s;
    for (const auto &blob : m.content)
      s.append (blob.begin (), blob.end ());
    return s;
  };
  const auto with_marathon
      = [&] (const std::vector<cortext::Cortext::Context::Memory> &ms) {
          for (const auto &m : ms)
            if (joined (m).find ("marathon training plan")
                != std::string::npos)
              return joined (m);
          return std::string ();
        };
  const std::string query
      = "When is the marathon long run in my training plan?";
  const auto packet = ctx->ProcessTextAt (query, "query/public", ts + 1000,
                                          cortext::Retention::Ephemeral);
  const std::string focused = with_marathon (packet.retrieved_memory);
  REQUIRE (!focused.empty ());
  REQUIRE (focused.size () <= 3072);
  for (std::size_t i = 0; i < topics.size (); ++i)
    if (i != 3)
      REQUIRE (focused.find (topics[i]) == std::string::npos);

  // Ingest-time hydration returns the stored memory whole.
  const auto ingest = ctx->ProcessTextAt (query, "chat/other", ts + 2000,
                                          cortext::Retention::Durable);
  REQUIRE (with_marathon (ingest.retrieved_memory).size () > 3072);
}

TEST_CASE ("A query finds a grouped memory by its best member",
           "[cortext][retrieval][retention][focus][aist]")
{
  cortext::testing::ScopedEnvVar disable_natural_boundary (
      "CORTEXT_BOUNDARY_DISABLE_NATURAL", "1");
  cortext::testing::ScopedEnvVar disable_pressure_boundary (
      "CORTEXT_BOUNDARY_DISABLE_PRESSURE", "1");
  cortext::testing::ScopedEnvVar disable_surprisal_boundary (
      "CORTEXT_BOUNDARY_DISABLE_SURPRISAL", "1");
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  auto store = std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (temp_db.path ().c_str ()));
  auto ctx = cortext::Cortext::Create (cfg, store);
  REQUIRE (ctx != nullptr);
  std::uint64_t ts = 1000;
  // One grouped memory: the queried member among four unrelated ones, so
  // its pooled vector sits far from the query.
  const std::vector<std::string> topics = {
    "My sourdough starter needs feeding twice a day with rye flour. ",
    "The quarterly tax filing for the bakery is due on the fifteenth. ",
    "The orchid conservatory renewal costs ninety dollars a year. ",
    "Our cat Miso hates the vacuum and hides under the porch. ",
    "The bathroom tiles are cracked and the plumber comes on Tuesday. ",
  };
  for (const auto &topic : topics)
    {
      std::string turn;
      while (turn.size () < 600)
        turn += topic;
      (void)ctx->ProcessTextAt (turn, "chat/unit", ts,
                                cortext::Retention::Natural);
      ts += 60000;
    }
  ctx->Flush ();
  // Single turns near the query's subject fill the anchor slots.
  for (int i = 0; i < 40; ++i)
    {
      ts += 3ULL * 60ULL * 60ULL * 1000ULL;
      (void)ctx->ProcessTextAt (
          "Greenhouse visit " + std::to_string (i)
              + ": the botanical garden has ferns, cacti and a koi pond.",
          "chat/garden/" + std::to_string (i), ts,
          cortext::Retention::Durable);
    }
  const auto packet = ctx->ProcessTextAt (
      "What do I pay annually for the flower greenhouse pass?",
      "query/public", ts + 1000, cortext::Retention::Ephemeral);
  bool found = false;
  for (const auto &m : packet.retrieved_memory)
    for (const auto &blob : m.content)
      found = found
              || std::string (blob.begin (), blob.end ())
                         .find ("orchid conservatory renewal")
                     != std::string::npos;
  REQUIRE (found);
  REQUIRE (packet.output.operation_ms.at ("GraphRetrieve.member_index_size")
           >= 5.0);
}

TEST_CASE ("Contiguity attaches adjacent turns of any modality",
           "[cortext][retrieval][contiguity][media]")
{
  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  cfg.signal_filter_image_enabled = false;
  auto store = std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (temp_db.path ().c_str ()));
  auto clock = std::make_shared<cortext::FixedClock> (1000);
  auto ctx = cortext::Cortext::Create (cfg, store, clock);
  std::uint64_t ts = 1000;
  AddUnrelatedTurns (*ctx, clock.get (), ts);
  ts += 3ULL * 60ULL * 60ULL * 1000ULL;
  clock->SetNowMillis (ts);
  const auto text_turn = ctx->ProcessTextAt (
      "Here is the receipt from the hardware store run.", "chat/errand/1", ts,
      cortext::Retention::Durable);
  REQUIRE (text_turn.output.stored_memory_id.has_value ());
  // The next turn is a photo alone.
  clock->SetNowMillis (ts + 60000);
  std::vector<std::uint8_t> pixels (8 * 8 * 3, 90);
  cortext::Cortext::Media photo;
  photo.data = kFakeJpeg.data ();
  photo.size = kFakeJpeg.size ();
  photo.mimetype = "image/jpeg";
  const auto image_turn = ctx->ProcessImage (
      pixels.data (), 8, 8, 3, "chat/errand/2", photo,
      cortext::Retention::Durable);
  REQUIRE (image_turn.output.stored_memory_id.has_value ());
  ts += 3ULL * 60ULL * 60ULL * 1000ULL;
  AddUnrelatedTurns (*ctx, clock.get (), ts);
  clock->SetNowMillis (ts + 1000);

  const auto packet = ctx->ProcessTextAt ("receipt from the hardware store",
                                          "query/public", ts + 1000,
                                          cortext::Retention::Ephemeral);
  const auto *anchor = FindMemoryWithText (packet.retrieved_memory,
                                           "receipt from the hardware store");
  REQUIRE (anchor != nullptr);
  bool photo_in_packet = false;
  for (const auto &memory : packet.retrieved_memory)
    for (const auto &blob : memory.content)
      photo_in_packet = photo_in_packet
                        || blob == std::vector<unsigned char> (
                               kFakeJpeg.begin (), kFakeJpeg.end ());
  REQUIRE (photo_in_packet);
  // One anchor for the two turns: the photo rides along with the text turn
  // (or the text with the photo), whichever ranks first.
  std::size_t errand_anchors = 0;
  for (const auto &memory : packet.retrieved_memory)
    if (memory.id == *image_turn.output.stored_memory_id
        || memory.id == *text_turn.output.stored_memory_id)
      ++errand_anchors;
  REQUIRE (errand_anchors == 1);
}
