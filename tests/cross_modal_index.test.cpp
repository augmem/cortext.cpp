#include "../src/operations/cross_modal_index_internal.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cortext/core/utils.hpp>
#include <cortext/processor/processor_context.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <any>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace
{
namespace xm = cortext::operations::cross_modal_index_internal;
constexpr int kDim = 8;

std::vector<float>
Vec (int seed)
{
  std::vector<float> v (kDim);
  for (int i = 0; i < kDim; ++i)
    v[static_cast<std::size_t> (i)]
        = static_cast<float> (std::sin (0.7 * seed + 1.3 * i) + 0.01 * seed);
  return v;
}

std::vector<unsigned char>
Blob (const std::string &name)
{
  return std::vector<unsigned char> (name.begin (), name.end ());
}

// One memory of `kind` whose signals point at the named payload vectors.
void
AddMemory (cortext::Store &store, long long memory_id, const std::string &kind,
           const std::vector<std::pair<std::string, std::string>> &signals)
{
  store.Execute ("INSERT INTO memories (memory_id, source_id, kind, start_ts, "
                 "end_ts, n_signals, modality, s_max, s_avg, strength, "
                 "created_at) VALUES (?, 's', ?, 0, 1, 1, 'mixed', 0.5, 0.5, "
                 "1.0, 0)",
                 { memory_id, kind });
  for (const auto &[blob, modality] : signals)
    store.Execute ("INSERT INTO signals (memory_id, embedding_id, source_id, "
                   "timestamp, modality, mime, blob_id, serial_position, "
                   "created_at) VALUES (?, 0, 's', 1, ?, 'x', ?, 0, 1)",
                   { memory_id, modality, Blob (blob) });
}

void
AddVector (cortext::Store &store, const std::string &blob,
           const std::string &modality, int seed)
{
  store.Execute ("INSERT OR IGNORE INTO memory_signal_vectors "
                 "(blob_id, modality, embedding) VALUES (?, ?, ?)",
                 { Blob (blob), modality, Vec (seed) });
}

// What the former per-query SQL path produced, in signal-id order.
std::vector<std::pair<long long, double>>
SqlTruth (cortext::Store &store, const std::string &modality,
          const Eigen::VectorXf &query)
{
  std::vector<std::pair<long long, double>> out;
  Eigen::VectorXf vec;
  for (const auto &row : store.Execute (
           "SELECT s.memory_id AS memory_id, v.embedding AS embedding "
           "FROM memory_signal_vectors v "
           "JOIN signals s ON s.blob_id = v.blob_id "
           "JOIN memories m ON m.memory_id = s.memory_id "
           "WHERE v.modality = ? AND m.kind = 'LONG_TERM' "
           "ORDER BY s.signal_id",
           { modality }))
    {
      REQUIRE (cortext::core::DecodeFloatBlob (row.at ("embedding"), kDim,
                                               vec));
      const double denom = static_cast<double> (query.norm () * vec.norm ());
      out.emplace_back (std::any_cast<long long> (row.at ("memory_id")),
                        denom <= 1e-12 ? 0.0
                                       : static_cast<double> (query.dot (vec))
                                             / denom);
    }
  return out;
}

std::vector<std::pair<long long, double>>
Indexed (const cortext::ProcessorContext &ctx, cortext::Store &store,
         const std::string &modality, const Eigen::VectorXf &query)
{
  auto tx = store.Begin ();
  std::vector<std::pair<long long, double>> out;
  xm::ScoreModality (ctx, *tx, modality, query, out);
  tx->Commit ();
  return out;
}

void
RequireBitwiseEqual (const std::vector<std::pair<long long, double>> &a,
                     const std::vector<std::pair<long long, double>> &b)
{
  REQUIRE (a.size () == b.size ());
  for (std::size_t i = 0; i < a.size (); ++i)
    {
      REQUIRE (a[i].first == b[i].first);
      REQUIRE (std::memcmp (&a[i].second, &b[i].second, sizeof (double)) == 0);
    }
}
} // namespace

TEST_CASE ("Cross-modal index scores exactly the LONG_TERM signal population",
           "[graph_retrieval][cross_modal]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  cortext::store::ApplyMigrations (*store);
  cortext::ProcessorContext ctx;
  // The index is keyed by context address, like the processor's other
  // registries; SignalProcessor teardown erases it, so does this test.
  struct EraseOnExit
  {
    const cortext::ProcessorContext &ctx;
    ~EraseOnExit () { xm::Erase (ctx); }
  } erase_on_exit{ ctx };
  const Eigen::VectorXf query
      = Eigen::Map<const Eigen::VectorXf> (Vec (100).data (), kDim);

  // Images of long-term memories, a shared payload (two signals, one row
  // each), a text signal (other modality) and a working memory's image.
  for (int i = 0; i < 6; ++i)
    AddVector (*store, "img" + std::to_string (i), "image", i);
  AddVector (*store, "txt0", "text", 50);
  AddMemory (*store, 1, "LONG_TERM", { { "img0", "image" }, { "txt0", "text" } });
  AddMemory (*store, 2, "LONG_TERM", { { "img1", "image" } });
  AddMemory (*store, 3, "WORKING", { { "img2", "image" } });
  AddMemory (*store, 4, "LONG_TERM", { { "img1", "image" }, { "img3", "image" } });
  RequireBitwiseEqual (Indexed (ctx, *store, "image", query),
                       SqlTruth (*store, "image", query));
  REQUIRE (xm::ResidentRows (ctx, "image") == 4);
  REQUIRE (xm::ResidentRows (ctx, "text") == 0);  // never asked for

  SECTION ("new rows are absorbed incrementally, working tails skipped")
  {
    AddMemory (*store, 5, "WORKING", { { "img4", "image" } });
    AddMemory (*store, 6, "LONG_TERM", { { "img5", "image" } });
    RequireBitwiseEqual (Indexed (ctx, *store, "image", query),
                         SqlTruth (*store, "image", query));
    REQUIRE (xm::ResidentRows (ctx, "image") == 5);
    RequireBitwiseEqual (Indexed (ctx, *store, "text", query),
                         SqlTruth (*store, "text", query));
    REQUIRE (xm::ResidentRows (ctx, "text") == 1);
  }

  SECTION ("rows absorbed inside a rolled-back transaction are dropped")
  {
    {
      auto tx = store->Begin ();
      tx->Execute ("INSERT INTO memories (memory_id, source_id, kind, "
                   "start_ts, end_ts, n_signals, modality, s_max, s_avg, "
                   "strength, created_at) VALUES (7, 's', 'LONG_TERM', 0, 1, "
                   "1, 'image', 0.5, 0.5, 1.0, 0)");
      tx->Execute ("INSERT INTO signals (memory_id, embedding_id, source_id, "
                   "timestamp, modality, mime, blob_id, serial_position, "
                   "created_at) VALUES (7, 0, 's', 1, 'image', 'x', ?, 0, 1)",
                   { Blob ("img5") });
      std::vector<std::pair<long long, double>> inside;
      xm::ScoreModality (ctx, *tx, "image", query, inside);
      REQUIRE (inside.size () == 5);
      tx->Rollback ();
    }
    // The same signal id now belongs to a different memory.
    AddMemory (*store, 8, "LONG_TERM", { { "img4", "image" } });
    RequireBitwiseEqual (Indexed (ctx, *store, "image", query),
                         SqlTruth (*store, "image", query));
  }

  SECTION ("Erase drops the resident populations")
  {
    xm::Erase (ctx);
    REQUIRE (xm::ResidentRows (ctx, "image") == 0);
    RequireBitwiseEqual (Indexed (ctx, *store, "image", query),
                         SqlTruth (*store, "image", query));
  }
}
