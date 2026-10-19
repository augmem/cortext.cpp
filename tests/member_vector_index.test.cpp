#include "../src/operations/member_vector_index_internal.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <cmath>
#include <string>
#include <vector>

namespace
{
namespace members = cortext::operations::member_vector_index_internal;
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

// One memory of `kind` with `n` signals, each with its own payload vector.
// Returns the highest signal id written.
long long
AddMemory (cortext::Store &store, long long memory_id, const std::string &kind,
           int n)
{
  store.Execute ("INSERT INTO memories (memory_id, source_id, kind, start_ts, "
                 "end_ts, n_signals, modality, s_max, s_avg, strength, "
                 "created_at) VALUES (?, 's', ?, 0, 1, ?, 'text', 0.5, 0.5, "
                 "1.0, 0)",
                 { memory_id, kind, static_cast<long long> (n) });
  for (int i = 0; i < n; ++i)
    {
      const std::string name
          = "m" + std::to_string (memory_id) + "_" + std::to_string (i);
      const std::vector<unsigned char> blob (name.begin (), name.end ());
      store.Execute ("INSERT INTO memory_signal_vectors "
                     "(blob_id, modality, embedding) VALUES (?, 'text', ?)",
                     { blob, Vec (static_cast<int> (memory_id) * 10 + i) });
      store.Execute ("INSERT INTO signals (memory_id, embedding_id, source_id, "
                     "timestamp, modality, mime, blob_id, serial_position, "
                     "created_at) VALUES (?, 0, 's', 1, 'text', 'x', ?, ?, 1)",
                     { memory_id, blob, static_cast<long long> (i) });
    }
  const auto rows = store.Execute ("SELECT MAX(signal_id) AS id FROM signals");
  return std::any_cast<long long> (rows.front ().at ("id"));
}

std::size_t
Refresh (cortext::internal::MemberVectorIndex &index, cortext::Store &store)
{
  auto tx = store.Begin ();
  const auto added = members::Refresh (index, *tx, kDim);
  tx->Commit ();
  return added;
}

} // namespace

TEST_CASE ("Member index refresh moves past long-term rows that are not "
           "members",
           "[graph_retrieval][member_index]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  cortext::store::ApplyMigrations (*store);
  cortext::internal::MemberVectorIndex index;

  // A store of single-signal memories (Durable ingest) has no members, but a
  // refresh still reads every one of its rows once; the next refresh must
  // start above them instead of rereading the whole store on every query.
  long long top = 0;
  for (long long id = 1; id <= 5; ++id)
    top = AddMemory (*store, id, "LONG_TERM", 1);
  REQUIRE (Refresh (index, *store) == 0);
  REQUIRE (index.signal_ids.empty ());
  REQUIRE (index.read_through == top);

  // Working rows above the mark never become members (a closed unit is
  // written as new LONG_TERM rows), so they do not move it.
  AddMemory (*store, 6, "WORKING", 2);
  REQUIRE (Refresh (index, *store) == 0);
  REQUIRE (index.read_through == top);

  // A grouped memory stored later is read once, all of its members.
  top = AddMemory (*store, 7, "LONG_TERM", 3);
  REQUIRE (Refresh (index, *store) == 3);
  REQUIRE (index.signal_ids.size () == 3);
  REQUIRE (index.read_through == top);
  REQUIRE (Refresh (index, *store) == 0);
  REQUIRE (index.signal_ids.size () == 3);
}
