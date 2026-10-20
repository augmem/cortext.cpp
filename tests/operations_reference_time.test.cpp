// Reference-time retrieval (Cortext::RetrievalTime, Signal::reference_ts):
// the temporal seed channel, its fusion with the dense seeds, and the as-of
// cutoff, at the retrieval operation with hand-placed embeddings and times.

#include "test_helpers.hpp"
#include "../src/operations/retrieval_trace_state.hpp"
#include "../src/operations/temporal_seed_internal.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>

#include <algorithm>
#include <any>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace cortext;
using cortext::operations::GraphAugmentedRetrieveCandidates;
namespace temporal = cortext::operations::temporal_seed_internal;

namespace
{
constexpr int kDim = 256;

// cos(v, e0) = cosine; residual on its own axis, so two such vectors have
// cosine c1 * c2 and never collapse into one family.
Eigen::VectorXf
TopicVec (float cosine, int axis)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kDim);
  v[0] = cosine;
  v[axis] = std::sqrt (std::max (0.0f, 1.0f - cosine * cosine));
  return v;
}

Eigen::VectorXf
Query ()
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kDim);
  v[0] = 1.0f;
  return v;
}

void
SeedEpisode (Store &store, long long memory_id, const Eigen::VectorXf &v,
             long long start_ts, long long end_ts)
{
  store.Execute ("INSERT INTO embeddings(embedding_id, embedding, created_at) "
                 "VALUES(?, ?, ?)",
                 { memory_id * 10, store::EigenToFloatVec (v), start_ts });
  store.Execute ("INSERT INTO memories(memory_id, embedding_id, source_id, "
                 "kind, start_ts, end_ts, created_at) "
                 "VALUES(?, ?, 'test', 'LONG_TERM', ?, ?, ?)",
                 { memory_id, memory_id * 10, start_ts, end_ts, end_ts });
}

// Opens retrieval with the write exclusion at a chosen time: the start of
// the still-open write unit, as the accumulator sets it.
class GateAt : public IOperation
{
public:
  explicit GateAt (std::optional<std::uint64_t> exclusion_ts)
      : exclusion_ts_ (exclusion_ts)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetShouldCheckRetrieval (true);
    ctx.SetWriteExclusionTs (exclusion_ts_.value_or (ctx.GetSignal ().timestamp));
  }

private:
  std::optional<std::uint64_t> exclusion_ts_;
};

struct Engine
{
  std::shared_ptr<Store> store;
  std::unique_ptr<SignalProcessor> processor;
};

// The processor loads the stored surface when it is built, so memories are
// seeded first.
template <typename Seed>
Engine
MakeEngine (Seed seed, std::optional<std::uint64_t> exclusion_ts = std::nullopt)
{
  Engine e;
  e.store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*e.store);
  seed (*e.store);
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  e.processor = std::make_unique<SignalProcessor> (
      cfg, e.store,
      std::make_unique<DynamicOperationSet> (
          std::make_unique<GateAt> (exclusion_ts),
          std::make_unique<GraphAugmentedRetrieveCandidates> ()));
  return e;
}

struct Recall
{
  std::vector<long long> ids;
  std::vector<double> scores;
};

// One query at `now`. A warm engine first takes one storing-retention
// signal, which installs the in-memory search cache the query then reads; a
// cold one answers from the processor surface loaded at construction.
Recall
Ask (Engine &e, std::uint64_t now,
     std::optional<std::uint64_t> reference_ts = std::nullopt,
     bool as_of = false)
{
  Signal s;
  s.embedding = Query ();
  s.timestamp = now;
  s.source_id = "query";
  s.retention = Retention::Ephemeral;
  s.reference_ts = reference_ts;
  s.reference_as_of = as_of;
  const auto out = e.processor->Process (s);
  Recall r;
  r.ids = out.candidate_memory_ids;
  for (const auto &c : operations::retrieval_trace::GetLastRankedCandidates ())
    r.scores.push_back (c.score);
  return r;
}

void
Warm (Engine &e, std::uint64_t now)
{
  Signal s;
  s.embedding = TopicVec (0.1f, 255);
  s.timestamp = now;
  s.source_id = "warm";
  s.retention = Retention::Natural;
  (void)e.processor->Process (s);
}

bool
Has (const Recall &r, long long id)
{
  return std::find (r.ids.begin (), r.ids.end (), id) != r.ids.end ();
}

constexpr std::uint64_t kNow = 50'000'000;
} // namespace

TEST_CASE ("Reference time inside an old episode recalls its weak memory",
           "[operations][graph][retrieval][reference_time]")
{
  const bool warm = GENERATE (false, true);
  constexpr long long kWeak = 7;
  auto e = MakeEngine ([] (Store &store) {
    // Twenty-five recent memories that match the query well, and one older
    // episode whose memory matches it weakly: it is a dense seed, but ranked
    // below the output cut.
    for (int i = 0; i < 25; ++i)
      SeedEpisode (store, 100 + i, TopicVec (0.95f - 0.004f * i, 2 + i),
                   40'000'000 + i * 10'000, 40'000'000 + i * 10'000 + 5'000);
    SeedEpisode (store, kWeak, TopicVec (0.55f, 200), 1'000'000, 2'000'000);
  });
  if (warm)
    Warm (e, kNow - 1);

  const auto without = Ask (e, kNow);
  REQUIRE_FALSE (without.ids.empty ());
  CHECK_FALSE (Has (without, kWeak));

  // T inside the episode [1'000'000, 2'000'000].
  const auto with = Ask (e, kNow, 1'500'000);
  CHECK (Has (with, kWeak));
  // The best dense match stays retrieved.
  CHECK (Has (with, 100));
}

TEST_CASE ("Reference time sets no implicit cutoff",
           "[operations][graph][retrieval][reference_time]")
{
  const bool warm = GENERATE (false, true);
  constexpr long long kBefore = 1;
  constexpr long long kAfter = 2;
  auto e = MakeEngine ([] (Store &store) {
    SeedEpisode (store, kBefore, TopicVec (0.97f, 2), 10'000'000, 10'001'000);
    SeedEpisode (store, kAfter, TopicVec (0.96f, 3), 30'000'000, 30'001'000);
  });
  if (warm)
    Warm (e, kNow - 1);

  // T between the two memories: the one after T stays retrievable.
  const auto between = Ask (e, kNow, 20'000'000);
  CHECK (Has (between, kBefore));
  CHECK (Has (between, kAfter));
  // T after now: nothing is held back by the engine clock either.
  const auto future = Ask (e, kNow, kNow + 86'400'000ULL * 365ULL);
  CHECK (Has (future, kBefore));
  CHECK (Has (future, kAfter));
}

TEST_CASE ("As-of reference time hides memories stored after it",
           "[operations][graph][retrieval][reference_time]")
{
  const bool warm = GENERATE (false, true);
  constexpr long long kBefore = 1;
  constexpr long long kHolding = 2;
  constexpr long long kAfter = 3;
  auto e = MakeEngine ([] (Store &store) {
    SeedEpisode (store, kBefore, TopicVec (0.90f, 2), 10'000'000, 10'001'000);
    // Starts at T exactly: stored by T, so it stays.
    SeedEpisode (store, kHolding, TopicVec (0.80f, 3), 20'000'000,
                 20'500'000);
    // The best match, stored after T.
    SeedEpisode (store, kAfter, TopicVec (0.99f, 4), 30'000'000, 30'001'000);
  });
  if (warm)
    Warm (e, kNow - 1);

  const auto open = Ask (e, kNow, 20'000'000, false);
  CHECK (Has (open, kAfter));

  const auto as_of = Ask (e, kNow, 20'000'000, true);
  CHECK_FALSE (Has (as_of, kAfter));
  CHECK (Has (as_of, kBefore));
  CHECK (Has (as_of, kHolding));
  for (const auto &c : operations::retrieval_trace::GetLastSeedCandidates ())
    CHECK (c.memory_id != kAfter);
}

TEST_CASE ("Reference time never retrieves the open write unit",
           "[operations][graph][retrieval][reference_time]")
{
  const bool warm = GENERATE (false, true);
  const bool as_of = GENERATE (false, true);
  // The open unit started at kOpen; retrieval excludes from there on.
  constexpr std::uint64_t kOpen = 40'000'000;
  constexpr long long kOld = 1;
  constexpr long long kUnit = 2;
  auto e = MakeEngine (
      [] (Store &store) {
        SeedEpisode (store, kOld, TopicVec (0.90f, 2), 10'000'000,
                     10'001'000);
        SeedEpisode (store, kUnit, TopicVec (0.99f, 3), kOpen, kOpen + 1'000);
      },
      kOpen);
  if (warm)
    Warm (e, kNow - 1);

  for (const std::optional<std::uint64_t> t :
       { std::optional<std::uint64_t> (), std::optional<std::uint64_t> (kOpen),
         std::optional<std::uint64_t> (kOpen + 500),
         std::optional<std::uint64_t> (kNow) })
    {
      const auto r = Ask (e, kNow, t, as_of);
      CHECK (Has (r, kOld));
      CHECK_FALSE (Has (r, kUnit));
    }
}

TEST_CASE ("No reference time leaves retrieval unchanged",
           "[operations][graph][retrieval][reference_time]")
{
  const bool warm = GENERATE (false, true);
  auto seed = [] (Store &store) {
    for (int i = 0; i < 40; ++i)
      SeedEpisode (store, 100 + i, TopicVec (0.95f - 0.01f * i, 2 + i),
                   1'000'000 + i * 100'000, 1'000'000 + i * 100'000 + 5'000);
  };
  auto plain = MakeEngine (seed);
  auto used = MakeEngine (seed);
  if (warm)
    {
      Warm (plain, kNow - 1);
      Warm (used, kNow - 1);
    }
  // A reference-time query leaves nothing behind for the next query.
  (void)Ask (used, kNow, 1'000'000, true);
  (void)Ask (used, kNow, 3'000'000, false);
  const auto a = Ask (plain, kNow);
  const auto b = Ask (used, kNow);
  CHECK (a.ids == b.ids);
  CHECK (a.scores == b.scores);
  // Without a reference time the ranking is the dense ranking: best match
  // first, and the time of a memory plays no part.
  REQUIRE_FALSE (a.ids.empty ());
  CHECK (a.ids.front () == 100);
}

TEST_CASE ("Temporal channel ranks by interval distance from bounded scans",
           "[operations][graph][retrieval][reference_time]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  // Episodes [1000k, 1100k), one per 1000k.
  for (int i = 0; i < 50; ++i)
    SeedEpisode (*store, 1 + i, TopicVec (0.5f, 2 + i),
                 (1 + i) * 1'000'000LL, (1 + i) * 1'000'000LL + 100'000);
  auto tx = store->Begin ();

  // Inside episode 10 ([10'000'000, 10'100'000]): it is at distance 0, then
  // the end of episode 9 and the start of episode 11, both 950k away (ties
  // by memory id).
  auto hits = temporal::Nearest (*tx, 10'050'000, 1'000'000'000, 4);
  REQUIRE (hits.size () >= 3);
  CHECK (hits[0].memory_id == 10);
  CHECK (hits[0].distance == 0);
  CHECK (hits[1].memory_id == 9);
  CHECK (hits[1].distance == 950'000);
  CHECK (hits[2].memory_id == 11);
  CHECK (hits[2].distance == 950'000);
  // Each scan reads at most its limit: three scans of 4 rows.
  CHECK (hits.size () <= 12);

  // Exclusion bounds the start: nothing at or after it.
  hits = temporal::Nearest (*tx, 10'050'000, 10'000'000, 4);
  for (const auto &hit : hits)
    CHECK (hit.start_ts < 10'000'000);
  REQUIRE_FALSE (hits.empty ());
  CHECK (hits[0].memory_id == 9);

  CHECK (temporal::Distance (100, 200, 150) == 0);
  CHECK (temporal::Distance (100, 200, 50) == 50);
  CHECK (temporal::Distance (100, 200, 260) == 60);
  CHECK (temporal::Distance (100, 0, 130) == 30);

  // Every arm reads an index range; none scans the table.
  const auto plan = tx->Execute (
      "EXPLAIN QUERY PLAN " + std::string (temporal::NearestSql ()),
      { 10'050'000LL, 1'000'000'000LL, 4LL });
  int index_arms = 0;
  for (const auto &row : plan)
    {
      const auto detail = row.find ("detail");
      REQUIRE (detail != row.end ());
      const auto text = std::any_cast<std::string> (detail->second);
      CHECK (text.find ("SCAN memories") == std::string::npos);
      if (text.find ("USING INDEX idx_memories_ltm_") != std::string::npos)
        ++index_arms;
    }
  CHECK (index_arms == 3);
  tx->Rollback ();
}
