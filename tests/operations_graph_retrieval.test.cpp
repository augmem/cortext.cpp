#include "test_helpers.hpp"
#include "../src/operations/constructive_recall_internal.hpp"
#include "../src/operations/historical_surface_search_cache_internal.hpp"
#include "../src/operations/retrieval_trace_state.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cortext/core/utils.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace cortext;
using cortext::operations::GraphAugmentedRetrieveCandidates;

namespace
{
constexpr int kEmbeddingDim = 256;

Eigen::VectorXf
UnitVec (int dim)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[dim] = 1.0f;
  return v;
}

Eigen::VectorXf
VectorWithCosineToDim0 (float cosine)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = cosine;
  v[1] = std::sqrt (std::max (0.0f, 1.0f - cosine * cosine));
  return v;
}

Eigen::VectorXf
VectorWithCosineAndDiverseResidual (float cosine, std::uint64_t seed)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = cosine;
  const float residual_scale
      = std::sqrt (std::max (0.0f, 1.0f - cosine * cosine))
        / std::sqrt (static_cast<float> (kEmbeddingDim - 1));
  std::uint64_t state = seed + 0x9e3779b97f4a7c15ULL;
  for (int dimension = 1; dimension < kEmbeddingDim; ++dimension)
    {
      state += 0x9e3779b97f4a7c15ULL;
      std::uint64_t mixed = state;
      mixed = (mixed ^ (mixed >> 30U)) * 0xbf58476d1ce4e5b9ULL;
      mixed = (mixed ^ (mixed >> 27U)) * 0x94d049bb133111ebULL;
      mixed ^= mixed >> 31U;
      v[dimension] = (mixed & 1ULL) != 0 ? residual_scale
                                         : -residual_scale;
    }
  return v;
}

Eigen::VectorXf
HadamardRow (unsigned int row)
{
  Eigen::VectorXf v (kEmbeddingDim);
  const float scale = 1.0f / std::sqrt (static_cast<float> (kEmbeddingDim));
  for (unsigned int column = 0;
       column < static_cast<unsigned int> (kEmbeddingDim); ++column)
    {
      v[static_cast<Eigen::Index> (column)]
          = (std::popcount (row & column) % 2 == 0) ? scale : -scale;
    }
  return v;
}

std::pair<Eigen::VectorXf, Eigen::VectorXf>
NearDuplicateFamilyPair ()
{
  Eigen::VectorXf first = Eigen::VectorXf::Zero (kEmbeddingDim);
  Eigen::VectorXf second = Eigen::VectorXf::Zero (kEmbeddingDim);
  for (int dimension = 0; dimension < 7; ++dimension)
    {
      first[dimension] = 1.0f;
      second[dimension] = 1.0f;
    }
  first[7] = 1.0f;
  first[8] = 0.9999f;
  second[7] = 0.9999f;
  second[8] = 1.0f;
  first.normalize ();
  second.normalize ();
  return { first, second };
}

Eigen::VectorXf
VectorWithCosineToQuery (const Eigen::VectorXf &query, float cosine)
{
  Eigen::VectorXf orthogonal = UnitVec (kEmbeddingDim - 1);
  Eigen::VectorXf result
      = cosine * query
        + std::sqrt (std::max (0.0f, 1.0f - cosine * cosine)) * orthogonal;
  result.normalize ();
  return result;
}

std::vector<float>
ToFloatVec (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

Signal
MakeSignal (const Eigen::VectorXf &embedding, std::uint64_t ts)
{
  Signal s;
  s.embedding = embedding;
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

class ForceRetrievalGateOp : public IOperation
{
public:
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetShouldCheckRetrieval (true);
    ctx.SetWriteExclusionTs (ctx.GetSignal ().timestamp);
  }
};

class CaptureReloadedRetrievalSurfaceOp : public IOperation
{
public:
  explicit CaptureReloadedRetrievalSurfaceOp (bool &found) : found_ (found) {}

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &p_ctx = ctx.GetProcessorContext ();
    auto anchor_it = p_ctx.retrieval_surface_index.find (10);
    auto neighbor_it = p_ctx.retrieval_surface_index.find (20);
    if (anchor_it == p_ctx.retrieval_surface_index.end ()
        || neighbor_it == p_ctx.retrieval_surface_index.end ())
      {
        found_ = false;
        return;
      }
    const auto &anchor = p_ctx.retrieval_surface_cache[anchor_it->second];
    const auto &neighbor = p_ctx.retrieval_surface_cache[neighbor_it->second];
    auto source_it
        = p_ctx.retrieval_surface_source_index.find ("conversation/reload");
    found_ = anchor.embedding_id == 100 && neighbor.embedding_id == 200
             && anchor.embedding.size () == kEmbeddingDim
             && neighbor.embedding.size () == kEmbeddingDim
             && source_it != p_ctx.retrieval_surface_source_index.end ()
             && source_it->second.size () == 2;
  }

private:
  bool &found_;
};

void
SeedMemory (Store &store, long long memory_id, long long embedding_id,
            const Eigen::VectorXf &embedding, std::uint64_t ts,
            const std::string &source_id = "test")
{
  store.Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { embedding_id, ToFloatVec (embedding), static_cast<long long> (ts) });
  store.Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, start_ts, "
      "created_at) VALUES(?, ?, ?, 'LONG_TERM', ?, ?)",
      { memory_id, embedding_id, source_id, static_cast<long long> (ts),
        static_cast<long long> (ts) });
}
} // namespace

TEST_CASE ("Graph retrieval returns nearest retained memory",
           "[operations][graph][retrieval]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  SeedMemory (*store, 20, 200, UnitVec (1), 1000);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
  const auto selected
      = operations::retrieval_trace::GetLastSelectedEmbeddingOrder ();
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 10)
           != out.candidate_memory_ids.end ());
  REQUIRE_FALSE (selected.empty ());
  auto current_rows = store->Execute (
      "SELECT embedding_id FROM current_memory_embeddings "
      "WHERE memory_id = ?",
      { 10LL });
  REQUIRE (current_rows.size () == 1);
  REQUIRE (selected.front ()
           == std::any_cast<long long> (current_rows[0].at ("embedding_id")));
}

TEST_CASE ("Graph retrieval expands through retained associations",
           "[operations][graph][retrieval]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  SeedMemory (*store, 20, 200, UnitVec (1), 1000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'co_occurs', ?, ?)",
      { 10LL, 20LL, 0.95, 1000LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 20)
           != out.candidate_memory_ids.end ());
}

TEST_CASE ("Graph retrieval reaches derived associations without direct seeding",
           "[operations][graph][retrieval][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (1), 1000);
  SeedMemory (*store, 20, 200, UnitVec (0), 1000);
  store->Execute (
      "UPDATE memories SET kind = 'ASSOCIATION' WHERE memory_id = ?",
      { 20LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto make_ops = [] {
    return std::make_unique<DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
  };

  {
    SignalProcessor processor (cfg, store, make_ops ());
    const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
    REQUIRE (std::find (out.candidate_memory_ids.begin (),
                        out.candidate_memory_ids.end (), 20)
             == out.candidate_memory_ids.end ());
  }

  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'derived_from', ?, ?)",
      { 20LL, 10LL, 0.95, 1000LL });
  {
    SignalProcessor processor (cfg, store, make_ops ());
    const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
    REQUIRE (std::find (out.candidate_memory_ids.begin (),
                        out.candidate_memory_ids.end (), 20)
             != out.candidate_memory_ids.end ());
  }
}

TEST_CASE ("Graph retrieval protects direct family before derived centroid",
           "[operations][graph][retrieval][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  SeedMemory (*store, 20, 200, UnitVec (0), 1000);
  store->Execute (
      "UPDATE memories SET kind = 'ASSOCIATION' WHERE memory_id = ?",
      { 20LL });
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'derived_from', ?, ?)",
      { 20LL, 10LL, 1.0, 1000LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 10)
           != out.candidate_memory_ids.end ());
}

TEST_CASE ("Graph retrieval expands through sequential episode edges",
           "[operations][graph][retrieval]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  SeedMemory (*store, 20, 200, UnitVec (1), 1100);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'next_in_episode', ?, ?)",
      { 10LL, 20LL, 0.95, 1100LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 20)
           != out.candidate_memory_ids.end ());
}

TEST_CASE ("Graph retrieval demotes superseded stale memories",
           "[operations][graph][retrieval][supersession][eval]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  SeedMemory (*store, 20, 200, VectorWithCosineToDim0 (0.92f), 2000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'supersedes', ?, ?)",
      { 20LL, 10LL, 1.0, 2000LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 3000));
  const auto ranked = operations::retrieval_trace::GetLastRankedCandidates ();
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE_FALSE (ranked.empty ());
  REQUIRE (ranked.front ().memory_id == 20LL);

  auto correction_it = std::find_if (
      ranked.begin (), ranked.end (),
      [] (const auto &candidate) { return candidate.memory_id == 20LL; });
  REQUIRE (correction_it != ranked.end ());
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 10LL)
           == out.candidate_memory_ids.end ());
}

TEST_CASE ("Graph retrieval scores older exact matches beyond old recency cap",
           "[operations][graph][retrieval]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 1, 100, UnitVec (0), 1000);
  for (long long i = 0; i < 450; ++i)
    {
      SeedMemory (*store, 1000 + i, 10000 + i, UnitVec (1),
                  1000);
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 100000));
  const auto selected
      = operations::retrieval_trace::GetLastSelectedEmbeddingOrder ();
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE_FALSE (selected.empty ());
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 1)
           != out.candidate_memory_ids.end ());
  REQUIRE (out.candidate_memory_ids.front () == 1LL);
}

TEST_CASE ("Graph retrieval keeps historical memory before its replacement time",
           "[operations][graph][retrieval][supersession]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  SeedMemory (*store, 20, 200, VectorWithCosineToDim0 (0.92f), 5000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'supersedes', ?, ?)",
      { 20LL, 10LL, 1.0, 5000LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 3000));
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE (out.candidate_memory_ids.front () == 10LL);
}

TEST_CASE ("Graph retrieval temporal score decays across multi-month ages",
           "[operations][graph][retrieval][temporal]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  constexpr long long kDayMs = 24LL * 60LL * 60LL * 1000LL;
  constexpr std::uint64_t now = 200ULL * static_cast<std::uint64_t> (kDayMs);
  SeedMemory (*store, 10, 100, UnitVec (0), now - 60ULL * 1000ULL);
  SeedMemory (*store, 20, 200, UnitVec (1), now - 30ULL * kDayMs);
  SeedMemory (*store, 30, 300, UnitVec (2), now - 90ULL * kDayMs);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  operations::retrieval_trace::ClearLastRankedCandidates ();
  const auto out = processor.Process (MakeSignal (UnitVec (0), now));
  const auto ranked = operations::retrieval_trace::GetLastRankedCandidates ();

  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  std::unordered_map<long long, double> temporal_by_memory;
  for (const auto &candidate : ranked)
    {
      temporal_by_memory.emplace (candidate.memory_id,
                                  candidate.temporal_score);
    }

  REQUIRE (temporal_by_memory.count (10) == 1);
  REQUIRE (temporal_by_memory.count (20) == 1);
  REQUIRE (temporal_by_memory.count (30) == 1);
  REQUIRE (temporal_by_memory.at (10) <= 1.0);
  REQUIRE (temporal_by_memory.at (10) > temporal_by_memory.at (20));
  REQUIRE (temporal_by_memory.at (20) > temporal_by_memory.at (30));
  REQUIRE (temporal_by_memory.at (30) >= 0.0);
  REQUIRE (temporal_by_memory.at (30) < 0.05);
}

TEST_CASE ("Graph retrieval KNN finds older exact matches outside stride buckets",
           "[operations][graph][retrieval]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  constexpr long long target_memory_id = 504;
  constexpr long long target_embedding_id = 90504;
  for (long long i = 1; i <= 1800; ++i)
    {
      if (i == target_memory_id)
        {
          continue;
        }
      SeedMemory (*store, i, 10000 + i, UnitVec (1), 1000);
    }
  SeedMemory (*store, target_memory_id, target_embedding_id, UnitVec (0),
              1000);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 100000));
  const auto selected
      = operations::retrieval_trace::GetLastSelectedEmbeddingOrder ();
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE_FALSE (selected.empty ());
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), target_memory_id)
           != out.candidate_memory_ids.end ());
}

TEST_CASE ("Graph retrieval KNN searches current memory surface",
           "[operations][graph][retrieval]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  constexpr long long target_memory_id = 504;
  constexpr long long base_embedding_id = 90504;
  constexpr long long current_embedding_id = 99504;
  for (long long i = 1; i <= 1800; ++i)
    {
      if (i == target_memory_id)
        {
          continue;
        }
      SeedMemory (*store, i, 10000 + i, UnitVec (1), 1000);
    }
  SeedMemory (*store, target_memory_id, base_embedding_id, UnitVec (2),
              1000);
  cortext::testing::SeedEmbeddingV2 (*store, current_embedding_id,
                                     ToFloatVec (UnitVec (0)), 2000);
  cortext::testing::SeedCurrentMemoryEmbeddingV2 (*store, target_memory_id,
                                                  current_embedding_id);
  auto current_rows = store->Execute (
      "SELECT memory_id, embedding_id FROM current_memory_embeddings "
      "WHERE embedding MATCH ? AND k = ? "
      "ORDER BY distance",
      { ToFloatVec (UnitVec (0)), 5LL });
  REQUIRE (std::find_if (
               current_rows.begin (), current_rows.end (),
               [&] (const auto &row) {
                 auto it = row.find ("memory_id");
                 return it != row.end ()
                        && std::any_cast<long long> (it->second)
                               == target_memory_id;
               })
           != current_rows.end ());
  auto joined_rows = store->Execute (
      "SELECT m.memory_id, cme.embedding_id, m.start_ts, cme.embedding, "
      "       m.source_id "
      "FROM ("
      "  SELECT memory_id, embedding_id, embedding, distance "
      "  FROM current_memory_embeddings "
      "  WHERE embedding MATCH ? "
      "    AND k = ?"
      ") cme "
      "JOIN memories m ON m.memory_id = cme.memory_id "
      "WHERE m.kind IN ('LONG_TERM', 'ASSOCIATION') "
      "  AND COALESCE(m.start_ts, 0) < ? "
      "ORDER BY distance",
      { ToFloatVec (UnitVec (0)), 5LL, 100000LL });
  REQUIRE (std::find_if (
               joined_rows.begin (), joined_rows.end (),
               [&] (const auto &row) {
                 auto it = row.find ("memory_id");
                 return it != row.end ()
                        && std::any_cast<long long> (it->second)
                               == target_memory_id;
               })
           != joined_rows.end ());

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 100000));
  const auto selected
      = operations::retrieval_trace::GetLastSelectedEmbeddingOrder ();
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE_FALSE (selected.empty ());
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), target_memory_id)
           != out.candidate_memory_ids.end ());
  REQUIRE (std::find (selected.begin (), selected.end (), current_embedding_id)
           != selected.end ());
}

TEST_CASE ("Graph retrieval reloads latest reconstruction when current writes "
           "are disabled",
           "[operations][graph][retrieval][constructive_recall][restart]")
{
  cortext::testing::ScopedEnvVar disable_current (
      "CORTEXT_DISABLE_CURRENT_MEMORY_SURFACE_WRITES", "1");
  if (!operations::constructive_recall::CurrentSurfaceWritesDisabled ())
    {
      SKIP ("current-surface write hook is disabled in this build");
    }
  cortext::testing::ScopedEnvVar reconstruction_interval (
      "CORTEXT_RECONSTRUCTION_MIN_UPDATE_MS", "999999999");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (2), 1000);
  SeedMemory (*store, 20, 200, VectorWithCosineToDim0 (0.90f), 1000);
  auto reconstruction_tx = store->Begin ();
  const long long reconstruction_id
      = operations::constructive_recall::AppendReconstructionWithEmbedding (
          *reconstruction_tx, 10, UnitVec (0), {}, 1500, 0.1, "retrieval",
          1.0, 1.0);
  REQUIRE (reconstruction_id > 0);
  reconstruction_tx->Commit ();
  const auto latest_rows = store->Execute (
      "SELECT embedding_id FROM memory_reconstructions "
      "WHERE reconstruction_id = ?",
      { reconstruction_id });
  REQUIRE (latest_rows.size () == 1);
  const long long latest_embedding_id = cortext::store::AnyToLongLong (
      latest_rows[0].at ("embedding_id")).value_or (0);
  REQUIRE (latest_embedding_id > 0);
  REQUIRE (store->Execute (
               "SELECT 1 FROM current_memory_embeddings WHERE memory_id = ?",
               { 10LL })
               .empty ());

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  // Construction is the restart boundary: it reloads processor surfaces from
  // the durable base/current/reconstruction tables.
  SignalProcessor processor (cfg, store, std::move (ops));

  operations::retrieval_trace::ClearLastRankedCandidates ();
  const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
  const auto ranked = operations::retrieval_trace::GetLastRankedCandidates ();
  const auto target = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 10LL;
      });
  const auto competitor = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 20LL;
      });
  REQUIRE (target != ranked.end ());
  REQUIRE (competitor != ranked.end ());
  REQUIRE (target->embedding_id == latest_embedding_id);
  REQUIRE (target->score > competitor->score);
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE (out.candidate_memory_ids.front () == 10LL);
}

TEST_CASE ("Graph retrieval reloads base embedding when constructive recall is "
           "disabled",
           "[operations][graph][retrieval][constructive_recall][restart][ablation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 101, UnitVec (1), 1250);
  SeedMemory (*store, 20, 200, VectorWithCosineToDim0 (0.90f), 1000);
  auto reconstruction_tx = store->Begin ();
  const long long reconstruction_id
      = operations::constructive_recall::AppendReconstructionWithEmbedding (
          *reconstruction_tx, 10, UnitVec (2), {}, 1500, 0.1, "retrieval",
          1.0, 1.0);
  REQUIRE (reconstruction_id > 0);
  reconstruction_tx->Commit ();
  cortext::testing::SeedCurrentMemoryEmbeddingV2 (*store, 10, 101);

  const auto latest_rows = store->Execute (
      "SELECT embedding_id FROM memory_reconstructions "
      "WHERE reconstruction_id = ?",
      { reconstruction_id });
  REQUIRE (latest_rows.size () == 1);
  const long long latest_embedding_id = cortext::store::AnyToLongLong (
      latest_rows[0].at ("embedding_id")).value_or (0);
  REQUIRE (latest_embedding_id > 0);
  REQUIRE (latest_embedding_id != 100LL);
  REQUIRE (latest_embedding_id != 101LL);

  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  if (!operations::constructive_recall::Disabled ())
    {
      SKIP ("constructive-recall disable hook is disabled in this build");
    }
  const auto authoritative
      = operations::constructive_recall::LoadCurrentEmbedding (
          store.get (), 10, 100, kEmbeddingDim);
  REQUIRE (authoritative.has_value ());
  REQUIRE ((*authoritative - UnitVec (0)).norm () < 1e-5f);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  operations::retrieval_trace::ClearLastRankedCandidates ();
  const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
  const auto ranked = operations::retrieval_trace::GetLastRankedCandidates ();
  const auto target = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 10LL;
      });
  const auto competitor = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 20LL;
      });
  REQUIRE (target != ranked.end ());
  REQUIRE (competitor != ranked.end ());
  REQUIRE (target->embedding_id == 100LL);
  REQUIRE (target->score > competitor->score);
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE (out.candidate_memory_ids.front () == 10LL);
}

TEST_CASE ("Graph retrieval cache rebuild uses base embedding when constructive "
           "recall is disabled",
           "[operations][graph][retrieval][constructive_recall][cache_rebuild]"
           "[ablation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 101, UnitVec (1), 1250);
  auto reconstruction_tx = store->Begin ();
  const long long reconstruction_id
      = operations::constructive_recall::AppendReconstructionWithEmbedding (
          *reconstruction_tx, 10, UnitVec (2), {}, 1500, 0.1, "retrieval",
          1.0, 1.0);
  REQUIRE (reconstruction_id > 0);
  reconstruction_tx->Commit ();
  cortext::testing::SeedCurrentMemoryEmbeddingV2 (*store, 10, 101);

  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  if (!operations::constructive_recall::Disabled ())
    {
      SKIP ("constructive-recall disable hook is disabled in this build");
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  pctx.UpsertRetrievalSurface (
      { 10, 100, 1000, 1000, 0, 0, 0, 0, "LONG_TERM", "test", "", -1.0,
        0, 0.0, 0.0, 0.0, false, true, UnitVec (0) });
  auto signal = MakeSignal (UnitVec (0), 2000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);
  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  const auto ranked = operations::retrieval_trace::GetLastRankedCandidates ();
  const auto target = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 10LL;
      });
  REQUIRE (target != ranked.end ());
  REQUIRE (target->embedding_id == 100LL);
  REQUIRE (target->score > 0.99);
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval returns refreshed output after reconstruction",
           "[operations][graph][retrieval]")
{
  cortext::testing::ScopedEnvVar enable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL");
  cortext::testing::ScopedEnvVar no_reconstruction_interval (
      "CORTEXT_RECONSTRUCTION_MIN_UPDATE_MS", "0");

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = UnitVec (0);
  Eigen::VectorXf memory = VectorWithCosineToDim0 (0.20f);
  SeedMemory (*store, 10, 100, memory, 1000);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  auto signal = MakeSignal (query, 2'000'000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (2'000'000);

  GraphAugmentedRetrieveCandidates op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto recon_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memory_reconstructions "
      "WHERE memory_id = ?",
      { 10LL });
  REQUIRE (recon_rows.size () == 1);
  REQUIRE (std::any_cast<long long> (recon_rows[0].at ("cnt")) > 0);
  auto latest_rows = store->Execute (
      "SELECT mr.embedding_id, e.embedding FROM memory_reconstructions mr "
      "JOIN embeddings e ON e.embedding_id = mr.embedding_id "
      "WHERE mr.memory_id = ? "
      "ORDER BY mr.reconstruction_id DESC LIMIT 1",
      { 10LL });
  REQUIRE (latest_rows.size () == 1);
  const long long latest_embedding_id
      = std::any_cast<long long> (latest_rows[0].at ("embedding_id"));
  REQUIRE (latest_embedding_id != 100LL);
  Eigen::VectorXf latest;
  REQUIRE (cortext::core::DecodeFloatBlob (
      latest_rows[0].at ("embedding"), kEmbeddingDim, latest));
  auto current_rows = store->Execute (
      "SELECT embedding_id, embedding FROM current_memory_embeddings "
      "WHERE memory_id = ?",
      { 10LL });
  REQUIRE (current_rows.size () == 1);
  const long long current_embedding_id
      = std::any_cast<long long> (current_rows[0].at ("embedding_id"));
  REQUIRE (current_embedding_id == latest_embedding_id);
  Eigen::VectorXf current;
  REQUIRE (cortext::core::DecodeFloatBlob (
      current_rows[0].at ("embedding"), kEmbeddingDim, current));
  REQUIRE ((current - latest).norm () < 1e-6f);

  const auto &candidates = ctx.GetRetrievedMemoryCandidates ();
  REQUIRE_FALSE (candidates.empty ());
  REQUIRE (candidates.front ().memory_id == 10LL);
  REQUIRE (candidates.front ().embedding_id == current_embedding_id);
  REQUIRE ((candidates.front ().embedding - current).norm () < 1e-6f);
}

TEST_CASE ("Graph retrieval reloads base memory surfaces without current rows",
           "[operations][graph][retrieval][source]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000, "conversation/reload");
  SeedMemory (*store, 20, 200, UnitVec (1), 1100, "conversation/reload");

  bool found = false;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<CaptureReloadedRetrievalSurfaceOp> (found));
  SignalProcessor processor (cfg, store, std::move (ops));

  (void)processor.Process (MakeSignal (UnitVec (0), 2000));
  REQUIRE (found);
}

TEST_CASE ("Graph retrieval queries supersession family representatives before KNN cap",
           "[operations][graph][retrieval][family]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  constexpr long long kTargetMemoryId = 1;
  SeedMemory (*store, kTargetMemoryId, 1,
              VectorWithCosineToDim0 (0.95f), 1000, "target");
  constexpr long long kFamilyBegin = 1000;
  constexpr long long kFamilySize = 430;
  const long long representative_id = kFamilyBegin + kFamilySize - 1;
  for (long long offset = 0; offset < kFamilySize; ++offset)
    {
      const long long memory_id = kFamilyBegin + offset;
      SeedMemory (*store, memory_id, 10000 + offset,
                  VectorWithCosineToDim0 (0.99f), 2000 + offset,
                  "duplicate/" + std::to_string (offset));
      if (memory_id != representative_id)
        {
          store->Execute (
              "INSERT INTO associations "
              "(source_memory_id, target_memory_id, edge_type, weight, "
              " last_reinforced) VALUES (?, ?, 'supersedes', 1.0, 0)",
              { representative_id, memory_id });
        }
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 100000));
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), kTargetMemoryId)
           != out.candidate_memory_ids.end ());
}

TEST_CASE ("Graph retrieval ranking is invariant to source and modality labels",
           "[operations][graph][retrieval][invariance]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto run = [] (bool shared_source, const std::string &modality) {
    auto unique_store = SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<Store> (std::move (unique_store));
    cortext::testing::InitializeCoreSchema (*store);
    SeedMemory (*store, 10, 100, UnitVec (0), 1000, "anchor");
    SeedMemory (*store, 20, 200, UnitVec (2), 1100,
                shared_source ? "anchor" : "unrelated");
    store->Execute ("UPDATE memories SET modality = ?", { modality });
    for (long long i = 0; i < 20; ++i)
      {
        SeedMemory (*store, 1000 + i, 10000 + i,
                    VectorWithCosineToDim0 (0.10f + 0.01f * i), 1200 + i,
                    "distractor/" + std::to_string (i));
        store->Execute (
            "UPDATE memories SET modality = ? WHERE memory_id = ?",
            { modality, 1000 + i });
      }

    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.focus = 1.0;
    cfg.sensitivity = 1.0;
    cfg.stability = 0.5;
    auto ops = std::make_unique<DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    (void)processor.Process (MakeSignal (UnitVec (0), 100000));
    const auto ranked = operations::retrieval_trace::GetLastRankedCandidates ();
    std::vector<std::pair<long long, double>> signature;
    signature.reserve (ranked.size ());
    for (const auto &candidate : ranked)
      {
        signature.emplace_back (candidate.memory_id, candidate.score);
      }
    return signature;
  };

  const auto shared_text = run (true, "text");
  const auto unique_audio = run (false, "audio");
  REQUIRE (shared_text == unique_audio);
}

TEST_CASE ("Graph retrieval soundly collapses cosine-near families before KNN cap",
           "[operations][graph][retrieval][family]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const auto [family_first, family_second] = NearDuplicateFamilyPair ();
  REQUIRE (family_first.dot (family_second) > 0.999999f);
  constexpr long long kTargetMemoryId = 1;
  SeedMemory (*store, kTargetMemoryId, 1,
              VectorWithCosineToQuery (family_first, 0.95f), 1000,
              "target");
  for (long long offset = 0; offset < 430; ++offset)
    {
      SeedMemory (*store, 1000 + offset, 10000 + offset,
                  offset % 2 == 0 ? family_first : family_second,
                  2000 + offset,
                  "duplicate/" + std::to_string (offset));
    }
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (family_first, 100000));
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), kTargetMemoryId)
           != out.candidate_memory_ids.end ());
}

TEST_CASE ("Graph retrieval bounds exact family checks on diverse embeddings",
           "[operations][graph][retrieval][family][performance]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  for (int dimension = 0; dimension < kEmbeddingDim; ++dimension)
    {
      SeedMemory (*store, 1000 + dimension, 10000 + dimension,
                  UnitVec (dimension), 1000 + dimension,
                  "diverse/" + std::to_string (dimension));
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  (void)processor.Process (MakeSignal (UnitVec (0), 100000));
  const std::size_t exact_comparisons
      = operations::retrieval_trace::GetLastFamilyExactComparisonCount ();
  constexpr std::size_t kAllPairs
      = static_cast<std::size_t> (kEmbeddingDim)
        * static_cast<std::size_t> (kEmbeddingDim - 1) / 2;
  REQUIRE (exact_comparisons > 0);
  REQUIRE (exact_comparisons < kAllPairs / 4);
}

TEST_CASE ("Graph retrieval bounds exact family checks on dense orthogonal "
           "embeddings",
           "[operations][graph][retrieval][family][performance]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  for (int row = 0; row < kEmbeddingDim; ++row)
    {
      SeedMemory (*store, 1000 + row, 10000 + row,
                  HadamardRow (static_cast<unsigned int> (row)), 1000 + row,
                  "diverse/" + std::to_string (row));
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  (void)processor.Process (MakeSignal (HadamardRow (0), 100000));
  const std::size_t exact_comparisons
      = operations::retrieval_trace::GetLastFamilyExactComparisonCount ();
  constexpr std::size_t kAllPairs
      = static_cast<std::size_t> (kEmbeddingDim)
        * static_cast<std::size_t> (kEmbeddingDim - 1) / 2;
  REQUIRE (exact_comparisons > 0);
  REQUIRE (exact_comparisons < kAllPairs / 4);
}

TEST_CASE ("Graph retrieval SQL fallback collapses families before candidate cap",
           "[operations][graph][retrieval][family][sql]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const auto [family_first, family_second] = NearDuplicateFamilyPair ();
  constexpr long long kTargetMemoryId = 1;
  SeedMemory (*store, kTargetMemoryId, 1,
              VectorWithCosineToQuery (family_first, 0.95f), 1000,
              "target");
  for (long long offset = 0; offset < 430; ++offset)
    {
      SeedMemory (*store, 1000 + offset, 10000 + offset,
                  offset % 2 == 0 ? family_first : family_second,
                  2000 + offset,
                  "duplicate/" + std::to_string (offset));
    }
  for (long long offset = 0; offset < 900; ++offset)
    {
      SeedMemory (*store, 2000 + offset, 20000 + offset, UnitVec (31),
                  3000 + offset, "distant/" + std::to_string (offset));
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx);
  const auto recovery_state
      = operations::historical_surface_search_cache_internal::Find (pctx);
  auto signal = MakeSignal (family_first, 100000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  const auto &candidates = ctx.GetRetrievedMemoryCandidates ();
  REQUIRE (std::find_if (candidates.begin (), candidates.end (),
                         [] (const auto &candidate) {
                           return candidate.memory_id == kTargetMemoryId;
                         })
           != candidates.end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 1);
  const int seed_limit = std::max (1, core::RetrievalMaxResults (cfg.focus));
  const int seed_search_limit = core::RetrievalSeedSearchK (
      cfg.focus, cfg.sensitivity, cfg.stability, seed_limit);
  const int fallback_materialization_limit = core::RetrievalSeedSearchK (
      cfg.focus, cfg.sensitivity, cfg.stability, seed_search_limit);
  REQUIRE (
      operations::retrieval_trace::GetLastSqlFallbackMaterializedRowCount ()
      <= static_cast<std::size_t> (fallback_materialization_limit));
  REQUIRE (
      operations::historical_surface_search_cache_internal::Find (pctx)
      == recovery_state);
  REQUIRE (recovery_state->embedding_dim == 0);
  REQUIRE (recovery_state->entries.empty ());
  REQUIRE (recovery_state->current_entries.empty ());

  OperationContext second_ctx (signal, pctx, cfg, store.get ());
  second_ctx.SetShouldCheckRetrieval (true);
  second_ctx.SetWriteExclusionTs (signal.timestamp);
  auto second_tx = store->Begin ();
  operation.Execute (second_ctx, *second_tx);
  second_tx->Rollback ();
  REQUIRE (
      operations::historical_surface_search_cache_internal::Find (pctx)
      == recovery_state);
  const auto second_target = std::find_if (
      second_ctx.GetRetrievedMemoryCandidates ().begin (),
      second_ctx.GetRetrievedMemoryCandidates ().end (),
      [] (const auto &candidate) {
        return candidate.memory_id == kTargetMemoryId;
      });
  REQUIRE (second_target
           != second_ctx.GetRetrievedMemoryCandidates ().end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 1);
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval SQL fallback filters ineligible nearest rows before K",
           "[operations][graph][retrieval][sql][eligibility]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  constexpr long long kTargetMemoryId = 9000;
  SeedMemory (*store, kTargetMemoryId, 19000,
              VectorWithCosineToDim0 (0.90f), 1000, "eligible");
  for (long long offset = 0; offset < 900; ++offset)
    {
      const long long memory_id = 1000 + offset;
      SeedMemory (*store, memory_id, 10000 + offset, UnitVec (0), 1000,
                  "ineligible");
      store->Execute ("UPDATE memories SET kind = 'WORKING' "
                      "WHERE memory_id = ?",
                      { memory_id });
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx);
  auto signal = MakeSignal (UnitVec (0), 100000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) {
                 return candidate.memory_id == kTargetMemoryId;
               })
           != ctx.GetRetrievedMemoryCandidates ().end ());
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval SQL fallback performance probe",
           "[.benchmark][operations][graph][retrieval][sql]")
{
  const char *row_count_env = std::getenv ("LTM_FALLBACK_BENCH_ROWS");
  const char *repeats_env = std::getenv ("LTM_FALLBACK_BENCH_REPEATS");
  const char *reconstruct_every_env
      = std::getenv ("LTM_FALLBACK_BENCH_RECONSTRUCT_EVERY");
  const char *processor_complete_env
      = std::getenv ("LTM_FALLBACK_BENCH_PROCESSOR_COMPLETE");
  const long long row_count
      = row_count_env == nullptr ? 1915 : std::stoll (row_count_env);
  const int repeats = repeats_env == nullptr ? 12 : std::stoi (repeats_env);
  const long long reconstruct_every
      = reconstruct_every_env == nullptr
            ? 0
            : std::stoll (reconstruct_every_env);
  const bool processor_surface_complete
      = processor_complete_env != nullptr
        && std::string (processor_complete_env) == "1";
  REQUIRE (row_count > 0);
  REQUIRE (repeats > 0);
  REQUIRE (reconstruct_every >= 0);

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  ProcessorContext pctx;
  for (long long offset = 0; offset < row_count; ++offset)
    {
      Eigen::VectorXf embedding = UnitVec (static_cast<int> (offset % 255));
      embedding[255]
          = static_cast<float> ((offset % 97) + 1) / 10000.0f;
      embedding.normalize ();
      const long long memory_id = 1000 + offset;
      const long long embedding_id = 10000 + offset;
      const long long timestamp = 1000 + offset;
      SeedMemory (*store, memory_id, embedding_id, embedding, timestamp);
      pctx.UpsertRetrievalSurface (
          { memory_id, embedding_id, timestamp, timestamp, 0, 0, 0, 0,
            "LONG_TERM", "benchmark", "", -1.0, 0, 0.0, 0.0, 0.0,
            false, true, embedding });
    }

  long long reconstruction_count = 0;
  if (reconstruct_every > 0)
    {
      operations::constructive_recall::ReconstructionUpdatePolicy policy;
      policy.update_current_surface = false;
      auto reconstruction_tx = store->Begin ();
      for (long long offset = 0; offset < row_count;
           offset += reconstruct_every)
        {
          Eigen::VectorXf embedding
              = UnitVec (static_cast<int> (offset % 255));
          embedding[255]
              = static_cast<float> ((offset % 97) + 1) / 10000.0f;
          embedding.normalize ();
          const long long memory_id = 1000 + offset;
          REQUIRE (operations::constructive_recall::
                       AppendReconstructionWithEmbedding (
                           *reconstruction_tx, memory_id, embedding, {},
                           100000 + offset, 0.1, "benchmark", 1.0, 1.0,
                           policy)
                   > 0);
          const auto latest = operations::constructive_recall::
              LoadLatestReconstruction (*reconstruction_tx, memory_id);
          REQUIRE (latest.has_value ());
          pctx.UpsertRetrievalSurface (
              { memory_id, latest->embedding_id, 1000 + offset,
                1000 + offset, 0, 0, 0, 0, "LONG_TERM", "benchmark", "",
                -1.0, 0, 0.0, 0.0, 0.0, false, true, embedding });
          ++reconstruction_count;
        }
      reconstruction_tx->Commit ();
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  operations::historical_surface_search_cache_internal::
      SetCurrentSurfaceDatabaseCurrent (pctx, reconstruction_count == 0);
  operations::historical_surface_search_cache_internal::
      SetProcessorSurfaceComplete (pctx, processor_surface_complete);
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx);
  auto signal = MakeSignal (UnitVec (0), 1000000000);
  signal.retention = Retention::Ephemeral;
  std::vector<double> elapsed_ms;
  elapsed_ms.reserve (static_cast<std::size_t> (repeats));
  for (int repeat = 0; repeat < repeats; ++repeat)
    {
      OperationContext ctx (signal, pctx, cfg, store.get ());
      ctx.SetShouldCheckRetrieval (true);
      ctx.SetWriteExclusionTs (signal.timestamp);
      GraphAugmentedRetrieveCandidates operation;
      auto tx = store->Begin ();
      const auto start = std::chrono::steady_clock::now ();
      operation.Execute (ctx, *tx);
      const auto end = std::chrono::steady_clock::now ();
      tx->Rollback ();
      REQUIRE_FALSE (ctx.GetRetrievedMemoryCandidates ().empty ());
      elapsed_ms.push_back (
          std::chrono::duration<double, std::milli> (end - start).count ());
    }
  std::sort (elapsed_ms.begin (), elapsed_ms.end ());
  const auto percentile = [&elapsed_ms] (double fraction) {
    const std::size_t index = std::min (
        elapsed_ms.size () - 1,
        static_cast<std::size_t> (
            std::ceil (fraction * static_cast<double> (elapsed_ms.size ())))
            - 1);
    return elapsed_ms[index];
  };
  std::cout << "CORTEXT_FALLBACK_BENCH {\"rows\":" << row_count
            << ",\"reconstructions\":" << reconstruction_count
            << ",\"current_surface_database_current\":"
            << (reconstruction_count == 0 ? "true" : "false")
            << ",\"processor_surface_complete\":"
            << (processor_surface_complete ? "true" : "false")
            << ",\"repeats\":" << repeats
            << ",\"p50_ms\":" << percentile (0.50)
            << ",\"p95_ms\":" << percentile (0.95) << "}" << std::endl;
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval ephemeral SQL fallback does not install cache",
           "[operations][graph][retrieval][sql][ephemeral]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  SeedMemory (*store, 10, 100, UnitVec (0), 1000);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  auto signal = MakeSignal (UnitVec (0), 100000);
  signal.retention = Retention::Ephemeral;
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  REQUIRE_FALSE (ctx.GetRetrievedMemoryCandidates ().empty ());
  REQUIRE (
      operations::historical_surface_search_cache_internal::Find (pctx)
      == nullptr);
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 1);
}

TEST_CASE ("Graph retrieval cache rebuild accepts shared base embeddings",
           "[operations][graph][retrieval][cache][shared-embedding]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  store->Execute ("UPDATE memories SET kind = 'WORKING' WHERE memory_id = 10");
  store->Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
      "start_ts, created_at) VALUES(?, ?, ?, 'LONG_TERM', ?, ?)",
      { 20LL, 100LL, std::string ("shared"), 500LL, 500LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  auto signal = MakeSignal (UnitVec (0), 100000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  const auto state
      = operations::historical_surface_search_cache_internal::Find (pctx);
  REQUIRE (state != nullptr);
  REQUIRE_FALSE (state->recovery_failed);
  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) { return candidate.memory_id == 20; })
           != ctx.GetRetrievedMemoryCandidates ().end ());
  const auto shared_entry = std::find_if (
      state->entries.begin (), state->entries.end (),
      [] (const auto &entry) { return entry.embedding_id == 100; });
  REQUIRE (shared_entry != state->entries.end ());
  REQUIRE (shared_entry->memory_references.size () == 1);
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval bounded fallback keeps eligible shared sibling",
           "[operations][graph][retrieval][sql][shared-embedding][supersession]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  store->Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
      "start_ts, created_at) VALUES(?, ?, ?, 'LONG_TERM', ?, ?)",
      { 11LL, 100LL, std::string ("shared"), 1100LL, 1100LL });
  SeedMemory (*store, 20, 200, VectorWithCosineToDim0 (0.92f), 2000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'supersedes', ?, ?)",
      { 20LL, 10LL, 1.0, 2000LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  pctx.UpsertRetrievalSurface (
      { 20, 200, 2000, 2000, 0, 0, 0, 0, "LONG_TERM", "replacement", "",
        -1.0, 0, 0.0, 0.0, 0.0, false, true,
        VectorWithCosineToDim0 (0.92f) });
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx);
  auto signal = MakeSignal (UnitVec (0), 3000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) { return candidate.memory_id == 10; })
           == ctx.GetRetrievedMemoryCandidates ().end ());
  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) { return candidate.memory_id == 11; })
           != ctx.GetRetrievedMemoryCandidates ().end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 1);
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval fallback pages past cosine family saturation",
           "[operations][graph][retrieval][sql][pagination][family]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  for (long long offset = 0; offset < 600; ++offset)
    {
      auto embedding = UnitVec (0);
      embedding[1] = static_cast<float> (offset + 1) / 1000000.0f;
      embedding.normalize ();
      SeedMemory (*store, 1000 + offset, 10000 + offset, embedding,
                  1100 + offset, "cosine-family");
    }
  constexpr long long kTargetMemoryId = 9000;
  SeedMemory (*store, kTargetMemoryId, 200,
              VectorWithCosineToDim0 (0.95f), 2000, "target");

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx);
  auto signal = MakeSignal (UnitVec (0), 100000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) {
                 return candidate.memory_id == kTargetMemoryId;
           })
           != ctx.GetRetrievedMemoryCandidates ().end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 2);
  const int seed_limit = std::max (1, core::RetrievalMaxResults (cfg.focus));
  const int seed_search_limit = core::RetrievalSeedSearchK (
      cfg.focus, cfg.sensitivity, cfg.stability, seed_limit);
  const int page_limit = core::RetrievalSeedSearchK (
      cfg.focus, cfg.sensitivity, cfg.stability, seed_search_limit);
  const auto materialized
      = operations::retrieval_trace::GetLastSqlFallbackMaterializedRowCount ();
  REQUIRE (materialized > static_cast<std::size_t> (page_limit));
  REQUIRE (materialized
           <= static_cast<std::size_t> (2 * page_limit));
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval fallback preserves overfetch after surface refresh",
           "[operations][graph][retrieval][sql][pagination][reconstruction]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  const int seed_limit = std::max (1, core::RetrievalMaxResults (cfg.focus));
  ProcessorContext pctx;
  operations::constructive_recall::ReconstructionUpdatePolicy policy;
  policy.update_current_surface = false;
  operations::historical_surface_search_cache_internal::
      SetCurrentSurfaceDatabaseCurrent (pctx, true);
  for (int offset = 0; offset < seed_limit; ++offset)
    {
      Eigen::VectorXf base = Eigen::VectorXf::Zero (kEmbeddingDim);
      constexpr float cosine = 0.90f;
      base[0] = cosine;
      base[1 + offset]
          = std::sqrt (std::max (0.0f, 1.0f - cosine * cosine));
      const long long memory_id = 1000 + offset;
      const long long embedding_id = 10000 + offset * 100;
      SeedMemory (*store, memory_id, embedding_id, base, 1000 + offset,
                  "reconstructed-family");
      auto reconstruction_tx = store->Begin ();
      REQUIRE (operations::constructive_recall::
                   AppendReconstructionWithEmbedding (
                       *reconstruction_tx, memory_id, UnitVec (0), {},
                       5000 + offset, 0.1, "surface-refresh", 1.0, 1.0,
                       policy, &pctx)
               > 0);
      const auto latest = operations::constructive_recall::
          LoadLatestReconstruction (*reconstruction_tx, memory_id);
      REQUIRE (latest.has_value ());
      reconstruction_tx->Commit ();
      pctx.UpsertRetrievalSurface (
          { memory_id, latest->embedding_id, 1000 + offset, 1000 + offset,
            0, 0, 0, 0, "LONG_TERM", "reconstructed-family", "", -1.0,
            0, 0.0, 0.0, 0.0, false, true, UnitVec (0) });
    }
  REQUIRE_FALSE (operations::historical_surface_search_cache_internal::
                     CurrentSurfaceDatabaseCurrent (pctx));

  constexpr long long kTargetMemoryId = 9000;
  SeedMemory (*store, kTargetMemoryId, 30000,
              VectorWithCosineToDim0 (0.75f), 2000, "target");
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx);

  auto signal = MakeSignal (UnitVec (0), 100000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) {
                 return candidate.memory_id == kTargetMemoryId;
               })
           != ctx.GetRetrievedMemoryCandidates ().end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 1);
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval fallback preserves overfetch after database reconstruction refresh",
           "[operations][graph][retrieval][sql][pagination][reconstruction]")
{
  cortext::testing::ScopedEnvVar disable_current (
      "CORTEXT_DISABLE_CURRENT_MEMORY_SURFACE_WRITES", "1");
  if (!operations::constructive_recall::CurrentSurfaceWritesDisabled ())
    {
      SKIP ("current-surface write hook is disabled in this build");
    }

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  const int seed_limit = std::max (1, core::RetrievalMaxResults (cfg.focus));
  ProcessorContext pctx;
  pctx.UpsertRetrievalSurface (
      { 99999, 99999, 1, 1, 0, 0, 0, 0, "LONG_TERM", "unrelated", "",
        -1.0, 0, 0.0, 0.0, 0.0, false, true, UnitVec (4) });

  for (int offset = 0; offset < seed_limit; ++offset)
    {
      Eigen::VectorXf base = Eigen::VectorXf::Zero (kEmbeddingDim);
      constexpr float cosine = 0.90f;
      base[0] = cosine;
      base[1 + offset]
          = std::sqrt (std::max (0.0f, 1.0f - cosine * cosine));
      const long long memory_id = 1000 + offset;
      SeedMemory (*store, memory_id, 10000 + offset, base, 1000 + offset,
                  "database-reconstructed-family");
    }
  for (int offset = 0; offset < seed_limit; ++offset)
    {
      auto reconstruction_tx = store->Begin ();
      REQUIRE (operations::constructive_recall::
                   AppendReconstructionWithEmbedding (
                       *reconstruction_tx, 1000 + offset, UnitVec (0), {},
                       5000 + offset, 0.1, "review", 1.0, 1.0)
               > 0);
      reconstruction_tx->Commit ();
    }

  constexpr long long kTargetMemoryId = 9000;
  SeedMemory (*store, kTargetMemoryId, 30000,
              VectorWithCosineToDim0 (0.75f), 2000, "target");
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx);

  auto signal = MakeSignal (UnitVec (0), 100000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) {
                 return candidate.memory_id == kTargetMemoryId;
               })
           != ctx.GetRetrievedMemoryCandidates ().end ());
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval bypasses stale valid cache for latest reconstructions",
           "[operations][graph][retrieval][cache][sql][reconstruction]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  const int seed_limit = std::max (1, core::RetrievalMaxResults (cfg.focus));
  const int seed_search_limit = core::RetrievalSeedSearchK (
      cfg.focus, cfg.sensitivity, cfg.stability, seed_limit);
  ProcessorContext pctx;
  std::vector<operations::historical_surface_search_cache_internal::Entry>
      historical_entries;
  std::vector<operations::historical_surface_search_cache_internal::Entry>
      current_entries;
  historical_entries.reserve (static_cast<std::size_t> (seed_search_limit + 1));
  current_entries.reserve (static_cast<std::size_t> (seed_search_limit + 1));

  for (int offset = 0; offset < seed_search_limit; ++offset)
    {
      Eigen::VectorXf base = VectorWithCosineAndDiverseResidual (
          0.90f, static_cast<std::uint64_t> (offset));
      const long long memory_id = 1000 + offset;
      const long long embedding_id = 10000 + offset;
      SeedMemory (*store, memory_id, embedding_id, base, 1000 + offset,
                  "stale-cache-family");
      pctx.UpsertRetrievalSurface (
          { memory_id, embedding_id, 1000 + offset, 1000 + offset, 0, 0,
            0, 0, "LONG_TERM", "stale-cache-family", "", -1.0, 0, 0.0,
            0.0, 0.0, false, true, base });
      historical_entries.push_back (
          { embedding_id, memory_id, 1000 + offset, "LONG_TERM",
            "stale-cache-family", base });
      current_entries.push_back (
          { embedding_id, memory_id, 0, std::string (), std::string (),
            base });
    }

  constexpr long long kTargetMemoryId = 9000;
  constexpr long long kTargetEmbeddingId = 30000;
  const Eigen::VectorXf target = VectorWithCosineToDim0 (0.75f);
  SeedMemory (*store, kTargetMemoryId, kTargetEmbeddingId, target, 2000,
              "target");
  pctx.UpsertRetrievalSurface (
      { kTargetMemoryId, kTargetEmbeddingId, 2000, 2000, 0, 0, 0, 0,
        "LONG_TERM", "target", "", -1.0, 0, 0.0, 0.0, 0.0, false, true,
        target });
  historical_entries.push_back (
      { kTargetEmbeddingId, kTargetMemoryId, 2000, "LONG_TERM", "target",
        target });
  current_entries.push_back (
      { kTargetEmbeddingId, kTargetMemoryId, 0, std::string (),
        std::string (), target });

  REQUIRE (operations::historical_surface_search_cache_internal::Reset (
      pctx, std::move (historical_entries), std::move (current_entries)));
  operations::historical_surface_search_cache_internal::
      SetCurrentSurfaceDatabaseCurrent (pctx, true);

  operations::constructive_recall::ReconstructionUpdatePolicy policy;
  policy.update_current_surface = false;
  for (int offset = 0; offset < seed_search_limit; ++offset)
    {
      auto reconstruction_tx = store->Begin ();
      const long long memory_id = 1000 + offset;
      REQUIRE (operations::constructive_recall::
                   AppendReconstructionWithEmbedding (
                       *reconstruction_tx, memory_id, UnitVec (0), {},
                       5000 + offset, 0.1, "stale-cache", 1.0, 1.0, policy,
                       &pctx)
               > 0);
      const auto latest = operations::constructive_recall::
          LoadLatestReconstruction (*reconstruction_tx, memory_id);
      REQUIRE (latest.has_value ());
      reconstruction_tx->Commit ();
      pctx.UpsertRetrievalSurface (
          { memory_id, latest->embedding_id, 1000 + offset, 1000 + offset,
            0, 0, 0, 0, "LONG_TERM", "stale-cache-family", "", -1.0, 0,
            0.0, 0.0, 0.0, false, true, UnitVec (0) });
    }
  const auto stale_state
      = operations::historical_surface_search_cache_internal::Find (pctx);
  REQUIRE (stale_state != nullptr);
  REQUIRE_FALSE (stale_state->recovery_failed);
  REQUIRE_FALSE (stale_state->current_surface_database_current);

  auto signal = MakeSignal (UnitVec (0), 100000);
  signal.retention = Retention::Ephemeral;
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) {
                 return candidate.memory_id == kTargetMemoryId;
               })
           != ctx.GetRetrievedMemoryCandidates ().end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 0);

  operations::historical_surface_search_cache_internal::
      SetProcessorSurfaceComplete (pctx, true);
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx);
  OperationContext recovery_ctx (signal, pctx, cfg, store.get ());
  recovery_ctx.SetShouldCheckRetrieval (true);
  recovery_ctx.SetWriteExclusionTs (signal.timestamp);
  auto recovery_tx = store->Begin ();
  operation.Execute (recovery_ctx, *recovery_tx);
  recovery_tx->Rollback ();
  REQUIRE (std::find_if (
               recovery_ctx.GetRetrievedMemoryCandidates ().begin (),
               recovery_ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) {
                 return candidate.memory_id == kTargetMemoryId;
               })
           != recovery_ctx.GetRetrievedMemoryCandidates ().end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 0);
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval keeps eligible sibling of superseded shared embedding",
           "[operations][graph][retrieval][cache][shared-embedding][supersession]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  store->Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
      "start_ts, created_at) VALUES(?, ?, ?, 'LONG_TERM', ?, ?)",
      { 11LL, 100LL, std::string ("shared"), 1100LL, 1100LL });
  SeedMemory (*store, 20, 200, VectorWithCosineToDim0 (0.92f), 2000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'supersedes', ?, ?)",
      { 20LL, 10LL, 1.0, 2000LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 3000));
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 10LL)
           == out.candidate_memory_ids.end ());
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 11LL)
           != out.candidate_memory_ids.end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 0);
}

TEST_CASE ("Historical search cache append retains memory alternative",
           "[operations][graph][retrieval][cache][append]")
{
  ProcessorContext pctx;
  REQUIRE (
      operations::historical_surface_search_cache_internal::Reset (pctx, {}));
  operations::historical_surface_search_cache_internal::Append (
      pctx, { 100, 10, 1000, "LONG_TERM", "opaque", UnitVec (0) });

  const auto state
      = operations::historical_surface_search_cache_internal::Find (pctx);
  REQUIRE (state != nullptr);
  REQUIRE (state->entries.size () == 1);
  REQUIRE (state->entries.front ().memory_references.size () == 1);
  REQUIRE (state->entries.front ().memory_references.front ().memory_id == 10);
  operations::historical_surface_search_cache_internal::Erase (pctx);
}
