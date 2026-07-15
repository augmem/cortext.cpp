#include "test_helpers.hpp"
#include "../src/operations/constructive_recall_internal.hpp"
#include "../src/operations/retrieval_trace_state.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cortext/core/utils.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
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

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
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
}
