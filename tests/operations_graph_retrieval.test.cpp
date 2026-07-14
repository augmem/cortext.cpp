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
#include <cmath>
#include <string>
#include <unordered_map>
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

class InjectRetrievalSurfaceOp : public IOperation
{
public:
  explicit InjectRetrievalSurfaceOp (
      std::vector<ProcessorContext::RetrievalSurfaceEntry> entries)
      : entries_ (std::move (entries))
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &p_ctx = ctx.GetProcessorContext ();
    p_ctx.retrieval_surface_cache.clear ();
    p_ctx.retrieval_surface_index.clear ();
    p_ctx.retrieval_surface_embedding_index.clear ();
    p_ctx.retrieval_surface_source_index.clear ();
    p_ctx.retrieval_surface_source_index_dirty.clear ();
    for (auto entry : entries_)
      {
        p_ctx.UpsertRetrievalSurface (std::move (entry));
      }
  }

private:
  std::vector<ProcessorContext::RetrievalSurfaceEntry> entries_;
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

ProcessorContext::RetrievalSurfaceEntry
MakeSurfaceEntry (long long memory_id, long long embedding_id,
                  const Eigen::VectorXf &embedding,
                  const std::string &source_id, long long start_ts)
{
  ProcessorContext::RetrievalSurfaceEntry entry;
  entry.memory_id = memory_id;
  entry.embedding_id = embedding_id;
  entry.embedding = embedding;
  entry.kind = "LONG_TERM";
  entry.source_id = source_id;
  entry.start_ts = start_ts;
  entry.created_at = start_ts;
  return entry;
}

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
  cortext::testing::ScopedEnvVar disable_source_expansion (
      "CORTEXT_DISABLE_SOURCE_SEED_GRAPH_EXPANSION", "1");

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

  auto stale_it = std::find_if (
      ranked.begin (), ranked.end (),
      [] (const auto &candidate) { return candidate.memory_id == 10LL; });
  auto correction_it = std::find_if (
      ranked.begin (), ranked.end (),
      [] (const auto &candidate) { return candidate.memory_id == 20LL; });
  REQUIRE (stale_it != ranked.end ());
  REQUIRE (correction_it != ranked.end ());
  REQUIRE (stale_it->score < correction_it->score);
  REQUIRE (stale_it->activation.partial_match_penalty > 0.0);
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
}

TEST_CASE ("Graph retrieval temporal score decays across multi-month ages",
           "[operations][graph][retrieval][temporal]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  cortext::testing::ScopedEnvVar disable_source_expansion (
      "CORTEXT_DISABLE_SOURCE_SEED_GRAPH_EXPANSION", "1");

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  constexpr long long kDayMs = 24LL * 60LL * 60LL * 1000LL;
  constexpr std::uint64_t now = 200ULL * static_cast<std::uint64_t> (kDayMs);
  SeedMemory (*store, 10, 100, UnitVec (0), now - 60ULL * 1000ULL);
  SeedMemory (*store, 20, 200, UnitVec (0), now - 30ULL * kDayMs);
  SeedMemory (*store, 30, 300, UnitVec (0), now - 90ULL * kDayMs);

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

TEST_CASE ("Graph retrieval expands bounded same-source neighbors from cache",
           "[operations][graph][retrieval][source]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  cortext::testing::SeedEmbeddingV2 (*store, 100, ToFloatVec (UnitVec (0)),
                                     1000);
  cortext::testing::SeedMemoryV2 (*store, 10, 100, "conversation/db-anchor",
                                  "LONG_TERM", 1.0, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 200, ToFloatVec (UnitVec (2)),
                                     1100);
  cortext::testing::SeedMemoryV2 (*store, 20, 200, "conversation/db-neighbor",
                                  "LONG_TERM", 1.0, 1100);
  for (long long i = 0; i < 7; ++i)
    {
      const long long memory_id = 1000 + i;
      const long long embedding_id = 10000 + i;
      cortext::testing::SeedEmbeddingV2 (
          *store, embedding_id,
          ToFloatVec (VectorWithCosineToDim0 (0.01f)),
          1000);
      cortext::testing::SeedMemoryV2 (*store, memory_id, embedding_id,
                                      "distractor/run", "LONG_TERM", 1.0,
                                      1000);
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;
  std::vector<ProcessorContext::RetrievalSurfaceEntry> cache_entries;
  cache_entries.push_back (
      MakeSurfaceEntry (10, 100, UnitVec (0), "conversation/cache", 1000));
  cache_entries.push_back (
      MakeSurfaceEntry (20, 200, UnitVec (2), "conversation/cache", 1100));
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<InjectRetrievalSurfaceOp> (std::move (cache_entries)),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 100000));
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 20)
           != out.candidate_memory_ids.end ());
}

TEST_CASE ("Graph retrieval expands adjacent turn source ids",
           "[operations][graph][retrieval][source]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000,
              "chat/run-1/turn-0080/caller");
  SeedMemory (*store, 20, 200, UnitVec (2), 1100,
              "chat/run-1/turn-0081/model");
  SeedMemory (*store, 21, 201, UnitVec (2), 1200,
              "chat/run-1/turn-00810/model");
  for (long long i = 0; i < 12; ++i)
    {
      SeedMemory (*store, 1000 + i, 10000 + i,
                  VectorWithCosineToDim0 (0.01f), 1000,
                  "chat/other/turn-0081/model");
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

  const auto out = processor.Process (MakeSignal (UnitVec (0), 100000));
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 20)
           != out.candidate_memory_ids.end ());
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 21)
           == out.candidate_memory_ids.end ());
}

TEST_CASE ("Graph retrieval escapes wildcard characters in turn source ids",
           "[operations][graph][retrieval][source]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000,
              "chat/run_1/turn-0080/caller");
  SeedMemory (*store, 20, 200, UnitVec (2), 1100,
              "chat/run_1/turn-0081/model");
  SeedMemory (*store, 21, 201, UnitVec (2), 1100,
              "chat/runA1/turn-0081/model");
  for (long long i = 0; i < 12; ++i)
    {
      SeedMemory (*store, 1000 + i, 10000 + i,
                  VectorWithCosineToDim0 (0.01f), 1000,
                  "chat/other/turn-0081/model");
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

  const auto out = processor.Process (MakeSignal (UnitVec (0), 100000));
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 20)
           != out.candidate_memory_ids.end ());
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 21)
           == out.candidate_memory_ids.end ());
}
