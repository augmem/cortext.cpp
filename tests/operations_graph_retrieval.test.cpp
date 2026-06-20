#include "test_helpers.hpp"
#include "../src/operations/retrieval_trace_state.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <algorithm>

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

void
SeedMemory (Store &store, long long memory_id, long long embedding_id,
            const Eigen::VectorXf &embedding, std::uint64_t ts)
{
  store.Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { embedding_id, ToFloatVec (embedding), static_cast<long long> (ts) });
  store.Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, start_ts, "
      "created_at) VALUES(?, ?, 'test', 'LONG_TERM', ?, ?)",
      { memory_id, embedding_id, static_cast<long long> (ts),
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
  REQUIRE_FALSE (selected.empty ());
  REQUIRE (selected.front () == 100);
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
                      out.candidate_memory_ids.end (), 200)
           != out.candidate_memory_ids.end ());
}
