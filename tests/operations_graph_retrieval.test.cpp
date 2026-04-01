// tests/operations_graph_retrieval.test.cpp
#include "test_helpers.hpp"
#include <any>
#include <catch2/catch_test_macros.hpp>

#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;
using cortext::operations::GraphAugmentedRetrieveCandidates;

namespace
{
constexpr int kEmbeddingDim = 256;

/// @brief Creates a 256-dim unit vector with value at index 0.
static Eigen::VectorXf
UnitVec256 (float first_val)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = first_val;
  const float n = v.norm ();
  if (n > 1e-9f)
    v /= n;
  return v;
}

/// @brief Creates a 256-dim unit vector with value at index 1.
static Eigen::VectorXf
UnitVec256Second (float second_val)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[1] = second_val;
  const float n = v.norm ();
  if (n > 1e-9f)
    v /= n;
  return v;
}

/// @brief Converts Eigen vector to std::vector<float> for DB storage.
static std::vector<float>
ToFloatVec (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

static Signal
MakeSignal (const Eigen::VectorXf &emb, uint64_t ts)
{
  Signal s;
  s.embedding = emb;
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
    auto &p_ctx = ctx.GetProcessorContext ();
    if (p_ctx.memory_stream.empty ())
      {
        p_ctx.memory_stream.push_back (ctx.GetSignal ().embedding);
      }
    auto &acc = p_ctx.accumulator_states[ctx.GetSignal ().source_id];
    acc.mu_acc = ctx.GetSignal ().embedding;
    acc.c_t = ctx.GetSignal ().embedding;
  }
};
} // namespace

TEST_CASE ("V2: Alg31 expands vector seeds via ASSOCIATIONS and returns expanded ids",
           "[operations][graph][alg31][v2]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // Create 256-dim embeddings for vec0 compatibility.
  Eigen::VectorXf emb1 = UnitVec256 (1.0f);       // First dimension = 1
  Eigen::VectorXf emb2 = UnitVec256Second (1.0f); // Second dimension = 1

  // id=1 aligns with query, id=2 does not.
  // V2: Insert into embeddings (minimal vec0 table)
  store->Execute ("INSERT INTO embeddings(embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 1LL, ToFloatVec (emb1), 0LL });
  store->Execute ("INSERT INTO embeddings(embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 2LL, ToFloatVec (emb2), 0LL });

  // V2: Insert into memories (comprehensive metadata)
  store->Execute ("INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
                  "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
                  "VALUES (?, ?, 'test', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, 1.0, 0)",
                  { 1LL, 1LL });
  store->Execute ("INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
                  "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
                  "VALUES (?, ?, 'test', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, 1.0, 0)",
                  { 2LL, 2LL });

  // V2: Create ASSOCIATIONS edge directly between memories (no intermediate label)
  // memory_id 1 -> memory_id 2 via 'co_occurs' edge
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'co_occurs', 1.0)",
      { 1LL, 2LL });

  // Low focus => depth=2 (GraphDepth(F)), ensuring expansion reaches memory 2.
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto ops = std::make_unique<OperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  auto out = processor.Process (MakeSignal (UnitVec256 (1.0f), 10));

  // Should include both seed (1) and expanded (2).
  bool has1 = false, has2 = false;
  for (const auto id : out.candidate_memory_ids)
    {
      if (id == 1LL)
        has1 = true;
      if (id == 2LL)
        has2 = true;
    }
  REQUIRE (has1);
  REQUIRE (has2);
}
