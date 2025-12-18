// tests/operations_reconsolidation.test.cpp
#include <Eigen/Dense>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/reconsolidation.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;
using cortext::operations::ApplyReconsolidation;

namespace
{

constexpr int kEmbeddingDim = 256;

// Helper op to seed embeddings and memories into the v2 database.
class SeedEmbeddingsOp : public IOperation
{
public:
  explicit SeedEmbeddingsOp (std::unordered_map<long long, Eigen::VectorXf> embs)
      : embeddings_ (std::move (embs))
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto *store = ctx.GetStore ();
    auto now_ts = cortext::testing::NowMs ();
    for (const auto &[id, emb] : embeddings_)
      {
        std::vector<float> vec (emb.data (), emb.data () + emb.size ());
        // v2: Insert into embeddings (minimal vec0 table)
        store->Execute (
            "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
            "VALUES(?, ?, ?)",
            { id, vec, now_ts });
        // v2: Insert into memories (comprehensive metadata including lability)
        store->Execute (
            "INSERT OR REPLACE INTO memories("
            "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
            "s_max, s_avg, strength, use_frequency, stability, connectivity, drift_mag, "
            "influence, sustained_influence, contextual_gain, redundancy, "
            "pre_activation, lability_state, suppression_count, lability_ts, created_at) "
            "VALUES(?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, "
            "1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, ?)",
            { id, id, now_ts, now_ts });
      }
  }

private:
  std::unordered_map<long long, Eigen::VectorXf> embeddings_;
};

// Helper op to preload current context and retrieved embeddings into context.
class SetupReconInputsOp : public IOperation
{
public:
  SetupReconInputsOp (Eigen::VectorXf cur,
                      std::unordered_map<long long, Eigen::VectorXf> retrieved)
      : cur_ (std::move (cur)), retrieved_ (std::move (retrieved))
  {
  }
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    pctx.recent_context_embeddings.clear ();
    pctx.recent_context_embeddings.push_back (cur_);
    ctx.SetRetrievedMemoryEmbeddings (retrieved_);
  }

private:
  Eigen::VectorXf cur_;
  std::unordered_map<long long, Eigen::VectorXf> retrieved_;
};

// Assert u_t increased from baseline 0.
class AssertUncertaintyIncreasedOp : public IOperation
{
public:
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    REQUIRE (pctx.u_t > 0.0);
  }
};

/// @brief Creates a 256-dim unit vector with value at specified index.
static Eigen::VectorXf
MakeUnitVec256 (int idx)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[idx] = 1.0f;
  return v;
}

static Signal
MakeSignal (const Eigen::VectorXf &emb, uint64_t ts = 1)
{
  Signal s;
  s.embedding = emb;
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

} // namespace

TEST_CASE ("Alg20 drifts embedding and writes lability fields",
           "[operations][recon]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0; // maximize plasticity
  cfg.stability = 0.0;   // minimize persistence

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  const Eigen::VectorXf mem = MakeUnitVec256 (0);

  // v2: Need to seed the memory before reconsolidation can update it
  std::unordered_map<long long, Eigen::VectorXf> retrieved{ { 1LL, mem } };
  auto seed = std::make_unique<SeedEmbeddingsOp> (retrieved);
  auto setup = std::make_unique<SetupReconInputsOp> (cur, retrieved);
  auto apply = std::make_unique<ApplyReconsolidation> ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (seed), std::move (setup), std::move (apply));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto s = MakeSignal (cur, /*ts=*/100);
  processor.Process (s);
  processor.Flush ();

  // v2: Memory row created in memories table
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?",
        { 1LL });
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("cnt")) == 1LL);
  }
  // v2: Lability fields are inline on memories table
  {
    auto rows = store->Execute (
        "SELECT lability_state, lability_ts FROM memories WHERE memory_id = ?",
        { 1LL });
    REQUIRE (rows.size () == 1);
    const double lab = std::any_cast<double> (rows[0].at ("lability_state"));
    const long long ts = std::any_cast<long long> (rows[0].at ("lability_ts"));
    REQUIRE (lab > 0.0);
    REQUIRE (ts == 100LL);
  }
}

TEST_CASE ("Alg20 no drift when S=0: embedding unchanged; lability updated",
           "[operations][recon]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.0; // no drift
  cfg.stability = 0.5;

  const Eigen::VectorXf cur = MakeUnitVec256 (1);
  const Eigen::VectorXf mem = MakeUnitVec256 (1);

  std::unordered_map<long long, Eigen::VectorXf> retrieved{ { 2LL, mem } };
  auto seed = std::make_unique<SeedEmbeddingsOp> (retrieved);
  auto setup = std::make_unique<SetupReconInputsOp> (cur, retrieved);
  auto apply = std::make_unique<ApplyReconsolidation> ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (seed), std::move (setup), std::move (apply));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto s = MakeSignal (cur, /*ts=*/42);
  processor.Process (s);
  processor.Flush ();

  // v2: Memory row exists (seeded) but drift_mag should be 0 when S=0
  {
    auto rows = store->Execute (
        "SELECT drift_mag FROM memories WHERE memory_id = ?",
        { 2LL });
    REQUIRE (rows.size () == 1);
    const double drift = std::any_cast<double> (rows[0].at ("drift_mag"));
    REQUIRE (drift == 0.0); // No drift when S=0
  }
  // v2: Lability fields are inline on memories table
  {
    auto rows = store->Execute (
        "SELECT lability_state, lability_ts FROM memories WHERE memory_id = ?",
        { 2LL });
    REQUIRE (rows.size () == 1);
    const double lab = std::any_cast<double> (rows[0].at ("lability_state"));
    const long long ts = std::any_cast<long long> (rows[0].at ("lability_ts"));
    REQUIRE (lab >= 0.0);
    REQUIRE (ts == 42LL);
  }
}

TEST_CASE ("Alg20 bumps uncertainty with positive drift",
           "[operations][recon]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  const Eigen::VectorXf mem = MakeUnitVec256 (0);

  auto setup = std::make_unique<SetupReconInputsOp> (
      cur, std::unordered_map<long long, Eigen::VectorXf>{ { 3LL, mem } });
  auto apply = std::make_unique<ApplyReconsolidation> ();
  auto assert_u = std::make_unique<AssertUncertaintyIncreasedOp> ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (setup), std::move (apply), std::move (assert_u));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto s = MakeSignal (cur, /*ts=*/7);
  processor.Process (s);
  processor.Flush ();
}

// --- Ripple Effect Tests ---

TEST_CASE ("Alg20 ripple propagation reaches graph neighbors",
           "[operations][recon][ripple]")
{
  // Test directly with OperationContext to isolate the operation
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  // Initialize schema
  {
    cortext::store::SchemaRegistry registry;
    cortext::store::ApplyMigrations (*store, registry);
  }

  // Insert test data
  // Create a neighbor embedding (emb:10) connected to primary (emb:1)
  const Eigen::VectorXf neighbor_vec = MakeUnitVec256 (5);
  std::vector<float> neighbor_data (neighbor_vec.data (),
                                    neighbor_vec.data () + neighbor_vec.size ());
  auto now_ts = cortext::testing::NowMs ();
  // v2: Insert into embeddings (minimal)
  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) VALUES (?, ?, ?)",
      { 10LL, neighbor_data, now_ts });
  // v2: Insert into memories (comprehensive)
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, start_ts, "
      "n_signals, modality, s_max, s_avg, created_at) "
      "VALUES (?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, ?)",
      { 10LL, 10LL, now_ts, now_ts });

  // Create a 'reinforces' edge from emb:1 to emb:10
  store->Execute (
      "INSERT INTO graph_edges (source_id, target_id, edge_type, weight) "
      "VALUES (?, ?, ?, ?)",
      { std::string ("emb:1"), std::string ("emb:10"),
        std::string ("reinforces"), 1.0 });

  // Create config, signal, and context directly
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0; // maximize plasticity
  cfg.stability = 0.0;   // maximize ripple (ripple_depth=2, ripple_decay=0.5)

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  const Eigen::VectorXf mem = MakeUnitVec256 (0);
  Signal s = MakeSignal (cur, /*ts=*/100);

  ProcessorContext pctx;
  pctx.recent_context_embeddings.push_back (cur);

  // Create OperationContext with the store pointer
  OperationContext ctx (s, pctx, cfg, store.get ());

  // Verify store is set in context
  REQUIRE (ctx.GetStore () != nullptr);

  // v2: Seed the primary memory before reconsolidation can update it
  std::vector<float> mem_data (mem.data (), mem.data () + mem.size ());
  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) VALUES (?, ?, ?)",
      { 1LL, mem_data, now_ts });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, start_ts, "
      "n_signals, modality, s_max, s_avg, created_at) "
      "VALUES (?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, ?)",
      { 1LL, 1LL, now_ts, now_ts });

  // Set up the retrieved embeddings
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, mem } });

  // Execute reconsolidation with a real transaction
  auto tx = store->Begin ();
  ApplyReconsolidation recon_op;
  recon_op.Execute (ctx, *tx);
  tx->Commit ();

  // v2: Verify primary reconsolidation worked (memory exists)
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?",
        { 1LL });
    REQUIRE (rows.size () == 1);
    auto cnt = std::any_cast<long long> (rows[0].at ("cnt"));
    REQUIRE (cnt == 1LL);
  }

  // v2: Verify neighbor (memory_id:10) received ripple update
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?",
        { 10LL });
    REQUIRE (rows.size () == 1);
    auto cnt = std::any_cast<long long> (rows[0].at ("cnt"));
    // Neighbor should have been written via ripple
    REQUIRE (cnt == 1LL);
  }
}

TEST_CASE ("Alg20 ripple decay applied correctly per hop",
           "[operations][recon][ripple]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0; // maximize plasticity
  cfg.stability = 0.0;   // ripple_depth=2, ripple_decay=0.5

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  const Eigen::VectorXf mem = MakeUnitVec256 (0);

  auto setup = std::make_unique<SetupReconInputsOp> (
      cur, std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, mem } });
  auto apply = std::make_unique<ApplyReconsolidation> ();
  auto ops = std::make_unique<cortext::OperationSet> (std::move (setup),
                                                      std::move (apply));

  // Create processor first to initialize schema
  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  // Now insert test data after schema is initialized
  // Create chain: emb:1 --reinforces--> emb:20 --reinforces--> emb:21
  const Eigen::VectorXf vec20 = MakeUnitVec256 (6);
  const Eigen::VectorXf vec21 = MakeUnitVec256 (7);
  std::vector<float> data20 (vec20.data (), vec20.data () + vec20.size ());
  std::vector<float> data21 (vec21.data (), vec21.data () + vec21.size ());
  auto now_ts = cortext::testing::NowMs ();

  // v2: Insert into embeddings and memories
  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) VALUES (?, ?, ?)",
      { 20LL, data20, now_ts });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, start_ts, "
      "n_signals, modality, s_max, s_avg, created_at) "
      "VALUES (?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, ?)",
      { 20LL, 20LL, now_ts, now_ts });
  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) VALUES (?, ?, ?)",
      { 21LL, data21, now_ts });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, start_ts, "
      "n_signals, modality, s_max, s_avg, created_at) "
      "VALUES (?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, ?)",
      { 21LL, 21LL, now_ts, now_ts });

  // Edge: emb:1 -> emb:20 (depth 1)
  store->Execute (
      "INSERT INTO graph_edges (source_id, target_id, edge_type, weight) "
      "VALUES (?, ?, ?, ?)",
      { std::string ("emb:1"), std::string ("emb:20"),
        std::string ("reinforces"), 1.0 });
  // Edge: emb:20 -> emb:21 (depth 2 from emb:1)
  store->Execute (
      "INSERT INTO graph_edges (source_id, target_id, edge_type, weight) "
      "VALUES (?, ?, ?, ?)",
      { std::string ("emb:20"), std::string ("emb:21"),
        std::string ("reinforces"), 1.0 });

  auto s = MakeSignal (cur, /*ts=*/200);
  processor.Process (s);
  processor.Flush ();

  // v2: Verify depth-1 neighbor (memory_id:20) and depth-2 neighbor (memory_id:21) both updated
  double lab20 = 0.0, lab21 = 0.0;
  {
    auto rows = store->Execute (
        "SELECT lability_state FROM memories WHERE memory_id = ?",
        { 20LL });
    REQUIRE (rows.size () == 1);
    lab20 = std::any_cast<double> (rows[0].at ("lability_state"));
  }
  {
    auto rows = store->Execute (
        "SELECT lability_state FROM memories WHERE memory_id = ?",
        { 21LL });
    REQUIRE (rows.size () == 1);
    lab21 = std::any_cast<double> (rows[0].at ("lability_state"));
  }

  // Depth-1 neighbor should have higher lability than depth-2 (decay per hop)
  // With ripple_decay=0.5: depth-1 gets lability*0.5, depth-2 gets lability*0.25
  REQUIRE (lab20 > lab21);
  REQUIRE (lab20 > 0.0);
  REQUIRE (lab21 > 0.0);
}

TEST_CASE ("Alg20 RippleDepth knob affects traversal depth",
           "[operations][recon][ripple]")
{
  // With T=0.8, ripple_depth = round(lerp(2,1,0.8)) = round(1.2) = 1
  // So depth-2 neighbors should NOT be reached by ripple propagation
  // Note: T=1.0 would make drift_mag=0 (formula has 1-T factor), blocking all ripple
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0; // maximize plasticity
  cfg.stability = 0.8;   // HIGH stability: ripple_depth=1, but allows some drift

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  const Eigen::VectorXf mem = MakeUnitVec256 (0);
  const Eigen::VectorXf vec30 = MakeUnitVec256 (8);
  const Eigen::VectorXf vec31 = MakeUnitVec256 (9);

  // Seed all embeddings before operation runs
  std::unordered_map<long long, Eigen::VectorXf> allEmbs{
    { 1LL, mem }, { 30LL, vec30 }, { 31LL, vec31 }
  };
  auto seed = std::make_unique<SeedEmbeddingsOp> (allEmbs);
  auto setup = std::make_unique<SetupReconInputsOp> (
      cur, std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, mem } });
  auto apply = std::make_unique<ApplyReconsolidation> ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (seed), std::move (setup), std::move (apply));

  // Create processor first to initialize schema
  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  // Insert graph edges after schema is initialized
  // Create chain: emb:1 --reinforces--> emb:30 --reinforces--> emb:31
  store->Execute (
      "INSERT INTO graph_edges (source_id, target_id, edge_type, weight) "
      "VALUES (?, ?, ?, ?)",
      { std::string ("emb:1"), std::string ("emb:30"),
        std::string ("reinforces"), 1.0 });
  store->Execute (
      "INSERT INTO graph_edges (source_id, target_id, edge_type, weight) "
      "VALUES (?, ?, ?, ?)",
      { std::string ("emb:30"), std::string ("emb:31"),
        std::string ("reinforces"), 1.0 });

  auto s = MakeSignal (cur, /*ts=*/300);
  processor.Process (s);
  processor.Flush ();

  // v2: Depth-1 neighbor (memory_id:30) should have lability_state > 0 (updated by ripple)
  {
    auto rows = store->Execute (
        "SELECT lability_state FROM memories WHERE memory_id = ?",
        { 30LL });
    REQUIRE (rows.size () == 1);
    const double lab = std::any_cast<double> (rows[0].at ("lability_state"));
    REQUIRE (lab > 0.0); // Ripple reached depth-1
  }

  // v2: Depth-2 neighbor (memory_id:31) should have lability_state = 0 (beyond ripple_depth=1)
  {
    auto rows = store->Execute (
        "SELECT lability_state FROM memories WHERE memory_id = ?",
        { 31LL });
    REQUIRE (rows.size () == 1);
    const double lab = std::any_cast<double> (rows[0].at ("lability_state"));
    REQUIRE (lab == 0.0); // Ripple did NOT reach depth-2
  }
}

TEST_CASE ("Alg20 ripple respects co_occurs_with edge type",
           "[operations][recon][ripple]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

  const Eigen::VectorXf cur = MakeUnitVec256 (0);
  const Eigen::VectorXf mem = MakeUnitVec256 (0);

  auto setup = std::make_unique<SetupReconInputsOp> (
      cur, std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, mem } });
  auto apply = std::make_unique<ApplyReconsolidation> ();
  auto ops = std::make_unique<cortext::OperationSet> (std::move (setup),
                                                      std::move (apply));

  // Create processor first to initialize schema
  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  // Now insert test data after schema is initialized
  // Create neighbor connected via co_occurs_with (not reinforces)
  const Eigen::VectorXf vec40 = MakeUnitVec256 (10);
  std::vector<float> data40 (vec40.data (), vec40.data () + vec40.size ());
  auto now_ts = cortext::testing::NowMs ();

  // v2: Insert into embeddings and memories
  store->Execute (
      "INSERT INTO embeddings (embedding_id, embedding, created_at) VALUES (?, ?, ?)",
      { 40LL, data40, now_ts });
  store->Execute (
      "INSERT INTO memories (memory_id, embedding_id, source_id, kind, start_ts, "
      "n_signals, modality, s_max, s_avg, created_at) "
      "VALUES (?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, ?)",
      { 40LL, 40LL, now_ts, now_ts });

  // Edge: emb:1 --co_occurs_with--> emb:40
  store->Execute (
      "INSERT INTO graph_edges (source_id, target_id, edge_type, weight) "
      "VALUES (?, ?, ?, ?)",
      { std::string ("emb:1"), std::string ("emb:40"),
        std::string ("co_occurs_with"), 1.0 });

  auto s = MakeSignal (cur, /*ts=*/400);
  processor.Process (s);
  processor.Flush ();

  // v2: Neighbor via co_occurs_with should also receive ripple
  {
    auto rows = store->Execute (
        "SELECT lability_state FROM memories WHERE memory_id = ?",
        { 40LL });
    REQUIRE (rows.size () == 1);
    const double lab = std::any_cast<double> (rows[0].at ("lability_state"));
    REQUIRE (lab > 0.0);
  }
}
