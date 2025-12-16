// tests/operations_reconsolidation.test.cpp
#include <Eigen/Dense>
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
  Execute (OperationContext &ctx) const override
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
  Execute (OperationContext &ctx) const override
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

  auto setup = std::make_unique<SetupReconInputsOp> (
      cur, std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, mem } });
  auto apply = std::make_unique<ApplyReconsolidation> ();
  auto ops = std::make_unique<cortext::OperationSet> (std::move (setup),
                                                      std::move (apply));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto s = MakeSignal (cur, /*ts=*/100);
  processor.Process (s);
  processor.Flush ();

  // Embedding row created in vec_embeddings
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM vec_embeddings WHERE embedding_id = ?",
        { 1LL });
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("cnt")) == 1LL);
  }
  // Lability fields updated
  {
    auto rows = store->Execute (
        "SELECT em.lability_state, mf.lability_ts FROM "
        "embeddings_meta em "
        "JOIN memory_feedback mf ON em.embedding_id = mf.embedding_id "
        "WHERE em.embedding_id = ?",
        { 1LL });
    REQUIRE (rows.size () == 1);
    const double lab = std::any_cast<double> (rows[0].at ("lability_state"));
    const long long ts = std::any_cast<long long> (rows[0].at ("lability_ts"));
    REQUIRE (lab > 0.0);
    REQUIRE (ts == 100LL);
  }
}

TEST_CASE ("Alg20 no drift when S=0: no embedding row; lability updated",
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

  auto setup = std::make_unique<SetupReconInputsOp> (
      cur, std::unordered_map<long long, Eigen::VectorXf>{ { 2LL, mem } });
  auto apply = std::make_unique<ApplyReconsolidation> ();
  auto ops = std::make_unique<cortext::OperationSet> (std::move (setup),
                                                      std::move (apply));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto s = MakeSignal (cur, /*ts=*/42);
  processor.Process (s);
  processor.Flush ();

  // No vec_embeddings row (drift skipped when S=0)
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM vec_embeddings WHERE embedding_id = ?",
        { 2LL });
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("cnt")) == 0LL);
  }
  // Lability fields updated
  {
    auto rows = store->Execute (
        "SELECT em.lability_state, mf.lability_ts FROM "
        "embeddings_meta em "
        "JOIN memory_feedback mf ON em.embedding_id = mf.embedding_id "
        "WHERE em.embedding_id = ?",
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
  store->Execute (
      "INSERT INTO vec_embeddings (embedding_id, embedding) VALUES (?, ?)",
      { 10LL, neighbor_data });

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

  std::vector<BufferedWriteInstruction> write_buffer;

  // Create OperationContext with the store pointer
  OperationContext ctx (s, pctx, cfg, write_buffer, store.get ());

  // Verify store is set in context
  REQUIRE (ctx.GetStore () != nullptr);

  // Set up the retrieved embeddings
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, mem } });

  // Execute reconsolidation
  ApplyReconsolidation recon_op;
  recon_op.Execute (ctx);

  // Execute buffered writes
  for (const auto &instr : write_buffer)
    {
      store->Execute (instr.query, instr.params);
    }

  // Verify primary reconsolidation worked
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM embeddings_meta WHERE embedding_id = ?",
        { 1LL });
    REQUIRE (rows.size () == 1);
    auto cnt = std::any_cast<long long> (rows[0].at ("cnt"));
    REQUIRE (cnt == 1LL);
  }

  // Verify neighbor (emb:10) received ripple update
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM embeddings_meta WHERE embedding_id = ?",
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

  store->Execute (
      "INSERT INTO vec_embeddings (embedding_id, embedding) VALUES (?, ?)",
      { 20LL, data20 });
  store->Execute (
      "INSERT INTO vec_embeddings (embedding_id, embedding) VALUES (?, ?)",
      { 21LL, data21 });

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

  // Verify depth-1 neighbor (emb:20) and depth-2 neighbor (emb:21) both updated
  double lab20 = 0.0, lab21 = 0.0;
  {
    auto rows = store->Execute (
        "SELECT lability_state FROM embeddings_meta WHERE embedding_id = ?",
        { 20LL });
    REQUIRE (rows.size () == 1);
    lab20 = std::any_cast<double> (rows[0].at ("lability_state"));
  }
  {
    auto rows = store->Execute (
        "SELECT lability_state FROM embeddings_meta WHERE embedding_id = ?",
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
  // So depth-2 neighbors should NOT be reached
  // Note: T=1.0 would make drift_mag=0 (formula has 1-T factor), blocking all ripple
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0; // maximize plasticity
  cfg.stability = 0.8;   // HIGH stability: ripple_depth=1, but allows some drift

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
  // Create chain: emb:1 --reinforces--> emb:30 --reinforces--> emb:31
  const Eigen::VectorXf vec30 = MakeUnitVec256 (8);
  const Eigen::VectorXf vec31 = MakeUnitVec256 (9);
  std::vector<float> data30 (vec30.data (), vec30.data () + vec30.size ());
  std::vector<float> data31 (vec31.data (), vec31.data () + vec31.size ());

  store->Execute (
      "INSERT INTO vec_embeddings (embedding_id, embedding) VALUES (?, ?)",
      { 30LL, data30 });
  store->Execute (
      "INSERT INTO vec_embeddings (embedding_id, embedding) VALUES (?, ?)",
      { 31LL, data31 });

  // Edge: emb:1 -> emb:30 (depth 1)
  store->Execute (
      "INSERT INTO graph_edges (source_id, target_id, edge_type, weight) "
      "VALUES (?, ?, ?, ?)",
      { std::string ("emb:1"), std::string ("emb:30"),
        std::string ("reinforces"), 1.0 });
  // Edge: emb:30 -> emb:31 (depth 2 from emb:1)
  store->Execute (
      "INSERT INTO graph_edges (source_id, target_id, edge_type, weight) "
      "VALUES (?, ?, ?, ?)",
      { std::string ("emb:30"), std::string ("emb:31"),
        std::string ("reinforces"), 1.0 });

  auto s = MakeSignal (cur, /*ts=*/300);
  processor.Process (s);
  processor.Flush ();

  // Depth-1 neighbor (emb:30) should be updated
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM embeddings_meta WHERE embedding_id = ?",
        { 30LL });
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("cnt")) == 1LL);
  }

  // Depth-2 neighbor (emb:31) should NOT be updated (beyond ripple_depth=1)
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM embeddings_meta WHERE embedding_id = ?",
        { 31LL });
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("cnt")) == 0LL);
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

  store->Execute (
      "INSERT INTO vec_embeddings (embedding_id, embedding) VALUES (?, ?)",
      { 40LL, data40 });

  // Edge: emb:1 --co_occurs_with--> emb:40
  store->Execute (
      "INSERT INTO graph_edges (source_id, target_id, edge_type, weight) "
      "VALUES (?, ?, ?, ?)",
      { std::string ("emb:1"), std::string ("emb:40"),
        std::string ("co_occurs_with"), 1.0 });

  auto s = MakeSignal (cur, /*ts=*/400);
  processor.Process (s);
  processor.Flush ();

  // Neighbor via co_occurs_with should also receive ripple
  {
    auto rows = store->Execute (
        "SELECT lability_state FROM embeddings_meta WHERE embedding_id = ?",
        { 40LL });
    REQUIRE (rows.size () == 1);
    const double lab = std::any_cast<double> (rows[0].at ("lability_state"));
    REQUIRE (lab > 0.0);
  }
}
