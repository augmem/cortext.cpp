// tests/operations_reconsolidation.test.cpp
#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/reconsolidation.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
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

  // Embedding row created
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM embeddings WHERE embedding_id = ?",
        { 1LL });
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("cnt")) == 1LL);
  }
  // Lability fields updated
  {
    auto rows = store->Execute ("SELECT lability_state, lability_ts FROM "
                                "memory_feedback WHERE embedding_id = ?",
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

  // No embeddings row
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM embeddings WHERE embedding_id = ?",
        { 2LL });
    REQUIRE (rows.size () == 1);
    REQUIRE (std::any_cast<long long> (rows[0].at ("cnt")) == 0LL);
  }
  // Lability fields updated
  {
    auto rows = store->Execute ("SELECT lability_state, lability_ts FROM "
                                "memory_feedback WHERE embedding_id = ?",
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
