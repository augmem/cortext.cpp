// tests/operations_consolidation.test.cpp
#include <Eigen/Dense>
#include <any>
#include <iostream>
#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/consolidation.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/schema.hpp>
#include <string>

using namespace cortext;
using cortext::operations::EvaluateConsolidation;
using cortext::operations::ScoreConsolidation;

namespace
{

struct SetupConsolidationInputsOp : IOperation
{
  SetupConsolidationInputsOp (int tokens_in_flight, int queue_depth,
                              double m_rate, double rate_target,
                              uint64_t last_retrieval_ts,
                              std::optional<uint64_t> last_consolidation_ts)
      : tokens_in_flight_ (tokens_in_flight), queue_depth_ (queue_depth),
        m_rate_ (m_rate), rate_target_ (rate_target),
        last_retrieval_ts_ (last_retrieval_ts),
        last_consolidation_ts_ (last_consolidation_ts)
  {
  }
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetTokensInFlight (tokens_in_flight_);
    ctx.SetRetrievalQueueDepth (queue_depth_);
    auto &p = ctx.GetProcessorContext ();
    p.m_rate = m_rate_;
    p.rate_target = rate_target_;
    p.last_retrieval_ts = last_retrieval_ts_;
    if (last_consolidation_ts_.has_value ())
      {
        p.last_consolidation_ts = *last_consolidation_ts_;
      }
  }
  int tokens_in_flight_;
  int queue_depth_;
  double m_rate_;
  double rate_target_;
  uint64_t last_retrieval_ts_;
  std::optional<uint64_t> last_consolidation_ts_;
};

static Signal
MakeSignal (uint64_t ts)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

static Signal
MakeConsolidationSignal (uint64_t ts)
{
  auto s = MakeSignal (ts);
  s.source_id = "test/consolidation";
  s.force_consolidation = true;
  return s;
}

} // namespace

// Helper op to verify consolidation flag and timestamp after execution.
struct AssertConsolidationStartedOp : IOperation
{
  AssertConsolidationStartedOp (uint64_t expected_ts)
      : expected_ts_ (expected_ts)
  {
  }
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    REQUIRE (ctx.GetConsolidationShouldStart () == true);
    auto &p = ctx.GetProcessorContext ();
    REQUIRE (p.last_consolidation_ts == expected_ts_);
  }
  uint64_t expected_ts_;
};

struct AssertConsolidationNotStartedOp : IOperation
{
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    REQUIRE (ctx.GetConsolidationShouldStart () == false);
  }
};

TEST_CASE ("Alg28 rate trigger starts when idle",
           "[operations][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0; // min_cluster_size = 3, so three below-floor rows avoid fallback
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Compute the knob-derived rate target
  // rate_consolidate = (1/interval) * (0.3+0.7T) * (1-0.5S)
  // At T=0.5, S=0.5: rate ≈ 0.00025
  const double rate_target = core::ConsolidationRate (cfg.stability, cfg.sensitivity);

  // Timestamps are in milliseconds
  const uint64_t now_ts = 100'000ULL; // 100 seconds in ms
  const int idle_required_s = core::IdleRequiredSeconds (cfg.stability);
  const uint64_t last_ret = now_ts - static_cast<uint64_t> (idle_required_s) * 1000ULL;

  // Set m_rate below half of the knob-derived rate to trigger
  const double m_rate = rate_target * 0.4; // Below 50% threshold

  auto setup = std::make_unique<SetupConsolidationInputsOp> (
      /*tokens_in_flight=*/0,
      /*queue_depth=*/0,
      /*m_rate=*/m_rate,
      /*rate_target=*/rate_target, // Not used anymore, but kept for setup
      /*last_retrieval_ts=*/last_ret,
      /*last_consolidation_ts=*/std::nullopt);
  auto eval = std::make_unique<EvaluateConsolidation> ();
  auto assert_op = std::make_unique<AssertConsolidationStartedOp> (now_ts);
  auto ops = std::make_unique<DynamicOperationSet> (std::move (setup),
                                              std::move (eval),
                                              std::move (assert_op));
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeConsolidationSignal (now_ts));
  processor.Flush ();
}

TEST_CASE ("Alg28 explicit consolidation signal starts even when busy",
           "[operations][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Compute the knob-derived rate target
  const double rate_target = core::ConsolidationRate (cfg.stability, cfg.sensitivity);

  // Timestamps are in milliseconds
  const uint64_t now_ts = 100'000ULL; // 100 seconds in ms
  const int idle_required_s = core::IdleRequiredSeconds (cfg.stability);
  const uint64_t last_ret = now_ts - static_cast<uint64_t> (idle_required_s) * 1000ULL;

  // Set m_rate below threshold to trigger rate check, but busy so should defer
  const double m_rate = rate_target * 0.4;

  auto setup = std::make_unique<SetupConsolidationInputsOp> (
      /*tokens_in_flight=*/3,
      /*queue_depth=*/5,
      /*m_rate=*/m_rate,
      /*rate_target=*/rate_target,
      /*last_retrieval_ts=*/last_ret,
      /*last_consolidation_ts=*/std::nullopt);
  auto eval = std::make_unique<EvaluateConsolidation> ();
  auto assert_op = std::make_unique<AssertConsolidationStartedOp> (now_ts);
  auto ops = std::make_unique<DynamicOperationSet> (std::move (setup),
                                              std::move (eval),
                                              std::move (assert_op));
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeConsolidationSignal (now_ts));
  processor.Flush ();
}

TEST_CASE ("Alg28 interval trigger starts when elapsed exceeds interval",
           "[operations][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Timestamps in milliseconds, interval/idle in seconds
  const int interval_s = core::ConsolidationIntervalSeconds (cfg.stability);
  const int idle_required_s = core::IdleRequiredSeconds (cfg.stability);
  const uint64_t now_ts = static_cast<uint64_t>(interval_s + 10) * 1000ULL; // enough time passed
  const uint64_t last_cons = now_ts - static_cast<uint64_t> (interval_s + 1) * 1000ULL;
  const uint64_t last_ret
      = now_ts - static_cast<uint64_t> (idle_required_s + 1) * 1000ULL; // ensure idle OK

  auto setup = std::make_unique<SetupConsolidationInputsOp> (
      /*tokens_in_flight=*/0,
      /*queue_depth=*/0,
      /*m_rate=*/3.0,
      /*rate_target=*/2.0,
      /*last_retrieval_ts=*/last_ret,
      /*last_consolidation_ts=*/last_cons);
  auto eval = std::make_unique<EvaluateConsolidation> ();
  auto assert_op = std::make_unique<AssertConsolidationStartedOp> (now_ts);
  auto ops = std::make_unique<DynamicOperationSet> (std::move (setup),
                                              std::move (eval),
                                              std::move (assert_op));
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeConsolidationSignal (now_ts));
  processor.Flush ();
}

// Helper op that seeds embeddings and memories with test data.
struct SeedVecEmbeddingsOp : IOperation
{
  SeedVecEmbeddingsOp (long long count) : count_ (count) {}
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto *store = ctx.GetStore ();
    std::vector<float> emb (256, 0.0f);
    emb[0] = 1.0f;
    for (long long i = 1; i <= count_; ++i)
      {
        // v2: Insert into embeddings (minimal vec0 table)
        store->Execute (
            "INSERT INTO embeddings(embedding_id, embedding, created_at) VALUES (?,?,?)",
            { i, emb, 0LL });
        // v2: Insert into memories (capacity check counts memories)
        store->Execute (
            "INSERT INTO memories(memory_id, embedding_id, source_id, kind, start_ts, "
            "n_signals, modality, s_max, s_avg, strength, created_at) "
            "VALUES (?, ?, 'test', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, 1.0, 0)",
            { i, i });
      }
  }
  long long count_;
};

TEST_CASE ("Alg28 capacity trigger starts when db_size exceeds threshold",
           "[operations][consolidation][capacity]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Use low stability to keep threshold small-ish.
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.0;

  const long long threshold = core::ConsolidationThresholdCount (cfg.stability);
  const long long want = threshold + 1;

  // Timestamps in milliseconds, idle_required in seconds
  const int idle_required_s = core::IdleRequiredSeconds (cfg.stability);
  const uint64_t now_ts = static_cast<uint64_t>(idle_required_s + 10) * 1000ULL;
  const uint64_t last_ret = now_ts - static_cast<uint64_t> (idle_required_s + 1) * 1000ULL;

  // Seed data first, then set up consolidation inputs, then eval.
  auto seed = std::make_unique<SeedVecEmbeddingsOp> (want);
  auto setup = std::make_unique<SetupConsolidationInputsOp> (
      /*tokens_in_flight=*/0,
      /*queue_depth=*/0,
      /*m_rate=*/3.0,
      /*rate_target=*/2.0,
      /*last_retrieval_ts=*/last_ret,
      /*last_consolidation_ts=*/std::nullopt);
  auto eval = std::make_unique<EvaluateConsolidation> ();
  auto assert_op = std::make_unique<AssertConsolidationStartedOp> (now_ts);
  auto ops = std::make_unique<DynamicOperationSet> (std::move (seed),
                                              std::move (setup),
                                              std::move (eval),
                                              std::move (assert_op));
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeConsolidationSignal (now_ts));
  processor.Flush ();
}

TEST_CASE ("Alg28 no trigger does not set start flag",
           "[operations][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  const uint64_t now_ts = 40'000ULL;
  const uint64_t last_cons = now_ts; // no interval trigger

  auto setup = std::make_unique<SetupConsolidationInputsOp> (
      /*tokens_in_flight=*/0,
      /*queue_depth=*/0,
      /*m_rate=*/2.0,
      /*rate_target=*/2.0,
      /*last_retrieval_ts=*/now_ts,
      /*last_consolidation_ts=*/last_cons);
  auto eval = std::make_unique<EvaluateConsolidation> ();
  auto assert_op = std::make_unique<AssertConsolidationNotStartedOp> ();
  auto ops = std::make_unique<DynamicOperationSet> (std::move (setup),
                                              std::move (eval),
                                              std::move (assert_op));
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (now_ts));
  processor.Flush ();
}

TEST_CASE ("ScoreConsolidation identifies low-strength candidates",
           "[operations][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  cortext::store::ApplyMigrations (*store);

  // Knobs: high T raises floor via periphery cutoff; F=0 keeps
  // min_cluster_size at 3 so the three below-floor rows avoid fallback.
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 1.0; // T=1.0 -> floor=0.20

  // Initialize store and context
  Signal dummy;
  dummy.timestamp = 50'000ULL;
  dummy.source_id = "test/consolidation";
  dummy.force_consolidation = true;
  dummy.embedding = Eigen::VectorXf::Zero (4); // Not used by op directly
  ProcessorContext p_ctx;
  OperationContext ctx (dummy, p_ctx, cfg, store.get ());

  // v2: Insert memories. IDs 1-3 are below the floor; ID 4 is above.
  // score = T*strength - F*redundancy + S*connectivity + T*stability
  // For T=1.0, S=0.5 and floor=0.20, strength is the effective score.
  for (long long id = 1; id <= 4; ++id)
    {
      std::vector<float> emb (256, 0.0f);
      emb[static_cast<size_t> (id - 1)] = 1.0f;
      const double strength = id == 4 ? 0.80 : 0.04 * id;
      store->Execute (
          "INSERT INTO embeddings(embedding_id, embedding, created_at) "
          "VALUES(?, ?, ?)",
          { id, emb, 0LL });
      store->Execute (
          "INSERT INTO memories(memory_id, embedding_id, source_id, kind, start_ts, "
          "n_signals, modality, s_max, s_avg, strength, stability, created_at) "
          "VALUES(?, ?, 'test', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, ?, ?, 0)",
          { id, id, strength, 0.0 });
    }

  // Run ScoreConsolidation
  ScoreConsolidation op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);

  // Verify candidates in context
  const auto &candidates = ctx.GetConsolidationCandidates ();
  REQUIRE (candidates.size () == 3);
  REQUIRE (candidates[0].embedding_id == 1LL);
  // score = T*strength = 1.0 * 0.04 = 0.04
  REQUIRE (candidates[0].score == Catch::Approx (0.04).margin (1e-6));
  REQUIRE (candidates[1].embedding_id == 2LL);
  REQUIRE (candidates[2].embedding_id == 3LL);
  // Verify embedding loaded correctly
  REQUIRE (candidates[0].embedding.size() == 256);
  REQUIRE (candidates[0].embedding(0) == Catch::Approx(1.0f));
}

TEST_CASE ("ScoreConsolidation forced mode broadens partial candidate sets",
           "[operations][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  cortext::store::ApplyMigrations (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;      // min_cluster_size = 3
  cfg.sensitivity = 0.5;
  cfg.stability = 1.0;  // score = strength, floor = 0.20

  Signal dummy;
  dummy.timestamp = 120'000ULL;
  dummy.source_id = "test/consolidation";
  dummy.force_consolidation = true;
  dummy.embedding = Eigen::VectorXf::Zero (4);
  ProcessorContext p_ctx;
  OperationContext ctx (dummy, p_ctx, cfg, store.get ());

  for (long long id = 1; id <= 6; ++id)
    {
      std::vector<float> emb (256, 0.0f);
      emb[static_cast<size_t> (id - 1)] = 1.0f;
      std::vector<unsigned char> blob = {
        static_cast<unsigned char> ('a' + id - 1)
      };
      const double strength = (id == 1) ? 0.10 : 0.40 + 0.05 * id;
      store->Execute (
          "INSERT INTO embeddings(embedding_id, embedding, created_at) "
          "VALUES(?, ?, ?)",
          { id, emb, 0LL });
      store->Execute (
          "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
          "start_ts, n_signals, modality, s_max, s_avg, strength, stability, "
          "redundancy, blob_id, created_at) "
          "VALUES(?, ?, 'test', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, "
          "?, 0.0, 0.0, ?, 0)",
          { id, id, strength, blob });
    }

  ScoreConsolidation op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);

  const auto &candidates = ctx.GetConsolidationCandidates ();
  REQUIRE (candidates.size () >= static_cast<size_t> (core::MinClusterSize (cfg.focus)));
  REQUIRE (candidates.size () == 6);
  REQUIRE (candidates[0].embedding_id == 1LL);
}

TEST_CASE ("ScoreConsolidation forced mode includes memories without blobs",
           "[operations][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  cortext::store::ApplyMigrations (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;      // min_cluster_size = 3
  cfg.sensitivity = 0.5;
  cfg.stability = 1.0;  // score = strength, floor = 0.20

  Signal dummy;
  dummy.timestamp = 130'000ULL;
  dummy.source_id = "test/consolidation";
  dummy.force_consolidation = true;
  dummy.embedding = Eigen::VectorXf::Zero (4);
  ProcessorContext p_ctx;
  OperationContext ctx (dummy, p_ctx, cfg, store.get ());

  for (long long id = 1; id <= 4; ++id)
    {
      std::vector<float> emb (256, 0.0f);
      emb[static_cast<size_t> (id - 1)] = 1.0f;
      const double strength = (id == 1) ? 0.10 : 0.45 + 0.05 * id;
      store->Execute (
          "INSERT INTO embeddings(embedding_id, embedding, created_at) "
          "VALUES(?, ?, ?)",
          { id, emb, 0LL });
      store->Execute (
          "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
          "start_ts, n_signals, modality, s_max, s_avg, strength, stability, "
          "redundancy, created_at) "
          "VALUES(?, ?, 'test', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, "
          "?, 0.0, 0.0, 0)",
          { id, id, strength });
    }

  ScoreConsolidation op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);

  const auto &candidates = ctx.GetConsolidationCandidates ();
  REQUIRE (candidates.size () >= static_cast<size_t> (core::MinClusterSize (cfg.focus)));
  REQUIRE (candidates.size () == 4);
  REQUIRE (candidates[0].embedding_id == 1LL);
}
