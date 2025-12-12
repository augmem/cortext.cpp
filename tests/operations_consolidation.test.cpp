// tests/operations_consolidation.test.cpp
#include <Eigen/Dense>
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/consolidation.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
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
  Execute (OperationContext &ctx) const override
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

} // namespace

TEST_CASE ("Alg28 rate trigger starts when idle",
           "[operations][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  const uint64_t now_ts = 10'000ULL; const int idle_required
      = core::IdleRequiredSeconds (cfg.stability);
  const uint64_t last_ret = now_ts - static_cast<uint64_t> (idle_required);

  auto setup = std::make_unique<SetupConsolidationInputsOp> (
      /*tokens_in_flight=*/0,
      /*queue_depth=*/0,
      /*m_rate=*/0.8,
      /*rate_target=*/2.0,
      /*last_retrieval_ts=*/last_ret,
      /*last_consolidation_ts=*/std::nullopt);
  auto eval = std::make_unique<EvaluateConsolidation> ();
  auto ops
      = std::make_unique<OperationSet> (std::move (setup), std::move (eval));
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (now_ts));
  processor.Flush ();

  auto rows
      = store->Execute ("SELECT reason, action, m_rate, rate_target, idle_for "
                        "FROM consolidation_events ORDER BY id DESC LIMIT 1");
  REQUIRE (rows.size () == 1);
  const auto &r = rows[0];
  REQUIRE (std::any_cast<std::string> (r.at ("reason")) == "rate");
  REQUIRE (std::any_cast<std::string> (r.at ("action")) == "start");
  REQUIRE (std::any_cast<double> (r.at ("m_rate"))
           == Catch::Approx (0.8).margin (1e-6));
  REQUIRE (std::any_cast<double> (r.at ("rate_target"))
           == Catch::Approx (2.0).margin (1e-6));
  REQUIRE (
      std::any_cast<double> (r.at ("idle_for"))
      >= Catch::Approx (static_cast<double> (idle_required)).margin (1e-6));
}

TEST_CASE ("Alg28 rate trigger defers when busy",
           "[operations][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  const uint64_t now_ts = 20'000ULL; const int idle_required
      = core::IdleRequiredSeconds (cfg.stability);
  const uint64_t last_ret = now_ts - static_cast<uint64_t> (idle_required);

  auto setup = std::make_unique<SetupConsolidationInputsOp> (
      /*tokens_in_flight=*/3,
      /*queue_depth=*/5,
      /*m_rate=*/0.8,
      /*rate_target=*/2.0,
      /*last_retrieval_ts=*/last_ret,
      /*last_consolidation_ts=*/std::nullopt);
  auto eval = std::make_unique<EvaluateConsolidation> ();
  auto ops
      = std::make_unique<OperationSet> (std::move (setup), std::move (eval));
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (now_ts));
  processor.Flush ();

  auto rows = store->Execute ("SELECT reason, action FROM "
                              "consolidation_events ORDER BY id DESC LIMIT 1");
  REQUIRE (rows.size () == 1);
  const auto &r = rows[0];
  REQUIRE (std::any_cast<std::string> (r.at ("reason")) == "rate");
  REQUIRE (std::any_cast<std::string> (r.at ("action")) == "defer");
}

TEST_CASE ("Alg28 interval trigger starts when elapsed exceeds interval",
           "[operations][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  const uint64_t now_ts = 30'000ULL; const int interval
      = core::ConsolidationIntervalSeconds (cfg.stability);
  const int idle_required = core::IdleRequiredSeconds (cfg.stability);
  const uint64_t last_cons = now_ts - static_cast<uint64_t> (interval + 1);
  const uint64_t last_ret
      = now_ts - static_cast<uint64_t> (idle_required + 1); // ensure idle OK

  auto setup = std::make_unique<SetupConsolidationInputsOp> (
      /*tokens_in_flight=*/0,
      /*queue_depth=*/0,
      /*m_rate=*/3.0,
      /*rate_target=*/2.0,
      /*last_retrieval_ts=*/last_ret,
      /*last_consolidation_ts=*/last_cons);
  auto eval = std::make_unique<EvaluateConsolidation> ();
  auto ops
      = std::make_unique<OperationSet> (std::move (setup), std::move (eval));
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (now_ts));
  processor.Flush ();

  auto rows = store->Execute ("SELECT reason, action FROM "
                              "consolidation_events ORDER BY id DESC LIMIT 1");
  REQUIRE (rows.size () == 1);
  const auto &r = rows[0];
  REQUIRE (std::any_cast<std::string> (r.at ("reason")) == "interval");
  REQUIRE (std::any_cast<std::string> (r.at ("action")) == "start");
}

TEST_CASE ("Alg28 capacity trigger starts when db_size exceeds threshold",
           "[operations][consolidation][capacity]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Use low stability to keep threshold small-ish.
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.0;

  // Create embeddings table and insert just over threshold.
  store->Execute ("CREATE TABLE IF NOT EXISTS embeddings ("
                  "embedding_id INTEGER PRIMARY KEY,"
                  "embedding BLOB"
                  ");");
  const long long threshold = core::ConsolidationThresholdCount (cfg.stability);
  const long long want = threshold + 1;
  const std::vector<float> emb = { 1.0f, 0.0f };
  for (long long i = 1; i <= want; ++i)
    {
      store->Execute ("INSERT INTO embeddings(embedding_id, embedding) VALUES "
                      "(?,?)",
                      { i, emb });
    }

  const uint64_t now_ts = 70'000ULL;
  const int idle_required = core::IdleRequiredSeconds (cfg.stability);
  const uint64_t last_ret = now_ts - static_cast<uint64_t> (idle_required + 1);

  auto setup = std::make_unique<SetupConsolidationInputsOp> (
      /*tokens_in_flight=*/0,
      /*queue_depth=*/0,
      /*m_rate=*/3.0,
      /*rate_target=*/2.0,
      /*last_retrieval_ts=*/last_ret,
      /*last_consolidation_ts=*/std::nullopt);
  auto eval = std::make_unique<EvaluateConsolidation> ();
  auto ops
      = std::make_unique<OperationSet> (std::move (setup), std::move (eval));
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (now_ts));
  processor.Flush ();

  auto rows = store->Execute ("SELECT reason, action FROM "
                              "consolidation_events ORDER BY id DESC LIMIT 1");
  REQUIRE (rows.size () == 1);
  const auto &r = rows[0];
  REQUIRE (std::any_cast<std::string> (r.at ("reason")) == "capacity");
  REQUIRE (std::any_cast<std::string> (r.at ("action")) == "start");
}

TEST_CASE ("Alg28 no trigger emits no event table rows",
           "[operations][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  const uint64_t now_ts = 40'000ULL; const uint64_t last_cons
      = now_ts; // no interval trigger

  auto setup = std::make_unique<SetupConsolidationInputsOp> (
      /*tokens_in_flight=*/0,
      /*queue_depth=*/0,
      /*m_rate=*/2.0,
      /*rate_target=*/2.0,
      /*last_retrieval_ts=*/now_ts,
      /*last_consolidation_ts=*/last_cons);
  auto eval = std::make_unique<EvaluateConsolidation> ();
  auto ops
      = std::make_unique<OperationSet> (std::move (setup), std::move (eval));
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (now_ts));
  processor.Flush ();

  // Ensure table exists then confirm zero rows; EvaluateConsolidation
  // doesn't create the table when no triggers fire.
  store->Execute (
      "CREATE TABLE IF NOT EXISTS consolidation_events ("
      "id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, reason TEXT, action "
      "TEXT, m_rate REAL, rate_target REAL, idle_for REAL, "
      "tokens_in_flight INTEGER, retrieval_queue_depth INTEGER)");
  auto rows
      = store->Execute ("SELECT COUNT(*) AS c FROM consolidation_events");
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 0LL);
}

TEST_CASE ("Alg29 scores and marks low-strength candidates",
           "[operations][consolidation][alg29]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Knobs: high T raises floor via periphery cutoff (~0.25 at T=1.0)
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 1.0; // T=1.0 → floor≈0.25

  // Seed memory_feedback with two rows: one below, one above the floor.
  store->Execute ("CREATE TABLE IF NOT EXISTS memory_feedback ("
                  "embedding_id INTEGER PRIMARY KEY,"
                  "retrieved_count INTEGER NOT NULL DEFAULT 0,"
                  "used_count INTEGER NOT NULL DEFAULT 0,"
                  "contextual_gain REAL NOT NULL DEFAULT 0.0,"
                  "use_frequency REAL NOT NULL DEFAULT 0.0,"
                  "strength REAL NOT NULL DEFAULT 1.0)");
  store->Execute (
      "INSERT INTO memory_feedback(embedding_id, strength) VALUES(?,?)",
      { 1LL, 0.10 });
  store->Execute (
      "INSERT INTO memory_feedback(embedding_id, strength) VALUES(?,?)",
      { 2LL, 0.80 });

  const uint64_t now_ts = 50'000ULL; auto score_op
      = std::make_unique<ScoreConsolidation> ();
  auto ops = std::make_unique<OperationSet> (std::move (score_op));
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (now_ts));
  processor.Flush ();

  // Only id=1 should be marked as candidate with score ~= 0.10 (T*strength).
  auto rows = store->Execute (
      "SELECT embedding_id, score, created_at, reason "
      "FROM consolidation_candidates ORDER BY embedding_id ASC");
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<long long> (rows[0].at ("embedding_id")) == 1LL);
  REQUIRE (std::any_cast<double> (rows[0].at ("score"))
           == Catch::Approx (0.10).margin (1e-6));
  REQUIRE (std::any_cast<long long> (rows[0].at ("created_at"))
           == static_cast<long long> (now_ts));
  REQUIRE (std::any_cast<std::string> (rows[0].at ("reason"))
           == std::string ("score_below_floor"));
}

TEST_CASE ("Alg29 is idempotent on repeated runs",
           "[operations][consolidation][alg29]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 1.0;

  store->Execute ("CREATE TABLE IF NOT EXISTS memory_feedback ("
                  "embedding_id INTEGER PRIMARY KEY,"
                  "retrieved_count INTEGER NOT NULL DEFAULT 0,"
                  "used_count INTEGER NOT NULL DEFAULT 0,"
                  "contextual_gain REAL NOT NULL DEFAULT 0.0,"
                  "use_frequency REAL NOT NULL DEFAULT 0.0,"
                  "strength REAL NOT NULL DEFAULT 1.0)");
  store->Execute (
      "INSERT INTO memory_feedback(embedding_id, strength) VALUES(?,?)",
      { 3LL, 0.15 }); // below floor at T=1.0

  auto ops = std::make_unique<OperationSet> (
      std::make_unique<ScoreConsolidation> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (60'000ULL));
  processor.Flush ();
  processor.Process (MakeSignal (60'100ULL));
  processor.Flush ();

  auto rows = store->Execute (
      "SELECT COUNT(*) AS c FROM consolidation_candidates WHERE embedding_id=?",
      { 3LL });
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 1LL);
}
