// tests/operations_consolidation.test.cpp
#include <Eigen/Dense>
#include <array>
#include <any>
#include <iostream>
#include <catch2/catch_approx.hpp>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/consolidation_state.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/consolidation.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/schema.hpp>
#include "../src/operations/consolidation_throughput_state_internal.hpp"
#include "../src/operations/sparse_retrieval_knobs_internal.hpp"
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

struct SetupConsolidationHintOp : IOperation
{
  SetupConsolidationHintOp (double floor_in, double peak_in,
                            double current_rate_in, int backlog_in)
      : floor (floor_in), peak (peak_in), current_rate (current_rate_in),
        backlog (backlog_in)
  {
  }

  double floor = 0.0;
  double peak = 0.0;
  double current_rate = 0.0;
  int backlog = 0;

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    pctx.m_rate = current_rate;
    pctx.memories_since_consolidation = backlog;
    operations::consolidation_throughput_state_internal::Reset (
        pctx, { floor, peak, true });
  }
};

SignalProcessor::Output
RunConsolidationHint (const SignalProcessor::Config &cfg,
                      const SetupConsolidationHintOp &setup,
                      std::string source_id = "test/source",
                      std::string modality = "text",
                      bool force_consolidation = false)
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<SetupConsolidationHintOp> (setup));
  SignalProcessor processor (cfg, store, std::move (ops));
  Signal signal;
  signal.embedding = Eigen::VectorXf::Ones (4);
  signal.timestamp = 1'000'000ULL;
  signal.source_id = std::move (source_id);
  signal.modality = std::move (modality);
  signal.force_consolidation = force_consolidation;
  return processor.Process (signal);
}

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

TEST_CASE ("Consolidation throughput trigger fraction is monotonic in F/S/T",
           "[operations][consolidation][hint]")
{
  namespace throughput
      = operations::consolidation_throughput_state_internal;
  REQUIRE (throughput::TriggerFraction (0.8, 0.5, 0.5)
           < throughput::TriggerFraction (0.2, 0.5, 0.5));
  REQUIRE (throughput::TriggerFraction (0.5, 0.8, 0.5)
           > throughput::TriggerFraction (0.5, 0.2, 0.5));
  REQUIRE (throughput::TriggerFraction (0.5, 0.5, 0.8)
           < throughput::TriggerFraction (0.5, 0.5, 0.2));
  REQUIRE (throughput::RequiredTriggerFraction (0.5, 0.5, 0.5)
           < throughput::TriggerFraction (0.5, 0.5, 0.5));
  REQUIRE (throughput::DriftRearmFraction (0.8, 0.5, 0.5)
           > throughput::DriftRearmFraction (0.2, 0.5, 0.5));
  REQUIRE (throughput::DriftRearmFraction (0.5, 0.8, 0.5)
           < throughput::DriftRearmFraction (0.5, 0.2, 0.5));
  REQUIRE (throughput::DriftRearmFraction (0.5, 0.5, 0.8)
           > throughput::DriftRearmFraction (0.5, 0.5, 0.2));
  REQUIRE (throughput::DriftRearmFraction (0.5, 0.5, 1.0) >= 0.30);
  REQUIRE (throughput::MinimumBacklog (0.0, 0.0, 0.0) == 64);
  REQUIRE (throughput::MinimumBacklog (0.5, 0.5, 0.5) == 128);
  REQUIRE (throughput::MinimumBacklog (1.0, 1.0, 1.0) == 192);
}

TEST_CASE ("Consolidation throughput observation tracks moving floor and peak",
           "[operations][consolidation][hint]")
{
  namespace throughput
      = operations::consolidation_throughput_state_internal;
  ProcessorContext pctx;
  throughput::Reset (pctx);
  throughput::Observe (pctx, 5.0, 0.5, 0.5, 0.5);
  auto state = throughput::Find (pctx);
  REQUIRE (state.floor == 5.0);
  REQUIRE (state.peak == 5.0);

  throughput::Observe (pctx, 10.0, 0.5, 0.5, 0.5);
  state = throughput::Find (pctx);
  REQUIRE (state.floor > 5.0);
  REQUIRE (state.floor < 10.0);
  REQUIRE (state.peak == 10.0);

  throughput::Observe (pctx, 2.0, 0.5, 0.5, 0.5);
  state = throughput::Find (pctx);
  REQUIRE (state.floor == 2.0);
  REQUIRE (state.peak == 10.0);

  throughput::Reset (pctx);
  throughput::Observe (pctx, 0.0, 0.5, 0.5, 0.5);
  state = throughput::Find (pctx);
  REQUIRE (state.initialized);
  REQUIRE (state.floor == 0.0);
  REQUIRE (state.peak == 0.0);

  throughput::Observe (pctx, 10.0, 0.5, 0.5, 0.5);
  state = throughput::Find (pctx);
  REQUIRE (state.initialized);
  REQUIRE (state.floor > 0.0);
  REQUIRE (state.floor < 10.0);
  REQUIRE (state.peak == 10.0);
  throughput::Erase (pctx);
}

TEST_CASE ("Consolidation state boundaries are inclusive and required-first",
           "[operations][consolidation][hint]")
{
  namespace throughput
      = operations::consolidation_throughput_state_internal;
  constexpr double focus = 0.5;
  constexpr double sensitivity = 0.5;
  constexpr double stability = 0.5;
  const throughput::State state { 0.0, 1.0, true };
  const double recommended
      = throughput::TriggerFraction (focus, sensitivity, stability);
  const double required
      = throughput::RequiredTriggerFraction (focus, sensitivity, stability);
  const long long backlog
      = throughput::MinimumBacklog (focus, sensitivity, stability);

  REQUIRE (throughput::Classify (
               state, required, backlog - 1, focus, sensitivity, stability)
           == ConsolidationState::None);

  REQUIRE (throughput::Classify (
               state, std::nextafter (recommended, 1.0), backlog,
               focus, sensitivity, stability)
           == ConsolidationState::None);
  REQUIRE (throughput::Classify (
               state, recommended, backlog, focus, sensitivity, stability)
           == ConsolidationState::Recommended);
  REQUIRE (throughput::Classify (
               state, std::nextafter (required, 1.0), backlog,
               focus, sensitivity, stability)
           == ConsolidationState::Recommended);
  REQUIRE (throughput::Classify (
               state, required, backlog, focus, sensitivity, stability)
           == ConsolidationState::Required);
}

TEST_CASE ("Consolidation state follows the ordered throughput drawdown",
           "[operations][consolidation][hint]")
{
  namespace throughput
      = operations::consolidation_throughput_state_internal;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  const int backlog = static_cast<int> (
      throughput::MinimumBacklog (
          cfg.focus, cfg.sensitivity, cfg.stability));

  SECTION ("bootstrap has no range")
    {
      const auto output = RunConsolidationHint (
          cfg, { 4.0, 4.0, 4.0, backlog });
      REQUIRE (output.consolidation_state == ConsolidationState::None);
    }

  SECTION ("rate above recommendation boundary is none")
    {
      const auto output = RunConsolidationHint (
          cfg, { 2.0, 10.0, 7.0, backlog });
      REQUIRE (output.consolidation_state == ConsolidationState::None);
    }

  SECTION ("drawdown enters recommended band")
    {
      const auto output = RunConsolidationHint (
          cfg, { 2.0, 10.0, 5.0, backlog });
      REQUIRE (output.consolidation_state
               == ConsolidationState::Recommended);
    }

  SECTION ("deeper drawdown enters required band")
    {
      const auto output = RunConsolidationHint (
          cfg, { 2.0, 10.0, 3.0, backlog });
      REQUIRE (output.consolidation_state == ConsolidationState::Required);
    }

  SECTION ("no backlog suppresses throughput state")
    {
      const auto output = RunConsolidationHint (
          cfg, { 2.0, 10.0, 2.0, 0 });
      REQUIRE (output.consolidation_state == ConsolidationState::None);
    }

  SECTION ("large backlog cannot replace a throughput range")
    {
      const auto output = RunConsolidationHint (
          cfg, { 4.0, 4.0, 4.0, 100'000 });
      REQUIRE (output.consolidation_state == ConsolidationState::None);
    }

  SECTION ("explicit consolidation input emits none")
    {
      const auto output = RunConsolidationHint (
          cfg, { 2.0, 10.0, 2.0, backlog }, "test/source", "text", true);
      REQUIRE (output.consolidation_state == ConsolidationState::None);
    }
}

TEST_CASE ("Consolidation recommendation rearms after a lower throughput regime",
           "[operations][consolidation][hint]")
{
  namespace throughput
      = operations::consolidation_throughput_state_internal;
  constexpr double focus = 0.5;
  constexpr double sensitivity = 0.5;
  constexpr double stability = 0.5;
  const long long backlog
      = throughput::MinimumBacklog (focus, sensitivity, stability);
  ProcessorContext pctx;
  throughput::Reset (pctx, { 2.0, 100.0, true, true });

  REQUIRE (throughput::Classify (throughput::Find (pctx), 3.0, backlog, focus,
                                 sensitivity, stability)
           != ConsolidationState::None);
  throughput::Acknowledge (pctx, 3.0);
  REQUIRE_FALSE (throughput::Find (pctx).armed);
  REQUIRE (throughput::Find (pctx).floor == 3.0);
  REQUIRE (throughput::Find (pctx).peak == 3.0);
  REQUIRE (throughput::Classify (throughput::Find (pctx), 3.0, backlog, focus,
                                 sensitivity, stability)
           == ConsolidationState::None);

  // The next recovery is far below the old 100-unit spike but establishes a
  // new event-derived range and rearms the classifier.
  throughput::Observe (pctx, 20.0, focus, sensitivity, stability);
  REQUIRE (throughput::Find (pctx).armed);
  for (long long observation = 1; observation < backlog; ++observation)
    throughput::Observe (pctx, 3.0, focus, sensitivity, stability);
  REQUIRE (throughput::Classify (throughput::Find (pctx), 3.0, backlog, focus,
                                 sensitivity, stability)
           != ConsolidationState::None);

  // A second acknowledgment and still-lower regime can rearm independently.
  throughput::Acknowledge (pctx, 3.0);
  throughput::Observe (pctx, 12.0, focus, sensitivity, stability);
  REQUIRE (throughput::Find (pctx).armed);
  for (long long observation = 1; observation < backlog; ++observation)
    throughput::Observe (pctx, 3.0, focus, sensitivity, stability);
  REQUIRE (throughput::Classify (throughput::Find (pctx), 3.0, backlog, focus,
                                 sensitivity, stability)
           != ConsolidationState::None);
  throughput::Erase (pctx);
}

TEST_CASE ("Sustained throughput drift rearms consolidation after acknowledgment",
           "[operations][consolidation][hint][drift]")
{
  namespace throughput
      = operations::consolidation_throughput_state_internal;
  ProcessorContext pctx;
  const double knob_values[] = { 0.0, 0.5, 1.0 };
  for (const double focus : knob_values)
    {
      for (const double sensitivity : knob_values)
        {
          for (const double stability : knob_values)
            {
              CAPTURE (focus, sensitivity, stability);
              const long long backlog
                  = throughput::MinimumBacklog (
                      focus, sensitivity, stability);
              throughput::Reset (pctx, { 20.0, 100.0, true, true });
              REQUIRE (throughput::Classify (throughput::Find (pctx), 20.0,
                                             backlog, focus, sensitivity, stability)
                       != ConsolidationState::None);
              throughput::Acknowledge (pctx, 100.0);

              ConsolidationState later_hint = ConsolidationState::None;
              for (int rate = 99; rate >= 0; --rate)
                {
                  throughput::Observe (pctx, static_cast<double> (rate), focus,
                                       sensitivity, stability);
                  later_hint = throughput::Classify (
                      throughput::Find (pctx), static_cast<double> (rate),
                      backlog,
                      focus, sensitivity, stability);
                  if (later_hint != ConsolidationState::None)
                    {
                      break;
                    }
                }

              while (later_hint == ConsolidationState::None
                     && throughput::Find (pctx).observations_since_ack
                            < backlog)
                {
                  throughput::Observe (pctx, 0.0, focus, sensitivity,
                                       stability);
                  later_hint = throughput::Classify (
                      throughput::Find (pctx), 0.0, backlog, focus,
                      sensitivity, stability);
                }

              REQUIRE (later_hint != ConsolidationState::None);
            }
        }
    }
  throughput::Erase (pctx);
}

TEST_CASE ("Stable noisy throughput stays disarmed after acknowledgment",
           "[operations][consolidation][hint][drift]")
{
  namespace throughput
      = operations::consolidation_throughput_state_internal;
  ProcessorContext pctx;
  const double knob_values[] = { 0.0, 0.5, 1.0 };
  const double ordinary_noise[] = { 99.999, 97.0, 100.0, 98.0, 96.0,
                                    99.0,   95.0, 98.0,  100.0 };
  for (const double focus : knob_values)
    {
      for (const double sensitivity : knob_values)
        {
          for (const double stability : knob_values)
            {
              CAPTURE (focus, sensitivity, stability);
              const long long backlog
                  = throughput::MinimumBacklog (
                      focus, sensitivity, stability);
              throughput::Reset (pctx, { 20.0, 100.0, true, true });
              throughput::Acknowledge (pctx, 100.0);
              for (long long observation = 0;
                   observation < backlog
                                     + static_cast<long long> (
                                         std::size (ordinary_noise));
                   ++observation)
                {
                  const double rate = ordinary_noise[
                      static_cast<std::size_t> (observation)
                      % std::size (ordinary_noise)];
                  throughput::Observe (pctx, rate, focus, sensitivity,
                                       stability);
                  REQUIRE (throughput::Classify (
                               throughput::Find (pctx), rate, backlog, focus,
                               sensitivity, stability)
                           == ConsolidationState::None);
                }
              REQUIRE_FALSE (throughput::Find (pctx).armed);
            }
        }
    }
  throughput::Erase (pctx);
}

TEST_CASE ("Elapsed time alone does not recommend consolidation",
           "[operations][consolidation][hint]")
{
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  const auto output = RunConsolidationHint (
      cfg, { 4.0, 4.0, 4.0, 1 });
  REQUIRE (output.consolidation_state == ConsolidationState::None);
}

TEST_CASE ("Consolidation throughput hint ignores source and modality",
           "[operations][consolidation][hint][invariance]")
{
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  const SetupConsolidationHintOp setup {
    2.0, 10.0, 5.0,
    static_cast<int> (
        operations::consolidation_throughput_state_internal::MinimumBacklog (
            cfg.focus, cfg.sensitivity, cfg.stability))
  };
  const auto text = RunConsolidationHint (
      cfg, setup, "one/source", "text");
  const auto image = RunConsolidationHint (
      cfg, setup, "unrelated/provenance", "image");
  REQUIRE (text.consolidation_state == image.consolidation_state);
}

// Helper op to verify consolidation start flag after execution.
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
    REQUIRE (p.last_consolidation_ts != expected_ts_);
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

TEST_CASE ("Alg28 explicit force starts regardless of rate and idle inputs",
           "[operations][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0; // min_cluster_size = 3, so three below-floor rows avoid fallback
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Timestamps are in milliseconds
  const uint64_t now_ts = 100'000ULL; // 100 seconds in ms
  const uint64_t last_ret = 50'000ULL;
  const double rate_target = 2.0;
  const double m_rate = 0.8;

  auto setup = std::make_unique<SetupConsolidationInputsOp> (
      /*tokens_in_flight=*/0,
      /*queue_depth=*/0,
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

TEST_CASE ("Alg28 explicit force starts regardless of busy inputs",
           "[operations][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Timestamps are in milliseconds
  const uint64_t now_ts = 100'000ULL; // 100 seconds in ms
  const uint64_t last_ret = 50'000ULL;
  const double rate_target = 2.0;
  const double m_rate = 0.8;

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

TEST_CASE ("Alg28 explicit force starts regardless of interval inputs",
           "[operations][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  const uint64_t now_ts = 4'000'000ULL;
  const uint64_t last_cons = 1'000ULL;
  const uint64_t last_ret = 2'000ULL;

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

TEST_CASE ("Alg28 ordinary signal does not start consolidation",
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
  const uint64_t last_cons = now_ts;

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

TEST_CASE ("Alg28 explicit start does not advance completion frontier before persistence",
           "[operations][consolidation]")
{
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  pctx.last_consolidation_ts = 1234ULL;
  pctx.consolidation_count = 7;
  pctx.memories_since_consolidation = 3;

  auto signal = MakeConsolidationSignal (5000ULL);
  OperationContext ctx (signal, pctx, cfg);
  EvaluateConsolidation eval;
  eval.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (ctx.GetConsolidationShouldStart ());
  REQUIRE_FALSE (ctx.GetConsolidationPersisted ());
  REQUIRE (pctx.last_consolidation_ts == 1234ULL);
  REQUIRE (pctx.consolidation_count == 7);
  REQUIRE (pctx.memories_since_consolidation == 3);
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
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, "
      "edge_type, weight) VALUES(4, 4, 'reinforces', 1.0)",
      {});

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
  const auto connectivity_rows = tx->Execute (
      "SELECT COUNT(*) AS c FROM memories "
      "WHERE ABS(COALESCE(connectivity, 0.0)) > 1e-12",
      {});
  REQUIRE (std::any_cast<long long> (connectivity_rows[0].at ("c")) == 0);
}

TEST_CASE ("ScoreConsolidation bounds its active candidate frontier from "
           "F S T knobs",
           "[operations][consolidation][bounded][knobs][regression]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::store::ApplyMigrations (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;
  const auto candidate_limit = static_cast<std::size_t> (
      operations::sparse_retrieval_knobs_internal::
          ActivationIdentityTarget (
              cfg.focus, cfg.sensitivity, cfg.stability));
  REQUIRE (candidate_limit == 640);

  Signal signal;
  signal.timestamp = 50'000ULL;
  signal.source_id = "opaque/consolidation";
  signal.modality = "audio";
  signal.force_consolidation = true;
  signal.embedding = Eigen::VectorXf::Zero (4);
  ProcessorContext p_ctx;
  OperationContext ctx (signal, p_ctx, cfg, store.get ());

  for (std::size_t index = 0; index <= candidate_limit; ++index)
    {
      const long long id = static_cast<long long> (index + 1);
      std::vector<float> embedding (256, 0.0f);
      embedding[index % embedding.size ()] = 1.0f;
      store->Execute (
          "INSERT INTO embeddings(embedding_id, embedding, created_at) "
          "VALUES(?, ?, ?)",
          { id, embedding, id });
      store->Execute (
          "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
          "start_ts, n_signals, modality, strength, stability, redundancy, "
          "created_at) VALUES(?, ?, ?, 'LONG_TERM', ?, 1, ?, 0.0, 0.0, "
          "0.0, ?)",
          { id, id, "opaque/" + std::to_string (index % 5), id,
            index % 2 == 0 ? "audio" : "image", id });
    }

  ScoreConsolidation scorer;
  auto tx = store->Begin ();
  scorer.Execute (ctx, *tx);

  const auto &candidates = ctx.GetConsolidationCandidates ();
  REQUIRE (candidates.size () == candidate_limit);
  REQUIRE (candidates.front ().memory_id == 1);
  REQUIRE (candidates.back ().memory_id
           == static_cast<long long> (candidate_limit));
  REQUIRE (ctx.GetOperationTimings ().at (
               "ScoreConsolidation.candidate_input_limit")
           == static_cast<double> (candidate_limit));
  REQUIRE (ctx.GetOperationTimings ().at (
               "ScoreConsolidation.candidate_input_count")
           == static_cast<double> (candidate_limit));
}

TEST_CASE ("ScoreConsolidation emits its full F S T derived work envelope",
           "[operations][consolidation][bounded][knobs][ablation]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::store::ApplyMigrations (*store);
  std::vector<float> embedding (256, 0.0f);
  embedding[0] = 1.0f;
  store->Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(1, ?, 1)",
      { embedding });
  store->Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
      "start_ts, n_signals, modality, strength, stability, redundancy, "
      "created_at) VALUES(1, 1, 'opaque/history', 'LONG_TERM', 1, 1, "
      "'image', 0.0, 0.0, 0.0, 1)",
      {});

  constexpr std::array<double, 3> kValues { 0.0, 0.5, 1.0 };
  std::size_t ablation_points = 0;
  for (const double focus : kValues)
    for (const double sensitivity : kValues)
      for (const double stability : kValues)
        {
          SignalProcessor::Config cfg;
          cortext::testing::RequireEncoder (cfg);
          cfg.focus = focus;
          cfg.sensitivity = sensitivity;
          cfg.stability = stability;
          Signal signal;
          signal.timestamp = 60'000ULL + ablation_points;
          signal.source_id
              = "opaque/" + std::to_string (ablation_points % 4);
          signal.modality
              = std::array<const char *, 3> { "text", "audio", "image" }
                    [ablation_points % 3];
          signal.force_consolidation = true;
          signal.embedding = Eigen::VectorXf::Zero (4);
          ProcessorContext p_ctx;
          OperationContext ctx (signal, p_ctx, cfg, store.get ());
          auto tx = store->Begin ();
          ScoreConsolidation{}.Execute (ctx, *tx);

          const double expected_candidates = static_cast<double> (
              operations::sparse_retrieval_knobs_internal::
                  ActivationIdentityTarget (
                      focus, sensitivity, stability));
          const double expected_edges = static_cast<double> (
              operations::sparse_retrieval_knobs_internal::
                  GraphNeighborCount (
                      focus, sensitivity, stability));
          REQUIRE (ctx.GetOperationTimings ().at (
                       "ScoreConsolidation.candidate_input_limit")
                   == expected_candidates);
          REQUIRE (ctx.GetOperationTimings ().at (
                       "ScoreConsolidation.association_edge_limit")
                   == expected_edges);
          REQUIRE (ctx.GetOperationTimings ().at (
                       "ScoreConsolidation.candidate_input_count")
                   == 1.0);
          ++ablation_points;
        }
  REQUIRE (ablation_points == 27);
}

TEST_CASE ("ScoreConsolidation forced mode preserves eligibility threshold",
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
  REQUIRE (candidates.size () == 1);
  REQUIRE (candidates[0].embedding_id == 1LL);
}

TEST_CASE ("ScoreConsolidation uses current memory embedding surface",
           "[operations][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  cortext::store::ApplyMigrations (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 1.0;

  Signal dummy;
  dummy.timestamp = 125'000ULL;
  dummy.source_id = "test/consolidation";
  dummy.force_consolidation = true;
  dummy.embedding = Eigen::VectorXf::Zero (4);
  ProcessorContext p_ctx;
  OperationContext ctx (dummy, p_ctx, cfg, store.get ());

  for (long long id = 1; id <= 3; ++id)
    {
      std::vector<float> base (256, 0.0f);
      base[static_cast<size_t> (id)] = 1.0f;
      store->Execute (
          "INSERT INTO embeddings(embedding_id, embedding, created_at) "
          "VALUES(?, ?, ?)",
          { id, base, 0LL });
      store->Execute (
          "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
          "start_ts, n_signals, modality, s_max, s_avg, strength, stability, "
          "redundancy, created_at) "
          "VALUES(?, ?, 'test', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, "
          "0.05, 0.0, 0.0, 0)",
          { id, id });
    }

  std::vector<float> current (256, 0.0f);
  current[7] = 1.0f;
  store->Execute (
      "INSERT INTO current_memory_embeddings("
      "memory_id, embedding, embedding_id, created_at"
      ") VALUES (?, ?, ?, ?)",
      { 1LL, current, 1LL, 1LL });

  ScoreConsolidation op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);

  const auto &candidates = ctx.GetConsolidationCandidates ();
  REQUIRE (candidates.size () == 3);
  REQUIRE (candidates[0].memory_id == 1LL);
  REQUIRE (candidates[0].embedding.size () == 256);
  REQUIRE (candidates[0].embedding (7) == Catch::Approx (1.0f));
  REQUIRE (candidates[0].embedding (1) == Catch::Approx (0.0f));
}

TEST_CASE ("ScoreConsolidation eligibility does not depend on blobs",
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
  REQUIRE (candidates.size () == 1);
  REQUIRE (candidates[0].embedding_id == 1LL);
}
