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
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/schema.hpp>
#include "../src/operations/consolidation_throughput_state_internal.hpp"
#include <string>

using namespace cortext;

namespace
{

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
