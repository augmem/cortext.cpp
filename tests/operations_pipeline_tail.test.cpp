#include "test_helpers.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cortext/core/knobs.hpp>
#include <cortext/core/utils.hpp>
#include <cortext/operations/accumulator_reset.hpp>
#include <cortext/operations/accumulator_scores.hpp>
#include <cortext/operations/neuromodulators.hpp>
#include <cortext/operations/signal_metrics_persistence.hpp>
#include <cortext/processor/accumulator_state.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>
#include <limits>

using namespace cortext;
using namespace cortext::operations;

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

Signal
MakeSignal (const std::string &source_id = "src", uint64_t timestamp = 1000)
{
  Signal signal;
  signal.source_id = source_id;
  signal.timestamp = timestamp;
  signal.modality = "text";
  signal.embedding = UnitVec (0);
  return signal;
}

std::shared_ptr<Store>
MakeStore ()
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  return store;
}

AccumulatorState
MakeAccumulator (const Signal &signal)
{
  AccumulatorState acc;
  acc.Reset (signal.embedding, signal.timestamp);
  acc.s_sum = 0.2;
  acc.s_max = 0.3;
  SignalRecord rec;
  rec.embedding = signal.embedding;
  rec.timestamp = signal.timestamp;
  rec.modality = signal.modality;
  rec.serial_position = 0;
  rec.score = 0.2;
  acc.signals.push_back (std::move (rec));
  return acc;
}

} // namespace

#if defined(CORTEXT_EXPERIMENT_HOOKS)
#endif

TEST_CASE ("UpdateNeuromodulators derives bounded control state",
           "[operations][neuromodulators]")
{
  ProcessorContext pctx;
  pctx.last_signal_timestamp = 1000;
  pctx.delta_reward = 0.4;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;
  const auto signal = MakeSignal ("src", 2500);
  OperationContext ctx (signal, pctx, cfg);
  ctx.SetMetric (Metric::rarity, 0.8);
  ctx.SetMetric (Metric::embedding_surprisal, 0.7);
  ctx.SetArousal (0.6);
  ctx.SetRetrievalQueueDepth (3);

  UpdateNeuromodulators op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.neuromod_ach >= 0.0);
  REQUIRE (pctx.neuromod_ach <= 1.0);
  REQUIRE (pctx.neuromod_ne > 0.0);
  REQUIRE (pctx.neuromod_da >= 0.4);
  REQUIRE (pctx.encode_bias >= 0.0);
  REQUIRE (pctx.encode_bias <= 1.0);
  REQUIRE (pctx.retrieval_bias == Catch::Approx (1.0 - pctx.encode_bias));
  REQUIRE (pctx.osc_phase > 0.0);
}

TEST_CASE ("PersistSignalMetrics updates signal metric columns",
           "[operations][signal_metrics_persistence]")
{
  auto store = MakeStore ();
  const Signal signal = MakeSignal ("metrics-src", 1234);
  cortext::testing::SeedEmbeddingV2 (*store, 1LL, signal.embedding, 1234);
  cortext::testing::SeedMemoryV2 (*store, 1LL, 1LL, signal.source_id,
                                  "LONG_TERM", 1.0, 1234);
  store->Execute (
      "INSERT INTO signals(memory_id, embedding_id, source_id, timestamp, "
      "modality, serial_position, created_at) "
      "VALUES(?, ?, ?, ?, 'text', 0, ?)",
      { 1LL, 1LL, signal.source_id, static_cast<long long> (signal.timestamp),
        static_cast<long long> (signal.timestamp) });

  ProcessorContext pctx;
  pctx.accumulator_states[signal.source_id] = MakeAccumulator (signal);
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetAccumulatorWriteDecision (true);
  ctx.SetWriteDecision (true);
  ctx.SetCompositeScore (0.77);
  ctx.SetThresholdTDynamic (0.55);
  ctx.SetCoherence (0.66);
  ctx.SetEffectiveFocus (0.44);
  ctx.SetMetric (Metric::relevance, 0.91);
  ctx.SetMetric (Metric::mismatch, 0.12);
  ctx.SetMetric (Metric::focus_spread, 0.33);

  PersistSignalMetrics op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto rows = store->Execute (
      "SELECT relevance, mismatch, score, threshold_t, write_decision, "
      "coherence, focus_spread, f_effective FROM signals WHERE source_id = ?",
      { signal.source_id });
  REQUIRE (rows.size () == 1);
  REQUIRE (cortext::testing::GetDouble (rows[0], "relevance")
           == Catch::Approx (0.91));
  REQUIRE (cortext::testing::GetDouble (rows[0], "mismatch")
           == Catch::Approx (0.12));
  REQUIRE (cortext::testing::GetDouble (rows[0], "score")
           == Catch::Approx (0.77));
  REQUIRE (cortext::testing::GetDouble (rows[0], "threshold_t")
           == Catch::Approx (0.55));
  REQUIRE (cortext::testing::GetInt64 (rows[0], "write_decision") == 1);
  REQUIRE (cortext::testing::GetDouble (rows[0], "coherence")
           == Catch::Approx (0.66));
  REQUIRE (cortext::testing::GetDouble (rows[0], "focus_spread")
           == Catch::Approx (0.33));
  REQUIRE (cortext::testing::GetDouble (rows[0], "f_effective")
           == Catch::Approx (0.44));
}

TEST_CASE ("UpdateAccumulatorScores refreshes aggregate score state",
           "[operations][accumulator_scores]")
{
  const Signal signal = MakeSignal ();
  ProcessorContext pctx;
  pctx.accumulator_states[signal.source_id] = MakeAccumulator (signal);
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  OperationContext ctx (signal, pctx, cfg);
  ctx.SetCompositeScore (0.8);

  UpdateAccumulatorScores op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  const auto &acc = pctx.accumulator_states.at (signal.source_id);
  REQUIRE (acc.s_sum == Catch::Approx (1.0));
  REQUIRE (acc.s_max == Catch::Approx (0.8));
  REQUIRE (acc.e_peak.size () == signal.embedding.size ());
  REQUIRE (core::CosineSimilarity (acc.e_peak, signal.embedding)
           == Catch::Approx (1.0));
  REQUIRE (acc.signals.back ().score == Catch::Approx (0.8));
}

TEST_CASE ("Accumulator reset operations clear or mark pending state",
           "[operations][accumulator_reset]")
{
  const Signal signal = MakeSignal ("reset-src", 7777);
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);

  {
    ProcessorContext pctx;
    pctx.accumulator_states[signal.source_id] = MakeAccumulator (signal);
    OperationContext ctx (signal, pctx, cfg);
    ctx.SetFlushRequired (true);
    ResetAccumulatorAfterFlush op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());
    const auto &acc = pctx.accumulator_states.at (signal.source_id);
    REQUIRE (acc.n_signals == 0);
    REQUIRE (acc.signals.empty ());
    REQUIRE (acc.t_start == signal.timestamp);
  }

  {
    ProcessorContext pctx;
    pctx.accumulator_states[signal.source_id] = MakeAccumulator (signal);
    OperationContext ctx (signal, pctx, cfg);
    ctx.SetInterruptAllowed (true);
    ctx.SetSelectedCandidateId (42LL);
    ctx.SetRetrievedMemoryEmbeddings ({ { 42LL, UnitVec (3) } });
    ResetAccumulatorOnInterrupt op;
    op.Execute (ctx, cortext::testing::GetNullTransaction ());
    const auto &acc = pctx.accumulator_states.at (signal.source_id);
    REQUIRE (acc.n_signals == 1);
    REQUIRE (acc.pending_interrupt_abort);
    REQUIRE (core::CosineSimilarity (acc.pending_interrupt_embedding,
                                     UnitVec (3))
             == Catch::Approx (1.0));
  }
}
