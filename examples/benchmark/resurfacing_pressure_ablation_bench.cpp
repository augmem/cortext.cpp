#include "../../src/operations/eviction_ablation.hpp"
#include "../../src/operations/retrieval_debug_state.hpp"
#include "../../src/operations/temporal_retrieval.hpp"

#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/operations/metacognitive.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace
{

constexpr int kEmbeddingDim = 256;
constexpr long long kTargetMemoryId = 77LL;
constexpr long long kStorageThresholdBytes = 1000LL;
constexpr std::uint64_t kIdleGapMs = 120000ULL;
constexpr std::uint64_t kRetrievalAgeMs = 168ULL * 60ULL * 60ULL * 1000ULL;

Eigen::VectorXf
UnitVec256 ()
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = 1.0f;
  return v;
}

std::vector<float>
ToFloatVec (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

class ForceRetrievalGateOp : public cortext::IOperation
{
public:
  ForceRetrievalGateOp (
      std::optional<cortext::ProcessorContext::MetacognitiveMode>
          metacognitive_mode = std::nullopt,
      std::optional<double> metacognitive_confidence = std::nullopt)
      : metacognitive_mode_ (metacognitive_mode),
        metacognitive_confidence_ (metacognitive_confidence)
  {
  }

  void
  Execute (cortext::OperationContext &ctx,
           cortext::Transaction & /*tx*/) const override
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
    if (metacognitive_mode_.has_value ())
      {
        p_ctx.metacognitive_mode = *metacognitive_mode_;
        p_ctx.metacognitive_mode_expires_at = ctx.GetSignal ().timestamp + 1000;
        p_ctx.metacognitive_certainty_satisfied = false;
      }
    if (metacognitive_confidence_.has_value ())
      {
        p_ctx.metacognitive_confidence = *metacognitive_confidence_;
      }
  }

private:
  std::optional<cortext::ProcessorContext::MetacognitiveMode>
      metacognitive_mode_;
  std::optional<double> metacognitive_confidence_;
};

class BenchEncoder : public cortext::Encoder
{
public:
  void
  EncodeText (const std::string &, std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }

  void
  EncodeAudio (const float *, std::size_t,
               std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }

  void
  EncodeImage (const std::uint8_t *, int, int, int,
               std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }
};

void
SeedMemory (cortext::Store &store, long long memory_id,
            const Eigen::VectorXf &embedding, long long created_at)
{
  store.Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?)",
      { memory_id, ToFloatVec (embedding), created_at });
  store.Execute (
      "INSERT INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, created_at, source_origin, source_reliability, "
      "source_contradiction_count) "
      "VALUES(?, ?, 'chat/user', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, 1.0, ?, "
      "'external', 0.46, 2)",
      { memory_id, memory_id, created_at, created_at });
}

struct Scenario
{
  std::string name;
  std::optional<cortext::operations::temporal::ResurfacingDecayMode> decay_mode;
  long long used_storage_bytes = 0;
};

struct ScenarioRow
{
  std::string scenario;
  double confidence = 0.0;
  bool unknown_detected = false;
  bool surfaced = false;
  long long top1 = 0;
};

ScenarioRow
RunScenario (const Scenario &scenario,
             const cortext::SignalProcessor::Config &cfg)
{
  cortext::operations::temporal::RetrievalAblationOverride retrieval_override;
  retrieval_override.resurfacing_decay_mode = scenario.decay_mode;
  cortext::operations::temporal::ScopedRetrievalAblationOverride
      retrieval_guard (retrieval_override);

  cortext::operations::eviction::EvictionAblationOverride eviction_override;
  eviction_override.storage_gate_enabled = true;
  eviction_override.min_storage_bytes = kStorageThresholdBytes;
  eviction_override.used_storage_bytes = scenario.used_storage_bytes;
  cortext::operations::eviction::ScopedEvictionAblationOverride eviction_guard (
      eviction_override);

  ScenarioRow row;
  row.scenario = scenario.name;

  {
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
    cortext::store::ApplyMigrations (*store);

    cortext::ProcessorContext p_ctx;
    p_ctx.metacognitive_confidence = 1.0;
    p_ctx.last_signal_timestamp = 1000;

    cortext::Signal signal;
    signal.embedding = UnitVec256 ();
    signal.timestamp = 1000 + kIdleGapMs;
    signal.source_id = "meta";

    cortext::OperationContext ctx (signal, p_ctx, cfg);
    ctx.SetFeelingOfKnowing (0.2);
    ctx.SetMemoryUsageEvents ({ { 1LL, true, -1.0 } });

    auto tx = store->Begin ();
    cortext::operations::MetacognitiveMonitoring op;
    op.Execute (ctx, *tx);
    tx->Commit ();

    row.confidence = p_ctx.metacognitive_confidence;
    row.unknown_detected = ctx.GetMetacogUnknownDetected ();
  }

  {
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
    cortext::store::ApplyMigrations (*store);
    SeedMemory (*store, kTargetMemoryId, UnitVec256 (), 1);

    auto ops = std::make_unique<cortext::DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (
            cortext::ProcessorContext::MetacognitiveMode::UnknownCaution, 0.0),
        std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
    cortext::SignalProcessor processor (cfg, store, std::move (ops));

    cortext::Signal signal;
    signal.embedding = UnitVec256 ();
    signal.timestamp = 1000 + kRetrievalAgeMs;
    signal.source_id = "retrieval";

    cortext::operations::retrieval_debug::ClearLastSelectedEmbeddingOrder ();
    const auto out = processor.Process (signal);
    processor.Flush ();
    row.surfaced = std::find (out.candidate_memory_ids.begin (),
                              out.candidate_memory_ids.end (),
                              kTargetMemoryId)
                   != out.candidate_memory_ids.end ();
    if (!out.candidate_memory_ids.empty ())
      {
        row.top1 = out.candidate_memory_ids.front ();
      }
  }

  return row;
}

bool
Check (const char *name, bool condition)
{
  std::cout << name << "=" << (condition ? 1 : 0) << "\n";
  return condition;
}

} // namespace

int
main ()
{
  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  BenchEncoder encoder;
  cfg.encoder = &encoder;

  const std::vector<Scenario> scenarios = {
    { "default_low_pressure", std::nullopt, 100 },
    { "default_high_pressure", std::nullopt, 1000 },
    { "time_only_low_pressure",
      cortext::operations::temporal::ResurfacingDecayMode::TimeOnly, 100 },
    { "time_plus_pressure_gate_low",
      cortext::operations::temporal::ResurfacingDecayMode::PressureGate, 100 },
    { "time_plus_pressure_ramp_low",
      cortext::operations::temporal::ResurfacingDecayMode::PressureRamp, 100 },
    { "time_plus_pressure_ramp_high",
      cortext::operations::temporal::ResurfacingDecayMode::PressureRamp, 1000 },
  };

  std::vector<ScenarioRow> rows;
  rows.reserve (scenarios.size ());
  for (const auto &scenario : scenarios)
    {
      rows.push_back (RunScenario (scenario, cfg));
    }

  std::cout << std::fixed << std::setprecision (6);
  std::cout << "scenario confidence unknown_detected surfaced top1\n";
  for (const auto &row : rows)
    {
      std::cout << row.scenario << " " << row.confidence << " "
                << (row.unknown_detected ? 1 : 0) << " "
                << (row.surfaced ? 1 : 0) << " " << row.top1 << "\n";
    }

  auto find_row = [&rows] (const std::string &name) -> const ScenarioRow * {
    for (const auto &row : rows)
      {
        if (row.scenario == name)
          {
            return &row;
          }
      }
    return nullptr;
  };

  const auto *time_only = find_row ("time_only_low_pressure");
  const auto *default_low = find_row ("default_low_pressure");
  const auto *default_high = find_row ("default_high_pressure");
  const auto *gate_low = find_row ("time_plus_pressure_gate_low");
  const auto *ramp_low = find_row ("time_plus_pressure_ramp_low");
  const auto *ramp_high = find_row ("time_plus_pressure_ramp_high");

  bool ok = true;
  ok &= Check ("default_low_matches_pressure_ramp_low_confidence",
               default_low && ramp_low
                   && std::abs (default_low->confidence - ramp_low->confidence)
                          < 1e-6);
  ok &= Check ("default_low_surfaces_target",
               default_low && default_low->surfaced
                   && default_low->top1 == kTargetMemoryId);
  ok &= Check ("default_high_matches_pressure_ramp_high_confidence",
               default_high && ramp_high
                   && std::abs (default_high->confidence - ramp_high->confidence)
                          < 1e-6);
  ok &= Check ("default_high_suppresses_target",
               default_high && !default_high->surfaced);
  ok &= Check ("pressure_gate_low_confidence_gt_time_only",
               gate_low && time_only
                   && gate_low->confidence > time_only->confidence);
  ok &= Check ("pressure_ramp_low_confidence_gt_time_only",
               ramp_low && time_only
                   && ramp_low->confidence > time_only->confidence);
  ok &= Check ("pressure_gate_low_surfaces_target",
               gate_low && gate_low->surfaced
                   && gate_low->top1 == kTargetMemoryId);
  ok &= Check ("pressure_ramp_low_surfaces_target",
               ramp_low && ramp_low->surfaced
                   && ramp_low->top1 == kTargetMemoryId);
  ok &= Check ("time_only_low_pressure_suppresses_target",
               time_only && !time_only->surfaced);
  ok &= Check ("pressure_ramp_high_matches_time_only_confidence",
               ramp_high && time_only
                   && std::abs (ramp_high->confidence - time_only->confidence)
                          < 1e-6);
  ok &= Check ("pressure_ramp_high_suppresses_target",
               ramp_high && !ramp_high->surfaced);

  return ok ? 0 : 1;
}
