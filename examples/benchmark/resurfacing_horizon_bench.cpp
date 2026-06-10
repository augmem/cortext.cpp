#include "../../src/operations/eviction_ablation.hpp"
#include "../../src/operations/temporal_retrieval.hpp"

#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/operations/metacognitive.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <algorithm>
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
constexpr double kConfidenceFloor = 0.2;

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

struct Gap
{
  std::string label;
  std::uint64_t ms = 0;
};

struct ScenarioRow
{
  std::string scenario;
  std::string gap;
  double confidence = 0.0;
  bool surfaced = false;
  long long top1 = 0;
};

struct ScenarioSummary
{
  std::string scenario;
  std::string first_floor_gap = "never";
  std::string first_suppress_gap = "never";
  std::string last_surface_gap = "never";
};

double
RunConfidence (const Scenario &scenario, const cortext::SignalProcessor::Config &cfg,
               std::uint64_t gap_ms)
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

  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  cortext::ProcessorContext p_ctx;
  p_ctx.metacognitive_confidence = 1.0;
  p_ctx.last_signal_timestamp = 1000;

  cortext::Signal signal;
  signal.embedding = UnitVec256 ();
  signal.timestamp = 1000 + gap_ms;
  signal.source_id = "meta";

  cortext::OperationContext ctx (signal, p_ctx, cfg);
  ctx.SetFeelingOfKnowing (0.2);
  ctx.SetMemoryUsageEvents ({ { 1LL, true, -1.0 } });

  auto tx = store->Begin ();
  cortext::operations::MetacognitiveMonitoring op;
  op.Execute (ctx, *tx);
  tx->Commit ();

  return p_ctx.metacognitive_confidence;
}

std::pair<bool, long long>
RunRetrieval (const Scenario &scenario, const cortext::SignalProcessor::Config &cfg,
              std::uint64_t gap_ms)
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
  signal.timestamp = 1000 + gap_ms;
  signal.source_id = "retrieval";

  const auto out = processor.Process (signal);
  processor.Flush ();

  const bool surfaced
      = std::find (out.candidate_memory_ids.begin (),
                   out.candidate_memory_ids.end (),
                   kTargetMemoryId)
        != out.candidate_memory_ids.end ();
  const long long top1
      = out.candidate_memory_ids.empty () ? 0LL : out.candidate_memory_ids.front ();
  return { surfaced, top1 };
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

  const std::vector<Gap> gaps = {
    { "1m", 60ULL * 1000ULL },
    { "5m", 5ULL * 60ULL * 1000ULL },
    { "15m", 15ULL * 60ULL * 1000ULL },
    { "30m", 30ULL * 60ULL * 1000ULL },
    { "2h", 2ULL * 60ULL * 60ULL * 1000ULL },
    { "6h", 6ULL * 60ULL * 60ULL * 1000ULL },
    { "24h", 24ULL * 60ULL * 60ULL * 1000ULL },
    { "3d", 3ULL * 24ULL * 60ULL * 60ULL * 1000ULL },
    { "7d", 7ULL * 24ULL * 60ULL * 60ULL * 1000ULL },
    { "30d", 30ULL * 24ULL * 60ULL * 60ULL * 1000ULL },
    { "90d", 90ULL * 24ULL * 60ULL * 60ULL * 1000ULL },
    { "180d", 180ULL * 24ULL * 60ULL * 60ULL * 1000ULL },
    { "365d", 365ULL * 24ULL * 60ULL * 60ULL * 1000ULL },
    { "3650d", 3650ULL * 24ULL * 60ULL * 60ULL * 1000ULL },
  };

  std::vector<ScenarioRow> rows;
  rows.reserve (scenarios.size () * gaps.size ());
  for (const auto &scenario : scenarios)
    {
      for (const auto &gap : gaps)
        {
          ScenarioRow row;
          row.scenario = scenario.name;
          row.gap = gap.label;
          row.confidence = RunConfidence (scenario, cfg, gap.ms);
          const auto retrieval = RunRetrieval (scenario, cfg, gap.ms);
          row.surfaced = retrieval.first;
          row.top1 = retrieval.second;
          rows.push_back (row);
        }
    }

  std::cout << std::fixed << std::setprecision (6);
  std::cout << "scenario gap confidence surfaced top1\n";
  for (const auto &row : rows)
    {
      std::cout << row.scenario << " " << row.gap << " " << row.confidence
                << " " << (row.surfaced ? 1 : 0) << " " << row.top1 << "\n";
    }

  std::vector<ScenarioSummary> summaries;
  summaries.reserve (scenarios.size ());
  for (const auto &scenario : scenarios)
    {
      ScenarioSummary summary;
      summary.scenario = scenario.name;
      for (const auto &gap : gaps)
        {
          auto it = std::find_if (
              rows.begin (), rows.end (),
              [&] (const ScenarioRow &row) {
                return row.scenario == scenario.name && row.gap == gap.label;
              });
          if (it == rows.end ())
            {
              continue;
            }
          if (summary.first_floor_gap == "never"
              && it->confidence <= kConfidenceFloor + 1e-6)
            {
              summary.first_floor_gap = gap.label;
            }
          if (summary.first_suppress_gap == "never" && !it->surfaced)
            {
              summary.first_suppress_gap = gap.label;
            }
          if (it->surfaced)
            {
              summary.last_surface_gap = gap.label;
            }
        }
      summaries.push_back (summary);
    }

  std::cout << "summary scenario first_floor_gap first_suppress_gap last_surface_gap\n";
  for (const auto &summary : summaries)
    {
      std::cout << summary.scenario << " " << summary.first_floor_gap << " "
                << summary.first_suppress_gap << " "
                << summary.last_surface_gap << "\n";
    }

  auto find_summary = [&summaries] (const std::string &name)
      -> const ScenarioSummary * {
    for (const auto &summary : summaries)
      {
        if (summary.scenario == name)
          {
            return &summary;
          }
      }
    return nullptr;
  };

  const auto *time_only = find_summary ("time_only_low_pressure");
  const auto *default_low = find_summary ("default_low_pressure");
  const auto *default_high = find_summary ("default_high_pressure");
  const auto *gate_low = find_summary ("time_plus_pressure_gate_low");
  const auto *ramp_low = find_summary ("time_plus_pressure_ramp_low");
  const auto *ramp_high = find_summary ("time_plus_pressure_ramp_high");

  bool ok = true;
  ok &= Check ("default_low_matches_pressure_ramp_low_suppression",
               default_low && ramp_low
                   && default_low->first_suppress_gap
                          == ramp_low->first_suppress_gap
                   && default_low->last_surface_gap
                          == ramp_low->last_surface_gap);
  ok &= Check ("default_low_surfaces_through_3650d",
               default_low && default_low->last_surface_gap == "3650d"
                   && default_low->first_suppress_gap == "never");
  ok &= Check ("default_high_matches_pressure_ramp_high_suppression",
               default_high && ramp_high
                   && default_high->first_suppress_gap
                          == ramp_high->first_suppress_gap
                   && default_high->last_surface_gap
                          == ramp_high->last_surface_gap);
  ok &= Check ("time_only_reaches_confidence_floor_by_1m",
               time_only && time_only->first_floor_gap == "1m");
  ok &= Check ("time_only_first_suppression_by_1m",
               time_only && time_only->first_suppress_gap == "1m");
  ok &= Check ("pressure_gate_surfaces_through_3650d",
               gate_low && gate_low->last_surface_gap == "3650d"
                   && gate_low->first_suppress_gap == "never");
  ok &= Check ("pressure_ramp_low_surfaces_through_3650d",
               ramp_low && ramp_low->last_surface_gap == "3650d"
                   && ramp_low->first_suppress_gap == "never");
  ok &= Check ("pressure_ramp_high_matches_time_only_suppression",
               ramp_high && time_only
                   && ramp_high->first_suppress_gap
                          == time_only->first_suppress_gap);

  return ok ? 0 : 1;
}
