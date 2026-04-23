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

struct KnobLevel
{
  std::string label;
  double value = 0.5;
};

struct KnobPoint
{
  std::string label;
  double focus = 0.5;
  double sensitivity = 0.5;
  double stability = 0.5;
};

struct ScenarioRow
{
  std::string scenario;
  std::string knob_label;
  double focus = 0.5;
  double sensitivity = 0.5;
  double stability = 0.5;
  std::string gap;
  double confidence = 0.0;
  bool surfaced = false;
};

struct ScenarioSummary
{
  std::string scenario;
  std::string knob_label;
  double focus = 0.5;
  double sensitivity = 0.5;
  double stability = 0.5;
  int first_floor_index = 0;
  int first_suppress_index = 0;
  int last_surface_index = 0;
};

struct MarginalAggregate
{
  std::string scenario;
  std::string dimension;
  std::string level_label;
  double level_value = 0.5;
  double avg_first_floor_index = 0.0;
  double avg_first_suppress_index = 0.0;
  double avg_last_surface_index = 0.0;
  int no_suppression_count = 0;
  int count = 0;
};

double
RunConfidence (const Scenario &scenario, const KnobPoint &knobs,
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

  cortext::SignalProcessor::Config cfg;
  cfg.focus = knobs.focus;
  cfg.sensitivity = knobs.sensitivity;
  cfg.stability = knobs.stability;
  BenchEncoder encoder;
  cfg.encoder = &encoder;

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

bool
RunRetrieval (const Scenario &scenario, const KnobPoint &knobs,
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

  cortext::SignalProcessor::Config cfg;
  cfg.focus = knobs.focus;
  cfg.sensitivity = knobs.sensitivity;
  cfg.stability = knobs.stability;
  BenchEncoder encoder;
  cfg.encoder = &encoder;

  auto ops = std::make_unique<cortext::OperationSet> (
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
  return std::find (out.candidate_memory_ids.begin (),
                    out.candidate_memory_ids.end (),
                    kTargetMemoryId)
         != out.candidate_memory_ids.end ();
}

int
GapIndexOrNever (const std::string &gap_label, const std::vector<Gap> &gaps)
{
  for (std::size_t i = 0; i < gaps.size (); ++i)
    {
      if (gaps[i].label == gap_label)
        {
          return static_cast<int> (i);
        }
    }
  return static_cast<int> (gaps.size ());
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
  const std::vector<Scenario> scenarios = {
    { "default_low_pressure", std::nullopt, 100 },
    { "default_high_pressure", std::nullopt, 1000 },
    { "time_only_low_pressure",
      cortext::operations::temporal::ResurfacingDecayMode::TimeOnly, 100 },
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

  const std::vector<KnobLevel> levels = {
    { "low", 0.2 },
    { "mid", 0.5 },
    { "high", 0.9 },
  };

  std::vector<KnobPoint> knob_points;
  knob_points.reserve (levels.size () * levels.size () * levels.size ());
  for (const auto &focus : levels)
    {
      for (const auto &sensitivity : levels)
        {
          for (const auto &stability : levels)
            {
              knob_points.push_back (
                  { "F=" + focus.label + ",S=" + sensitivity.label + ",T="
                        + stability.label,
                    focus.value, sensitivity.value, stability.value });
            }
        }
    }

  std::vector<ScenarioRow> rows;
  rows.reserve (scenarios.size () * knob_points.size () * gaps.size ());
  for (const auto &scenario : scenarios)
    {
      for (const auto &knobs : knob_points)
        {
          for (const auto &gap : gaps)
            {
              rows.push_back ({ scenario.name,
                                knobs.label,
                                knobs.focus,
                                knobs.sensitivity,
                                knobs.stability,
                                gap.label,
                                RunConfidence (scenario, knobs, gap.ms),
                                RunRetrieval (scenario, knobs, gap.ms) });
            }
        }
    }

  std::vector<ScenarioSummary> summaries;
  summaries.reserve (scenarios.size () * knob_points.size ());
  for (const auto &scenario : scenarios)
    {
      for (const auto &knobs : knob_points)
        {
          ScenarioSummary summary;
          summary.scenario = scenario.name;
          summary.knob_label = knobs.label;
          summary.focus = knobs.focus;
          summary.sensitivity = knobs.sensitivity;
          summary.stability = knobs.stability;
          summary.first_floor_index = static_cast<int> (gaps.size ());
          summary.first_suppress_index = static_cast<int> (gaps.size ());
          summary.last_surface_index = -1;

          for (std::size_t i = 0; i < gaps.size (); ++i)
            {
              const auto it = std::find_if (
                  rows.begin (), rows.end (),
                  [&] (const ScenarioRow &row) {
                    return row.scenario == scenario.name
                           && row.knob_label == knobs.label
                           && row.gap == gaps[i].label;
                  });
              if (it == rows.end ())
                {
                  continue;
                }
              if (summary.first_floor_index == static_cast<int> (gaps.size ())
                  && it->confidence <= kConfidenceFloor + 1e-6)
                {
                  summary.first_floor_index = static_cast<int> (i);
                }
              if (summary.first_suppress_index
                      == static_cast<int> (gaps.size ())
                  && !it->surfaced)
                {
                  summary.first_suppress_index = static_cast<int> (i);
                }
              if (it->surfaced)
                {
                  summary.last_surface_index = static_cast<int> (i);
                }
            }

          summaries.push_back (summary);
        }
    }

  auto summarize_dimension = [&] (const std::string &scenario_name,
                                  const std::string &dimension,
                                  const KnobLevel &level) {
    MarginalAggregate aggregate;
    aggregate.scenario = scenario_name;
    aggregate.dimension = dimension;
    aggregate.level_label = level.label;
    aggregate.level_value = level.value;

    for (const auto &summary : summaries)
      {
        if (summary.scenario != scenario_name)
          {
            continue;
          }
        const bool match = (dimension == "focus" && summary.focus == level.value)
                           || (dimension == "sensitivity"
                               && summary.sensitivity == level.value)
                           || (dimension == "stability"
                               && summary.stability == level.value);
        if (!match)
          {
            continue;
          }
        aggregate.avg_first_floor_index += summary.first_floor_index;
        aggregate.avg_first_suppress_index += summary.first_suppress_index;
        aggregate.avg_last_surface_index += summary.last_surface_index;
        aggregate.no_suppression_count
            += summary.first_suppress_index == static_cast<int> (gaps.size ())
                   ? 1
                   : 0;
        aggregate.count++;
      }

    if (aggregate.count > 0)
      {
        aggregate.avg_first_floor_index
            /= static_cast<double> (aggregate.count);
        aggregate.avg_first_suppress_index
            /= static_cast<double> (aggregate.count);
        aggregate.avg_last_surface_index
            /= static_cast<double> (aggregate.count);
      }
    return aggregate;
  };

  std::vector<MarginalAggregate> marginals;
  for (const auto &scenario : scenarios)
    {
      for (const auto &level : levels)
        {
          marginals.push_back (
              summarize_dimension (scenario.name, "focus", level));
          marginals.push_back (
              summarize_dimension (scenario.name, "sensitivity", level));
          marginals.push_back (
              summarize_dimension (scenario.name, "stability", level));
        }
    }

  std::cout << std::fixed << std::setprecision (3);
  std::cout << "summary scenario knobs focus sensitivity stability "
               "first_floor_gap first_suppress_gap last_surface_gap\n";
  for (const auto &summary : summaries)
    {
      const auto floor_gap = summary.first_floor_index >= static_cast<int> (gaps.size ())
                                 ? std::string ("never")
                                 : gaps[summary.first_floor_index].label;
      const auto suppress_gap
          = summary.first_suppress_index >= static_cast<int> (gaps.size ())
                ? std::string ("never")
                : gaps[summary.first_suppress_index].label;
      const auto last_surface_gap = summary.last_surface_index < 0
                                        ? std::string ("never")
                                        : gaps[summary.last_surface_index].label;
      std::cout << summary.scenario << " " << summary.knob_label << " "
                << summary.focus << " " << summary.sensitivity << " "
                << summary.stability << " " << floor_gap << " " << suppress_gap
                << " " << last_surface_gap << "\n";
    }

  std::cout << "marginal scenario dimension level value avg_first_floor_idx "
               "avg_first_suppress_idx avg_last_surface_idx no_suppression count\n";
  for (const auto &aggregate : marginals)
    {
      std::cout << aggregate.scenario << " " << aggregate.dimension << " "
                << aggregate.level_label << " " << aggregate.level_value << " "
                << aggregate.avg_first_floor_index << " "
                << aggregate.avg_first_suppress_index << " "
                << aggregate.avg_last_surface_index << " "
                << aggregate.no_suppression_count << " " << aggregate.count
                << "\n";
    }

  auto find_summary = [&] (const std::string &scenario, const std::string &label)
      -> const ScenarioSummary * {
    for (const auto &summary : summaries)
      {
        if (summary.scenario == scenario && summary.knob_label == label)
          {
            return &summary;
          }
      }
    return nullptr;
  };

  bool default_low_no_worse_than_time_only = true;
  bool default_high_matches_time_only = true;
  for (const auto &knobs : knob_points)
    {
      const auto *default_low
          = find_summary ("default_low_pressure", knobs.label);
      const auto *default_high
          = find_summary ("default_high_pressure", knobs.label);
      const auto *time_only
          = find_summary ("time_only_low_pressure", knobs.label);
      if (!default_low || !default_high || !time_only)
        {
          default_low_no_worse_than_time_only = false;
          default_high_matches_time_only = false;
          continue;
        }
      default_low_no_worse_than_time_only
          = default_low_no_worse_than_time_only
            && default_low->first_suppress_index >= time_only->first_suppress_index
            && default_low->first_floor_index >= time_only->first_floor_index;
      default_high_matches_time_only
          = default_high_matches_time_only
            && default_high->first_suppress_index == time_only->first_suppress_index
            && default_high->first_floor_index == time_only->first_floor_index;
    }

  const auto *mid_default_low
      = find_summary ("default_low_pressure", "F=mid,S=mid,T=mid");
  const auto *mid_default_high
      = find_summary ("default_high_pressure", "F=mid,S=mid,T=mid");
  const auto *mid_time_only
      = find_summary ("time_only_low_pressure", "F=mid,S=mid,T=mid");

  bool ok = true;
  ok &= Check ("knob_sweep_default_low_no_worse_than_time_only",
               default_low_no_worse_than_time_only);
  ok &= Check ("knob_sweep_default_high_matches_time_only",
               default_high_matches_time_only);
  ok &= Check ("knob_sweep_mid_default_low_survives_horizon",
               mid_default_low
                   && mid_default_low->first_suppress_index
                          == static_cast<int> (gaps.size ()));
  ok &= Check ("knob_sweep_mid_default_high_suppresses_at_1m",
               mid_default_high
                   && GapIndexOrNever ("1m", gaps)
                          == mid_default_high->first_suppress_index);
  ok &= Check ("knob_sweep_mid_time_only_suppresses_at_1m",
               mid_time_only
                   && GapIndexOrNever ("1m", gaps)
                          == mid_time_only->first_suppress_index);

  return ok ? 0 : 1;
}
