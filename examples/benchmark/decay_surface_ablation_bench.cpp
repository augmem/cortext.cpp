#include "../../src/operations/eviction_ablation.hpp"
#include "../../src/operations/retrieval_debug_state.hpp"

#include <cortext/core/knobs.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/operations/memory_strength.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/signal.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <any>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{

constexpr int kEmbeddingDim = 256;
constexpr long long kTargetMemoryId = 101LL;
constexpr long long kDistractorMemoryId = 102LL;
constexpr long long kIrrelevantMemoryId = 103LL;
constexpr double kDaySeconds = 24.0 * 60.0 * 60.0;
constexpr double kFrozenHalfLifeSeconds = 3650.0 * kDaySeconds;
constexpr long long kBlockedStorageFloor = 1LL << 40;

Eigen::VectorXf
MakeVec (float first, float second)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = first;
  v[1] = second;
  const float norm = v.norm ();
  if (norm > 1e-9f)
    {
      v /= norm;
    }
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
  }
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
            const Eigen::VectorXf &embedding, long long created_at,
            double strength)
{
  store.Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?)",
      { memory_id, ToFloatVec (embedding), created_at });
  store.Execute (
      "INSERT INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, trace_fast, trace_med, trace_slow, trace_ultra, "
      "use_frequency, stability, connectivity, drift_mag, influence, sustained_influence, "
      "contextual_gain, redundancy, pre_activation, lability_state, suppression_count, "
      "created_at) "
      "VALUES(?, ?, ?, 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, ?, ?, ?, ?, ?, "
      "0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, ?)",
      { memory_id, memory_id, std::string ("chat/user"), created_at, strength,
        strength, strength * 0.5, strength * 0.2, strength * 0.05, created_at });
}

long long
CountMemory (cortext::Store &store, long long memory_id)
{
  auto rows = store.Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?",
      { memory_id });
  return std::any_cast<long long> (rows[0].at ("cnt"));
}

double
ReadStrength (cortext::Store &store, long long memory_id)
{
  auto rows = store.Execute (
      "SELECT strength FROM memories WHERE memory_id = ?",
      { memory_id });
  if (rows.empty ())
    {
      return 0.0;
    }
  return std::any_cast<double> (rows[0].at ("strength"));
}

void
ApplyGapDecay (cortext::Store &store, const cortext::SignalProcessor::Config &cfg,
               long long gap_ms)
{
  cortext::ProcessorContext p_ctx;
  p_ctx.half_life = cortext::core::BaseHalfLifePrior (cfg.stability);

  cortext::Signal signal;
  signal.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  signal.timestamp = static_cast<std::uint64_t> (gap_ms);
  signal.source_id = "gap";

  cortext::OperationContext ctx (signal, p_ctx, cfg, &store);
  ctx.SetMemoryUsageEvents (
      { { kTargetMemoryId, false }, { kDistractorMemoryId, false },
        { kIrrelevantMemoryId, false } });
  auto tx = store.Begin ();
  cortext::operations::UpdateMemoryStrength op;
  op.Execute (ctx, *tx);
  tx->Commit ();
}

struct RetrievalOutcome
{
  long long top1 = 0;
  int rank = -1;
  double top1_score = 0.0;
};

RetrievalOutcome
RunRetrieval (std::shared_ptr<cortext::Store> store,
              const cortext::SignalProcessor::Config &cfg,
              const Eigen::VectorXf &query,
              std::uint64_t signal_ts)
{
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  cortext::operations::retrieval_debug::ClearLastSelectedEmbeddingOrder ();
  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();

  cortext::Signal signal;
  signal.embedding = query;
  signal.timestamp = signal_ts;
  signal.source_id = "bench";
  (void)processor.Process (signal);

  RetrievalOutcome out;
  const auto &order
      = cortext::operations::retrieval_debug::GetLastSelectedEmbeddingOrder ();
  const auto &ranked
      = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  if (!order.empty ())
    {
      out.top1 = order.front ();
    }
  for (std::size_t i = 0; i < order.size (); ++i)
    {
      if (order[i] == kTargetMemoryId)
        {
          out.rank = static_cast<int> (i) + 1;
          break;
        }
    }
  if (!ranked.empty ())
    {
      out.top1_score = ranked.front ().score;
    }
  return out;
}

struct Scenario
{
  std::string name;
  double half_life_seconds = 0.0;
  long long min_storage_bytes = 0;
};

struct ScenarioRow
{
  std::string scenario;
  int gap_hours = 0;
  bool target_exists = false;
  double target_strength = 0.0;
  RetrievalOutcome exact;
  RetrievalOutcome paraphrase;
};

ScenarioRow
RunScenario (const Scenario &scenario, int gap_hours,
             const cortext::SignalProcessor::Config &cfg,
             const Eigen::VectorXf &target,
             const Eigen::VectorXf &distractor,
             const Eigen::VectorXf &irrelevant,
             const Eigen::VectorXf &query_exact,
             const Eigen::VectorXf &query_paraphrase)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  SeedMemory (*store, kTargetMemoryId, target, 1000LL, 1.0);
  SeedMemory (*store, kDistractorMemoryId, distractor, 1000LL, 1.0);
  SeedMemory (*store, kIrrelevantMemoryId, irrelevant, 1000LL, 1.0);

  cortext::operations::eviction::EvictionAblationOverride override;
  override.half_life = scenario.half_life_seconds;
  override.consolidation_gate_enabled = false;
  override.storage_gate_enabled = true;
  override.min_storage_bytes = scenario.min_storage_bytes;
  override.used_storage_bytes = 0;
  cortext::operations::eviction::ScopedEvictionAblationOverride scoped (
      override);

  const long long gap_ms
      = static_cast<long long> (gap_hours) * 60LL * 60LL * 1000LL;
  ApplyGapDecay (*store, cfg, gap_ms);

  ScenarioRow row;
  row.scenario = scenario.name;
  row.gap_hours = gap_hours;
  row.target_exists = (CountMemory (*store, kTargetMemoryId) > 0);
  row.target_strength = ReadStrength (*store, kTargetMemoryId);

  const std::uint64_t retrieval_ts = static_cast<std::uint64_t> (gap_ms + 1);
  row.exact = RunRetrieval (store, cfg, query_exact, retrieval_ts);
  row.paraphrase
      = RunRetrieval (store, cfg, query_paraphrase, retrieval_ts + 1);
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
  bool ok = true;

  const Eigen::VectorXf target = MakeVec (0.96f, 0.28f);
  const Eigen::VectorXf distractor = MakeVec (0.86f, 0.50f);
  const Eigen::VectorXf irrelevant = MakeVec (0.10f, 0.99f);
  const Eigen::VectorXf query_exact = MakeVec (0.98f, 0.20f);
  const Eigen::VectorXf query_paraphrase = MakeVec (0.93f, 0.36f);

  BenchEncoder encoder;
  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.encoder = &encoder;

  const std::vector<Scenario> scenarios = {
    { "decay_eviction_allowed", kDaySeconds, 0LL },
    { "decay_eviction_blocked", kDaySeconds, kBlockedStorageFloor },
    { "frozen_eviction_blocked", kFrozenHalfLifeSeconds, kBlockedStorageFloor },
  };
  const std::vector<int> gaps_hours = { 0, 1, 6, 24, 72, 168 };

  std::vector<ScenarioRow> rows;
  rows.reserve (scenarios.size () * gaps_hours.size ());
  for (const auto &scenario : scenarios)
    {
      for (int gap_hours : gaps_hours)
        {
          rows.push_back (RunScenario (scenario, gap_hours, cfg, target,
                                       distractor, irrelevant, query_exact,
                                       query_paraphrase));
        }
    }

  std::cout << "scenario gap_h target_exists target_strength exact_top1 "
               "exact_rank paraphrase_top1 paraphrase_rank\n";
  std::cout << std::fixed << std::setprecision (6);
  for (const auto &row : rows)
    {
      std::cout << row.scenario << " " << row.gap_hours << " "
                << (row.target_exists ? 1 : 0) << " " << row.target_strength
                << " " << row.exact.top1 << " " << row.exact.rank << " "
                << row.paraphrase.top1 << " " << row.paraphrase.rank << "\n";
    }

  auto find_row = [&rows] (const std::string &scenario,
                           int gap_hours) -> const ScenarioRow * {
    for (const auto &row : rows)
      {
        if (row.scenario == scenario && row.gap_hours == gap_hours)
          {
            return &row;
          }
      }
    return nullptr;
  };

  bool blocked_exact_all = true;
  bool blocked_paraphrase_all = true;
  bool frozen_exact_all = true;
  bool frozen_paraphrase_all = true;
  bool blocked_matches_frozen_all = true;
  for (int gap_hours : gaps_hours)
    {
      const auto *blocked = find_row ("decay_eviction_blocked", gap_hours);
      const auto *frozen = find_row ("frozen_eviction_blocked", gap_hours);
      if (!blocked || !frozen)
        {
          ok = false;
          continue;
        }
      blocked_exact_all
          = blocked_exact_all && blocked->exact.top1 == kTargetMemoryId;
      blocked_paraphrase_all
          = blocked_paraphrase_all && blocked->paraphrase.top1 == kTargetMemoryId;
      frozen_exact_all
          = frozen_exact_all && frozen->exact.top1 == kTargetMemoryId;
      frozen_paraphrase_all
          = frozen_paraphrase_all && frozen->paraphrase.top1 == kTargetMemoryId;
      blocked_matches_frozen_all
          = blocked_matches_frozen_all
            && blocked->exact.top1 == frozen->exact.top1
            && blocked->paraphrase.top1 == frozen->paraphrase.top1
            && blocked->exact.rank == frozen->exact.rank
            && blocked->paraphrase.rank == frozen->paraphrase.rank;
    }

  const auto *blocked_168 = find_row ("decay_eviction_blocked", 168);
  const auto *frozen_168 = find_row ("frozen_eviction_blocked", 168);
  const auto *allowed_6 = find_row ("decay_eviction_allowed", 6);
  const auto *allowed_24 = find_row ("decay_eviction_allowed", 24);
  const auto *allowed_168 = find_row ("decay_eviction_allowed", 168);

  ok &= Check ("surfacing_blocked_decay_exact_top1_all", blocked_exact_all);
  ok &= Check ("surfacing_blocked_decay_paraphrase_top1_all",
               blocked_paraphrase_all);
  ok &= Check ("surfacing_frozen_exact_top1_all", frozen_exact_all);
  ok &= Check ("surfacing_frozen_paraphrase_top1_all", frozen_paraphrase_all);
  ok &= Check ("surfacing_blocked_decay_matches_frozen_all",
               blocked_matches_frozen_all);
  ok &= Check ("surfacing_blocked_decay_strength_lt_frozen_168h",
               blocked_168 && frozen_168
                   && blocked_168->target_strength < frozen_168->target_strength);
  ok &= Check ("surfacing_eviction_allowed_exists_at_6h",
               allowed_6 && allowed_6->target_exists);
  ok &= Check ("surfacing_eviction_allowed_missing_at_24h",
               allowed_24 && !allowed_24->target_exists);
  ok &= Check ("surfacing_eviction_allowed_missing_at_168h",
               allowed_168 && !allowed_168->target_exists);

  return ok ? 0 : 1;
}
