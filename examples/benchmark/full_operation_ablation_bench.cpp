#include "../../src/operations/constructive_recall_internal.hpp"
#include "../../src/operations/meta_learning_internal.hpp"
#include "../../src/operations/retrieval_debug_state.hpp"

#include <cortext/core/sparse.hpp>
#include <cortext/core/constants.hpp>
#include <cortext/core/utils.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/competition.hpp>
#include <cortext/operations/detect_memory_usage.hpp>
#include <cortext/operations/emotion.hpp>
#include <cortext/operations/graph_build.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/operations/interrupt_gate.hpp>
#include <cortext/operations/metacognitive.hpp>
#include <cortext/operations/reconsolidation.hpp>
#include <cortext/operations/write_gate.hpp>
#include <cortext/operations/focus.hpp>
#include <cortext/operations/sensitivity.hpp>
#include <cortext/operations/stability.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/accumulator_state.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
constexpr int kEmbeddingDim = 256;
constexpr double kScoreEpsilon = 1e-9;

enum FamilyBit : int
{
  kSourceConfidence = 0,
  kPredictiveRetrieval,
  kConstructiveRecall,
  kProceduralMemory,
  kMetacognitiveControl,
  kAffectInterrupt,
  kAffectRetrieval,
  kFlashbulbConsolidation,
  kNeuromodulation,
  kMetaLearning,
  kReinforcementEdges,
  kSequentialEdges,
  kFamilyCount
};

struct FamilyInfo
{
  const char *name;
  const char *slug;
};

constexpr std::array<FamilyInfo, kFamilyCount> kFamilies = {
  FamilyInfo{ "source_confidence", "source_confidence" },
  FamilyInfo{ "predictive_retrieval", "predictive_retrieval" },
  FamilyInfo{ "constructive_recall", "constructive_recall" },
  FamilyInfo{ "procedural_memory", "procedural_memory" },
  FamilyInfo{ "metacognitive_control", "metacognitive_control" },
  FamilyInfo{ "affect_interrupt", "affect_interrupt" },
  FamilyInfo{ "affect_retrieval", "affect_retrieval" },
  FamilyInfo{ "flashbulb_consolidation", "flashbulb_consolidation" },
  FamilyInfo{ "neuromodulation", "neuromodulation" },
  FamilyInfo{ "meta_learning", "meta_learning" },
  FamilyInfo{ "reinforcement_edges", "reinforcement_edges" },
  FamilyInfo{ "sequential_edges", "sequential_edges" },
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

BenchEncoder &
GetBenchEncoder ()
{
  static BenchEncoder encoder;
  return encoder;
}

class ScopedEnvVar
{
public:
  explicit ScopedEnvVar (const char *name) : name_ (name)
  {
    const char *existing = std::getenv (name);
    if (existing != nullptr)
      {
        had_value_ = true;
        old_value_ = existing;
      }
    unsetenv (name_);
  }

  ScopedEnvVar (const char *name, const std::string &value) : ScopedEnvVar (name)
  {
    setenv (name_, value.c_str (), 1);
  }

  ~ScopedEnvVar ()
  {
    if (had_value_)
      {
        setenv (name_, old_value_.c_str (), 1);
      }
    else
      {
        unsetenv (name_);
      }
  }

private:
  const char *name_;
  bool had_value_ = false;
  std::string old_value_;
};

class ScopedMaskEnv
{
public:
  explicit ScopedMaskEnv (std::uint32_t mask)
  {
    ConfigureToggle ("CORTEXT_DISABLE_SOURCE_CONF",
                     FamilyEnabled (mask, kSourceConfidence));
    ConfigureToggle ("CORTEXT_DISABLE_PREDICTIVE_RETRIEVAL_BONUS",
                     FamilyEnabled (mask, kPredictiveRetrieval));
    ConfigureToggle ("CORTEXT_DISABLE_CONSTRUCTIVE_RECALL",
                     FamilyEnabled (mask, kConstructiveRecall));
    ConfigureToggle ("CORTEXT_DISABLE_META_LEARNING",
                     FamilyEnabled (mask, kMetaLearning));

    ConfigureToggle ("CORTEXT_DISABLE_METACOG_TOT_RECOVERY",
                     FamilyEnabled (mask, kMetacognitiveControl));
    ConfigureToggle ("CORTEXT_DISABLE_METACOG_UNKNOWN_CAUTION",
                     FamilyEnabled (mask, kMetacognitiveControl));
    ConfigureToggle ("CORTEXT_DISABLE_METACOG_CONFIDENCE_DECAY",
                     FamilyEnabled (mask, kMetacognitiveControl));

    ConfigureToggle ("CORTEXT_DISABLE_NEUROMOD_WRITE_SCALE",
                     FamilyEnabled (mask, kNeuromodulation));
    ConfigureToggle ("CORTEXT_DISABLE_NEUROMOD_RECONSOLIDATION_SCALE",
                     FamilyEnabled (mask, kNeuromodulation));
    ConfigureToggle ("CORTEXT_DISABLE_NEUROMOD_COMPETITION_SCALE",
                     FamilyEnabled (mask, kNeuromodulation));
    ConfigureToggle ("CORTEXT_DISABLE_NEUROMOD_VALUE_GAIN",
                     FamilyEnabled (mask, kNeuromodulation));
  }

private:
  static bool
  FamilyEnabled (std::uint32_t mask, int bit)
  {
    return (mask & (1u << bit)) != 0u;
  }

  void
  ConfigureToggle (const char *name, bool enabled)
  {
    if (enabled)
      {
        guards_.push_back (std::make_unique<ScopedEnvVar> (name));
      }
    else
      {
        guards_.push_back (std::make_unique<ScopedEnvVar> (name, "1"));
      }
  }

  std::vector<std::unique_ptr<ScopedEnvVar>> guards_;
};

class NullTransaction : public cortext::Transaction
{
public:
  std::unique_ptr<cortext::Transaction>
  Begin () override
  {
    return std::make_unique<NullTransaction> ();
  }

  std::vector<std::map<std::string, std::any>>
  Execute (const std::string &, const std::vector<std::any> & = {}) override
  {
    return {};
  }

  void
  Commit () override
  {
  }

  void
  Rollback () override
  {
  }
};

Eigen::VectorXf
Norm (Eigen::VectorXf v)
{
  const float n = v.norm ();
  if (n > 1e-9f)
    {
      v /= n;
    }
  return v;
}

Eigen::VectorXf
MakeVec (std::initializer_list<std::pair<int, float>> values)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  for (const auto &[idx, value] : values)
    {
      if (idx >= 0 && idx < kEmbeddingDim)
        {
          v[idx] = value;
        }
    }
  return Norm (v);
}

std::vector<float>
ToFloatVec (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

long long
AnyToInt64 (const std::any &value)
{
  if (value.type () == typeid (long long))
    {
      return std::any_cast<long long> (value);
    }
  if (value.type () == typeid (int))
    {
      return static_cast<long long> (std::any_cast<int> (value));
    }
  return 0;
}

double
AnyToDouble (const std::any &value)
{
  if (value.type () == typeid (double))
    {
      return std::any_cast<double> (value);
    }
  if (value.type () == typeid (float))
    {
      return static_cast<double> (std::any_cast<float> (value));
    }
  if (value.type () == typeid (int))
    {
      return static_cast<double> (std::any_cast<int> (value));
    }
  if (value.type () == typeid (long long))
    {
      return static_cast<double> (std::any_cast<long long> (value));
    }
  return 0.0;
}

cortext::Signal
MakeSignal (const Eigen::VectorXf &embedding, std::uint64_t ts,
            const std::string &source_id = "bench")
{
  cortext::Signal signal;
  signal.embedding = embedding;
  signal.timestamp = ts;
  signal.source_id = source_id;
  signal.modality = "text";
  signal.mimetype = "text/plain";
  return signal;
}

std::shared_ptr<cortext::Store>
CreateStore ()
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);
  return store;
}

void
ResetStore (cortext::Store &store)
{
  const std::array<const char *, 11> tables = {
    "associations",
    "memory_reconstructions",
    "fact_retrieval_events",
    "fact_support",
    "facts",
    "signals",
    "episodes",
    "memories",
    "embeddings",
    "meta_learning_coeffs",
    "summaries",
  };
  for (const char *table : tables)
    {
      try
        {
          store.Execute (std::string ("DELETE FROM ") + table, {});
        }
      catch (const std::exception &)
        {
        }
    }
}

std::shared_ptr<cortext::Store>
GetScenarioStore (int slot)
{
  static std::array<std::shared_ptr<cortext::Store>, kFamilyCount> stores;
  auto &store = stores[static_cast<std::size_t> (slot)];
  if (!store)
    {
      store = CreateStore ();
    }
  ResetStore (*store);
  return store;
}

void
SeedMemory (cortext::Store &store, long long memory_id, long long embedding_id,
            const Eigen::VectorXf &embedding, const std::string &kind = "LONG_TERM",
            long long created_at = 1, double strength = 1.0)
{
  store.Execute (
      "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { embedding_id, ToFloatVec (embedding), created_at });
  store.Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, created_at) "
      "VALUES(?, ?, 'bench', ?, ?, 1, 'text', 0.5, 0.5, ?, ?)",
      { memory_id, embedding_id, kind, created_at, strength, created_at });
}

bool
FamilyEnabled (std::uint32_t mask, int bit)
{
  return (mask & (1u << bit)) != 0u;
}

cortext::SignalProcessor::Config
MakeConfig (std::uint32_t mask, double focus, double sensitivity,
            double stability)
{
  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = focus;
  cfg.sensitivity = sensitivity;
  cfg.stability = stability;
  cfg.procedural_enabled = FamilyEnabled (mask, kProceduralMemory);
  cfg.affect_interrupt = FamilyEnabled (mask, kAffectInterrupt);
  cfg.affect_retrieval = FamilyEnabled (mask, kAffectRetrieval);
  cfg.reinforcement_enabled = FamilyEnabled (mask, kReinforcementEdges);
  cfg.sequential_edges_enabled = FamilyEnabled (mask, kSequentialEdges);
  return cfg;
}

class ForceRetrievalGateOp : public cortext::IOperation
{
public:
  explicit ForceRetrievalGateOp (
      std::optional<cortext::ProcessorContext::MetacognitiveMode> mode
          = std::nullopt,
      std::optional<double> metacognitive_confidence = std::nullopt)
      : mode_ (mode), metacognitive_confidence_ (metacognitive_confidence)
  {
  }

  void
  Execute (cortext::OperationContext &ctx,
           cortext::Transaction & /*tx*/) const override
  {
    ctx.SetShouldCheckRetrieval (true);
    auto &pctx = ctx.GetProcessorContext ();
    if (pctx.memory_stream.empty ())
      {
        pctx.memory_stream.push_back (ctx.GetSignal ().embedding);
      }
    auto &acc = pctx.accumulator_states[ctx.GetSignal ().source_id];
    acc.mu_acc = ctx.GetSignal ().embedding;
    acc.c_t = ctx.GetSignal ().embedding;
    if (mode_.has_value ())
      {
        pctx.metacognitive_mode = *mode_;
        pctx.metacognitive_mode_expires_at = ctx.GetSignal ().timestamp + 1000;
        pctx.metacognitive_certainty_satisfied = false;
      }
    if (metacognitive_confidence_.has_value ())
      {
        pctx.metacognitive_confidence = *metacognitive_confidence_;
      }
  }

private:
  std::optional<cortext::ProcessorContext::MetacognitiveMode> mode_;
  std::optional<double> metacognitive_confidence_;
};

class SeedProceduralStoreOp : public cortext::IOperation
{
public:
  SeedProceduralStoreOp (long long memory_id, double score)
      : memory_id_ (memory_id), score_ (score)
  {
  }

  void
  Execute (cortext::OperationContext &ctx,
           cortext::Transaction & /*tx*/) const override
  {
    const int key_size = cortext::core::SparseKeySize (ctx.GetConfig ().focus, ctx.GetConfig ().sensitivity, ctx.GetConfig ().stability);
    const std::string key = cortext::core::SparseKey (ctx.GetSignal ().embedding,
                                                      key_size);
    ctx.GetProcessorContext ().procedural_store[key][memory_id_] = score_;
  }

private:
  long long memory_id_ = 0;
  double score_ = 0.0;
};

class SetAffectInputsOp : public cortext::IOperation
{
public:
  SetAffectInputsOp (double emotion_intensity, double arousal, double salience)
      : emotion_intensity_ (emotion_intensity), arousal_ (arousal),
        salience_ (salience)
  {
  }

  void
  Execute (cortext::OperationContext &ctx,
           cortext::Transaction & /*tx*/) const override
  {
    ctx.SetEmotionIntensity (emotion_intensity_);
    ctx.SetArousal (arousal_);
    ctx.SetMetric (cortext::operations::Metric::salience, salience_);
  }

private:
  double emotion_intensity_ = 0.0;
  double arousal_ = 0.0;
  double salience_ = 0.0;
};

class SetupStoredIdOp : public cortext::IOperation
{
public:
  explicit SetupStoredIdOp (std::optional<long long> stored_id)
      : stored_id_ (stored_id)
  {
  }

  void
  Execute (cortext::OperationContext &ctx,
           cortext::Transaction & /*tx*/) const override
  {
    ctx.SetStoredEmbeddingId (stored_id_);
  }

private:
  std::optional<long long> stored_id_;
};

class SetConsolidationStartOp : public cortext::IOperation
{
public:
  void
  Execute (cortext::OperationContext &ctx,
           cortext::Transaction & /*tx*/) const override
  {
    ctx.SetConsolidationShouldStart (true);
  }
};

class SetNeuromodOp : public cortext::IOperation
{
public:
  SetNeuromodOp (double ach, double ne) : ach_ (ach), ne_ (ne) {}

  void
  Execute (cortext::OperationContext &ctx,
           cortext::Transaction & /*tx*/) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    pctx.neuromod_ach = ach_;
    pctx.neuromod_ne = ne_;
  }

private:
  double ach_ = 0.0;
  double ne_ = 0.0;
};

class SetupReconInputsOp : public cortext::IOperation
{
public:
  SetupReconInputsOp (Eigen::VectorXf cur,
                      std::unordered_map<long long, Eigen::VectorXf> retrieved)
      : cur_ (std::move (cur)), retrieved_ (std::move (retrieved))
  {
  }

  void
  Execute (cortext::OperationContext &ctx,
           cortext::Transaction & /*tx*/) const override
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

void
SetupAccumulatorState (cortext::ProcessorContext &pctx, const cortext::Signal &s,
                       double s_max, double s_sum, int n_signals)
{
  cortext::AccumulatorState acc;
  acc.mu_acc = s.embedding;
  acc.e_peak = s.embedding;
  acc.s_max = s_max;
  acc.s_sum = s_sum;
  acc.n_signals = n_signals;
  acc.t_start = s.timestamp - 5000;
  acc.last_write_ts = 0;
  pctx.accumulator_states[s.source_id] = std::move (acc);
}

double
RunSourceConfidenceScenario (std::uint32_t mask)
{
  ScopedMaskEnv guards (mask);
  auto store = GetScenarioStore (kSourceConfidence);

  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf low_conf = query;
  const Eigen::VectorXf high_conf = MakeVec ({ { 0, 0.94f }, { 1, 0.341f } });
  const long long ts = 7200000;

  SeedMemory (*store, 10LL, 10LL, low_conf, "LONG_TERM", 1);
  SeedMemory (*store, 20LL, 20LL, high_conf, "LONG_TERM", 1);
  store->Execute ("UPDATE memories SET source_contradiction_count = ? WHERE memory_id = ?",
                  { 7LL, 10LL });
  store->Execute ("UPDATE memories SET source_contradiction_count = ? WHERE memory_id = ?",
                  { 0LL, 20LL });

  auto cfg = MakeConfig (mask, 0.5, 0.5, 1.0);
  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (query, ts));
  processor.Flush ();
  const auto ranked = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  if (ranked.empty ())
    {
      return 0.0;
    }
  return ranked.front ().memory_id == 20LL ? 1.0 : 0.0;
}

double
RunPredictiveScenario (std::uint32_t mask)
{
  ScopedMaskEnv guards (mask);
  auto store = GetScenarioStore (kPredictiveRetrieval);

  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf predictive_target
      = MakeVec ({ { 0, 0.8f }, { 1, 0.6f } });
  const Eigen::VectorXf raw_best
      = MakeVec ({ { 0, 0.95f }, { 1, 0.3122499f } });

  SeedMemory (*store, 11LL, 11LL, predictive_target, "LONG_TERM", 1);
  SeedMemory (*store, 22LL, 22LL, raw_best, "LONG_TERM", 1);
  store->Execute ("UPDATE memories SET pre_activation = 1.0 WHERE memory_id = ?",
                  { 11LL });

  auto cfg = MakeConfig (mask, 1.0, 0.5, 0.5);
  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (query, 10));
  processor.Flush ();
  const auto ranked = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  if (ranked.empty ())
    {
      return 0.0;
    }
  return ranked.front ().memory_id == 11LL ? 1.0 : 0.0;
}

double
RunConstructiveRecallScenario (std::uint32_t mask)
{
  ScopedMaskEnv guards (mask);
  auto store = GetScenarioStore (kConstructiveRecall);

  const Eigen::VectorXf query = MakeVec ({ { 0, 0.98f }, { 1, 0.19f } });
  const Eigen::VectorXf evidence = MakeVec ({ { 0, 0.84f }, { 1, 0.54f } });
  // Competitor sits between the stale evidence (sim ~0.93) and the
  // reconstructed current embedding (sim 1.0): off -> competitor wins,
  // on -> reconstruction routing wins, with margins that survive the
  // post-remap score terms instead of a 0.006 knife edge.
  const Eigen::VectorXf competitor = MakeVec ({ { 0, 0.90f }, { 1, 0.436f } });
  const Eigen::VectorXf reconstructed = query;

  SeedMemory (*store, 11LL, 11LL, evidence, "LONG_TERM", 1);
  SeedMemory (*store, 22LL, 22LL, competitor, "LONG_TERM", 1);
  auto tx = store->Begin ();
  cortext::operations::constructive_recall::AppendReconstructionWithEmbedding (
      *tx, 11LL, reconstructed, {}, 2, 0.2, "bench");
  tx->Commit ();

  auto cfg = MakeConfig (mask, 0.5, 0.5, 0.5);
  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (query, 10));
  processor.Flush ();
  const auto ranked = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  if (ranked.empty ())
    {
      return 0.0;
    }
  return ranked.front ().memory_id == 11LL ? 1.0 : 0.0;
}

double
RunProceduralScenario (std::uint32_t mask)
{
  ScopedMaskEnv guards (mask);
  auto store = GetScenarioStore (kProceduralMemory);

  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf routine_target = MakeVec ({ { 0, 0.30f }, { 1, 0.9539392f } });
  SeedMemory (*store, 500LL, 500LL, routine_target, "LONG_TERM", 1);
  for (int i = 0; i < 9; ++i)
    {
      const float x = 0.44f - 0.01f * static_cast<float> (i);
      const float y = std::sqrt (std::max (0.0f, 1.0f - x * x));
      const long long id = 600LL + i;
      SeedMemory (*store, id, id, MakeVec ({ { 0, x }, { 1, y } }), "LONG_TERM",
                  1);
    }

  auto cfg = MakeConfig (mask, 1.0, 1.0, 0.5);
  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<SeedProceduralStoreOp> (500LL, 1.0),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (query, 10));
  processor.Flush ();
  const auto ranked = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  return std::any_of (ranked.begin (), ranked.end (), [] (const auto &candidate) {
    return candidate.memory_id == 500LL;
  }) ? 1.0 : 0.0;
}

std::vector<long long>
RunTotRetrievalForMask (std::uint32_t mask, double confidence)
{
  ScopedMaskEnv guards (mask);
  auto store = GetScenarioStore (kMetacognitiveControl);

  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  SeedMemory (*store, 1LL, 1LL, query, "LONG_TERM", 1);
  for (long long id = 2; id <= 4; ++id)
    {
      SeedMemory (*store, id, id, MakeVec ({ { 1, 1.0f } }), "ASSOCIATION", 1);
    }
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'co_occurs', 1.0)",
      { 1LL, 2LL });
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'co_occurs', 1.0)",
      { 2LL, 3LL });
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'co_occurs', 1.0)",
      { 3LL, 4LL });

  auto cfg = MakeConfig (mask, 0.5, 0.5, 1.0);
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (
          cortext::ProcessorContext::MetacognitiveMode::TotRecovery,
          confidence),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto out = processor.Process (MakeSignal (query, 10));
  processor.Flush ();
  return out.candidate_memory_ids;
}

double
RunMetacognitiveScenario (std::uint32_t mask)
{
  const auto tot_ids = RunTotRetrievalForMask (mask, 1.0);
  const bool tot_hit
      = std::find (tot_ids.begin (), tot_ids.end (), 4LL) != tot_ids.end ();

  bool unknown_empty = false;
  {
    ScopedMaskEnv guards (mask);
    auto store = GetScenarioStore (kMetacognitiveControl);
    const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
    SeedMemory (*store, 77LL, 77LL, query, "LONG_TERM", 1);
    store->Execute ("UPDATE memories SET source_contradiction_count = ? WHERE memory_id = ?",
                    { 5LL, 77LL });
    auto cfg = MakeConfig (mask, 0.5, 0.5, 0.5);
    auto ops = std::make_unique<cortext::DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (
            cortext::ProcessorContext::MetacognitiveMode::UnknownCaution, 0.0),
        std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
    cortext::SignalProcessor processor (cfg, store, std::move (ops));
    auto out = processor.Process (MakeSignal (query, 20));
    processor.Flush ();
    unknown_empty = out.candidate_memory_ids.empty ();
  }

  bool decay_blocks = false;
  {
    ScopedMaskEnv guards (mask);
    cortext::Signal signal = MakeSignal (MakeVec ({ { 0, 1.0f } }), 11000);
    cortext::ProcessorContext pctx;
    pctx.metacognitive_confidence = 1.0;
    pctx.last_signal_timestamp = 1000;
    auto cfg = MakeConfig (mask, 0.5, 0.5, 0.0);
    cortext::operations::MetacognitiveMonitoring op;
    cortext::OperationContext ctx (signal, pctx, cfg);
    ctx.SetFeelingOfKnowing (0.2);
    ctx.SetMemoryUsageEvents ({ { 1LL, true, -1.0 } });
    NullTransaction tx;
    op.Execute (ctx, tx);
    const auto low_ids = RunTotRetrievalForMask (mask, pctx.metacognitive_confidence);
    const bool low_hit
        = std::find (low_ids.begin (), low_ids.end (), 4LL) != low_ids.end ();
    decay_blocks = !low_hit;
  }

  double total = 0.0;
  total += tot_hit ? 1.0 : 0.0;
  total += unknown_empty ? 1.0 : 0.0;
  total += decay_blocks ? 1.0 : 0.0;
  return total / 3.0;
}

double
RunAffectInterruptScenario (std::uint32_t mask)
{
  auto store = GetScenarioStore (kAffectInterrupt);
  const Eigen::VectorXf signal_emb = MakeVec ({ { 0, 1.0f }, { 1, 0.02f } });
  const Eigen::VectorXf cand_emb = MakeVec ({ { 0, 0.93f }, { 1, 0.367f } });
  SeedMemory (*store, 1LL, 1LL, cand_emb, "LONG_TERM", 0);

  auto evaluate = [&] (bool affect_interrupt_enabled, double threshold_t) {
    auto cfg = MakeConfig (mask, 0.7, 0.9, 0.2);
    cfg.affect_interrupt = affect_interrupt_enabled;

    cortext::ProcessorContext pc;
    pc.signals_processed = 100;
    pc.last_interrupt_tick = 0;
    pc.recent_memory_centroids.push_back (
        MakeVec ({ { 0, 1.0f }, { 1, 0.0f } }));

    cortext::Signal signal = MakeSignal (signal_emb, 1000, "affect_interrupt");
    cortext::OperationContext oc (signal, pc, cfg, store.get ());
    oc.SetCoherence (1.0);
    oc.SetThresholdTDynamic (threshold_t);
    oc.SetAtBoundary (false);
    oc.SetEmotionIntensity (1.0);
    oc.SetArousal (1.0);
    oc.SetMetric (cortext::operations::Metric::salience, 1.0);
    oc.SetRetrievedMemoryEmbeddings ({ { 1LL, cand_emb } });

    cortext::operations::ComputeMniGateDecision op;
    auto tx = store->Begin ();
    op.Execute (oc, *tx);
    return oc.GetInterruptAllowed ();
  };

  if (!FamilyEnabled (mask, kAffectInterrupt))
    {
      return 0.0;
    }

  for (double threshold_t = 0.20; threshold_t <= 0.80; threshold_t += 0.02)
    {
      const bool off_allowed = evaluate (false, threshold_t);
      const bool on_allowed = evaluate (true, threshold_t);
      if (on_allowed && !off_allowed)
        {
          return 1.0;
        }
    }
  return 0.0;
}

double
RunAffectRetrievalScenario (std::uint32_t mask)
{
  ScopedMaskEnv guards (mask);
  auto store = GetScenarioStore (kAffectRetrieval);
  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf affect_target = MakeVec ({ { 0, 0.90f }, { 1, 0.43589f } });
  const Eigen::VectorXf raw_best = MakeVec ({ { 0, 0.96f }, { 1, 0.28f } });

  SeedMemory (*store, 11LL, 11LL, affect_target, "LONG_TERM", 1);
  SeedMemory (*store, 22LL, 22LL, raw_best, "LONG_TERM", 1);
  store->Execute (
      "UPDATE memories SET emotional_intensity = 1.0, s_arousal_avg = 1.0 WHERE memory_id = ?",
      { 11LL });
  store->Execute (
      "UPDATE memories SET emotional_intensity = 0.0, s_arousal_avg = 0.0 WHERE memory_id = ?",
      { 22LL });

  auto cfg = MakeConfig (mask, 0.5, 1.0, 0.5);
  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<SetAffectInputsOp> (1.0, 1.0, 1.0),
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (query, 10));
  processor.Flush ();
  const auto ranked = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  if (ranked.empty ())
    {
      return 0.0;
    }
  return ranked.front ().memory_id == 11LL ? 1.0 : 0.0;
}

double
RunFlashbulbScenario (std::uint32_t mask)
{
  ScopedMaskEnv guards (mask);
  auto store = GetScenarioStore (kFlashbulbConsolidation);
  SeedMemory (*store, 101LL, 101LL, MakeVec ({ { 0, 1.0f } }), "LONG_TERM", 1);
  store->Execute (
      "UPDATE memories SET s_emotion_max = ?, s_arousal_avg = ? WHERE memory_id = ?",
      { 0.95, 0.85, 101LL });

  auto cfg = MakeConfig (mask, 0.4, 0.8, 0.5);
  std::unique_ptr<cortext::DynamicOperationSet> ops;
  if (FamilyEnabled (mask, kFlashbulbConsolidation))
    {
      ops = std::make_unique<cortext::DynamicOperationSet> (
          std::make_unique<SetupStoredIdOp> (101LL),
          std::make_unique<cortext::operations::ApplyEmotionalConsolidation> ());
    }
  else
    {
      ops = std::make_unique<cortext::DynamicOperationSet> (
          std::make_unique<SetupStoredIdOp> (101LL));
    }
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (MakeVec ({ { 0, 1.0f } }), 12345));
  processor.Flush ();

  auto rows = store->Execute ("SELECT flashbulb FROM memories WHERE memory_id = ?",
                              { 101LL });
  if (rows.empty ())
    {
      return 0.0;
    }
  return AnyToInt64 (rows[0].at ("flashbulb")) == 1 ? 1.0 : 0.0;
}

double
RunNeuromodulationScenario (std::uint32_t mask)
{
  ScopedMaskEnv guards (mask);
  cortext::Signal signal = MakeSignal (MakeVec ({ { 0, 1.0f } }), 100000,
                                       "neuromod");
  cortext::ProcessorContext pctx;
  auto cfg = MakeConfig (mask, 0.5, 0.5, 0.5);
  SetupAccumulatorState (pctx, signal, 0.5, 1.0, 2);
  pctx.neuromod_ne = 1.0;

  cortext::OperationContext ctx (signal, pctx, cfg);
  ctx.SetFlushRequired (true);
  ctx.SetThresholdTDynamic (0.6);
  cortext::operations::ComputeWriteGate op;
  NullTransaction tx;
  op.Execute (ctx, tx);
  return ctx.GetAccumulatorWriteDecision () ? 1.0 : 0.0;
}

double
RunMetaLearningScenario (std::uint32_t mask)
{
  ScopedMaskEnv guards (mask);
  auto store = GetScenarioStore (kMetaLearning);
  cortext::ProcessorContext pctx;
  auto cfg = MakeConfig (mask, 0.5, 0.5, 0.5);

  auto tx = store->Begin ();
  cortext::OperationContext init_ctx (MakeSignal (MakeVec ({ { 0, 1.0f } }), 1000),
                                      pctx, cfg, store.get ());
  cortext::operations::InitializeFocusPriors focus;
  cortext::operations::InitializeSensitivityPriors sensitivity;
  cortext::operations::InitializeStabilityPriors stability;
  focus.Execute (init_ctx, *tx);
  sensitivity.Execute (init_ctx, *tx);
  stability.Execute (init_ctx, *tx);

  cortext::operations::ApplyMetaLearning meta;
  for (int i = 0; i < 8; ++i)
    {
      cortext::OperationContext ctx (
          MakeSignal (MakeVec ({ { 0, 1.0f } }),
                      2000 + static_cast<std::uint64_t> (i) * 1000),
          pctx, cfg, store.get ());
      pctx.attention_width = static_cast<double> (cortext::core::kAttentionWidthMin);
      pctx.rate_target = 4.8;
      pctx.rho_hat_prev = 4.8;
      pctx.hysteresis = 0.22;
      pctx.u_t = 0.1;
      pctx.last_used_flag = 1.0;
      pctx.delta_reward = 0.6;
      ctx.SetWriteDecision (true);
      meta.Execute (ctx, *tx);
    }

  auto rows = tx->Execute (
      "SELECT COALESCE(SUM(update_count), 0) AS updates FROM meta_learning_coeffs");
  const long long updates
      = rows.empty () ? 0LL : AnyToInt64 (rows[0].at ("updates"));
  return updates > 0 ? 1.0 : 0.0;
}

double
RunReinforcementScenario (std::uint32_t mask)
{
  ScopedMaskEnv guards (mask);
  auto store = GetScenarioStore (kReinforcementEdges);
  // Reinforcement steps are support-weighted (geometric mean of contextual
  // support), so an exactly-orthogonal co-retrieved pair earns no edge by
  // design. The claim under test: co-retrieved memories with shared context
  // gain a reinforces edge.
  const Eigen::VectorXf emb_a = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf emb_b = MakeVec ({ { 0, 0.6f }, { 1, 0.8f } });
  SeedMemory (*store, 1LL, 1LL, emb_a, "LONG_TERM", 1);
  SeedMemory (*store, 2LL, 2LL, emb_b, "LONG_TERM", 1);

  auto cfg = MakeConfig (mask, 0.5, 0.5, 0.5);
  cortext::ProcessorContext pctx;
  cortext::Signal signal = MakeSignal (emb_a, 100);
  cortext::OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetRetrievedMemoryEmbeddings ({ { 1LL, emb_a }, { 2LL, emb_b } });
  ctx.SetSelectedCandidateId (1LL);
  ctx.SetInterruptAllowed (true);
  cortext::operations::DetectMemoryUsage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto rows = store->Execute (
      "SELECT COUNT(*) AS c FROM associations WHERE edge_type = 'reinforces'",
      {});
  const long long count = rows.empty () ? 0LL : AnyToInt64 (rows[0].at ("c"));
  return count > 0 ? 1.0 : 0.0;
}

double
RunSequentialScenario (std::uint32_t mask)
{
  ScopedMaskEnv guards (mask);
  auto store = GetScenarioStore (kSequentialEdges);
  SeedMemory (*store, 1LL, 1LL, MakeVec ({ { 0, 1.0f } }), "LONG_TERM", 1);
  SeedMemory (*store, 2LL, 2LL, MakeVec ({ { 0, 0.95f }, { 1, 0.31f } }),
              "LONG_TERM", 2);
  store->Execute (
      "UPDATE memories SET cluster_id = 1, boundary_score = 0.0 WHERE memory_id IN (1, 2)");

  auto cfg = MakeConfig (mask, 0.5, 0.5, 0.5);
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<SetConsolidationStartOp> (),
      std::make_unique<cortext::operations::BuildGraphFromConsolidation> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (MakeVec ({ { 0, 1.0f } }), 1000));
  processor.Flush ();

  auto rows = store->Execute (
      "SELECT COUNT(*) AS c FROM associations WHERE edge_type = 'next_in_episode'",
      {});
  const long long count = rows.empty () ? 0LL : AnyToInt64 (rows[0].at ("c"));
  return count > 0 ? 1.0 : 0.0;
}

struct ComboResult
{
  std::uint32_t mask = 0;
  std::array<double, kFamilyCount> scores{};
  double total = 0.0;
};

ComboResult
RunSuite (std::uint32_t mask)
{
  ComboResult result;
  result.mask = mask;
  result.scores[kSourceConfidence] = RunSourceConfidenceScenario (mask);
  result.scores[kPredictiveRetrieval] = RunPredictiveScenario (mask);
  result.scores[kConstructiveRecall] = RunConstructiveRecallScenario (mask);
  result.scores[kProceduralMemory] = RunProceduralScenario (mask);
  result.scores[kMetacognitiveControl] = RunMetacognitiveScenario (mask);
  result.scores[kAffectInterrupt] = RunAffectInterruptScenario (mask);
  result.scores[kAffectRetrieval] = RunAffectRetrievalScenario (mask);
  result.scores[kFlashbulbConsolidation] = RunFlashbulbScenario (mask);
  result.scores[kNeuromodulation] = RunNeuromodulationScenario (mask);
  result.scores[kMetaLearning] = RunMetaLearningScenario (mask);
  result.scores[kReinforcementEdges] = RunReinforcementScenario (mask);
  result.scores[kSequentialEdges] = RunSequentialScenario (mask);
  result.total = std::accumulate (result.scores.begin (), result.scores.end (),
                                  0.0);
  return result;
}

int
BitCount (std::uint32_t mask)
{
  int count = 0;
  while (mask != 0u)
    {
      count += static_cast<int> (mask & 1u);
      mask >>= 1u;
    }
  return count;
}

std::string
MaskSummary (std::uint32_t mask)
{
  std::ostringstream out;
  bool first = true;
  for (int i = 0; i < kFamilyCount; ++i)
    {
      if (!FamilyEnabled (mask, i))
        {
          continue;
        }
      if (!first)
        {
          out << ",";
        }
      first = false;
      out << kFamilies[static_cast<std::size_t> (i)].slug;
    }
  return first ? "none" : out.str ();
}

std::string
DisabledSummary (std::uint32_t mask)
{
  std::ostringstream out;
  bool first = true;
  for (int i = 0; i < kFamilyCount; ++i)
    {
      if (FamilyEnabled (mask, i))
        {
          continue;
        }
      if (!first)
        {
          out << ",";
        }
      first = false;
      out << kFamilies[static_cast<std::size_t> (i)].slug;
    }
  return first ? "none" : out.str ();
}

} // namespace

int
main ()
{
  const std::uint32_t full_mask = (1u << kFamilyCount) - 1u;
  const std::uint32_t combo_count = (1u << kFamilyCount);
  std::vector<ComboResult> results;
  results.reserve (combo_count);

  for (std::uint32_t mask = 0; mask < combo_count; ++mask)
    {
      results.push_back (RunSuite (mask));
    }

  const auto best_it = std::max_element (
      results.begin (), results.end (),
      [] (const ComboResult &a, const ComboResult &b) {
        return a.total < b.total;
      });
  const double best_total
      = best_it == results.end () ? 0.0 : best_it->total;
  double full_total = 0.0;
  std::array<double, kFamilyCount> full_scores{};
  for (const auto &result : results)
    {
      if (result.mask == full_mask)
        {
          full_total = result.total;
          full_scores = result.scores;
          break;
        }
    }

  std::vector<const ComboResult *> best_results;
  int minimal_best_bits = kFamilyCount + 1;
  for (const auto &result : results)
    {
      if (std::abs (result.total - best_total) > kScoreEpsilon)
        {
          continue;
        }
      best_results.push_back (&result);
      minimal_best_bits = std::min (minimal_best_bits, BitCount (result.mask));
    }

  std::vector<const ComboResult *> minimal_best_masks;
  for (const auto *result : best_results)
    {
      if (BitCount (result->mask) == minimal_best_bits)
        {
          minimal_best_masks.push_back (result);
        }
    }

  std::cout << std::fixed << std::setprecision (6);
  std::cout << "operation_family_count=" << kFamilyCount << "\n";
  std::cout << "operation_family_combo_count=" << combo_count << "\n";
  std::cout << "operation_family_full_mask_score=" << full_total << "\n";
  std::cout << "operation_family_best_score=" << best_total << "\n";
  std::cout << "operation_family_full_mask_is_best="
            << ((std::abs (full_total - best_total) <= kScoreEpsilon) ? 1 : 0)
            << "\n";
  std::cout << "operation_family_best_mask_count=" << best_results.size ()
            << "\n";
  std::cout << "operation_family_minimal_best_bits=" << minimal_best_bits
            << "\n";
  if (!minimal_best_masks.empty ())
    {
      std::cout << "operation_family_minimal_best_enabled="
                << MaskSummary (minimal_best_masks.front ()->mask) << "\n";
      std::cout << "operation_family_minimal_best_disabled="
                << DisabledSummary (minimal_best_masks.front ()->mask) << "\n";
    }
  for (int bit = 0; bit < kFamilyCount; ++bit)
    {
      std::cout << "operation_family_full_mask_component_"
                << kFamilies[static_cast<std::size_t> (bit)].slug << "="
                << full_scores[static_cast<std::size_t> (bit)] << "\n";
    }

  std::cout << "\n| family | mean_marginal | positive_rate | negative_rate | max_score_without | essential_for_best | verdict |\n";
  std::cout << "|---|---:|---:|---:|---:|---:|---|\n";

  for (int bit = 0; bit < kFamilyCount; ++bit)
    {
      double marginal_sum = 0.0;
      int pair_count = 0;
      int positive_count = 0;
      int negative_count = 0;
      double max_without = -std::numeric_limits<double>::infinity ();
      for (const auto &result : results)
        {
          if (FamilyEnabled (result.mask, bit))
            {
              continue;
            }
          max_without = std::max (max_without, result.total);
          const std::uint32_t neighbor_mask = result.mask | (1u << bit);
          const double neighbor_total = results[neighbor_mask].total;
          const double delta = neighbor_total - result.total;
          marginal_sum += delta;
          ++pair_count;
          if (delta > kScoreEpsilon)
            {
              ++positive_count;
            }
          else if (delta < -kScoreEpsilon)
            {
              ++negative_count;
            }
        }
      const double mean_marginal
          = pair_count > 0 ? marginal_sum / static_cast<double> (pair_count) : 0.0;
      const double positive_rate
          = pair_count > 0
                ? static_cast<double> (positive_count)
                      / static_cast<double> (pair_count)
                : 0.0;
      const double negative_rate
          = pair_count > 0
                ? static_cast<double> (negative_count)
                      / static_cast<double> (pair_count)
                : 0.0;
      const bool essential = max_without + kScoreEpsilon < best_total;
      std::string verdict;
      if (essential && mean_marginal >= 0.75)
        {
          verdict = "core";
        }
      else if (essential)
        {
          verdict = "narrow_but_required";
        }
      else if (mean_marginal > 0.10)
        {
          verdict = "contributory_not_essential";
        }
      else
        {
          verdict = "cut_candidate";
        }
      std::cout << "| `" << kFamilies[static_cast<std::size_t> (bit)].name
                << "` | " << mean_marginal << " | " << positive_rate << " | "
                << negative_rate << " | " << max_without << " | "
                << (essential ? 1 : 0) << " | " << verdict << " |\n";
    }

  std::vector<const ComboResult *> sorted;
  sorted.reserve (results.size ());
  for (const auto &result : results)
    {
      sorted.push_back (&result);
    }
  std::sort (sorted.begin (), sorted.end (),
             [] (const ComboResult *a, const ComboResult *b) {
               if (std::abs (a->total - b->total) > kScoreEpsilon)
                 {
                   return a->total > b->total;
                 }
               return BitCount (a->mask) < BitCount (b->mask);
             });

  std::cout << "\n| rank | score | bits_on | enabled_families | disabled_families |\n";
  std::cout << "|---|---:|---:|---|---|\n";
  for (std::size_t i = 0; i < std::min<std::size_t> (10, sorted.size ()); ++i)
    {
      const auto *result = sorted[i];
      std::cout << "| " << (i + 1) << " | " << result->total << " | "
                << BitCount (result->mask) << " | " << MaskSummary (result->mask)
                << " | " << DisabledSummary (result->mask) << " |\n";
    }

  return 0;
}
