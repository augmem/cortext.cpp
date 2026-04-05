#include "../../src/operations/constructive_recall_internal.hpp"
#include "../../src/operations/meta_learning_internal.hpp"
#include "../../src/operations/retrieval_debug_state.hpp"

#include <cortext/core/algorithms.hpp>
#include <cortext/core/sparse.hpp>
#include <cortext/core/utils.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/competition.hpp>
#include <cortext/operations/detect_memory_usage.hpp>
#include <cortext/operations/emotion.hpp>
#include <cortext/operations/focus.hpp>
#include <cortext/operations/graph_build.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/operations/interrupt_gate.hpp>
#include <cortext/operations/metacognitive.hpp>
#include <cortext/operations/reconsolidation.hpp>
#include <cortext/operations/sensitivity.hpp>
#include <cortext/operations/stability.hpp>
#include <cortext/operations/write_gate.hpp>
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

enum UnitBit : int
{
  kSourceConfidence = 0,
  kPredictiveRetrieval,
  kConstructiveRecall,
  kProceduralProactive,
  kMetacogTot,
  kMetacogUnknown,
  kMetacogDecay,
  kAffectInterrupt,
  kAffectRetrieval,
  kFlashbulbPercentile,
  kFlashbulbRate,
  kFlashbulbArousal,
  kNeuromodWrite,
  kNeuromodCompetition,
  kNeuromodReconsolidation,
  kNeuromodValue,
  kMetaLearning,
  kReinforcementEdges,
  kSequentialEdges,
  kUnitCount
};

struct UnitInfo
{
  const char *slug;
  const char *cluster;
};

constexpr std::array<UnitInfo, kUnitCount> kUnits = {
  UnitInfo{ "source_confidence", "singleton" },
  UnitInfo{ "predictive_retrieval", "singleton" },
  UnitInfo{ "constructive_recall", "singleton" },
  UnitInfo{ "procedural_proactive", "singleton" },
  UnitInfo{ "metacog_tot_recovery", "metacognitive" },
  UnitInfo{ "metacog_unknown_caution", "metacognitive" },
  UnitInfo{ "metacog_confidence_decay", "metacognitive" },
  UnitInfo{ "affect_interrupt", "affect" },
  UnitInfo{ "affect_retrieval", "affect" },
  UnitInfo{ "flashbulb_percentile", "flashbulb" },
  UnitInfo{ "flashbulb_rate", "flashbulb" },
  UnitInfo{ "flashbulb_arousal", "flashbulb" },
  UnitInfo{ "neuromod_write_scale", "neuromod" },
  UnitInfo{ "neuromod_competition_scale", "neuromod" },
  UnitInfo{ "neuromod_reconsolidation_scale", "neuromod" },
  UnitInfo{ "neuromod_value_gain", "neuromod" },
  UnitInfo{ "meta_learning", "singleton" },
  UnitInfo{ "reinforcement_edges", "singleton" },
  UnitInfo{ "sequential_edges", "singleton" },
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
SeedMemory (cortext::Store &store, long long memory_id, long long embedding_id,
            const Eigen::VectorXf &embedding,
            const std::string &kind = "LONG_TERM", long long created_at = 1,
            double strength = 1.0)
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

cortext::SignalProcessor::Config
BaseConfig (double focus, double sensitivity, double stability)
{
  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = focus;
  cfg.sensitivity = sensitivity;
  cfg.stability = stability;
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
    const int key_size = cortext::core::SparseKeySize (ctx.GetConfig ().focus);
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

class SetupStoredEmotionOp : public cortext::IOperation
{
public:
  SetupStoredEmotionOp (long long stored_id, std::vector<double> history,
                        double rate_ewma)
      : stored_id_ (stored_id), history_ (std::move (history)),
        rate_ewma_ (rate_ewma)
  {
  }

  void
  Execute (cortext::OperationContext &ctx,
           cortext::Transaction & /*tx*/) const override
  {
    ctx.SetStoredEmbeddingId (stored_id_);
    auto &pctx = ctx.GetProcessorContext ();
    pctx.recent_emotion_intensities.assign (history_.begin (), history_.end ());
    pctx.flashbulb_rate_ewma = rate_ewma_;
  }

private:
  long long stored_id_ = 0;
  std::vector<double> history_;
  double rate_ewma_ = 0.0;
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

class SetupCompetitionInputsOp : public cortext::IOperation
{
public:
  SetupCompetitionInputsOp (Eigen::VectorXf cur,
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

class SeedCompetitionMemoriesOp : public cortext::IOperation
{
public:
  explicit SeedCompetitionMemoriesOp (
      std::unordered_map<long long, Eigen::VectorXf> embeddings)
      : embeddings_ (std::move (embeddings))
  {
  }

  void
  Execute (cortext::OperationContext &ctx,
           cortext::Transaction & /*tx*/) const override
  {
    auto *store = ctx.GetStore ();
    const long long now_ts = 1;
    for (const auto &[id, emb] : embeddings_)
      {
        store->Execute (
            "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
            "VALUES(?, ?, ?)",
            { id, ToFloatVec (emb), now_ts });
        store->Execute (
            "INSERT OR REPLACE INTO memories("
            "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
            "s_max, s_avg, strength, created_at) "
            "VALUES(?, ?, 'bench', 'LONG_TERM', 1, 1, 'text', 0.5, 0.5, 1.0, 1)",
            { id, id });
      }
  }

private:
  std::unordered_map<long long, Eigen::VectorXf> embeddings_;
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

struct SingletonBundle
{
  bool source_confidence = true;
  bool predictive_retrieval = true;
  bool constructive_recall = true;
  bool procedural_proactive = true;
  bool meta_learning = true;
  bool reinforcement_edges = true;
  bool sequential_edges = true;
};

struct MetacogBundle
{
  bool tot = true;
  bool unknown = true;
  bool decay = true;
};

struct AffectBundle
{
  bool interrupt = true;
  bool retrieval = true;
};

struct FlashbulbBundle
{
  bool percentile = true;
  bool rate = true;
  bool arousal = true;
};

struct NeuromodBundle
{
  bool write = true;
  bool competition = true;
  bool recon = true;
  bool value = true;
};

class ScopedSingletonEnv
{
public:
  explicit ScopedSingletonEnv (const SingletonBundle &bundle)
  {
    Configure ("CORTEXT_DISABLE_SOURCE_CONF", bundle.source_confidence);
    Configure ("CORTEXT_DISABLE_PREDICTIVE_RETRIEVAL_BONUS",
               bundle.predictive_retrieval);
    Configure ("CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", bundle.constructive_recall);
    Configure ("CORTEXT_DISABLE_PROCEDURAL_PROACTIVE_RETRIEVAL",
               bundle.procedural_proactive);
    Configure ("CORTEXT_DISABLE_META_LEARNING", bundle.meta_learning);
  }

private:
  void
  Configure (const char *name, bool enabled)
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

class ScopedMetacogEnv
{
public:
  explicit ScopedMetacogEnv (const MetacogBundle &bundle)
  {
    Configure ("CORTEXT_DISABLE_METACOG_TOT_RECOVERY", bundle.tot);
    Configure ("CORTEXT_DISABLE_METACOG_UNKNOWN_CAUTION", bundle.unknown);
    Configure ("CORTEXT_DISABLE_METACOG_CONFIDENCE_DECAY", bundle.decay);
  }

private:
  void
  Configure (const char *name, bool enabled)
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

class ScopedFlashbulbEnv
{
public:
  explicit ScopedFlashbulbEnv (const FlashbulbBundle &bundle)
  {
    Configure ("CORTEXT_FLASHBULB_DISABLE_PERCENTILE", bundle.percentile);
    Configure ("CORTEXT_FLASHBULB_DISABLE_RATE", bundle.rate);
    Configure ("CORTEXT_FLASHBULB_DISABLE_AROUSAL", bundle.arousal);
  }

private:
  void
  Configure (const char *name, bool enabled)
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

class ScopedNeuromodEnv
{
public:
  explicit ScopedNeuromodEnv (const NeuromodBundle &bundle)
  {
    Configure ("CORTEXT_DISABLE_NEUROMOD_WRITE_SCALE", bundle.write);
    Configure ("CORTEXT_DISABLE_NEUROMOD_COMPETITION_SCALE", bundle.competition);
    Configure ("CORTEXT_DISABLE_NEUROMOD_RECONSOLIDATION_SCALE", bundle.recon);
    Configure ("CORTEXT_DISABLE_NEUROMOD_VALUE_GAIN", bundle.value);
  }

private:
  void
  Configure (const char *name, bool enabled)
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

// Singleton scenarios ---------------------------------------------------------

double
RunSourceConfidenceScenario (const SingletonBundle &bundle)
{
  ScopedSingletonEnv guards (bundle);
  auto store = CreateStore ();

  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf low_conf = query;
  const Eigen::VectorXf high_conf = MakeVec ({ { 0, 0.94f }, { 1, 0.341f } });
  const long long ts = 7200000;

  SeedMemory (*store, 10LL, 10LL, low_conf);
  SeedMemory (*store, 20LL, 20LL, high_conf);
  store->Execute (
      "UPDATE memories SET source_contradiction_count = ? WHERE memory_id = ?",
      { 7LL, 10LL });
  store->Execute (
      "UPDATE memories SET source_contradiction_count = ? WHERE memory_id = ?",
      { 0LL, 20LL });

  auto cfg = BaseConfig (0.5, 0.5, 1.0);
  cfg.procedural_enabled = bundle.procedural_proactive;
  cfg.reinforcement_enabled = bundle.reinforcement_edges;
  cfg.sequential_edges_enabled = bundle.sequential_edges;
  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (query, ts));
  processor.Flush ();
  const auto ranked = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  return (!ranked.empty () && ranked.front ().memory_id == 20LL) ? 1.0 : 0.0;
}

double
RunPredictiveScenario (const SingletonBundle &bundle)
{
  ScopedSingletonEnv guards (bundle);
  auto store = CreateStore ();

  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf predictive_target
      = MakeVec ({ { 0, 0.8f }, { 1, 0.6f } });
  const Eigen::VectorXf raw_best
      = MakeVec ({ { 0, 0.95f }, { 1, 0.3122499f } });

  SeedMemory (*store, 11LL, 11LL, predictive_target);
  SeedMemory (*store, 22LL, 22LL, raw_best);
  store->Execute ("UPDATE memories SET pre_activation = 1.0 WHERE memory_id = ?",
                  { 11LL });

  auto cfg = BaseConfig (1.0, 0.5, 0.5);
  cfg.procedural_enabled = bundle.procedural_proactive;
  cfg.reinforcement_enabled = bundle.reinforcement_edges;
  cfg.sequential_edges_enabled = bundle.sequential_edges;
  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (query, 10));
  processor.Flush ();
  const auto ranked = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  return (!ranked.empty () && ranked.front ().memory_id == 11LL) ? 1.0 : 0.0;
}

double
RunConstructiveRecallScenario (const SingletonBundle &bundle)
{
  ScopedSingletonEnv guards (bundle);
  auto store = CreateStore ();

  const Eigen::VectorXf query = MakeVec ({ { 0, 0.98f }, { 1, 0.19f } });
  const Eigen::VectorXf evidence = MakeVec ({ { 0, 0.84f }, { 1, 0.54f } });
  const Eigen::VectorXf competitor = MakeVec ({ { 0, 0.96f }, { 1, 0.28f } });
  const Eigen::VectorXf reconstructed = query;

  SeedMemory (*store, 11LL, 11LL, evidence);
  SeedMemory (*store, 22LL, 22LL, competitor);
  auto tx = store->Begin ();
  cortext::operations::constructive_recall::AppendReconstructionWithEmbedding (
      *tx, 11LL, reconstructed, {}, 2, 0.2, "bench");
  tx->Commit ();

  auto cfg = BaseConfig (0.5, 0.5, 0.5);
  cfg.procedural_enabled = bundle.procedural_proactive;
  cfg.reinforcement_enabled = bundle.reinforcement_edges;
  cfg.sequential_edges_enabled = bundle.sequential_edges;
  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (query, 10));
  processor.Flush ();
  const auto ranked = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  return (!ranked.empty () && ranked.front ().memory_id == 11LL) ? 1.0 : 0.0;
}

double
RunProceduralScenario (const SingletonBundle &bundle)
{
  ScopedSingletonEnv guards (bundle);
  auto store = CreateStore ();

  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf routine_target = MakeVec ({ { 0, 0.30f }, { 1, 0.9539392f } });
  SeedMemory (*store, 500LL, 500LL, routine_target);
  for (int i = 0; i < 9; ++i)
    {
      const float x = 0.44f - 0.01f * static_cast<float> (i);
      const float y = std::sqrt (std::max (0.0f, 1.0f - x * x));
      const long long id = 600LL + i;
      SeedMemory (*store, id, id, MakeVec ({ { 0, x }, { 1, y } }));
    }

  auto cfg = BaseConfig (1.0, 1.0, 0.5);
  cfg.procedural_enabled = true;
  cfg.reinforcement_enabled = bundle.reinforcement_edges;
  cfg.sequential_edges_enabled = bundle.sequential_edges;
  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  auto ops = std::make_unique<cortext::OperationSet> (
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

double
RunMetaLearningScenario (const SingletonBundle &bundle)
{
  ScopedSingletonEnv guards (bundle);
  auto store = CreateStore ();
  cortext::ProcessorContext pctx;
  auto cfg = BaseConfig (0.5, 0.5, 0.5);

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
      pctx.attention_width = 2.0;
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
RunReinforcementScenario (const SingletonBundle &bundle)
{
  ScopedSingletonEnv guards (bundle);
  auto store = CreateStore ();
  const Eigen::VectorXf emb_a = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf emb_b = MakeVec ({ { 1, 1.0f } });
  SeedMemory (*store, 1LL, 1LL, emb_a);
  SeedMemory (*store, 2LL, 2LL, emb_b);

  auto cfg = BaseConfig (0.5, 0.5, 0.5);
  cfg.procedural_enabled = bundle.procedural_proactive;
  cfg.reinforcement_enabled = bundle.reinforcement_edges;
  cfg.sequential_edges_enabled = bundle.sequential_edges;
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
RunSequentialScenario (const SingletonBundle &bundle)
{
  ScopedSingletonEnv guards (bundle);
  auto store = CreateStore ();
  SeedMemory (*store, 1LL, 1LL, MakeVec ({ { 0, 1.0f } }), "LONG_TERM", 1);
  SeedMemory (*store, 2LL, 2LL, MakeVec ({ { 0, 0.95f }, { 1, 0.31f } }),
              "LONG_TERM", 2);
  store->Execute (
      "UPDATE memories SET cluster_id = 1, boundary_score = 0.0 WHERE memory_id IN (1, 2)");

  auto cfg = BaseConfig (0.5, 0.5, 0.5);
  cfg.procedural_enabled = bundle.procedural_proactive;
  cfg.reinforcement_enabled = bundle.reinforcement_edges;
  cfg.sequential_edges_enabled = bundle.sequential_edges;
  auto ops = std::make_unique<cortext::OperationSet> (
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

std::array<double, 7>
RunSingletonBundle (const SingletonBundle &bundle)
{
  return {
    RunSourceConfidenceScenario (bundle),
    RunPredictiveScenario (bundle),
    RunConstructiveRecallScenario (bundle),
    RunProceduralScenario (bundle),
    RunMetaLearningScenario (bundle),
    RunReinforcementScenario (bundle),
    RunSequentialScenario (bundle),
  };
}

// Metacognitive cluster -------------------------------------------------------

std::vector<long long>
RunTotRetrieval (const MetacogBundle &bundle, double confidence)
{
  ScopedMetacogEnv guards (bundle);
  auto store = CreateStore ();

  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  SeedMemory (*store, 1LL, 1LL, query);
  for (long long id = 2; id <= 4; ++id)
    {
      SeedMemory (*store, id, id, MakeVec ({ { 1, 1.0f } }), "ASSOCIATION");
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

  auto cfg = BaseConfig (0.5, 0.5, 1.0);
  auto ops = std::make_unique<cortext::OperationSet> (
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
RunMetacogTotScenario (const MetacogBundle &bundle)
{
  const auto ids = RunTotRetrieval (bundle, 1.0);
  return std::find (ids.begin (), ids.end (), 4LL) != ids.end () ? 1.0 : 0.0;
}

double
RunMetacogUnknownScenario (const MetacogBundle &bundle)
{
  ScopedMetacogEnv guards (bundle);
  auto store = CreateStore ();
  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  SeedMemory (*store, 77LL, 77LL, query);
  store->Execute ("UPDATE memories "
                  "SET source_origin = 'external', source_reliability = 0.1, "
                  "source_contradiction_count = 2 "
                  "WHERE memory_id = ?",
                  { 77LL });

  auto cfg = BaseConfig (0.5, 0.5, 0.5);
  auto ops = std::make_unique<cortext::OperationSet> (
      std::make_unique<ForceRetrievalGateOp> (
          cortext::ProcessorContext::MetacognitiveMode::UnknownCaution, 0.0),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto out = processor.Process (MakeSignal (query, 20));
  processor.Flush ();
  return out.candidate_memory_ids.empty () ? 1.0 : 0.0;
}

double
RunMetacogDecayScenario (const MetacogBundle &bundle)
{
  ScopedMetacogEnv guards (bundle);
  cortext::Signal signal = MakeSignal (MakeVec ({ { 0, 1.0f } }), 11000);
  cortext::ProcessorContext pctx;
  pctx.metacognitive_confidence = 1.0;
  pctx.last_signal_timestamp = 1000;

  auto cfg = BaseConfig (0.5, 0.5, 0.0);
  cortext::operations::MetacognitiveMonitoring op;
  cortext::OperationContext ctx (signal, pctx, cfg);
  ctx.SetFeelingOfKnowing (0.2);
  ctx.SetMemoryUsageEvents ({ { 1LL, true, -1.0 } });
  NullTransaction tx;
  op.Execute (ctx, tx);

  const auto ids = RunTotRetrieval (bundle, pctx.metacognitive_confidence);
  const bool hit = std::find (ids.begin (), ids.end (), 4LL) != ids.end ();
  return hit ? 0.0 : 1.0;
}

std::array<double, 3>
RunMetacogBundle (const MetacogBundle &bundle)
{
  return {
    RunMetacogTotScenario (bundle),
    RunMetacogUnknownScenario (bundle),
    RunMetacogDecayScenario (bundle),
  };
}

// Affect cluster --------------------------------------------------------------

double
RunAffectInterruptScenario (const AffectBundle &bundle)
{
  auto store = CreateStore ();
  const Eigen::VectorXf signal_emb = MakeVec ({ { 0, 1.0f }, { 1, 0.02f } });
  const Eigen::VectorXf cand_emb = MakeVec ({ { 0, 0.93f }, { 1, 0.367f } });
  SeedMemory (*store, 1LL, 1LL, cand_emb);

  auto cfg = BaseConfig (0.7, 0.9, 0.2);
  cfg.affect_interrupt = bundle.interrupt;
  cfg.affect_retrieval = bundle.retrieval;

  cortext::ProcessorContext pc;
  pc.signals_processed = 100;
  pc.last_interrupt_tick = 0;
  pc.recent_memory_centroids.push_back (
      MakeVec ({ { 0, 1.0f }, { 1, 0.0f } }));

  cortext::Signal signal = MakeSignal (signal_emb, 1000, "affect_interrupt");
  cortext::OperationContext oc (signal, pc, cfg, store.get ());
  oc.SetCoherence (1.0);
  oc.SetThresholdTDynamic (0.55);
  oc.SetAtBoundary (false);
  oc.SetEmotionIntensity (1.0);
  oc.SetArousal (1.0);
  oc.SetMetric (cortext::operations::Metric::salience, 1.0);
  oc.SetRetrievedMemoryEmbeddings ({ { 1LL, cand_emb } });

  cortext::operations::ComputeMniGateDecision op;
  auto tx = store->Begin ();
  op.Execute (oc, *tx);
  return oc.GetInterruptAllowed () ? 1.0 : 0.0;
}

double
RunAffectRetrievalScenario (const AffectBundle &bundle)
{
  auto store = CreateStore ();
  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf affect_target = MakeVec ({ { 0, 0.90f }, { 1, 0.43589f } });
  const Eigen::VectorXf raw_best = MakeVec ({ { 0, 0.96f }, { 1, 0.28f } });

  SeedMemory (*store, 11LL, 11LL, affect_target);
  SeedMemory (*store, 22LL, 22LL, raw_best);
  store->Execute (
      "UPDATE memories SET emotional_intensity = 1.0, s_arousal_avg = 1.0 WHERE memory_id = ?",
      { 11LL });
  store->Execute (
      "UPDATE memories SET emotional_intensity = 0.0, s_arousal_avg = 0.0 WHERE memory_id = ?",
      { 22LL });

  auto cfg = BaseConfig (0.5, 1.0, 0.5);
  cfg.affect_interrupt = bundle.interrupt;
  cfg.affect_retrieval = bundle.retrieval;
  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  auto ops = std::make_unique<cortext::OperationSet> (
      std::make_unique<SetAffectInputsOp> (1.0, 1.0, 1.0),
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (query, 10));
  processor.Flush ();
  const auto ranked = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  return (!ranked.empty () && ranked.front ().memory_id == 11LL) ? 1.0 : 0.0;
}

std::array<double, 2>
RunAffectBundle (const AffectBundle &bundle)
{
  return {
    RunAffectInterruptScenario (bundle),
    RunAffectRetrievalScenario (bundle),
  };
}

// Flashbulb cluster -----------------------------------------------------------

bool
RunFlashbulbCase (const FlashbulbBundle &bundle, double emotion, double arousal,
                  const std::vector<double> &history, double rate_ewma)
{
  ScopedFlashbulbEnv guards (bundle);
  auto store = CreateStore ();
  SeedMemory (*store, 101LL, 101LL, MakeVec ({ { 0, 1.0f } }));
  store->Execute (
      "UPDATE memories SET s_emotion_max = ?, s_arousal_avg = ? WHERE memory_id = ?",
      { emotion, arousal, 101LL });

  auto cfg = BaseConfig (0.4, 0.8, 0.5);
  auto ops = std::make_unique<cortext::OperationSet> (
      std::make_unique<SetupStoredEmotionOp> (101LL, history, rate_ewma),
      std::make_unique<cortext::operations::ApplyEmotionalConsolidation> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (MakeVec ({ { 0, 1.0f } }), 12345));
  processor.Flush ();

  auto rows = store->Execute ("SELECT flashbulb FROM memories WHERE memory_id = ?",
                              { 101LL });
  return !rows.empty () && AnyToInt64 (rows[0].at ("flashbulb")) == 1LL;
}

double
RunFlashbulbPercentileScenario (const FlashbulbBundle &bundle)
{
  const std::vector<double> history (16, 0.98);
  return RunFlashbulbCase (bundle, 0.90, 0.90, history, 0.0) ? 0.0 : 1.0;
}

double
RunFlashbulbRateScenario (const FlashbulbBundle &bundle)
{
  return RunFlashbulbCase (bundle, 0.76, 0.88, {}, 1.0) ? 0.0 : 1.0;
}

double
RunFlashbulbArousalScenario (const FlashbulbBundle &bundle)
{
  return RunFlashbulbCase (bundle, 0.96, 0.45, {}, 0.0) ? 0.0 : 1.0;
}

std::array<double, 3>
RunFlashbulbBundle (const FlashbulbBundle &bundle)
{
  return {
    RunFlashbulbPercentileScenario (bundle),
    RunFlashbulbRateScenario (bundle),
    RunFlashbulbArousalScenario (bundle),
  };
}

// Neuromod cluster ------------------------------------------------------------

double
RunNeuromodWriteScenario (const NeuromodBundle &bundle)
{
  ScopedNeuromodEnv guards (bundle);
  cortext::Signal s = MakeSignal (MakeVec ({ { 0, 1.0f } }), 100000);
  cortext::ProcessorContext pctx;
  auto cfg = BaseConfig (0.5, 0.5, 0.5);

  cortext::AccumulatorState acc;
  acc.mu_acc = s.embedding;
  acc.e_peak = s.embedding;
  acc.s_max = 0.5;
  acc.s_sum = 1.0;
  acc.n_signals = 2;
  acc.t_start = s.timestamp - 5000;
  pctx.accumulator_states[s.source_id] = std::move (acc);
  pctx.neuromod_ne = 1.0;

  cortext::OperationContext ctx (s, pctx, cfg);
  ctx.SetFlushRequired (true);
  ctx.SetThresholdTDynamic (0.6);
  cortext::operations::ComputeWriteGate op;
  NullTransaction tx;
  op.Execute (ctx, tx);
  return ctx.GetAccumulatorWriteDecision () ? 1.0 : 0.0;
}

double
RunNeuromodCompetitionScenario (const NeuromodBundle &bundle)
{
  ScopedNeuromodEnv guards (bundle);
  auto store = CreateStore ();

  const Eigen::VectorXf ctx = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf w1 = MakeVec ({ { 0, 0.99f }, { 1, 0.05f } });
  const Eigen::VectorXf w2 = MakeVec ({ { 0, 0.98f }, { 1, 0.06f } });
  const Eigen::VectorXf w3 = MakeVec ({ { 0, 0.95f }, { 1, 0.10f } });
  const Eigen::VectorXf loser = MakeVec ({ { 0, 0.88f }, { 1, 0.47f } });
  std::unordered_map<long long, Eigen::VectorXf> retrieved{
    { 10LL, w1 }, { 11LL, w2 }, { 12LL, w3 }, { 13LL, loser }
  };

  auto cfg = BaseConfig (1.0, 1.0, 0.0);
  auto pipeline = std::make_unique<cortext::OperationSet> (
      std::make_unique<SeedCompetitionMemoriesOp> (retrieved),
      std::make_unique<SetupCompetitionInputsOp> (ctx, retrieved),
      std::make_unique<SetNeuromodOp> (0.0, 1.0),
      std::make_unique<cortext::operations::ApplyRetrievalCompetition> ());
  cortext::SignalProcessor processor (cfg, store, std::move (pipeline));
  processor.Process (MakeSignal (ctx, 100));
  processor.Flush ();

  auto rows = store->Execute (
      "SELECT strength FROM memories WHERE memory_id = ?",
      { 13LL });
  const double loser_strength
      = rows.empty () ? 1.0 : AnyToDouble (rows[0].at ("strength"));
  return loser_strength < 0.999 ? 1.0 : 0.0;
}

double
RunNeuromodReconScenario (const NeuromodBundle &bundle)
{
  ScopedNeuromodEnv guards (bundle);
  auto store = CreateStore ();

  const Eigen::VectorXf current = MakeVec ({ { 0, 1.0f }, { 1, 0.9f } });
  const Eigen::VectorXf mem = MakeVec ({ { 0, 1.0f } });
  SeedMemory (*store, 9LL, 9LL, mem);

  auto cfg = BaseConfig (0.5, 1.0, 0.0);
  auto pipeline = std::make_unique<cortext::OperationSet> (
      std::make_unique<SetupReconInputsOp> (
          current, std::unordered_map<long long, Eigen::VectorXf>{ { 9LL, mem } }),
      std::make_unique<SetNeuromodOp> (1.0, 0.0),
      std::make_unique<cortext::operations::ApplyReconsolidation> ());
  cortext::SignalProcessor processor (cfg, store, std::move (pipeline));
  ScopedEnvVar disable_constructive ("CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  processor.Process (MakeSignal (current, 100));
  processor.Flush ();

  auto rows = store->Execute (
      "SELECT embedding FROM embeddings WHERE embedding_id = ?",
      { 9LL });
  Eigen::VectorXf updated;
  if (rows.empty ()
      || !cortext::core::DecodeFloatBlob (rows[0].at ("embedding"),
                                          kEmbeddingDim, updated))
    {
      return 0.0;
    }
  const double similarity = cortext::core::CosineSimilarity (updated, current);
  return similarity > 0.82 ? 1.0 : 0.0;
}

double
RunValueStudy (const NeuromodBundle &bundle, double neuromod_da)
{
  ScopedNeuromodEnv guards (bundle);
  auto store = CreateStore ();
  const Eigen::VectorXf emb = MakeVec ({ { 0, 1.0f } });
  SeedMemory (*store, 7LL, 7LL, emb);

  cortext::ProcessorContext pctx;
  pctx.neuromod_da = neuromod_da;
  pctx.delta_reward = 0.8;

  auto cfg = BaseConfig (1.0, 0.5, 0.5);
  cfg.procedural_enabled = true;

  cortext::Signal signal = MakeSignal (emb, 10);
  cortext::OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetInterruptAllowed (true);
  ctx.SetSelectedCandidateId (7LL);
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{ { 7LL, emb } });

  cortext::operations::DetectMemoryUsage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  const int key_size = cortext::core::SparseKeySize (cfg.focus);
  const std::string key = cortext::core::SparseKey (signal.embedding, key_size);
  return pctx.procedural_store[key][7LL];
}

double
RunNeuromodValueScenario (const NeuromodBundle &bundle)
{
  return RunValueStudy (bundle, 1.0) > RunValueStudy (bundle, 0.0) ? 1.0 : 0.0;
}

std::array<double, 4>
RunNeuromodBundle (const NeuromodBundle &bundle)
{
  return {
    RunNeuromodWriteScenario (bundle),
    RunNeuromodCompetitionScenario (bundle),
    RunNeuromodReconScenario (bundle),
    RunNeuromodValueScenario (bundle),
  };
}

template <std::size_t N>
struct ClusterResult
{
  std::array<double, N> scores{};
  double total = 0.0;
};

template <std::size_t N>
ClusterResult<N>
FinalizeScores (const std::array<double, N> &scores)
{
  ClusterResult<N> result;
  result.scores = scores;
  result.total = std::accumulate (scores.begin (), scores.end (), 0.0);
  return result;
}

template <std::size_t N>
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

template <std::size_t N>
std::string
MaskSummary (std::uint32_t mask, const std::array<const char *, N> &names)
{
  std::ostringstream out;
  bool first = true;
  for (std::size_t i = 0; i < N; ++i)
    {
      if ((mask & (1u << i)) == 0u)
        {
          continue;
        }
      if (!first)
        {
          out << ",";
        }
      first = false;
      out << names[i];
    }
  return first ? "none" : out.str ();
}

std::string
GlobalMaskSummary (std::uint32_t mask)
{
  std::ostringstream out;
  bool first = true;
  for (int i = 0; i < kUnitCount; ++i)
    {
      if ((mask & (1u << i)) == 0u)
        {
          continue;
        }
      if (!first)
        {
          out << ",";
        }
      first = false;
      out << kUnits[static_cast<std::size_t> (i)].slug;
    }
  return first ? "none" : out.str ();
}

std::string
GlobalDisabledSummary (std::uint32_t mask)
{
  std::ostringstream out;
  bool first = true;
  for (int i = 0; i < kUnitCount; ++i)
    {
      if ((mask & (1u << i)) != 0u)
        {
          continue;
        }
      if (!first)
        {
          out << ",";
        }
      first = false;
      out << kUnits[static_cast<std::size_t> (i)].slug;
    }
  return first ? "none" : out.str ();
}

struct GlobalResult
{
  std::uint32_t mask = 0;
  std::array<double, kUnitCount> scores{};
  double total = 0.0;
};

} // namespace

int
main ()
{
  constexpr std::array<const char *, 7> kSingletonNames = {
    "source_confidence",
    "predictive_retrieval",
    "constructive_recall",
    "procedural_proactive",
    "meta_learning",
    "reinforcement_edges",
    "sequential_edges",
  };
  constexpr std::array<const char *, 3> kMetacogNames = {
    "metacog_tot_recovery",
    "metacog_unknown_caution",
    "metacog_confidence_decay",
  };
  constexpr std::array<const char *, 2> kAffectNames = {
    "affect_interrupt",
    "affect_retrieval",
  };
  constexpr std::array<const char *, 3> kFlashbulbNames = {
    "flashbulb_percentile",
    "flashbulb_rate",
    "flashbulb_arousal",
  };
  constexpr std::array<const char *, 4> kNeuromodNames = {
    "neuromod_write_scale",
    "neuromod_competition_scale",
    "neuromod_reconsolidation_scale",
    "neuromod_value_gain",
  };

  constexpr std::uint32_t kSingletonCombos = (1u << 7);
  constexpr std::uint32_t kMetacogCombos = (1u << 3);
  constexpr std::uint32_t kAffectCombos = (1u << 2);
  constexpr std::uint32_t kFlashbulbCombos = (1u << 3);
  constexpr std::uint32_t kNeuromodCombos = (1u << 4);
  constexpr std::uint32_t kGlobalCombos = (1u << kUnitCount);

  std::array<ClusterResult<7>, kSingletonCombos> singleton_results{};
  std::array<ClusterResult<3>, kMetacogCombos> metacog_results{};
  std::array<ClusterResult<2>, kAffectCombos> affect_results{};
  std::array<ClusterResult<3>, kFlashbulbCombos> flashbulb_results{};
  std::array<ClusterResult<4>, kNeuromodCombos> neuromod_results{};

  for (std::uint32_t mask = 0; mask < kSingletonCombos; ++mask)
    {
      SingletonBundle bundle;
      bundle.source_confidence = (mask & (1u << 0)) != 0u;
      bundle.predictive_retrieval = (mask & (1u << 1)) != 0u;
      bundle.constructive_recall = (mask & (1u << 2)) != 0u;
      bundle.procedural_proactive = (mask & (1u << 3)) != 0u;
      bundle.meta_learning = (mask & (1u << 4)) != 0u;
      bundle.reinforcement_edges = (mask & (1u << 5)) != 0u;
      bundle.sequential_edges = (mask & (1u << 6)) != 0u;
      singleton_results[mask] = FinalizeScores<7> (RunSingletonBundle (bundle));
    }

  for (std::uint32_t mask = 0; mask < kMetacogCombos; ++mask)
    {
      MetacogBundle bundle;
      bundle.tot = (mask & (1u << 0)) != 0u;
      bundle.unknown = (mask & (1u << 1)) != 0u;
      bundle.decay = (mask & (1u << 2)) != 0u;
      metacog_results[mask] = FinalizeScores<3> (RunMetacogBundle (bundle));
    }

  for (std::uint32_t mask = 0; mask < kAffectCombos; ++mask)
    {
      AffectBundle bundle;
      bundle.interrupt = (mask & (1u << 0)) != 0u;
      bundle.retrieval = (mask & (1u << 1)) != 0u;
      affect_results[mask] = FinalizeScores<2> (RunAffectBundle (bundle));
    }

  for (std::uint32_t mask = 0; mask < kFlashbulbCombos; ++mask)
    {
      FlashbulbBundle bundle;
      bundle.percentile = (mask & (1u << 0)) != 0u;
      bundle.rate = (mask & (1u << 1)) != 0u;
      bundle.arousal = (mask & (1u << 2)) != 0u;
      flashbulb_results[mask] = FinalizeScores<3> (RunFlashbulbBundle (bundle));
    }

  for (std::uint32_t mask = 0; mask < kNeuromodCombos; ++mask)
    {
      NeuromodBundle bundle;
      bundle.write = (mask & (1u << 0)) != 0u;
      bundle.competition = (mask & (1u << 1)) != 0u;
      bundle.recon = (mask & (1u << 2)) != 0u;
      bundle.value = (mask & (1u << 3)) != 0u;
      neuromod_results[mask] = FinalizeScores<4> (RunNeuromodBundle (bundle));
    }

  std::vector<GlobalResult> results;
  results.reserve (kGlobalCombos);
  for (std::uint32_t mask = 0; mask < kGlobalCombos; ++mask)
    {
      std::uint32_t singleton_mask = 0u;
      singleton_mask |= ((mask & (1u << kSourceConfidence)) != 0u) ? (1u << 0) : 0u;
      singleton_mask |= ((mask & (1u << kPredictiveRetrieval)) != 0u) ? (1u << 1) : 0u;
      singleton_mask |= ((mask & (1u << kConstructiveRecall)) != 0u) ? (1u << 2) : 0u;
      singleton_mask |= ((mask & (1u << kProceduralProactive)) != 0u) ? (1u << 3) : 0u;
      singleton_mask |= ((mask & (1u << kMetaLearning)) != 0u) ? (1u << 4) : 0u;
      singleton_mask |= ((mask & (1u << kReinforcementEdges)) != 0u) ? (1u << 5) : 0u;
      singleton_mask |= ((mask & (1u << kSequentialEdges)) != 0u) ? (1u << 6) : 0u;
      const std::uint32_t metacog_mask = (mask >> 4) & ((1u << 3) - 1u);
      const std::uint32_t affect_mask = (mask >> 7) & ((1u << 2) - 1u);
      const std::uint32_t flashbulb_mask = (mask >> 9) & ((1u << 3) - 1u);
      const std::uint32_t neuromod_mask = (mask >> 12) & ((1u << 4) - 1u);

      GlobalResult result;
      result.mask = mask;

      const auto &single = singleton_results[singleton_mask];
      result.scores[kSourceConfidence] = single.scores[0];
      result.scores[kPredictiveRetrieval] = single.scores[1];
      result.scores[kConstructiveRecall] = single.scores[2];
      result.scores[kProceduralProactive] = single.scores[3];
      result.scores[kMetaLearning] = single.scores[4];
      result.scores[kReinforcementEdges] = single.scores[5];
      result.scores[kSequentialEdges] = single.scores[6];

      const auto &metacog = metacog_results[metacog_mask];
      result.scores[kMetacogTot] = metacog.scores[0];
      result.scores[kMetacogUnknown] = metacog.scores[1];
      result.scores[kMetacogDecay] = metacog.scores[2];

      const auto &affect = affect_results[affect_mask];
      result.scores[kAffectInterrupt] = affect.scores[0];
      result.scores[kAffectRetrieval] = affect.scores[1];

      const auto &flashbulb = flashbulb_results[flashbulb_mask];
      result.scores[kFlashbulbPercentile] = flashbulb.scores[0];
      result.scores[kFlashbulbRate] = flashbulb.scores[1];
      result.scores[kFlashbulbArousal] = flashbulb.scores[2];

      const auto &neuromod = neuromod_results[neuromod_mask];
      result.scores[kNeuromodWrite] = neuromod.scores[0];
      result.scores[kNeuromodCompetition] = neuromod.scores[1];
      result.scores[kNeuromodReconsolidation] = neuromod.scores[2];
      result.scores[kNeuromodValue] = neuromod.scores[3];

      result.total = std::accumulate (result.scores.begin (), result.scores.end (),
                                      0.0);
      results.push_back (result);
    }

  const std::uint32_t full_mask = kGlobalCombos - 1u;
  const auto best_it = std::max_element (
      results.begin (), results.end (),
      [] (const GlobalResult &a, const GlobalResult &b) {
        return a.total < b.total;
      });
  const double best_total = best_it == results.end () ? 0.0 : best_it->total;
  double full_total = 0.0;
  std::array<double, kUnitCount> full_scores{};
  for (const auto &result : results)
    {
      if (result.mask == full_mask)
        {
          full_total = result.total;
          full_scores = result.scores;
          break;
        }
    }

  std::vector<const GlobalResult *> best_results;
  int minimal_best_bits = kUnitCount + 1;
  for (const auto &result : results)
    {
      if (std::abs (result.total - best_total) > kScoreEpsilon)
        {
          continue;
        }
      best_results.push_back (&result);
      minimal_best_bits = std::min (minimal_best_bits, BitCount<kUnitCount> (result.mask));
    }

  std::vector<const GlobalResult *> minimal_best_masks;
  for (const auto *result : best_results)
    {
      if (BitCount<kUnitCount> (result->mask) == minimal_best_bits)
        {
          minimal_best_masks.push_back (result);
        }
    }

  std::cout << std::fixed << std::setprecision (6);
  std::cout << "operation_unit_count=" << kUnitCount << "\n";
  std::cout << "operation_unit_combo_count=" << kGlobalCombos << "\n";
  std::cout << "operation_unit_factorized_singleton_combo_count=" << kSingletonCombos
            << "\n";
  std::cout << "operation_unit_factorized_metacog_combo_count=" << kMetacogCombos
            << "\n";
  std::cout << "operation_unit_factorized_affect_combo_count=" << kAffectCombos
            << "\n";
  std::cout << "operation_unit_factorized_flashbulb_combo_count=" << kFlashbulbCombos
            << "\n";
  std::cout << "operation_unit_factorized_neuromod_combo_count=" << kNeuromodCombos
            << "\n";
  std::cout << "operation_unit_factorized_local_evals="
            << (kSingletonCombos + kMetacogCombos + kAffectCombos
                + kFlashbulbCombos + kNeuromodCombos)
            << "\n";
  std::cout << "operation_unit_full_mask_score=" << full_total << "\n";
  std::cout << "operation_unit_best_score=" << best_total << "\n";
  std::cout << "operation_unit_full_mask_is_best="
            << ((std::abs (full_total - best_total) <= kScoreEpsilon) ? 1 : 0)
            << "\n";
  std::cout << "operation_unit_best_mask_count=" << best_results.size () << "\n";
  std::cout << "operation_unit_minimal_best_bits=" << minimal_best_bits << "\n";
  if (!minimal_best_masks.empty ())
    {
      std::cout << "operation_unit_minimal_best_enabled="
                << GlobalMaskSummary (minimal_best_masks.front ()->mask) << "\n";
      std::cout << "operation_unit_minimal_best_disabled="
                << GlobalDisabledSummary (minimal_best_masks.front ()->mask) << "\n";
    }
  for (int bit = 0; bit < kUnitCount; ++bit)
    {
      std::cout << "operation_unit_full_mask_component_"
                << kUnits[static_cast<std::size_t> (bit)].slug << "="
                << full_scores[static_cast<std::size_t> (bit)] << "\n";
    }

  std::cout << "\n| unit | cluster | mean_marginal | max_score_without | essential_for_best |\n";
  std::cout << "|---|---|---:|---:|---:|\n";
  for (int bit = 0; bit < kUnitCount; ++bit)
    {
      double sum = 0.0;
      int count = 0;
      double max_without = -std::numeric_limits<double>::infinity ();
      for (const auto &result : results)
        {
          if ((result.mask & (1u << bit)) != 0u)
            {
              continue;
            }
          max_without = std::max (max_without, result.total);
          const auto neighbor = result.mask | (1u << bit);
          sum += results[neighbor].total - result.total;
          ++count;
        }
      const double mean = count > 0 ? sum / static_cast<double> (count) : 0.0;
      std::cout << "| `" << kUnits[static_cast<std::size_t> (bit)].slug << "` | `"
                << kUnits[static_cast<std::size_t> (bit)].cluster << "` | "
                << mean << " | " << max_without << " | "
                << (max_without < best_total ? 1 : 0) << " |\n";
    }

  std::vector<const GlobalResult *> sorted;
  for (const auto &result : results)
    {
      sorted.push_back (&result);
    }
  std::sort (sorted.begin (), sorted.end (),
             [] (const GlobalResult *a, const GlobalResult *b) {
               if (std::abs (a->total - b->total) > kScoreEpsilon)
                 {
                   return a->total > b->total;
                 }
               return BitCount<kUnitCount> (a->mask)
                      < BitCount<kUnitCount> (b->mask);
             });

  std::cout << "\n| rank | score | bits_on | enabled | disabled |\n";
  std::cout << "|---|---:|---:|---|---|\n";
  for (std::size_t i = 0; i < std::min<std::size_t> (8, sorted.size ()); ++i)
    {
      std::cout << "| " << (i + 1) << " | " << sorted[i]->total << " | "
                << BitCount<kUnitCount> (sorted[i]->mask) << " | "
                << GlobalMaskSummary (sorted[i]->mask) << " | "
                << GlobalDisabledSummary (sorted[i]->mask) << " |\n";
    }

  return 0;
}
