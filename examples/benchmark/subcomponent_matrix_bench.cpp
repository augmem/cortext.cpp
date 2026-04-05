#include <cortext/core/algorithms.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/core/sparse.hpp>
#include <cortext/core/utils.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/boundary.hpp>
#include <cortext/operations/competition.hpp>
#include <cortext/operations/detect_memory_usage.hpp>
#include <cortext/operations/emotion.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/operations/interrupt_gate.hpp>
#include <cortext/operations/metacognitive.hpp>
#include <cortext/operations/reconsolidation.hpp>
#include <cortext/operations/write_gate.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/accumulator_state.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include "../../src/operations/retrieval_debug_state.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
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
  long long stored_id_;
  std::vector<double> history_;
  double rate_ewma_ = 0.0;
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

template <std::size_t N>
struct LocalResult
{
  std::uint32_t mask = 0;
  std::array<double, N> scores{};
  double total = 0.0;
};

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
MaskSummary (std::uint32_t mask,
             const std::array<const char *, N> &families)
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
      out << families[i];
    }
  return first ? "none" : out.str ();
}

template <std::size_t N>
std::string
DisabledSummary (std::uint32_t mask,
                 const std::array<const char *, N> &families)
{
  std::ostringstream out;
  bool first = true;
  for (std::size_t i = 0; i < N; ++i)
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
      out << families[i];
    }
  return first ? "none" : out.str ();
}

template <std::size_t N>
void
PrintMatrix (const std::string &name,
             const std::array<const char *, N> &families,
             const std::vector<LocalResult<N>> &results)
{
  const std::uint32_t full_mask = (1u << N) - 1u;
  const std::uint32_t combo_count = (1u << N);
  double best_score = 0.0;
  double full_score = 0.0;
  for (const auto &r : results)
    {
      best_score = std::max (best_score, r.total);
      if (r.mask == full_mask)
        {
          full_score = r.total;
        }
    }

  std::cout << name << "_family_count=" << N << "\n";
  std::cout << name << "_combo_count=" << combo_count << "\n";
  std::cout << name << "_full_mask_score=" << full_score << "\n";
  std::cout << name << "_best_score=" << best_score << "\n";
  std::cout << name << "_full_mask_is_best="
            << (std::abs (full_score - best_score) <= kScoreEpsilon ? 1 : 0)
            << "\n";
  for (const auto &r : results)
    {
      if (r.mask != full_mask)
        {
          continue;
        }
      for (std::size_t i = 0; i < N; ++i)
        {
          std::cout << name << "_full_mask_component_" << families[i] << "="
                    << r.scores[i] << "\n";
        }
      break;
    }

  std::cout << "\n| family | mean_marginal | max_score_without | essential_for_best |\n";
  std::cout << "|---|---:|---:|---:|\n";
  for (std::size_t bit = 0; bit < N; ++bit)
    {
      double sum = 0.0;
      int count = 0;
      double max_without = -1e9;
      for (const auto &r : results)
        {
          if ((r.mask & (1u << bit)) != 0u)
            {
              continue;
            }
          const auto neighbor = r.mask | (1u << bit);
          sum += results[neighbor].total - r.total;
          count++;
          max_without = std::max (max_without, r.total);
        }
      const double mean = count > 0 ? sum / static_cast<double> (count) : 0.0;
      std::cout << "| `" << families[bit] << "` | " << mean << " | "
                << max_without << " | " << (max_without < best_score ? 1 : 0)
                << " |\n";
    }

  std::vector<const LocalResult<N> *> sorted;
  for (const auto &r : results)
    {
      sorted.push_back (&r);
    }
  std::sort (sorted.begin (), sorted.end (),
             [] (const LocalResult<N> *a, const LocalResult<N> *b) {
               if (std::abs (a->total - b->total) > kScoreEpsilon)
                 {
                   return a->total > b->total;
                 }
               return BitCount<N> (a->mask) < BitCount<N> (b->mask);
             });
  std::cout << "\n| rank | score | bits_on | enabled | disabled |\n";
  std::cout << "|---|---:|---:|---|---|\n";
  for (std::size_t i = 0; i < std::min<std::size_t> (6, sorted.size ()); ++i)
    {
      std::cout << "| " << (i + 1) << " | " << sorted[i]->total << " | "
                << BitCount<N> (sorted[i]->mask) << " | "
                << MaskSummary<N> (sorted[i]->mask, families) << " | "
                << DisabledSummary<N> (sorted[i]->mask, families) << " |\n";
    }
  std::cout << "\n";
}

// Boundary matrix -------------------------------------------------------------

enum BoundaryBit : int
{
  kBoundaryPressure = 0,
  kBoundarySurprisal,
  kBoundaryNatural,
  kBoundaryBits
};

class ScopedBoundaryEnv
{
public:
  explicit ScopedBoundaryEnv (std::uint32_t mask)
  {
    Configure ("CORTEXT_BOUNDARY_DISABLE_PRESSURE",
               (mask & (1u << kBoundaryPressure)) != 0u);
    Configure ("CORTEXT_BOUNDARY_DISABLE_SURPRISAL",
               (mask & (1u << kBoundarySurprisal)) != 0u);
    Configure ("CORTEXT_BOUNDARY_DISABLE_NATURAL",
               (mask & (1u << kBoundaryNatural)) != 0u);
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

double
RunBoundaryPressureScenario (std::uint32_t mask)
{
  ScopedBoundaryEnv guards (mask);
  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  cortext::ProcessorContext pctx;
  cortext::AccumulatorState state;
  state.Reset (MakeVec ({ { 0, 1.0f } }), 1000);
  state.n_signals = 6;
  state.eta_acc = 0.05;
  state.last_signal_ts = 1100;
  state.drift_acc = 2.2;
  state.coherence_prev = 1.0;
  pctx.accumulator_states["bench"] = std::move (state);

  cortext::OperationContext ctx (
      MakeSignal (MakeVec ({ { 0, 1.0f } }), 1200), pctx, cfg);
  ctx.SetAccumulatorDriftStep (0.0);
  ctx.SetAccumulatorCoherence (1.0);
  ctx.SetMetric (cortext::operations::Metric::embedding_surprisal, 0.0);
  cortext::operations::DetectBoundary op;
  NullTransaction tx;
  op.Execute (ctx, tx);
  return ctx.GetFlushRequired () ? 1.0 : 0.0;
}

double
RunBoundarySurprisalScenario (std::uint32_t mask)
{
  ScopedBoundaryEnv guards (mask);
  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;

  cortext::ProcessorContext pctx;
  cortext::AccumulatorState state;
  state.Reset (MakeVec ({ { 0, 1.0f } }), 1000);
  state.n_signals = 6;
  state.eta_acc = 0.05;
  state.last_signal_ts = 1500;
  state.coherence_prev = 1.0;
  pctx.accumulator_states["bench"] = std::move (state);

  cortext::OperationContext ctx (
      MakeSignal (MakeVec ({ { 0, 1.0f } }), 2500), pctx, cfg);
  ctx.SetAccumulatorDriftStep (0.0);
  ctx.SetAccumulatorCoherence (1.0);
  ctx.SetMetric (cortext::operations::Metric::embedding_surprisal, 1.0);
  cortext::operations::DetectBoundary op;
  NullTransaction tx;
  op.Execute (ctx, tx);
  return ctx.GetFlushRequired () ? 1.0 : 0.0;
}

double
RunBoundaryNaturalScenario (std::uint32_t mask)
{
  ScopedBoundaryEnv guards (mask);
  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 0.6;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.3;

  cortext::ProcessorContext pctx;
  cortext::AccumulatorState state;
  state.Reset (MakeVec ({ { 0, 1.0f } }), 1000);
  state.n_signals = 6;
  state.eta_acc = 0.05;
  state.last_signal_ts = 1800;
  state.coherence_prev = 1.0;
  pctx.accumulator_states["bench"] = std::move (state);

  cortext::OperationContext ctx (
      MakeSignal (MakeVec ({ { 0, 0.2f }, { 1, 0.98f } }), 2400), pctx, cfg);
  ctx.SetAccumulatorDriftStep (0.8);
  ctx.SetAccumulatorCoherence (0.1);
  ctx.SetMetric (cortext::operations::Metric::embedding_surprisal, 0.9);
  cortext::operations::DetectBoundary op;
  NullTransaction tx;
  op.Execute (ctx, tx);
  return ctx.GetFlushRequired () ? 1.0 : 0.0;
}

LocalResult<kBoundaryBits>
RunBoundaryMatrixMask (std::uint32_t mask)
{
  LocalResult<kBoundaryBits> result;
  result.mask = mask;
  result.scores[kBoundaryPressure] = RunBoundaryPressureScenario (mask);
  result.scores[kBoundarySurprisal] = RunBoundarySurprisalScenario (mask);
  result.scores[kBoundaryNatural] = RunBoundaryNaturalScenario (mask);
  for (double score : result.scores)
    {
      result.total += score;
    }
  return result;
}

// Flashbulb matrix ------------------------------------------------------------

enum FlashbulbBit : int
{
  kFlashbulbPercentile = 0,
  kFlashbulbRate,
  kFlashbulbArousal,
  kFlashbulbBits
};

class ScopedFlashbulbEnv
{
public:
  explicit ScopedFlashbulbEnv (std::uint32_t mask)
  {
    Configure ("CORTEXT_FLASHBULB_DISABLE_PERCENTILE",
               (mask & (1u << kFlashbulbPercentile)) != 0u);
    Configure ("CORTEXT_FLASHBULB_DISABLE_RATE",
               (mask & (1u << kFlashbulbRate)) != 0u);
    Configure ("CORTEXT_FLASHBULB_DISABLE_AROUSAL",
               (mask & (1u << kFlashbulbArousal)) != 0u);
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

bool
RunFlashbulbCase (std::uint32_t mask, double emotion, double arousal,
                  const std::vector<double> &history, double rate_ewma)
{
  ScopedFlashbulbEnv guards (mask);
  auto store = CreateStore ();
  SeedMemory (*store, 101LL, 101LL, MakeVec ({ { 0, 1.0f } }));
  store->Execute (
      "UPDATE memories SET s_emotion_max = ?, s_arousal_avg = ? WHERE memory_id = ?",
      { emotion, arousal, 101LL });

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 0.4;
  cfg.sensitivity = 0.8;
  cfg.stability = 0.5;

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
RunFlashbulbPercentileScenario (std::uint32_t mask)
{
  const std::vector<double> history (16, 0.98);
  const bool flashbulb = RunFlashbulbCase (mask, 0.90, 0.90, history, 0.0);
  return flashbulb ? 0.0 : 1.0;
}

double
RunFlashbulbRateScenario (std::uint32_t mask)
{
  const bool flashbulb = RunFlashbulbCase (mask, 0.76, 0.88, {}, 1.0);
  return flashbulb ? 0.0 : 1.0;
}

double
RunFlashbulbArousalScenario (std::uint32_t mask)
{
  const bool flashbulb = RunFlashbulbCase (mask, 0.96, 0.45, {}, 0.0);
  return flashbulb ? 0.0 : 1.0;
}

LocalResult<kFlashbulbBits>
RunFlashbulbMatrixMask (std::uint32_t mask)
{
  LocalResult<kFlashbulbBits> result;
  result.mask = mask;
  result.scores[kFlashbulbPercentile] = RunFlashbulbPercentileScenario (mask);
  result.scores[kFlashbulbRate] = RunFlashbulbRateScenario (mask);
  result.scores[kFlashbulbArousal] = RunFlashbulbArousalScenario (mask);
  for (double score : result.scores)
    {
      result.total += score;
    }
  return result;
}

// Metacognitive matrix --------------------------------------------------------

enum MetacogBit : int
{
  kMetacogTot = 0,
  kMetacogUnknown,
  kMetacogDecay,
  kMetacogBits
};

class ScopedMetacogEnv
{
public:
  explicit ScopedMetacogEnv (std::uint32_t mask)
  {
    Configure ("CORTEXT_DISABLE_METACOG_TOT_RECOVERY",
               (mask & (1u << kMetacogTot)) != 0u);
    Configure ("CORTEXT_DISABLE_METACOG_UNKNOWN_CAUTION",
               (mask & (1u << kMetacogUnknown)) != 0u);
    Configure ("CORTEXT_DISABLE_METACOG_CONFIDENCE_DECAY",
               (mask & (1u << kMetacogDecay)) != 0u);
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

std::vector<long long>
RunTotRetrievalForMask (std::uint32_t mask, double metacognitive_confidence)
{
  ScopedMetacogEnv guards (mask);
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

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 1.0;

  auto ops = std::make_unique<cortext::OperationSet> (
      std::make_unique<ForceRetrievalGateOp> (
          cortext::ProcessorContext::MetacognitiveMode::TotRecovery,
          metacognitive_confidence),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto out = processor.Process (MakeSignal (query, 10));
  processor.Flush ();
  return out.candidate_memory_ids;
}

double
RunMetacogTotScenario (std::uint32_t mask)
{
  const auto ids = RunTotRetrievalForMask (mask, 1.0);
  const bool hit = std::find (ids.begin (), ids.end (), 4LL) != ids.end ();
  return hit ? 1.0 : 0.0;
}

double
RunMetacogUnknownScenario (std::uint32_t mask)
{
  ScopedMetacogEnv guards (mask);
  auto store = CreateStore ();
  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  SeedMemory (*store, 77LL, 77LL, query);
  store->Execute ("UPDATE memories "
                  "SET source_origin = 'external', source_reliability = 0.1, "
                  "source_contradiction_count = 2 "
                  "WHERE memory_id = ?",
                  { 77LL });

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

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
RunMetacogDecayScenario (std::uint32_t mask)
{
  ScopedMetacogEnv guards (mask);
  cortext::Signal signal = MakeSignal (MakeVec ({ { 0, 1.0f } }), 11000);
  cortext::ProcessorContext pctx;
  pctx.metacognitive_confidence = 1.0;
  pctx.last_signal_timestamp = 1000;

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.0;

  cortext::operations::MetacognitiveMonitoring op;
  cortext::OperationContext ctx (signal, pctx, cfg);
  ctx.SetFeelingOfKnowing (0.2);
  ctx.SetMemoryUsageEvents ({ { 1LL, true, -1.0 } });
  NullTransaction tx;
  op.Execute (ctx, tx);

  const auto ids = RunTotRetrievalForMask (mask, pctx.metacognitive_confidence);
  const bool hit = std::find (ids.begin (), ids.end (), 4LL) != ids.end ();
  return hit ? 0.0 : 1.0;
}

LocalResult<kMetacogBits>
RunMetacogMatrixMask (std::uint32_t mask)
{
  LocalResult<kMetacogBits> result;
  result.mask = mask;
  result.scores[kMetacogTot] = RunMetacogTotScenario (mask);
  result.scores[kMetacogUnknown] = RunMetacogUnknownScenario (mask);
  result.scores[kMetacogDecay] = RunMetacogDecayScenario (mask);
  for (double score : result.scores)
    {
      result.total += score;
    }
  return result;
}

// Affect matrix ---------------------------------------------------------------

enum AffectBit : int
{
  kAffectInterruptBit = 0,
  kAffectRetrievalBit,
  kAffectBits
};

cortext::SignalProcessor::Config
MakeAffectConfig (std::uint32_t mask, double focus, double sensitivity,
                  double stability)
{
  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = focus;
  cfg.sensitivity = sensitivity;
  cfg.stability = stability;
  cfg.affect_interrupt = (mask & (1u << kAffectInterruptBit)) != 0u;
  cfg.affect_retrieval = (mask & (1u << kAffectRetrievalBit)) != 0u;
  return cfg;
}

double
RunAffectInterruptScenario (std::uint32_t mask)
{
  auto store = CreateStore ();
  const Eigen::VectorXf signal_emb = MakeVec ({ { 0, 1.0f }, { 1, 0.02f } });
  const Eigen::VectorXf cand_emb = MakeVec ({ { 0, 0.93f }, { 1, 0.367f } });
  SeedMemory (*store, 1LL, 1LL, cand_emb);

  auto cfg = MakeAffectConfig (mask, 0.7, 0.9, 0.2);
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
RunAffectRetrievalScenario (std::uint32_t mask)
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

  auto cfg = MakeAffectConfig (mask, 0.5, 1.0, 0.5);
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

LocalResult<kAffectBits>
RunAffectMatrixMask (std::uint32_t mask)
{
  LocalResult<kAffectBits> result;
  result.mask = mask;
  result.scores[kAffectInterruptBit] = RunAffectInterruptScenario (mask);
  result.scores[kAffectRetrievalBit] = RunAffectRetrievalScenario (mask);
  for (double score : result.scores)
    {
      result.total += score;
    }
  return result;
}

// Neuromodulation matrix ------------------------------------------------------

enum NeuromodBit : int
{
  kNeuromodWrite = 0,
  kNeuromodCompetition,
  kNeuromodReconsolidation,
  kNeuromodValueGain,
  kNeuromodBits
};

class ScopedNeuromodEnv
{
public:
  explicit ScopedNeuromodEnv (std::uint32_t mask)
  {
    Configure ("CORTEXT_DISABLE_NEUROMOD_WRITE_SCALE",
               (mask & (1u << kNeuromodWrite)) != 0u);
    Configure ("CORTEXT_DISABLE_NEUROMOD_COMPETITION_SCALE",
               (mask & (1u << kNeuromodCompetition)) != 0u);
    Configure ("CORTEXT_DISABLE_NEUROMOD_RECONSOLIDATION_SCALE",
               (mask & (1u << kNeuromodReconsolidation)) != 0u);
    Configure ("CORTEXT_DISABLE_NEUROMOD_VALUE_GAIN",
               (mask & (1u << kNeuromodValueGain)) != 0u);
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

double
RunNeuromodWriteScenario (std::uint32_t mask)
{
  ScopedNeuromodEnv guards (mask);
  cortext::Signal s = MakeSignal (MakeVec ({ { 0, 1.0f } }), 100000);
  cortext::ProcessorContext pctx;
  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

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
RunNeuromodCompetitionScenario (std::uint32_t mask)
{
  ScopedNeuromodEnv guards (mask);
  auto store = CreateStore ();

  const Eigen::VectorXf ctx = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf w1 = MakeVec ({ { 0, 0.99f }, { 1, 0.05f } });
  const Eigen::VectorXf w2 = MakeVec ({ { 0, 0.98f }, { 1, 0.06f } });
  const Eigen::VectorXf w3 = MakeVec ({ { 0, 0.95f }, { 1, 0.10f } });
  const Eigen::VectorXf loser = MakeVec ({ { 0, 0.88f }, { 1, 0.47f } });
  std::unordered_map<long long, Eigen::VectorXf> retrieved{
    { 10LL, w1 }, { 11LL, w2 }, { 12LL, w3 }, { 13LL, loser }
  };

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 1.0;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

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
      = rows.empty () ? 1.0 : std::any_cast<double> (rows[0].at ("strength"));
  return loser_strength < 0.999 ? 1.0 : 0.0;
}

double
RunNeuromodReconScenario (std::uint32_t mask)
{
  ScopedNeuromodEnv guards (mask);
  auto store = CreateStore ();

  const Eigen::VectorXf current = MakeVec ({ { 0, 1.0f }, { 1, 0.9f } });
  const Eigen::VectorXf mem = MakeVec ({ { 0, 1.0f } });
  SeedMemory (*store, 9LL, 9LL, mem);

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

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
  const double similarity
      = cortext::core::CosineSimilarity (updated, current);
  return similarity > 0.82 ? 1.0 : 0.0;
}

double
RunValueStudyForMask (std::uint32_t mask, double neuromod_da)
{
  ScopedNeuromodEnv guards (mask);
  auto store = CreateStore ();
  const Eigen::VectorXf emb = MakeVec ({ { 0, 1.0f } });
  SeedMemory (*store, 7LL, 7LL, emb);

  cortext::ProcessorContext pctx;
  pctx.neuromod_da = neuromod_da;
  pctx.delta_reward = 0.8;

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
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

  const int k_key = cortext::core::SparseKeySize (cfg.focus);
  const std::string key = cortext::core::SparseKey (signal.embedding, k_key);
  return pctx.procedural_store[key][7LL];
}

double
RunNeuromodValueScenario (std::uint32_t mask)
{
  const double high = RunValueStudyForMask (mask, 1.0);
  const double low = RunValueStudyForMask (mask, 0.0);
  return high > low ? 1.0 : 0.0;
}

LocalResult<kNeuromodBits>
RunNeuromodMatrixMask (std::uint32_t mask)
{
  LocalResult<kNeuromodBits> result;
  result.mask = mask;
  result.scores[kNeuromodWrite] = RunNeuromodWriteScenario (mask);
  result.scores[kNeuromodCompetition] = RunNeuromodCompetitionScenario (mask);
  result.scores[kNeuromodReconsolidation] = RunNeuromodReconScenario (mask);
  result.scores[kNeuromodValueGain] = RunNeuromodValueScenario (mask);
  for (double score : result.scores)
    {
      result.total += score;
    }
  return result;
}

} // namespace

int
main ()
{
  std::cout << std::fixed << std::setprecision (6);

  {
    std::vector<LocalResult<kBoundaryBits>> results;
    const std::uint32_t combos = (1u << kBoundaryBits);
    results.reserve (combos);
    for (std::uint32_t mask = 0; mask < combos; ++mask)
      {
        results.push_back (RunBoundaryMatrixMask (mask));
      }
    PrintMatrix<kBoundaryBits> (
        "subcomponent_boundary",
        { "pressure", "surprisal", "natural" }, results);
  }

  {
    std::vector<LocalResult<kFlashbulbBits>> results;
    const std::uint32_t combos = (1u << kFlashbulbBits);
    results.reserve (combos);
    for (std::uint32_t mask = 0; mask < combos; ++mask)
      {
        results.push_back (RunFlashbulbMatrixMask (mask));
      }
    PrintMatrix<kFlashbulbBits> (
        "subcomponent_flashbulb",
        { "percentile", "rate", "arousal" }, results);
  }

  {
    std::vector<LocalResult<kMetacogBits>> results;
    const std::uint32_t combos = (1u << kMetacogBits);
    results.reserve (combos);
    for (std::uint32_t mask = 0; mask < combos; ++mask)
      {
        results.push_back (RunMetacogMatrixMask (mask));
      }
    PrintMatrix<kMetacogBits> (
        "subcomponent_metacognitive",
        { "tot_recovery", "unknown_caution", "confidence_decay" }, results);
  }

  {
    std::vector<LocalResult<kAffectBits>> results;
    const std::uint32_t combos = (1u << kAffectBits);
    results.reserve (combos);
    for (std::uint32_t mask = 0; mask < combos; ++mask)
      {
        results.push_back (RunAffectMatrixMask (mask));
      }
    PrintMatrix<kAffectBits> (
        "subcomponent_affect",
        { "interrupt_path", "retrieval_path" }, results);
  }

  {
    std::vector<LocalResult<kNeuromodBits>> results;
    const std::uint32_t combos = (1u << kNeuromodBits);
    results.reserve (combos);
    for (std::uint32_t mask = 0; mask < combos; ++mask)
      {
        results.push_back (RunNeuromodMatrixMask (mask));
      }
    PrintMatrix<kNeuromodBits> (
        "subcomponent_neuromod",
        { "write_scale", "competition_scale", "reconsolidation_scale",
          "value_gain" },
        results);
  }

  return 0;
}
