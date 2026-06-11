#include "../../src/operations/retrieval_debug_state.hpp"

#include <cortext/operations/predictive.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <any>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
constexpr int kEmbeddingDim = 256;

class BenchEncoder : public cortext::Encoder
{
public:
  void
  EncodeText (const std::string & /*text*/,
              std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }

  void
  EncodeAudio (const float * /*pcm*/, std::size_t /*num_samples*/,
               std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }

  void
  EncodeImage (const std::uint8_t * /*data*/, int /*width*/, int /*height*/,
               int /*channels*/, std::vector<float> &out_embedding) override
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
MakeSignal (const Eigen::VectorXf &embedding, std::uint64_t ts)
{
  cortext::Signal signal;
  signal.embedding = embedding;
  signal.timestamp = ts;
  signal.source_id = "bench";
  return signal;
}

void
SeedMemory (cortext::Store &store, long long id, const Eigen::VectorXf &embedding,
            double pre_activation = 0.0)
{
  store.Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) VALUES(?, ?, ?)",
      { id, ToFloatVec (embedding), 1LL });
  store.Execute (
      "INSERT INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, pre_activation, created_at) "
      "VALUES(?, ?, 'bench', 'LONG_TERM', 1, 1, 'text', 0.5, 0.5, 1.0, ?, 1)",
      { id, id, pre_activation });
}

class ForceRetrievalGateOp : public cortext::IOperation
{
public:
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
  }
};

class SetupPredictiveInputsOp : public cortext::IOperation
{
public:
  SetupPredictiveInputsOp (std::vector<Eigen::VectorXf> recent,
                           std::unordered_map<long long, Eigen::VectorXf> retrieved,
                           double surprise)
      : recent_ (std::move (recent)),
        retrieved_ (std::move (retrieved)),
        surprise_ (surprise)
  {
  }

  void
  Execute (cortext::OperationContext &ctx,
           cortext::Transaction & /*tx*/) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    pctx.recent_context_embeddings.clear ();
    for (const auto &embedding : recent_)
      {
        pctx.recent_context_embeddings.push_back (embedding);
      }
    ctx.SetRetrievedMemoryEmbeddings (retrieved_);
    ctx.SetMetric (cortext::operations::Metric::surprise, surprise_);
  }

private:
  std::vector<Eigen::VectorXf> recent_;
  std::unordered_map<long long, Eigen::VectorXf> retrieved_;
  double surprise_ = 0.0;
};

struct StudyResult
{
  std::string name;
  bool passed = false;
};

StudyResult
RunPredictiveRankingStudy ()
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf predictive_target
      = MakeVec ({ { 0, 0.8f }, { 1, 0.6f } });
  const Eigen::VectorXf raw_best
      = MakeVec ({ { 0, 0.95f }, { 1, 0.3122499f } });
  SeedMemory (*store, 11LL, predictive_target, 1.0);
  SeedMemory (*store, 22LL, raw_best, 0.0);

  cortext::SignalProcessor::Config cfg;
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.encoder = &GetBenchEncoder ();

  auto run = [&] {
    cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
    auto ops = std::make_unique<cortext::DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
    cortext::SignalProcessor processor (cfg, store, std::move (ops));
    processor.Process (MakeSignal (query, 10));
    processor.Flush ();
    return cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  };

  long long top_on = 0;
  long long top_off = 0;
  {
    ScopedEnvVar disable ("CORTEXT_DISABLE_PREDICTIVE_RETRIEVAL_BONUS", "1");
    const auto ranked = run ();
    top_off = ranked.empty () ? 0LL : ranked.front ().memory_id;
  }
  {
    ScopedEnvVar enable ("CORTEXT_DISABLE_PREDICTIVE_RETRIEVAL_BONUS");
    const auto ranked = run ();
    top_on = ranked.empty () ? 0LL : ranked.front ().memory_id;
  }

  {
    const auto ranked = run ();
    for (const auto &candidate : ranked)
      {
        std::cout << "predictive_ranking_candidate id=" << candidate.memory_id
                  << " score=" << candidate.score
                  << " relevance=" << candidate.relevance
                  << " pre_activation=" << candidate.pre_activation
                  << " predictive_bonus=" << candidate.predictive_bonus
                  << " durable_boost=" << candidate.durable_source_boost
                  << " durable_count=" << candidate.durable_source_count
                  << " label_boost=" << candidate.label_graph_boost
                  << "\n";
      }
  }
  std::cout << "predictive_target_top1_on=" << (top_on == 11LL ? 1 : 0)
            << " predictive_target_top1_off=" << (top_off == 11LL ? 1 : 0)
            << "\n";
  return { "predictive_ranking",
           top_on == 11LL && top_off == 22LL };
}

StudyResult
RunPredictiveDecayStudy ()
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  const Eigen::VectorXf aligned = MakeVec ({ { 0, 1.0f } });
  SeedMemory (*store, 404LL, aligned, 0.8);

  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.encoder = &GetBenchEncoder ();

  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<cortext::operations::ApplyPredictivePreActivation> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (aligned, 20));
  processor.Flush ();

  const auto rows = store->Execute (
      "SELECT pre_activation FROM memories WHERE memory_id = ?",
      { 404LL });
  const double pre_activation
      = rows.empty () ? 0.0 : std::any_cast<double> (rows[0].at ("pre_activation"));
  std::cout << "preactivation_after_decay=" << pre_activation << "\n";
  return { "predictive_decay",
           std::abs (pre_activation - 0.4) < 1e-6 };
}

double
RunPredictiveRefresh (double surprise)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  const Eigen::VectorXf e0 = MakeVec ({ { 0, 1.0f }, { 1, 0.0f } });
  const Eigen::VectorXf e1 = MakeVec ({ { 0, 1.0f }, { 1, 0.1f } });
  const Eigen::VectorXf e2 = MakeVec ({ { 0, 1.0f }, { 1, 0.2f } });
  const Eigen::VectorXf aligned = MakeVec ({ { 0, 1.0f }, { 1, 0.22f } });
  SeedMemory (*store, 505LL, aligned, 0.0);

  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.8;
  cfg.stability = 0.3;
  cfg.encoder = &GetBenchEncoder ();

  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<SetupPredictiveInputsOp> (
          std::vector<Eigen::VectorXf>{ e0, e1, e2 },
          std::unordered_map<long long, Eigen::VectorXf>{ { 505LL, aligned } },
          surprise),
      std::make_unique<cortext::operations::ApplyPredictivePreActivation> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (aligned, 30));
  processor.Flush ();

  const auto rows = store->Execute (
      "SELECT pre_activation FROM memories WHERE memory_id = ?",
      { 505LL });
  return rows.empty () ? 0.0
                       : std::any_cast<double> (rows[0].at ("pre_activation"));
}

StudyResult
RunPredictiveSurpriseStudy ()
{
  const double low = RunPredictiveRefresh (0.0);
  const double high = RunPredictiveRefresh (1.0);
  std::cout << "surprise_refresh_low=" << low
            << " surprise_refresh_high=" << high << "\n";
  return { "predictive_surprise_refresh", high > low };
}
} // namespace

int
main ()
{
  const std::vector<StudyResult> results{
    RunPredictiveRankingStudy (),
    RunPredictiveDecayStudy (),
    RunPredictiveSurpriseStudy (),
  };

  int passed = 0;
  for (const auto &result : results)
    {
      std::cout << "study=" << result.name
                << " passed=" << (result.passed ? 1 : 0) << "\n";
      if (result.passed)
        {
          ++passed;
        }
    }
  std::cout << "summary=" << passed << "/" << results.size () << " passed\n";
  return passed == static_cast<int> (results.size ()) ? 0 : 1;
}
