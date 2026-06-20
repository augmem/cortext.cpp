#include "../../src/operations/retrieval_trace_state.hpp"

#include <cortext/core/sparse.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace
{
constexpr int kEmbeddingDim = 256;

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
SeedMemory (cortext::Store &store, long long id, const Eigen::VectorXf &embedding)
{
  store.Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) VALUES(?, ?, ?)",
      { id, ToFloatVec (embedding), 1LL });
  store.Execute (
      "INSERT INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, created_at) "
      "VALUES(?, ?, 'bench', 'LONG_TERM', 1, 1, 'text', 0.5, 0.5, 1.0, 1)",
      { id, id });
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

struct StudyResult
{
  bool selected = false;
  bool top1 = false;
  double proc_score = 0.0;
};

StudyResult
RunProceduralStudy (bool disable_proactive)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf routine_target
      = MakeVec ({ { 0, 0.30f }, { 1, 0.9539392f } });
  SeedMemory (*store, 500LL, routine_target);
  for (int i = 0; i < 9; ++i)
    {
      const float x = 0.44f - 0.01f * static_cast<float> (i);
      const float y = std::sqrt (std::max (0.0f, 1.0f - x * x));
      SeedMemory (*store, 600LL + i, MakeVec ({ { 0, x }, { 1, y } }));
    }

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 1.0;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;
  cfg.procedural_enabled = true;

  cortext::operations::retrieval_trace::ClearLastRankedCandidates ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<SeedProceduralStoreOp> (500LL, 1.0),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  std::optional<ScopedEnvVar> disable_guard;
  if (disable_proactive)
    {
      disable_guard.emplace ("CORTEXT_DISABLE_PROCEDURAL_PROACTIVE_RETRIEVAL",
                             "1");
    }
  processor.Process (MakeSignal (query, 10));
  processor.Flush ();

  StudyResult result;
  const auto &ranked
      = cortext::operations::retrieval_trace::GetLastRankedCandidates ();
  for (const auto &candidate : ranked)
    {
      if (candidate.memory_id == 500LL)
        {
          result.selected = true;
          result.proc_score = candidate.proc_score;
        }
    }
  result.top1 = !ranked.empty () && ranked.front ().memory_id == 500LL;
  return result;
}

} // namespace

int
main ()
{
  const StudyResult on = RunProceduralStudy (false);
  const StudyResult off = RunProceduralStudy (true);

  std::cout << "procedural_target_selected_on=" << (on.selected ? "1" : "0")
            << " procedural_target_selected_off=" << (off.selected ? "1" : "0")
            << "\n";
  std::cout << "procedural_target_top1_on=" << (on.top1 ? "1" : "0")
            << " procedural_target_top1_off=" << (off.top1 ? "1" : "0")
            << "\n";
  std::cout << "procedural_target_proc_score_on=" << on.proc_score << "\n";

  const bool pass = on.selected && on.top1 && !off.selected && !off.top1
                    && on.proc_score > 0.99;
  std::cout << "procedural_bench_passed=" << (pass ? "1" : "0") << "\n";
  return pass ? 0 : 1;
}
