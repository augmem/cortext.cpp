#include <cortext/core/algorithms.hpp>
#include <cortext/core/utils.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/detect_memory_usage.hpp>
#include <cortext/operations/reconsolidation.hpp>
#include <cortext/operations/write_gate.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/accumulator_state.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <any>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
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
    if (existing)
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
MakeSignal (const Eigen::VectorXf &embedding, std::uint64_t ts)
{
  cortext::Signal s;
  s.embedding = embedding;
  s.timestamp = ts;
  s.source_id = "bench";
  s.modality = "text";
  s.mimetype = "text/plain";
  return s;
}

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
RunWriteStudy (bool disable_scale)
{
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
  std::optional<ScopedEnvVar> disable_guard;
  if (disable_scale)
    {
      disable_guard.emplace ("CORTEXT_DISABLE_NEUROMOD_WRITE_SCALE", "1");
    }

  cortext::operations::ComputeWriteGate op;
  NullTransaction tx;
  op.Execute (ctx, tx);
  return ctx.GetAccumulatorWriteDecision () ? 1.0 : 0.0;
}

double
RunReconStudy (bool disable_scale)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  Eigen::VectorXf current = MakeVec ({ { 0, 1.0f }, { 1, 0.9f } });
  const Eigen::VectorXf mem = MakeVec ({ { 0, 1.0f } });

  store->Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) VALUES(?, ?, ?)",
      { 9LL, ToFloatVec (mem), 1LL });
  store->Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, start_ts, n_signals, "
      "modality, s_max, s_avg, strength, created_at) "
      "VALUES(?, ?, 'bench', 'LONG_TERM', 1, 1, 'text', 0.5, 0.5, 1.0, 1)",
      { 9LL, 9LL });

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

  auto pipeline = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<SetupReconInputsOp> (
          current, std::unordered_map<long long, Eigen::VectorXf>{ { 9LL, mem } }),
      std::make_unique<SetNeuromodOp> (1.0, 0.0),
      std::make_unique<cortext::operations::ApplyReconsolidation> ());
  cortext::SignalProcessor processor (cfg, store, std::move (pipeline));

  ScopedEnvVar disable_constructive ("CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  std::optional<ScopedEnvVar> disable_guard;
  if (disable_scale)
    {
      disable_guard.emplace (
          "CORTEXT_DISABLE_NEUROMOD_RECONSOLIDATION_SCALE", "1");
    }
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
  return cortext::core::CosineSimilarity (updated, current);
}

} // namespace

int
main ()
{
  const double write_on = RunWriteStudy (false);
  const double write_off = RunWriteStudy (true);
  const double recon_on = RunReconStudy (false);
  const double recon_off = RunReconStudy (true);

  std::cout << "neuromod_write_accept_on=" << write_on
            << " neuromod_write_accept_off=" << write_off << "\n";
  std::cout << "neuromod_recon_similarity_on=" << recon_on
            << " neuromod_recon_similarity_off=" << recon_off << "\n";

  const bool pass = write_on > write_off
                    && recon_on > recon_off;
  std::cout << "neuromodulator_bench_passed=" << (pass ? "1" : "0") << "\n";
  return pass ? 0 : 1;
}
