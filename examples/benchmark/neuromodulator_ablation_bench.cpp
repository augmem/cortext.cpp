#include <cortext/core/algorithms.hpp>
#include <cortext/core/utils.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/detect_memory_usage.hpp>
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

} // namespace

int
main ()
{
  const double write_on = RunWriteStudy (false);
  const double write_off = RunWriteStudy (true);

  std::cout << "neuromod_write_accept_on=" << write_on
            << " neuromod_write_accept_off=" << write_off << "\n";

  const bool pass = write_on > write_off;
  std::cout << "neuromodulator_bench_passed=" << (pass ? "1" : "0") << "\n";
  return pass ? 0 : 1;
}
