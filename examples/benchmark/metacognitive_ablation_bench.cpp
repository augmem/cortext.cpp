#include <cortext/core/knobs.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/operations/metacognitive.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <any>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>
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
  explicit ForceRetrievalGateOp (
      cortext::ProcessorContext::MetacognitiveMode mode,
      double metacognitive_confidence = 1.0)
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
    pctx.metacognitive_mode = mode_;
    pctx.metacognitive_mode_expires_at = ctx.GetSignal ().timestamp + 1000;
    pctx.metacognitive_certainty_satisfied = false;
    pctx.metacognitive_confidence = metacognitive_confidence_;
  }

private:
  cortext::ProcessorContext::MetacognitiveMode mode_;
  double metacognitive_confidence_ = 1.0;
};

struct StudyResult
{
  std::string name;
  bool passed = false;
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
  Execute (const std::string & /*query*/,
           const std::vector<std::any> & /*params*/ = {}) override
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

std::vector<long long>
RunTotRetrieval (bool disable_tot, double metacognitive_confidence = 1.0)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  SeedMemory (*store, 1LL, query);
  for (long long id = 2; id <= 4; ++id)
    {
      store->Execute (
          "INSERT INTO embeddings(embedding_id, embedding, created_at) "
          "VALUES(?, ?, ?)",
          { id, ToFloatVec (MakeVec ({ { 1, 1.0f } })), 1LL });
      store->Execute (
          "INSERT INTO memories("
          "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
          "s_max, s_avg, strength, created_at) "
          "VALUES(?, ?, 'bench', 'ASSOCIATION', 1, 1, 'text', 0.5, 0.5, 1.0, 1)",
          { id, id });
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
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 1.0;
  cfg.encoder = &GetBenchEncoder ();

  std::unique_ptr<ScopedEnvVar> env;
  if (disable_tot)
    {
      env = std::make_unique<ScopedEnvVar> (
          "CORTEXT_DISABLE_METACOG_TOT_RECOVERY", "1");
    }
  else
    {
      env = std::make_unique<ScopedEnvVar> ("CORTEXT_DISABLE_METACOG_TOT_RECOVERY");
    }

  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (
          cortext::ProcessorContext::MetacognitiveMode::TotRecovery,
          metacognitive_confidence),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto out = processor.Process (MakeSignal (query, 10));
  processor.Flush ();
  return out.candidate_memory_ids;
}

StudyResult
RunTotStudy ()
{
  const auto off_ids = RunTotRetrieval (true);
  const auto on_ids = RunTotRetrieval (false);
  const bool hit_off
      = std::find (off_ids.begin (), off_ids.end (), 4LL) != off_ids.end ();
  const bool hit_on
      = std::find (on_ids.begin (), on_ids.end (), 4LL) != on_ids.end ();
  std::cout << "tot_recovery_hits_on=" << (hit_on ? 1 : 0)
            << " tot_recovery_hits_off=" << (hit_off ? 1 : 0) << "\n";
  return { "metacognitive_tot_recovery", hit_on && !hit_off };
}

std::vector<long long>
RunUnknownRetrieval (bool disable_unknown)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  SeedMemory (*store, 77LL, query);
  store->Execute ("UPDATE memories "
                  "SET source_origin = 'external', source_reliability = 0.1, "
                  "source_contradiction_count = 2 "
                  "WHERE memory_id = ?",
                  { 77LL });

  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.encoder = &GetBenchEncoder ();

  std::unique_ptr<ScopedEnvVar> env;
  if (disable_unknown)
    {
      env = std::make_unique<ScopedEnvVar> (
          "CORTEXT_DISABLE_METACOG_UNKNOWN_CAUTION", "1");
    }
  else
    {
      env = std::make_unique<ScopedEnvVar> (
          "CORTEXT_DISABLE_METACOG_UNKNOWN_CAUTION");
    }

  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (
          cortext::ProcessorContext::MetacognitiveMode::UnknownCaution),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  auto out = processor.Process (MakeSignal (query, 20));
  processor.Flush ();
  return out.candidate_memory_ids;
}

StudyResult
RunUnknownStudy ()
{
  const auto off_ids = RunUnknownRetrieval (true);
  const auto on_ids = RunUnknownRetrieval (false);
  std::cout << "unknown_empty_on=" << (on_ids.empty () ? 1 : 0)
            << " unknown_empty_off=" << (off_ids.empty () ? 1 : 0) << "\n";
  return { "metacognitive_unknown_caution",
           on_ids.empty () && !off_ids.empty () };
}

StudyResult
RunConfidenceDecayStudy ()
{
  cortext::Signal signal = MakeSignal (MakeVec ({ { 0, 1.0f } }), 11000);
  cortext::ProcessorContext pctx;
  pctx.metacognitive_confidence = 1.0;
  pctx.last_signal_timestamp = 1000;

  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.0;
  cfg.encoder = &GetBenchEncoder ();

  cortext::operations::MetacognitiveMonitoring op;
  cortext::OperationContext ctx (signal, pctx, cfg);
  ctx.SetFeelingOfKnowing (0.2);
  ctx.SetMemoryUsageEvents ({ { 1LL, true, -1.0 } });
  NullTransaction tx;
  op.Execute (ctx, tx);
  const double decayed_confidence = pctx.metacognitive_confidence;

  cortext::ProcessorContext pctx_no_decay;
  pctx_no_decay.metacognitive_confidence = 1.0;
  pctx_no_decay.last_signal_timestamp = 1000;
  cortext::OperationContext ctx_no_decay (signal, pctx_no_decay, cfg);
  ctx_no_decay.SetFeelingOfKnowing (0.2);
  ctx_no_decay.SetMemoryUsageEvents ({ { 1LL, true, -1.0 } });
  {
    ScopedEnvVar disable ("CORTEXT_DISABLE_METACOG_CONFIDENCE_DECAY", "1");
    op.Execute (ctx_no_decay, tx);
  }
  const double no_decay_confidence = pctx_no_decay.metacognitive_confidence;

  const auto low_ids = RunTotRetrieval (false, decayed_confidence);
  const auto high_ids = RunTotRetrieval (false, no_decay_confidence);
  const bool low_hit
      = std::find (low_ids.begin (), low_ids.end (), 4LL) != low_ids.end ();
  const bool high_hit
      = std::find (high_ids.begin (), high_ids.end (), 4LL) != high_ids.end ();

  std::cout << "confidence_decay_on=" << decayed_confidence
            << " confidence_decay_off=" << no_decay_confidence << "\n";
  std::cout << "tot_decay_hit_on=" << (low_hit ? 1 : 0)
            << " tot_decay_hit_off=" << (high_hit ? 1 : 0) << "\n";
  return { "metacognitive_confidence_decay",
           decayed_confidence < no_decay_confidence && high_hit && !low_hit };
}

StudyResult
RunExpiryStudy ()
{
  cortext::Signal signal = MakeSignal (MakeVec ({ { 0, 1.0f } }), 5000);
  cortext::ProcessorContext pctx;
  pctx.metacognitive_mode
      = cortext::ProcessorContext::MetacognitiveMode::TotRecovery;
  pctx.metacognitive_mode_expires_at = 4000;

  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.encoder = &GetBenchEncoder ();

  cortext::operations::MetacognitiveMonitoring op;
  cortext::OperationContext ctx (signal, pctx, cfg);
  ctx.SetFeelingOfKnowing (0.1);
  ctx.SetMemoryUsageEvents ({ { 1LL, true, 1.0 } });
  NullTransaction tx;
  op.Execute (ctx, tx);

  const bool ok
      = pctx.metacognitive_mode
            == cortext::ProcessorContext::MetacognitiveMode::Normal
        && pctx.metacognitive_mode_expires_at == 0;
  std::cout << "mode_expiry_ok=" << (ok ? 1 : 0) << "\n";
  return { "metacognitive_mode_expiry", ok };
}
} // namespace

int
main ()
{
  const std::vector<StudyResult> results{
    RunTotStudy (),
    RunUnknownStudy (),
    RunConfidenceDecayStudy (),
    RunExpiryStudy (),
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
