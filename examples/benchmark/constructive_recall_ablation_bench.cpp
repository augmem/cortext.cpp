#include "../../src/operations/constructive_recall_internal.hpp"
#include "../../src/operations/retrieval_trace_state.hpp"

#include <cortext/core/algorithms.hpp>
#include <cortext/core/utils.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/operations/memory_storage.hpp>
#include <cortext/operations/reconsolidation.hpp>
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
  signal.modality = "text";
  signal.mimetype = "text/plain";
  return signal;
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
  return 0.0;
}

Eigen::VectorXf
LoadEmbeddingById (cortext::Store &store, long long embedding_id)
{
  auto rows = store.Execute (
      "SELECT embedding FROM embeddings WHERE embedding_id = ?",
      { embedding_id });
  Eigen::VectorXf out;
  if (rows.empty ()
      || !cortext::core::DecodeFloatBlob (rows[0].at ("embedding"),
                                          kEmbeddingDim, out))
    {
      return Eigen::VectorXf ();
    }
  return out;
}

int
RunInitialSeedStudy ()
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  cortext::Signal signal = MakeSignal (MakeVec ({ { 0, 1.0f } }), 12345);

  cortext::ProcessorContext pctx;
  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  cortext::AccumulatorState acc;
  acc.mu_acc = signal.embedding;
  acc.n_signals = 1;
  acc.s_sum = 0.5;
  acc.s_max = 0.5;
  acc.t_start = signal.timestamp - 1000;
  {
    cortext::SignalRecord rec;
    rec.embedding = signal.embedding;
    rec.timestamp = signal.timestamp;
    rec.modality = signal.modality;
    rec.mime = signal.mimetype;
    rec.score = 0.5;
    rec.serial_position = 0;
    acc.signals.push_back (std::move (rec));
  }
  pctx.accumulator_states[signal.source_id] = std::move (acc);

  cortext::OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetAccumulatorWriteDecision (true);
  ctx.SetRepresentativeEmbedding (signal.embedding);

  cortext::operations::MemoryStorage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  const auto stored_id = ctx.GetStoredEmbeddingId ();
  if (!stored_id.has_value ())
    {
      std::cout << "constructive_initial_seeded=0\n";
      return 1;
    }

  auto mem_rows = store->Execute (
      "SELECT memory_id FROM memories WHERE embedding_id = ?",
      { *stored_id });
  const long long memory_id
      = mem_rows.empty () ? 0LL : AnyToInt64 (mem_rows[0].at ("memory_id"));
  auto recon_rows = store->Execute (
      "SELECT embedding_id, trigger FROM memory_reconstructions "
      "WHERE memory_id = ? ORDER BY reconstruction_id",
      { memory_id });
  const int seeded = (recon_rows.size () == 1
                      && AnyToInt64 (recon_rows[0].at ("embedding_id")) == *stored_id
                      && std::any_cast<std::string> (recon_rows[0].at ("trigger"))
                             == "initial")
                         ? 1
                         : 0;
  std::cout << "constructive_initial_seeded=" << seeded << "\n";
  return seeded == 1 ? 0 : 1;
}

int
RunRetrievalAblationStudy ()
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  const Eigen::VectorXf query = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf target_base = MakeVec ({ { 1, 1.0f } });
  const Eigen::VectorXf target_reconstructed = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf rival = MakeVec ({ { 0, 0.92f }, { 1, 0.39f } });

  store->Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) VALUES(?, ?, ?)",
      { 11LL, ToFloatVec (target_base), 1LL });
  store->Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, start_ts, "
      "n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES(?, ?, 'bench', 'LONG_TERM', 1, 1, 'text', 0.5, 0.5, 1.0, 1)",
      { 11LL, 11LL });
  store->Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) VALUES(?, ?, ?)",
      { 22LL, ToFloatVec (rival), 1LL });
  store->Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, start_ts, "
      "n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES(?, ?, 'bench', 'LONG_TERM', 1, 1, 'text', 0.5, 0.5, 1.0, 1)",
      { 22LL, 22LL });
  store->Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) VALUES(?, ?, ?)",
      { 111LL, ToFloatVec (target_reconstructed), 2LL });
  store->Execute (
      "INSERT INTO memory_reconstructions("
      "memory_id, embedding_id, created_at, uncertainty, trigger"
      ") VALUES(?, ?, ?, ?, 'initial')",
      { 11LL, 111LL, 2LL, 0.0 });

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto run = [&] {
    cortext::operations::retrieval_trace::ClearLastRankedCandidates ();
    auto ops = std::make_unique<cortext::DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
    cortext::SignalProcessor processor (cfg, store, std::move (ops));
    processor.Process (MakeSignal (query, 10));
    processor.Flush ();
    return cortext::operations::retrieval_trace::GetLastRankedCandidates ();
  };

  long long top_off = 0;
  long long top_on = 0;
  long long recon_count_off = 0;
  long long recon_count_on = 0;
  double latest_uncertainty = 0.0;

  {
    ScopedEnvVar disable ("CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
    const auto ranked = run ();
    top_off = ranked.empty () ? 0LL : ranked.front ().memory_id;
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM memory_reconstructions WHERE memory_id = ?",
        { 11LL });
    recon_count_off = rows.empty () ? 0LL : AnyToInt64 (rows[0].at ("cnt"));
  }
  {
    ScopedEnvVar enable ("CORTEXT_DISABLE_CONSTRUCTIVE_RECALL");
    const auto ranked = run ();
    top_on = ranked.empty () ? 0LL : ranked.front ().memory_id;
    auto rows = store->Execute (
        "SELECT COUNT(*) AS cnt FROM memory_reconstructions WHERE memory_id = ?",
        { 11LL });
    recon_count_on = rows.empty () ? 0LL : AnyToInt64 (rows[0].at ("cnt"));
    auto latest = store->Execute (
        "SELECT uncertainty FROM memory_reconstructions "
        "WHERE memory_id = ? ORDER BY reconstruction_id DESC LIMIT 1",
        { 11LL });
    latest_uncertainty
        = latest.empty () ? 0.0 : AnyToDouble (latest[0].at ("uncertainty"));
  }

  std::cout << "constructive_target_top1_on=" << (top_on == 11LL ? 1 : 0)
            << " constructive_target_top1_off=" << (top_off == 11LL ? 1 : 0)
            << " constructive_recon_rows_on=" << recon_count_on
            << " constructive_recon_rows_off=" << recon_count_off
            << " constructive_latest_uncertainty=" << latest_uncertainty
            << "\n";

  const bool passed = top_off == 22LL && top_on == 11LL
                      && recon_count_off == 1 && recon_count_on == 2
                      && latest_uncertainty > 0.0;
  return passed ? 0 : 1;
}

int
RunReconsolidationStudy ()
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  const Eigen::VectorXf evidence = MakeVec ({ { 0, 1.0f } });
  const Eigen::VectorXf current = MakeVec ({ { 0, 0.4f }, { 1, 0.9165f } });

  store->Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) VALUES(?, ?, ?)",
      { 1LL, ToFloatVec (evidence), 1LL });
  store->Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, start_ts, "
      "n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES(?, ?, 'bench', 'LONG_TERM', 1, 1, 'text', 0.5, 0.5, 1.0, 1)",
      { 1LL, 1LL });
  store->Execute (
      "INSERT INTO memory_reconstructions("
      "memory_id, embedding_id, created_at, uncertainty, trigger"
      ") VALUES(?, ?, ?, ?, 'initial')",
      { 1LL, 1LL, 1LL, 0.0 });

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<SetupReconInputsOp> (
          current,
          std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, evidence } }),
      std::make_unique<cortext::operations::ApplyReconsolidation> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (current, 100));
  processor.Flush ();

  auto mem_rows = store->Execute (
      "SELECT embedding_id FROM memories WHERE memory_id = ?",
      { 1LL });
  const long long evidence_embedding_id
      = mem_rows.empty () ? 0LL : AnyToInt64 (mem_rows[0].at ("embedding_id"));

  auto recon_rows = store->Execute (
      "SELECT embedding_id FROM memory_reconstructions "
      "WHERE memory_id = ? ORDER BY reconstruction_id",
      { 1LL });
  const long long latest_embedding_id
      = recon_rows.empty () ? 0LL : AnyToInt64 (recon_rows.back ().at ("embedding_id"));

  const Eigen::VectorXf evidence_after = LoadEmbeddingById (*store, 1LL);
  const Eigen::VectorXf reconstructed_after
      = LoadEmbeddingById (*store, latest_embedding_id);
  const double current_sim
      = cortext::core::CosineSimilarity (reconstructed_after, current);
  const double evidence_sim
      = cortext::core::CosineSimilarity (evidence_after, current);

  std::cout << "constructive_recon_preserves_evidence="
            << (evidence_embedding_id == 1LL ? 1 : 0)
            << " constructive_recon_versions="
            << static_cast<long long> (recon_rows.size ())
            << " constructive_recon_current_sim=" << current_sim
            << " constructive_evidence_current_sim=" << evidence_sim
            << "\n";

  const bool passed = evidence_embedding_id == 1LL && recon_rows.size () == 2
                      && latest_embedding_id > 1LL && current_sim > 0.45
                      && current_sim > evidence_sim;
  return passed ? 0 : 1;
}

} // namespace

int
main ()
{
  int passed = 0;
  const int total = 3;
  passed += (RunInitialSeedStudy () == 0) ? 1 : 0;
  passed += (RunRetrievalAblationStudy () == 0) ? 1 : 0;
  passed += (RunReconsolidationStudy () == 0) ? 1 : 0;
  std::cout << "constructive_recall_bench_passed=" << passed << "/" << total
            << "\n";
  return passed == total ? 0 : 1;
}
