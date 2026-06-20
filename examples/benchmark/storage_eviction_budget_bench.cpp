#include "../../src/operations/eviction_policy_override.hpp"

#include <cortext/core/knobs.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/memory_strength.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <any>
#include <cstdint>
#include <iostream>
#include <memory>
#include <limits>
#include <vector>

namespace
{
constexpr int kEmbeddingDim = 256;

class BenchEncoder : public cortext::Encoder
{
public:
  void EncodeText (const std::string &, std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }
  void EncodeAudio (const float *, std::size_t, std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }
  void EncodeImage (const std::uint8_t *, int, int, int, std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }
};

void
SeedMemory (cortext::Store &store, long long id, double strength)
{
  std::vector<float> vec (kEmbeddingDim, 0.0f);
  vec[0] = 1.0f;
  store.Execute (
      "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) VALUES(?, ?, ?)",
      { id, vec, 1LL });
  store.Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, trace_fast, trace_med, trace_slow, trace_ultra, "
      "use_frequency, stability, connectivity, drift_mag, influence, sustained_influence, "
      "contextual_gain, redundancy, pre_activation, lability_state, suppression_count, "
      "created_at) "
      "VALUES(?, ?, 'bench', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, ?, ?, ?, ?, ?, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)",
      { id, id, strength, strength, strength * 0.5, strength * 0.2, strength * 0.05 });
}

long long
CountMemory (cortext::Store &store, long long id)
{
  auto rows = store.Execute ("SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?", { id });
  return std::any_cast<long long> (rows[0].at ("cnt"));
}

void
RunDecay (cortext::Store &store,
          const cortext::SignalProcessor::Config &cfg,
          cortext::ProcessorContext &pctx,
          long long id)
{
  for (int i = 0; i < 240; ++i)
    {
      cortext::Signal signal;
      signal.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
      signal.timestamp = static_cast<uint64_t> ((i + 1) * 1000);
      signal.source_id = "bench";
      cortext::OperationContext ctx (signal, pctx, cfg, &store);
      ctx.SetMemoryUsageEvents ({ { id, false } });
      auto tx = store.Begin ();
      cortext::operations::UpdateMemoryStrength op;
      op.Execute (ctx, *tx);
      tx->Commit ();
    }
}

bool
Check (const char *name, bool condition)
{
  std::cout << name << "=" << (condition ? 1 : 0) << "\n";
  return condition;
}

} // namespace

int main ()
{
  bool ok = true;

  BenchEncoder encoder;
  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;
  cfg.encoder = &encoder;

  {
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
    cortext::store::ApplyMigrations (*store);
    SeedMemory (*store, 100LL, 0.1);
    cortext::ProcessorContext pctx;
    pctx.half_life = cortext::core::BaseHalfLifePrior (cfg.stability);
    pctx.last_consolidation_ts = std::numeric_limits<uint64_t>::max ();

    cortext::operations::eviction::EvictionPolicyOverride override;
    override.consolidation_gate_enabled = false;
    override.storage_gate_enabled = true;
    override.min_storage_bytes = 1LL << 30;
    cortext::operations::eviction::ScopedEvictionPolicyOverride scoped (override);
    RunDecay (*store, cfg, pctx, 100LL);
    std::cout << "storage_gate_high_budget_count=" << CountMemory (*store, 100LL) << "\n";
    ok &= Check ("storage_gate_blocks_early_eviction", CountMemory (*store, 100LL) == 1);
  }

  {
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
    cortext::store::ApplyMigrations (*store);
    SeedMemory (*store, 101LL, 0.1);
    cortext::ProcessorContext pctx;
    pctx.half_life = cortext::core::BaseHalfLifePrior (cfg.stability);
    pctx.last_consolidation_ts = std::numeric_limits<uint64_t>::max ();

    cortext::operations::eviction::EvictionPolicyOverride override;
    override.consolidation_gate_enabled = false;
    override.storage_gate_enabled = true;
    override.min_storage_bytes = 0;
    cortext::operations::eviction::ScopedEvictionPolicyOverride scoped (override);
    RunDecay (*store, cfg, pctx, 101LL);
    std::cout << "storage_gate_zero_budget_count=" << CountMemory (*store, 101LL) << "\n";
    ok &= Check ("storage_gate_allows_eviction_after_limit", CountMemory (*store, 101LL) == 0);
  }

  return ok ? 0 : 1;
}
