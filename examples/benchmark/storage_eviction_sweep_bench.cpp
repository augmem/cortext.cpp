#include "../../src/operations/eviction_ablation.hpp"

#include <cortext/core/knobs.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/memory_strength.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <any>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

namespace
{
constexpr int kEmbeddingDim = 256;
constexpr long long kGrowthPerStepBytes = 5LL * 1000LL * 1000LL;
constexpr int kMaxSteps = 450;

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

std::string
MakeTempDbPath ()
{
  const auto temp_dir = std::filesystem::temp_directory_path ();
  const auto stamp = std::chrono::high_resolution_clock::now ()
                         .time_since_epoch ()
                         .count ();
  return (temp_dir / ("cortext_storage_eviction_sweep_"
                      + std::to_string (stamp) + ".db"))
      .string ();
}

void
CleanupTempDb (const std::string &path)
{
  std::error_code ec;
  std::filesystem::remove (path, ec);
  std::filesystem::remove (path + "-wal", ec);
  std::filesystem::remove (path + "-shm", ec);
}

class ScopedTempDb
{
public:
  ScopedTempDb () : path_ (MakeTempDbPath ())
  {
    store_ = cortext::SQLiteStore::Create (path_.c_str ());
    cortext::store::ApplyMigrations (*store_);
    store_->Execute ("PRAGMA journal_mode = DELETE", {});
    store_->Execute (
        "CREATE TABLE IF NOT EXISTS bench_padding("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  payload BLOB NOT NULL"
        ")",
        {});
  }

  ~ScopedTempDb ()
  {
    store_.reset ();
    CleanupTempDb (path_);
  }

  cortext::Store &
  store ()
  {
    return *store_;
  }

  const std::string &
  path () const
  {
    return path_;
  }

private:
  std::string path_;
  std::unique_ptr<cortext::Store> store_;
};

void
SeedMemory (cortext::Store &store, long long id, double strength)
{
  std::vector<float> vec (kEmbeddingDim, 0.0f);
  vec[0] = 1.0f;
  store.Execute (
      "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { id, vec, 1LL });
  store.Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, trace_fast, trace_med, trace_slow, trace_ultra, "
      "use_frequency, stability, connectivity, drift_mag, influence, sustained_influence, "
      "contextual_gain, redundancy, pre_activation, lability_state, suppression_count, "
      "created_at) "
      "VALUES(?, ?, 'bench', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, ?, ?, ?, ?, ?, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)",
      { id, id, strength, strength, strength * 0.5, strength * 0.2,
        strength * 0.05 });
}

long long
CountMemory (cortext::Store &store, long long id)
{
  auto rows = store.Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?", { id });
  return std::any_cast<long long> (rows[0].at ("cnt"));
}

long long
CurrentDbBytes (const std::string &path)
{
  std::error_code ec;
  long long total = 0;
  const auto main_size = std::filesystem::file_size (path, ec);
  if (!ec)
    total += static_cast<long long> (main_size);
  const auto wal_size = std::filesystem::file_size (path + "-wal", ec);
  if (!ec)
    total += static_cast<long long> (wal_size);
  return total;
}

void
AppendPadding (cortext::Store &store, long long bytes)
{
  store.Execute ("INSERT INTO bench_padding(payload) VALUES (zeroblob(?))",
                 { bytes });
}

struct SweepResult
{
  std::string name;
  long long threshold_bytes = 0;
  long long crossing_step = -1;
  long long eviction_step = -1;
  long long crossing_bytes = 0;
  long long eviction_bytes = 0;
  long long final_bytes = 0;
};

SweepResult
RunScenario (const std::string &name,
             const cortext::SignalProcessor::Config &cfg,
             long long threshold_bytes,
             long long memory_id)
{
  ScopedTempDb db;
  auto &store = db.store ();
  SeedMemory (store, memory_id, 0.1);

  cortext::ProcessorContext pctx;
  pctx.half_life = cortext::core::BaseHalfLifePrior (cfg.stability);
  pctx.last_consolidation_ts = std::numeric_limits<uint64_t>::max ();

  cortext::operations::eviction::EvictionAblationOverride override;
  override.consolidation_gate_enabled = false;
  override.storage_gate_enabled = true;
  override.min_storage_bytes = threshold_bytes;
  override.min_storage_fraction_of_available = 0.0;
  cortext::operations::eviction::ScopedEvictionAblationOverride scoped (
      override);

  SweepResult result;
  result.name = name;
  result.threshold_bytes = threshold_bytes;

  for (int step = 1; step <= kMaxSteps; ++step)
    {
      AppendPadding (store, kGrowthPerStepBytes);
      const long long used_bytes = CurrentDbBytes (db.path ());
      if (result.crossing_step < 0 && used_bytes >= threshold_bytes)
        {
          result.crossing_step = step;
          result.crossing_bytes = used_bytes;
        }

      cortext::Signal signal;
      signal.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
      signal.timestamp = static_cast<uint64_t> (step * 1000);
      signal.source_id = "bench";
      cortext::OperationContext ctx (signal, pctx, cfg, &store);
      ctx.SetMemoryUsageEvents ({ { memory_id, false } });
      auto tx = store.Begin ();
      cortext::operations::UpdateMemoryStrength op;
      op.Execute (ctx, *tx);
      tx->Commit ();

      if (CountMemory (store, memory_id) == 0)
        {
          result.eviction_step = step;
          result.eviction_bytes = CurrentDbBytes (db.path ());
          break;
        }
    }

  result.final_bytes = CurrentDbBytes (db.path ());
  return result;
}

bool
Check (const char *name, bool condition)
{
  std::cout << name << "=" << (condition ? 1 : 0) << "\n";
  return condition;
}

} // namespace

int
main ()
{
  bool ok = true;

  BenchEncoder encoder;
  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;
  cfg.encoder = &encoder;

  const std::vector<std::pair<std::string, long long>> scenarios = {
    { "0mb", 0LL },
    { "64mb", 64LL * 1000LL * 1000LL },
    { "128mb", 128LL * 1000LL * 1000LL },
    { "256mb", 256LL * 1000LL * 1000LL },
    { "500mb", 500LL * 1000LL * 1000LL },
    { "1gb", 1000LL * 1000LL * 1000LL },
    { "2gb", 2000LL * 1000LL * 1000LL },
  };

  std::vector<SweepResult> results;
  results.reserve (scenarios.size ());
  long long memory_id = 200LL;
  for (const auto &[name, threshold] : scenarios)
    {
      auto result = RunScenario (name, cfg, threshold, memory_id++);
      results.push_back (result);
      std::cout << "storage_sweep_" << name << "_cross_step="
                << result.crossing_step << "\n";
      std::cout << "storage_sweep_" << name << "_eviction_step="
                << result.eviction_step << "\n";
      std::cout << "storage_sweep_" << name << "_cross_bytes="
                << result.crossing_bytes << "\n";
      std::cout << "storage_sweep_" << name << "_eviction_bytes="
                << result.eviction_bytes << "\n";
    }

  for (std::size_t i = 1; i < results.size (); ++i)
    {
      ok &= Check (
          ("storage_sweep_monotonic_" + results[i - 1].name + "_to_"
           + results[i].name)
              .c_str (),
          results[i].eviction_step >= results[i - 1].eviction_step);
    }

  ok &= Check ("storage_sweep_default_500mb_delays_vs_64mb",
               results[4].eviction_step > results[1].eviction_step);
  ok &= Check ("storage_sweep_1gb_delays_vs_500mb",
               results[5].eviction_step > results[4].eviction_step);
  ok &= Check ("storage_sweep_2gb_delays_vs_1gb",
               results[6].eviction_step > results[5].eviction_step);
  ok &= Check ("storage_sweep_zero_budget_is_fastest",
               results.front ().eviction_step <= results[1].eviction_step);

  return ok ? 0 : 1;
}
