#include "../../src/operations/eviction_ablation.hpp"

#include <cortext/core/knobs.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/extraction.hpp>
#include <cortext/operations/memory_strength.hpp>
#include <cortext/operations/process_extraction_results.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <algorithm>
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
  return (temp_dir / ("cortext_storage_cons_e2e_" + std::to_string (stamp)
                      + ".db"))
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
SeedLongTermMemory (cortext::Store &store, long long embedding_id,
                    long long memory_id, double strength,
                    long long created_at)
{
  std::vector<float> vec (kEmbeddingDim, 0.0f);
  vec[0] = 1.0f;
  store.Execute (
      "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { embedding_id, vec, created_at });
  store.Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, trace_fast, trace_med, trace_slow, trace_ultra, "
      "use_frequency, stability, connectivity, drift_mag, influence, sustained_influence, "
      "contextual_gain, redundancy, pre_activation, lability_state, suppression_count, "
      "created_at, last_access) "
      "VALUES(?, ?, 'bench', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, ?, ?, ?, ?, ?, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, ?, ?)",
      { memory_id, embedding_id, strength, strength, strength * 0.5,
        strength * 0.2, strength * 0.05, created_at, created_at });
}

void
SeedSummaryMemory (cortext::Store &store, long long embedding_id,
                   long long memory_id, const std::string &summary_id,
                   long long start_ts)
{
  std::vector<float> vec (kEmbeddingDim, 0.0f);
  vec[0] = 1.0f;
  store.Execute (
      "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?)",
      { embedding_id, vec, start_ts });
  store.Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, ?, 'ASSOCIATION', ?, 1, 'text', 1.0, 1.0, 1.0, ?)",
      { memory_id, embedding_id, summary_id, start_ts, start_ts });
}

void
AppendPadding (cortext::Store &store, long long bytes)
{
  store.Execute ("INSERT INTO bench_padding(payload) VALUES (zeroblob(?))",
                 { bytes });
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

long long
CountMemoryById (cortext::Store &store, long long id)
{
  auto rows = store.Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?", { id });
  return std::any_cast<long long> (rows[0].at ("cnt"));
}

long long
CountLabels (cortext::Store &store)
{
  auto rows = store.Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE kind = 'LABEL'", {});
  return std::any_cast<long long> (rows[0].at ("cnt"));
}

long long
CountLabelEdges (cortext::Store &store)
{
  auto rows = store.Execute (
      "SELECT COUNT(*) AS cnt FROM associations WHERE edge_type = 'has_label'",
      {});
  return std::any_cast<long long> (rows[0].at ("cnt"));
}

double
GetStrength (cortext::Store &store, long long id)
{
  auto rows = store.Execute (
      "SELECT strength FROM memories WHERE memory_id = ?", { id });
  if (rows.empty ())
    {
      return -1.0;
    }
  return std::any_cast<double> (rows[0].at ("strength"));
}

void
RunExtraction (cortext::Store &store, cortext::ProcessorContext &pctx,
               BenchEncoder &encoder,
               const std::string &summary_id, std::uint64_t signal_ts)
{
  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.encoder = &encoder;

  cortext::operations::ExtractionResult result;
  result.summary_id = summary_id;
  result.labels.push_back ({ "Gabriel", 0.0 });
  result.labels.push_back ({ "Chicago", 0.0 });
  result.labels.push_back ({ "Cortext", 0.0 });
  pctx.pending_extraction_results.push_back (std::move (result));

  cortext::Signal signal;
  signal.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  signal.timestamp = signal_ts;
  signal.source_id = "bench";
  cortext::OperationContext ctx (signal, pctx, cfg, &store);
  cortext::operations::ProcessExtractionResults op;
  auto tx = store.Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();
}

struct ScenarioResult
{
  std::string name;
  long long storage_floor_bytes = 0;
  int consolidation_interval = 0;
  long long eviction_step = -1;
  long long first_label_step = -1;
  long long final_labels = 0;
  long long final_label_edges = 0;
  long long final_memory_alive = 0;
  double final_strength = -1.0;
};

ScenarioResult
RunScenario (const std::string &name, long long storage_floor_bytes,
             int consolidation_interval, long long memory_id)
{
  ScopedTempDb db;
  auto &store = db.store ();
  const long long target_embedding_id = memory_id;
  SeedLongTermMemory (store, target_embedding_id, memory_id, 0.1, 0);

  BenchEncoder encoder;
  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;
  cfg.encoder = &encoder;

  cortext::ProcessorContext pctx;
  pctx.half_life = cortext::core::BaseHalfLifePrior (cfg.stability);
  // Simulate an ongoing chat system in its steady-state regime. The target
  // memory is already in the post-initial-consolidation pool, so the
  // question is whether storage buys enough time for the next label-producing
  // consolidation window.
  pctx.last_consolidation_ts = 1;

  cortext::operations::eviction::EvictionAblationOverride override;
  override.consolidation_gate_enabled = false;
  override.storage_gate_enabled = true;
  override.min_storage_bytes = storage_floor_bytes;
  override.min_storage_fraction_of_available = 0.0;
  cortext::operations::eviction::ScopedEvictionAblationOverride scoped (
      override);

  ScenarioResult result;
  result.name = name;
  result.storage_floor_bytes = storage_floor_bytes;
  result.consolidation_interval = consolidation_interval;

  bool labels_created = false;
  long long summary_embedding_id = 5000;
  long long summary_memory_id = 6000;

  for (int step = 1; step <= kMaxSteps; ++step)
    {
      AppendPadding (store, kGrowthPerStepBytes);

      if (!labels_created && consolidation_interval > 0
          && (step % consolidation_interval) == 0
          && CountMemoryById (store, memory_id) == 1)
        {
          const std::string summary_id = "summary-" + std::to_string (step);
          SeedSummaryMemory (store, summary_embedding_id++, summary_memory_id++,
                             summary_id, step * 1000LL);
          RunExtraction (store, pctx, encoder, summary_id, step * 1000ULL);
          pctx.last_consolidation_ts = static_cast<uint64_t> (step * 1000ULL);
          labels_created = true;
          result.first_label_step = step;
        }

      cortext::Signal signal;
      signal.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
      signal.timestamp = static_cast<uint64_t> (step * 1000ULL);
      signal.source_id = "bench";
      cortext::OperationContext ctx (signal, pctx, cfg, &store);
      ctx.SetMemoryUsageEvents ({ { target_embedding_id, false } });
      auto tx = store.Begin ();
      cortext::operations::UpdateMemoryStrength op;
      op.Execute (ctx, *tx);
      tx->Commit ();

      if (result.eviction_step < 0 && CountMemoryById (store, memory_id) == 0)
        {
          result.eviction_step = step;
        }
    }

  result.final_memory_alive = CountMemoryById (store, memory_id);
  result.final_labels = CountLabels (store);
  result.final_label_edges = CountLabelEdges (store);
  result.final_strength = GetStrength (store, memory_id);
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

  const std::vector<std::pair<std::string, long long>> floors = {
    { "0mb", 0LL },
    { "64mb", 64LL * 1000LL * 1000LL },
    { "128mb", 128LL * 1000LL * 1000LL },
    { "256mb", 256LL * 1000LL * 1000LL },
    { "500mb", 500LL * 1000LL * 1000LL },
    { "1gb", 1000LL * 1000LL * 1000LL },
    { "2gb", 2000LL * 1000LL * 1000LL },
  };
  const std::vector<int> intervals = { 0, 30, 60, 90, 120, 150 };

  std::vector<ScenarioResult> results;
  long long memory_id = 7000;
  for (const auto &[floor_name, floor_bytes] : floors)
    {
      for (int interval : intervals)
        {
          const std::string scenario_name
              = floor_name + "_cons_" + std::to_string (interval);
          auto result = RunScenario (scenario_name, floor_bytes, interval,
                                     memory_id++);
          results.push_back (result);
          std::cout << "storage_cons_" << scenario_name
                    << "_eviction_step=" << result.eviction_step << "\n";
          std::cout << "storage_cons_" << scenario_name
                    << "_first_label_step=" << result.first_label_step << "\n";
          std::cout << "storage_cons_" << scenario_name
                    << "_final_labels=" << result.final_labels << "\n";
          std::cout << "storage_cons_" << scenario_name
                    << "_final_label_edges=" << result.final_label_edges
                    << "\n";
          std::cout << "storage_cons_" << scenario_name
                    << "_final_strength=" << result.final_strength << "\n";
        }
    }

  auto find_result = [&](const std::string &name) -> const ScenarioResult & {
    auto it = std::find_if (
        results.begin (), results.end (),
        [&](const ScenarioResult &r) { return r.name == name; });
    return *it;
  };

  ok &= Check ("storage_cons_no_consolidation_creates_no_labels",
               find_result ("500mb_cons_0").final_labels == 0);
  ok &= Check ("storage_cons_64mb_interval60_captures_labels",
               find_result ("64mb_cons_60").final_labels == 3);
  ok &= Check ("storage_cons_64mb_interval90_misses_labels",
               find_result ("64mb_cons_90").final_labels == 0);
  ok &= Check ("storage_cons_500mb_interval90_captures_labels",
               find_result ("500mb_cons_90").final_labels == 3);
  ok &= Check ("storage_cons_500mb_interval90_delays_vs_64mb",
               find_result ("500mb_cons_90").eviction_step
                   > find_result ("64mb_cons_90").eviction_step);
  ok &= Check ("storage_cons_500mb_interval120_misses_labels",
               find_result ("500mb_cons_120").final_labels == 0);
  ok &= Check ("storage_cons_1gb_interval120_captures_labels",
               find_result ("1gb_cons_120").final_labels == 3);
  ok &= Check ("storage_cons_1gb_interval150_captures_labels",
               find_result ("1gb_cons_150").final_labels == 3);
  ok &= Check ("storage_cons_2gb_interval150_captures_labels",
               find_result ("2gb_cons_150").final_labels == 3);
  ok &= Check ("storage_cons_1gb_interval150_delays_vs_500mb",
               find_result ("1gb_cons_150").eviction_step
                   > find_result ("500mb_cons_150").eviction_step);

  return ok ? 0 : 1;
}
