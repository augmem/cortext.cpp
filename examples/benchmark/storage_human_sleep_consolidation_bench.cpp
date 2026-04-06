#include "../../src/operations/eviction_ablation.hpp"

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
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

namespace
{
constexpr int kEmbeddingDim = 256;
constexpr long long kHourMs = 60LL * 60LL * 1000LL;
constexpr long long kGrowthPerHourBytes = 12LL * 1000LL * 1000LL;
constexpr double kHumanHalfLifeSeconds = 24.0 * 60.0 * 60.0;
constexpr int kMaxHours = 240;

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
  return (temp_dir / ("cortext_storage_human_sleep_" + std::to_string (stamp)
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

private:
  std::string path_;
  std::unique_ptr<cortext::Store> store_;
};

void
SeedLongTermMemory (cortext::Store &store, long long id)
{
  std::vector<float> vec (kEmbeddingDim, 0.0f);
  vec[0] = 1.0f;
  store.Execute (
      "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { id, vec, 0LL });
  store.Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, trace_fast, trace_med, trace_slow, trace_ultra, "
      "use_frequency, stability, connectivity, drift_mag, influence, sustained_influence, "
      "contextual_gain, redundancy, pre_activation, lability_state, suppression_count, "
      "created_at, last_access) "
      "VALUES(?, ?, 'bench', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, ?, ?, ?, ?, ?, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)",
      { id, id, 0.1, 0.1, 0.05, 0.02, 0.005 });
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

struct SleepProfile
{
  const char *name;
  double total_sleep_min_hours;
  double total_sleep_max_hours;
  double naps_min;
  double naps_max;
};

struct ScenarioResult
{
  std::string name;
  std::string profile_name;
  long long storage_floor_bytes = 0;
  int consolidation_every_hours = 0;
  long long eviction_hour = -1;
  long long first_label_hour = -1;
  long long final_labels = 0;
};

ScenarioResult
RunScenario (const SleepProfile &profile, long long storage_floor_bytes,
             long long memory_id)
{
  ScopedTempDb db;
  auto &store = db.store ();
  SeedLongTermMemory (store, memory_id);

  BenchEncoder encoder;
  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;
  cfg.encoder = &encoder;

  cortext::ProcessorContext pctx;
  pctx.half_life = kHumanHalfLifeSeconds;
  pctx.last_consolidation_ts = 1;

  cortext::operations::eviction::EvictionAblationOverride override;
  override.consolidation_gate_enabled = false;
  override.storage_gate_enabled = true;
  override.min_storage_bytes = storage_floor_bytes;
  override.min_storage_fraction_of_available = 0.0;
  override.half_life = kHumanHalfLifeSeconds;

  ScenarioResult result;
  result.name = std::string (profile.name) + "_"
                + std::to_string (storage_floor_bytes);
  result.profile_name = profile.name;
  result.storage_floor_bytes = storage_floor_bytes;
  const double average_naps = 0.5 * (profile.naps_min + profile.naps_max);
  const double consolidation_bouts_per_day = 1.0 + average_naps;
  result.consolidation_every_hours = std::max (
      1, static_cast<int> (std::llround (24.0 / consolidation_bouts_per_day)));

  bool labels_created = false;
  long long summary_embedding_id = memory_id + 100000;
  long long summary_memory_id = memory_id + 200000;
  long long synthetic_used_bytes = 0;

  for (int hour = 1; hour <= kMaxHours; ++hour)
    {
      synthetic_used_bytes += kGrowthPerHourBytes;
      override.used_storage_bytes = synthetic_used_bytes;
      cortext::operations::eviction::ScopedEvictionAblationOverride scoped (
          override);

      if (!labels_created
          && (hour % result.consolidation_every_hours) == 0
          && CountMemoryById (store, memory_id) == 1)
        {
          const std::string summary_id
              = std::string ("summary-") + profile.name + "-"
                + std::to_string (hour);
          SeedSummaryMemory (store, summary_embedding_id++, summary_memory_id++,
                             summary_id, hour * kHourMs);
          RunExtraction (store, pctx, encoder, summary_id, hour * kHourMs);
          labels_created = true;
          result.first_label_hour = hour;
        }

      cortext::Signal signal;
      signal.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
      signal.timestamp = static_cast<uint64_t> (hour * kHourMs);
      signal.source_id = "bench";
      cortext::OperationContext ctx (signal, pctx, cfg, &store);
      ctx.SetMemoryUsageEvents ({ { memory_id, false } });
      auto tx = store.Begin ();
      cortext::operations::UpdateMemoryStrength op;
      op.Execute (ctx, *tx);
      tx->Commit ();

      if (result.eviction_hour < 0 && CountMemoryById (store, memory_id) == 0)
        {
          result.eviction_hour = hour;
        }
    }

  result.final_labels = CountLabels (store);
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

  const std::vector<SleepProfile> profiles = {
    { "newborn", 14.0, 17.0, 4.0, 6.0 },
    { "infant", 12.0, 16.0, 2.0, 3.0 },
    { "toddler", 11.0, 14.0, 1.0, 2.0 },
    { "preschool", 10.0, 13.0, 0.0, 1.0 },
    { "school_age", 9.0, 12.0, 0.0, 0.0 },
    { "teen", 8.0, 10.0, 0.0, 0.0 },
    { "adult", 7.0, 9.0, 0.0, 0.0 },
    { "older_adult", 7.0, 8.0, 0.0, 0.0 },
  };

  const std::vector<std::pair<std::string, long long>> floors = {
    { "0mb", 0LL },
    { "64mb", 64LL * 1000LL * 1000LL },
    { "256mb", 256LL * 1000LL * 1000LL },
    { "500mb", 500LL * 1000LL * 1000LL },
    { "1gb", 1000LL * 1000LL * 1000LL },
    { "2gb", 2000LL * 1000LL * 1000LL },
  };

  std::vector<ScenarioResult> results;
  long long memory_id = 9000;
  for (const auto &profile : profiles)
    {
      for (const auto &[floor_name, floor_bytes] : floors)
        {
          auto result = RunScenario (profile, floor_bytes, memory_id++);
          results.push_back (result);
          std::cout << "human_sleep_" << profile.name << "_" << floor_name
                    << "_eviction_hour=" << result.eviction_hour << "\n";
          std::cout << "human_sleep_" << profile.name << "_" << floor_name
                    << "_label_hour=" << result.first_label_hour << "\n";
          std::cout << "human_sleep_" << profile.name << "_" << floor_name
                    << "_labels=" << result.final_labels << "\n";
        }
    }

  auto find_result = [&](const std::string &profile_name,
                         long long floor_bytes) -> const ScenarioResult & {
    auto it = std::find_if (
        results.begin (), results.end (),
        [&](const ScenarioResult &r) {
          return r.profile_name == profile_name
                 && r.storage_floor_bytes == floor_bytes;
        });
    return *it;
  };

  ok &= Check ("human_sleep_newborn_zero_budget_captures",
               find_result ("newborn", 0LL).final_labels == 3);
  ok &= Check ("human_sleep_adult_zero_budget_misses",
               find_result ("adult", 0LL).final_labels == 0);
  ok &= Check ("human_sleep_adult_256mb_misses",
               find_result ("adult", 256LL * 1000LL * 1000LL).final_labels
                   == 0);
  ok &= Check ("human_sleep_adult_500mb_captures",
               find_result ("adult", 500LL * 1000LL * 1000LL).final_labels
                   == 3);
  ok &= Check ("human_sleep_adult_1gb_matches_500mb",
               find_result ("adult", 1000LL * 1000LL * 1000LL).final_labels
                   == 3);
  ok &= Check ("human_sleep_older_adult_500mb_captures",
               find_result ("older_adult",
                            500LL * 1000LL * 1000LL).final_labels
                   == 3);

  return ok ? 0 : 1;
}
