#include "../../src/operations/eviction_policy_override.hpp"

#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/memory_strength.hpp>
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
constexpr int kMaxHours = 72;

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
  return (temp_dir / ("cortext_storage_human_sleep_token_"
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
SeedLongTermMemory (cortext::Store &store, long long id, long long created_at)
{
  std::vector<float> vec (kEmbeddingDim, 0.0f);
  vec[0] = 1.0f;
  store.Execute (
      "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { id, vec, created_at });
  store.Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, trace_fast, trace_med, trace_slow, trace_ultra, "
      "use_frequency, stability, connectivity, drift_mag, influence, sustained_influence, "
      "contextual_gain, redundancy, pre_activation, lability_state, suppression_count, "
      "created_at, last_access) "
      "VALUES(?, ?, 'bench', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, ?, ?, ?, ?, ?, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, ?, ?)",
      { id, id, created_at, 0.1, 0.1, 0.05, 0.02, 0.005, created_at,
        created_at });
}

long long
CountMemoryById (cortext::Store &store, long long id)
{
  auto rows = store.Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?", { id });
  return std::any_cast<long long> (rows[0].at ("cnt"));
}

std::int64_t
EstimateTokenCountFromChars (std::size_t chars)
{
  if (chars == 0)
    return 0;
  return std::max<std::int64_t> (
      1, static_cast<std::int64_t> (
             std::ceil (static_cast<double> (chars) / 4.0)));
}

std::string
MakeTraceText (int hour)
{
  return "Hour " + std::to_string (hour)
         + ": Gabriel discussed Cortext, Chicago, caregiver routines, and a "
           "memory-support task that should survive until sleep-driven "
           "consolidation.";
}

struct SleepProfile
{
  const char *name;
  double naps_min;
  double naps_max;
};

struct TraceState
{
  long long memory_id = 0;
  std::string text;
  bool consolidated = false;
  bool lost = false;
};

struct ScenarioResult
{
  std::string profile_name;
  long long storage_floor_bytes = 0;
  int consolidation_every_hours = 0;
  int consolidation_events = 0;
  int traces_consolidated = 0;
  int traces_lost = 0;
  long long estimated_prompt_tokens = 0;
  long long estimated_completion_tokens = 0;
  long long estimated_total_tokens = 0;
};

ScenarioResult
RunScenario (const SleepProfile &profile, long long storage_floor_bytes,
             long long base_memory_id)
{
  ScopedTempDb db;
  auto &store = db.store ();

  BenchEncoder encoder;
  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;
  cfg.encoder = &encoder;

  cortext::ProcessorContext pctx;
  pctx.half_life = kHumanHalfLifeSeconds;
  pctx.last_consolidation_ts = 1;

  cortext::operations::eviction::EvictionPolicyOverride override;
  override.consolidation_gate_enabled = false;
  override.storage_gate_enabled = true;
  override.min_storage_bytes = storage_floor_bytes;
  override.min_storage_fraction_of_available = 0.0;
  override.half_life = kHumanHalfLifeSeconds;

  const double average_naps = 0.5 * (profile.naps_min + profile.naps_max);
  const double consolidation_bouts_per_day = 1.0 + average_naps;

  ScenarioResult result;
  result.profile_name = profile.name;
  result.storage_floor_bytes = storage_floor_bytes;
  result.consolidation_every_hours = std::max (
      1, static_cast<int> (std::llround (24.0 / consolidation_bouts_per_day)));

  long long synthetic_used_bytes = 0;
  std::vector<TraceState> traces;

  for (int hour = 1; hour <= kMaxHours; ++hour)
    {
      TraceState trace;
      trace.memory_id = base_memory_id + hour;
      trace.text = MakeTraceText (hour);
      SeedLongTermMemory (store, trace.memory_id, hour * kHourMs);
      traces.push_back (trace);

      synthetic_used_bytes += kGrowthPerHourBytes;
      override.used_storage_bytes = synthetic_used_bytes;
      cortext::operations::eviction::ScopedEvictionPolicyOverride scoped (
          override);

      cortext::Signal signal;
      signal.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
      signal.timestamp = static_cast<uint64_t> (hour * kHourMs);
      signal.source_id = "bench";
      cortext::OperationContext ctx (signal, pctx, cfg, &store);
      ctx.SetMemoryUsageEvents ({});
      auto tx = store.Begin ();
      cortext::operations::UpdateMemoryStrength op;
      op.Execute (ctx, *tx);
      tx->Commit ();

      for (auto &candidate : traces)
        {
          if (!candidate.consolidated && !candidate.lost
              && CountMemoryById (store, candidate.memory_id) == 0)
            {
              candidate.lost = true;
            }
        }

      if ((hour % result.consolidation_every_hours) == 0)
        {
          std::vector<TraceState *> batch;
          for (auto &candidate : traces)
            {
              if (!candidate.consolidated && !candidate.lost
                  && CountMemoryById (store, candidate.memory_id) == 1)
                {
                  batch.push_back (&candidate);
                }
            }

          if (!batch.empty ())
            {
              std::size_t source_chars = 0;
              for (const auto *candidate : batch)
                {
                  source_chars += candidate->text.size () + 24;
                }

              const std::size_t summary_prompt_chars = 256 + source_chars;
              const std::size_t summary_completion_chars = 180;
              const std::size_t extraction_prompt_chars = 192 + 180;
              const std::size_t extraction_completion_chars = 96;
              const long long prompt_tokens
                  = EstimateTokenCountFromChars (summary_prompt_chars)
                    + EstimateTokenCountFromChars (extraction_prompt_chars);
              const long long completion_tokens
                  = EstimateTokenCountFromChars (summary_completion_chars)
                    + EstimateTokenCountFromChars (
                        extraction_completion_chars);

              result.estimated_prompt_tokens += prompt_tokens;
              result.estimated_completion_tokens += completion_tokens;
              result.estimated_total_tokens += prompt_tokens + completion_tokens;
              result.consolidation_events += 1;
              result.traces_consolidated
                  += static_cast<int> (batch.size ());

              for (auto *candidate : batch)
                {
                  candidate->consolidated = true;
                }
            }
        }
    }

  for (const auto &candidate : traces)
    {
      if (candidate.lost)
        {
          result.traces_lost += 1;
        }
    }
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
    { "newborn", 4.0, 6.0 },
    { "infant", 2.0, 3.0 },
    { "toddler", 1.0, 2.0 },
    { "preschool", 0.0, 1.0 },
    { "school_age", 0.0, 0.0 },
    { "teen", 0.0, 0.0 },
    { "adult", 0.0, 0.0 },
    { "older_adult", 0.0, 0.0 },
  };

  const std::vector<std::pair<std::string, long long>> floors = {
    { "0mb", 0LL },
    { "256mb", 256LL * 1000LL * 1000LL },
    { "500mb", 500LL * 1000LL * 1000LL },
    { "1gb", 1000LL * 1000LL * 1000LL },
  };

  std::vector<ScenarioResult> results;
  long long base_memory_id = 50000;
  for (const auto &profile : profiles)
    {
      for (const auto &[floor_name, floor_bytes] : floors)
        {
          auto result = RunScenario (profile, floor_bytes, base_memory_id);
          base_memory_id += 1000;
          results.push_back (result);
          std::cout << "human_sleep_tokens_" << profile.name << "_"
                    << floor_name << "_consolidation_events="
                    << result.consolidation_events << "\n";
          std::cout << "human_sleep_tokens_" << profile.name << "_"
                    << floor_name << "_traces_consolidated="
                    << result.traces_consolidated << "\n";
          std::cout << "human_sleep_tokens_" << profile.name << "_"
                    << floor_name << "_traces_lost=" << result.traces_lost
                    << "\n";
          std::cout << "human_sleep_tokens_" << profile.name << "_"
                    << floor_name << "_prompt_tokens="
                    << result.estimated_prompt_tokens << "\n";
          std::cout << "human_sleep_tokens_" << profile.name << "_"
                    << floor_name << "_completion_tokens="
                    << result.estimated_completion_tokens << "\n";
          std::cout << "human_sleep_tokens_" << profile.name << "_"
                    << floor_name << "_total_tokens="
                    << result.estimated_total_tokens << "\n";
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

  ok &= Check (
      "human_sleep_tokens_newborn_cost_exceeds_adult_500mb",
      find_result ("newborn", 500LL * 1000LL * 1000LL).estimated_total_tokens
          > find_result ("adult", 500LL * 1000LL * 1000LL)
                .estimated_total_tokens);
  ok &= Check (
      "human_sleep_tokens_newborn_events_exceed_adult_500mb",
      find_result ("newborn", 500LL * 1000LL * 1000LL).consolidation_events
          > find_result ("adult", 500LL * 1000LL * 1000LL)
                .consolidation_events);
  ok &= Check ("human_sleep_tokens_adult_500mb_cost_matches_0mb",
               find_result ("adult", 500LL * 1000LL * 1000LL)
                       .estimated_total_tokens
                   == find_result ("adult", 0LL)
                          .estimated_total_tokens);
  ok &= Check ("human_sleep_tokens_adult_1gb_matches_500mb_retention",
               find_result ("adult", 1000LL * 1000LL * 1000LL)
                       .traces_consolidated
                   == find_result ("adult", 500LL * 1000LL * 1000LL)
                          .traces_consolidated);
  ok &= Check ("human_sleep_tokens_adult_1gb_matches_500mb_cost",
               find_result ("adult", 1000LL * 1000LL * 1000LL)
                       .estimated_total_tokens
                   == find_result ("adult", 500LL * 1000LL * 1000LL)
                          .estimated_total_tokens);

  return ok ? 0 : 1;
}
