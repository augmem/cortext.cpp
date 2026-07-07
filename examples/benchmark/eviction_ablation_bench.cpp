#include "../../src/operations/eviction_policy_override.hpp"
#include "include/benchmark_text_encoder.hpp"

#include <cortext/core/knobs.hpp>
#include <cortext/operations/memory_strength.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/signal.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <any>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace
{

using BenchEncoder = cortext::benchmark::BenchmarkTextEncoder;

cortext::Signal
MakeSignal (BenchEncoder &encoder, const std::string &text, std::uint64_t ts)
{
  cortext::Signal s;
  s.embedding = encoder.EncodeTextEigen (text);
  s.timestamp = ts;
  s.source_id = "bench";
  return s;
}

void
SeedMemory (cortext::Store &store, BenchEncoder &encoder, long long id,
            double strength,
            long long created_at, int flashbulb = 0,
            double half_life_bonus = 0.0)
{
  std::vector<float> vec;
  encoder.EncodeText (
      "eviction benchmark memory " + std::to_string (id), vec);
  store.Execute (
      "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { id, vec, created_at });
  store.Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, trace_fast, trace_med, trace_slow, trace_ultra, "
      "use_frequency, stability, connectivity, drift_mag, "
      "influence, sustained_influence, contextual_gain, redundancy, "
      "pre_activation, lability_state, suppression_count, "
      "flashbulb, half_life_bonus, created_at, last_access) "
      "VALUES(?, ?, 'bench', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, "
      "?, ?, ?, ?, ?, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "
      "?, ?, ?, ?)",
      { id, id, created_at, strength, strength, strength * 0.5,
        strength * 0.2, strength * 0.05, flashbulb, half_life_bonus,
        created_at, created_at });
}

long long
CountMemories (cortext::Store &store)
{
  auto rows = store.Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE kind = 'LONG_TERM'", {});
  return std::any_cast<long long> (rows[0].at ("cnt"));
}

long long
CountMemoryById (cortext::Store &store, long long id)
{
  auto rows = store.Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?", { id });
  return std::any_cast<long long> (rows[0].at ("cnt"));
}

double
GetStrength (cortext::Store &store, long long id)
{
  auto rows = store.Execute (
      "SELECT strength FROM memories WHERE memory_id = ?", { id });
  if (rows.empty ())
    return -1.0;
  return std::any_cast<double> (rows[0].at ("strength"));
}

std::vector<double>
GetAllStrengths (cortext::Store &store)
{
  auto rows = store.Execute (
      "SELECT strength FROM memories WHERE kind = 'LONG_TERM' "
      "ORDER BY strength ASC",
      {});
  std::vector<double> result;
  result.reserve (rows.size ());
  for (const auto &row : rows)
    result.push_back (std::any_cast<double> (row.at ("strength")));
  return result;
}

// Create a ProcessorContext with half_life derived from T (like the real
// system does after StabilityFeedback initializes it).
// last_consolidation_ts is set to max so all memories are evictable by default.
// Tests that need the consolidation gate to block eviction should set it to 0.
cortext::ProcessorContext
MakeProcessorContext (double T)
{
  cortext::ProcessorContext pctx;
  pctx.half_life = cortext::core::BaseHalfLifePrior (T);
  pctx.last_consolidation_ts = std::numeric_limits<uint64_t>::max ();
  return pctx;
}

// Run one UpdateMemoryStrength cycle for the given usage events.
// Bypasses SignalProcessor to avoid working-memory persistence interference.
void
RunStrengthUpdate (cortext::Store &store, BenchEncoder & /*encoder*/,
                   const cortext::SignalProcessor::Config &cfg,
                   const std::vector<cortext::OperationContext::MemoryUsageEvent> &events,
                   std::uint64_t signal_ts,
                   cortext::ProcessorContext &pctx)
{
  auto &encoder = static_cast<BenchEncoder &> (*cfg.encoder);
  auto signal = MakeSignal (encoder, "eviction benchmark update pulse", signal_ts);
  cortext::OperationContext ctx (signal, pctx, cfg, &store);
  ctx.SetMemoryUsageEvents (events);

  auto tx = store.Begin ();
  cortext::operations::UpdateMemoryStrength op;
  op.Execute (ctx, *tx);
  tx->Commit ();
}

// --- Configuration types ---

struct EvictionAblationConfig
{
  std::string name;
  double focus = 0.5;
  double sensitivity = 0.5;
  double stability = 0.5;
  bool representative_subset_only = false;
  cortext::operations::eviction::EvictionPolicyOverride override;
};

// --- Metrics ---

struct EvictionAggregateMetrics
{
  std::string name;
  // Retention
  double retention_rate_50 = 0.0;
  double retention_rate_200 = 0.0;
  double retention_rate_500 = 0.0;
  double mean_eviction_step = 0.0;
  // Selectivity
  double used_survival_rate = 0.0;
  double unused_survival_rate = 0.0;
  double selectivity_ratio = 0.0;
  // Flashbulb
  double flashbulb_survival = 0.0;
  double normal_survival = 0.0;
  double flashbulb_advantage = 0.0;
  // Recovery
  double recovery_success_rate = 0.0;
  // Long-horizon
  double steady_state_count = 0.0;
  double eviction_rate_per_hour = 0.0;
  double oldest_surviving_age_hours = 0.0;
  // Distribution
  double strength_mean = 0.0;
  double strength_std = 0.0;
  double strength_min = 0.0;
  double strength_max = 0.0;
  double strength_p25 = 0.0;
  double strength_p50 = 0.0;
  double strength_p75 = 0.0;
  // Timing
  double update_ms_mean = 0.0;
  // Crowding
  double crowding_equilibrium_count = 0.0;
};

double
SafeRatio (double num, double den)
{
  return den <= 0.0 ? 0.0 : num / den;
}

double
Percentile (const std::vector<double> &sorted, double p)
{
  if (sorted.empty ())
    return 0.0;
  const double idx = p * static_cast<double> (sorted.size () - 1);
  const std::size_t lo = static_cast<std::size_t> (std::floor (idx));
  const std::size_t hi = std::min (lo + 1, sorted.size () - 1);
  const double frac = idx - static_cast<double> (lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

// --- Scenario runners ---

// Scenario 1: Pure Decay
void
RunPureDecay (EvictionAggregateMetrics &metrics,
              const EvictionAblationConfig &config, BenchEncoder &encoder)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto &store = *unique_store;
  cortext::store::ApplyMigrations (store);

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &encoder;
  cfg.focus = config.focus;
  cfg.sensitivity = config.sensitivity;
  cfg.stability = config.stability;

  auto pctx = MakeProcessorContext (config.stability);
  constexpr int kCount = 50;
  constexpr int kSteps = 500;
  // 10s per step = 5000s total (~2 half-lives of fast trace at T=0.5)
  constexpr long long kIntervalMs = 60000;

  for (int i = 1; i <= kCount; ++i)
    SeedMemory (store, encoder, i, 1.0, 0);

  std::vector<cortext::OperationContext::MemoryUsageEvent> events;
  events.reserve (kCount);
  for (int i = 1; i <= kCount; ++i)
    events.push_back ({ static_cast<long long> (i), false });

  double eviction_step_sum = 0.0;
  int eviction_count = 0;
  long long prev_count = kCount;

  for (int step = 1; step <= kSteps; ++step)
    {
      const auto ts = static_cast<std::uint64_t> (step * kIntervalMs);
      RunStrengthUpdate (store, encoder, cfg, events, ts, pctx);

      const long long current = CountMemories (store);
      const long long evicted_this_step = prev_count - current;
      for (long long e = 0; e < evicted_this_step; ++e)
        {
          eviction_step_sum += step;
          eviction_count++;
        }
      prev_count = current;

      // Remove events for evicted memories
      events.erase (
          std::remove_if (events.begin (), events.end (),
                          [&store] (const auto &ev) {
                            return CountMemoryById (store, ev.embedding_id) == 0;
                          }),
          events.end ());

      if (step == 50)
        metrics.retention_rate_50 = SafeRatio (current, kCount);
      if (step == 200)
        metrics.retention_rate_200 = SafeRatio (current, kCount);
      if (step == 500)
        metrics.retention_rate_500 = SafeRatio (current, kCount);
    }

  metrics.mean_eviction_step
      = eviction_count > 0 ? eviction_step_sum / eviction_count : kSteps;
}

// Scenario 2: Selective Reinforcement
void
RunSelectiveReinforcement (EvictionAggregateMetrics &metrics,
                           const EvictionAblationConfig &config,
                           BenchEncoder &encoder)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto &store = *unique_store;
  cortext::store::ApplyMigrations (store);

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &encoder;
  cfg.focus = config.focus;
  cfg.sensitivity = config.sensitivity;
  cfg.stability = config.stability;

  auto pctx = MakeProcessorContext (config.stability);
  constexpr int kCount = 50;
  constexpr int kUsed = 10;
  constexpr int kSteps = 300;
  constexpr long long kIntervalMs = 60000;

  for (int i = 1; i <= kCount; ++i)
    SeedMemory (store, encoder, i, 1.0, 0);

  for (int step = 1; step <= kSteps; ++step)
    {
      const auto ts = static_cast<std::uint64_t> (step * kIntervalMs);
      std::vector<cortext::OperationContext::MemoryUsageEvent> events;
      for (int i = 1; i <= kCount; ++i)
        {
          if (CountMemoryById (store, i) == 0)
            continue;
          const bool used = (i <= kUsed) && (step % 20 == 0);
          events.push_back (
              { static_cast<long long> (i), used, used ? 0.5 : std::optional<double>{} });
        }
      if (!events.empty ())
        RunStrengthUpdate (store, encoder, cfg, events, ts, pctx);
    }

  int used_surviving = 0;
  int unused_surviving = 0;
  for (int i = 1; i <= kUsed; ++i)
    {
      if (CountMemoryById (store, i) > 0)
        ++used_surviving;
    }
  for (int i = kUsed + 1; i <= kCount; ++i)
    {
      if (CountMemoryById (store, i) > 0)
        ++unused_surviving;
    }
  metrics.used_survival_rate = SafeRatio (used_surviving, kUsed);
  metrics.unused_survival_rate = SafeRatio (unused_surviving, kCount - kUsed);
  metrics.selectivity_ratio
      = SafeRatio (metrics.used_survival_rate,
                   std::max (metrics.unused_survival_rate, 0.01));
}

// Scenario 3: Flashbulb Protection
void
RunFlashbulbProtection (EvictionAggregateMetrics &metrics,
                        const EvictionAblationConfig &config,
                        BenchEncoder &encoder)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto &store = *unique_store;
  cortext::store::ApplyMigrations (store);

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &encoder;
  cfg.focus = config.focus;
  cfg.sensitivity = config.sensitivity;
  cfg.stability = config.stability;

  auto pctx = MakeProcessorContext (config.stability);
  constexpr int kTotal = 20;
  constexpr int kFlashbulb = 10;
  constexpr int kSteps = 400;
  constexpr long long kIntervalMs = 60000;

  for (int i = 1; i <= kTotal; ++i)
    {
      if (i <= kFlashbulb)
        SeedMemory (store, encoder, i, 1.0, 0, 1, 3.0);
      else
        SeedMemory (store, encoder, i, 1.0, 0, 0, 0.0);
    }

  for (int step = 1; step <= kSteps; ++step)
    {
      const auto ts = static_cast<std::uint64_t> (step * kIntervalMs);
      std::vector<cortext::OperationContext::MemoryUsageEvent> events;
      for (int i = 1; i <= kTotal; ++i)
        {
          if (CountMemoryById (store, i) > 0)
            events.push_back ({ static_cast<long long> (i), false });
        }
      if (!events.empty ())
        RunStrengthUpdate (store, encoder, cfg, events, ts, pctx);
    }

  int flashbulb_surviving = 0;
  int normal_surviving = 0;
  for (int i = 1; i <= kFlashbulb; ++i)
    {
      if (CountMemoryById (store, i) > 0)
        ++flashbulb_surviving;
    }
  for (int i = kFlashbulb + 1; i <= kTotal; ++i)
    {
      if (CountMemoryById (store, i) > 0)
        ++normal_surviving;
    }
  metrics.flashbulb_survival = SafeRatio (flashbulb_surviving, kFlashbulb);
  metrics.normal_survival = SafeRatio (normal_surviving, kTotal - kFlashbulb);
  metrics.flashbulb_advantage
      = metrics.flashbulb_survival - metrics.normal_survival;
}

// Scenario 4: Recovery from Near-Eviction
void
RunRecovery (EvictionAggregateMetrics &metrics,
             const EvictionAblationConfig &config, BenchEncoder &encoder)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto &store = *unique_store;
  cortext::store::ApplyMigrations (store);

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &encoder;
  cfg.focus = config.focus;
  cfg.sensitivity = config.sensitivity;
  cfg.stability = config.stability;

  auto pctx = MakeProcessorContext (config.stability);
  constexpr int kTrials = 20;
  constexpr long long kIntervalMs = 60000;
  int recovered = 0;

  for (int trial = 0; trial < kTrials; ++trial)
    {
      // Reset store each trial
      store.Execute ("DELETE FROM memories", {});
      store.Execute ("DELETE FROM embeddings", {});
      store.Execute ("DELETE FROM memory_evictions", {});

      const long long id = trial + 1;
      SeedMemory (store, encoder, id, 0.5, 0);

      // Decay phase: let it decay significantly (80 steps × 60s = 4800s)
      for (int step = 1; step <= 80; ++step)
        {
          if (CountMemoryById (store, id) == 0)
            break;
          RunStrengthUpdate (
              store, encoder, cfg,
              { { id, false } },
              static_cast<std::uint64_t> (step * kIntervalMs), pctx);
        }

      if (CountMemoryById (store, id) == 0)
        continue; // Already evicted, can't test recovery

      // Reinforce with use (30 steps of active retrieval)
      for (int step = 81; step <= 110; ++step)
        {
          if (CountMemoryById (store, id) == 0)
            break;
          RunStrengthUpdate (
              store, encoder, cfg,
              { { id, true, 0.8 } },
              static_cast<std::uint64_t> (step * kIntervalMs), pctx);
        }

      // Continue decay to see if it survives (90 more steps)
      for (int step = 111; step <= 200; ++step)
        {
          if (CountMemoryById (store, id) == 0)
            break;
          RunStrengthUpdate (
              store, encoder, cfg,
              { { id, false } },
              static_cast<std::uint64_t> (step * kIntervalMs), pctx);
        }

      if (CountMemoryById (store, id) > 0)
        ++recovered;
    }

  metrics.recovery_success_rate = SafeRatio (recovered, kTrials);
}

// Scenario 5: Long-Horizon Simulation
void
RunLongHorizon (EvictionAggregateMetrics &metrics,
                const EvictionAblationConfig &config, BenchEncoder &encoder)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto &store = *unique_store;
  cortext::store::ApplyMigrations (store);

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &encoder;
  cfg.focus = config.focus;
  cfg.sensitivity = config.sensitivity;
  cfg.stability = config.stability;

  auto pctx = MakeProcessorContext (config.stability);
  // Simulate 7 days at 1-minute steps = 10080 steps
  constexpr long long kStepMs = 60000; // 1 minute
  constexpr int kTotalSteps = 10080;
  constexpr int kSeedInterval = 60; // New memory every hour
  constexpr int kUseInterval = 120; // Some memories used every 2 hours

  long long next_id = 1;
  int total_evictions = 0;
  long long oldest_surviving_created = 0;

  for (int step = 0; step < kTotalSteps; ++step)
    {
      const auto ts = static_cast<std::uint64_t> (step * kStepMs);

      // Seed a new memory every hour
      if (step % kSeedInterval == 0)
        {
          SeedMemory (store, encoder, next_id, 1.0, static_cast<long long> (ts));
          ++next_id;
        }

      // Build events for all living memories
      auto all_rows = store.Execute (
          "SELECT memory_id FROM memories WHERE kind = 'LONG_TERM'", {});
      std::vector<cortext::OperationContext::MemoryUsageEvent> events;
      for (const auto &row : all_rows)
        {
          const long long mid
              = std::any_cast<long long> (row.at ("memory_id"));
          // Every other memory gets periodic use
          const bool used = (mid % 5 == 0) && (step % kUseInterval == 0);
          events.push_back ({ mid, used, used ? 0.3 : std::optional<double>{} });
        }

      const long long before = CountMemories (store);
      if (!events.empty ())
        RunStrengthUpdate (store, encoder, cfg, events, ts, pctx);
      const long long after = CountMemories (store);
      total_evictions += static_cast<int> (before - after);
    }

  metrics.steady_state_count = static_cast<double> (CountMemories (store));
  const double total_hours = (kTotalSteps * kStepMs) / 3600000.0;
  metrics.eviction_rate_per_hour = SafeRatio (total_evictions, total_hours);

  // Find oldest surviving memory
  auto oldest = store.Execute (
      "SELECT MIN(created_at) AS oldest FROM memories WHERE kind = 'LONG_TERM'",
      {});
  if (!oldest.empty () && oldest[0].at ("oldest").type () != typeid (std::nullptr_t))
    {
      const long long oldest_ts
          = std::any_cast<long long> (oldest[0].at ("oldest"));
      const long long final_ts
          = static_cast<long long> ((kTotalSteps - 1) * kStepMs);
      metrics.oldest_surviving_age_hours
          = static_cast<double> (final_ts - oldest_ts) / 3600000.0;
    }
}

// Scenario 6: Strength Distribution
void
RunStrengthDistribution (EvictionAggregateMetrics &metrics,
                         const EvictionAblationConfig &config,
                         BenchEncoder &encoder)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto &store = *unique_store;
  cortext::store::ApplyMigrations (store);

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &encoder;
  cfg.focus = config.focus;
  cfg.sensitivity = config.sensitivity;
  cfg.stability = config.stability;

  auto pctx = MakeProcessorContext (config.stability);
  constexpr int kCount = 100;
  constexpr int kSteps = 200;
  constexpr long long kIntervalMs = 60000;

  // Varied initial strengths
  for (int i = 1; i <= kCount; ++i)
    {
      const double init_strength = 0.1 + 0.9 * (static_cast<double> (i) / kCount);
      SeedMemory (store, encoder, i, init_strength, 0);
    }

  for (int step = 1; step <= kSteps; ++step)
    {
      const auto ts = static_cast<std::uint64_t> (step * kIntervalMs);
      std::vector<cortext::OperationContext::MemoryUsageEvent> events;
      for (int i = 1; i <= kCount; ++i)
        {
          if (CountMemoryById (store, i) == 0)
            continue;
          // Top 20% get periodic use
          const bool used = (i > 80) && (step % 10 == 0);
          events.push_back (
              { static_cast<long long> (i), used,
                used ? 0.5 : std::optional<double>{} });
        }
      if (!events.empty ())
        RunStrengthUpdate (store, encoder, cfg, events, ts, pctx);
    }

  auto strengths = GetAllStrengths (store);
  if (!strengths.empty ())
    {
      double sum = 0.0;
      for (double s : strengths)
        sum += s;
      metrics.strength_mean = sum / strengths.size ();

      double var_sum = 0.0;
      for (double s : strengths)
        var_sum += (s - metrics.strength_mean) * (s - metrics.strength_mean);
      metrics.strength_std = std::sqrt (var_sum / strengths.size ());

      metrics.strength_min = strengths.front ();
      metrics.strength_max = strengths.back ();
      metrics.strength_p25 = Percentile (strengths, 0.25);
      metrics.strength_p50 = Percentile (strengths, 0.50);
      metrics.strength_p75 = Percentile (strengths, 0.75);
    }
}

// Scenario 8: Crowding Pressure
void
RunCrowdingPressure (EvictionAggregateMetrics &metrics,
                     const EvictionAblationConfig &config,
                     BenchEncoder &encoder)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto &store = *unique_store;
  cortext::store::ApplyMigrations (store);

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &encoder;
  cfg.focus = config.focus;
  cfg.sensitivity = config.sensitivity;
  cfg.stability = config.stability;

  auto pctx = MakeProcessorContext (config.stability);
  constexpr int kSteps = 600;
  constexpr long long kIntervalMs = 60000;
  long long next_id = 1;

  for (int step = 1; step <= kSteps; ++step)
    {
      const auto ts = static_cast<std::uint64_t> (step * kIntervalMs);

      // Seed a new memory every step
      SeedMemory (store, encoder, next_id, 1.0, static_cast<long long> (ts));
      ++next_id;

      // Update all living memories
      auto rows = store.Execute (
          "SELECT memory_id FROM memories WHERE kind = 'LONG_TERM'", {});
      std::vector<cortext::OperationContext::MemoryUsageEvent> events;
      for (const auto &row : rows)
        {
          const long long mid
              = std::any_cast<long long> (row.at ("memory_id"));
          events.push_back ({ mid, false });
        }
      if (!events.empty ())
        RunStrengthUpdate (store, encoder, cfg, events, ts, pctx);
    }

  metrics.crowding_equilibrium_count
      = static_cast<double> (CountMemories (store));
}

// --- Timing helper ---
double
MeasureUpdateLatency (const EvictionAblationConfig &config,
                      BenchEncoder &encoder)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto &store = *unique_store;
  cortext::store::ApplyMigrations (store);

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &encoder;
  cfg.focus = config.focus;
  cfg.sensitivity = config.sensitivity;
  cfg.stability = config.stability;

  constexpr int kCount = 50;
  for (int i = 1; i <= kCount; ++i)
    SeedMemory (store, encoder, i, 1.0, 0);

  std::vector<cortext::OperationContext::MemoryUsageEvent> events;
  for (int i = 1; i <= kCount; ++i)
    events.push_back ({ static_cast<long long> (i), false });

  auto pctx = MakeProcessorContext (config.stability);
  constexpr int kRuns = 20;
  double total_ms = 0.0;
  for (int run = 0; run < kRuns; ++run)
    {
      const auto start = std::chrono::steady_clock::now ();
      RunStrengthUpdate (store, encoder, cfg, events,
                         static_cast<std::uint64_t> ((run + 1) * 1000), pctx);
      const auto end = std::chrono::steady_clock::now ();
      total_ms += std::chrono::duration<double, std::milli> (end - start).count ();
    }
  return total_ms / kRuns;
}

// --- Evaluate a full config ---
EvictionAggregateMetrics
EvaluateConfig (const EvictionAblationConfig &config, BenchEncoder &encoder)
{
  EvictionAggregateMetrics metrics;
  metrics.name = config.name;

  cortext::operations::eviction::ScopedEvictionPolicyOverride guard (
      config.override);

  RunPureDecay (metrics, config, encoder);
  RunSelectiveReinforcement (metrics, config, encoder);
  RunFlashbulbProtection (metrics, config, encoder);
  RunRecovery (metrics, config, encoder);

  if (!config.representative_subset_only)
    {
      RunLongHorizon (metrics, config, encoder);
      RunStrengthDistribution (metrics, config, encoder);
      RunCrowdingPressure (metrics, config, encoder);
      metrics.update_ms_mean = MeasureUpdateLatency (config, encoder);
    }

  return metrics;
}

// --- Output ---

void
PrintMetrics (const EvictionAggregateMetrics &m)
{
  std::cout << std::fixed << std::setprecision (4);
  std::cout << "config=" << m.name << "\n";
  std::cout << "retention_rate_50=" << m.retention_rate_50 << "\n";
  std::cout << "retention_rate_200=" << m.retention_rate_200 << "\n";
  std::cout << "retention_rate_500=" << m.retention_rate_500 << "\n";
  std::cout << "mean_eviction_step=" << m.mean_eviction_step << "\n";
  std::cout << "used_survival_rate=" << m.used_survival_rate << "\n";
  std::cout << "unused_survival_rate=" << m.unused_survival_rate << "\n";
  std::cout << "selectivity_ratio=" << m.selectivity_ratio << "\n";
  std::cout << "flashbulb_survival=" << m.flashbulb_survival << "\n";
  std::cout << "normal_survival=" << m.normal_survival << "\n";
  std::cout << "flashbulb_advantage=" << m.flashbulb_advantage << "\n";
  std::cout << "recovery_success_rate=" << m.recovery_success_rate << "\n";
  std::cout << "steady_state_count=" << m.steady_state_count << "\n";
  std::cout << "eviction_rate_per_hour=" << m.eviction_rate_per_hour << "\n";
  std::cout << "oldest_surviving_age_hours="
            << m.oldest_surviving_age_hours << "\n";
  std::cout << "strength_mean=" << m.strength_mean << "\n";
  std::cout << "strength_std=" << m.strength_std << "\n";
  std::cout << "strength_min=" << m.strength_min << "\n";
  std::cout << "strength_max=" << m.strength_max << "\n";
  std::cout << "strength_p25=" << m.strength_p25 << "\n";
  std::cout << "strength_p50=" << m.strength_p50 << "\n";
  std::cout << "strength_p75=" << m.strength_p75 << "\n";
  std::cout << "crowding_equilibrium_count="
            << m.crowding_equilibrium_count << "\n";
  std::cout << "update_ms_mean=" << m.update_ms_mean << "\n";
  std::cout << "\n";
}

void
PrintDelta (const std::string &label, double baseline, double variant)
{
  std::cout << std::fixed << std::setprecision (4);
  std::cout << label << "_delta=" << (variant - baseline) << "\n";
}

} // namespace

int
main ()
{
  BenchEncoder encoder;
  std::cout << "encoder_backend=" << encoder.backend_name () << "\n";
  std::cout << "encoder_model_path=" << encoder.resolved_model_path ().string ()
            << "\n";

  using namespace cortext::operations::eviction;

  // --- Build ablation configs ---
  std::vector<EvictionAblationConfig> configs;

  // Default
  configs.push_back ({ "default", 0.5, 0.5, 0.5, false, {} });

  // Trace count ablations
  configs.push_back (
      { "single_trace", 0.5, 0.5, 0.5, false, { 1 } });
  configs.push_back (
      { "two_traces", 0.5, 0.5, 0.5, false, { 2 } });
  configs.push_back (
      { "four_traces", 0.5, 0.5, 0.5, false, { 4 } });

  // Coupling ablations
  {
    EvictionPolicyOverride o;
    o.coupling_enabled = false;
    configs.push_back ({ "coupling_off", 0.5, 0.5, 0.5, false, o });
  }
  {
    EvictionPolicyOverride o;
    o.coupling_strength = 0.25;
    configs.push_back ({ "coupling_strong", 0.5, 0.5, 0.5, false, o });
  }

  // Reinforcement ablations
  {
    EvictionPolicyOverride o;
    o.reinforcement = ReinforcementStrength::Off;
    configs.push_back ({ "boost_off", 0.5, 0.5, 0.5, false, o });
  }
  {
    EvictionPolicyOverride o;
    o.reinforcement = ReinforcementStrength::Weak;
    configs.push_back ({ "boost_weak", 0.5, 0.5, 0.5, false, o });
  }
  {
    EvictionPolicyOverride o;
    o.reinforcement = ReinforcementStrength::Strong;
    configs.push_back ({ "boost_strong", 0.5, 0.5, 0.5, false, o });
  }

  // Cutoff ablations
  {
    EvictionPolicyOverride o;
    o.periphery_cutoff = 0.03;
    configs.push_back ({ "cutoff_low", 0.5, 0.5, 0.5, false, o });
  }
  {
    EvictionPolicyOverride o;
    o.periphery_cutoff = 0.20;
    configs.push_back ({ "cutoff_high", 0.5, 0.5, 0.5, false, o });
  }

  // Half-life ablations
  {
    EvictionPolicyOverride o;
    o.half_life = 60.0;
    configs.push_back ({ "halflife_short", 0.5, 0.5, 0.5, false, o });
  }
  {
    EvictionPolicyOverride o;
    o.half_life = 7200.0;
    configs.push_back ({ "halflife_long", 0.5, 0.5, 0.5, false, o });
  }

  // Flashbulb ablation
  {
    EvictionPolicyOverride o;
    o.flashbulb_enabled = false;
    configs.push_back ({ "flashbulb_off", 0.5, 0.5, 0.5, false, o });
  }

  // Weight distribution ablation
  {
    EvictionPolicyOverride o;
    o.weights = WeightDistribution::Equal;
    configs.push_back ({ "equal_weights", 0.5, 0.5, 0.5, false, o });
  }

  // Consolidation gate ablation
  {
    EvictionPolicyOverride o;
    o.consolidation_gate_enabled = false;
    configs.push_back ({ "consolidation_gate_off", 0.5, 0.5, 0.5, false, o });
  }

  // F/S/T sweep (representative subset)
  const double fst_values[] = { 0.2, 0.5, 0.9 };
  for (double f : fst_values)
    {
      for (double s : fst_values)
        {
          for (double t : fst_values)
            {
              std::ostringstream name;
              name << "fst_f" << std::fixed << std::setprecision (1)
                   << f << "_s" << s << "_t" << t;
              configs.push_back ({ name.str (), f, s, t, true, {} });
            }
        }
    }

  // --- Evaluate all configs ---
  std::vector<EvictionAggregateMetrics> results;
  results.reserve (configs.size ());

  for (const auto &config : configs)
    {
      std::cerr << "Evaluating: " << config.name << "\n";
      results.push_back (EvaluateConfig (config, encoder));
    }

  // --- Print results ---
  for (const auto &r : results)
    PrintMetrics (r);

  // --- Comparison deltas vs default ---
  const auto &baseline = results[0];
  std::cout << "--- deltas vs default ---\n";
  for (std::size_t i = 1; i < results.size (); ++i)
    {
      const auto &r = results[i];
      std::cout << "config=" << r.name << "\n";
      PrintDelta ("selectivity_ratio", baseline.selectivity_ratio,
                  r.selectivity_ratio);
      PrintDelta ("retention_rate_200", baseline.retention_rate_200,
                  r.retention_rate_200);
      PrintDelta ("flashbulb_advantage", baseline.flashbulb_advantage,
                  r.flashbulb_advantage);
      PrintDelta ("recovery_success_rate", baseline.recovery_success_rate,
                  r.recovery_success_rate);
      std::cout << "\n";
    }

  // --- Gate checks ---
  bool gates_pass = true;

  if (baseline.selectivity_ratio < 2.0)
    {
      std::cerr << "GATE FAIL: selectivity_ratio "
                << baseline.selectivity_ratio << " < 2.0\n";
      gates_pass = false;
    }
  if (baseline.flashbulb_advantage <= 0.0)
    {
      std::cerr << "GATE FAIL: flashbulb_advantage "
                << baseline.flashbulb_advantage << " <= 0.0\n";
      gates_pass = false;
    }
  if (baseline.recovery_success_rate < 0.5)
    {
      std::cerr << "GATE FAIL: recovery_success_rate "
                << baseline.recovery_success_rate << " < 0.5\n";
      gates_pass = false;
    }

  // Default should outperform coupling_off on mean eviction step
  // (coupling extends memory lifetime via cascade to slower traces)
  const auto coupling_off_it = std::find_if (
      results.begin (), results.end (),
      [] (const auto &r) { return r.name == "coupling_off"; });
  if (coupling_off_it != results.end ()
      && baseline.mean_eviction_step <= coupling_off_it->mean_eviction_step)
    {
      std::cerr << "GATE FAIL: default mean_eviction_step ("
                << baseline.mean_eviction_step
                << ") <= coupling_off ("
                << coupling_off_it->mean_eviction_step << ")\n";
      gates_pass = false;
    }

  // Default should outperform single_trace on mean eviction step
  const auto single_trace_it = std::find_if (
      results.begin (), results.end (),
      [] (const auto &r) { return r.name == "single_trace"; });
  if (single_trace_it != results.end ()
      && baseline.mean_eviction_step <= single_trace_it->mean_eviction_step)
    {
      std::cerr << "GATE FAIL: default mean_eviction_step ("
                << baseline.mean_eviction_step
                << ") <= single_trace ("
                << single_trace_it->mean_eviction_step << ")\n";
      gates_pass = false;
    }

  // consolidation_gate_off should behave like default (both have
  // last_consolidation_ts=max so memories are evictable). The gate only
  // matters when last_consolidation_ts=0 — tested in memory_system_ablation.
  const auto consolidation_gate_off_it = std::find_if (
      results.begin (), results.end (),
      [] (const auto &r) { return r.name == "consolidation_gate_off"; });
  if (consolidation_gate_off_it != results.end ()
      && std::abs (consolidation_gate_off_it->mean_eviction_step
                   - baseline.mean_eviction_step)
             > 50.0)
    {
      std::cerr << "GATE FAIL: consolidation_gate_off mean_eviction_step ("
                << consolidation_gate_off_it->mean_eviction_step
                << ") diverges from default ("
                << baseline.mean_eviction_step << ") by >50\n";
      gates_pass = false;
    }

  std::cout << "gates_pass=" << (gates_pass ? "true" : "false") << "\n";
  return gates_pass ? 0 : 1;
}
