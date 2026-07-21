#include "cortext/operations/memory_strength.hpp"
#include "historical_surface_search_cache_internal.hpp"
#include "retrieval_trace_state.hpp"
#include "emotional_metadata_cache_internal.hpp"
#include "execution_cache_sidecar_internal.hpp"
#include "association_fanout_cache_internal.hpp"
#include <algorithm>
#include <limits>
#include <cstdint>
#include <chrono>

#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "eviction_policy_override.hpp"
#include "eviction_policy.hpp"
#include "rif_state_internal.hpp"
#include <cmath>
#include <functional>
#include <optional>
#include <set>
#include <typeinfo>
#include <unordered_set>
#include <vector>

namespace cortext::operations
{

namespace
{
using SteadyClock = std::chrono::steady_clock;
constexpr std::size_t kEvictionSqlChunkSize = 400;
constexpr std::size_t kEvictionSelectLimit = 512;
constexpr int kEvictionScanIntervalSignals = 128;
constexpr int kEvictionPressureScanIntervalSignals = 16;

double
ElapsedMillis (SteadyClock::time_point start)
{
  return std::chrono::duration<double, std::milli> (SteadyClock::now () - start)
      .count ();
}

std::vector<std::any>
MakeParams (const std::vector<long long> &values, std::size_t begin,
            std::size_t end)
{
  std::vector<std::any> params;
  params.reserve (end - begin);
  for (std::size_t i = begin; i < end; ++i)
    {
      params.push_back (values[i]);
    }
  return params;
}

void
AppendParams (std::vector<std::any> &params,
              const std::vector<long long> &values, std::size_t begin,
              std::size_t end)
{
  for (std::size_t i = begin; i < end; ++i)
    {
      params.push_back (values[i]);
    }
}

void
ForEachChunk (std::size_t count, const std::function<void (std::size_t,
                                                           std::size_t)> &fn)
{
  for (std::size_t begin = 0; begin < count; begin += kEvictionSqlChunkSize)
    {
      const std::size_t end
          = std::min (count, begin + kEvictionSqlChunkSize);
      fn (begin, end);
    }
}

std::vector<std::vector<unsigned char> >
SelectEvictionBlobIds (Transaction &tx,
                       const std::vector<long long> &memory_ids)
{
  std::vector<std::vector<unsigned char> > blob_ids;
  std::set<std::vector<unsigned char> > seen_blob_ids;
  ForEachChunk (memory_ids.size (), [&] (std::size_t begin,
                                         std::size_t end) {
    const std::string placeholders
        = eviction_policy::MakePlaceholders (end - begin);
    std::vector<std::any> params;
    params.reserve ((end - begin) * 3);
    AppendParams (params, memory_ids, begin, end);
    AppendParams (params, memory_ids, begin, end);
    AppendParams (params, memory_ids, begin, end);
    auto rows = tx.Execute (
        "SELECT blob_id FROM memories "
        "WHERE memory_id IN (" + placeholders + ") "
        "  AND blob_id IS NOT NULL "
        "UNION "
        "SELECT blob_id FROM signals "
        "WHERE memory_id IN (" + placeholders + ") "
        "  AND blob_id IS NOT NULL "
        "UNION "
        "SELECT blob_id FROM memory_reconstructions "
        "WHERE memory_id IN (" + placeholders + ") "
        "  AND blob_id IS NOT NULL",
        params);
    for (const auto &row : rows)
      {
        const auto it = row.find ("blob_id");
        if (it == row.end ())
          {
            continue;
          }
        auto blob_id = store::BlobFromAny (it->second);
        if (!blob_id.empty () && seen_blob_ids.insert (blob_id).second)
          {
            blob_ids.push_back (std::move (blob_id));
          }
      }
  });
  return blob_ids;
}

bool
ObjstoreBlobStillReferenced (Transaction &tx,
                             const std::vector<unsigned char> &blob_id)
{
  auto rows = tx.Execute (
      "SELECT ("
      "  EXISTS(SELECT 1 FROM memories WHERE blob_id = ?) "
      "  OR EXISTS(SELECT 1 FROM signals WHERE blob_id = ?) "
      "  OR EXISTS(SELECT 1 FROM memory_reconstructions WHERE blob_id = ?)"
      ") AS referenced",
      { blob_id, blob_id, blob_id });
  if (rows.empty () || rows[0].count ("referenced") == 0)
    {
      return false;
    }
  return store::AnyToLongLong (rows[0].at ("referenced")).value_or (0) != 0;
}

long long
DeleteOrphanedObjstoreBlobs (
    Transaction &tx, const std::vector<std::vector<unsigned char> > &blob_ids)
{
  long long deleted = 0;
  for (const auto &blob_id : blob_ids)
    {
      if (ObjstoreBlobStillReferenced (tx, blob_id))
        {
          continue;
        }
      auto rows = tx.Execute ("SELECT objstore_delete(?1) AS deleted",
                              { blob_id });
      if (!rows.empty () && rows[0].count ("deleted") != 0
          && store::AnyToLongLong (rows[0].at ("deleted")).value_or (0) != 0)
        {
          ++deleted;
        }
    }
  return deleted;
}

const std::any *
FindValue (const std::map<std::string, std::any> &row, const char *key)
{
  auto it = row.find (key);
  return it == row.end () ? nullptr : &it->second;
}

bool
IsNull (const std::any &value)
{
  return !value.has_value () || value.type () == typeid (std::nullptr_t);
}

double
GetDouble (const std::map<std::string, std::any> &row, const char *key,
           double fallback)
{
  const std::any *value = FindValue (row, key);
  if (!value || IsNull (*value))
    {
      return fallback;
    }
  if (value->type () == typeid (double))
    return std::any_cast<double> (*value);
  if (value->type () == typeid (float))
    return static_cast<double> (std::any_cast<float> (*value));
  if (value->type () == typeid (int))
    return static_cast<double> (std::any_cast<int> (*value));
  if (value->type () == typeid (long))
    return static_cast<double> (std::any_cast<long> (*value));
  if (value->type () == typeid (long long))
    return static_cast<double> (std::any_cast<long long> (*value));
  return fallback;
}

long long
GetInt64 (const std::map<std::string, std::any> &row, const char *key,
          long long fallback)
{
  const std::any *value = FindValue (row, key);
  if (!value || IsNull (*value))
    {
      return fallback;
    }
  if (value->type () == typeid (long long))
    return std::any_cast<long long> (*value);
  if (value->type () == typeid (long))
    return static_cast<long long> (std::any_cast<long> (*value));
  if (value->type () == typeid (int))
    return static_cast<long long> (std::any_cast<int> (*value));
  if (value->type () == typeid (double))
    return static_cast<long long> (std::any_cast<double> (*value));
  return fallback;
}

std::optional<long long>
GetInt64Maybe (const std::map<std::string, std::any> &row, const char *key)
{
  const std::any *value = FindValue (row, key);
  if (!value || IsNull (*value))
    {
      return std::nullopt;
    }
  if (value->type () == typeid (long long))
    return std::any_cast<long long> (*value);
  if (value->type () == typeid (long))
    return static_cast<long long> (std::any_cast<long> (*value));
  if (value->type () == typeid (int))
    return static_cast<long long> (std::any_cast<int> (*value));
  if (value->type () == typeid (double))
    return static_cast<long long> (std::any_cast<double> (*value));
  return std::nullopt;
}

} // namespace

void
UpdateMemoryStrength::Execute (OperationContext &context, Transaction &tx) const
{
  if (context.GetSignal ().retention == Retention::Ephemeral)
    {
      return;
    }
  const auto &cfg = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();

  const auto eviction_override = eviction::GetEvictionPolicyOverride ();

  const double F_raw = cfg.focus;
  const double S_raw = cfg.sensitivity;
  const double T = cfg.stability;
  const double F_eff = core::FocusBias (F_raw);
  const double S_eff = core::SensitivityBias (S_raw);
  const double alpha = core::AlphaS (S_raw, p_ctx.u_t);
  const double half_life
      = eviction_override.half_life.has_value ()
            ? *eviction_override.half_life
            : ((p_ctx.half_life > constants::kNormEpsilon)
                   ? p_ctx.half_life
                   : core::BaseHalfLifePrior (T));
  const double L_cg = std::round (
      core::MemoryContextualGainWindow (F_raw, S_raw, T));
  const double alpha_cg
      = (L_cg > 0.0) ? (constants::kTwo / (L_cg + 1.0)) : 1.0;
  const double serial_mult
      = std::max (0.0,
                  context.GetSerialPositionMultiplier ().value_or (1.0));

  int64_t update_count = 0;

  const auto &events = context.GetMemoryUsageEvents ();
  for (const auto &e : events)
    {
      const double used_flag = e.used ? 1.0 : 0.0;
      const double cg_event = core::Clamp (
          e.contextual_gain.value_or (constants::kNormalizedMin), -1.0, 1.0);
      const long long id = static_cast<long long> (e.embedding_id);
      const long long memory_id = static_cast<long long> (e.memory_id);
      const bool has_memory_id = memory_id > 0;
      const char *lookup_column = has_memory_id ? "memory_id" : "embedding_id";
      const long long lookup_id = has_memory_id ? memory_id : id;
      const long long ts
          = static_cast<long long> (context.GetSignal ().timestamp);

      const std::string select_sql
          = std::string (
                "SELECT m.memory_id, rif.strength AS strength, "
                "       m.use_frequency, m.contextual_gain, "
                "       m.retrieved_count, m.used_count, m.last_access, "
                "       m.created_at, m.flashbulb, m.half_life_bonus, "
                "       m.trace_fast, m.trace_med, m.trace_slow, "
                "       m.trace_ultra "
                "FROM memories m "
                "JOIN rif_effective_memories rif "
                "  ON rif.memory_id = m.memory_id WHERE m.")
            + lookup_column + " = ?";
      const auto select_start = SteadyClock::now ();
      auto rows = tx.Execute (select_sql, { lookup_id });
      context.AddOperationTiming ("MemoryStrength.feedback_select_sql",
                                  ElapsedMillis (select_start));
      if (rows.empty ())
        {
          continue;
        }

      const double strength_prev
          = GetDouble (rows[0], "strength", 1.0);
      const double use_freq_prev
          = GetDouble (rows[0], "use_frequency", 0.0);
      const double contextual_gain_prev
          = GetDouble (rows[0], "contextual_gain", 0.0);
      const long long row_memory_id
          = GetInt64 (rows[0], "memory_id", memory_id);
      const long long retrieved_prev
          = GetInt64 (rows[0], "retrieved_count", 0LL);
      const long long used_prev
          = GetInt64 (rows[0], "used_count", 0LL);
      const std::optional<long long> last_access_prev
          = GetInt64Maybe (rows[0], "last_access");
      const long long created_at
          = GetInt64 (rows[0], "created_at", ts);
      const int flashbulb
          = GetInt64 (rows[0], "flashbulb", 0LL) != 0 ? 1 : 0;
      const double half_life_bonus_raw
          = GetDouble (rows[0], "half_life_bonus", 0.0);
      const auto trace_fallback = core::MemoryStoredTraceFallbackPolicy (
          F_raw, S_raw, T);
      const double trace_fast_prev
          = GetDouble (rows[0], "trace_fast",
                       strength_prev * trace_fallback.fast);
      const double trace_med_prev
          = GetDouble (rows[0], "trace_med",
                       strength_prev * trace_fallback.medium);
      const double trace_slow_prev
          = GetDouble (rows[0], "trace_slow",
                       strength_prev * trace_fallback.slow);
      const double trace_ultra_prev
          = GetDouble (rows[0], "trace_ultra",
                       strength_prev * trace_fallback.ultra);

      const long long access_base
          = last_access_prev.value_or (created_at);
      const double delta_seconds = std::max (
          0.0, static_cast<double> (ts - access_base) / 1000.0);
      const bool flashbulb_active
          = eviction_override.flashbulb_enabled.value_or (true)
            && (flashbulb != 0);
      const double half_life_bonus
          = flashbulb_active
                ? std::max (1.0, half_life_bonus_raw)
                : 1.0;
      const double memory_half_life
          = std::max (half_life * half_life_bonus, constants::kNormEpsilon);
      const double use_frequency
          = core::Clamp (core::Ewma (use_freq_prev, used_flag, alpha),
                         constants::kNormalizedMin,
                         constants::kNormalizedMax);
      const double cg_sample = e.used ? cg_event : 0.0;
      const double contextual_gain
          = core::Clamp (core::Ewma (contextual_gain_prev, cg_sample, alpha_cg),
                         -1.0, 1.0);

      const long long retrieved_new = retrieved_prev + 1;
      const long long used_new = used_prev + static_cast<long long> (used_flag);
      const double influence_factor
          = (retrieved_new > 0)
                ? (static_cast<double> (used_new)
                   / static_cast<double> (std::max (retrieved_new, 1LL)))
                      * contextual_gain
                : 0.0;

      const int n_traces
          = eviction_override.trace_count.has_value ()
                ? core::Clamp (*eviction_override.trace_count, 1, 4)
                : core::MemoryTraceCount (F_raw, S_raw, T);
      const auto tau_policy = core::MemoryTraceTauPolicy (F_raw, S_raw, T);
      const double tau_fast = tau_policy.fast * memory_half_life;
      const double tau_med = tau_policy.medium * memory_half_life;
      const double tau_slow = tau_policy.slow * memory_half_life;
      const double tau_ultra = tau_policy.ultra * memory_half_life;
      const double taus[4] = { tau_fast, tau_med, tau_slow, tau_ultra };
      double traces[4] = { trace_fast_prev, trace_med_prev,
                           trace_slow_prev, trace_ultra_prev };

      // Reinforcement: S and F drive a per-trace injection that is
      // distributed uniformly across active traces (not front-loaded).
      // The alpha EWMA increment handles gradual learning; the
      // reinforcement injection handles immediate retrieval-use feedback.
      const double reinf_scale
          = eviction_override.reinforcement.has_value ()
                ? (*eviction_override.reinforcement
                           == eviction::ReinforcementStrength::Off
                       ? 0.0
                       : (*eviction_override.reinforcement
                                  == eviction::ReinforcementStrength::Weak
                              ? 0.5
                              : 1.0))
                : 1.0;
      const double reinforcement
          = core::Clamp (
                reinf_scale
                    * (S_eff * used_flag
                       + F_eff
                             * core::Clamp (influence_factor,
                                            constants::kNormalizedMin,
                                            constants::kNormalizedMax))
                    * serial_mult,
                constants::kNormalizedMin, constants::kNormalizedMax);
      const double reinf_per_trace
          = reinforcement / std::max (1, n_traces);

      for (int i = 0; i < 4; ++i)
        {
          const double tau = std::max (taus[i], constants::kNormEpsilon);
          const double decay_i
              = std::exp (-std::log (constants::kTwo) / tau
                          * std::max (0.0, delta_seconds));
          const double increment = (alpha * used_flag) / std::max (1, n_traces);
          traces[i] = core::Clamp (
              traces[i] * decay_i + increment + reinf_per_trace,
              constants::kNormalizedMin, constants::kNormalizedMax);
        }

      const double coupling
          = eviction_override.coupling_enabled.has_value ()
                    && !*eviction_override.coupling_enabled
                ? 0.0
                : eviction_override.coupling_strength.value_or (
                    core::MemoryTraceCoupling (F_raw, S_raw, T));
      traces[1] = core::Clamp (traces[1] + coupling * traces[0],
                               constants::kNormalizedMin,
                               constants::kNormalizedMax);
      traces[2] = core::Clamp (traces[2] + coupling * traces[1],
                               constants::kNormalizedMin,
                               constants::kNormalizedMax);
      traces[3] = core::Clamp (traces[3] + coupling * traces[2],
                               constants::kNormalizedMin,
                               constants::kNormalizedMax);

      const bool equal_weights
          = eviction_override.weights.has_value ()
            && *eviction_override.weights
                   == eviction::WeightDistribution::Equal;
      double w_raw[4];
      if (equal_weights)
        {
          w_raw[0] = 0.25;
          w_raw[1] = 0.25;
          w_raw[2] = 0.25;
          w_raw[3] = 0.25;
        }
      else
        {
          const auto weights = core::MemoryTraceWeightPolicy (F_raw, S_raw, T);
          w_raw[0] = weights.fast;
          w_raw[1] = weights.medium;
          w_raw[2] = weights.slow;
          w_raw[3] = weights.ultra;
        }
      double w_sum = 0.0;
      for (int i = 0; i < 4; ++i)
        {
          if (i < n_traces)
            w_sum += w_raw[i];
        }
      if (w_sum <= constants::kNormEpsilon)
        w_sum = 1.0;
      double strength = 0.0;
      for (int i = 0; i < 4; ++i)
        {
          if (i < n_traces)
            {
              strength += (w_raw[i] / w_sum) * traces[i];
            }
        }
      strength = core::Clamp (strength, constants::kNormalizedMin,
                              constants::kNormalizedMax);

      const std::string update_sql
          = std::string (
                "UPDATE memories "
                "SET retrieved_count = ?, "
                "    used_count = ?, "
                "    last_used = CASE WHEN ? > 0 THEN ? ELSE last_used END, "
                "    contextual_gain = ?, "
                "    influence_factor = ?, "
                "    use_frequency = ?, "
                "    strength = ?, "
                "    trace_fast = ?, "
                "    trace_med = ?, "
                "    trace_slow = ?, "
                "    trace_ultra = ?, "
                "    last_access = ? "
                "WHERE ")
            + lookup_column + " = ?";
      const auto update_start = SteadyClock::now ();
      tx.Execute (
          update_sql,
          { retrieved_new, used_new, used_flag, ts, contextual_gain,
            influence_factor, use_frequency, strength, traces[0], traces[1],
            traces[2], traces[3], ts, lookup_id });
      rif_state_internal::RefreshActiveStrengthWhere (
          tx, std::string (lookup_column) + " = ?", { lookup_id });
      if (row_memory_id > 0)
        {
          p_ctx.UpdateRetrievalSurfaceUsageByMemory (row_memory_id,
                                                     retrieved_new, used_new, ts);
        }
      else
        {
          p_ctx.UpdateRetrievalSurfaceUsageByEmbedding (id, retrieved_new,
                                                        used_new, ts);
        }
      context.AddOperationTiming ("MemoryStrength.feedback_update_sql",
                                  ElapsedMillis (update_start));

      ++update_count;
    }

  // Evict weak memories below periphery cutoff (v2: memories table).
  // Also delete corresponding signals, graph edges, and embeddings.
  const long long evicted_at
      = static_cast<long long> (context.GetSignal ().timestamp);
  const auto frontier_start = SteadyClock::now ();
  const auto frontier = eviction_policy::ResolveEvictionFrontier (
      p_ctx, tx, T, static_cast<long long> (std::min<std::uint64_t> (
          p_ctx.last_consolidation_ts,
          static_cast<std::uint64_t> (
              std::numeric_limits<long long>::max ()))),
      eviction_override);
  context.AddOperationTiming ("MemoryStrength.eviction_frontier",
                              ElapsedMillis (frontier_start));

  // Consolidation gate: no memory is evicted until it has existed through
  // at least one consolidation cycle (created_at < last_consolidation_ts).
  // When last_consolidation_ts == 0, no memories are evictable.
  std::string eviction_where
      = eviction_policy::EvictionWhereClause (frontier);
  const std::vector<std::any> delete_params
      = eviction_policy::EvictionWhereParams (frontier);

  if (!frontier.storage_allows_eviction)
    {
      telemetry::LogDebug (
          "cortext.memory_strength",
          { telemetry::Attribute::Int64 ("update_count", update_count),
            telemetry::Attribute::Int64 ("eviction_count", 0),
            telemetry::Attribute::Int64 ("storage_used_bytes",
                                         frontier.storage_used_bytes),
            telemetry::Attribute::Int64 ("storage_threshold_bytes",
                                         frontier.storage_threshold_bytes) });
      return;
    }

  if (frontier.consolidation_gate_active && frontier.consolidation_ts <= 0)
    {
      context.AddOperationTiming ("MemoryStrength.eviction_scan_deferred",
                                  0.0);
      telemetry::LogDebug (
          "cortext.memory_strength",
          { telemetry::Attribute::Int64 ("update_count", update_count),
            telemetry::Attribute::Int64 ("eviction_count", 0),
            telemetry::Attribute::Int64 ("storage_used_bytes",
                                         frontier.storage_used_bytes),
            telemetry::Attribute::Int64 ("storage_threshold_bytes",
                                         frontier.storage_threshold_bytes) });
      return;
    }

  const bool consolidation_gate_disabled
      = !frontier.consolidation_gate_active;
  const bool storage_pressure_active
      = frontier.storage_gate_active && frontier.storage_allows_eviction;
  const bool consolidation_advanced
      = p_ctx.last_eviction_scan_consolidation_ts
        != p_ctx.last_consolidation_ts;
  const bool first_scan = p_ctx.last_eviction_scan_signal < 0;
  const int scan_interval = storage_pressure_active
                                ? kEvictionPressureScanIntervalSignals
                                : kEvictionScanIntervalSignals;
  const bool periodic_scan_due
      = (p_ctx.signals_processed - p_ctx.last_eviction_scan_signal)
        >= scan_interval;
  const bool periodic_scan
      = periodic_scan_due
        && (consolidation_gate_disabled || storage_pressure_active);
  if (!consolidation_gate_disabled && !consolidation_advanced && !first_scan
      && !periodic_scan)
    {
      context.AddOperationTiming ("MemoryStrength.eviction_scan_deferred",
                                  0.0);
      telemetry::LogDebug (
          "cortext.memory_strength",
          { telemetry::Attribute::Int64 ("update_count", update_count),
            telemetry::Attribute::Int64 ("eviction_count", 0),
            telemetry::Attribute::Int64 ("storage_used_bytes",
                                         frontier.storage_used_bytes),
            telemetry::Attribute::Int64 ("storage_threshold_bytes",
                                         frontier.storage_threshold_bytes) });
      return;
    }

  p_ctx.last_eviction_scan_signal = p_ctx.signals_processed;
  p_ctx.last_eviction_scan_consolidation_ts = p_ctx.last_consolidation_ts;

  const auto eviction_select_start = SteadyClock::now ();
  std::vector<std::any> select_params = delete_params;
  select_params.push_back (static_cast<long long> (kEvictionSelectLimit));
  auto evictable_rows = tx.Execute (
      "SELECT m.memory_id, m.embedding_id "
      "FROM memories m INDEXED BY idx_memories_ltm_strength_created"
      " JOIN rif_effective_memories rif ON rif.memory_id = m.memory_id"
          " " + eviction_where
          + " ORDER BY rif.strength ASC, m.created_at ASC LIMIT ?",
      select_params);
  context.AddOperationTiming ("MemoryStrength.eviction_select_ids_sql",
                              ElapsedMillis (eviction_select_start));

  std::vector<long long> evictable_memory_ids;
  std::vector<long long> evictable_embedding_ids;
  std::unordered_set<long long> seen_memory_ids;
  std::unordered_set<long long> seen_embedding_ids;
  evictable_memory_ids.reserve (evictable_rows.size ());
  evictable_embedding_ids.reserve (evictable_rows.size ());
  for (const auto &row : evictable_rows)
    {
      const auto memory_it = row.find ("memory_id");
      if (memory_it == row.end ())
        {
          continue;
        }
      const auto memory_id = store::AnyToLongLong (memory_it->second);
      if (!memory_id.has_value () || !seen_memory_ids.insert (*memory_id).second)
        {
          continue;
        }
      evictable_memory_ids.push_back (*memory_id);

      const auto embedding_it = row.find ("embedding_id");
      if (embedding_it == row.end ())
        {
          continue;
        }
      const auto embedding_id = store::AnyToLongLong (embedding_it->second);
      if (embedding_id.has_value ()
          && seen_embedding_ids.insert (*embedding_id).second)
        {
          evictable_embedding_ids.push_back (*embedding_id);
        }
    }

  if (evictable_memory_ids.empty ())
    {
      telemetry::LogDebug (
          "cortext.memory_strength",
          { telemetry::Attribute::Int64 ("update_count", update_count),
            telemetry::Attribute::Int64 ("eviction_count", 0),
            telemetry::Attribute::Int64 ("storage_used_bytes",
                                         frontier.storage_used_bytes),
            telemetry::Attribute::Int64 ("storage_threshold_bytes",
                                         frontier.storage_threshold_bytes) });
      return;
    }

  for (const long long memory_id : evictable_memory_ids)
    {
      p_ctx.RemoveRetrievalSurface (memory_id);
      association_fanout_cache::NotifyRetrievalSurfaceChanged (p_ctx,
                                                               memory_id);
      retrieval_trace::RecordSurfaceRemove (memory_id);
      historical_surface_search_cache_internal::RemoveCurrent (p_ctx,
                                                                memory_id);
    }
  // The embedding deletion phase can preserve rows still referenced by other
  // owners. Drop the private accelerator rather than guess at post-transaction
  // membership; the exact SQL path remains authoritative after eviction.
  historical_surface_search_cache_internal::
      InvalidateHistoricalPreserveCurrent (p_ctx);

  const auto reconstruction_embedding_select_start = SteadyClock::now ();
  ForEachChunk (evictable_memory_ids.size (),
                [&] (std::size_t begin, std::size_t end) {
                  const std::string placeholders
                      = eviction_policy::MakePlaceholders (end - begin);
                  auto params = MakeParams (evictable_memory_ids, begin, end);
                  auto rows = tx.Execute (
                      "SELECT embedding_id FROM memory_reconstructions "
                      "WHERE memory_id IN (" + placeholders + ") "
                      "UNION "
                      "SELECT embedding_id FROM current_memory_embeddings "
                      "WHERE memory_id IN (" + placeholders + ") "
                      "UNION "
                      "SELECT embedding_id FROM signals "
                      "WHERE memory_id IN (" + placeholders + ")",
                      [&] {
                        std::vector<std::any> p = params;
                        AppendParams (p, evictable_memory_ids, begin, end);
                        AppendParams (p, evictable_memory_ids, begin, end);
                        return p;
                      } ());
                  for (const auto &row : rows)
                    {
                      auto it = row.find ("embedding_id");
                      if (it == row.end ())
                        {
                          continue;
                        }
                      const auto embedding_id = store::AnyToLongLong (
                          it->second);
                      if (embedding_id.has_value ()
                          && seen_embedding_ids.insert (*embedding_id).second)
                        {
                          evictable_embedding_ids.push_back (*embedding_id);
                        }
                    }
                });
  context.AddOperationTiming (
      "MemoryStrength.eviction_select_owned_embeddings_sql",
      ElapsedMillis (reconstruction_embedding_select_start));

  const auto objstore_blob_select_start = SteadyClock::now ();
  const auto evictable_blob_ids
      = SelectEvictionBlobIds (tx, evictable_memory_ids);
  context.AddOperationTiming (
      "MemoryStrength.eviction_select_objstore_blobs_sql",
      ElapsedMillis (objstore_blob_select_start));

  const auto eviction_insert_start = SteadyClock::now ();
  ForEachChunk (evictable_memory_ids.size (),
                [&] (std::size_t begin, std::size_t end) {
                  const std::string placeholders
                      = eviction_policy::MakePlaceholders (end - begin);
                  std::vector<std::any> params = { evicted_at };
                  AppendParams (params, evictable_memory_ids, begin, end);
                  tx.Execute (
                      "INSERT INTO memory_evictions ("
                      "  memory_id, embedding_id, source_id, kind, label, "
                      "  start_ts, end_ts, created_at, last_access, strength, "
                      "  use_frequency, contextual_gain, retrieved_count, "
                      "  used_count, n_signals, modality, eviction_reason, "
                      "  evicted_at"
                      ") "
                      "SELECT m.memory_id, m.embedding_id, m.source_id, "
                      "       m.kind, COALESCE(m.label, ''), m.start_ts, "
                      "       m.end_ts, m.created_at, m.last_access, "
                      "       rif.strength, m.use_frequency, "
                      "       m.contextual_gain, m.retrieved_count, "
                      "       m.used_count, m.n_signals, m.modality, "
                      "       'periphery_cutoff', ? "
                      "FROM memories m "
                      "JOIN rif_effective_memories rif "
                      "  ON rif.memory_id = m.memory_id "
                      "WHERE m.memory_id IN (" + placeholders + ")",
                      params);
                });
  context.AddOperationTiming ("MemoryStrength.eviction_insert_sql",
                              ElapsedMillis (eviction_insert_start));

  const auto assoc_delete_start = SteadyClock::now ();
  ForEachChunk (evictable_memory_ids.size (),
                [&] (std::size_t begin, std::size_t end) {
                  const std::string placeholders
                      = eviction_policy::MakePlaceholders (end - begin);
                  auto params
                      = MakeParams (evictable_memory_ids, begin, end);
                  AppendParams (params, evictable_memory_ids, begin, end);
                  tx.Execute (
                      "DELETE FROM associations "
                      "WHERE source_memory_id IN (" + placeholders + ") "
                      "   OR target_memory_id IN (" + placeholders + ")",
                      params);
                });
  context.AddOperationTiming ("MemoryStrength.eviction_delete_associations_sql",
                              ElapsedMillis (assoc_delete_start));
  p_ctx.association_fanout_cache.valid = false;

  const auto reconstruction_delete_start = SteadyClock::now ();
  ForEachChunk (evictable_memory_ids.size (),
                [&] (std::size_t begin, std::size_t end) {
                  const std::string placeholders
                      = eviction_policy::MakePlaceholders (end - begin);
                  tx.Execute (
                      "DELETE FROM memory_reconstructions "
                      "WHERE memory_id IN (" + placeholders + ")",
                      MakeParams (evictable_memory_ids, begin, end));
                });
  context.AddOperationTiming (
      "MemoryStrength.eviction_delete_reconstructions_sql",
      ElapsedMillis (reconstruction_delete_start));

  const auto signal_delete_start = SteadyClock::now ();
  ForEachChunk (evictable_memory_ids.size (),
                [&] (std::size_t begin, std::size_t end) {
                  const std::string placeholders
                      = eviction_policy::MakePlaceholders (end - begin);
                  tx.Execute (
                      "DELETE FROM signals WHERE memory_id IN ("
                          + placeholders + ")",
                      MakeParams (evictable_memory_ids, begin, end));
                });
  context.AddOperationTiming ("MemoryStrength.eviction_delete_signals_sql",
                              ElapsedMillis (signal_delete_start));

  const auto current_surface_delete_start = SteadyClock::now ();
  for (const long long memory_id : evictable_memory_ids)
    {
      tx.Execute ("DELETE FROM current_memory_embeddings WHERE memory_id = ?",
                  { memory_id });
    }
  context.AddOperationTiming (
      "MemoryStrength.eviction_delete_current_surface_sql",
      ElapsedMillis (current_surface_delete_start));

  const auto memory_delete_start = SteadyClock::now ();
  ForEachChunk (evictable_memory_ids.size (),
                [&] (std::size_t begin, std::size_t end) {
                  const std::string placeholders
                      = eviction_policy::MakePlaceholders (end - begin);
                  tx.Execute (
                      "DELETE FROM memories WHERE memory_id IN ("
                          + placeholders + ")",
                      MakeParams (evictable_memory_ids, begin, end));
                });
  context.AddOperationTiming ("MemoryStrength.eviction_delete_memories_sql",
                              ElapsedMillis (memory_delete_start));
  for (const long long memory_id : evictable_memory_ids)
    {
      emotional_metadata_cache_internal::Remove (
          context.GetProcessorContext (), memory_id);
    }
  rif_active_epoch_cache_internal::StageMemories (
      execution_cache_sidecar_internal::Ensure (
          context.GetProcessorContext ())->rif_active_epoch,
      evictable_memory_ids);

  const auto objstore_delete_start = SteadyClock::now ();
  const long long evicted_blob_count
      = DeleteOrphanedObjstoreBlobs (tx, evictable_blob_ids);
  context.AddOperationTiming ("MemoryStrength.eviction_delete_objstore_blobs_sql",
                              ElapsedMillis (objstore_delete_start));

  if (!evictable_embedding_ids.empty ())
    {
      // Some candidates may remain referenced after eviction. Invalidate the
      // private population rather than guessing which conditional deletes won.
      historical_surface_search_cache_internal::
          InvalidateHistoricalPreserveCurrent (
              context.GetProcessorContext ());
    }
  const auto embedding_delete_start = SteadyClock::now ();
  ForEachChunk (evictable_embedding_ids.size (),
                [&] (std::size_t begin, std::size_t end) {
                  const std::string placeholders
                      = eviction_policy::MakePlaceholders (end - begin);
                  tx.Execute (
                      "DELETE FROM embeddings "
                      "WHERE embedding_id IN (" + placeholders + ") "
                      "  AND NOT EXISTS ("
                      "    SELECT 1 FROM memories m "
                      "    WHERE m.embedding_id = embeddings.embedding_id"
                      "  ) "
                      "  AND NOT EXISTS ("
                      "    SELECT 1 FROM signals s "
                      "    WHERE s.embedding_id = embeddings.embedding_id"
                      "  ) "
                      "  AND NOT EXISTS ("
                      "    SELECT 1 FROM memory_reconstructions mr "
                      "    WHERE mr.embedding_id = embeddings.embedding_id"
                      "  )",
                      MakeParams (evictable_embedding_ids, begin, end));
                });
  context.AddOperationTiming ("MemoryStrength.eviction_delete_embeddings_sql",
                              ElapsedMillis (embedding_delete_start));
  const int64_t eviction_count
      = static_cast<int64_t> (evictable_memory_ids.size ());

  telemetry::LogDebug ("cortext.memory_strength",
                       { telemetry::Attribute::Int64 ("update_count",
                                                      update_count),
                         telemetry::Attribute::Int64 ("eviction_count",
                                                      eviction_count),
                         telemetry::Attribute::Int64 ("storage_used_bytes",
                                                      frontier.storage_used_bytes),
                         telemetry::Attribute::Int64 ("storage_threshold_bytes",
                                                      frontier.storage_threshold_bytes),
                         telemetry::Attribute::Int64 ("evicted_blob_count",
                                                      evicted_blob_count) });
}

} // namespace cortext::operations
