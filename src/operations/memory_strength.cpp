#include "cortext/operations/memory_strength.hpp"

#include "cortext/store/store.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "eviction_ablation.hpp"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <typeinfo>
#include <vector>

namespace cortext::operations
{

namespace
{

constexpr long long kDefaultEvictionMinBytes = 500LL * 1000LL * 1000LL;

long long
GetAnyInt64 (const std::map<std::string, std::any> &row, const std::string &key,
             long long fallback = 0)
{
  auto it = row.find (key);
  if (it == row.end () || !it->second.has_value ())
    return fallback;
  if (it->second.type () == typeid (long long))
    return std::any_cast<long long> (it->second);
  if (it->second.type () == typeid (int))
    return static_cast<long long> (std::any_cast<int> (it->second));
  if (it->second.type () == typeid (double))
    return static_cast<long long> (std::any_cast<double> (it->second));
  return fallback;
}

double
ParseEnvDouble (const char *name, double fallback)
{
  const char *raw = std::getenv (name);
  if (!raw || !*raw)
    return fallback;
  char *end = nullptr;
  const double value = std::strtod (raw, &end);
  if (end == raw)
    return fallback;
  return value;
}

long long
ParseEnvInt64 (const char *name, long long fallback)
{
  const char *raw = std::getenv (name);
  if (!raw || !*raw)
    return fallback;
  char *end = nullptr;
  const long long value = std::strtoll (raw, &end, 10);
  if (end == raw)
    return fallback;
  return value;
}

std::string
ResolveMainDbPath (Transaction &tx)
{
  auto rows = tx.Execute ("PRAGMA database_list", {});
  for (const auto &row : rows)
    {
      auto it_name = row.find ("name");
      auto it_file = row.find ("file");
      if (it_name == row.end () || it_file == row.end ()
          || !it_name->second.has_value () || !it_file->second.has_value ())
        {
          continue;
        }
      if (it_name->second.type () == typeid (std::string)
          && std::any_cast<std::string> (it_name->second) == "main"
          && it_file->second.type () == typeid (std::string))
        {
          return std::any_cast<std::string> (it_file->second);
        }
    }
  return {};
}

long long
QueryPragmaInt64 (Transaction &tx, const std::string &pragma_name)
{
  auto rows = tx.Execute ("PRAGMA " + pragma_name, {});
  if (rows.empty ())
    return 0;
  auto it = rows[0].find (pragma_name);
  if (it != rows[0].end ())
    return GetAnyInt64 (rows[0], pragma_name, 0);
  if (!rows[0].empty ())
    return GetAnyInt64 (rows[0], rows[0].begin ()->first, 0);
  return 0;
}

struct StorageGateInfo
{
  bool active = false;
  long long used_bytes = 0;
  long long threshold_bytes = 0;
};

StorageGateInfo
ComputeStorageGateInfo (Transaction &tx,
                        const eviction::EvictionAblationOverride &override)
{
  StorageGateInfo info;

  const std::string db_path = ResolveMainDbPath (tx);
  const bool file_backed = !db_path.empty ();
  info.active = override.storage_gate_enabled.value_or (file_backed);
  if (!info.active)
    return info;

  long long used_bytes = 0;
  if (override.used_storage_bytes.has_value ())
    {
      used_bytes = std::max<long long> (*override.used_storage_bytes, 0);
    }
  else if (file_backed)
    {
      std::error_code ec;
      const auto main_size = std::filesystem::file_size (db_path, ec);
      if (!ec)
        used_bytes += static_cast<long long> (main_size);
      const auto wal_path = db_path + "-wal";
      const auto wal_size = std::filesystem::file_size (wal_path, ec);
      if (!ec)
        used_bytes += static_cast<long long> (wal_size);
    }

  if (used_bytes <= 0)
    {
      const long long page_count = QueryPragmaInt64 (tx, "page_count");
      const long long page_size = QueryPragmaInt64 (tx, "page_size");
      used_bytes = page_count * page_size;
    }
  info.used_bytes = std::max<long long> (used_bytes, 0);

  const long long fixed_floor = override.min_storage_bytes.value_or (
      ParseEnvInt64 ("CORTEXT_EVICTION_MIN_DB_BYTES",
                     kDefaultEvictionMinBytes));
  const double fraction_floor
      = override.min_storage_fraction_of_available.value_or (
          ParseEnvDouble ("CORTEXT_EVICTION_MIN_DB_AVAIL_PCT", 0.0));
  long long threshold_bytes = std::max<long long> (fixed_floor, 0);
  if (file_backed && fraction_floor > 0.0)
    {
      std::error_code ec;
      const auto db_parent = std::filesystem::path (db_path).parent_path ();
      const auto space = std::filesystem::space (
          db_parent.empty () ? std::filesystem::current_path () : db_parent,
          ec);
      if (!ec)
        {
          const long long from_fraction = static_cast<long long> (
              std::floor (fraction_floor
                          * static_cast<double> (space.available)));
          threshold_bytes = std::max (threshold_bytes,
                                      std::max<long long> (from_fraction, 0));
        }
    }
  info.threshold_bytes = threshold_bytes;
  return info;
}

} // namespace

void
UpdateMemoryStrength::Execute (OperationContext &context, Transaction &tx) const
{
  const auto &cfg = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();

  const auto eviction_override = eviction::GetEvictionAblationOverride ();

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
  const double cutoff
      = eviction_override.periphery_cutoff.value_or (
          core::PeripheryCutoff (T));
  const double L_cg = std::round (core::Lerp (8.0, 32.0, T));
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
      const long long ts
          = static_cast<long long> (context.GetSignal ().timestamp);

      auto rows = tx.Execute (
          "SELECT strength, use_frequency, contextual_gain, retrieved_count, "
          "       used_count, last_access, created_at, flashbulb, "
          "       half_life_bonus, trace_fast, trace_med, trace_slow, trace_ultra "
          "FROM memories WHERE embedding_id = ?",
          { id });
      if (rows.empty ())
        {
          continue;
        }

      const double strength_prev
          = std::any_cast<double> (rows[0].at ("strength"));
      const double use_freq_prev
          = std::any_cast<double> (rows[0].at ("use_frequency"));
      const double contextual_gain_prev
          = std::any_cast<double> (rows[0].at ("contextual_gain"));
      auto get_double = [] (const std::any &v, double def) -> double {
        if (v.type () == typeid (double))
          return std::any_cast<double> (v);
        if (v.type () == typeid (float))
          return static_cast<double> (std::any_cast<float> (v));
        if (v.type () == typeid (int))
          return static_cast<double> (std::any_cast<int> (v));
        if (v.type () == typeid (long long))
          return static_cast<double> (std::any_cast<long long> (v));
        return def;
      };
      auto get_int = [] (const std::any &v, long long def) -> long long {
        if (v.type () == typeid (long long))
          return std::any_cast<long long> (v);
        if (v.type () == typeid (int))
          return static_cast<long long> (std::any_cast<int> (v));
        if (v.type () == typeid (double))
          return static_cast<long long> (std::any_cast<double> (v));
        return def;
      };
      const long long retrieved_prev
          = std::any_cast<long long> (rows[0].at ("retrieved_count"));
      const long long used_prev
          = std::any_cast<long long> (rows[0].at ("used_count"));
      const auto &last_access_any = rows[0].at ("last_access");
      const bool has_last_access
          = (last_access_any.type () != typeid (std::nullptr_t));
      const long long last_access_prev
          = has_last_access ? std::any_cast<long long> (last_access_any) : 0LL;
      const long long created_at
          = std::any_cast<long long> (rows[0].at ("created_at"));
      const int flashbulb
          = get_int (rows[0].at ("flashbulb"), 0) != 0 ? 1 : 0;
      const double half_life_bonus_raw
          = get_double (rows[0].at ("half_life_bonus"), 0.0);
      const double trace_fast_prev
          = get_double (rows[0].at ("trace_fast"), strength_prev);
      const double trace_med_prev
          = get_double (rows[0].at ("trace_med"), strength_prev * 0.5);
      const double trace_slow_prev
          = get_double (rows[0].at ("trace_slow"), strength_prev * 0.2);
      const double trace_ultra_prev
          = get_double (rows[0].at ("trace_ultra"), strength_prev * 0.05);

      const long long access_base
          = has_last_access ? last_access_prev : created_at;
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
                : 2 + static_cast<int> (std::round (2.0 * T));
      const double tau_fast = 0.10 * memory_half_life;
      const double tau_med = 0.50 * memory_half_life;
      const double tau_slow = 2.00 * memory_half_life;
      const double tau_ultra = 8.00 * memory_half_life;
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
                    0.05 + 0.10 * T);
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
          w_raw[0] = 0.40 - 0.25 * T;
          w_raw[1] = 0.25;
          w_raw[2] = 0.20 + 0.15 * T;
          w_raw[3] = 0.15 + 0.10 * T;
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

      tx.Execute (
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
          "WHERE embedding_id = ?",
          { retrieved_new, used_new, used_flag, ts, contextual_gain,
            influence_factor, use_frequency, strength,
            traces[0], traces[1], traces[2], traces[3], ts, id });

      ++update_count;
    }

  // Evict weak memories below periphery cutoff (v2: memories table)
  // Also delete corresponding signals, graph edges, and embeddings.
  // Fact-evidence floor: memories supporting active facts are protected
  // when their strength >= fact_floor(T), scaled by Stability.
  const long long evicted_at
      = static_cast<long long> (context.GetSignal ().timestamp);
  const bool fact_floor_active
      = eviction_override.fact_floor_enabled.value_or (true);
  const double fact_floor
      = fact_floor_active ? core::FactEvictionFloor (T) : 0.0;
  const StorageGateInfo storage_gate
      = ComputeStorageGateInfo (tx, eviction_override);

  // Eviction condition: strength < cutoff AND either (a) not fact-linked
  // or (b) strength < fact_floor. Expressed in SQL as a LEFT JOIN that
  // excludes protected memories.
  // Consolidation gate: no memory is evicted until it has existed through
  // at least one consolidation cycle (created_at < last_consolidation_ts).
  // When last_consolidation_ts == 0, no memories are evictable.
  const bool consolidation_gate_active
      = eviction_override.consolidation_gate_enabled.value_or (true);
  const long long consolidation_ts
      = static_cast<long long> (p_ctx.last_consolidation_ts);
  const std::string consolidation_gate
      = consolidation_gate_active ? " AND m.created_at < ?" : "";
  const std::string eviction_where
      = fact_floor_active
            ? "WHERE m.strength < ? AND m.kind = 'LONG_TERM'"
              " AND (fe_active.source_memory_id IS NULL"
              "      OR m.strength < ?)"
              + consolidation_gate
            : "WHERE m.strength < ? AND m.kind = 'LONG_TERM'"
              + consolidation_gate;
  const std::string fact_join
      = fact_floor_active
            ? " LEFT JOIN ("
              "   SELECT DISTINCT fe.source_memory_id"
              "   FROM fact_evidence fe"
              "   JOIN fact_assertions fa ON fe.fact_id = fa.fact_id"
              "   WHERE fa.lifecycle_state != 'archived'"
              " ) fe_active ON fe_active.source_memory_id = m.memory_id"
            : "";
  const std::vector<std::any> eviction_params
      = [&] {
          std::vector<std::any> p = { evicted_at, cutoff };
          if (fact_floor_active)
            p.push_back (fact_floor);
          if (consolidation_gate_active)
            p.push_back (consolidation_ts);
          return p;
        }();
  const std::vector<std::any> delete_params
      = [&] {
          std::vector<std::any> p = { cutoff };
          if (fact_floor_active)
            p.push_back (fact_floor);
          if (consolidation_gate_active)
            p.push_back (consolidation_ts);
          return p;
        }();

  if (storage_gate.active
      && storage_gate.used_bytes < storage_gate.threshold_bytes)
    {
      telemetry::LogDebug (
          "cortext.memory_strength",
          { telemetry::Attribute::Int64 ("update_count", update_count),
            telemetry::Attribute::Int64 ("eviction_count", 0),
            telemetry::Attribute::Int64 ("storage_used_bytes",
                                         storage_gate.used_bytes),
            telemetry::Attribute::Int64 ("storage_threshold_bytes",
                                         storage_gate.threshold_bytes) });
      return;
    }

  tx.Execute (
      "INSERT INTO memory_evictions ("
      "  memory_id, embedding_id, source_id, kind, label, start_ts, end_ts, "
      "  created_at, last_access, strength, use_frequency, contextual_gain, "
      "  retrieved_count, used_count, n_signals, modality, eviction_reason, "
      "  evicted_at"
      ") "
      "SELECT m.memory_id, m.embedding_id, m.source_id, m.kind, "
      "       COALESCE(m.label, ''), m.start_ts, m.end_ts, m.created_at, "
      "       m.last_access, m.strength, m.use_frequency, m.contextual_gain, "
      "       m.retrieved_count, m.used_count, m.n_signals, m.modality, "
      "       'periphery_cutoff', ? "
      "FROM memories m" + fact_join + " " + eviction_where,
      eviction_params);

  // Build subquery for evictable memory_ids
  const std::string evictable_ids
      = "SELECT m.memory_id FROM memories m" + fact_join + " " + eviction_where;

  tx.Execute (
      "DELETE FROM associations "
      "WHERE source_memory_id IN (" + evictable_ids + ") "
      "   OR target_memory_id IN (" + evictable_ids + ")",
      [&] {
        auto p = delete_params;
        p.insert (p.end (), delete_params.begin (), delete_params.end ());
        return p;
      }());
  tx.Execute (
      "DELETE FROM signals WHERE memory_id IN (" + evictable_ids + ")",
      delete_params);
  tx.Execute (
      "DELETE FROM embeddings WHERE embedding_id IN "
      "(SELECT m.embedding_id FROM memories m" + fact_join + " "
          + eviction_where + ")",
      delete_params);
  auto eviction_result = tx.Execute (
      "DELETE FROM memories WHERE memory_id IN (" + evictable_ids + ")",
      delete_params);
  const int64_t eviction_count = eviction_result.size ();

  telemetry::LogDebug ("cortext.memory_strength",
                       { telemetry::Attribute::Int64 ("update_count",
                                                      update_count),
                         telemetry::Attribute::Int64 ("eviction_count",
                                                      eviction_count),
                         telemetry::Attribute::Int64 ("storage_used_bytes",
                                                      storage_gate.used_bytes),
                         telemetry::Attribute::Int64 ("storage_threshold_bytes",
                                                      storage_gate.threshold_bytes) });
}

} // namespace cortext::operations
