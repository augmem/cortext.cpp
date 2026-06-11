#include "storage_pressure.hpp"

#include "cortext/store/store.hpp"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

namespace cortext::operations::pressure
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

} // namespace

StoragePressureState
ComputeStoragePressureState (Transaction &tx,
                             const eviction::EvictionAblationOverride &override)
{
  StoragePressureState info;

  const std::string db_path = ResolveMainDbPath (tx);
  const bool file_backed = !db_path.empty ();
  // An explicit used-bytes override is a simulated storage state: harnesses
  // drive in-memory stores through pressure scenarios, so the gate must be
  // considered active even though there is no backing file to measure.
  info.active = override.storage_gate_enabled.value_or (
      file_backed || override.used_storage_bytes.has_value ());
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

double
GateScale (const StoragePressureState &state, double low_pressure_scale)
{
  if (!state.active || state.threshold_bytes <= 0)
    {
      return 1.0;
    }
  return state.used_bytes >= state.threshold_bytes
             ? 1.0
             : std::clamp (low_pressure_scale, 0.0, 1.0);
}

double
RampScale (const StoragePressureState &state, double low_pressure_scale)
{
  if (!state.active || state.threshold_bytes <= 0)
    {
      return 1.0;
    }
  const double pressure_ratio
      = std::clamp (static_cast<double> (state.used_bytes)
                        / static_cast<double> (state.threshold_bytes),
                    0.0, 1.0);
  return std::clamp (low_pressure_scale, 0.0, 1.0)
         + (1.0 - std::clamp (low_pressure_scale, 0.0, 1.0))
               * pressure_ratio;
}

} // namespace cortext::operations::pressure
