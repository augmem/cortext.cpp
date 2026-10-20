#include "temporal_seed_internal.hpp"

#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"

#include <algorithm>
#include <any>
#include <limits>
#include <string>
#include <unordered_set>

namespace cortext::operations::temporal_seed_internal
{

std::uint64_t
Distance (long long start_ts, long long end_ts, std::uint64_t reference_ts)
{
  const long long t = static_cast<long long> (std::min<std::uint64_t> (
      reference_ts,
      static_cast<std::uint64_t> (std::numeric_limits<long long>::max ())));
  const long long end = std::max (start_ts, end_ts);
  if (start_ts > t)
    return static_cast<std::uint64_t> (start_ts - t);
  if (end < t)
    return static_cast<std::uint64_t> (t - end);
  return 0;
}

std::string_view
NearestSql ()
{
  // INDEXED BY pins each arm to its LONG_TERM partial index: the kind index
  // would otherwise be an equally valid plan that reads every long-term row.
  static constexpr std::string_view kSql
      = "SELECT memory_id, start_ts, end_ts FROM ("
        "  SELECT memory_id, start_ts, end_ts "
        "  FROM memories INDEXED BY idx_memories_ltm_start "
        "  WHERE kind = 'LONG_TERM' AND start_ts > ?1 AND start_ts < ?2 "
        "  ORDER BY start_ts ASC LIMIT ?3) "
        "UNION ALL "
        "SELECT memory_id, start_ts, end_ts FROM ("
        "  SELECT memory_id, start_ts, end_ts "
        "  FROM memories INDEXED BY idx_memories_ltm_start "
        "  WHERE kind = 'LONG_TERM' AND start_ts <= ?1 AND start_ts < ?2 "
        "  ORDER BY start_ts DESC LIMIT ?3) "
        "UNION ALL "
        "SELECT memory_id, start_ts, end_ts FROM ("
        "  SELECT memory_id, start_ts, end_ts "
        "  FROM memories INDEXED BY idx_memories_ltm_end "
        "  WHERE kind = 'LONG_TERM' AND end_ts <= ?1 AND start_ts < ?2 "
        "  ORDER BY end_ts DESC LIMIT ?3)";
  return kSql;
}

std::vector<Hit>
Nearest (Transaction &tx, std::uint64_t reference_ts, long long exclusion_ts,
         std::size_t scan_limit)
{
  std::vector<Hit> hits;
  if (scan_limit == 0)
    return hits;
  const long long t = static_cast<long long> (std::min<std::uint64_t> (
      reference_ts,
      static_cast<std::uint64_t> (std::numeric_limits<long long>::max ())));
  const auto rows
      = tx.Execute (std::string (NearestSql ()),
                    { t, exclusion_ts, static_cast<long long> (scan_limit) });
  std::unordered_set<long long> seen;
  hits.reserve (rows.size ());
  for (const auto &row : rows)
    {
      const auto id = row.find ("memory_id");
      const auto start = row.find ("start_ts");
      const auto end = row.find ("end_ts");
      const long long memory_id
          = id == row.end () ? 0 : store::AnyToLongLong (id->second).value_or (0);
      if (memory_id <= 0 || !seen.insert (memory_id).second)
        continue;
      Hit hit;
      hit.memory_id = memory_id;
      hit.start_ts = start == row.end ()
                         ? 0
                         : store::AnyToLongLong (start->second).value_or (0);
      hit.end_ts = end == row.end ()
                       ? hit.start_ts
                       : store::AnyToLongLong (end->second)
                             .value_or (hit.start_ts);
      hit.distance = Distance (hit.start_ts, hit.end_ts, reference_ts);
      hits.push_back (hit);
    }
  std::sort (hits.begin (), hits.end (), [] (const Hit &a, const Hit &b) {
    if (a.distance != b.distance)
      return a.distance < b.distance;
    return a.memory_id < b.memory_id;
  });
  return hits;
}

} // namespace cortext::operations::temporal_seed_internal
