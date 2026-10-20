#pragma once

// Temporal seed channel for queries that carry a reference time
// (Cortext::RetrievalTime, Signal::reference_ts). It returns the long-term
// memories nearest the reference time T by time distance, in both
// directions: 0 when T falls inside a memory's [start_ts, end_ts], else the
// distance to its nearer edge. Time is stored signal metadata, so the
// channel is the same for every modality.
//
// Cost: three bounded range scans over the LONG_TERM partial indexes of
// schema migration 55 (idx_memories_ltm_start, idx_memories_ltm_end), each
// LIMIT scan_limit, so O(log N + scan_limit) rows; no full scan:
//   after   start_ts > T, nearest first          (exact for memories after T)
//   before  start_ts <= T, latest start first    (memories that hold T)
//   ended   end_ts <= T, latest end first        (exact for memories before T)
// The union is ranked by exact interval distance. A memory that holds T is
// missed only when more than scan_limit memories start between its start
// and T; such a memory ties at distance 0 with the ones that are found.

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace cortext
{
class Transaction;
}

namespace cortext::operations::temporal_seed_internal
{

/// Candidates the channel contributes to a query's seed fusion (K).
inline constexpr std::size_t kLimit = 10;

/// Reciprocal-rank weight of the channel's list. The dense seed list has
/// weight 1 and the keyword list 1; half of either keeps time an added cue
/// that cannot outrank semantic relevance on its own: a memory found only by
/// time ranks after the whole dense seed list (1 / (60 + r) > 0.5 / 61 for
/// every dense rank r <= 62), while a dense or keyword hit that is also near
/// T rises.
inline constexpr double kWeight = 0.5;

/// One memory near the reference time.
struct Hit
{
  long long memory_id = 0;
  long long start_ts = 0;
  long long end_ts = 0;
  std::uint64_t distance = 0;
};

/// Interval distance from reference_ts to [start_ts, max(start_ts, end_ts)].
std::uint64_t Distance (long long start_ts, long long end_ts,
                        std::uint64_t reference_ts);

/// The channel statement. Parameters: ?1 reference time, ?2 start_ts upper
/// bound (exclusive; the seed exclusion time), ?3 rows per scan.
std::string_view NearestSql ();

/// Distinct LONG_TERM memories with start_ts < exclusion_ts from the three
/// scans, nearest first (distance, then memory_id). Callers apply the
/// remaining seed eligibility rules and their own limit.
std::vector<Hit> Nearest (Transaction &tx, std::uint64_t reference_ts,
                          long long exclusion_ts, std::size_t scan_limit);

} // namespace cortext::operations::temporal_seed_internal
