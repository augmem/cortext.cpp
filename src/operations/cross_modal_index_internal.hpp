#pragma once

// Resident per-modality population of stored signal vectors for the
// cross-modal seed channel (graph_retrieval.cpp, CrossModalSeedConfig).
//
// The channel scores a query against every stored signal vector of each
// other modality that belongs to a LONG_TERM memory. Reading that population
// from SQLite on every query (join + blob decode per row) made each query
// O(N_modality) in SQL work. This index keeps one modality's population
// resident as an aligned row matrix and advances it incrementally:
//
//  * The population is append-only inside the engine: LONG_TERM memories and
//    their signal rows are inserted together by MemoryStorage and are never
//    deleted, re-kinded or re-pointed (only WORKING memories and their
//    signals are deleted or rewritten). A sync therefore reads only the
//    signal rows above a mark, the highest LONG_TERM signal id (of any
//    modality) already scanned (a primary-key range), so per-query SQL work
//    is bounded by what was written since the previous query. The mark is a
//    LONG_TERM row because only those rows are never deleted: a deleted
//    working row's id can be handed out again.
//  * The mark row is re-checked on every sync; if it vanished or changed (a
//    rolled-back outer transaction, external SQL) the modality is rebuilt
//    from scratch. Processor rollback and custom SQL also Erase the
//    index explicitly.
//  * A modality is loaded only when a query of another modality first needs
//    it, so a text-only workload never holds its text vectors twice.
//
// Scores are computed exactly as the per-query SQL path did (float dot and
// norms over aligned contiguous rows, double division), in signal-id order.
// Nothing here depends on what a modality is; modality names are opaque keys.

#include "cortext/processor/processor_context.hpp"
#include "cortext/store/store.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cortext::operations::cross_modal_index_internal
{

/// One modality's resident population, in signal-id order.
struct Population
{
  int dim = 0;
  /// Highest LONG_TERM signal id (any modality) scanned through, and its
  /// memory: rows at or below it are absorbed or not in this modality.
  long long mark_signal_id = 0;
  long long mark_memory_id = 0;
  std::vector<float, Eigen::aligned_allocator<float>> rows;  ///< size*dim
  std::vector<float> norms;
  std::vector<long long> memory_ids;

  std::size_t size () const { return memory_ids.size (); }
};

/// Advance `modality`'s resident population to the store's current LONG_TERM
/// signal rows, then append the cosine of `query` against each of its rows
/// to `out` as (memory_id, cosine), in signal-id order. Scores match the
/// former per-query SQL path's Cosine (float dot / float norm product, then
/// double division) bit for bit. Store errors propagate.
void ScoreModality (const ProcessorContext &ctx, Transaction &tx,
                    const std::string &modality, const Eigen::VectorXf &query,
                    std::vector<std::pair<long long, double>> &out);

/// Rows currently resident for `modality` (0 when never loaded).
std::size_t ResidentRows (const ProcessorContext &ctx,
                          const std::string &modality);

/// Drop every resident population of this processor (teardown, rollback,
/// custom SQL that may mutate).
void Erase (const ProcessorContext &ctx);

} // namespace cortext::operations::cross_modal_index_internal
