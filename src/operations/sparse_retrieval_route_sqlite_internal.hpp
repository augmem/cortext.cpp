#pragma once

#include "sparse_retrieval_knobs_internal.hpp"
#include "sparse_retrieval_route_internal.hpp"
#include "../experimental_env.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace cortext
{
class Store;
}

namespace cortext::operations::sparse_retrieval_route_sqlite_internal
{

struct Parameters
{
  std::size_t route_capacity = 0;
  std::size_t activation_identity_target = 0;
  std::size_t activation_snapshot_capacity = 0;
  std::size_t total_query_row_budget = 0;
  std::size_t route_bootstrap_limit = 0;
  std::size_t search_node_budget = 0;
  std::size_t activation_search_node_budget_min = 0;
  std::size_t activation_search_node_budget_step = 0;
  std::size_t search_expansion_batch = 0;
  std::size_t search_ef = 0;
  std::size_t activation_search_ef_min = 0;
  std::size_t activation_search_ef_step = 0;
  std::size_t shadow_node_capacity = 0;
  std::size_t backfill_batch_size = 0;
  std::size_t backfill_search_node_budget = 0;
  std::size_t backfill_search_ef = 0;
  std::size_t row_addressed_neighbor_count = 0;
  std::size_t row_addressed_level_zero_links = 0;
  std::size_t family_exact_comparison_limit = 0;
  std::size_t maximum_level = 0;
  std::size_t reciprocal_update_count = 0;
  sparse_retrieval_route_internal::Parameters hnsw;
};

inline Parameters
DeriveParameters (double focus, double sensitivity, double stability)
{
  const auto route_capacity = static_cast<std::size_t> (
      sparse_retrieval_knobs_internal::RouteCapacity (
          focus, sensitivity, stability));
  const auto backfill_batch_size = static_cast<std::size_t> (
      sparse_retrieval_knobs_internal::BackfillBatchSize (
          focus, sensitivity, stability));
  const auto retrieval_sawtooth_floor = static_cast<std::size_t> (
      sparse_retrieval_knobs_internal::RetrievalSawtoothFloor (
          focus, sensitivity, stability));
  const auto retrieval_sawtooth_peak = static_cast<std::size_t> (
      sparse_retrieval_knobs_internal::RetrievalSawtoothPeak (
          focus, sensitivity, stability));
  const auto retrieval_sawtooth_step = static_cast<std::size_t> (
      sparse_retrieval_knobs_internal::RetrievalSawtoothStep (
          focus, sensitivity, stability));
  Parameters parameters {
    route_capacity,
    static_cast<std::size_t> (
        sparse_retrieval_knobs_internal::ActivationIdentityTarget (
            focus, sensitivity, stability)),
    static_cast<std::size_t> (
        sparse_retrieval_knobs_internal::ActivationIdentityTarget (
            focus, sensitivity, stability)),
    retrieval_sawtooth_peak
        + static_cast<std::size_t> (
            sparse_retrieval_knobs_internal::ActivationIdentityTarget (
                focus, sensitivity, stability)),
    route_capacity * 2,
    retrieval_sawtooth_peak,
    retrieval_sawtooth_floor,
    retrieval_sawtooth_step,
    static_cast<std::size_t> (
        sparse_retrieval_knobs_internal::SearchExpansionBatch (
            focus, sensitivity, stability)),
    retrieval_sawtooth_peak,
    retrieval_sawtooth_floor,
    retrieval_sawtooth_step,
    route_capacity * 24,
    backfill_batch_size,
    route_capacity + backfill_batch_size,
    backfill_batch_size * 2,
    static_cast<std::size_t> (sparse_retrieval_knobs_internal::
                                  GraphNeighborCount (
        focus, sensitivity, stability)),
    static_cast<std::size_t> (sparse_retrieval_knobs_internal::
                                  GraphLevelZeroLinks (
        focus, sensitivity, stability)),
    static_cast<std::size_t> (sparse_retrieval_knobs_internal::
                                  FamilyExactComparisonLimit (
        focus, sensitivity, stability)),
    static_cast<std::size_t> (sparse_retrieval_knobs_internal::MaximumLevel (
        focus, sensitivity, stability)),
    static_cast<std::size_t> (sparse_retrieval_knobs_internal::
                                  ReciprocalUpdateCount (
        focus, sensitivity, stability)),
    sparse_retrieval_route_internal::DeriveParameters (
        focus, sensitivity, stability),
  };

  // Private historical-ablation hook: every selectable node envelope is
  // composed only from the F/S/T-derived C, B, and A formulas. Hooks-off
  // production uses the 8C-to-9C lifecycle initialized above.
  if (const auto envelope = internal::experimental_env::Int (
          "CORTEXT_EXPERIMENT_SQLITE_SPARSE_NODE_ENVELOPE"))
    {
      const std::size_t legacy_query_budget = route_capacity * 5;
      std::size_t node_budget = legacy_query_budget;
      bool dynamic_activation_envelope = false;
      parameters.search_ef = legacy_query_budget;
      parameters.activation_search_ef_min = legacy_query_budget;
      parameters.activation_search_ef_step = 0;
      parameters.activation_search_node_budget_min = legacy_query_budget;
      parameters.activation_search_node_budget_step = 0;
      parameters.total_query_row_budget
          = legacy_query_budget + parameters.activation_identity_target;
      switch (*envelope)
        {
        case 1:
          node_budget = route_capacity;
          break;
        case 2:
          node_budget = parameters.activation_identity_target;
          break;
        case 3:
          node_budget = route_capacity + backfill_batch_size;
          break;
        case 4:
          node_budget = route_capacity * 2;
          break;
        case 5:
          node_budget = route_capacity * 5;
          break;
        case 6:
          node_budget = route_capacity * 6;
          break;
        case 8:
          node_budget = route_capacity * 4;
          break;
        case 18:
          node_budget
              = route_capacity * 4 + parameters.reciprocal_update_count;
          break;
        case 19:
          node_budget
              = route_capacity * 4 + parameters.reciprocal_update_count;
          parameters.search_ef = node_budget;
          parameters.activation_search_ef_min = node_budget;
          parameters.total_query_row_budget
              = node_budget + parameters.activation_identity_target;
          break;
        case 20:
          dynamic_activation_envelope = true;
          parameters.activation_search_ef_min
              = route_capacity * 4 + parameters.reciprocal_update_count;
          parameters.activation_search_ef_step
              = parameters.reciprocal_update_count;
          parameters.activation_search_node_budget_min
              = parameters.activation_search_ef_min;
          parameters.activation_search_node_budget_step
              = parameters.reciprocal_update_count;
          break;
        case 21:
          dynamic_activation_envelope = true;
          node_budget = route_capacity * 6;
          parameters.search_ef = node_budget;
          parameters.activation_search_ef_min = route_capacity * 5;
          parameters.activation_search_ef_step
              = parameters.reciprocal_update_count;
          parameters.activation_search_node_budget_min
              = parameters.activation_search_ef_min;
          parameters.activation_search_node_budget_step
              = parameters.reciprocal_update_count;
          parameters.total_query_row_budget
              = node_budget + parameters.activation_identity_target;
          break;
        case 22:
          dynamic_activation_envelope = true;
          node_budget = route_capacity * 7;
          parameters.search_ef = node_budget;
          parameters.activation_search_ef_min = route_capacity * 6;
          parameters.activation_search_ef_step
              = parameters.reciprocal_update_count;
          parameters.activation_search_node_budget_min
              = parameters.activation_search_ef_min;
          parameters.activation_search_node_budget_step
              = parameters.reciprocal_update_count;
          parameters.total_query_row_budget
              = node_budget + parameters.activation_identity_target;
          break;
        case 23:
          dynamic_activation_envelope = true;
          node_budget = route_capacity * 8;
          parameters.search_ef = node_budget;
          parameters.activation_search_ef_min = route_capacity * 7;
          parameters.activation_search_ef_step
              = parameters.reciprocal_update_count;
          parameters.activation_search_node_budget_min
              = parameters.activation_search_ef_min;
          parameters.activation_search_node_budget_step
              = parameters.reciprocal_update_count;
          parameters.total_query_row_budget
              = node_budget + parameters.activation_identity_target;
          break;
        case 24:
          dynamic_activation_envelope = true;
          node_budget = route_capacity * 9;
          parameters.search_ef = node_budget;
          parameters.activation_search_ef_min = route_capacity * 8;
          parameters.activation_search_ef_step
              = parameters.reciprocal_update_count;
          parameters.activation_search_node_budget_min
              = parameters.activation_search_ef_min;
          parameters.activation_search_node_budget_step
              = parameters.reciprocal_update_count;
          parameters.total_query_row_budget
              = node_budget + parameters.activation_identity_target;
          break;
        case 10:
          node_budget = route_capacity * 10;
          break;
        case 12:
          node_budget = route_capacity * 12;
          break;
        case 16:
          node_budget = route_capacity * 16;
          break;
        default:
          break;
        }
      parameters.search_node_budget = node_budget;
      if (!dynamic_activation_envelope)
        parameters.activation_search_node_budget_min = node_budget;
    }
  return parameters;
}

inline Parameters
DefaultParameters ()
{
  return DeriveParameters (0.5, 0.5, 0.5);
}

/// SQLite-authoritative, row-addressed mirror of the existing HNSW route.
/// Restart loads only graph metadata. Queries walk persisted HNSW adjacency
/// under an F/S/T-derived node envelope and rerank every activated row exactly.
class Route
{
public:
  ~Route ();

  static std::shared_ptr<Route> Create (
      Store &store, int embedding_dim,
      const std::vector<std::pair<long long, Eigen::VectorXf>> &entries,
      const sparse_retrieval_route_internal::Route &hnsw_route,
      Parameters parameters);

  static std::shared_ptr<Route> Open (
      Store &store, int embedding_dim,
      Parameters parameters);

  /// Open or begin an unpublished generation used for bounded legacy-store
  /// backfill. The generation is invisible to normal retrieval until
  /// PublishBuild succeeds.
  static std::shared_ptr<Route> OpenOrBeginBuild (Store &store,
                                                  int embedding_dim,
                                                  Parameters parameters);

  bool Upsert (long long memory_id, const Eigen::VectorXf &embedding);
  bool Remove (long long memory_id);
  bool StagePendingUpsert (long long memory_id,
                           const Eigen::VectorXf &embedding);
  bool StagePendingRemove (long long memory_id);
  bool Seal (const sparse_retrieval_route_internal::Route *hnsw_route
             = nullptr);
  bool Invalidate ();
  bool SetBuildCursor (long long memory_id);
  bool PublishBuild (std::size_t expected_active_count);

  std::optional<std::vector<long long>> Search (
      const Eigen::VectorXf &query, std::size_t route_capacity) const;
  /// Return every exactly reranked identity activated within the fixed query
  /// envelope. Callers apply operation-specific eligibility before reducing
  /// to the route-capacity result set.
  std::optional<std::vector<long long>> SearchActivated (
      const Eigen::VectorXf &query) const;
  /// Persist the level-zero activation pivot nearest the consolidation
  /// embedding. Persisted graph rows and the canonical hierarchy entry remain
  /// authoritative.
  bool Recenter (const Eigen::VectorXf &consolidation_embedding);

  std::size_t DeltaSize () const;
  std::size_t ActiveCount () const;
  long long BuildCursor () const;
  bool IsBuilding () const;
  std::optional<std::vector<long long>> DirtyMemoryIds (
      std::size_t limit) const;
  bool HasDirtyMemoryIds () const;
  std::size_t RestartRowsLoaded () const;
  std::size_t LastSearchNodeRows () const;
  /// Rows evaluated from the consolidation snapshot in addition to the
  /// current 8C-to-9C HNSW envelope. This is independently bounded by A.
  std::size_t LastActivationSnapshotRows () const;
  /// Snapshot rows physically loaded from SQLite for the last query. A
  /// repeated query may reuse the bounded generation-qualified shadow cache,
  /// while SQLite remains authoritative across seals and restarts.
  std::size_t LastActivationSnapshotCacheMissRows () const;
  /// Activation-snapshot identities admitted as level-zero frontier seeds by
  /// the last query. Recenter rebuilds must report zero because the prior
  /// snapshot cannot steer its replacement.
  std::size_t LastActivationFrontierSeedCount () const;
  std::size_t LastSearchDistanceEvaluations () const;
  std::size_t CachedNodeRows () const;
  std::size_t CacheOrderRows () const;
  int LastSearchFailureCode () const;
  int LastSealFailureCode () const;
  long long ActivationEntryMemoryId () const;
  std::size_t ActivationIdentityCount () const;
  std::vector<long long> ActivationIdentityIds () const;
  std::size_t ActivationSearchEffort () const;
  std::size_t ActivationSearchNodeBudget () const;

private:
  std::optional<std::vector<long long>> SearchWithEnvelope (
      const Eigen::VectorXf &query, std::size_t route_capacity,
      bool construction_search, bool return_all_activated,
      bool rebuild_activation = false) const;

  struct Impl;
  explicit Route (std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

} // namespace cortext::operations::sparse_retrieval_route_sqlite_internal
