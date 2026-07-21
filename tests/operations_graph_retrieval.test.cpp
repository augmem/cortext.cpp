#include "test_helpers.hpp"
#include "../src/operations/association_fanout_cache_internal.hpp"
#include "../src/operations/constructive_recall_internal.hpp"
#include "../src/operations/historical_surface_search_cache_internal.hpp"
#include "../src/operations/retrieval_trace_state.hpp"
#include "../src/operations/sparse_retrieval_route_internal.hpp"
#include "../src/operations/sparse_retrieval_route_sqlite_internal.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cortext/core/utils.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace cortext;
using cortext::operations::GraphAugmentedRetrieveCandidates;

TEST_CASE ("retrieval surface mutation capture is explicit and ordered",
           "[graph-retrieval][trace]")
{
  using namespace cortext::operations::retrieval_trace;
  SetSurfaceMutationCaptureEnabled (false);
  RecordSurfaceUpsert (1, 2, { 1.0f, 0.0f });
  REQUIRE (GetSurfaceMutations ().empty ());

  SetSurfaceMutationCaptureEnabled (true);
  ClearSurfaceMutations ();
  RecordSurfaceUpsert (7, 11, { 0.25f, 0.75f });
  RecordSurfaceRemove (7);
  const auto &mutations = GetSurfaceMutations ();
  REQUIRE (mutations.size () == 2);
  CHECK (mutations[0].action == SurfaceMutation::Action::Upsert);
  CHECK (mutations[0].memory_id == 7);
  CHECK (mutations[0].embedding_id == 11);
  CHECK ((mutations[0].embedding
          == std::vector<float>{ 0.25f, 0.75f }));
  CHECK (mutations[1].action == SurfaceMutation::Action::Remove);
  CHECK (mutations[1].memory_id == 7);

  SetSurfaceMutationCaptureEnabled (false);
  REQUIRE (GetSurfaceMutations ().empty ());
}

namespace
{
constexpr int kEmbeddingDim = 256;

Eigen::VectorXf
UnitVec (int dim)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[dim] = 1.0f;
  return v;
}

TEST_CASE ("Sparse retrieval route is deterministic across upsert and remove",
           "[operations][graph][retrieval][hnsw]")
{
  using operations::sparse_retrieval_route_internal::Route;
  const std::vector<std::pair<long long, Eigen::VectorXf>> entries {
    { 10, UnitVec (0) }, { 20, UnitVec (1) }, { 30, UnitVec (2) }
  };
  const auto parameters
      = operations::sparse_retrieval_route_internal::DefaultParameters ();
  auto first = Route::Create (kEmbeddingDim, entries, parameters);
  auto second = Route::Create (kEmbeddingDim, entries, parameters);
  REQUIRE (first != nullptr);
  REQUIRE (second != nullptr);
  REQUIRE (first->DeltaSize () == 0);
  REQUIRE (first->Search (UnitVec (0), 2)
           == second->Search (UnitVec (0), 2));
  REQUIRE (first->Search (UnitVec (0), 2)->front () == 10);

  REQUIRE (first->Upsert (10, UnitVec (3)));
  REQUIRE (first->DeltaSize () == 1);
  REQUIRE (first->Search (UnitVec (3), 3)->front () == 10);
  REQUIRE (first->Sync (
      { { 10, UnitVec (0) }, { 20, UnitVec (1) }, { 30, UnitVec (2) } }));
  REQUIRE (first->DeltaSize () == 0);
  REQUIRE (first->Remove (10));
  REQUIRE (first->DeltaSize () == 1);
  const auto after_remove = first->Search (UnitVec (3), 3);
  REQUIRE (after_remove.has_value ());
  REQUIRE (std::find (after_remove->begin (), after_remove->end (), 10)
           == after_remove->end ());
  REQUIRE (first->Size () == 2);
  REQUIRE (first->SealDelta ());
  REQUIRE (first->DeltaSize () == 0);
  REQUIRE (first->Size () == 2);
  REQUIRE (first->Search (UnitVec (0), 3)->front () != 10);
  REQUIRE (first->Upsert (10, UnitVec (0)));
  REQUIRE (first->SealDelta ());
  REQUIRE (first->DeltaSize () == 0);
  REQUIRE (first->Size () == 3);
  REQUIRE (first->Search (UnitVec (0), 3)->front () == 10);
}

TEST_CASE ("Sparse retrieval route repairs below an empty upper-layer entry",
           "[operations][graph][retrieval][hnsw][regression]")
{
  using operations::sparse_retrieval_route_internal::NodeSnapshot;
  using operations::sparse_retrieval_route_internal::Route;
  auto route = Route::CreateWithLevelsForTest (
      kEmbeddingDim, { { 1, UnitVec (1) }, { 2, UnitVec (2) } },
      { 2, 1 },
      operations::sparse_retrieval_route_internal::DeriveParameters (
          0.0, 0.0, 0.0));
  REQUIRE (route);
  const auto snapshot = route->Snapshot ({});
  REQUIRE (snapshot);
  REQUIRE (snapshot->entry_memory_id == 1);
  REQUIRE (snapshot->max_level == 2);
  const auto entry = std::find_if (
      snapshot->nodes.begin (), snapshot->nodes.end (),
      [&] (const NodeSnapshot &node) {
        return node.memory_id == snapshot->entry_memory_id;
      });
  REQUIRE (entry != snapshot->nodes.end ());
  REQUIRE (entry->links.size () == 3);
  REQUIRE (entry->links[2].empty ());
  const auto target = std::find_if (
      snapshot->nodes.begin (), snapshot->nodes.end (),
      [] (const NodeSnapshot &node) { return node.memory_id == 2; });
  REQUIRE (target != snapshot->nodes.end ());
  REQUIRE (target->level == 1);

  const long long target_memory_id = target->memory_id;
  REQUIRE (route->Upsert (target_memory_id, UnitVec (200)));
  REQUIRE (route->SealDelta ());
  const auto result = route->Search (UnitVec (200), 1);
  REQUIRE (result);
  REQUIRE (result->size () == 1);
  REQUIRE (result->front () == target_memory_id);
}

TEST_CASE ("SQLite sparse route restart and query work stay row bounded",
           "[operations][graph][retrieval][hnsw][sqlite]")
{
  using HnswRoute
      = operations::sparse_retrieval_route_internal::Route;
  using SQLiteRoute
      = operations::sparse_retrieval_route_sqlite_internal::Route;
  const auto parameters = operations::sparse_retrieval_route_sqlite_internal::
      DefaultParameters ();
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<std::pair<long long, Eigen::VectorXf>> entries;
  for (long long index = 0; index < 600; ++index)
    entries.emplace_back (index + 1,
                          UnitVec (static_cast<int> (
                              index % kEmbeddingDim)));
  auto hnsw = HnswRoute::Create (
      kEmbeddingDim, entries,
      operations::sparse_retrieval_route_internal::DefaultParameters ());
  REQUIRE (hnsw != nullptr);
  auto route = SQLiteRoute::Create (
      *store, kEmbeddingDim, entries, *hnsw, parameters);
  REQUIRE (route != nullptr);
  REQUIRE (route->RestartRowsLoaded () == 1);
  REQUIRE (route->ActivationSearchEffort () == parameters.search_ef);
  REQUIRE (route->ActivationSearchNodeBudget ()
           == parameters.search_node_budget);

  const auto initial = route->Search (UnitVec (0), 32);
  REQUIRE (initial.has_value ());
  REQUIRE_FALSE (initial->empty ());
  REQUIRE (initial->front () == 1);
  REQUIRE (route->LastSearchNodeRows () == entries.size ());
  REQUIRE (route->LastSearchNodeRows () <= parameters.search_node_budget);
  REQUIRE (route->CachedNodeRows () <= parameters.shadow_node_capacity);
  REQUIRE (route->CacheOrderRows () <= parameters.shadow_node_capacity);

  REQUIRE (route->ActivationEntryMemoryId () == 0);
  const auto recenter_target
      = route->Search (UnitVec (5), parameters.route_capacity);
  REQUIRE (recenter_target);
  REQUIRE_FALSE (recenter_target->empty ());
  REQUIRE (route->Recenter (UnitVec (5)));
  REQUIRE (route->ActivationEntryMemoryId () == recenter_target->front ());
  REQUIRE (route->ActivationSearchEffort ()
           == parameters.activation_search_ef_min);
  REQUIRE (route->ActivationSearchNodeBudget ()
           == parameters.activation_search_node_budget_min);
  REQUIRE (route->Search (UnitVec (5), 32));
  REQUIRE (route->ActivationSearchEffort ()
           == std::min (parameters.search_ef,
                        parameters.activation_search_ef_min
                            + parameters.activation_search_ef_step));
  REQUIRE (
      route->ActivationSearchNodeBudget ()
      == std::min (parameters.search_node_budget,
                   parameters.activation_search_node_budget_min
                       + parameters.activation_search_node_budget_step));
  const auto effort_before_failed_recenter
      = route->ActivationSearchEffort ();
  const auto budget_before_failed_recenter
      = route->ActivationSearchNodeBudget ();
  store->Execute (
      "CREATE TRIGGER fail_sparse_route_recenter BEFORE UPDATE "
      "ON cortext_sparse_route_meta BEGIN "
      "SELECT RAISE(ABORT, 'injected sparse route recenter failure'); END");
  REQUIRE_FALSE (route->Recenter (UnitVec (6)));
  REQUIRE (route->ActivationSearchEffort ()
           == effort_before_failed_recenter);
  REQUIRE (route->ActivationSearchNodeBudget ()
           == budget_before_failed_recenter);
  store->Execute ("DROP TRIGGER fail_sparse_route_recenter");
  const auto before_restart = route->Search (UnitVec (0), 32);
  REQUIRE (before_restart);

  auto restarted = SQLiteRoute::Open (*store, kEmbeddingDim, parameters);
  REQUIRE (restarted != nullptr);
  REQUIRE (restarted->RestartRowsLoaded () == 1);
  REQUIRE (restarted->ActivationEntryMemoryId () == recenter_target->front ());
  REQUIRE (restarted->ActivationSearchEffort () == parameters.search_ef);
  REQUIRE (restarted->ActivationSearchNodeBudget ()
           == parameters.search_node_budget);
  REQUIRE (restarted->Search (UnitVec (0), 32) == before_restart);
  REQUIRE (restarted->Upsert (1000, UnitVec (0)));
  REQUIRE (restarted->Remove (1));
  REQUIRE (restarted->StagePendingUpsert (1000, UnitVec (0)));
  REQUIRE (restarted->StagePendingRemove (1));
  REQUIRE (hnsw->Upsert (1000, UnitVec (0)));
  REQUIRE (hnsw->Remove (1));
  REQUIRE (hnsw->SealDelta ());
  REQUIRE (restarted->DeltaSize () == 2);
  store->Execute (
      "CREATE TRIGGER fail_sparse_route_seal BEFORE UPDATE "
      "ON cortext_sparse_route_nodes WHEN NEW.generation > 0 "
      "BEGIN SELECT RAISE(ABORT, 'injected sparse route seal failure'); END");
  REQUIRE_FALSE (restarted->Seal (hnsw.get ()));
  REQUIRE (restarted->DeltaSize () == 2);
  const auto pending = restarted->Search (UnitVec (0), 32);
  REQUIRE (pending.has_value ());
  REQUIRE (std::find (pending->begin (), pending->end (), 1)
           == pending->end ());
  REQUIRE (std::find (pending->begin (), pending->end (), 1000)
           != pending->end ());
  auto persisted_before_retry = SQLiteRoute::Open (
      *store, kEmbeddingDim, parameters);
  REQUIRE (persisted_before_retry);
  REQUIRE (persisted_before_retry->HasDirtyMemoryIds ());
  store->Execute ("DROP TRIGGER fail_sparse_route_seal");
  REQUIRE (restarted->Seal (hnsw.get ()));
  REQUIRE (restarted->DeltaSize () == 0);
  REQUIRE (restarted->CachedNodeRows () == 0);
  REQUIRE (restarted->CacheOrderRows () == 0);

  restarted = SQLiteRoute::Open (*store, kEmbeddingDim, parameters);
  REQUIRE (restarted != nullptr);
  const auto inserted_rows = store->Execute (
      "SELECT active FROM cortext_sparse_route_nodes WHERE memory_id = ?",
      { 1000LL });
  REQUIRE (inserted_rows.size () == 1);
  REQUIRE (store::AnyToLongLong (inserted_rows.front ().at ("active"))
               .value_or (0)
           == 1);
  const auto after_seal = restarted->Search (UnitVec (0), 32);
  REQUIRE (after_seal.has_value ());
  REQUIRE (std::find (after_seal->begin (), after_seal->end (), 1)
           == after_seal->end ());
  const auto inserted = restarted->SearchActivated (UnitVec (0));
  REQUIRE (inserted.has_value ());
  REQUIRE (std::find (inserted->begin (), inserted->end (), 1000)
           != inserted->end ());
  REQUIRE (restarted->LastSearchNodeRows ()
           <= parameters.search_node_budget);

  Eigen::VectorXf row_addressed = Eigen::VectorXf::Zero (kEmbeddingDim);
  row_addressed[0] = 0.6f;
  row_addressed[1] = 0.8f;
  REQUIRE (restarted->Upsert (2000, row_addressed));
  REQUIRE (restarted->StagePendingUpsert (2000, row_addressed));
  REQUIRE (restarted->DeltaSize () == 1);
  REQUIRE (restarted->Seal ());
  REQUIRE (restarted->LastSearchNodeRows ()
           <= parameters.backfill_search_node_budget);
  REQUIRE (restarted->ActivationSearchEffort () == parameters.search_ef);
  REQUIRE (restarted->ActivationSearchNodeBudget ()
           == parameters.search_node_budget);
  REQUIRE (restarted->DeltaSize () == 0);
  restarted = SQLiteRoute::Open (*store, kEmbeddingDim, parameters);
  REQUIRE (restarted != nullptr);
  const auto row_addressed_inserted
      = restarted->Search (row_addressed, 32);
  REQUIRE (row_addressed_inserted.has_value ());
  REQUIRE_FALSE (row_addressed_inserted->empty ());
  REQUIRE (row_addressed_inserted->front () == 2000);
  REQUIRE (restarted->LastSearchNodeRows ()
           <= parameters.search_node_budget);
  REQUIRE (restarted->Remove (2000));
  REQUIRE (restarted->StagePendingRemove (2000));
  REQUIRE (restarted->Seal ());
  restarted = SQLiteRoute::Open (*store, kEmbeddingDim, parameters);
  REQUIRE (restarted != nullptr);
  const auto row_addressed_removed
      = restarted->Search (row_addressed, 32);
  REQUIRE (row_addressed_removed.has_value ());
  REQUIRE (std::find (row_addressed_removed->begin (),
                      row_addressed_removed->end (), 2000)
           == row_addressed_removed->end ());

}

TEST_CASE ("Consolidation resets the experimental derived retrieval sawtooth",
           "[operations][graph][retrieval][hnsw][sqlite][consolidation]"
           "[activation][knobs][experiment_hooks]")
{
  using HnswRoute
      = operations::sparse_retrieval_route_internal::Route;
  using SQLiteRoute
      = operations::sparse_retrieval_route_sqlite_internal::Route;
  cortext::testing::ScopedEnvVar envelope (
      "CORTEXT_EXPERIMENT_SQLITE_SPARSE_NODE_ENVELOPE", "20");
  const auto parameters
      = operations::sparse_retrieval_route_sqlite_internal::
          DeriveParameters (0.0, 0.0, 0.0);
  const std::size_t step = parameters.reciprocal_update_count;
  REQUIRE (parameters.activation_search_node_budget_min
           == parameters.route_capacity * 4 + step);
  REQUIRE (parameters.activation_search_node_budget_step == step);
  REQUIRE (parameters.activation_search_ef_min
           == parameters.activation_search_node_budget_min);
  REQUIRE (parameters.activation_search_ef_step == step);

  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  std::vector<std::pair<long long, Eigen::VectorXf>> entries;
  entries.reserve (700);
  for (long long memory_id = 1; memory_id <= 700; ++memory_id)
    entries.emplace_back (
        memory_id,
        UnitVec (static_cast<int> (memory_id % kEmbeddingDim)));
  auto hnsw = HnswRoute::Create (kEmbeddingDim, entries, parameters.hnsw);
  REQUIRE (hnsw);
  auto route = SQLiteRoute::Create (
      *store, kEmbeddingDim, entries, *hnsw, parameters);
  REQUIRE (route);
  REQUIRE (route->ActivationSearchNodeBudget ()
           == parameters.search_node_budget);
  REQUIRE (route->Recenter (UnitVec (0)));
  REQUIRE (route->ActivationSearchNodeBudget ()
           == parameters.activation_search_node_budget_min);
  REQUIRE (route->ActivationSearchEffort ()
           == parameters.activation_search_ef_min);

  REQUIRE (route->SearchActivated (UnitVec (0)));
  REQUIRE (route->ActivationSearchNodeBudget ()
           == parameters.activation_search_node_budget_min + step);
  REQUIRE (route->ActivationSearchEffort ()
           == parameters.activation_search_ef_min + step);
  while (route->ActivationSearchNodeBudget ()
         < parameters.search_node_budget)
    REQUIRE (route->SearchActivated (UnitVec (0)));
  REQUIRE (route->ActivationSearchNodeBudget ()
           == parameters.search_node_budget);
  REQUIRE (route->ActivationSearchEffort () == parameters.search_ef);
}

TEST_CASE ("Quality-preserving experimental retrieval sawtooth keeps the "
           "canonical floor",
           "[operations][graph][retrieval][hnsw][sqlite][consolidation]"
           "[activation][knobs][experiment_hooks]")
{
  using HnswRoute
      = operations::sparse_retrieval_route_internal::Route;
  using SQLiteRoute
      = operations::sparse_retrieval_route_sqlite_internal::Route;
  cortext::testing::ScopedEnvVar envelope (
      "CORTEXT_EXPERIMENT_SQLITE_SPARSE_NODE_ENVELOPE", "21");
  const auto parameters
      = operations::sparse_retrieval_route_sqlite_internal::
          DeriveParameters (0.0, 0.0, 0.0);
  const std::size_t step = parameters.reciprocal_update_count;
  REQUIRE (parameters.activation_search_node_budget_min
           == parameters.route_capacity * 5);
  REQUIRE (parameters.activation_search_node_budget_step == step);
  REQUIRE (parameters.search_node_budget == parameters.route_capacity * 6);
  REQUIRE (parameters.search_ef == parameters.search_node_budget);
  REQUIRE (parameters.activation_search_ef_min
           == parameters.activation_search_node_budget_min);
  REQUIRE (parameters.activation_search_ef_step == step);
  REQUIRE (parameters.total_query_row_budget
           == parameters.search_node_budget
                  + parameters.activation_identity_target);

  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  std::vector<std::pair<long long, Eigen::VectorXf>> entries;
  entries.reserve (500);
  for (long long memory_id = 1; memory_id <= 500; ++memory_id)
    entries.emplace_back (
        memory_id,
        UnitVec (static_cast<int> (memory_id % kEmbeddingDim)));
  auto hnsw = HnswRoute::Create (kEmbeddingDim, entries, parameters.hnsw);
  REQUIRE (hnsw);
  auto route = SQLiteRoute::Create (
      *store, kEmbeddingDim, entries, *hnsw, parameters);
  REQUIRE (route);
  REQUIRE (route->ActivationSearchNodeBudget ()
           == parameters.search_node_budget);
  REQUIRE (route->Recenter (UnitVec (0)));
  REQUIRE (route->ActivationSearchNodeBudget ()
           == parameters.activation_search_node_budget_min);
  REQUIRE (route->ActivationSearchEffort ()
           == parameters.activation_search_ef_min);
  REQUIRE (route->SearchActivated (UnitVec (0)));
  REQUIRE (route->ActivationSearchNodeBudget ()
           == parameters.activation_search_node_budget_min + step);
  REQUIRE (route->ActivationSearchEffort ()
           == parameters.activation_search_ef_min + step);
}

TEST_CASE ("Experimental retrieval sawtooth can raise only its derived floor",
           "[operations][graph][retrieval][hnsw][sqlite][consolidation]"
           "[activation][knobs][experiment_hooks]")
{
  cortext::testing::ScopedEnvVar envelope (
      "CORTEXT_EXPERIMENT_SQLITE_SPARSE_NODE_ENVELOPE", "22");
  const auto parameters
      = operations::sparse_retrieval_route_sqlite_internal::
          DeriveParameters (0.5, 0.5, 0.5);
  REQUIRE (parameters.activation_search_node_budget_min
           == parameters.route_capacity * 6);
  REQUIRE (parameters.activation_search_ef_min
           == parameters.route_capacity * 6);
  REQUIRE (parameters.search_node_budget == parameters.route_capacity * 7);
  REQUIRE (parameters.search_ef == parameters.route_capacity * 7);
  REQUIRE (parameters.activation_search_node_budget_step
           == parameters.reciprocal_update_count);
  REQUIRE (parameters.activation_search_ef_step
           == parameters.reciprocal_update_count);
  REQUIRE (parameters.total_query_row_budget
           == parameters.search_node_budget
                  + parameters.activation_identity_target);
}

TEST_CASE ("Experimental retrieval floor ablation remains knob-derived",
           "[operations][graph][retrieval][hnsw][sqlite][consolidation]"
           "[activation][knobs][experiment_hooks]")
{
  cortext::testing::ScopedEnvVar envelope (
      "CORTEXT_EXPERIMENT_SQLITE_SPARSE_NODE_ENVELOPE", "23");
  const auto parameters
      = operations::sparse_retrieval_route_sqlite_internal::
          DeriveParameters (0.5, 0.5, 0.5);
  REQUIRE (parameters.activation_search_node_budget_min
           == parameters.route_capacity * 7);
  REQUIRE (parameters.activation_search_ef_min
           == parameters.route_capacity * 7);
  REQUIRE (parameters.search_node_budget == parameters.route_capacity * 8);
  REQUIRE (parameters.search_ef == parameters.route_capacity * 8);
  REQUIRE (parameters.activation_search_node_budget_step
           == parameters.reciprocal_update_count);
  REQUIRE (parameters.activation_search_ef_step
           == parameters.reciprocal_update_count);
  REQUIRE (parameters.total_query_row_budget
           == parameters.search_node_budget
                  + parameters.activation_identity_target);
}

TEST_CASE ("Final adjacent retrieval floor ablation remains knob-derived",
           "[operations][graph][retrieval][hnsw][sqlite][consolidation]"
           "[activation][knobs][experiment_hooks]")
{
  const auto production
      = operations::sparse_retrieval_route_sqlite_internal::
          DeriveParameters (0.5, 0.5, 0.5);
  cortext::testing::ScopedEnvVar envelope (
      "CORTEXT_EXPERIMENT_SQLITE_SPARSE_NODE_ENVELOPE", "24");
  const auto parameters
      = operations::sparse_retrieval_route_sqlite_internal::
          DeriveParameters (0.5, 0.5, 0.5);
  REQUIRE (parameters.activation_search_node_budget_min
           == parameters.route_capacity * 8);
  REQUIRE (parameters.activation_search_ef_min
           == parameters.route_capacity * 8);
  REQUIRE (parameters.search_node_budget == parameters.route_capacity * 9);
  REQUIRE (parameters.search_ef == parameters.route_capacity * 9);
  REQUIRE (parameters.activation_search_node_budget_step
           == parameters.reciprocal_update_count);
  REQUIRE (parameters.activation_search_ef_step
           == parameters.reciprocal_update_count);
  REQUIRE (parameters.total_query_row_budget
           == parameters.search_node_budget
                  + parameters.activation_identity_target);
  REQUIRE (parameters.search_node_budget == production.search_node_budget);
  REQUIRE (parameters.activation_search_node_budget_min
           == production.activation_search_node_budget_min);
  REQUIRE (parameters.activation_search_node_budget_step
           == production.activation_search_node_budget_step);
  REQUIRE (parameters.search_ef == production.search_ef);
  REQUIRE (parameters.activation_search_ef_min
           == production.activation_search_ef_min);
  REQUIRE (parameters.activation_search_ef_step
           == production.activation_search_ef_step);
  REQUIRE (parameters.total_query_row_budget
           == production.total_query_row_budget);
}

TEST_CASE ("Mature SQLite activation fills its knob-derived node envelope "
           "across inactive graph rows",
           "[operations][graph][retrieval][hnsw][sqlite][consolidation]"
           "[regression]")
{
  using HnswRoute
      = operations::sparse_retrieval_route_internal::Route;
  using SQLiteRoute
      = operations::sparse_retrieval_route_sqlite_internal::Route;
  const auto parameters
      = operations::sparse_retrieval_route_sqlite_internal::
          DeriveParameters (0.0, 0.0, 0.0);
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);

  const long long active_count
      = static_cast<long long> (parameters.search_node_budget - 1);
  std::vector<std::pair<long long, Eigen::VectorXf>> entries;
  entries.reserve (static_cast<std::size_t> (active_count));
  for (long long memory_id = 1; memory_id <= active_count; ++memory_id)
    entries.emplace_back (
        memory_id,
        UnitVec (static_cast<int> (memory_id % kEmbeddingDim)));
  auto hnsw = HnswRoute::Create (
      kEmbeddingDim, entries, parameters.hnsw);
  REQUIRE (hnsw);
  auto route = SQLiteRoute::Create (
      *store, kEmbeddingDim, entries, *hnsw, parameters);
  REQUIRE (route);
  const std::vector<unsigned char> empty_level_zero_links {
    1, 0, 0, 0, 0, 0, 0, 0
  };
  const long long inactive_memory_id = active_count + 1;
  store->Execute (
      "INSERT INTO cortext_sparse_route_nodes("
      "memory_id, embedding, level, links, active, generation) "
      "SELECT ?, embedding, 0, ?, 0, generation "
      "FROM cortext_sparse_route_nodes WHERE memory_id = 1",
      { inactive_memory_id, empty_level_zero_links });
  route.reset ();
  route = SQLiteRoute::Open (*store, kEmbeddingDim, parameters);
  REQUIRE (route);

  const auto activated = route->SearchActivated (UnitVec (0));
  REQUIRE (activated);
  REQUIRE (activated->size () == parameters.activation_identity_target);
  REQUIRE (std::find (activated->begin (), activated->end (),
                      inactive_memory_id)
           == activated->end ());
  REQUIRE (route->LastSearchNodeRows ()
           == parameters.search_node_budget);
}

TEST_CASE ("Consolidation rebuilds and restarts a knob-bounded SQLite "
           "activation centroid",
           "[operations][graph][retrieval][hnsw][sqlite][consolidation]"
           "[activation][knobs][restart]")
{
  using HnswRoute
      = operations::sparse_retrieval_route_internal::Route;
  using SQLiteRoute
      = operations::sparse_retrieval_route_sqlite_internal::Route;
  const auto parameters = operations::sparse_retrieval_route_sqlite_internal::
      DeriveParameters (0.0, 0.0, 0.0);
  REQUIRE (parameters.activation_identity_target == 640);
  REQUIRE (parameters.search_node_budget == 2304);
  REQUIRE (parameters.activation_search_node_budget_min == 2048);
  REQUIRE (parameters.activation_search_node_budget_step == 4);
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<std::pair<long long, Eigen::VectorXf>> entries;
  entries.reserve (1400);
  for (long long memory_id = 1; memory_id <= 1400; ++memory_id)
    entries.emplace_back (memory_id,
                          UnitVec (memory_id <= 700 ? 0 : 1));
  auto hnsw = HnswRoute::Create (
      kEmbeddingDim, entries, parameters.hnsw);
  REQUIRE (hnsw);
  auto route = SQLiteRoute::Create (
      *store, kEmbeddingDim, entries, *hnsw, parameters);
  REQUIRE (route);
  const Eigen::VectorXf observer_query = UnitVec (2);
  const auto canonical_observer = route->Search (observer_query, 16);
  REQUIRE (canonical_observer);
  REQUIRE_FALSE (canonical_observer->empty ());
  const auto initial_activation = route->SearchActivated (observer_query);
  REQUIRE (initial_activation);
  REQUIRE (initial_activation->size ()
           == parameters.activation_identity_target);

  REQUIRE (route->Recenter (UnitVec (0)));
  const long long first_entry = route->ActivationEntryMemoryId ();
  REQUIRE (first_entry > 0);
  REQUIRE (first_entry <= 700);
  const auto first_snapshot = route->ActivationIdentityIds ();
  REQUIRE (first_snapshot.size ()
           == parameters.activation_identity_target);
  REQUIRE (first_snapshot.front () == first_entry);
  const auto first_activation = route->SearchActivated (observer_query);
  REQUIRE (first_activation);
  REQUIRE (first_activation->size ()
           == parameters.activation_identity_target);
  REQUIRE (route->LastSearchNodeRows ()
           <= parameters.search_node_budget);

  route = SQLiteRoute::Open (*store, kEmbeddingDim, parameters);
  REQUIRE (route);
  REQUIRE (route->ActivationEntryMemoryId () == first_entry);
  REQUIRE (route->SearchActivated (observer_query) == first_activation);
  REQUIRE (route->LastActivationSnapshotCacheMissRows () > 0);
  REQUIRE (route->LastActivationSnapshotCacheMissRows ()
           <= parameters.activation_identity_target);
  REQUIRE (route->SearchActivated (observer_query) == first_activation);
  REQUIRE (route->LastActivationSnapshotCacheMissRows () == 0);
  REQUIRE (route->RestartRowsLoaded () == 1);

  REQUIRE (route->Recenter (UnitVec (1)));
  const long long second_entry = route->ActivationEntryMemoryId ();
  REQUIRE (second_entry > 700);
  REQUIRE (second_entry != first_entry);
  const auto second_snapshot = route->ActivationIdentityIds ();
  REQUIRE (second_snapshot.size ()
           == parameters.activation_identity_target);
  REQUIRE (second_snapshot.front () == second_entry);
  REQUIRE (second_snapshot != first_snapshot);
  const auto second_activation = route->SearchActivated (observer_query);
  REQUIRE (second_activation);
  REQUIRE (second_activation->size ()
           == parameters.activation_identity_target);
  REQUIRE (*second_activation != *first_activation);
  REQUIRE (second_activation->front () == canonical_observer->front ());
  const auto shifted_query = route->SearchActivated (UnitVec (0));
  REQUIRE (shifted_query);
  REQUIRE (shifted_query->front () == 1);
  REQUIRE (static_cast<std::size_t> (std::count_if (
               shifted_query->begin (), shifted_query->end (),
               [] (const long long memory_id) { return memory_id > 700; }))
           == parameters.reciprocal_update_count);
  REQUIRE (route->LastActivationSnapshotRows ()
           <= parameters.activation_identity_target);
  REQUIRE (route->LastSearchNodeRows ()
           <= parameters.search_node_budget);

  const auto metadata = store->Execute (
      "SELECT activation_entry_memory_id, activation_generation, "
      "length(activation_centroid) AS centroid_bytes, "
      "length(activation_identity_ids) AS identity_bytes "
      "FROM cortext_sparse_route_meta WHERE singleton = 1");
  REQUIRE (metadata.size () == 1);
  REQUIRE (std::any_cast<long long> (
               metadata[0].at ("activation_entry_memory_id"))
           == second_entry);
  REQUIRE (std::any_cast<long long> (
               metadata[0].at ("activation_generation"))
           == 1);
  REQUIRE (std::any_cast<long long> (metadata[0].at ("centroid_bytes"))
           == static_cast<long long> (kEmbeddingDim * sizeof (float)));
  REQUIRE (std::any_cast<long long> (metadata[0].at ("identity_bytes"))
           == static_cast<long long> (
               sizeof (std::uint32_t)
               + parameters.activation_identity_target
                     * sizeof (std::uint64_t)));

  route = SQLiteRoute::Open (*store, kEmbeddingDim, parameters);
  REQUIRE (route);
  REQUIRE (route->ActivationEntryMemoryId () == second_entry);
  REQUIRE (route->ActivationIdentityIds () == second_snapshot);
  REQUIRE (route->SearchActivated (observer_query) == second_activation);
  REQUIRE (route->RestartRowsLoaded () == 1);
}

Eigen::VectorXf
VectorWithCosineToDim0 (float cosine)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = cosine;
  v[1] = std::sqrt (std::max (0.0f, 1.0f - cosine * cosine));
  return v;
}

Eigen::VectorXf
VectorWithCosineAndDiverseResidual (float cosine, std::uint64_t seed)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = cosine;
  const float residual_scale
      = std::sqrt (std::max (0.0f, 1.0f - cosine * cosine))
        / std::sqrt (static_cast<float> (kEmbeddingDim - 1));
  std::uint64_t state = seed + 0x9e3779b97f4a7c15ULL;
  for (int dimension = 1; dimension < kEmbeddingDim; ++dimension)
    {
      state += 0x9e3779b97f4a7c15ULL;
      std::uint64_t mixed = state;
      mixed = (mixed ^ (mixed >> 30U)) * 0xbf58476d1ce4e5b9ULL;
      mixed = (mixed ^ (mixed >> 27U)) * 0x94d049bb133111ebULL;
      mixed ^= mixed >> 31U;
      v[dimension] = (mixed & 1ULL) != 0 ? residual_scale
                                         : -residual_scale;
    }
  return v;
}

Eigen::VectorXf
HadamardRow (unsigned int row)
{
  Eigen::VectorXf v (kEmbeddingDim);
  const float scale = 1.0f / std::sqrt (static_cast<float> (kEmbeddingDim));
  for (unsigned int column = 0;
       column < static_cast<unsigned int> (kEmbeddingDim); ++column)
    {
      v[static_cast<Eigen::Index> (column)]
          = (std::popcount (row & column) % 2 == 0) ? scale : -scale;
    }
  return v;
}

std::pair<Eigen::VectorXf, Eigen::VectorXf>
NearDuplicateFamilyPair ()
{
  Eigen::VectorXf first = Eigen::VectorXf::Zero (kEmbeddingDim);
  Eigen::VectorXf second = Eigen::VectorXf::Zero (kEmbeddingDim);
  for (int dimension = 0; dimension < 7; ++dimension)
    {
      first[dimension] = 1.0f;
      second[dimension] = 1.0f;
    }
  first[7] = 1.0f;
  first[8] = 0.9999f;
  second[7] = 0.9999f;
  second[8] = 1.0f;
  first.normalize ();
  second.normalize ();
  return { first, second };
}

Eigen::VectorXf
VectorWithCosineToQuery (const Eigen::VectorXf &query, float cosine)
{
  Eigen::VectorXf orthogonal = UnitVec (kEmbeddingDim - 1);
  Eigen::VectorXf result
      = cosine * query
        + std::sqrt (std::max (0.0f, 1.0f - cosine * cosine)) * orthogonal;
  result.normalize ();
  return result;
}

std::vector<float>
ToFloatVec (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

Signal
MakeSignal (const Eigen::VectorXf &embedding, std::uint64_t ts)
{
  Signal s;
  s.embedding = embedding;
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

class ForceRetrievalGateOp : public IOperation
{
public:
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetShouldCheckRetrieval (true);
    ctx.SetWriteExclusionTs (ctx.GetSignal ().timestamp);
  }
};

class CaptureReloadedRetrievalSurfaceOp : public IOperation
{
public:
  explicit CaptureReloadedRetrievalSurfaceOp (bool &found) : found_ (found) {}

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &p_ctx = ctx.GetProcessorContext ();
    auto anchor_it = p_ctx.retrieval_surface_index.find (10);
    auto neighbor_it = p_ctx.retrieval_surface_index.find (20);
    if (anchor_it == p_ctx.retrieval_surface_index.end ()
        || neighbor_it == p_ctx.retrieval_surface_index.end ())
      {
        found_ = false;
        return;
      }
    const auto &anchor = p_ctx.retrieval_surface_cache[anchor_it->second];
    const auto &neighbor = p_ctx.retrieval_surface_cache[neighbor_it->second];
    auto source_it
        = p_ctx.retrieval_surface_source_index.find ("conversation/reload");
    found_ = anchor.embedding_id == 100 && neighbor.embedding_id == 200
             && anchor.embedding.size () == kEmbeddingDim
             && neighbor.embedding.size () == kEmbeddingDim
             && source_it != p_ctx.retrieval_surface_source_index.end ()
             && source_it->second.size () == 2;
  }

private:
  bool &found_;
};

void
SeedMemory (Store &store, long long memory_id, long long embedding_id,
            const Eigen::VectorXf &embedding, std::uint64_t ts,
            const std::string &source_id = "test")
{
  store.Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { embedding_id, ToFloatVec (embedding), static_cast<long long> (ts) });
  store.Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, start_ts, "
      "created_at) VALUES(?, ?, ?, 'LONG_TERM', ?, ?)",
      { memory_id, embedding_id, source_id, static_cast<long long> (ts),
        static_cast<long long> (ts) });
}
} // namespace

TEST_CASE ("Graph retrieval returns nearest retained memory",
           "[operations][graph][retrieval]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  SeedMemory (*store, 20, 200, UnitVec (1), 1000);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
  const auto selected
      = operations::retrieval_trace::GetLastSelectedEmbeddingOrder ();
  const auto seeds = operations::retrieval_trace::GetLastSeedCandidates ();
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 10)
           != out.candidate_memory_ids.end ());
  REQUIRE_FALSE (selected.empty ());
  REQUIRE_FALSE (seeds.empty ());
  REQUIRE (seeds.front ().memory_id == 10LL);
  REQUIRE (seeds.front ().embedding_id > 0);
  REQUIRE (seeds.front ().score >= seeds.back ().score);
  auto current_rows = store->Execute (
      "SELECT embedding_id FROM current_memory_embeddings "
      "WHERE memory_id = ?",
      { 10LL });
  REQUIRE (current_rows.size () == 1);
  REQUIRE (selected.front ()
           == std::any_cast<long long> (current_rows[0].at ("embedding_id")));
}

TEST_CASE ("Graph retrieval expands through retained associations",
           "[operations][graph][retrieval]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  SeedMemory (*store, 20, 200, UnitVec (1), 1000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'co_occurs', ?, ?)",
      { 10LL, 20LL, 0.95, 1000LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 20)
           != out.candidate_memory_ids.end ());
}

TEST_CASE ("Graph retrieval reaches derived associations without direct seeding",
           "[operations][graph][retrieval][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (1), 1000);
  SeedMemory (*store, 20, 200, UnitVec (0), 1000);
  store->Execute (
      "UPDATE memories SET kind = 'ASSOCIATION' WHERE memory_id = ?",
      { 20LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto make_ops = [] {
    return std::make_unique<DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
  };

  {
    SignalProcessor processor (cfg, store, make_ops ());
    const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
    REQUIRE (std::find (out.candidate_memory_ids.begin (),
                        out.candidate_memory_ids.end (), 20)
             == out.candidate_memory_ids.end ());
  }

  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'derived_from', ?, ?)",
      { 20LL, 10LL, 0.95, 1000LL });
  {
    SignalProcessor processor (cfg, store, make_ops ());
    const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
    REQUIRE (std::find (out.candidate_memory_ids.begin (),
                        out.candidate_memory_ids.end (), 20)
             != out.candidate_memory_ids.end ());
  }
}

TEST_CASE ("Graph retrieval protects direct family before derived centroid",
           "[operations][graph][retrieval][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  SeedMemory (*store, 20, 200, UnitVec (0), 1000);
  store->Execute (
      "UPDATE memories SET kind = 'ASSOCIATION' WHERE memory_id = ?",
      { 20LL });
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'derived_from', ?, ?)",
      { 20LL, 10LL, 1.0, 1000LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 10)
           != out.candidate_memory_ids.end ());
}

TEST_CASE ("Graph retrieval expands through sequential episode edges",
           "[operations][graph][retrieval]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  SeedMemory (*store, 20, 200, UnitVec (1), 1100);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'next_in_episode', ?, ?)",
      { 10LL, 20LL, 0.95, 1100LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 20)
           != out.candidate_memory_ids.end ());
}

TEST_CASE ("Graph retrieval demotes superseded stale memories",
           "[operations][graph][retrieval][supersession][eval]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  SeedMemory (*store, 20, 200, VectorWithCosineToDim0 (0.92f), 2000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'supersedes', ?, ?)",
      { 20LL, 10LL, 1.0, 2000LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 3000));
  const auto ranked = operations::retrieval_trace::GetLastRankedCandidates ();
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE_FALSE (ranked.empty ());
  REQUIRE (ranked.front ().memory_id == 20LL);

  auto correction_it = std::find_if (
      ranked.begin (), ranked.end (),
      [] (const auto &candidate) { return candidate.memory_id == 20LL; });
  REQUIRE (correction_it != ranked.end ());
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 10LL)
           == out.candidate_memory_ids.end ());
}

TEST_CASE ("Graph retrieval scores older exact matches beyond old recency cap",
           "[operations][graph][retrieval]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 1, 100, UnitVec (0), 1000);
  for (long long i = 0; i < 450; ++i)
    {
      SeedMemory (*store, 1000 + i, 10000 + i, UnitVec (1),
                  1000);
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 100000));
  const auto selected
      = operations::retrieval_trace::GetLastSelectedEmbeddingOrder ();
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE_FALSE (selected.empty ());
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 1)
           != out.candidate_memory_ids.end ());
  REQUIRE (out.candidate_memory_ids.front () == 1LL);
}

TEST_CASE ("Graph retrieval keeps historical memory before its replacement time",
           "[operations][graph][retrieval][supersession]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  SeedMemory (*store, 20, 200, VectorWithCosineToDim0 (0.92f), 5000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'supersedes', ?, ?)",
      { 20LL, 10LL, 1.0, 5000LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 3000));
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE (out.candidate_memory_ids.front () == 10LL);

  const auto at_replacement
      = processor.Process (MakeSignal (UnitVec (0), 5000));
  REQUIRE (std::find (at_replacement.candidate_memory_ids.begin (),
                      at_replacement.candidate_memory_ids.end (), 10LL)
           != at_replacement.candidate_memory_ids.end ());

  const auto after_replacement
      = processor.Process (MakeSignal (UnitVec (0), 5001));
  REQUIRE (std::find (after_replacement.candidate_memory_ids.begin (),
                      after_replacement.candidate_memory_ids.end (), 10LL)
           == after_replacement.candidate_memory_ids.end ());
}

TEST_CASE ("Family features are cached and replaced with their embedding",
           "[operations][graph][retrieval][cache][family]")
{
  namespace surface_cache
      = operations::historical_surface_search_cache_internal;
  ProcessorContext pctx;
  REQUIRE (surface_cache::Reset (
      pctx, {}, { surface_cache::Entry { 100, 10, 1000, "LONG_TERM",
                                        "source-a", UnitVec (0), 100 } }));

  auto state = surface_cache::Find (pctx);
  REQUIRE (state);
  REQUIRE_FALSE (state->current_entries[0].family_features.has_value ());
  const auto &first
      = surface_cache::FamilyFeatures (state->current_entries[0]);
  REQUIRE (first.normalized[0] == Catch::Approx (1.0f));
  REQUIRE (state->current_entries[0].family_features.has_value ());
  const auto *cached = &*state->current_entries[0].family_features;
  REQUIRE (&surface_cache::FamilyFeatures (state->current_entries[0])
           == cached);

  surface_cache::UpsertCurrent (
      pctx, { 200, 10, 1001, "LONG_TERM", "source-b", UnitVec (1), 100 });
  state = surface_cache::Find (pctx);
  REQUIRE (state);
  REQUIRE_FALSE (state->current_entries[0].family_features.has_value ());
  const auto &replacement
      = surface_cache::FamilyFeatures (state->current_entries[0]);
  REQUIRE (replacement.normalized[0] == Catch::Approx (0.0f));
  REQUIRE (replacement.normalized[1] == Catch::Approx (1.0f));
  surface_cache::Erase (pctx);
}

TEST_CASE ("Supersession eligibility index updates only affected targets",
           "[operations][graph][retrieval][supersession][cache][scaling]")
{
  ProcessorContext pctx;
  pctx.UpsertRetrievalSurface (
      { 20, 200, 5000, 5000, 0, 0, 0, 0, "LONG_TERM", "replacement",
        "text", -1.0, 0, 0.0, 0.0, 0.0, false, true, UnitVec (0) });
  auto &cache = pctx.association_fanout_cache;
  cache.valid = true;
  cache.out_by_source[20].push_back (
      { 10, 100, "supersedes", 1.0, 5000 });
  cache.in_by_target[10].push_back (
      { 20, 200, "supersedes", 1.0, 5000 });
  operations::association_fanout_cache::BuildSupersessionEligibility (
      cache, pctx);
  auto sidecar = operations::execution_cache_sidecar_internal::Find (pctx);
  REQUIRE (sidecar);
  REQUIRE (sidecar->supersession_eligibility.valid);
  REQUIRE (sidecar->supersession_eligibility.activation_ts_by_target.at (10)
           == 5000);

  pctx.UpsertRetrievalSurface (
      { 30, 300, 6000, 6000, 0, 0, 0, 0, "LONG_TERM", "unrelated",
        "audio", -1.0, 0, 0.0, 0.0, 0.0, false, true, UnitVec (1) });
  REQUIRE (sidecar->supersession_eligibility.valid);

  pctx.UpsertRetrievalSurface (
      { 20, 200, 7000, 7000, 0, 0, 0, 0, "LONG_TERM", "replacement",
        "image", -1.0, 0, 0.0, 0.0, 0.0, false, true, UnitVec (0) });
  operations::association_fanout_cache::NotifyRetrievalSurfaceChanged (
      pctx, 20);
  REQUIRE (sidecar->supersession_eligibility.valid);
  REQUIRE (sidecar->supersession_eligibility.activation_ts_by_target.at (10)
           == 7000);
}

TEST_CASE ("Graph retrieval temporal score decays across multi-month ages",
           "[operations][graph][retrieval][temporal]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  constexpr long long kDayMs = 24LL * 60LL * 60LL * 1000LL;
  constexpr std::uint64_t now = 200ULL * static_cast<std::uint64_t> (kDayMs);
  SeedMemory (*store, 10, 100, UnitVec (0), now - 60ULL * 1000ULL);
  SeedMemory (*store, 20, 200, UnitVec (1), now - 30ULL * kDayMs);
  SeedMemory (*store, 30, 300, UnitVec (2), now - 90ULL * kDayMs);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  operations::retrieval_trace::ClearLastRankedCandidates ();
  const auto out = processor.Process (MakeSignal (UnitVec (0), now));
  const auto ranked = operations::retrieval_trace::GetLastRankedCandidates ();

  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  std::unordered_map<long long, double> temporal_by_memory;
  for (const auto &candidate : ranked)
    {
      temporal_by_memory.emplace (candidate.memory_id,
                                  candidate.temporal_score);
    }

  REQUIRE (temporal_by_memory.count (10) == 1);
  REQUIRE (temporal_by_memory.count (20) == 1);
  REQUIRE (temporal_by_memory.count (30) == 1);
  REQUIRE (temporal_by_memory.at (10) <= 1.0);
  REQUIRE (temporal_by_memory.at (10) > temporal_by_memory.at (20));
  REQUIRE (temporal_by_memory.at (20) > temporal_by_memory.at (30));
  REQUIRE (temporal_by_memory.at (30) >= 0.0);
  REQUIRE (temporal_by_memory.at (30) < 0.05);
}

TEST_CASE ("Graph retrieval KNN finds older exact matches outside stride buckets",
           "[operations][graph][retrieval]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  constexpr long long target_memory_id = 504;
  constexpr long long target_embedding_id = 90504;
  for (long long i = 1; i <= 1800; ++i)
    {
      if (i == target_memory_id)
        {
          continue;
        }
      SeedMemory (*store, i, 10000 + i, UnitVec (1), 1000);
    }
  SeedMemory (*store, target_memory_id, target_embedding_id, UnitVec (0),
              1000);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 100000));
  const auto selected
      = operations::retrieval_trace::GetLastSelectedEmbeddingOrder ();
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE_FALSE (selected.empty ());
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), target_memory_id)
           != out.candidate_memory_ids.end ());
}

TEST_CASE ("Graph retrieval KNN searches current memory surface",
           "[operations][graph][retrieval]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  constexpr long long target_memory_id = 504;
  constexpr long long base_embedding_id = 90504;
  constexpr long long current_embedding_id = 99504;
  for (long long i = 1; i <= 1800; ++i)
    {
      if (i == target_memory_id)
        {
          continue;
        }
      SeedMemory (*store, i, 10000 + i, UnitVec (1), 1000);
    }
  SeedMemory (*store, target_memory_id, base_embedding_id, UnitVec (2),
              1000);
  cortext::testing::SeedEmbeddingV2 (*store, current_embedding_id,
                                     ToFloatVec (UnitVec (0)), 2000);
  cortext::testing::SeedCurrentMemoryEmbeddingV2 (*store, target_memory_id,
                                                  current_embedding_id);
  auto current_rows = store->Execute (
      "SELECT memory_id, embedding_id FROM current_memory_embeddings "
      "WHERE embedding MATCH ? AND k = ? "
      "ORDER BY distance",
      { ToFloatVec (UnitVec (0)), 5LL });
  REQUIRE (std::find_if (
               current_rows.begin (), current_rows.end (),
               [&] (const auto &row) {
                 auto it = row.find ("memory_id");
                 return it != row.end ()
                        && std::any_cast<long long> (it->second)
                               == target_memory_id;
               })
           != current_rows.end ());
  auto joined_rows = store->Execute (
      "SELECT m.memory_id, cme.embedding_id, m.start_ts, cme.embedding, "
      "       m.source_id "
      "FROM ("
      "  SELECT memory_id, embedding_id, embedding, distance "
      "  FROM current_memory_embeddings "
      "  WHERE embedding MATCH ? "
      "    AND k = ?"
      ") cme "
      "JOIN memories m ON m.memory_id = cme.memory_id "
      "WHERE m.kind IN ('LONG_TERM', 'ASSOCIATION') "
      "  AND COALESCE(m.start_ts, 0) < ? "
      "ORDER BY distance",
      { ToFloatVec (UnitVec (0)), 5LL, 100000LL });
  REQUIRE (std::find_if (
               joined_rows.begin (), joined_rows.end (),
               [&] (const auto &row) {
                 auto it = row.find ("memory_id");
                 return it != row.end ()
                        && std::any_cast<long long> (it->second)
                               == target_memory_id;
               })
           != joined_rows.end ());

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 100000));
  const auto selected
      = operations::retrieval_trace::GetLastSelectedEmbeddingOrder ();
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE_FALSE (selected.empty ());
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), target_memory_id)
           != out.candidate_memory_ids.end ());
  REQUIRE (std::find (selected.begin (), selected.end (), current_embedding_id)
           != selected.end ());
}

TEST_CASE ("Graph retrieval reloads latest reconstruction when current writes "
           "are disabled",
           "[operations][graph][retrieval][constructive_recall][restart]")
{
  cortext::testing::ScopedEnvVar disable_current (
      "CORTEXT_DISABLE_CURRENT_MEMORY_SURFACE_WRITES", "1");
  if (!operations::constructive_recall::CurrentSurfaceWritesDisabled ())
    {
      SKIP ("current-surface write hook is disabled in this build");
    }
  cortext::testing::ScopedEnvVar reconstruction_interval (
      "CORTEXT_RECONSTRUCTION_MIN_UPDATE_MS", "999999999");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (2), 1000);
  SeedMemory (*store, 20, 200, VectorWithCosineToDim0 (0.90f), 1000);
  auto reconstruction_tx = store->Begin ();
  const long long reconstruction_id
      = operations::constructive_recall::AppendReconstructionWithEmbedding (
          *reconstruction_tx, 10, UnitVec (0), {}, 1500, 0.1, "retrieval",
          1.0, 1.0);
  REQUIRE (reconstruction_id > 0);
  reconstruction_tx->Commit ();
  const auto latest_rows = store->Execute (
      "SELECT embedding_id FROM memory_reconstructions "
      "WHERE reconstruction_id = ?",
      { reconstruction_id });
  REQUIRE (latest_rows.size () == 1);
  const long long latest_embedding_id = cortext::store::AnyToLongLong (
      latest_rows[0].at ("embedding_id")).value_or (0);
  REQUIRE (latest_embedding_id > 0);
  REQUIRE (store->Execute (
               "SELECT 1 FROM current_memory_embeddings WHERE memory_id = ?",
               { 10LL })
               .empty ());

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  // Construction is the restart boundary: it reloads processor surfaces from
  // the durable base/current/reconstruction tables.
  SignalProcessor processor (cfg, store, std::move (ops));

  operations::retrieval_trace::ClearLastRankedCandidates ();
  const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
  const auto ranked = operations::retrieval_trace::GetLastRankedCandidates ();
  const auto target = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 10LL;
      });
  const auto competitor = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 20LL;
      });
  REQUIRE (target != ranked.end ());
  REQUIRE (competitor != ranked.end ());
  REQUIRE (target->embedding_id == latest_embedding_id);
  REQUIRE (target->score > competitor->score);
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE (out.candidate_memory_ids.front () == 10LL);
}

TEST_CASE ("Graph retrieval reloads base embedding when constructive recall is "
           "disabled",
           "[operations][graph][retrieval][constructive_recall][restart][ablation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 101, UnitVec (1), 1250);
  SeedMemory (*store, 20, 200, VectorWithCosineToDim0 (0.90f), 1000);
  auto reconstruction_tx = store->Begin ();
  const long long reconstruction_id
      = operations::constructive_recall::AppendReconstructionWithEmbedding (
          *reconstruction_tx, 10, UnitVec (2), {}, 1500, 0.1, "retrieval",
          1.0, 1.0);
  REQUIRE (reconstruction_id > 0);
  reconstruction_tx->Commit ();
  cortext::testing::SeedCurrentMemoryEmbeddingV2 (*store, 10, 101);

  const auto latest_rows = store->Execute (
      "SELECT embedding_id FROM memory_reconstructions "
      "WHERE reconstruction_id = ?",
      { reconstruction_id });
  REQUIRE (latest_rows.size () == 1);
  const long long latest_embedding_id = cortext::store::AnyToLongLong (
      latest_rows[0].at ("embedding_id")).value_or (0);
  REQUIRE (latest_embedding_id > 0);
  REQUIRE (latest_embedding_id != 100LL);
  REQUIRE (latest_embedding_id != 101LL);

  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  if (!operations::constructive_recall::Disabled ())
    {
      SKIP ("constructive-recall disable hook is disabled in this build");
    }
  const auto authoritative
      = operations::constructive_recall::LoadCurrentEmbedding (
          store.get (), 10, 100, kEmbeddingDim);
  REQUIRE (authoritative.has_value ());
  REQUIRE ((*authoritative - UnitVec (0)).norm () < 1e-5f);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  operations::retrieval_trace::ClearLastRankedCandidates ();
  const auto out = processor.Process (MakeSignal (UnitVec (0), 2000));
  const auto ranked = operations::retrieval_trace::GetLastRankedCandidates ();
  const auto target = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 10LL;
      });
  const auto competitor = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 20LL;
      });
  REQUIRE (target != ranked.end ());
  REQUIRE (competitor != ranked.end ());
  REQUIRE (target->embedding_id == 100LL);
  REQUIRE (target->score > competitor->score);
  REQUIRE_FALSE (out.candidate_memory_ids.empty ());
  REQUIRE (out.candidate_memory_ids.front () == 10LL);
}

TEST_CASE ("Graph retrieval cache rebuild uses base embedding when constructive "
           "recall is disabled",
           "[operations][graph][retrieval][constructive_recall][cache_rebuild]"
           "[ablation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 101, UnitVec (1), 1250);
  auto reconstruction_tx = store->Begin ();
  const long long reconstruction_id
      = operations::constructive_recall::AppendReconstructionWithEmbedding (
          *reconstruction_tx, 10, UnitVec (2), {}, 1500, 0.1, "retrieval",
          1.0, 1.0);
  REQUIRE (reconstruction_id > 0);
  reconstruction_tx->Commit ();
  cortext::testing::SeedCurrentMemoryEmbeddingV2 (*store, 10, 101);

  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  if (!operations::constructive_recall::Disabled ())
    {
      SKIP ("constructive-recall disable hook is disabled in this build");
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  pctx.UpsertRetrievalSurface (
      { 10, 100, 1000, 1000, 0, 0, 0, 0, "LONG_TERM", "test", "", -1.0,
        0, 0.0, 0.0, 0.0, false, true, UnitVec (0) });
  auto signal = MakeSignal (UnitVec (0), 2000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);
  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  const auto ranked = operations::retrieval_trace::GetLastRankedCandidates ();
  const auto target = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 10LL;
      });
  REQUIRE (target != ranked.end ());
  REQUIRE (target->embedding_id == 100LL);
  REQUIRE (target->score > 0.99);
  REQUIRE (operations::historical_surface_search_cache_internal::
               BaseEmbeddingIdForMemory (pctx, 10LL, 0)
           == 100LL);
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Current surface upsert preserves original base embedding lineage",
           "[operations][graph][retrieval][constructive_recall][cache]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  ProcessorContext pctx;
  const Eigen::VectorXf base = UnitVec (0);
  const Eigen::VectorXf reconstructed = UnitVec (1);
  REQUIRE (cache::Reset (
      pctx, { { 100LL, 10LL, 1000LL, "LONG_TERM", "source", base } },
      { { 100LL, 10LL, 0, std::string (), std::string (), base, 100LL } }));
  cache::UpsertCurrent (
      pctx, { 101LL, 10LL, 0, std::string (), std::string (), reconstructed });
  REQUIRE (cache::BaseEmbeddingIdForMemory (pctx, 10LL, 0) == 100LL);
  const auto state = cache::Find (pctx);
  REQUIRE (state);
  REQUIRE (state->current_entries.at (
               state->current_memory_index.at (10LL)).embedding_id
           == 101LL);
  cache::Erase (pctx);
}

TEST_CASE ("Graph retrieval returns refreshed output after reconstruction",
           "[operations][graph][retrieval]")
{
  cortext::testing::ScopedEnvVar enable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL");
  cortext::testing::ScopedEnvVar no_reconstruction_interval (
      "CORTEXT_RECONSTRUCTION_MIN_UPDATE_MS", "0");

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = UnitVec (0);
  Eigen::VectorXf memory = VectorWithCosineToDim0 (0.20f);
  SeedMemory (*store, 10, 100, memory, 1000);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  auto signal = MakeSignal (query, 2'000'000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (2'000'000);

  GraphAugmentedRetrieveCandidates op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto recon_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memory_reconstructions "
      "WHERE memory_id = ?",
      { 10LL });
  REQUIRE (recon_rows.size () == 1);
  REQUIRE (std::any_cast<long long> (recon_rows[0].at ("cnt")) > 0);
  auto latest_rows = store->Execute (
      "SELECT mr.embedding_id, e.embedding FROM memory_reconstructions mr "
      "JOIN embeddings e ON e.embedding_id = mr.embedding_id "
      "WHERE mr.memory_id = ? "
      "ORDER BY mr.reconstruction_id DESC LIMIT 1",
      { 10LL });
  REQUIRE (latest_rows.size () == 1);
  const long long latest_embedding_id
      = std::any_cast<long long> (latest_rows[0].at ("embedding_id"));
  REQUIRE (latest_embedding_id != 100LL);
  Eigen::VectorXf latest;
  REQUIRE (cortext::core::DecodeFloatBlob (
      latest_rows[0].at ("embedding"), kEmbeddingDim, latest));
  auto current_rows = store->Execute (
      "SELECT embedding_id, embedding FROM current_memory_embeddings "
      "WHERE memory_id = ?",
      { 10LL });
  REQUIRE (current_rows.size () == 1);
  const long long current_embedding_id
      = std::any_cast<long long> (current_rows[0].at ("embedding_id"));
  REQUIRE (current_embedding_id == latest_embedding_id);
  Eigen::VectorXf current;
  REQUIRE (cortext::core::DecodeFloatBlob (
      current_rows[0].at ("embedding"), kEmbeddingDim, current));
  REQUIRE ((current - latest).norm () < 1e-6f);

  const auto &candidates = ctx.GetRetrievedMemoryCandidates ();
  REQUIRE_FALSE (candidates.empty ());
  REQUIRE (candidates.front ().memory_id == 10LL);
  REQUIRE (candidates.front ().embedding_id == current_embedding_id);
  REQUIRE ((candidates.front ().embedding - current).norm () < 1e-6f);
}

TEST_CASE ("Graph retrieval reloads base memory surfaces without current rows",
           "[operations][graph][retrieval][source]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedMemory (*store, 10, 100, UnitVec (0), 1000, "conversation/reload");
  SeedMemory (*store, 20, 200, UnitVec (1), 1100, "conversation/reload");

  bool found = false;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<CaptureReloadedRetrievalSurfaceOp> (found));
  SignalProcessor processor (cfg, store, std::move (ops));

  (void)processor.Process (MakeSignal (UnitVec (0), 2000));
  REQUIRE (found);
}

TEST_CASE ("Graph retrieval queries supersession family representatives before KNN cap",
           "[operations][graph][retrieval][family]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  constexpr long long kTargetMemoryId = 1;
  SeedMemory (*store, kTargetMemoryId, 1,
              VectorWithCosineToDim0 (0.95f), 1000, "target");
  constexpr long long kFamilyBegin = 1000;
  constexpr long long kFamilySize = 430;
  const long long representative_id = kFamilyBegin + kFamilySize - 1;
  for (long long offset = 0; offset < kFamilySize; ++offset)
    {
      const long long memory_id = kFamilyBegin + offset;
      SeedMemory (*store, memory_id, 10000 + offset,
                  VectorWithCosineToDim0 (0.99f), 2000 + offset,
                  "duplicate/" + std::to_string (offset));
      if (memory_id != representative_id)
        {
          store->Execute (
              "INSERT INTO associations "
              "(source_memory_id, target_memory_id, edge_type, weight, "
              " last_reinforced) VALUES (?, ?, 'supersedes', 1.0, 0)",
              { representative_id, memory_id });
        }
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 100000));
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), kTargetMemoryId)
           != out.candidate_memory_ids.end ());
}

TEST_CASE ("Graph retrieval ranking is invariant to source and modality labels",
           "[operations][graph][retrieval][invariance]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto run = [] (bool shared_source, const std::string &modality,
                 bool sparse_route) {
    cortext::testing::ScopedEnvVar sparse_route_flag (
        "CORTEXT_HNSW_SPARSE_ROUTE", sparse_route ? "1" : "0");
    auto unique_store = SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<Store> (std::move (unique_store));
    cortext::testing::InitializeCoreSchema (*store);
    SeedMemory (*store, 10, 100, UnitVec (0), 1000, "anchor");
    SeedMemory (*store, 20, 200, UnitVec (2), 1100,
                shared_source ? "anchor" : "unrelated");
    store->Execute ("UPDATE memories SET modality = ?", { modality });
    for (long long i = 0; i < 20; ++i)
      {
        SeedMemory (*store, 1000 + i, 10000 + i,
                    VectorWithCosineToDim0 (0.10f + 0.01f * i), 1200 + i,
                    "distractor/" + std::to_string (i));
        store->Execute (
            "UPDATE memories SET modality = ? WHERE memory_id = ?",
            { modality, 1000 + i });
      }

    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.focus = 1.0;
    cfg.sensitivity = 1.0;
    cfg.stability = 0.5;
    auto ops = std::make_unique<DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    (void)processor.Process (MakeSignal (UnitVec (0), 100000));
    const auto ranked = operations::retrieval_trace::GetLastRankedCandidates ();
    std::vector<std::pair<long long, double>> signature;
    signature.reserve (ranked.size ());
    for (const auto &candidate : ranked)
      {
        signature.emplace_back (candidate.memory_id, candidate.score);
      }
    return signature;
  };

  const auto shared_text = run (true, "text", false);
  const auto unique_audio = run (false, "audio", true);
  const auto unique_image = run (false, "image", true);
  REQUIRE (shared_text == unique_audio);
  REQUIRE (shared_text == unique_image);
}

TEST_CASE ("Graph retrieval captures same-event exact control for sparse route",
           "[operations][graph][retrieval][hnsw][control]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  cortext::testing::ScopedEnvVar sparse_route_flag (
      "CORTEXT_HNSW_SPARSE_ROUTE", "1");
  cortext::testing::ScopedEnvVar exact_control_flag (
      "CORTEXT_CAPTURE_HNSW_EXACT_CONTROL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  for (long long index = 0; index < 600; ++index)
    SeedMemory (*store, index + 1, 1000 + index,
                VectorWithCosineToDim0 (
                    0.10f + static_cast<float> (index) * 0.001f),
                1000 + index, "opaque/" + std::to_string (index % 4));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  (void)processor.Process (MakeSignal (UnitVec (0), 100000));
  const auto candidate
      = operations::retrieval_trace::GetLastSeedCandidates ();
  const auto exact
      = operations::retrieval_trace::GetLastExactSeedCandidates ();
  REQUIRE_FALSE (candidate.empty ());
  REQUIRE_FALSE (exact.empty ());
  REQUIRE (candidate.size () == exact.size ());
  for (std::size_t index = 0; index < exact.size (); ++index)
    {
      REQUIRE (candidate[index].memory_id == exact[index].memory_id);
      REQUIRE (candidate[index].score == exact[index].score);
    }
}

TEST_CASE ("Graph retrieval reopens SQLite sparse route on the existing path",
           "[operations][graph][retrieval][hnsw][sqlite][integration]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  cortext::testing::ScopedEnvVar hnsw_route_flag (
      "CORTEXT_HNSW_SPARSE_ROUTE", "0");
  cortext::testing::ScopedEnvVar sqlite_route_flag (
      "CORTEXT_SQLITE_SPARSE_ROUTE", "1");
  cortext::testing::ScopedEnvVar exact_control_flag (
      "CORTEXT_CAPTURE_HNSW_EXACT_CONTROL", "1");
  cortext::testing::ScopedEnvVar profile_flag (
      "CORTEXT_PROFILE_GRAPH_RETRIEVAL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  const auto parameters
      = operations::sparse_retrieval_route_sqlite_internal::
          DefaultParameters ();
  std::vector<std::pair<long long, Eigen::VectorXf>> route_entries;
  route_entries.reserve (600);
  for (long long index = 0; index < 600; ++index)
    {
      auto embedding = VectorWithCosineToDim0 (
          0.10f + static_cast<float> (index) * 0.001f);
      SeedMemory (*store, index + 1, 1000 + index, embedding,
                  1000 + index,
                  "opaque/" + std::to_string (index % 4));
      route_entries.emplace_back (index + 1, std::move (embedding));
    }
  auto hnsw = operations::sparse_retrieval_route_internal::Route::Create (
      kEmbeddingDim, route_entries,
      operations::sparse_retrieval_route_internal::DefaultParameters ());
  REQUIRE (hnsw != nullptr);
  auto sqlite_route = operations::sparse_retrieval_route_sqlite_internal::
      Route::Create (
          *store, kEmbeddingDim, route_entries, *hnsw, parameters);
  REQUIRE (sqlite_route != nullptr);

  const auto meta = store->Execute (
      "SELECT active_count FROM cortext_sparse_route_meta "
      "WHERE singleton = 1");
  REQUIRE (meta.size () == 1);
  REQUIRE (store::AnyToLongLong (meta.front ().at ("active_count"))
               .value_or (0)
           == 600);

  struct RouteRun
  {
    std::vector<long long> candidate_ids;
    double node_rows = 0.0;
    double activated_identities = 0.0;
    double restart_rows = 0.0;
  };
  const auto run = [&] (const std::string &modality,
                        const std::string &query_source,
                        bool shared_memory_source) {
    store->Execute (
        shared_memory_source
            ? "UPDATE memories SET modality = ?, source_id = 'shared/source'"
            : "UPDATE memories SET modality = ?, source_id = 'opaque/' || "
              "CAST(memory_id % 4 AS TEXT)",
        { modality });
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.focus = 0.5;
    cfg.sensitivity = 0.5;
    cfg.stability = 0.5;
    auto ops = std::make_unique<DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    auto signal = MakeSignal (UnitVec (0), 100000);
    signal.source_id = query_source;
    signal.modality = modality;
    const auto output = processor.Process (signal);
    const auto candidates
        = operations::retrieval_trace::GetLastSeedCandidates ();
    RouteRun result;
    for (const auto &candidate : candidates)
      result.candidate_ids.push_back (candidate.memory_id);
    result.node_rows = output.operation_ms.at (
        "GraphRetrieve.sqlite_sparse_route_node_rows");
    result.activated_identities = output.operation_ms.at (
        "GraphRetrieve.sqlite_sparse_route_activated_identities");
    result.restart_rows = output.operation_ms.at (
        "GraphRetrieve.sqlite_sparse_route_restart_rows");
    return result;
  };

  const auto shared_text = run ("text", "query/shared", true);
  const auto opaque_audio = run ("audio", "query/opaque/audio", false);
  const auto opaque_image = run ("image", "query/opaque/image", false);
  REQUIRE_FALSE (shared_text.candidate_ids.empty ());
  REQUIRE (shared_text.candidate_ids == opaque_audio.candidate_ids);
  REQUIRE (shared_text.candidate_ids == opaque_image.candidate_ids);
  for (const auto &result : { shared_text, opaque_audio, opaque_image })
    {
      INFO ("sqlite route rows=" << result.node_rows);
      REQUIRE (result.node_rows > 0.0);
      REQUIRE (result.node_rows
               <= static_cast<double> (parameters.search_node_budget));
      REQUIRE (result.activated_identities
               <= static_cast<double> (
                   parameters.activation_identity_target));
      REQUIRE (result.restart_rows == 1.0);
    }
  auto restarted = operations::sparse_retrieval_route_sqlite_internal::Route::
      Open (*store, kEmbeddingDim, parameters);
  REQUIRE (restarted != nullptr);
  REQUIRE (restarted->RestartRowsLoaded () == 1);
  REQUIRE (
      restarted->Search (UnitVec (0), parameters.route_capacity).has_value ());
  REQUIRE (restarted->LastSearchNodeRows ()
           <= parameters.search_node_budget);
}

TEST_CASE ("SQLite sparse route restages active mutations after restart",
           "[operations][graph][retrieval][hnsw][sqlite][restart]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  const auto parameters
      = operations::sparse_retrieval_route_sqlite_internal::
          DefaultParameters ();
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<std::pair<long long, Eigen::VectorXf>> route_entries;
  route_entries.reserve (600);
  for (long long memory_id = 1; memory_id <= 600; ++memory_id)
    route_entries.emplace_back (
        memory_id,
        VectorWithCosineToDim0 (
            0.10f + static_cast<float> (memory_id - 1) * 0.001f));
  auto hnsw = operations::sparse_retrieval_route_internal::Route::Create (
      kEmbeddingDim, route_entries,
      operations::sparse_retrieval_route_internal::DefaultParameters ());
  REQUIRE (hnsw);
  auto route = operations::sparse_retrieval_route_sqlite_internal::Route::
      Create (*store, kEmbeddingDim, route_entries, *hnsw,
              parameters);
  REQUIRE (route);

  const Eigen::VectorXf revised = UnitVec (5);
  REQUIRE (route->Upsert (1, revised));
  REQUIRE (route->Remove (2));
  REQUIRE (store->Execute (
               "SELECT singleton FROM cortext_sparse_route_meta")
               .size ()
           == 1);
  REQUIRE (store->Execute (
               "SELECT memory_id FROM cortext_sparse_route_dirty")
               .size ()
           == 2);

  cache::State state;
  state.embedding_dim = kEmbeddingDim;
  state.current_entries.reserve (599);
  for (const auto &[memory_id, embedding] : route_entries)
    {
      if (memory_id == 2)
        continue;
      const std::size_t index = state.current_entries.size ();
      state.current_entries.emplace_back (
          memory_id, memory_id, 1000 + memory_id, "LONG_TERM", "opaque",
          memory_id == 1 ? revised : embedding, memory_id);
      state.current_memory_index.emplace (memory_id, index);
      state.current_memory_order.emplace (memory_id, index);
    }

  route.reset ();
  route = cache::OpenSQLiteSparseRoute (state, *store);
  REQUIRE (route);
  REQUIRE (route->RestartRowsLoaded () == 1);
  REQUIRE (route->DeltaSize () == 0);
  REQUIRE (route->HasDirtyMemoryIds ());
  const auto staged_dirty
      = cache::StageSQLiteSparseRouteDirtyForSearch (state, *route);
  REQUIRE (staged_dirty == 2);
  REQUIRE (route->DeltaSize () == 2);
  const auto unsealed_revised_result
      = route->Search (revised, parameters.route_capacity);
  REQUIRE (unsealed_revised_result);
  REQUIRE_FALSE (unsealed_revised_result->empty ());
  REQUIRE (unsealed_revised_result->front () == 1);
  const auto unsealed_removed_result
      = route->Search (route_entries[1].second, parameters.route_capacity);
  REQUIRE (unsealed_removed_result);
  REQUIRE (std::find (unsealed_removed_result->begin (),
                      unsealed_removed_result->end (), 2)
           == unsealed_removed_result->end ());
  REQUIRE (cache::ReconcileSQLiteSparseRoute (
      state, *store, route,
      std::shared_ptr<
          operations::sparse_retrieval_route_internal::Route>{}));
  route = state.sqlite_sparse_route;
  REQUIRE (route);
  REQUIRE_FALSE (route->HasDirtyMemoryIds ());
  const auto revised_result
      = route->Search (revised, parameters.route_capacity);
  REQUIRE (revised_result);
  REQUIRE_FALSE (revised_result->empty ());
  REQUIRE (revised_result->front () == 1);
  const auto removed_result
      = route->Search (route_entries[1].second, parameters.route_capacity);
  REQUIRE (removed_result);
  REQUIRE (std::find (removed_result->begin (), removed_result->end (), 2)
           == removed_result->end ());

  REQUIRE (store->Execute (
               "SELECT memory_id FROM cortext_sparse_route_dirty")
               .empty ());
  const auto meta = store->Execute (
      "SELECT active_count FROM cortext_sparse_route_meta "
      "WHERE singleton = 1");
  REQUIRE (meta.size () == 1);
  REQUIRE (store::AnyToLongLong (meta.front ().at ("active_count"))
               .value_or (0)
           == 599);

  state.sqlite_sparse_route.reset ();
  route = cache::OpenSQLiteSparseRoute (state, *store);
  REQUIRE (route);
  REQUIRE (route->DeltaSize () == 0);
  const auto sealed_result
      = route->Search (revised, parameters.route_capacity);
  REQUIRE (sealed_result);
  REQUIRE_FALSE (sealed_result->empty ());
  REQUIRE (sealed_result->front () == 1);

  // Dirty reconciliation and historical backfill are separate derived
  // surfaces. Dirty search stages at most C identities and reads C+1 only to
  // detect journal overflow; historical backfill remains B and checks its
  // logical B+1 boundary from the post-B iterator without reading that row.
  const std::size_t overflow_count = parameters.route_capacity + 1;
  for (std::size_t offset = 0; offset < overflow_count; ++offset)
    {
      const long long memory_id = 3 + static_cast<long long> (offset);
      REQUIRE (route->Upsert (
          memory_id,
          route_entries[static_cast<std::size_t> (memory_id - 1)].second));
    }
  REQUIRE (store->Execute (
               "SELECT memory_id FROM cortext_sparse_route_dirty")
               .size ()
           == overflow_count);
  REQUIRE_FALSE (
      cache::StageSQLiteSparseRouteDirtyForSearch (state, *route));
  REQUIRE (route->DeltaSize () == 0);
}

TEST_CASE ("SQLite sparse route backfills a large existing surface in fixed "
           "unpublished batches",
           "[operations][graph][retrieval][hnsw][sqlite][backfill]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  const auto parameters
      = operations::sparse_retrieval_route_sqlite_internal::
          DefaultParameters ();
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);

  cache::State state;
  state.embedding_dim = kEmbeddingDim;
  constexpr long long kEntryCount = 1300;
  state.current_entries.reserve (kEntryCount);
  for (long long memory_id = 1; memory_id <= kEntryCount; ++memory_id)
    {
      Eigen::VectorXf embedding (kEmbeddingDim);
      for (Eigen::Index dimension = 0; dimension < embedding.size ();
           ++dimension)
        embedding[dimension] = std::sin (
            static_cast<float> (memory_id * (dimension + 1)) * 0.013f);
      embedding.normalize ();
      const std::size_t index = state.current_entries.size ();
      state.current_entries.emplace_back (
          memory_id, memory_id, 1000 + memory_id, "LONG_TERM", "opaque",
          std::move (embedding), memory_id);
      state.current_memory_index.emplace (memory_id, index);
      state.current_memory_order.emplace (memory_id, index);
    }

  REQUIRE (cache::AdvanceSQLiteSparseRouteBackfill (state, *store));
  REQUIRE_FALSE (state.sqlite_sparse_route);
  REQUIRE (state.sqlite_sparse_route_backfill);
  REQUIRE (state.sqlite_sparse_route_backfill->BuildCursor ()
           == static_cast<long long> (parameters.backfill_batch_size));
  REQUIRE (state.sqlite_sparse_route_backfill->ActiveCount ()
           == static_cast<long long> (parameters.backfill_batch_size));
  REQUIRE (store->Execute (
               "SELECT singleton FROM cortext_sparse_route_meta")
               .empty ());
  REQUIRE (store->Execute (
               "SELECT singleton FROM cortext_sparse_route_build")
               .size ()
           == 1);
  REQUIRE (store->Execute (
               "SELECT memory_id FROM cortext_sparse_route_nodes")
               .size ()
           <= operations::sparse_retrieval_route_sqlite_internal::
                  DefaultParameters ()
                      .backfill_batch_size);

  // Construction progress is durable, but the incomplete generation remains
  // invisible to the normal retrieval path after restart.
  state.sqlite_sparse_route_backfill.reset ();
  REQUIRE_FALSE (
      operations::sparse_retrieval_route_sqlite_internal::Route::Open (
          *store, kEmbeddingDim, parameters));
  state.sqlite_sparse_route_backfill
      = operations::sparse_retrieval_route_sqlite_internal::Route::
          OpenOrBeginBuild (*store, kEmbeddingDim, parameters);
  REQUIRE (state.sqlite_sparse_route_backfill);
  REQUIRE (state.sqlite_sparse_route_backfill->BuildCursor ()
           == static_cast<long long> (parameters.backfill_batch_size));

  Eigen::VectorXf revised = Eigen::VectorXf::Zero (kEmbeddingDim);
  revised[0] = 1.0f;
  state.current_entries.front ().embedding = revised;
  REQUIRE (state.sqlite_sparse_route_backfill->Upsert (1, revised));
  REQUIRE (store->Execute (
               "SELECT singleton FROM cortext_sparse_route_build")
               .size ()
           == 1);
  REQUIRE (store->Execute (
               "SELECT memory_id FROM cortext_sparse_route_dirty")
               .size ()
           == 1);
  state.sqlite_sparse_route_backfill.reset ();
  state.sqlite_sparse_route_backfill
      = operations::sparse_retrieval_route_sqlite_internal::Route::
          OpenOrBeginBuild (*store, kEmbeddingDim, parameters);
  REQUIRE (state.sqlite_sparse_route_backfill);
  REQUIRE (state.sqlite_sparse_route_backfill->BuildCursor ()
           == static_cast<long long> (parameters.backfill_batch_size));
  REQUIRE (cache::AdvanceSQLiteSparseRouteBackfill (state, *store));
  REQUIRE (state.sqlite_sparse_route_backfill->BuildCursor ()
           == static_cast<long long> (parameters.backfill_batch_size * 2));
  REQUIRE (store->Execute (
               "SELECT singleton FROM cortext_sparse_route_build")
               .size ()
           == 1);
  REQUIRE (store->Execute (
               "SELECT memory_id FROM cortext_sparse_route_dirty")
               .empty ());

  std::size_t advances = 2;
  while (!state.sqlite_sparse_route && advances++ < 20)
    REQUIRE (cache::AdvanceSQLiteSparseRouteBackfill (state, *store));
  REQUIRE (state.sqlite_sparse_route);
  REQUIRE_FALSE (state.sqlite_sparse_route_backfill);
  REQUIRE (state.sqlite_sparse_route->ActiveCount () == kEntryCount);
  REQUIRE (store->Execute (
               "SELECT singleton FROM cortext_sparse_route_build")
               .empty ());
  const auto topology = store->Execute (
      "SELECT MAX(level) AS max_level, "
      "SUM(CASE WHEN level > 0 THEN 1 ELSE 0 END) AS upper_nodes "
      "FROM cortext_sparse_route_nodes WHERE active = 1");
  REQUIRE (topology.size () == 1);
  REQUIRE (store::AnyToLongLong (topology.front ().at ("max_level"))
               .value_or (0)
           > 0);
  REQUIRE (store::AnyToLongLong (topology.front ().at ("upper_nodes"))
               .value_or (0)
           > 0);

  auto restarted
      = operations::sparse_retrieval_route_sqlite_internal::Route::Open (
          *store, kEmbeddingDim, parameters);
  REQUIRE (restarted);
  REQUIRE (restarted->RestartRowsLoaded () == 1);
  const auto revised_result
      = restarted->Search (revised, parameters.route_capacity);
  REQUIRE (revised_result);
  REQUIRE_FALSE (revised_result->empty ());

  const auto activated = restarted->SearchActivated (revised);
  REQUIRE (activated);
  REQUIRE (activated->size () >= revised_result->size ());
  REQUIRE (activated->size () == parameters.activation_identity_target);
  REQUIRE (activated->size ()
           <= parameters.activation_identity_target);
  REQUIRE (restarted->LastSearchNodeRows () >= activated->size ());
  REQUIRE (restarted->LastSearchNodeRows ()
           <= parameters.search_node_budget);
  REQUIRE (restarted->LastSearchDistanceEvaluations () > 0);
  REQUIRE (restarted->LastSearchDistanceEvaluations ()
           <= restarted->LastSearchNodeRows () + restarted->DeltaSize ());
  REQUIRE (revised_result->front () == 1);
  for (const long long memory_id : { 1LL, 129LL, 777LL, 1300LL })
    {
      const auto result = restarted->Search (
          state.current_entries[static_cast<std::size_t> (memory_id - 1)]
              .embedding,
          parameters.route_capacity);
      REQUIRE (result);
      REQUIRE_FALSE (result->empty ());
      REQUIRE (result->front () == memory_id);
      REQUIRE (restarted->LastSearchNodeRows ()
               <= parameters.search_node_budget);
    }
}

TEST_CASE ("SQLite sparse route starts backfill at the knob-derived headroom "
           "watermark",
           "[operations][graph][retrieval][hnsw][sqlite][backfill][knobs]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  const auto parameters
      = operations::sparse_retrieval_route_sqlite_internal::
          DefaultParameters ();
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);

  cache::State state;
  state.embedding_dim = kEmbeddingDim;
  state.sparse_route_parameters = parameters;
  const std::size_t start_size
      = parameters.route_capacity - parameters.backfill_batch_size;
  auto append_entry = [&] (long long memory_id) {
    Eigen::VectorXf embedding = Eigen::VectorXf::Zero (kEmbeddingDim);
    embedding[static_cast<Eigen::Index> (memory_id % kEmbeddingDim)] = 1.0f;
    const std::size_t index = state.current_entries.size ();
    state.current_entries.emplace_back (
        memory_id, memory_id, 1000 + memory_id, "LONG_TERM", "opaque",
        std::move (embedding), memory_id);
    state.current_memory_index.emplace (memory_id, index);
    state.current_memory_order.emplace (memory_id, index);
  };
  for (std::size_t index = 1; index < start_size; ++index)
    append_entry (static_cast<long long> (index));

  REQUIRE (cache::AdvanceSQLiteSparseRouteBackfill (state, *store));
  REQUIRE_FALSE (state.sqlite_sparse_route_backfill);

  append_entry (static_cast<long long> (start_size));
  REQUIRE (cache::AdvanceSQLiteSparseRouteBackfill (state, *store));
  REQUIRE (state.sqlite_sparse_route_backfill);
  REQUIRE (state.sqlite_sparse_route_backfill_last_sealed_rows
           <= parameters.backfill_batch_size);
  REQUIRE (state.sqlite_sparse_route_backfill->ActiveCount ()
           <= static_cast<long long> (parameters.route_capacity
                                      + parameters.backfill_batch_size));
}

TEST_CASE ("SQLite sparse route keeps active live mutations in a bounded "
           "durable journal",
           "[operations][graph][retrieval][hnsw][sqlite][backfill]"
           "[regression]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  namespace sqlite_route
      = operations::sparse_retrieval_route_sqlite_internal;
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  const auto parameters = sqlite_route::DeriveParameters (0.0, 0.0, 0.0);

  cache::State state;
  state.embedding_dim = kEmbeddingDim;
  state.sparse_route_parameters = parameters;
  std::vector<std::pair<long long, Eigen::VectorXf>> entries;
  entries.reserve (600);
  for (long long memory_id = 1; memory_id <= 600; ++memory_id)
    {
      auto embedding = VectorWithCosineToDim0 (
          0.10f + static_cast<float> (memory_id - 1) * 0.001f);
      const std::size_t index = state.current_entries.size ();
      state.current_entries.emplace_back (
          memory_id, memory_id, 1000 + memory_id, "LONG_TERM", "opaque",
          embedding, memory_id);
      state.current_memory_index.emplace (memory_id, index);
      state.current_memory_order.emplace (memory_id, index);
      entries.emplace_back (memory_id, std::move (embedding));
    }
  auto hnsw = operations::sparse_retrieval_route_internal::Route::Create (
      kEmbeddingDim, entries, parameters.hnsw);
  REQUIRE (hnsw);
  auto route = sqlite_route::Route::Create (
      *store, kEmbeddingDim, entries, *hnsw, parameters);
  REQUIRE (route);

  for (long long memory_id = 1; memory_id <= 400; ++memory_id)
    {
      Eigen::VectorXf revised = UnitVec (
          static_cast<int> (memory_id % kEmbeddingDim));
      state.current_entries[static_cast<std::size_t> (memory_id - 1)]
          .embedding = revised;
      REQUIRE (route->Upsert (memory_id, revised));
    }
  REQUIRE (route->DeltaSize () == 0);
  REQUIRE (route->HasDirtyMemoryIds ());
  const auto first_slice = route->DirtyMemoryIds (parameters.route_capacity);
  REQUIRE (first_slice);
  REQUIRE (first_slice->size () == parameters.route_capacity);
  REQUIRE_FALSE (
      cache::StageSQLiteSparseRouteDirtyForSearch (state, *route));
  REQUIRE (route->DeltaSize () == 0);

  std::size_t remaining = 400;
  while (remaining > parameters.route_capacity)
    {
      REQUIRE_FALSE (
          cache::StageSQLiteSparseRouteDirtyForSearch (state, *route));
      REQUIRE (cache::ReconcileSQLiteSparseRoute (
          state, *store, route,
          std::shared_ptr<
              operations::sparse_retrieval_route_internal::Route>{}));
      remaining -= parameters.route_capacity;
      REQUIRE (store->Execute (
                   "SELECT memory_id FROM cortext_sparse_route_dirty")
                   .size ()
               == remaining);
    }
  const auto staged_remainder
      = cache::StageSQLiteSparseRouteDirtyForSearch (state, *route);
  REQUIRE (staged_remainder == remaining);
  REQUIRE (route->DeltaSize () == remaining);
  REQUIRE (route->Search (UnitVec (0), parameters.route_capacity));
  REQUIRE (cache::ReconcileSQLiteSparseRoute (
      state, *store, route,
      std::shared_ptr<
          operations::sparse_retrieval_route_internal::Route>{}));
  REQUIRE (route->DeltaSize () == 0);
  REQUIRE_FALSE (route->HasDirtyMemoryIds ());
}

TEST_CASE ("SQLite sparse route drains live build mutations in knob-bounded "
           "slices",
           "[operations][graph][retrieval][hnsw][sqlite][backfill]"
           "[regression]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  namespace sqlite_route
      = operations::sparse_retrieval_route_sqlite_internal;
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);

  cache::State state;
  state.embedding_dim = kEmbeddingDim;
  state.sparse_route_parameters
      = sqlite_route::DeriveParameters (0.0, 0.0, 0.0);
  constexpr long long kEntryCount = 800;
  state.current_entries.reserve (kEntryCount);
  for (long long memory_id = 1; memory_id <= kEntryCount; ++memory_id)
    {
      Eigen::VectorXf embedding (kEmbeddingDim);
      for (Eigen::Index dimension = 0; dimension < embedding.size ();
           ++dimension)
        embedding[dimension] = std::cos (
            static_cast<float> (memory_id * (dimension + 11)) * 0.004f);
      embedding.normalize ();
      const std::size_t index = state.current_entries.size ();
      state.current_entries.emplace_back (
          memory_id, memory_id, 1000 + memory_id, "LONG_TERM", "opaque",
          std::move (embedding), memory_id);
      state.current_memory_index.emplace (memory_id, index);
      state.current_memory_order.emplace (memory_id, index);
    }

  REQUIRE (cache::AdvanceSQLiteSparseRouteBackfill (state, *store));
  REQUIRE (state.sqlite_sparse_route_backfill);
  auto &build = state.sqlite_sparse_route_backfill;
  const std::size_t backfill_batch
      = state.sparse_route_parameters.backfill_batch_size;
  REQUIRE (build->BuildCursor ()
           == static_cast<long long> (backfill_batch));
  constexpr std::size_t kDirtyCount = 700;
  for (long long memory_id = 1;
       memory_id <= static_cast<long long> (kDirtyCount); ++memory_id)
    REQUIRE (build->Upsert (
        memory_id,
        state.current_entries[static_cast<std::size_t> (memory_id - 1)]
            .embedding));
  REQUIRE (build->DeltaSize () == 0);
  REQUIRE (store->Execute (
               "SELECT memory_id FROM cortext_sparse_route_dirty")
               .size ()
           == kDirtyCount);
  const auto bounded_dirty = build->DirtyMemoryIds (
      state.sparse_route_parameters.route_capacity);
  REQUIRE (bounded_dirty);
  REQUIRE (bounded_dirty->size ()
           == state.sparse_route_parameters.route_capacity);
  REQUIRE (build->HasDirtyMemoryIds ());

  REQUIRE (cache::AdvanceSQLiteSparseRouteBackfill (state, *store));
  REQUIRE (state.sqlite_sparse_route_backfill_failure_count == 0);
  REQUIRE (build->DeltaSize () == 0);
  REQUIRE (build->BuildCursor ()
           == static_cast<long long> (backfill_batch * 2));
  REQUIRE (state.sqlite_sparse_route_backfill_last_sealed_rows
           == backfill_batch);
  const std::size_t expected_remaining_dirty
      = kDirtyCount - state.sparse_route_parameters.route_capacity;
  REQUIRE (store->Execute (
               "SELECT memory_id FROM cortext_sparse_route_dirty")
               .size ()
           == expected_remaining_dirty);
  REQUIRE (build->DirtyMemoryIds (
               state.sparse_route_parameters.route_capacity)
               ->size ()
           == state.sparse_route_parameters.route_capacity);

  std::size_t remaining_dirty = expected_remaining_dirty;
  std::size_t drain_count = 0;
  while (remaining_dirty > 0 && drain_count++ < 10)
    {
      const bool advanced
          = cache::AdvanceSQLiteSparseRouteBackfill (state, *store);
      CAPTURE (drain_count, build->LastSealFailureCode (),
               build->LastSearchNodeRows (), build->DeltaSize (),
               build->ActivationEntryMemoryId (),
               build->ActivationSearchNodeBudget (),
               state.sqlite_sparse_route_backfill_failure_code);
      REQUIRE (advanced);
      REQUIRE (state.sqlite_sparse_route_backfill_failure_count == 0);
      REQUIRE (build->DeltaSize () == 0);
      const std::size_t next_remaining
          = store->Execute (
                     "SELECT memory_id FROM cortext_sparse_route_dirty")
                .size ();
      REQUIRE (next_remaining < remaining_dirty);
      REQUIRE (remaining_dirty - next_remaining
               <= state.sparse_route_parameters.route_capacity
                      + state.sparse_route_parameters.backfill_batch_size);
      remaining_dirty = next_remaining;
    }
  REQUIRE (remaining_dirty == 0);
}

TEST_CASE ("SQLite sparse route work budgets derive from every knob",
           "[operations][graph][retrieval][hnsw][sqlite][knobs]")
{
  namespace sqlite_route
      = operations::sparse_retrieval_route_sqlite_internal;
  for (const double focus : { 0.0, 0.5, 1.0 })
    for (const double sensitivity : { 0.0, 0.5, 1.0 })
      for (const double stability : { 0.0, 0.5, 1.0 })
        {
          CAPTURE (focus, sensitivity, stability);
          const auto parameters = sqlite_route::DeriveParameters (
              focus, sensitivity, stability);
          REQUIRE (
              parameters.route_capacity
              == static_cast<std::size_t> (
                  operations::sparse_retrieval_knobs_internal::RouteCapacity (
                      focus, sensitivity, stability)));
          REQUIRE (
              parameters.backfill_batch_size
              == static_cast<std::size_t> (
                  operations::sparse_retrieval_knobs_internal::
                      BackfillBatchSize (
                      focus, sensitivity, stability)));
          REQUIRE (
              operations::sparse_retrieval_knobs_internal::
                  HydrationFallbackSignalLimit (
                      focus, sensitivity, stability)
              == static_cast<int> (parameters.backfill_batch_size));
          REQUIRE (parameters.route_bootstrap_limit
                   == parameters.route_capacity * 2);
          REQUIRE (parameters.activation_identity_target
                   == parameters.route_capacity * 2
                          + parameters.backfill_batch_size * 2);
          REQUIRE (parameters.activation_snapshot_capacity
                   == parameters.activation_identity_target);
          REQUIRE (parameters.total_query_row_budget
                   == parameters.search_node_budget
                          + parameters.activation_snapshot_capacity);
          REQUIRE (parameters.search_node_budget
                   == static_cast<std::size_t> (
                       operations::sparse_retrieval_knobs_internal::
                           RetrievalSawtoothPeak (
                               focus, sensitivity, stability)));
          REQUIRE (parameters.activation_search_node_budget_min
                   == static_cast<std::size_t> (
                       operations::sparse_retrieval_knobs_internal::
                           RetrievalSawtoothFloor (
                               focus, sensitivity, stability)));
          REQUIRE (parameters.activation_search_node_budget_step
                   == parameters.reciprocal_update_count);
          REQUIRE (parameters.search_expansion_batch
                   == std::max<std::size_t> (
                       8, parameters.backfill_batch_size / 4));
          REQUIRE (
              operations::sparse_retrieval_knobs_internal::OrdinarySealBatch (
                  focus, sensitivity, stability)
              == static_cast<int> (parameters.backfill_batch_size
                                   - parameters.search_expansion_batch));
          REQUIRE (parameters.search_ef == parameters.search_node_budget);
          REQUIRE (parameters.activation_search_ef_min
                   == parameters.activation_search_node_budget_min);
          REQUIRE (parameters.activation_search_ef_step
                   == parameters.reciprocal_update_count);
          REQUIRE (parameters.shadow_node_capacity
                   == parameters.route_capacity * 24);
          REQUIRE (parameters.backfill_search_node_budget
                   == parameters.route_capacity
                          + parameters.backfill_batch_size);
          REQUIRE (parameters.backfill_search_ef
                   == parameters.backfill_batch_size * 2);
          REQUIRE (
              parameters.row_addressed_neighbor_count
              == static_cast<std::size_t> (
                  operations::sparse_retrieval_knobs_internal::
                      GraphNeighborCount (
                      focus, sensitivity, stability)));
          REQUIRE (
              parameters.row_addressed_level_zero_links
              == static_cast<std::size_t> (
                  operations::sparse_retrieval_knobs_internal::
                      GraphLevelZeroLinks (
                      focus, sensitivity, stability)));
          REQUIRE (
              parameters.family_exact_comparison_limit
              == static_cast<std::size_t> (
                  operations::sparse_retrieval_knobs_internal::
                      FamilyExactComparisonLimit (
                          focus, sensitivity, stability)));
          REQUIRE (
              parameters.maximum_level
              == static_cast<std::size_t> (
                  operations::sparse_retrieval_knobs_internal::MaximumLevel (
                      focus, sensitivity, stability)));
          REQUIRE (
              parameters.reciprocal_update_count
              == static_cast<std::size_t> (
                  operations::sparse_retrieval_knobs_internal::
                      ReciprocalUpdateCount (
                      focus, sensitivity, stability)));
          REQUIRE (parameters.hnsw.minimum_capacity
                   == parameters.route_bootstrap_limit);
          REQUIRE (parameters.hnsw.graph_neighbor_count
                   == parameters.row_addressed_neighbor_count);
          REQUIRE (
              parameters.hnsw.construction_effort
              == static_cast<std::size_t> (
                  operations::sparse_retrieval_knobs_internal::
                      ConstructionEffort (
                      focus, sensitivity, stability)));
          REQUIRE (
              parameters.hnsw.query_effort
              == static_cast<std::size_t> (
                  operations::sparse_retrieval_knobs_internal::QueryEffort (
                      focus, sensitivity, stability)));
          {
            cortext::testing::ScopedEnvVar envelope (
                "CORTEXT_EXPERIMENT_SQLITE_SPARSE_NODE_ENVELOPE", "6");
            const auto selected = sqlite_route::DeriveParameters (
                focus, sensitivity, stability);
            REQUIRE (selected.search_node_budget
                     == parameters.route_capacity * 6);
            REQUIRE (selected.activation_search_node_budget_min
                     == parameters.route_capacity * 6);
          }
        }

  const auto midpoint = sqlite_route::DeriveParameters (0.5, 0.5, 0.5);
  REQUIRE (midpoint.route_capacity == 512);
  REQUIRE (midpoint.activation_identity_target == 1280);
  REQUIRE (midpoint.backfill_batch_size == 128);
  REQUIRE (midpoint.row_addressed_neighbor_count == 64);
  REQUIRE (midpoint.row_addressed_level_zero_links == 128);
  REQUIRE (midpoint.family_exact_comparison_limit == 1024);
  REQUIRE (midpoint.maximum_level == 9);
  REQUIRE (midpoint.reciprocal_update_count == 8);
  REQUIRE (midpoint.hnsw.construction_effort == 200);
  REQUIRE (midpoint.hnsw.query_effort == 1280);

  for (const auto &[selector, expected] :
       std::vector<std::pair<std::string, std::size_t>>{
           { "1", midpoint.route_capacity },
           { "2", midpoint.activation_identity_target },
           { "3", midpoint.route_capacity + midpoint.backfill_batch_size },
           { "4", midpoint.route_capacity * 2 },
           { "5", midpoint.route_capacity * 5 },
           { "6", midpoint.route_capacity * 6 },
           { "8", midpoint.route_capacity * 4 },
           { "18", midpoint.route_capacity * 4
                       + midpoint.reciprocal_update_count },
           { "19", midpoint.route_capacity * 4
                       + midpoint.reciprocal_update_count },
           { "10", midpoint.route_capacity * 10 },
           { "12", midpoint.route_capacity * 12 },
           { "16", midpoint.route_capacity * 16 },
       })
    {
      CAPTURE (selector, expected);
      cortext::testing::ScopedEnvVar envelope (
          "CORTEXT_EXPERIMENT_SQLITE_SPARSE_NODE_ENVELOPE", selector);
      const auto selected = sqlite_route::DeriveParameters (0.5, 0.5, 0.5);
      REQUIRE (selected.search_node_budget == expected);
      REQUIRE (selected.activation_search_node_budget_min == expected);
      REQUIRE (selected.activation_identity_target == 1280);
      const std::size_t expected_effort
          = selector == "19" ? expected : selected.route_capacity * 5;
      REQUIRE (selected.search_ef == expected_effort);
      REQUIRE (selected.activation_search_ef_min
               == expected_effort);
      REQUIRE (
          selected.total_query_row_budget
          == (selector == "19"
                  ? expected + selected.activation_identity_target
                  : selected.route_capacity * 5
                        + selected.activation_identity_target));
      REQUIRE (selected.backfill_search_node_budget
               == selected.route_capacity + selected.backfill_batch_size);
    }

  {
    cortext::testing::ScopedEnvVar envelope (
        "CORTEXT_EXPERIMENT_SQLITE_SPARSE_NODE_ENVELOPE", "20");
    const auto selected = sqlite_route::DeriveParameters (0.5, 0.5, 0.5);
    const std::size_t minimum
        = selected.route_capacity * 4 + selected.reciprocal_update_count;
    REQUIRE (selected.search_node_budget == selected.route_capacity * 5);
    REQUIRE (selected.search_ef == selected.route_capacity * 5);
    REQUIRE (selected.activation_search_node_budget_min == minimum);
    REQUIRE (selected.activation_search_ef_min == minimum);
    REQUIRE (selected.activation_search_node_budget_step
             == selected.reciprocal_update_count);
    REQUIRE (selected.activation_search_ef_step
             == selected.reciprocal_update_count);
  }

  {
    cortext::testing::ScopedEnvVar envelope (
        "CORTEXT_EXPERIMENT_SQLITE_SPARSE_NODE_ENVELOPE", "7");
    const auto ignored = sqlite_route::DeriveParameters (0.5, 0.5, 0.5);
    REQUIRE (ignored.search_node_budget == ignored.route_capacity * 5);
    REQUIRE (ignored.activation_search_node_budget_min
             == ignored.route_capacity * 5);
  }

  const auto first_seal_rows = [] (double focus, double sensitivity,
                                   double stability) {
    namespace cache
        = operations::historical_surface_search_cache_internal;
    auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
    cortext::testing::InitializeCoreSchema (*store);
    cache::State state;
    state.embedding_dim = kEmbeddingDim;
    state.sparse_route_parameters = sqlite_route::DeriveParameters (
        focus, sensitivity, stability);
    constexpr long long kEntryCount = 800;
    state.current_entries.reserve (kEntryCount);
    for (long long memory_id = 1; memory_id <= kEntryCount; ++memory_id)
      {
        Eigen::VectorXf embedding (kEmbeddingDim);
        for (Eigen::Index dimension = 0; dimension < embedding.size ();
             ++dimension)
          embedding[dimension] = std::sin (
              static_cast<float> (memory_id * (dimension + 5)) * 0.009f);
        embedding.normalize ();
        const std::size_t index = state.current_entries.size ();
        state.current_entries.emplace_back (
            memory_id, memory_id, 1000 + memory_id, "LONG_TERM",
            "opaque/" + std::to_string (memory_id % 4),
            std::move (embedding), memory_id);
        state.current_memory_index.emplace (memory_id, index);
        state.current_memory_order.emplace (memory_id, index);
      }
    REQUIRE (cache::AdvanceSQLiteSparseRouteBackfill (state, *store));
    return state.sqlite_sparse_route_backfill_last_sealed_rows;
  };
  REQUIRE (first_seal_rows (0.0, 0.0, 0.0) == 64);
  REQUIRE (first_seal_rows (0.5, 0.5, 0.5) == 128);
  REQUIRE (first_seal_rows (1.0, 1.0, 1.0) == 192);

  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  std::vector<std::pair<long long, Eigen::VectorXf>> route_entries;
  route_entries.reserve (1300);
  for (long long memory_id = 1; memory_id <= 1300; ++memory_id)
    {
      Eigen::VectorXf embedding (kEmbeddingDim);
      for (Eigen::Index dimension = 0; dimension < embedding.size ();
           ++dimension)
        embedding[dimension] = std::sin (
            static_cast<float> (memory_id * (dimension + 7)) * 0.005f);
      embedding.normalize ();
      route_entries.emplace_back (memory_id, std::move (embedding));
    }
  const auto low_parameters = sqlite_route::DeriveParameters (0.0, 0.0, 0.0);
  auto hnsw = operations::sparse_retrieval_route_internal::Route::Create (
      kEmbeddingDim, route_entries, low_parameters.hnsw);
  REQUIRE (hnsw);
  auto sqlite = sqlite_route::Route::Create (
      *store, kEmbeddingDim, route_entries, *hnsw, low_parameters);
  REQUIRE (sqlite);
  for (std::size_t query = 0; query < 64; ++query)
    {
      const auto result = sqlite->Search (
          route_entries[(query * 19) % route_entries.size ()].second,
          low_parameters.route_capacity);
      REQUIRE (result);
      REQUIRE (sqlite->LastSearchNodeRows ()
               <= low_parameters.search_node_budget);
    }
}

TEST_CASE ("SQLite sparse route backfill survives an authoritative cache "
           "refresh",
           "[operations][graph][retrieval][hnsw][sqlite][backfill]"
           "[regression]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  const auto parameters
      = operations::sparse_retrieval_route_sqlite_internal::
          DefaultParameters ();
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);

  ProcessorContext pctx;
  std::vector<cache::Entry> current_entries;
  constexpr long long kEntryCount = 600;
  current_entries.reserve (kEntryCount);
  for (long long memory_id = 1; memory_id <= kEntryCount; ++memory_id)
    {
      Eigen::VectorXf embedding (kEmbeddingDim);
      for (Eigen::Index dimension = 0; dimension < embedding.size ();
           ++dimension)
        embedding[dimension] = std::cos (
            static_cast<float> (memory_id * (dimension + 3)) * 0.007f);
      embedding.normalize ();
      current_entries.emplace_back (
          memory_id, memory_id, 1000 + memory_id, "LONG_TERM",
          "opaque/" + std::to_string (memory_id % 4),
          std::move (embedding), memory_id);
    }

  REQUIRE (cache::Reset (pctx, {}, std::move (current_entries)));
  cache::SetSparseRouteParameters (pctx, parameters);
  cache::SetCurrentSurfaceDatabaseCurrent (pctx, true);
  cache::SetProcessorSurfaceComplete (pctx, true);
  auto state = cache::FindMutable (pctx);
  REQUIRE (state);
  REQUIRE (cache::AdvanceSQLiteSparseRouteBackfill (*state, *store));
  const auto build = state->sqlite_sparse_route_backfill;
  REQUIRE (build);
  REQUIRE (build->BuildCursor ()
           == static_cast<long long> (parameters.backfill_batch_size));
  REQUIRE (cache::IncrementalSparseRouteSealReady (*state, true));

  auto refreshed_entries = state->current_entries;
  REQUIRE (cache::Reset (pctx, {}, std::move (refreshed_entries)));
  state = cache::FindMutable (pctx);
  REQUIRE (state);
  REQUIRE (state->current_surface_database_current);
  REQUIRE (state->processor_surface_complete);
  REQUIRE (state->sqlite_sparse_route_backfill == build);
  REQUIRE (state->sqlite_sparse_route_backfill->BuildCursor ()
           == static_cast<long long> (parameters.backfill_batch_size));
  REQUIRE (cache::IncrementalSparseRouteSealReady (*state, true));

  REQUIRE (cache::AdvanceSQLiteSparseRouteBackfill (*state, *store));
  REQUIRE (state->sqlite_sparse_route_backfill);
  REQUIRE (state->sqlite_sparse_route_backfill->BuildCursor ()
           == static_cast<long long> (parameters.backfill_batch_size * 2));
  const auto generations = store->Execute (
      "SELECT COUNT(DISTINCT generation) AS generation_count, "
      "MIN(generation) AS min_generation, MAX(generation) AS max_generation "
      "FROM cortext_sparse_route_nodes");
  REQUIRE (generations.size () == 1);
  REQUIRE (store::AnyToLongLong (
               generations.front ().at ("generation_count"))
               .value_or (0)
           == 1);
  REQUIRE (store::AnyToLongLong (
               generations.front ().at ("min_generation"))
               .value_or (0)
           == 1);
  REQUIRE (store::AnyToLongLong (
               generations.front ().at ("max_generation"))
               .value_or (0)
           == 1);
  cache::Erase (pctx);
}

TEST_CASE ("Graph retrieval soundly collapses cosine-near families before KNN cap",
           "[operations][graph][retrieval][family]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const auto [family_first, family_second] = NearDuplicateFamilyPair ();
  REQUIRE (family_first.dot (family_second) > 0.999999f);
  constexpr long long kTargetMemoryId = 1;
  SeedMemory (*store, kTargetMemoryId, 1,
              VectorWithCosineToQuery (family_first, 0.95f), 1000,
              "target");
  for (long long offset = 0; offset < 430; ++offset)
    {
      SeedMemory (*store, 1000 + offset, 10000 + offset,
                  offset % 2 == 0 ? family_first : family_second,
                  2000 + offset,
                  "duplicate/" + std::to_string (offset));
    }
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (family_first, 100000));
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), kTargetMemoryId)
           != out.candidate_memory_ids.end ());
  const auto exact_comparison_count
      = operations::retrieval_trace::GetLastFamilyExactComparisonCount ();
  REQUIRE (exact_comparison_count > 0);
  REQUIRE (
      exact_comparison_count
      <= static_cast<std::size_t> (
          operations::sparse_retrieval_knobs_internal::
              FamilyExactComparisonLimit (
                  cfg.focus, cfg.sensitivity, cfg.stability)));
}

TEST_CASE ("Graph retrieval bounds exact family checks on diverse embeddings",
           "[operations][graph][retrieval][family][performance]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  for (int dimension = 0; dimension < kEmbeddingDim; ++dimension)
    {
      SeedMemory (*store, 1000 + dimension, 10000 + dimension,
                  UnitVec (dimension), 1000 + dimension,
                  "diverse/" + std::to_string (dimension));
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  (void)processor.Process (MakeSignal (UnitVec (0), 100000));
  const std::size_t exact_comparisons
      = operations::retrieval_trace::GetLastFamilyExactComparisonCount ();
  constexpr std::size_t kAllPairs
      = static_cast<std::size_t> (kEmbeddingDim)
        * static_cast<std::size_t> (kEmbeddingDim - 1) / 2;
  REQUIRE (exact_comparisons == 0);
  REQUIRE (exact_comparisons < kAllPairs / 4);
}

TEST_CASE ("Graph retrieval bounds exact family checks on dense orthogonal "
           "embeddings",
           "[operations][graph][retrieval][family][performance]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  for (int row = 0; row < kEmbeddingDim; ++row)
    {
      SeedMemory (*store, 1000 + row, 10000 + row,
                  HadamardRow (static_cast<unsigned int> (row)), 1000 + row,
                  "diverse/" + std::to_string (row));
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  (void)processor.Process (MakeSignal (HadamardRow (0), 100000));
  const std::size_t exact_comparisons
      = operations::retrieval_trace::GetLastFamilyExactComparisonCount ();
  constexpr std::size_t kAllPairs
      = static_cast<std::size_t> (kEmbeddingDim)
        * static_cast<std::size_t> (kEmbeddingDim - 1) / 2;
  REQUIRE (exact_comparisons == 0);
  REQUIRE (exact_comparisons < kAllPairs / 4);
}

TEST_CASE ("Graph retrieval SQL fallback collapses families before candidate cap",
           "[operations][graph][retrieval][family][sql]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const auto [family_first, family_second] = NearDuplicateFamilyPair ();
  constexpr long long kTargetMemoryId = 1;
  SeedMemory (*store, kTargetMemoryId, 1,
              VectorWithCosineToQuery (family_first, 0.95f), 1000,
              "target");
  for (long long offset = 0; offset < 430; ++offset)
    {
      SeedMemory (*store, 1000 + offset, 10000 + offset,
                  offset % 2 == 0 ? family_first : family_second,
                  2000 + offset,
                  "duplicate/" + std::to_string (offset));
    }
  for (long long offset = 0; offset < 900; ++offset)
    {
      SeedMemory (*store, 2000 + offset, 20000 + offset, UnitVec (31),
                  3000 + offset, "distant/" + std::to_string (offset));
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx);
  const auto recovery_state
      = operations::historical_surface_search_cache_internal::Find (pctx);
  auto signal = MakeSignal (family_first, 100000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  const auto &candidates = ctx.GetRetrievedMemoryCandidates ();
  REQUIRE (std::find_if (candidates.begin (), candidates.end (),
                         [] (const auto &candidate) {
                           return candidate.memory_id == kTargetMemoryId;
                         })
           != candidates.end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 1);
  const int seed_limit = std::max (1, core::RetrievalMaxResults (cfg.focus));
  const int seed_search_limit = core::RetrievalSeedSearchK (
      cfg.focus, cfg.sensitivity, cfg.stability, seed_limit);
  const int fallback_materialization_limit = core::RetrievalSeedSearchK (
      cfg.focus, cfg.sensitivity, cfg.stability, seed_search_limit);
  REQUIRE (
      operations::retrieval_trace::GetLastSqlFallbackMaterializedRowCount ()
      <= static_cast<std::size_t> (fallback_materialization_limit));
  REQUIRE (
      operations::historical_surface_search_cache_internal::Find (pctx)
      == recovery_state);
  REQUIRE (recovery_state->embedding_dim == 0);
  REQUIRE (recovery_state->entries.empty ());
  REQUIRE (recovery_state->current_entries.empty ());

  OperationContext second_ctx (signal, pctx, cfg, store.get ());
  second_ctx.SetShouldCheckRetrieval (true);
  second_ctx.SetWriteExclusionTs (signal.timestamp);
  auto second_tx = store->Begin ();
  operation.Execute (second_ctx, *second_tx);
  second_tx->Rollback ();
  REQUIRE (
      operations::historical_surface_search_cache_internal::Find (pctx)
      == recovery_state);
  const auto second_target = std::find_if (
      second_ctx.GetRetrievedMemoryCandidates ().begin (),
      second_ctx.GetRetrievedMemoryCandidates ().end (),
      [] (const auto &candidate) {
        return candidate.memory_id == kTargetMemoryId;
      });
  REQUIRE (second_target
           != second_ctx.GetRetrievedMemoryCandidates ().end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 1);
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval SQL fallback filters ineligible nearest rows before K",
           "[operations][graph][retrieval][sql][eligibility]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  constexpr long long kTargetMemoryId = 9000;
  SeedMemory (*store, kTargetMemoryId, 19000,
              VectorWithCosineToDim0 (0.90f), 1000, "eligible");
  for (long long offset = 0; offset < 900; ++offset)
    {
      const long long memory_id = 1000 + offset;
      SeedMemory (*store, memory_id, 10000 + offset, UnitVec (0), 1000,
                  "ineligible");
      store->Execute ("UPDATE memories SET kind = 'WORKING' "
                      "WHERE memory_id = ?",
                      { memory_id });
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx);
  auto signal = MakeSignal (UnitVec (0), 100000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) {
                 return candidate.memory_id == kTargetMemoryId;
               })
           != ctx.GetRetrievedMemoryCandidates ().end ());
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval SQL fallback performance probe",
           "[.benchmark][operations][graph][retrieval][sql]")
{
  const char *row_count_env = std::getenv ("LTM_FALLBACK_BENCH_ROWS");
  const char *repeats_env = std::getenv ("LTM_FALLBACK_BENCH_REPEATS");
  const char *reconstruct_every_env
      = std::getenv ("LTM_FALLBACK_BENCH_RECONSTRUCT_EVERY");
  const char *processor_complete_env
      = std::getenv ("LTM_FALLBACK_BENCH_PROCESSOR_COMPLETE");
  const long long row_count
      = row_count_env == nullptr ? 1915 : std::stoll (row_count_env);
  const int repeats = repeats_env == nullptr ? 12 : std::stoi (repeats_env);
  const long long reconstruct_every
      = reconstruct_every_env == nullptr
            ? 0
            : std::stoll (reconstruct_every_env);
  const bool processor_surface_complete
      = processor_complete_env != nullptr
        && std::string (processor_complete_env) == "1";
  REQUIRE (row_count > 0);
  REQUIRE (repeats > 0);
  REQUIRE (reconstruct_every >= 0);

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  ProcessorContext pctx;
  for (long long offset = 0; offset < row_count; ++offset)
    {
      Eigen::VectorXf embedding = UnitVec (static_cast<int> (offset % 255));
      embedding[255]
          = static_cast<float> ((offset % 97) + 1) / 10000.0f;
      embedding.normalize ();
      const long long memory_id = 1000 + offset;
      const long long embedding_id = 10000 + offset;
      const long long timestamp = 1000 + offset;
      SeedMemory (*store, memory_id, embedding_id, embedding, timestamp);
      pctx.UpsertRetrievalSurface (
          { memory_id, embedding_id, timestamp, timestamp, 0, 0, 0, 0,
            "LONG_TERM", "benchmark", "", -1.0, 0, 0.0, 0.0, 0.0,
            false, true, embedding });
    }

  long long reconstruction_count = 0;
  if (reconstruct_every > 0)
    {
      operations::constructive_recall::ReconstructionUpdatePolicy policy;
      policy.update_current_surface = false;
      auto reconstruction_tx = store->Begin ();
      for (long long offset = 0; offset < row_count;
           offset += reconstruct_every)
        {
          Eigen::VectorXf embedding
              = UnitVec (static_cast<int> (offset % 255));
          embedding[255]
              = static_cast<float> ((offset % 97) + 1) / 10000.0f;
          embedding.normalize ();
          const long long memory_id = 1000 + offset;
          REQUIRE (operations::constructive_recall::
                       AppendReconstructionWithEmbedding (
                           *reconstruction_tx, memory_id, embedding, {},
                           100000 + offset, 0.1, "benchmark", 1.0, 1.0,
                           policy)
                   > 0);
          const auto latest = operations::constructive_recall::
              LoadLatestReconstruction (*reconstruction_tx, memory_id);
          REQUIRE (latest.has_value ());
          pctx.UpsertRetrievalSurface (
              { memory_id, latest->embedding_id, 1000 + offset,
                1000 + offset, 0, 0, 0, 0, "LONG_TERM", "benchmark", "",
                -1.0, 0, 0.0, 0.0, 0.0, false, true, embedding });
          ++reconstruction_count;
        }
      reconstruction_tx->Commit ();
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  operations::historical_surface_search_cache_internal::
      SetCurrentSurfaceDatabaseCurrent (pctx, reconstruction_count == 0);
  operations::historical_surface_search_cache_internal::
      SetProcessorSurfaceComplete (pctx, processor_surface_complete);
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx, true);
  auto signal = MakeSignal (UnitVec (0), 1000000000);
  signal.retention = Retention::Ephemeral;
  std::vector<double> elapsed_ms;
  elapsed_ms.reserve (static_cast<std::size_t> (repeats));
  for (int repeat = 0; repeat < repeats; ++repeat)
    {
      OperationContext ctx (signal, pctx, cfg, store.get ());
      ctx.SetShouldCheckRetrieval (true);
      ctx.SetWriteExclusionTs (signal.timestamp);
      GraphAugmentedRetrieveCandidates operation;
      auto tx = store->Begin ();
      const auto start = std::chrono::steady_clock::now ();
      operation.Execute (ctx, *tx);
      const auto end = std::chrono::steady_clock::now ();
      tx->Rollback ();
      REQUIRE_FALSE (ctx.GetRetrievedMemoryCandidates ().empty ());
      elapsed_ms.push_back (
          std::chrono::duration<double, std::milli> (end - start).count ());
    }
  std::sort (elapsed_ms.begin (), elapsed_ms.end ());
  const auto percentile = [&elapsed_ms] (double fraction) {
    const std::size_t index = std::min (
        elapsed_ms.size () - 1,
        static_cast<std::size_t> (
            std::ceil (fraction * static_cast<double> (elapsed_ms.size ())))
            - 1);
    return elapsed_ms[index];
  };
  std::cout << "CORTEXT_FALLBACK_BENCH {\"rows\":" << row_count
            << ",\"reconstructions\":" << reconstruction_count
            << ",\"current_surface_database_current\":"
            << (reconstruction_count == 0 ? "true" : "false")
            << ",\"processor_surface_complete\":"
            << (processor_surface_complete ? "true" : "false")
            << ",\"repeats\":" << repeats
            << ",\"p50_ms\":" << percentile (0.50)
            << ",\"p95_ms\":" << percentile (0.95) << "}" << std::endl;
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval ephemeral SQL fallback does not install cache",
           "[operations][graph][retrieval][sql][ephemeral]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  SeedMemory (*store, 10, 100, UnitVec (0), 1000);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  auto signal = MakeSignal (UnitVec (0), 100000);
  signal.retention = Retention::Ephemeral;
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  REQUIRE_FALSE (ctx.GetRetrievedMemoryCandidates ().empty ());
  REQUIRE (
      operations::historical_surface_search_cache_internal::Find (pctx)
      == nullptr);
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 1);
}

TEST_CASE ("Graph retrieval cache rebuild accepts shared base embeddings",
           "[operations][graph][retrieval][cache][shared-embedding]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  store->Execute ("UPDATE memories SET kind = 'WORKING' WHERE memory_id = 10");
  store->Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
      "start_ts, created_at) VALUES(?, ?, ?, 'LONG_TERM', ?, ?)",
      { 20LL, 100LL, std::string ("shared"), 500LL, 500LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  auto signal = MakeSignal (UnitVec (0), 100000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  const auto state
      = operations::historical_surface_search_cache_internal::Find (pctx);
  REQUIRE (state != nullptr);
  REQUIRE_FALSE (state->recovery_failed);
  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) { return candidate.memory_id == 20; })
           != ctx.GetRetrievedMemoryCandidates ().end ());
  const auto shared_entry = std::find_if (
      state->entries.begin (), state->entries.end (),
      [] (const auto &entry) { return entry.embedding_id == 100; });
  REQUIRE (shared_entry != state->entries.end ());
  REQUIRE (shared_entry->memory_references.size () == 1);
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval cache rebuild covers association supersession "
           "candidates",
           "[operations][graph][retrieval][cache][supersession]"
           "[association]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  store->Execute (
      "UPDATE memories SET kind = 'ASSOCIATION' WHERE memory_id = 10");

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  ProcessorContext::RetrievalSurfaceEntry association_surface;
  association_surface.memory_id = 10;
  association_surface.embedding_id = 100;
  association_surface.start_ts = 1000;
  association_surface.kind = "ASSOCIATION";
  association_surface.vector_seed_eligible = false;
  association_surface.embedding = UnitVec (0);
  pctx.UpsertRetrievalSurface (std::move (association_surface));
  auto signal = MakeSignal (UnitVec (0), 2000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  const auto state
      = operations::historical_surface_search_cache_internal::Find (pctx);
  REQUIRE (state);
  REQUIRE (state->supersession_entry_by_memory.contains (10));
  REQUIRE (state->current_memory_index.contains (10));
  REQUIRE (operations::historical_surface_search_cache_internal::
               CurrentPopulationCoversHistorical (*state, -1));

  operations::historical_surface_search_cache_internal::
      SetProcessorSurfaceComplete (pctx, true);
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx, true);
  OperationContext recovery_ctx (signal, pctx, cfg, store.get ());
  recovery_ctx.SetShouldCheckRetrieval (true);
  recovery_ctx.SetWriteExclusionTs (signal.timestamp);
  auto recovery_tx = store->Begin ();
  operation.Execute (recovery_ctx, *recovery_tx);
  recovery_tx->Rollback ();
  const auto recovered
      = operations::historical_surface_search_cache_internal::Find (pctx);
  REQUIRE (recovered);
  REQUIRE (recovered->current_memory_index.contains (10));
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval bounded fallback keeps eligible shared sibling",
           "[operations][graph][retrieval][sql][shared-embedding][supersession]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  store->Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
      "start_ts, created_at) VALUES(?, ?, ?, 'LONG_TERM', ?, ?)",
      { 11LL, 100LL, std::string ("shared"), 1100LL, 1100LL });
  SeedMemory (*store, 20, 200, VectorWithCosineToDim0 (0.92f), 2000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'supersedes', ?, ?)",
      { 20LL, 10LL, 1.0, 2000LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  pctx.UpsertRetrievalSurface (
      { 20, 200, 2000, 2000, 0, 0, 0, 0, "LONG_TERM", "replacement", "",
        -1.0, 0, 0.0, 0.0, 0.0, false, true,
        VectorWithCosineToDim0 (0.92f) });
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx);
  auto signal = MakeSignal (UnitVec (0), 3000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) { return candidate.memory_id == 10; })
           == ctx.GetRetrievedMemoryCandidates ().end ());
  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) { return candidate.memory_id == 11; })
           != ctx.GetRetrievedMemoryCandidates ().end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 1);
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval fallback pages past cosine family saturation",
           "[operations][graph][retrieval][sql][pagination][family]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  for (long long offset = 0; offset < 600; ++offset)
    {
      auto embedding = UnitVec (0);
      embedding[1] = static_cast<float> (offset + 1) / 1000000.0f;
      embedding.normalize ();
      SeedMemory (*store, 1000 + offset, 10000 + offset, embedding,
                  1100 + offset, "cosine-family");
    }
  constexpr long long kTargetMemoryId = 9000;
  SeedMemory (*store, kTargetMemoryId, 200,
              VectorWithCosineToDim0 (0.95f), 2000, "target");

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx);
  auto signal = MakeSignal (UnitVec (0), 100000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) {
                 return candidate.memory_id == kTargetMemoryId;
           })
           != ctx.GetRetrievedMemoryCandidates ().end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 2);
  const int seed_limit = std::max (1, core::RetrievalMaxResults (cfg.focus));
  const int seed_search_limit = core::RetrievalSeedSearchK (
      cfg.focus, cfg.sensitivity, cfg.stability, seed_limit);
  const int page_limit = core::RetrievalSeedSearchK (
      cfg.focus, cfg.sensitivity, cfg.stability, seed_search_limit);
  const auto materialized
      = operations::retrieval_trace::GetLastSqlFallbackMaterializedRowCount ();
  REQUIRE (materialized > static_cast<std::size_t> (page_limit));
  REQUIRE (materialized
           <= static_cast<std::size_t> (2 * page_limit));
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval fallback preserves overfetch after surface refresh",
           "[operations][graph][retrieval][sql][pagination][reconstruction]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  const int seed_limit = std::max (1, core::RetrievalMaxResults (cfg.focus));
  ProcessorContext pctx;
  operations::constructive_recall::ReconstructionUpdatePolicy policy;
  policy.update_current_surface = false;
  operations::historical_surface_search_cache_internal::
      SetCurrentSurfaceDatabaseCurrent (pctx, true);
  for (int offset = 0; offset < seed_limit; ++offset)
    {
      Eigen::VectorXf base = Eigen::VectorXf::Zero (kEmbeddingDim);
      constexpr float cosine = 0.90f;
      base[0] = cosine;
      base[1 + offset]
          = std::sqrt (std::max (0.0f, 1.0f - cosine * cosine));
      const long long memory_id = 1000 + offset;
      const long long embedding_id = 10000 + offset * 100;
      SeedMemory (*store, memory_id, embedding_id, base, 1000 + offset,
                  "reconstructed-family");
      auto reconstruction_tx = store->Begin ();
      REQUIRE (operations::constructive_recall::
                   AppendReconstructionWithEmbedding (
                       *reconstruction_tx, memory_id, UnitVec (0), {},
                       5000 + offset, 0.1, "surface-refresh", 1.0, 1.0,
                       policy, &pctx)
               > 0);
      const auto latest = operations::constructive_recall::
          LoadLatestReconstruction (*reconstruction_tx, memory_id);
      REQUIRE (latest.has_value ());
      reconstruction_tx->Commit ();
      pctx.UpsertRetrievalSurface (
          { memory_id, latest->embedding_id, 1000 + offset, 1000 + offset,
            0, 0, 0, 0, "LONG_TERM", "reconstructed-family", "", -1.0,
            0, 0.0, 0.0, 0.0, false, true, UnitVec (0) });
    }
  REQUIRE_FALSE (operations::historical_surface_search_cache_internal::
                     CurrentSurfaceDatabaseCurrent (pctx));

  constexpr long long kTargetMemoryId = 9000;
  SeedMemory (*store, kTargetMemoryId, 30000,
              VectorWithCosineToDim0 (0.75f), 2000, "target");
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx);

  auto signal = MakeSignal (UnitVec (0), 100000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) {
                 return candidate.memory_id == kTargetMemoryId;
               })
           != ctx.GetRetrievedMemoryCandidates ().end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 1);
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval fallback preserves overfetch after database reconstruction refresh",
           "[operations][graph][retrieval][sql][pagination][reconstruction]")
{
  cortext::testing::ScopedEnvVar disable_current (
      "CORTEXT_DISABLE_CURRENT_MEMORY_SURFACE_WRITES", "1");
  if (!operations::constructive_recall::CurrentSurfaceWritesDisabled ())
    {
      SKIP ("current-surface write hook is disabled in this build");
    }

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  const int seed_limit = std::max (1, core::RetrievalMaxResults (cfg.focus));
  ProcessorContext pctx;
  pctx.UpsertRetrievalSurface (
      { 99999, 99999, 1, 1, 0, 0, 0, 0, "LONG_TERM", "unrelated", "",
        -1.0, 0, 0.0, 0.0, 0.0, false, true, UnitVec (4) });

  for (int offset = 0; offset < seed_limit; ++offset)
    {
      Eigen::VectorXf base = Eigen::VectorXf::Zero (kEmbeddingDim);
      constexpr float cosine = 0.90f;
      base[0] = cosine;
      base[1 + offset]
          = std::sqrt (std::max (0.0f, 1.0f - cosine * cosine));
      const long long memory_id = 1000 + offset;
      SeedMemory (*store, memory_id, 10000 + offset, base, 1000 + offset,
                  "database-reconstructed-family");
    }
  for (int offset = 0; offset < seed_limit; ++offset)
    {
      auto reconstruction_tx = store->Begin ();
      REQUIRE (operations::constructive_recall::
                   AppendReconstructionWithEmbedding (
                       *reconstruction_tx, 1000 + offset, UnitVec (0), {},
                       5000 + offset, 0.1, "review", 1.0, 1.0)
               > 0);
      reconstruction_tx->Commit ();
    }

  constexpr long long kTargetMemoryId = 9000;
  SeedMemory (*store, kTargetMemoryId, 30000,
              VectorWithCosineToDim0 (0.75f), 2000, "target");
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx);

  auto signal = MakeSignal (UnitVec (0), 100000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) {
                 return candidate.memory_id == kTargetMemoryId;
               })
           != ctx.GetRetrievedMemoryCandidates ().end ());
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Historical search recovery rejects recycled context state",
           "[operations][graph][retrieval][cache][regression]")
{
  alignas (ProcessorContext) std::byte storage[sizeof (ProcessorContext)];
  auto *prior = std::construct_at (
      reinterpret_cast<ProcessorContext *> (storage));
  operations::historical_surface_search_cache_internal::
      SetCurrentSurfaceDatabaseCurrent (*prior, true);
  operations::historical_surface_search_cache_internal::
      SetProcessorSurfaceComplete (*prior, true);
  std::destroy_at (prior);

  auto *current = std::construct_at (
      reinterpret_cast<ProcessorContext *> (storage));
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      *current);
  REQUIRE (operations::historical_surface_search_cache_internal::
               RecoveryFailed (*current));
  REQUIRE_FALSE (operations::historical_surface_search_cache_internal::
                     CurrentSurfaceDatabaseCurrent (*current));
  const auto state
      = operations::historical_surface_search_cache_internal::Find (*current);
  REQUIRE (state != nullptr);
  REQUIRE_FALSE (state->processor_surface_complete);
  operations::historical_surface_search_cache_internal::Erase (*current);
  std::destroy_at (current);
}

TEST_CASE ("Graph retrieval bypasses stale valid cache for latest reconstructions",
           "[operations][graph][retrieval][cache][sql][reconstruction]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  const int seed_limit = std::max (1, core::RetrievalMaxResults (cfg.focus));
  const int seed_search_limit = core::RetrievalSeedSearchK (
      cfg.focus, cfg.sensitivity, cfg.stability, seed_limit);
  ProcessorContext pctx;
  std::vector<operations::historical_surface_search_cache_internal::Entry>
      historical_entries;
  std::vector<operations::historical_surface_search_cache_internal::Entry>
      current_entries;
  historical_entries.reserve (static_cast<std::size_t> (seed_search_limit + 1));
  current_entries.reserve (static_cast<std::size_t> (seed_search_limit + 1));

  for (int offset = 0; offset < seed_search_limit; ++offset)
    {
      Eigen::VectorXf base = VectorWithCosineAndDiverseResidual (
          0.90f, static_cast<std::uint64_t> (offset));
      const long long memory_id = 1000 + offset;
      const long long embedding_id = 10000 + offset;
      SeedMemory (*store, memory_id, embedding_id, base, 1000 + offset,
                  "stale-cache-family");
      pctx.UpsertRetrievalSurface (
          { memory_id, embedding_id, 1000 + offset, 1000 + offset, 0, 0,
            0, 0, "LONG_TERM", "stale-cache-family", "", -1.0, 0, 0.0,
            0.0, 0.0, false, true, base });
      historical_entries.push_back (
          { embedding_id, memory_id, 1000 + offset, "LONG_TERM",
            "stale-cache-family", base });
      current_entries.push_back (
          { embedding_id, memory_id, 0, std::string (), std::string (),
            base });
    }

  constexpr long long kTargetMemoryId = 9000;
  constexpr long long kTargetEmbeddingId = 30000;
  const Eigen::VectorXf target = VectorWithCosineToDim0 (0.75f);
  SeedMemory (*store, kTargetMemoryId, kTargetEmbeddingId, target, 2000,
              "target");
  pctx.UpsertRetrievalSurface (
      { kTargetMemoryId, kTargetEmbeddingId, 2000, 2000, 0, 0, 0, 0,
        "LONG_TERM", "target", "", -1.0, 0, 0.0, 0.0, 0.0, false, true,
        target });
  historical_entries.push_back (
      { kTargetEmbeddingId, kTargetMemoryId, 2000, "LONG_TERM", "target",
        target });
  current_entries.push_back (
      { kTargetEmbeddingId, kTargetMemoryId, 0, std::string (),
        std::string (), target });

  REQUIRE (operations::historical_surface_search_cache_internal::Reset (
      pctx, std::move (historical_entries), std::move (current_entries)));
  operations::historical_surface_search_cache_internal::
      SetCurrentSurfaceDatabaseCurrent (pctx, true);

  operations::constructive_recall::ReconstructionUpdatePolicy policy;
  policy.update_current_surface = false;
  for (int offset = 0; offset < seed_search_limit; ++offset)
    {
      auto reconstruction_tx = store->Begin ();
      const long long memory_id = 1000 + offset;
      REQUIRE (operations::constructive_recall::
                   AppendReconstructionWithEmbedding (
                       *reconstruction_tx, memory_id, UnitVec (0), {},
                       5000 + offset, 0.1, "stale-cache", 1.0, 1.0, policy,
                       &pctx)
               > 0);
      const auto latest = operations::constructive_recall::
          LoadLatestReconstruction (*reconstruction_tx, memory_id);
      REQUIRE (latest.has_value ());
      reconstruction_tx->Commit ();
      pctx.UpsertRetrievalSurface (
          { memory_id, latest->embedding_id, 1000 + offset, 1000 + offset,
            0, 0, 0, 0, "LONG_TERM", "stale-cache-family", "", -1.0, 0,
            0.0, 0.0, 0.0, false, true, UnitVec (0) });
    }
  const auto stale_state
      = operations::historical_surface_search_cache_internal::Find (pctx);
  REQUIRE (stale_state != nullptr);
  REQUIRE_FALSE (stale_state->recovery_failed);
  REQUIRE_FALSE (stale_state->current_surface_database_current);

  auto signal = MakeSignal (UnitVec (0), 100000);
  signal.retention = Retention::Ephemeral;
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetShouldCheckRetrieval (true);
  ctx.SetWriteExclusionTs (signal.timestamp);

  GraphAugmentedRetrieveCandidates operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Rollback ();

  REQUIRE (std::find_if (
               ctx.GetRetrievedMemoryCandidates ().begin (),
               ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) {
                 return candidate.memory_id == kTargetMemoryId;
               })
           != ctx.GetRetrievedMemoryCandidates ().end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 0);

  operations::historical_surface_search_cache_internal::
      SetProcessorSurfaceComplete (pctx, true);
  operations::historical_surface_search_cache_internal::MarkRecoveryFailed (
      pctx, true);
  OperationContext recovery_ctx (signal, pctx, cfg, store.get ());
  recovery_ctx.SetShouldCheckRetrieval (true);
  recovery_ctx.SetWriteExclusionTs (signal.timestamp);
  auto recovery_tx = store->Begin ();
  operation.Execute (recovery_ctx, *recovery_tx);
  recovery_tx->Rollback ();
  REQUIRE (std::find_if (
               recovery_ctx.GetRetrievedMemoryCandidates ().begin (),
               recovery_ctx.GetRetrievedMemoryCandidates ().end (),
               [] (const auto &candidate) {
                 return candidate.memory_id == kTargetMemoryId;
               })
           != recovery_ctx.GetRetrievedMemoryCandidates ().end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 0);
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Graph retrieval keeps eligible sibling of superseded shared embedding",
           "[operations][graph][retrieval][cache][shared-embedding][supersession]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  SeedMemory (*store, 10, 100, UnitVec (0), 1000);
  store->Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
      "start_ts, created_at) VALUES(?, ?, ?, 'LONG_TERM', ?, ?)",
      { 11LL, 100LL, std::string ("shared"), 1100LL, 1100LL });
  SeedMemory (*store, 20, 200, VectorWithCosineToDim0 (0.92f), 2000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, "
      "weight, last_reinforced) VALUES(?, ?, 'supersedes', ?, ?)",
      { 20LL, 10LL, 1.0, 2000LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  const auto out = processor.Process (MakeSignal (UnitVec (0), 3000));
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 10LL)
           == out.candidate_memory_ids.end ());
  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (), 11LL)
           != out.candidate_memory_ids.end ());
  REQUIRE (operations::retrieval_trace::GetLastSqlFallbackQueryCount () == 0);
}

TEST_CASE ("Historical search cache append retains memory alternative",
           "[operations][graph][retrieval][cache][append]")
{
  ProcessorContext pctx;
  REQUIRE (
      operations::historical_surface_search_cache_internal::Reset (pctx, {}));
  operations::historical_surface_search_cache_internal::Append (
      pctx, { 100, 10, 1000, "LONG_TERM", "opaque", UnitVec (0) });

  const auto state
      = operations::historical_surface_search_cache_internal::Find (pctx);
  REQUIRE (state != nullptr);
  REQUIRE (state->entries.size () == 1);
  REQUIRE (state->entries.front ().memory_references.size () == 1);
  REQUIRE (state->entries.front ().memory_references.front ().memory_id == 10);
  REQUIRE (state->long_term_entry_indices
           == std::vector<std::size_t>{ 0 });
  REQUIRE (state->long_term_index_positions.at (0) == 0);
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Historical search cache tracks only long-term distance rows",
           "[operations][graph][retrieval][cache][eligibility]")
{
  using operations::historical_surface_search_cache_internal::Entry;
  ProcessorContext pctx;
  REQUIRE (operations::historical_surface_search_cache_internal::Reset (
      pctx,
      { Entry{ 100, 10, 1000, "LONG_TERM", "a", UnitVec (0) },
        Entry{ 101, 11, 1001, "SIGNAL", "b", UnitVec (1) },
        Entry{ 102, 12, 1002, "LONG_TERM", "c", UnitVec (2) } }));

  auto state
      = operations::historical_surface_search_cache_internal::Find (pctx);
  REQUIRE (state != nullptr);
  REQUIRE (state->long_term_entry_indices
           == std::vector<std::size_t>{ 0, 2 });

  operations::historical_surface_search_cache_internal::RemoveEmbedding (
      pctx, 100);
  state = operations::historical_surface_search_cache_internal::Find (pctx);
  REQUIRE (state->entries.size () == 2);
  REQUIRE (state->entries[0].embedding_id == 102);
  REQUIRE (state->long_term_entry_indices
           == std::vector<std::size_t>{ 0 });
  REQUIRE (state->long_term_index_positions.at (0) == 0);

  operations::historical_surface_search_cache_internal::Append (
      pctx, { 103, 13, 1003, "SIGNAL", "d", UnitVec (3) });
  REQUIRE (state->long_term_entry_indices
           == std::vector<std::size_t>{ 0 });
  operations::historical_surface_search_cache_internal::Erase (pctx);
}
