#include <any>
#include "test_helpers.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/clock.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include "../src/operations/association_fanout_cache_internal.hpp"
#include "../src/operations/bounded_activation_shadow_internal.hpp"
#include "../src/operations/emotional_metadata_cache_internal.hpp"
#include "../src/operations/historical_surface_search_cache_internal.hpp"
#include "../src/operations/rif_state_internal.hpp"
#include "../src/operations/consolidation_throughput_state_internal.hpp"
#include "../src/operations/signal_record_rollback_internal.hpp"
#include <thread>

using namespace cortext;

namespace cortext::testing
{
std::size_t ProcessorRollbackSnapshotOwnerCountForTest ();
std::size_t ProcessorRollbackSnapshotReuseCountForTest ();
std::size_t SignalRecordRollbackBackupCountForTest ();
std::size_t SignalRecordRollbackCopiedRecordCountForTest ();
void SetSignalRecordSnapshotSetupThrowStageForTest (int stage);
void SetRifActiveEpochPublishFailureStageForTest (int stage);
void SetRifActiveEpochPublishFailureMaskForTest (unsigned int mask);
void SetSQLiteCheckpointFailureOnceForTest ();
}

namespace cortext::store
{
void DebugApplyCoreMigrationsThroughForTest (Store &store,
                                             int64_t maximum_id);
}

static std::shared_ptr<operations::execution_cache_sidecar_internal::State>
ExecutionCacheState (const ProcessorContext &ctx)
{
  return operations::execution_cache_sidecar_internal::Find (ctx);
}

struct InsertOp : IOperation
{
  void
  Execute (OperationContext & /*ctx*/, Transaction &tx) const override
  {
    tx.Execute ("INSERT INTO t(v) VALUES(?)", { std::string{ "hello" } });
  }
};

struct InsertDeferredViolationOp : IOperation
{
  void
  Execute (OperationContext & /*ctx*/, Transaction &tx) const override
  {
    tx.Execute ("INSERT INTO deferred_child(parent_id) VALUES(?)", { 999LL });
  }
};

struct BoundedShadowNoOp : IOperation
{
  void
  Execute (OperationContext &, Transaction &) const override
  {
  }
};

struct BoundedShadowThrowOnceOp : IOperation
{
  void
  Execute (OperationContext &, Transaction &) const override
  {
    if (fail_once)
      {
        fail_once = false;
        throw std::runtime_error ("bounded shadow precommit failure");
      }
  }

  mutable bool fail_once = true;
};

struct BoundedShadowCancelOp : IOperation
{
  void
  Execute (OperationContext &, Transaction &) const override
  {
    internal::ThrowCancellation ();
  }
};

Signal
MakeBoundedShadowSignal (std::uint64_t event_index,
                         std::string source_id = "source-a",
                         std::string modality = "text")
{
  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (16);
  for (Eigen::Index dimension = 0; dimension < signal.embedding.size ();
       ++dimension)
    signal.embedding[dimension] = static_cast<float> (
        std::sin ((event_index + 1) * (dimension + 1) * 0.017)
        + std::cos ((event_index + 3) * (dimension + 2) * 0.011));
  signal.timestamp = 1000 + event_index;
  signal.source_id = std::move (source_id);
  signal.modality = std::move (modality);
  signal.mimetype = signal.modality == "audio" ? "audio/pcm;format=f32"
                    : signal.modality == "image" ? "image/png"
                                                  : "text/plain";
  return signal;
}

struct AdvanceRifEpochOp : IOperation
{
  void
  Execute (OperationContext &ctx, Transaction &tx) const override
  {
    auto sidecar = operations::execution_cache_sidecar_internal::Ensure (
        ctx.GetProcessorContext ());
    const auto result = operations::rif_state_internal::AdvanceRecovery (
        tx, static_cast<long long> (ctx.GetSignal ().timestamp), 10000.0,
        sidecar->rif_active_epoch.calibration_memory_ids);
    sidecar->rif_active_epoch.calibration_memory_ids.clear ();
    operations::rif_active_epoch_cache_internal::StageClock (
        sidecar->rif_active_epoch, result.clock.generation,
        result.clock.log_factor, result.clock.last_ts);
    operations::rif_active_epoch_cache_internal::StageMemories (
        sidecar->rif_active_epoch, result.changed_memory_ids);
  }
};

struct AdvanceRifAndFailCommitOnceOp : IOperation
{
  void
  Execute (OperationContext &ctx, Transaction &tx) const override
  {
    AdvanceRifEpochOp {}.Execute (ctx, tx);
    if (fail_once)
      {
        fail_once = false;
        tx.Execute ("INSERT INTO deferred_child(parent_id) VALUES(999)");
      }
  }

  mutable bool fail_once = true;
};

struct PrimeRifEpochBoundaryOp : IOperation
{
  void
  Execute (OperationContext &ctx, Transaction &) const override
  {
    if (ctx.GetSignal ().force_consolidation || primed)
      return;
    auto sidecar = operations::execution_cache_sidecar_internal::Ensure (
        ctx.GetProcessorContext ());
    sidecar->rif_active_epoch.event_count
        = operations::rif_active_epoch_cache_internal::kEventLimit - 1;
    primed = true;
  }

  mutable bool primed = false;
};

struct ConsolidationInsertAndFailCommitOnceOp : IOperation
{
  void
  Execute (OperationContext &ctx, Transaction &tx) const override
  {
    captured = &ctx.GetProcessorContext ();
    if (!ctx.GetSignal ().force_consolidation)
      return;
    tx.Execute ("INSERT INTO consolidation_derived(id) VALUES(1)");
    if (fail_once)
      {
        fail_once = false;
        tx.Execute ("INSERT INTO deferred_child(parent_id) VALUES(999)");
      }
  }

  mutable bool fail_once = true;
  mutable ProcessorContext *captured = nullptr;
};

struct ConsolidationInsertOp : IOperation
{
  void
  Execute (OperationContext &ctx, Transaction &tx) const override
  {
    if (ctx.GetSignal ().force_consolidation)
      tx.Execute ("INSERT OR IGNORE INTO consolidation_derived(id) VALUES(1)");
  }
};

struct CustomRifReadWriteOp : IOperation
{
  explicit CustomRifReadWriteOp (double *observed_strength,
                                 double *observed_suppression)
      : observed_strength (observed_strength),
        observed_suppression (observed_suppression)
  {
  }

  void
  Execute (OperationContext &, Transaction &tx) const override
  {
    const auto rows = tx.Execute (
        "SELECT * FROM memories WHERE memory_id = 1");
    if (observed_strength)
      *observed_strength
          = std::any_cast<double> (rows[0].at ("strength"));
    if (observed_suppression)
      *observed_suppression
          = std::any_cast<double> (rows[0].at ("suppression"));
    tx.Execute (
        "UPDATE memories SET strength = strength + 0.1 WHERE memory_id = 1");
  }

  double *observed_strength = nullptr;
  double *observed_suppression = nullptr;
};

struct CustomIndirectRifReadOp : IOperation
{
  explicit CustomIndirectRifReadOp (double *observed_strength)
      : observed_strength (observed_strength)
  {
  }

  void
  Execute (OperationContext &, Transaction &tx) const override
  {
    tx.Execute (
        "CREATE TEMP VIEW custom_memory_view AS SELECT * FROM memories");
    const auto rows = tx.Execute (
        "SELECT v.* FROM custom_memory_view v WHERE memory_id = 1");
    if (observed_strength)
      *observed_strength
          = std::any_cast<double> (rows[0].at ("strength"));
  }

  double *observed_strength = nullptr;
};

struct CustomMutationCacheProbeOp : IOperation
{
  CustomMutationCacheProbeOp (bool *fanout_current, bool *search_invalidated,
                              bool *supersession_invalidated)
      : fanout_current (fanout_current),
        search_invalidated (search_invalidated),
        supersession_invalidated (supersession_invalidated)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction &tx) const override
  {
    auto &p_ctx = ctx.GetProcessorContext ();
    const auto prior_sidecar = ExecutionCacheState (p_ctx);
    REQUIRE (prior_sidecar);
    prior_sidecar->supersession_eligibility.valid = true;
    prior_sidecar->supersession_eligibility.activation_ts_by_target[999]
        = 999;
    tx.Execute (
        "INSERT INTO associations(source_memory_id, target_memory_id, "
        "edge_type, weight, last_reinforced) "
        "VALUES(1, 2, 'similar_to', 0.75, 2000)");
    const auto &fanout = operations::association_fanout_cache::Ensure (
        ctx.GetStore (), p_ctx);
    if (fanout_current)
      {
        const auto it = fanout->out_by_source.find (1);
        *fanout_current
            = it != fanout->out_by_source.end () && !it->second.empty ()
              && it->second.front ().memory_id == 2;
      }
    if (search_invalidated)
      *search_invalidated
          = !operations::historical_surface_search_cache_internal::Find (p_ctx);
    if (supersession_invalidated)
      {
        const auto sidecar = ExecutionCacheState (p_ctx);
        *supersession_invalidated
            = sidecar && sidecar.get () != prior_sidecar.get ()
              && !sidecar->supersession_eligibility
                       .activation_ts_by_target.contains (999);
      }
  }

  bool *fanout_current = nullptr;
  bool *search_invalidated = nullptr;
  bool *supersession_invalidated = nullptr;
};

struct CaptureAssociationSupersessionCoverageOp : IOperation
{
  explicit CaptureAssociationSupersessionCoverageOp (bool *covered)
      : covered (covered)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction &) const override
  {
    const auto state
        = operations::historical_surface_search_cache_internal::Find (
            ctx.GetProcessorContext ());
    if (covered)
      *covered = state && state->supersession_entry_by_memory.contains (2)
                 && state->current_memory_index.contains (2);
  }

  bool *covered = nullptr;
};

struct UpdateFirstWorkingSlotStrengthOp : IOperation
{
  explicit UpdateFirstWorkingSlotStrengthOp (double strength,
                                             bool *updated,
                                             ProcessorContext **captured)
      : strength (strength), updated (updated), captured (captured)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction &) const override
  {
    auto &p_ctx = ctx.GetProcessorContext ();
    if (captured)
      *captured = &p_ctx;
    if (p_ctx.wm_slots.empty ())
      return;
    p_ctx.wm_slots.front ().strength = strength;
    p_ctx.wm_slots.front ().metadata_dirty = true;
    p_ctx.wm_slots_dirty = true;
    if (updated)
      *updated = true;
  }

  double strength = 0.0;
  bool *updated = nullptr;
  ProcessorContext **captured = nullptr;
};

struct RecordOrderOp : IOperation
{
  explicit RecordOrderOp (std::vector<int> *order, int id)
      : order (order), id (id)
  {
  }

  void
  Execute (OperationContext & /*ctx*/, Transaction & /*tx*/) const override
  {
    if (order)
      {
        order->push_back (id);
      }
  }

  std::vector<int> *order;
  int id = 0;
};

struct CaptureCacheIdentityOp : IOperation
{
  explicit CaptureCacheIdentityOp (bool *retained) : retained (retained) {}

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    const auto &surface = ctx.GetProcessorContext ().retrieval_surface_cache;
    ++call_count;
    if (call_count == 1)
      {
        first_data = surface.data ();
        first_size = surface.size ();
        auto &processor_context = ctx.GetProcessorContext ();
        auto &ids = processor_context.index_store["large/stable"];
        ids.reserve (4096);
        for (long long id = 1; id <= 4096; ++id)
          {
            ids.push_back (id);
            processor_context.index_reverse[id] = "large/stable";
            processor_context.procedural_store["large/stable"][id] = 0.5;
          }
      }
    else if (retained)
      {
        *retained = first_data == surface.data () && first_size == surface.size ();
      }
  }

  bool *retained = nullptr;
  mutable int call_count = 0;
  mutable const ProcessorContext::RetrievalSurfaceEntry *first_data = nullptr;
  mutable std::size_t first_size = 0;
};

struct ExerciseJournalAwareCommitFailureOp : IOperation
{
  explicit ExerciseJournalAwareCommitFailureOp (bool *restored)
      : restored (restored)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction &tx) const override
  {
    auto &processor_context = ctx.GetProcessorContext ();
    ++call_count;
    if (call_count == 1)
      {
        processor_context.index_store["stable"] = { 11LL, 12LL };
        processor_context.index_reverse[11LL] = "stable";
        processor_context.procedural_store["stable"][11LL] = 0.75;
        return;
      }
    if (call_count == 2)
      {
        processor_context.retrieval_surface_cache.clear ();
        processor_context.retrieval_surface_index.clear ();
        processor_context.retrieval_surface_embedding_index.clear ();
        processor_context.association_fanout_cache = {};
        operations::execution_cache_sidecar_internal::Erase (
            processor_context);
        operations::signal_record_rollback_internal::
            PreserveSparseIndexBeforeInsert (
                processor_context, "stable", 99LL);
        operations::signal_record_rollback_internal::
            PreserveProceduralValueBeforeUpdate (
                processor_context, "stable", 11LL);
        processor_context.index_store["stable"].push_back (99LL);
        processor_context.index_reverse[99LL] = "stable";
        processor_context.procedural_store["stable"][11LL] = 0.0;
        tx.Execute ("INSERT INTO deferred_child(parent_id) VALUES(?)",
                    { 999LL });
        return;
      }
    if (restored)
      {
        (void)operations::association_fanout_cache::Ensure (
            ctx.GetStore (), processor_context);
        const auto sidecar = ExecutionCacheState (processor_context);
        *restored
            = processor_context.retrieval_surface_index.contains (41LL)
              && processor_context.retrieval_surface_embedding_index.contains (
                  51LL)
              && sidecar && sidecar->emotional_metadata.valid
              && sidecar->emotional_metadata.rows_by_memory
                     .contains (41LL)
              && sidecar->emotional_metadata
                     .memory_ids_by_embedding.at (51LL)
                     == std::vector<long long> { 41LL }
              && processor_context.association_fanout_cache.valid
              && processor_context.association_fanout_cache.in_by_target
                     .at (41LL).front ().memory_id
                     == 42LL
              && sidecar->supersession_eligibility.valid
              && sidecar->supersession_eligibility.activation_ts_by_target.at (
                     41LL)
                     == 1100LL
              && processor_context.index_store.at ("stable")
                     == std::vector<long long> { 11LL, 12LL }
              && processor_context.index_reverse.at (11LL) == "stable"
              && processor_context.procedural_store.at ("stable").at (11LL)
                     == 0.75;
      }
  }

  bool *restored = nullptr;
  mutable int call_count = 0;
};

struct ExerciseJournalAwareReadOnlyRollbackOp : IOperation
{
  explicit ExerciseJournalAwareReadOnlyRollbackOp (bool *restored)
      : restored (restored)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &processor_context = ctx.GetProcessorContext ();
    ++call_count;
    if (call_count == 1)
      {
        processor_context.index_store["stable"] = { 7LL, 8LL };
        return;
      }
    if (call_count == 2)
      {
        processor_context.retrieval_surface_cache.clear ();
        processor_context.retrieval_surface_index.clear ();
        processor_context.retrieval_surface_embedding_index.clear ();
        operations::execution_cache_sidecar_internal::Erase (
            processor_context);
        operations::signal_record_rollback_internal::
            PreserveSparseIndexBeforeInsert (
                processor_context, "stable", 99LL);
        processor_context.index_store["stable"].push_back (99LL);
        processor_context.index_reverse[99LL] = "stable";
        return;
      }
    if (restored)
      {
        const auto sidecar = ExecutionCacheState (processor_context);
        *restored
            = processor_context.retrieval_surface_index.contains (41LL)
              && processor_context.retrieval_surface_embedding_index.contains (
                  51LL)
              && sidecar && sidecar->emotional_metadata.valid
              && sidecar->emotional_metadata.rows_by_memory
                     .contains (41LL)
              && sidecar->emotional_metadata
                     .memory_ids_by_embedding.at (51LL)
                     == std::vector<long long> { 41LL }
              && processor_context.index_store.at ("stable")
                     == std::vector<long long> { 7LL, 8LL };
      }
  }

  bool *restored = nullptr;
  mutable int call_count = 0;
};

struct CaptureAccumulatorCountOp : IOperation
{
  explicit CaptureAccumulatorCountOp (std::size_t *count) : count (count) {}

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    if (count)
      {
        *count = ctx.GetProcessorContext ().accumulator_states.size ();
      }
  }

  std::size_t *count;
};

struct CaptureBaseLineageAcrossFailureOp : IOperation
{
  explicit CaptureBaseLineageAcrossFailureOp (long long *base_embedding_id)
      : base_embedding_id (base_embedding_id)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ++call_count;
    if (call_count == 2)
      throw std::runtime_error ("force lineage rollback");
    if (base_embedding_id)
      *base_embedding_id
          = operations::historical_surface_search_cache_internal::
              BaseEmbeddingIdForMemory (
                  ctx.GetProcessorContext (), 10LL, 0);
  }

  long long *base_embedding_id = nullptr;
  mutable int call_count = 0;
};

struct ExerciseRollbackCacheRestoreOp : IOperation
{
  ExerciseRollbackCacheRestoreOp (bool *retrieval_restored,
                                  bool *volatile_restored)
      : retrieval_restored (retrieval_restored),
        volatile_restored (volatile_restored)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &processor_context = ctx.GetProcessorContext ();
    ++call_count;
    if (call_count == 1)
      {
        ProcessorContext::RetrievalSurfaceEntry cache_only_entry;
        cache_only_entry.memory_id = 77LL;
        cache_only_entry.embedding_id = 88LL;
        cache_only_entry.created_at = 1500LL;
        cache_only_entry.start_ts = 1500LL;
        cache_only_entry.kind = "LONG_TERM";
        cache_only_entry.source_id = "cache/only";
        cache_only_entry.modality = "text";
        cache_only_entry.embedding = Eigen::VectorXf::Constant (256, 0.25f);
        processor_context.UpsertRetrievalSurface (
            std::move (cache_only_entry));
        operations::emotional_metadata_cache_internal::Upsert (
            processor_context,
            { 77LL, 88LL, 1500LL, true, 0.8, 0.7, 2.0, 3, 0.5 });
        const auto sidecar
            = operations::execution_cache_sidecar_internal::Ensure (
                processor_context);
        sidecar->emotional_fixed_point.valid = true;
        sidecar->emotional_fixed_point.emotional_input_generation
            = sidecar->emotional_metadata.cascade_input_generation;
        sidecar->emotional_fixed_point.recent_window_ts = 1234LL;
        sidecar->supersession_eligibility.valid = true;
        sidecar->supersession_eligibility.activation_ts_by_target[41LL]
            = 1100LL;
        processor_context.index_store["stable"] = { 11LL, 12LL };
        processor_context.index_reverse[11LL] = "stable";
        processor_context.procedural_store["stable"][11LL] = 0.75;
        return;
      }
    if (call_count == 2)
      {
        processor_context.retrieval_surface_cache.clear ();
        processor_context.retrieval_surface_index.clear ();
        processor_context.retrieval_surface_embedding_index.clear ();
        const auto sidecar
            = operations::execution_cache_sidecar_internal::Ensure (
                processor_context);
        sidecar->emotional_metadata.rows_by_memory.clear ();
        sidecar->emotional_metadata.source_query_order.clear ();
        sidecar->emotional_fixed_point.valid = false;
        sidecar->emotional_fixed_point.recent_window_ts = 9999LL;
        sidecar->supersession_eligibility.valid = false;
        sidecar->supersession_eligibility.activation_ts_by_target[41LL]
            = 9999LL;
        processor_context.index_store["stable"] = { 99LL };
        processor_context.index_reverse[11LL] = "mutated";
        processor_context.procedural_store["stable"][11LL] = 0.0;
        throw std::runtime_error ("force rollback");
      }

    if (retrieval_restored)
      {
        const auto sidecar = ExecutionCacheState (processor_context);
        *retrieval_restored
            = processor_context.retrieval_surface_index.contains (41LL)
              && processor_context.retrieval_surface_index.contains (77LL)
              && processor_context.retrieval_surface_embedding_index.contains (
                  51LL)
              && processor_context.retrieval_surface_embedding_index.contains (
                  88LL)
              && processor_context.retrieval_surface_cache.size () == 2
              && processor_context.retrieval_surface_cache[0].memory_id == 41LL
              && processor_context.retrieval_surface_cache[1].memory_id == 77LL
              && sidecar && sidecar->emotional_metadata.valid
              && sidecar->emotional_metadata.rows_by_memory
                     .contains (41LL)
              && sidecar->emotional_metadata.rows_by_memory
                     .contains (77LL)
              && sidecar->emotional_metadata
                     .source_query_order
                     == std::vector<long long> { 77LL }
              && sidecar->emotional_fixed_point.valid
              && sidecar->emotional_fixed_point
                         .emotional_input_generation
                     == sidecar->emotional_metadata.cascade_input_generation
              && sidecar->emotional_fixed_point
                         .recent_window_ts
                     == 1234LL
              && sidecar->supersession_eligibility.valid
              && sidecar->supersession_eligibility.activation_ts_by_target.at (
                     41LL)
                     == 1100LL;
      }
    if (volatile_restored)
      {
        const auto index_it = processor_context.index_store.find ("stable");
        const auto reverse_it = processor_context.index_reverse.find (11LL);
        const auto procedural_it
            = processor_context.procedural_store.find ("stable");
        *volatile_restored
            = index_it != processor_context.index_store.end ()
              && index_it->second == std::vector<long long> { 11LL, 12LL }
              && reverse_it != processor_context.index_reverse.end ()
              && reverse_it->second == "stable"
              && procedural_it != processor_context.procedural_store.end ()
              && procedural_it->second.contains (11LL)
              && procedural_it->second.at (11LL) == 0.75;
      }
  }

  bool *retrieval_restored;
  bool *volatile_restored;
  mutable int call_count = 0;
};

struct ExerciseWholeContextRollbackOp : IOperation
{
  explicit ExerciseWholeContextRollbackOp (bool *restored)
      : restored (restored)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &processor_context = ctx.GetProcessorContext ();
    ++call_count;
    if (call_count == 1)
      {
        AccumulatorState active;
        active.n_signals = 1;
        SignalRecord active_record;
        active_record.payload.assign (8, 0x11);
        active.signals.push_back (std::move (active_record));
        processor_context.accumulator_states["test"] = std::move (active);
        AccumulatorState foreign;
        foreign.n_signals = 7;
        foreign.mu_acc = Eigen::VectorXf::Constant (256, 0.125f);
        foreign.primary_modality = "audio";
        processor_context.accumulator_states["foreign/source"]
            = std::move (foreign);
        processor_context.recent_scores = { 0.25, 0.75 };
        processor_context.soft_anchor_next_id = 17;
        return;
      }
    if (call_count == 2)
      {
        processor_context.accumulator_states.at ("test")
            .signals.front ().payload.assign (8, 0xFF);
        processor_context.accumulator_states.at ("test").signals.clear ();
        processor_context.accumulator_states.erase ("foreign/source");
        processor_context.accumulator_states["unexpected/source"].n_signals
            = 99;
        processor_context.recent_scores.clear ();
        processor_context.soft_anchor_next_id = 99;
        throw std::runtime_error ("force whole-context rollback");
      }

    const auto accumulator_it
        = processor_context.accumulator_states.find ("foreign/source");
    if (restored)
      {
        *restored
            = accumulator_it != processor_context.accumulator_states.end ()
              && accumulator_it->second.n_signals == 7
              && accumulator_it->second.mu_acc.size () == 256
              && accumulator_it->second.mu_acc[0] == 0.125f
              && accumulator_it->second.primary_modality == "audio"
              && processor_context.accumulator_states.at ("test")
                         .signals.front ().payload
                     == std::vector<unsigned char> (8, 0x11)
              && !processor_context.accumulator_states.contains (
                  "unexpected/source")
              && processor_context.recent_scores
                     == std::deque<double> { 0.25, 0.75 }
              && processor_context.soft_anchor_next_id == 17;
      }
  }

  bool *restored;
  mutable int call_count = 0;
};

struct ExercisePendingUnitRollbackOwnershipOp : IOperation
{
  ExercisePendingUnitRollbackOwnershipOp (std::size_t record_count,
                                          bool *restored_without_copy)
      : record_count (record_count),
        restored_without_copy (restored_without_copy)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &processor_context = ctx.GetProcessorContext ();
    ++call_count;
    if (call_count == 1)
      {
        AccumulatorState pending;
        pending.n_signals = static_cast<int> (record_count);
        for (std::size_t index = 0; index < record_count; ++index)
          {
            SignalRecord record;
            record.embedding = Eigen::VectorXf::Constant (
                256, static_cast<float> (index + 1));
            record.payload.assign (128, static_cast<unsigned char> (index));
            record.serial_position = static_cast<int> (index);
            pending.signals.push_back (std::move (record));
          }
        auto [inserted, _]
            = processor_context.accumulator_states.insert_or_assign (
                "test", std::move (pending));
        first_payload = inserted->second.signals.front ().payload.data ();
        middle_payload
            = inserted->second.signals[record_count / 2].payload.data ();
        last_payload = inserted->second.signals.back ().payload.data ();
        ProcessorContext::WMSlot slot;
        slot.memory_id = 91;
        slot.signal_records = inserted->second.signals;
        slot.blob_ids.push_back (std::vector<unsigned char> (32, 0x5A));
        processor_context.wm_slots.push_back (std::move (slot));
        wm_first_payload = processor_context.wm_slots.front ()
                               .signal_records.front ().payload.data ();
        wm_first_blob
            = processor_context.wm_slots.front ().blob_ids.front ().data ();
        return;
      }
    if (call_count == 2)
      {
        auto &pending = processor_context.accumulator_states.at (
            "test");
        SignalRecord appended;
        appended.embedding = Eigen::VectorXf::Ones (256);
        appended.payload.assign (128, 0xA5);
        appended.serial_position = static_cast<int> (pending.signals.size ());
        pending.signals.push_back (std::move (appended));
        pending.n_signals += 1;
        pending.s_sum = 99.0;
        throw std::runtime_error ("force pending-unit rollback");
      }

    const auto &signals = processor_context.accumulator_states.at (
        "test").signals;
    if (restored_without_copy)
      {
        *restored_without_copy
            = signals.size () == record_count
              && signals.front ().payload.data () == first_payload
              && signals[record_count / 2].payload.data ()
                     == middle_payload
              && signals.back ().payload.data () == last_payload
              && processor_context.wm_slots.size () == 1
              && processor_context.wm_slots.front ()
                         .signal_records.front ().payload.data ()
                     == wm_first_payload
              && processor_context.wm_slots.front ()
                         .blob_ids.front ().data ()
                     == wm_first_blob;
      }
  }

  std::size_t record_count = 0;
  bool *restored_without_copy = nullptr;
  mutable int call_count = 0;
  mutable const unsigned char *first_payload = nullptr;
  mutable const unsigned char *middle_payload = nullptr;
  mutable const unsigned char *last_payload = nullptr;
  mutable const unsigned char *wm_first_payload = nullptr;
  mutable const unsigned char *wm_first_blob = nullptr;
};

struct ExerciseFailingFlushRecordRollbackOp : IOperation
{
  explicit ExerciseFailingFlushRecordRollbackOp (bool *restored)
      : restored (restored)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &processor_context = ctx.GetProcessorContext ();
    ++call_count;
    if (call_count == 1)
      {
        AccumulatorState pending;
        pending.n_signals = 3;
        for (int index = 0; index < 3; ++index)
          {
            SignalRecord record;
            record.embedding = Eigen::VectorXf::Constant (
                256, static_cast<float> (index + 1));
            record.payload.assign (16,
                                   static_cast<unsigned char> (index + 10));
            record.serial_position = index;
            pending.signals.push_back (std::move (record));
          }
        processor_context.accumulator_states["test"] = std::move (pending);
        AccumulatorState unrelated;
        unrelated.n_signals = 4096;
        for (int index = 0; index < 4096; ++index)
          {
            SignalRecord record;
            record.payload.assign (8, 0xA5);
            unrelated.signals.push_back (std::move (record));
          }
        processor_context.accumulator_states["unrelated/large"]
            = std::move (unrelated);
        unrelated_first_payload
            = processor_context.accumulator_states.at ("unrelated/large")
                  .signals.front ()
                  .payload.data ();
        ProcessorContext::WMSlot unrelated_slot;
        unrelated_slot.memory_id = 404;
        for (int index = 0; index < 4096; ++index)
          {
            SignalRecord record;
            record.payload.assign (8, 0x5A);
            unrelated_slot.signal_records.push_back (std::move (record));
          }
        processor_context.wm_slots.push_back (std::move (unrelated_slot));
        unrelated_wm_first_payload
            = processor_context.wm_slots.front ()
                  .signal_records.front ().payload.data ();
        return;
      }
    if (call_count == 2)
      {
        operations::signal_record_rollback_internal::EnsureBackedUp (
            processor_context);
        auto &records = processor_context.accumulator_states.at (
            "test").signals;
        records.front ().payload.assign (4, 0xFF);
        records.front ().blob_id.assign (32, 0xEE);
        records.clear ();
        throw std::runtime_error ("force failing flush rollback");
      }

    const auto &records = processor_context.accumulator_states.at (
        "test").signals;
    if (restored)
      {
        *restored = records.size () == 3
                    && records.front ().payload
                           == std::vector<unsigned char> (16, 10)
                    && records.front ().blob_id.empty ()
                    && records.back ().payload
                           == std::vector<unsigned char> (16, 12)
                    && processor_context.accumulator_states
                               .at ("unrelated/large")
                               .signals.front ()
                               .payload.data ()
                           == unrelated_first_payload;
        *restored
            = *restored && processor_context.wm_slots.front ()
                                  .signal_records.front ().payload.data ()
                              == unrelated_wm_first_payload;
      }
  }

  bool *restored = nullptr;
  mutable int call_count = 0;
  mutable const unsigned char *unrelated_first_payload = nullptr;
  mutable const unsigned char *unrelated_wm_first_payload = nullptr;
};

struct ExerciseSnapshotSetupExceptionSafetyOp : IOperation
{
  explicit ExerciseSnapshotSetupExceptionSafetyOp (bool *restored)
      : restored (restored)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &processor_context = ctx.GetProcessorContext ();
    ++call_count;
    if (call_count == 1)
      {
        AccumulatorState active;
        active.n_signals = 1;
        SignalRecord record;
        record.payload.assign (32, 0x2A);
        active.signals.push_back (std::move (record));
        processor_context.accumulator_states["test"] = std::move (active);
        ProcessorContext::WMSlot slot;
        slot.memory_id = 17;
        slot.signal_records
            = processor_context.accumulator_states.at ("test").signals;
        processor_context.wm_slots.push_back (std::move (slot));
        accumulator_payload = processor_context.accumulator_states.at ("test")
                                  .signals.front ().payload.data ();
        working_memory_payload = processor_context.wm_slots.front ()
                                     .signal_records.front ().payload.data ();
        return;
      }
    *restored = processor_context.accumulator_states.at ("test")
                        .signals.front ().payload.data ()
                    == accumulator_payload
                && processor_context.wm_slots.front ()
                           .signal_records.front ().payload.data ()
                       == working_memory_payload;
  }

  bool *restored = nullptr;
  mutable int call_count = 0;
  mutable const unsigned char *accumulator_payload = nullptr;
  mutable const unsigned char *working_memory_payload = nullptr;
};

struct ExerciseWorkingMemoryEraseOwnershipOp : IOperation
{
  explicit ExerciseWorkingMemoryEraseOwnershipOp (bool *restored)
      : restored (restored)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &processor_context = ctx.GetProcessorContext ();
    ++call_count;
    if (call_count == 1)
      {
        ProcessorContext::WMSlot small;
        small.memory_id = 1;
        SignalRecord small_record;
        small_record.payload.assign (8, 0x11);
        small.signal_records.push_back (std::move (small_record));
        ProcessorContext::WMSlot large;
        large.memory_id = 2;
        for (int index = 0; index < 4096; ++index)
          {
            SignalRecord record;
            record.payload.assign (8, 0x22);
            large.signal_records.push_back (std::move (record));
          }
        processor_context.wm_slots.push_back (std::move (small));
        processor_context.wm_slots.push_back (std::move (large));
        small_payload = processor_context.wm_slots[0]
                            .signal_records.front ().payload.data ();
        large_payload = processor_context.wm_slots[1]
                            .signal_records.front ().payload.data ();
        return;
      }
    if (call_count == 2)
      {
        operations::signal_record_rollback_internal::
            PreserveWorkingMemorySlotBeforeErase (processor_context, 0);
        processor_context.wm_slots.erase (processor_context.wm_slots.begin ());
        throw std::runtime_error ("force working-memory erase rollback");
      }
    *restored = processor_context.wm_slots.size () == 2
                && processor_context.wm_slots[0]
                           .signal_records.front ().payload.data ()
                       == small_payload
                && processor_context.wm_slots[1]
                           .signal_records.front ().payload.data ()
                       == large_payload;
  }

  bool *restored = nullptr;
  mutable int call_count = 0;
  mutable const unsigned char *small_payload = nullptr;
  mutable const unsigned char *large_payload = nullptr;
};

struct ExerciseWorkingMemoryAppendOwnershipOp : IOperation
{
  explicit ExerciseWorkingMemoryAppendOwnershipOp (bool *restored)
      : restored (restored)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &processor_context = ctx.GetProcessorContext ();
    ++call_count;
    if (call_count == 1)
      {
        ProcessorContext::WMSlot slot;
        slot.memory_id = 81;
        for (int index = 0; index < 4096; ++index)
          {
            SignalRecord record;
            record.payload.assign (8, 0x31);
            slot.signal_records.push_back (std::move (record));
          }
        processor_context.wm_slots.push_back (std::move (slot));
        first_payload = processor_context.wm_slots.front ()
                            .signal_records.front ().payload.data ();
        return;
      }
    if (call_count == 2)
      {
        SignalRecord appended;
        appended.payload.assign (8, 0x42);
        processor_context.wm_slots.front ().signal_records.push_back (
            std::move (appended));
        throw std::runtime_error ("force working-memory append rollback");
      }
    *restored = processor_context.wm_slots.front ().signal_records.size ()
                    == 4096
                && processor_context.wm_slots.front ()
                           .signal_records.front ().payload.data ()
                       == first_payload;
  }

  bool *restored = nullptr;
  mutable int call_count = 0;
  mutable const unsigned char *first_payload = nullptr;
};

struct ExerciseActiveToFullBackupUpgradeOp : IOperation
{
  explicit ExerciseActiveToFullBackupUpgradeOp (bool *restored)
      : restored (restored)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &processor_context = ctx.GetProcessorContext ();
    ++call_count;
    if (call_count == 1)
      {
        AccumulatorState active;
        active.n_signals = 1;
        SignalRecord active_record;
        active_record.payload.assign (8, 0x19);
        active.signals.push_back (std::move (active_record));
        processor_context.accumulator_states["test"] = std::move (active);
        AccumulatorState foreign;
        foreign.n_signals = 7;
        processor_context.accumulator_states["foreign/source"]
            = std::move (foreign);
        return;
      }
    if (call_count == 2)
      {
        operations::signal_record_rollback_internal::EnsureBackedUp (
            processor_context);
        processor_context.accumulator_states.at ("test")
            .signals.front ().payload.assign (8, 0xFF);
        operations::signal_record_rollback_internal::EnsureAllBackedUp (
            processor_context);
        processor_context.accumulator_states.erase ("foreign/source");
        throw std::runtime_error ("force full-backup upgrade rollback");
      }
    *restored = processor_context.accumulator_states.at ("test")
                        .signals.front ().payload
                    == std::vector<unsigned char> (8, 0x19)
                && processor_context.accumulator_states
                           .at ("foreign/source").n_signals
                       == 7;
  }

  bool *restored = nullptr;
  mutable int call_count = 0;
};

struct ExerciseNestedDifferentSourceRollbackOp : IOperation
{
  explicit ExerciseNestedDifferentSourceRollbackOp (bool *restored)
      : restored (restored)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &processor_context = ctx.GetProcessorContext ();
    const auto &signal = ctx.GetSignal ();
    if (signal.source_id == "inner/source")
      {
        AccumulatorState inner;
        inner.n_signals = 1;
        SignalRecord record;
        record.payload.assign (8, 0x77);
        inner.signals.push_back (std::move (record));
        processor_context.accumulator_states[signal.source_id]
            = std::move (inner);
        return;
      }

    ++outer_call_count;
    if (outer_call_count == 1)
      {
        AccumulatorState outer;
        outer.n_signals = 1;
        SignalRecord record;
        record.payload.assign (8, 0x55);
        outer.signals.push_back (std::move (record));
        processor_context.accumulator_states[signal.source_id]
            = std::move (outer);
        return;
      }
    if (outer_call_count == 2)
      {
        Signal nested;
        nested.embedding = Eigen::VectorXf::Zero (256);
        nested.source_id = "inner/source";
        nested.timestamp = signal.timestamp + 1;
        processor->Process (nested);
        throw std::runtime_error ("force outer rollback after nested process");
      }
    *restored = !processor_context.accumulator_states.contains ("inner/source")
                && processor_context.accumulator_states.at (signal.source_id)
                           .signals.front ().payload
                       == std::vector<unsigned char> (8, 0x55);
  }

  SignalProcessor *processor = nullptr;
  bool *restored = nullptr;
  mutable int outer_call_count = 0;
};

struct ExerciseNestedShadowTransactionOp : IOperation
{
  explicit ExerciseNestedShadowTransactionOp (bool throw_after_nested)
      : throw_after_nested (throw_after_nested)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction &tx) const override
  {
    const auto &signal = ctx.GetSignal ();
    tx.Execute ("INSERT INTO nested_shadow_markers(source_id) VALUES(?)",
                { signal.source_id });
    if (signal.source_id == "inner-shadow")
      return;
    Signal nested = MakeBoundedShadowSignal (
        static_cast<std::uint64_t> (signal.timestamp + 1), "inner-shadow",
        "audio");
    processor->Process (nested);
    if (throw_after_nested)
      throw std::runtime_error ("force root rollback after nested commit");
  }

  SignalProcessor *processor = nullptr;
  bool throw_after_nested = false;
};

struct ExerciseThroughputRollbackOp : IOperation
{
  explicit ExerciseThroughputRollbackOp (bool *restored_in)
      : restored (restored_in)
  {
  }

  bool *restored = nullptr;
  mutable int call_count = 0;

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    namespace throughput
        = operations::consolidation_throughput_state_internal;
    auto &pctx = ctx.GetProcessorContext ();
    ++call_count;
    if (call_count == 1)
      {
        throughput::Reset (pctx, { 2.0, 10.0, true });
        return;
      }
    if (call_count == 2)
      {
        throughput::Reset (pctx, { 8.0, 9.0, true });
        throw std::runtime_error ("force throughput rollback");
      }
    const auto state = throughput::Find (pctx);
    if (restored)
      {
        *restored = state.floor == 2.0 && state.peak == 10.0
                    && state.initialized;
      }
  }
};

struct ExerciseThroughputEphemeralOp : IOperation
{
  explicit ExerciseThroughputEphemeralOp (bool *restored_in)
      : restored (restored_in)
  {
  }

  bool *restored = nullptr;
  mutable int call_count = 0;

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    namespace throughput
        = operations::consolidation_throughput_state_internal;
    auto &pctx = ctx.GetProcessorContext ();
    ++call_count;
    if (call_count == 1)
      {
        throughput::Reset (pctx, { 3.0, 12.0, true });
        return;
      }
    if (call_count == 2)
      {
        throughput::Reset (pctx, { 9.0, 9.5, true });
        return;
      }
    const auto state = throughput::Find (pctx);
    if (restored)
      {
        *restored = state.floor == 3.0 && state.peak == 12.0
                    && state.initialized;
      }
  }
};

struct SetOrCaptureThroughputOp : IOperation
{
  std::optional<operations::consolidation_throughput_state_internal::State>
      state_to_set;
  operations::consolidation_throughput_state_internal::State *captured
      = nullptr;
  ProcessorContext **context_address = nullptr;

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    namespace throughput
        = operations::consolidation_throughput_state_internal;
    auto &pctx = ctx.GetProcessorContext ();
    if (context_address)
      {
        *context_address = &pctx;
      }
    if (state_to_set)
      {
        throughput::Reset (pctx, *state_to_set);
      }
    if (captured)
      {
        *captured = throughput::Find (pctx);
      }
  }
};

TEST_CASE ("SignalProcessor processes and flushes to SQLite", "[processor]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  // Create table upfront
  store->Execute (
      "CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT);");

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  auto pipeline
      = std::make_unique<DynamicOperationSet> (std::make_unique<InsertOp> ());
  SignalProcessor proc (cfg, store, std::move (pipeline));

  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  s.timestamp = 0;
  s.source_id = "test";

  proc.Process (s);
  proc.Flush ();

  auto rows = store->Execute ("SELECT COUNT(*) AS c FROM t;");
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 1LL);
}

TEST_CASE ("Bounded activation shadow is default-off and publishes Natural and "
           "Durable at the shared post-commit edge",
           "[processor][bounded_activation_shadow][lifecycle]")
{
  namespace shadow
      = operations::bounded_activation_shadow_internal;
  const auto baseline_registry = shadow::RegistrySizeForTest ();
  {
    cortext::testing::ScopedEnvVar disabled (
        "CORTEXT_BOUNDED_ACTIVATION_SHADOW");
    auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    auto root = std::make_unique<DynamicOperationSet> (
        std::make_unique<BoundedShadowNoOp> ());
    SignalProcessor processor (cfg, store, std::move (root));
    processor.Process (MakeBoundedShadowSignal (0));
    REQUIRE (shadow::RegistrySizeForTest () == baseline_registry);
  }

  {
    cortext::testing::ScopedEnvVar enabled (
        "CORTEXT_BOUNDED_ACTIVATION_SHADOW", "1");
    auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    auto root = std::make_unique<DynamicOperationSet> (
        std::make_unique<BoundedShadowNoOp> ());
    SignalProcessor processor (cfg, store, std::move (root));

    auto natural = MakeBoundedShadowSignal (0, "source-a", "text");
    natural.retention = Retention::Natural;
    const auto natural_output = processor.Process (natural);
    REQUIRE (std::none_of (
        natural_output.operation_ms.begin (), natural_output.operation_ms.end (),
        [] (const auto &entry) {
          return entry.first.find ("bounded_activation_shadow")
                 != std::string::npos;
        }));
    auto snapshots = shadow::SnapshotsForTest ();
    REQUIRE (snapshots.size () == baseline_registry + 1);
    const auto first = snapshots.back ();
    REQUIRE (first.available);
    REQUIRE_FALSE (first.disabled);
    REQUIRE (first.generation == 1);
    REQUIRE (first.event_index == 1);

    auto durable = MakeBoundedShadowSignal (1, "source-b", "audio");
    durable.retention = Retention::Durable;
    cortext::testing::SetSQLiteCheckpointFailureOnceForTest ();
    processor.Process (durable);
    snapshots = shadow::SnapshotsForTest ();
    const auto second = snapshots.back ();
    REQUIRE (second.generation == 2);
    REQUIRE (second.event_index == 2);
    REQUIRE_FALSE (second.disabled);
  }
  REQUIRE (shadow::RegistrySizeForTest () == baseline_registry);
}

TEST_CASE ("Bounded activation shadow journals nested events until the root "
           "canonical commit",
           "[processor][bounded_activation_shadow][lifecycle][nested]")
{
  namespace shadow
      = operations::bounded_activation_shadow_internal;
  cortext::testing::ScopedEnvVar enabled (
      "CORTEXT_BOUNDED_ACTIVATION_SHADOW", "1");
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  store->Execute (
      "CREATE TABLE nested_shadow_markers("
      "id INTEGER PRIMARY KEY, source_id TEXT NOT NULL)");
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto operation
      = std::make_unique<ExerciseNestedShadowTransactionOp> (false);
  auto *operation_ptr = operation.get ();
  auto root = std::make_unique<DynamicOperationSet> (std::move (operation));
  SignalProcessor processor (cfg, store, std::move (root));
  operation_ptr->processor = &processor;

  processor.Process (MakeBoundedShadowSignal (0, "outer-shadow", "text"));
  const auto rows = store->Execute (
      "SELECT COUNT(*) AS n FROM nested_shadow_markers");
  REQUIRE (std::any_cast<long long> (rows.front ().at ("n")) == 2);
  const auto snapshots = shadow::SnapshotsForTest ();
  REQUIRE (snapshots.size () == 1);
  REQUIRE_FALSE (snapshots.front ().disabled);
  REQUIRE (snapshots.front ().generation == 2);
  REQUIRE (snapshots.front ().event_index == 2);
}

TEST_CASE ("Bounded activation shadow discards nested events when the root "
           "canonical transaction rolls back",
           "[processor][bounded_activation_shadow][lifecycle][nested]")
{
  namespace shadow
      = operations::bounded_activation_shadow_internal;
  cortext::testing::ScopedEnvVar enabled (
      "CORTEXT_BOUNDED_ACTIVATION_SHADOW", "1");
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  store->Execute (
      "CREATE TABLE nested_shadow_markers("
      "id INTEGER PRIMARY KEY, source_id TEXT NOT NULL)");
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto operation
      = std::make_unique<ExerciseNestedShadowTransactionOp> (true);
  auto *operation_ptr = operation.get ();
  auto root = std::make_unique<DynamicOperationSet> (std::move (operation));
  SignalProcessor processor (cfg, store, std::move (root));
  operation_ptr->processor = &processor;

  REQUIRE_THROWS (processor.Process (
      MakeBoundedShadowSignal (0, "outer-shadow", "image")));
  const auto rows = store->Execute (
      "SELECT COUNT(*) AS n FROM nested_shadow_markers");
  REQUIRE (std::any_cast<long long> (rows.front ().at ("n")) == 0);
  const auto snapshots = shadow::SnapshotsForTest ();
  REQUIRE (snapshots.size () == 1);
  REQUIRE_FALSE (snapshots.front ().disabled);
  REQUIRE (snapshots.front ().generation == 0);
  REQUIRE (snapshots.front ().event_index == 0);
}

TEST_CASE ("Bounded activation shadow discards a root journal when snapshot "
           "setup throws before execution",
           "[processor][bounded_activation_shadow][lifecycle][failure]")
{
  namespace shadow
      = operations::bounded_activation_shadow_internal;
  cortext::testing::ScopedEnvVar enabled (
      "CORTEXT_BOUNDED_ACTIVATION_SHADOW", "1");
  const auto baseline_registry = shadow::RegistrySizeForTest ();
  {
    auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    bool restored = false;
    auto pipeline = std::make_unique<DynamicOperationSet> (
        std::make_unique<ExerciseSnapshotSetupExceptionSafetyOp> (&restored));
    SignalProcessor processor (
        cfg, store,
        operations::signal_record_rollback_internal::MarkJournalAware (
            std::move (pipeline)));

    processor.Process (MakeBoundedShadowSignal (0, "test"));
    cortext::testing::SetSignalRecordSnapshotSetupThrowStageForTest (1);
    REQUIRE_THROWS (processor.Process (MakeBoundedShadowSignal (1, "test")));
    auto snapshots = shadow::SnapshotsForTest ();
    REQUIRE (snapshots.size () == baseline_registry + 1);
    REQUIRE_FALSE (snapshots.back ().disabled);
    REQUIRE (snapshots.back ().generation == 1);
    REQUIRE (snapshots.back ().event_index == 1);

    processor.Process (MakeBoundedShadowSignal (2, "test"));
    snapshots = shadow::SnapshotsForTest ();
    REQUIRE_FALSE (snapshots.back ().disabled);
    REQUIRE (snapshots.back ().generation == 2);
    REQUIRE (snapshots.back ().event_index == 2);
    REQUIRE (restored);
  }
  REQUIRE (shadow::RegistrySizeForTest () == baseline_registry);
}

TEST_CASE ("Bounded activation shadow discards rollback and fails closed after "
           "a post-commit publication fault",
           "[processor][bounded_activation_shadow][rollback][failure]")
{
  namespace shadow
      = operations::bounded_activation_shadow_internal;
  cortext::testing::ScopedEnvVar enabled (
      "CORTEXT_BOUNDED_ACTIVATION_SHADOW", "1");
  const auto baseline_registry = shadow::RegistrySizeForTest ();
  {
    auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    auto root = std::make_unique<DynamicOperationSet> (
        std::make_unique<BoundedShadowThrowOnceOp> ());
    SignalProcessor processor (cfg, store, std::move (root));
    REQUIRE_THROWS (processor.Process (MakeBoundedShadowSignal (0)));
    auto snapshots = shadow::SnapshotsForTest ();
    REQUIRE (snapshots.size () == baseline_registry + 1);
    REQUIRE (snapshots.back ().generation == 0);
    REQUIRE (snapshots.back ().event_index == 0);
    processor.Process (MakeBoundedShadowSignal (1));
    snapshots = shadow::SnapshotsForTest ();
    REQUIRE (snapshots.back ().generation == 1);
    REQUIRE (snapshots.back ().event_index == 1);
  }
  REQUIRE (shadow::RegistrySizeForTest () == baseline_registry);

  {
    auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
    store->Execute (
        "CREATE TABLE shadow_commit_probe(id INTEGER PRIMARY KEY, v INTEGER)");
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    struct InsertProbe final : IOperation
    {
      void
      Execute (OperationContext &, Transaction &tx) const override
      {
        tx.Execute ("INSERT INTO shadow_commit_probe(v) VALUES(1)");
      }
    };
    auto root = std::make_unique<DynamicOperationSet> (
        std::make_unique<InsertProbe> ());
    SignalProcessor processor (cfg, store, std::move (root));
    shadow::SetFailureStageForTest (2);
    REQUIRE_NOTHROW (processor.Process (MakeBoundedShadowSignal (0)));
    const auto rows
        = store->Execute ("SELECT COUNT(*) AS n FROM shadow_commit_probe");
    REQUIRE (std::any_cast<long long> (rows[0].at ("n")) == 1);
    const auto snapshots = shadow::SnapshotsForTest ();
    REQUIRE (snapshots.size () == baseline_registry + 1);
    REQUIRE (snapshots.back ().disabled);
    REQUIRE (snapshots.back ().failure == shadow::Failure::PublishInjected);
    REQUIRE (snapshots.back ().generation == 0);
  }
  REQUIRE (shadow::RegistrySizeForTest () == baseline_registry);
}

TEST_CASE ("Bounded activation shadow isolates preparation cancellation and "
           "persistent commit failures",
           "[processor][bounded_activation_shadow][lifecycle][failure]")
{
  namespace shadow
      = operations::bounded_activation_shadow_internal;
  cortext::testing::ScopedEnvVar enabled (
      "CORTEXT_BOUNDED_ACTIVATION_SHADOW", "1");
  const auto baseline_registry = shadow::RegistrySizeForTest ();

  {
    auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
    store->Execute (
        "CREATE TABLE shadow_prepare_probe(id INTEGER PRIMARY KEY, v INTEGER)");
    struct InsertPrepareProbe final : IOperation
    {
      void
      Execute (OperationContext &, Transaction &tx) const override
      {
        tx.Execute ("INSERT INTO shadow_prepare_probe(v) VALUES(1)");
      }
    };
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    auto root = std::make_unique<DynamicOperationSet> (
        std::make_unique<InsertPrepareProbe> ());
    SignalProcessor processor (cfg, store, std::move (root));
    shadow::SetFailureStageForTest (1);
    REQUIRE_NOTHROW (processor.Process (MakeBoundedShadowSignal (0)));
    const auto rows
        = store->Execute ("SELECT COUNT(*) AS n FROM shadow_prepare_probe");
    REQUIRE (std::any_cast<long long> (rows[0].at ("n")) == 1);
    const auto snapshots = shadow::SnapshotsForTest ();
    REQUIRE (snapshots.size () == baseline_registry + 1);
    REQUIRE (snapshots.back ().disabled);
    REQUIRE (snapshots.back ().failure == shadow::Failure::PrepareInjected);
    REQUIRE (snapshots.back ().generation == 0);
  }
  REQUIRE (shadow::RegistrySizeForTest () == baseline_registry);

  {
    auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    auto root = std::make_unique<DynamicOperationSet> (
        std::make_unique<BoundedShadowCancelOp> ());
    SignalProcessor processor (cfg, store, std::move (root));
    REQUIRE_THROWS_AS (processor.Process (MakeBoundedShadowSignal (0)),
                       internal::CancellationError);
    const auto snapshots = shadow::SnapshotsForTest ();
    REQUIRE (snapshots.size () == baseline_registry + 1);
    REQUIRE_FALSE (snapshots.back ().disabled);
    REQUIRE (snapshots.back ().generation == 0);
    REQUIRE (snapshots.back ().event_index == 0);
  }
  REQUIRE (shadow::RegistrySizeForTest () == baseline_registry);

  {
    auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
    store->Execute ("PRAGMA foreign_keys = ON");
    store->Execute ("CREATE TABLE deferred_parent(id INTEGER PRIMARY KEY)");
    store->Execute (
        "CREATE TABLE deferred_child("
        "parent_id INTEGER REFERENCES deferred_parent(id) "
        "DEFERRABLE INITIALLY DEFERRED)");
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    auto root = std::make_unique<DynamicOperationSet> (
        std::make_unique<InsertDeferredViolationOp> ());
    SignalProcessor processor (cfg, store, std::move (root));
    REQUIRE_THROWS (processor.Process (MakeBoundedShadowSignal (0)));
    const auto rows = store->Execute (
        "SELECT COUNT(*) AS n FROM deferred_child");
    REQUIRE (std::any_cast<long long> (rows[0].at ("n")) == 0);
    const auto snapshots = shadow::SnapshotsForTest ();
    REQUIRE (snapshots.size () == baseline_registry + 1);
    REQUIRE_FALSE (snapshots.back ().disabled);
    REQUIRE (snapshots.back ().generation == 0);
    REQUIRE (snapshots.back ().event_index == 0);
  }
  REQUIRE (shadow::RegistrySizeForTest () == baseline_registry);
}

TEST_CASE ("Bounded activation shadow rejects duplicate gap regression and "
           "digest mismatch publication",
           "[processor][bounded_activation_shadow][generation]")
{
  namespace shadow
      = operations::bounded_activation_shadow_internal;
  const auto make_state = [] {
    auto state = std::make_shared<shadow::State> (
        shadow::DeriveParameters (0.0, 0.0, 0.0));
    state->available = true;
    return state;
  };
  const auto signal = MakeBoundedShadowSignal (0);

  auto duplicate_state = make_state ();
  auto first = shadow::Prepare (duplicate_state, signal);
  REQUIRE (shadow::PublishAfterPersistentCommit (first).published);
  auto duplicate = shadow::Prepare (duplicate_state, signal);
  duplicate.generation = 1;
  const auto duplicate_result
      = shadow::PublishAfterPersistentCommit (duplicate);
  REQUIRE (duplicate_result.disabled);
  REQUIRE (duplicate_result.failure == shadow::Failure::DuplicateGeneration);

  auto gap_state = make_state ();
  auto gap = shadow::Prepare (gap_state, signal);
  gap.generation = 2;
  const auto gap_result = shadow::PublishAfterPersistentCommit (gap);
  REQUIRE (gap_result.disabled);
  REQUIRE (gap_result.failure == shadow::Failure::GenerationGap);

  auto regression_state = make_state ();
  regression_state->last_published_generation = 2;
  auto regression = shadow::Prepare (regression_state, signal);
  regression.generation = 1;
  const auto regression_result
      = shadow::PublishAfterPersistentCommit (regression);
  REQUIRE (regression_result.disabled);
  REQUIRE (regression_result.failure == shadow::Failure::GenerationRegression);

  auto digest_state = make_state ();
  auto digest = shadow::Prepare (digest_state, signal);
  digest.digest ^= 1;
  const auto digest_result = shadow::PublishAfterPersistentCommit (digest);
  REQUIRE (digest_result.disabled);
  REQUIRE (digest_result.failure == shadow::Failure::DigestMismatch);
}

TEST_CASE ("Bounded activation shadow mechanics and digest ignore source and "
           "modality labels",
           "[processor][bounded_activation_shadow][modality][source]")
{
  namespace shadow
      = operations::bounded_activation_shadow_internal;
  auto first = std::make_shared<shadow::State> (
      shadow::DeriveParameters (0.0, 0.0, 0.0));
  auto second = std::make_shared<shadow::State> (
      shadow::DeriveParameters (0.0, 0.0, 0.0));
  first->available = true;
  second->available = true;
  const std::array<std::string, 4> sources {
    "source-a", "source-b", "source-c", "source-d"
  };
  const std::array<std::string, 3> modalities { "text", "audio", "image" };
  for (std::uint64_t event = 0; event < 192; ++event)
    {
      auto control = MakeBoundedShadowSignal (event, "one-source", "text");
      auto mixed = MakeBoundedShadowSignal (
          event, sources[event % sources.size ()],
          modalities[event % modalities.size ()]);
      auto control_prepared = shadow::Prepare (first, control);
      auto mixed_prepared = shadow::Prepare (second, mixed);
      REQUIRE (control_prepared.recall_comparisons
               == mixed_prepared.recall_comparisons);
      REQUIRE (control_prepared.candidate_count
               == mixed_prepared.candidate_count);
      for (std::size_t candidate = 0;
           candidate < control_prepared.candidate_count; ++candidate)
        REQUIRE (control_prepared.candidate_identities[candidate]
                 == mixed_prepared.candidate_identities[candidate]);
      const auto control_result
          = shadow::PublishAfterPersistentCommit (control_prepared);
      const auto mixed_result
          = shadow::PublishAfterPersistentCommit (mixed_prepared);
      REQUIRE (control_result.normal_comparisons
               == mixed_result.normal_comparisons);
      REQUIRE (control_result.consolidation_comparisons
               == mixed_result.consolidation_comparisons);
      REQUIRE (shadow::StateDigest (*first) == shadow::StateDigest (*second));
    }
}

TEST_CASE ("Bounded activation shadow respects every FST-derived storage and "
           "work envelope after capacity initialization",
           "[processor][bounded_activation_shadow][bounds]")
{
  namespace shadow
      = operations::bounded_activation_shadow_internal;
  auto state = std::make_shared<shadow::State> (
      shadow::DeriveParameters (0.5, 0.5, 0.5));
  state->available = true;
  const auto initial_allocated_bytes = state->allocated_bytes;
  for (std::uint64_t event = 0; event < 2048; ++event)
    {
      auto prepared
          = shadow::Prepare (state, MakeBoundedShadowSignal (event));
      REQUIRE (prepared.valid);
      REQUIRE (prepared.recall_comparisons
               <= state->parameters.recall_comparison_bound);
      const auto result = shadow::PublishAfterPersistentCommit (prepared);
      REQUIRE (result.published);
      REQUIRE (result.normal_comparisons
               <= state->parameters.normal_comparison_bound);
      REQUIRE (result.consolidation_comparisons
               <= state->parameters.consolidation_comparison_bound);
      REQUIRE_FALSE (state->disabled);
      REQUIRE (state->allocated_bytes == initial_allocated_bytes);
      REQUIRE (state->root_count <= state->parameters.roots);
      REQUIRE (state->leaf_count <= state->parameters.leaf_capacity);
    }
  const auto snapshot = shadow::ReadSnapshot (*state);
  REQUIRE (snapshot.generation == 2048);
  REQUIRE (snapshot.metrics.normal_comparisons_max
           <= snapshot.parameters.normal_comparison_bound);
  REQUIRE (snapshot.metrics.recall_comparisons_max
           <= snapshot.parameters.recall_comparison_bound);
  REQUIRE (snapshot.metrics.consolidation_comparisons_max
           <= snapshot.parameters.consolidation_comparison_bound);
  REQUIRE (snapshot.metrics.allocation_after_initialization_count == 0);
  REQUIRE (snapshot.parameters.fixed_embedding_slots
           == snapshot.parameters.roots + snapshot.parameters.leaf_capacity
                  + snapshot.parameters.leaf_capacity
                        * snapshot.parameters.representatives_per_leaf);
  REQUIRE (snapshot.parameters.fixed_neighbor_slots
           == snapshot.parameters.leaf_capacity
                  * snapshot.parameters.neighbor_degree);
}

TEST_CASE ("Bounded activation shadow FST formulas are fixed across all corner "
           "and midpoint combinations",
           "[processor][bounded_activation_shadow][bounds][fst]")
{
  namespace shadow
      = operations::bounded_activation_shadow_internal;
  for (const double focus : { 0.0, 0.5, 1.0 })
    for (const double sensitivity : { 0.0, 0.5, 1.0 })
      for (const double stability : { 0.0, 0.5, 1.0 })
        {
          const auto parameters
              = shadow::DeriveParameters (focus, sensitivity, stability);
          REQUIRE (parameters.roots
                   == 8 + static_cast<std::size_t> (
                              std::lround (24 * sensitivity)));
          REQUIRE (parameters.leaf_capacity
                   == 128 + static_cast<std::size_t> (
                                std::lround (512 * focus + 512 * stability)));
          REQUIRE (parameters.children_per_root
                   == (parameters.leaf_capacity + parameters.roots - 1)
                          / parameters.roots);
          REQUIRE (parameters.root_beam
                   == 1 + static_cast<std::size_t> (
                              std::lround (3 * sensitivity)));
          REQUIRE (parameters.leaf_beam
                   == 2 + static_cast<std::size_t> (
                              std::lround (6 * focus)));
          REQUIRE (parameters.representatives_per_leaf
                   == 8 + static_cast<std::size_t> (
                              std::lround (32 * focus)));
          REQUIRE (parameters.neighbor_degree
                   == 4 + static_cast<std::size_t> (
                              std::lround (8 * sensitivity)));
          REQUIRE (parameters.activated_leaves_per_consolidation
                   == 8 + static_cast<std::size_t> (
                              std::lround (24 * focus)));
          REQUIRE (parameters.consolidation_interval
                   == 256 + static_cast<std::size_t> (
                                std::lround (512 * stability)));
          REQUIRE (parameters.fixed_embedding_slots
                   == parameters.roots + parameters.leaf_capacity
                          + parameters.leaf_capacity
                                * parameters.representatives_per_leaf);
          REQUIRE (parameters.fixed_neighbor_slots
                   == parameters.leaf_capacity
                          * parameters.neighbor_degree);
        }
}

TEST_CASE ("Bounded activation shadow restart is unavailable until explicit "
           "linear-history rebuild and keeps that rebuild as a production "
           "blocker",
           "[processor][bounded_activation_shadow][restart]")
{
  namespace shadow
      = operations::bounded_activation_shadow_internal;
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  for (long long index = 1; index <= 4; ++index)
    {
      std::vector<float> embedding (256, 0.0F);
      embedding[static_cast<std::size_t> (index)] = 1.0F;
      store->Execute (
          "INSERT INTO embeddings(embedding_id, embedding, created_at) "
          "VALUES(?, ?, ?)",
          { index, embedding, index });
      store->Execute (
          "INSERT INTO signals(signal_id, source_id, embedding_id, timestamp, "
          "modality, created_at) VALUES(?, ?, ?, ?, ?, ?)",
          { index, std::string ("source-a"), index, index,
            std::string ("text"), index });
    }

  cortext::testing::ScopedEnvVar enabled (
      "CORTEXT_BOUNDED_ACTIVATION_SHADOW", "1");
  cortext::testing::ScopedEnvVar rebuild_disabled (
      "CORTEXT_BOUNDED_ACTIVATION_SHADOW_REBUILD");
  ProcessorContext first_context;
  auto unavailable = shadow::Initialize (first_context, store.get (),
                                         0.5, 0.5, 0.5);
  REQUIRE (unavailable);
  REQUIRE_FALSE (unavailable->available);
  REQUIRE (unavailable->restart_rebuild_required);
  REQUIRE_FALSE (
      shadow::Prepare (unavailable, MakeBoundedShadowSignal (4)).valid);
  shadow::Erase (first_context);

  {
    cortext::testing::ScopedEnvVar rebuild_enabled (
        "CORTEXT_BOUNDED_ACTIVATION_SHADOW_REBUILD", "1");
    ProcessorContext rebuilt_context;
    auto rebuilt = shadow::Initialize (rebuilt_context, store.get (),
                                      0.5, 0.5, 0.5);
    REQUIRE (rebuilt);
    REQUIRE (rebuilt->available);
    REQUIRE_FALSE (rebuilt->restart_rebuild_required);
    REQUIRE (rebuilt->last_published_generation == 4);
    REQUIRE (rebuilt->metrics.restart_rows_visited == 4);
    REQUIRE (rebuilt->metrics.restart_rebuild_linear_history);
    REQUIRE_FALSE (rebuilt->metrics.restart_production_gate);
    shadow::Erase (rebuilt_context);
  }

  store->Execute (
      "INSERT INTO signals(signal_id, source_id, embedding_id, timestamp, "
      "modality, created_at) VALUES(5, 'source-a', 999, 5, 'text', 5)");
  {
    cortext::testing::ScopedEnvVar rebuild_enabled (
        "CORTEXT_BOUNDED_ACTIVATION_SHADOW_REBUILD", "1");
    ProcessorContext incomplete_context;
    auto incomplete = shadow::Initialize (incomplete_context, store.get (),
                                          0.5, 0.5, 0.5);
    REQUIRE (incomplete);
    REQUIRE_FALSE (incomplete->available);
    REQUIRE (incomplete->disabled);
    REQUIRE (incomplete->failure == shadow::Failure::RebuildFailure);
    shadow::Erase (incomplete_context);
  }
}

TEST_CASE ("SignalProcessor rebuilds the SQLite active epoch after a "
           "post-commit publication failure without replay",
           "[processor][rif][active_epoch][durable][failure]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  store->Execute (
      "INSERT INTO memories(memory_id, source_id, kind, start_ts, n_signals, "
      "modality, strength, suppression, suppression_ts, created_at) "
      "VALUES(1, 'source-a', 'LONG_TERM', 1000, 1, 'text', "
      "0.5, 0.5, 1000, 1000)");
  cortext::operations::rif_state_internal::RebuildFromMaterialized (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto root = std::make_unique<DynamicOperationSet> (
      std::make_unique<AdvanceRifEpochOp> ());
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::
          MarkEngineOwnedJournalAware (std::move (root)));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "source-a";
  signal.retention = Retention::Durable;
  signal.timestamp = 2000;
  cortext::testing::SetRifActiveEpochPublishFailureStageForTest (2);
  const auto first = processor.Process (signal);
  REQUIRE (first.operation_ms.at (
               "SignalProcessor.rif_epoch_publication_recovery_count")
           == 1.0);
  REQUIRE (first.operation_ms.contains (
      "SignalProcessor.sqlite_wal_checkpoint"));

  signal.timestamp = 3000;
  const auto second = processor.Process (signal);
  REQUIRE (second.operation_ms.at (
               "SignalProcessor.rif_epoch_publication_recovery_count")
           == 0.0);
  const auto rows = store->Execute (
      "SELECT strength, suppression, suppression_ts "
      "FROM rif_effective_memories WHERE memory_id = 1");
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<double> (rows[0].at ("strength"))
           == Catch::Approx (0.595));
  REQUIRE (std::any_cast<double> (rows[0].at ("suppression"))
           == Catch::Approx (0.405));
  REQUIRE (std::any_cast<long long> (rows[0].at ("suppression_ts"))
           == 3000);
}

TEST_CASE ("SignalProcessor discards an unpublished RIF epoch on persistent "
           "commit failure",
           "[processor][rif][active_epoch][rollback][commit]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  store->Execute (
      "INSERT INTO memories(memory_id, source_id, kind, start_ts, n_signals, "
      "modality, strength, suppression, suppression_ts, created_at) "
      "VALUES(1, 'source-a', 'LONG_TERM', 1000, 1, 'text', "
      "0.5, 0.5, 1000, 1000)");
  cortext::operations::rif_state_internal::RebuildFromMaterialized (*store);
  store->Execute ("CREATE TABLE deferred_parent(id INTEGER PRIMARY KEY)");
  store->Execute (
      "CREATE TABLE deferred_child("
      "parent_id INTEGER REFERENCES deferred_parent(id) "
      "DEFERRABLE INITIALLY DEFERRED)");

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto root = std::make_unique<DynamicOperationSet> (
      std::make_unique<AdvanceRifAndFailCommitOnceOp> ());
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::
          MarkEngineOwnedJournalAware (std::move (root)));
  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "source-a";
  signal.retention = Retention::Natural;
  signal.timestamp = 2000;
  REQUIRE_THROWS (processor.Process (signal));

  auto rows = store->Execute (
      "SELECT strength, suppression, suppression_ts "
      "FROM rif_effective_memories WHERE memory_id = 1");
  REQUIRE (std::any_cast<double> (rows[0].at ("strength"))
           == Catch::Approx (0.5));
  REQUIRE (std::any_cast<double> (rows[0].at ("suppression"))
           == Catch::Approx (0.5));
  REQUIRE (std::any_cast<long long> (rows[0].at ("suppression_ts"))
           == 1000);

  const auto success = processor.Process (signal);
  REQUIRE (success.operation_ms.at (
               "SignalProcessor.rif_epoch_publication_recovery_count")
           == 0.0);
  rows = store->Execute (
      "SELECT strength, suppression, suppression_ts "
      "FROM rif_effective_memories WHERE memory_id = 1");
  REQUIRE (std::any_cast<double> (rows[0].at ("strength"))
           == Catch::Approx (0.55));
  REQUIRE (std::any_cast<double> (rows[0].at ("suppression"))
           == Catch::Approx (0.45));
  REQUIRE (std::any_cast<long long> (rows[0].at ("suppression_ts"))
           == 2000);
}

TEST_CASE ("RIF epoch retry preserves an ordinary committed event after "
           "publication and recovery both fail",
           "[processor][rif][active_epoch][durable][failure][retry]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  store->Execute (
      "INSERT INTO memories(memory_id, source_id, kind, start_ts, n_signals, "
      "modality, strength, suppression, suppression_ts, created_at) "
      "VALUES(1, 'source-a', 'LONG_TERM', 1000, 1, 'text', "
      "0.5, 0.5, 1000, 1000)");
  cortext::operations::rif_state_internal::RebuildFromMaterialized (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto root = std::make_unique<DynamicOperationSet> (
      std::make_unique<AdvanceRifEpochOp> ());
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::
          MarkEngineOwnedJournalAware (std::move (root)));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "source-a";
  signal.retention = Retention::Durable;
  signal.timestamp = 2000;
  cortext::testing::SetRifActiveEpochPublishFailureMaskForTest (
      (1U << 2U) | (1U << 4U));
  const auto failed_publication = processor.Process (signal);
  REQUIRE (failed_publication.operation_ms.at (
               "SignalProcessor.rif_epoch_publication_recovery_count")
           == 1.0);
  REQUIRE (failed_publication.operation_ms.at (
               "SignalProcessor.rif_active_epoch_event_count")
           == 1.0);

  signal.timestamp = 3000;
  const auto retried = processor.Process (signal);
  REQUIRE (retried.operation_ms.at (
               "SignalProcessor.rif_epoch_publication_recovery_count")
           == 0.0);
  REQUIRE (retried.operation_ms.at (
               "SignalProcessor.rif_active_epoch_event_count")
           == 2.0);
}

TEST_CASE ("Custom operation SQL observes and preserves effective lazy RIF "
           "values",
           "[processor][rif][active_epoch][custom_operation]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  store->Execute (
      "INSERT INTO memories(memory_id, source_id, kind, start_ts, n_signals, "
      "modality, strength, suppression, suppression_ts, created_at) "
      "VALUES(1, 'source-a', 'LONG_TERM', 1000, 1, 'text', "
      "0.5, 0.5, 1000, 1000)");
  cortext::operations::rif_state_internal::RebuildFromMaterialized (*store);
  auto recovery_tx = store->Begin ();
  cortext::operations::rif_state_internal::AdvanceRecovery (
      *recovery_tx, 2000, 10000.0);
  recovery_tx->Commit ();

  double observed_strength = 0.0;
  double observed_suppression = 0.0;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto root = std::make_unique<DynamicOperationSet> (
      std::make_unique<CustomRifReadWriteOp> (
          &observed_strength, &observed_suppression));
  SignalProcessor processor (cfg, store, std::move (root));
  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "source-a";
  signal.retention = Retention::Natural;
  signal.timestamp = 2000;
  const auto first = processor.Process (signal);
  REQUIRE (first.operation_ms.at (
               "SignalProcessor.rif_active_epoch_event_count")
           == 1.0);

  REQUIRE (observed_strength == Catch::Approx (0.55));
  REQUIRE (observed_suppression == Catch::Approx (0.45));
  const auto rows = store->Execute (
      "SELECT strength, suppression FROM rif_effective_memories "
      "WHERE memory_id = 1");
  REQUIRE (std::any_cast<double> (rows[0].at ("strength"))
           == Catch::Approx (0.65));
  REQUIRE (std::any_cast<double> (rows[0].at ("suppression"))
           == Catch::Approx (0.45));

  signal.timestamp = 3000;
  const auto second = processor.Process (signal);
  REQUIRE (second.operation_ms.at (
               "SignalProcessor.rif_active_epoch_event_count")
           == 2.0);
}

TEST_CASE ("Custom operation indirect views observe effective lazy RIF values",
           "[processor][rif][custom_operation][indirect]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  store->Execute (
      "INSERT INTO memories(memory_id, source_id, kind, start_ts, n_signals, "
      "modality, strength, suppression, suppression_ts, created_at) "
      "VALUES(1, 'source-a', 'LONG_TERM', 1000, 1, 'text', "
      "0.5, 0.5, 1000, 1000)");
  cortext::operations::rif_state_internal::RebuildFromMaterialized (*store);
  auto recovery_tx = store->Begin ();
  cortext::operations::rif_state_internal::AdvanceRecovery (
      *recovery_tx, 2000, 10000.0);
  recovery_tx->Commit ();

  double observed_strength = 0.0;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto root = std::make_unique<DynamicOperationSet> (
      std::make_unique<CustomIndirectRifReadOp> (&observed_strength));
  SignalProcessor processor (cfg, store, std::move (root));
  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "source-a";
  signal.retention = Retention::Natural;
  signal.timestamp = 2000;
  processor.Process (signal);
  REQUIRE (observed_strength == Catch::Approx (0.55));
}

TEST_CASE ("Custom mutations invalidate and rebuild every database cache",
           "[processor][cache][custom_operation][mutation]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  cortext::testing::SeedEmbeddingV2 (
      *store, 11, std::vector<float> (256, 0.1f), 1000);
  cortext::testing::SeedEmbeddingV2 (
      *store, 12, std::vector<float> (256, 0.2f), 1000);
  cortext::testing::SeedMemoryV2 (
      *store, 1, 11, "source-a", "LONG_TERM", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (
      *store, 2, 12, "source-b", "ASSOCIATION", 1.0, 1000);

  bool fanout_current = false;
  bool search_invalidated = false;
  bool supersession_invalidated = false;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto root = std::make_unique<DynamicOperationSet> (
      std::make_unique<CustomMutationCacheProbeOp> (
          &fanout_current, &search_invalidated, &supersession_invalidated));
  SignalProcessor processor (cfg, store, std::move (root));
  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "source-c";
  signal.retention = Retention::Natural;
  signal.timestamp = 2000;
  processor.Process (signal);

  REQUIRE (fanout_current);
  REQUIRE (search_invalidated);
  REQUIRE (supersession_invalidated);
}

TEST_CASE ("Processor supersession cache covers association-only SQL rows",
           "[processor][cache][supersession][association]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  cortext::testing::SeedEmbeddingV2 (
      *store, 12, std::vector<float> (256, 0.2f), 1000);
  cortext::testing::SeedMemoryV2 (
      *store, 2, 12, "source-b", "ASSOCIATION", 1.0, 1000);
  const auto sql_rows = store->Execute (
      "SELECT m.memory_id FROM embeddings e "
      "JOIN memories m ON m.embedding_id = e.embedding_id "
      "WHERE m.kind IN ('LONG_TERM', 'ASSOCIATION')");
  REQUIRE (sql_rows.size () == 1);

  bool covered = false;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto root = std::make_unique<DynamicOperationSet> (
      std::make_unique<CaptureAssociationSupersessionCoverageOp> (&covered));
  SignalProcessor processor (cfg, store, std::move (root));
  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "source-c";
  signal.retention = Retention::Natural;
  signal.timestamp = 2000;
  processor.Process (signal);
  REQUIRE (covered);
}

TEST_CASE ("Processor calibrates staggered lazy RIF migration timestamps",
           "[processor][rif][migration][equivalence]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::store::DebugApplyCoreMigrationsThroughForTest (*store, 27);
  store->Execute (
      "INSERT INTO memories(memory_id, source_id, kind, start_ts, n_signals, "
      "modality, strength, suppression, suppression_ts, created_at) VALUES"
      "(1, 'source-a', 'LONG_TERM', 1000, 1, 'text', 0.5, 0.5, 1000, 1000),"
      "(2, 'source-b', 'ASSOCIATION', 1500, 1, 'image', 0.5, 0.5, 1500, 1500)");
  cortext::store::ApplyMigrations (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto root = std::make_unique<DynamicOperationSet> (
      std::make_unique<AdvanceRifEpochOp> ());
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::
          MarkEngineOwnedJournalAware (std::move (root)));
  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "source-c";
  signal.retention = Retention::Natural;
  signal.timestamp = 2000;
  processor.Process (signal);

  const auto rows = store->Execute (
      "SELECT memory_id, strength, suppression FROM rif_effective_memories "
      "ORDER BY memory_id");
  REQUIRE (rows.size () == 2);
  REQUIRE (std::any_cast<double> (rows[0].at ("strength"))
           == Catch::Approx (0.55));
  REQUIRE (std::any_cast<double> (rows[0].at ("suppression"))
           == Catch::Approx (0.45));
  REQUIRE (std::any_cast<double> (rows[1].at ("strength"))
           == Catch::Approx (0.525));
  REQUIRE (std::any_cast<double> (rows[1].at ("suppression"))
           == Catch::Approx (0.475));

  signal.timestamp = 3000;
  processor.Process (signal);
  const auto continued_rows = store->Execute (
      "SELECT memory_id, strength, suppression FROM rif_effective_memories "
      "ORDER BY memory_id");
  REQUIRE (continued_rows.size () == 2);
  REQUIRE (std::any_cast<double> (continued_rows[0].at ("strength"))
           == Catch::Approx (0.595));
  REQUIRE (std::any_cast<double> (continued_rows[0].at ("suppression"))
           == Catch::Approx (0.405));
  REQUIRE (std::any_cast<double> (continued_rows[1].at ("strength"))
           == Catch::Approx (0.5725));
  REQUIRE (std::any_cast<double> (continued_rows[1].at ("suppression"))
           == Catch::Approx (0.4275));
}

TEST_CASE ("Migrated active working slots reanchor persisted strength",
           "[processor][rif][migration][working_memory][restart]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::store::DebugApplyCoreMigrationsThroughForTest (*store, 27);
  cortext::testing::SeedEmbeddingV2 (
      *store, 100, std::vector<float> (256, 0.125f), 1000);
  store->Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
      "modality, start_ts, n_signals, strength, last_access, "
      "suppression, suppression_ts, created_at) "
      "VALUES(1, 100, 'working/source', 'WORKING', 'text', 1000, 1, "
      "0.5, 1000, 0.5, 1000, 1000)");
  cortext::store::ApplyMigrations (*store);

  bool updated = false;
  ProcessorContext *captured = nullptr;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.clock = std::make_shared<FixedClock> (1000);
  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "working/source";
  signal.timestamp = 1000;
  {
    auto root = std::make_unique<DynamicOperationSet> (
        std::make_unique<UpdateFirstWorkingSlotStrengthOp> (
            0.2, &updated, &captured));
    SignalProcessor processor (cfg, store, std::move (root));
    processor.Process (signal);
    REQUIRE (updated);
    REQUIRE (captured);
    const auto sidecar = ExecutionCacheState (*captured);
    REQUIRE (sidecar);
    REQUIRE (sidecar->rif_active_epoch.database);
    const auto epoch_rows = sidecar->rif_active_epoch.database->Execute (
        "SELECT recovery_total FROM active_state WHERE memory_id = 1");
    REQUIRE (epoch_rows.size () == 1);
    REQUIRE (std::any_cast<double> (epoch_rows[0].at ("recovery_total"))
             == Catch::Approx (0.7));
  }

  auto effective = store->Execute (
      "SELECT strength, suppression FROM rif_effective_memories "
      "WHERE memory_id = 1");
  REQUIRE (effective.size () == 1);
  REQUIRE (std::any_cast<double> (effective[0].at ("strength"))
           == Catch::Approx (0.2));
  REQUIRE (std::any_cast<double> (effective[0].at ("suppression"))
           == Catch::Approx (0.5));

  ProcessorContext *restarted_context = nullptr;
  {
    auto capture = std::make_unique<SetOrCaptureThroughputOp> ();
    capture->context_address = &restarted_context;
    auto root = std::make_unique<DynamicOperationSet> (
        std::move (capture));
    SignalProcessor restarted (cfg, store, std::move (root));
    restarted.Process (signal);
    REQUIRE (restarted_context);
    const auto restarted_sidecar = ExecutionCacheState (*restarted_context);
    REQUIRE (restarted_sidecar);
    REQUIRE (restarted_sidecar->rif_active_epoch.database);
    const auto restarted_rows
        = restarted_sidecar->rif_active_epoch.database->Execute (
            "SELECT recovery_total FROM active_state WHERE memory_id = 1");
    REQUIRE (restarted_rows.size () == 1);
    REQUIRE (std::any_cast<double> (
                 restarted_rows[0].at ("recovery_total"))
             == Catch::Approx (0.7));
  }

  auto recovery_tx = store->Begin ();
  cortext::operations::rif_state_internal::AdvanceRecovery (
      *recovery_tx, 2000, 10000.0);
  recovery_tx->Commit ();
  effective = store->Execute (
      "SELECT strength, suppression FROM rif_effective_memories "
      "WHERE memory_id = 1");
  REQUIRE (std::any_cast<double> (effective[0].at ("strength"))
           == Catch::Approx (0.25));
  REQUIRE (std::any_cast<double> (effective[0].at ("suppression"))
           == Catch::Approx (0.45));
}

TEST_CASE ("RIF active epoch requires consolidation at its bound and resets "
           "only after a successful maintenance commit",
           "[processor][rif][active_epoch][bound]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto root = std::make_unique<DynamicOperationSet> (
      std::make_unique<PrimeRifEpochBoundaryOp> ());
  SignalProcessor processor (cfg, store, std::move (root));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "source-a";
  signal.retention = Retention::Natural;
  signal.timestamp = 1000;
  const auto at_limit = processor.Process (signal);
  REQUIRE (at_limit.consolidation_state == ConsolidationState::Required);
  REQUIRE (at_limit.operation_ms.at (
               "SignalProcessor.rif_active_epoch_event_count")
           == 512.0);

  signal.timestamp = 2000;
  const auto ignored = processor.Process (signal);
  REQUIRE (ignored.consolidation_state == ConsolidationState::Required);
  REQUIRE (ignored.operation_ms.at (
               "SignalProcessor.rif_active_epoch_event_count")
           == 513.0);

  signal.force_consolidation = true;
  signal.timestamp = 3000;
  const auto reset = processor.Process (signal);
  REQUIRE (reset.consolidation_state == ConsolidationState::None);
  REQUIRE (reset.operation_ms.at (
               "SignalProcessor.rif_active_epoch_event_count")
           == 0.0);
  REQUIRE (reset.operation_ms.at (
               "SignalProcessor.rif_active_epoch_mutation_count")
           == 0.0);
  REQUIRE (reset.operation_ms.at (
               "SignalProcessor.rif_active_epoch_allocated_bytes")
           < static_cast<double> (
               operations::rif_active_epoch_cache_internal::
                   kAllocatedByteLimit));
}

TEST_CASE ("Failed consolidation keeps the prior RIF epoch and a later "
           "successful commit resets it exactly once",
           "[processor][rif][active_epoch][consolidation][rollback]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  store->Execute ("PRAGMA foreign_keys = ON");
  store->Execute ("CREATE TABLE consolidation_derived(id INTEGER PRIMARY KEY)");
  store->Execute ("CREATE TABLE deferred_parent(id INTEGER PRIMARY KEY)");
  store->Execute (
      "CREATE TABLE deferred_child("
      "parent_id INTEGER REFERENCES deferred_parent(id) "
      "DEFERRABLE INITIALLY DEFERRED)");
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto operation = std::make_unique<ConsolidationInsertAndFailCommitOnceOp> ();
  auto *capturing_operation = operation.get ();
  auto root = std::make_unique<DynamicOperationSet> (std::move (operation));
  SignalProcessor processor (cfg, store, std::move (root));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "source-a";
  signal.retention = Retention::Natural;
  signal.timestamp = 1000;
  const auto ordinary = processor.Process (signal);
  REQUIRE (ordinary.operation_ms.at (
               "SignalProcessor.rif_active_epoch_event_count")
           == 1.0);

  signal.force_consolidation = true;
  signal.timestamp = 2000;
  REQUIRE_THROWS (processor.Process (signal));
  REQUIRE (store->Execute (
               "SELECT COUNT(*) AS n FROM consolidation_derived")[0]
               .at ("n")
               .type ()
           == typeid (long long));
  REQUIRE (std::any_cast<long long> (
               store->Execute (
                   "SELECT COUNT(*) AS n FROM consolidation_derived")[0]
                   .at ("n"))
           == 0LL);
  REQUIRE (capturing_operation->captured != nullptr);
  const auto after_failure
      = ExecutionCacheState (*capturing_operation->captured);
  REQUIRE (after_failure);
  REQUIRE (after_failure->rif_active_epoch.event_count == 1);
  REQUIRE_FALSE (after_failure->rif_active_epoch.pending_rebuild);

  signal.timestamp = 3000;
  const auto success = processor.Process (signal);
  REQUIRE (success.operation_ms.at (
               "SignalProcessor.rif_active_epoch_event_count")
           == 0.0);
  REQUIRE (store->Execute (
               "SELECT COUNT(*) AS n FROM consolidation_derived")[0]
               .at ("n")
               .type ()
           == typeid (long long));
  REQUIRE (std::any_cast<long long> (
               store->Execute (
                   "SELECT COUNT(*) AS n FROM consolidation_derived")[0]
                   .at ("n"))
           == 1LL);
}

TEST_CASE ("Post-commit consolidation epoch reset failure rebuilds without "
           "replaying derived mutations",
           "[processor][rif][active_epoch][consolidation][publication]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  store->Execute ("CREATE TABLE consolidation_derived(id INTEGER PRIMARY KEY)");
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto root = std::make_unique<DynamicOperationSet> (
      std::make_unique<ConsolidationInsertOp> ());
  SignalProcessor processor (cfg, store, std::move (root));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "source-a";
  signal.retention = Retention::Natural;
  signal.force_consolidation = true;
  signal.timestamp = 1000;
  cortext::testing::SetRifActiveEpochPublishFailureStageForTest (3);
  const auto output = processor.Process (signal);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.rif_epoch_publication_recovery_count")
           == 1.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.rif_active_epoch_event_count")
           == 0.0);
  REQUIRE (store->Execute (
               "SELECT COUNT(*) AS n FROM consolidation_derived")[0]
               .at ("n")
               .type ()
           == typeid (long long));
  REQUIRE (std::any_cast<long long> (
               store->Execute (
                   "SELECT COUNT(*) AS n FROM consolidation_derived")[0]
                   .at ("n"))
           == 1LL);
}

TEST_CASE ("RIF epoch retry preserves a committed consolidation reset after "
           "publication and recovery both fail",
           "[processor][rif][active_epoch][consolidation][failure][retry]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  store->Execute ("CREATE TABLE consolidation_derived(id INTEGER PRIMARY KEY)");
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto root = std::make_unique<DynamicOperationSet> (
      std::make_unique<ConsolidationInsertOp> ());
  SignalProcessor processor (cfg, store, std::move (root));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "source-a";
  signal.retention = Retention::Natural;
  signal.timestamp = 1000;
  const auto ordinary = processor.Process (signal);
  REQUIRE (ordinary.operation_ms.at (
               "SignalProcessor.rif_active_epoch_event_count")
           == 1.0);

  signal.force_consolidation = true;
  signal.timestamp = 2000;
  cortext::testing::SetRifActiveEpochPublishFailureMaskForTest (
      (1U << 3U) | (1U << 4U));
  const auto failed_publication = processor.Process (signal);
  REQUIRE (failed_publication.operation_ms.at (
               "SignalProcessor.rif_epoch_publication_recovery_count")
           == 1.0);
  REQUIRE (failed_publication.operation_ms.at (
               "SignalProcessor.rif_active_epoch_event_count")
           == 0.0);

  signal.force_consolidation = false;
  signal.timestamp = 3000;
  const auto retried = processor.Process (signal);
  REQUIRE (retried.operation_ms.at (
               "SignalProcessor.rif_epoch_publication_recovery_count")
           == 0.0);
  REQUIRE (retried.operation_ms.at (
               "SignalProcessor.rif_active_epoch_event_count")
           == 1.0);
  REQUIRE (std::any_cast<long long> (
               store->Execute (
                   "SELECT COUNT(*) AS n FROM consolidation_derived")[0]
                   .at ("n"))
           == 1LL);
}

TEST_CASE ("Durable checkpoint failure cannot replay the committed signal",
           "[processor][rif][active_epoch][durable][barrier]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  store->Execute (
      "CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)");
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto root
      = std::make_unique<DynamicOperationSet> (std::make_unique<InsertOp> ());
  SignalProcessor processor (cfg, store, std::move (root));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "source-a";
  signal.retention = Retention::Durable;
  signal.timestamp = 1000;
  cortext::testing::SetSQLiteCheckpointFailureOnceForTest ();
  const auto first = processor.Process (signal);
  REQUIRE (first.operation_ms.at (
               "SignalProcessor.sqlite_wal_checkpoint_failure_count")
           == 1.0);
  REQUIRE (std::any_cast<long long> (
               store->Execute ("SELECT COUNT(*) AS n FROM t")[0].at ("n"))
           == 1LL);

  signal.timestamp = 2000;
  const auto second = processor.Process (signal);
  REQUIRE (second.operation_ms.at (
               "SignalProcessor.sqlite_wal_checkpoint_failure_count")
           == 0.0);
  REQUIRE (std::any_cast<long long> (
               store->Execute ("SELECT COUNT(*) AS n FROM t")[0].at ("n"))
           == 2LL);
}

TEST_CASE ("SignalProcessor executes pipeline in order", "[processor][order]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  std::vector<int> order;
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<RecordOrderOp> (&order, 1),
      std::make_unique<RecordOrderOp> (&order, 2),
      std::make_unique<RecordOrderOp> (&order, 3));

  SignalProcessor proc (cfg, store, std::move (pipeline));

  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  s.timestamp = 0;
  s.source_id = "test";

  proc.Process (s);

  REQUIRE (order == std::vector<int> { 1, 2, 3 });
}

TEST_CASE ("SignalProcessor treats persisted accumulators as volatile",
           "[processor][accumulator]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  cortext::testing::InitializeCoreSchema (*store);
  store->Execute (
      "INSERT INTO accumulators(source_id, n, t_start, last_signal_ts) "
      "VALUES('stale/source', 3, 1000, 2000)",
      {});

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  std::size_t accumulator_count = 999;
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<CaptureAccumulatorCountOp> (&accumulator_count));

  SignalProcessor proc (cfg, store, std::move (pipeline));

  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  s.timestamp = 3000;
  s.source_id = "test";

  proc.Process (s);

  REQUIRE (accumulator_count == 0);
}

TEST_CASE ("SignalProcessor restores exact durable and cache-only state after "
           "rollback",
           "[processor][rollback][cache]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  cortext::testing::InitializeCoreSchema (*store);
  cortext::testing::SeedEmbeddingV2 (
      *store, 51LL, std::vector<float> (256, 0.125f), 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 41LL, 51LL, "seed/source",
                                  "LONG_TERM", 1.0, 1000LL);
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  bool retrieval_restored = false;
  bool volatile_restored = false;
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<ExerciseRollbackCacheRestoreOp> (
          &retrieval_restored, &volatile_restored));
  SignalProcessor proc (cfg, store, std::move (pipeline));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 2000;
  const auto initial_output = proc.Process (signal);
  REQUIRE (initial_output.operation_ms.at (
               "SignalProcessor.snapshot_full_cache_copy_count")
           == 1.0);

  signal.timestamp = 3000;
  REQUIRE_THROWS (proc.Process (signal));

  signal.timestamp = 4000;
  proc.Process (signal);
  REQUIRE (retrieval_restored);
  REQUIRE (volatile_restored);
}

TEST_CASE ("SignalProcessor rollback rebuild preserves original base lineage",
           "[processor][rollback][cache][reconstruction][regression]")
{
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  cortext::testing::SeedEmbeddingV2 (
      *store, 100LL, std::vector<float> (256, 0.125f), 1000LL);
  cortext::testing::SeedEmbeddingV2 (
      *store, 101LL, std::vector<float> (256, 0.25f), 1100LL);
  cortext::testing::SeedMemoryV2 (*store, 10LL, 100LL, "lineage/source",
                                  "LONG_TERM", 1.0, 1000LL);
  store->Execute (
      "INSERT INTO memory_reconstructions("
      "reconstruction_id, memory_id, embedding_id, created_at) "
      "VALUES(1, 10, 101, 1100)");
  cortext::testing::SeedCurrentMemoryEmbeddingV2 (*store, 10LL, 101LL);

  long long observed_base = 0;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto root = std::make_unique<DynamicOperationSet> (
      std::make_unique<CaptureBaseLineageAcrossFailureOp> (&observed_base));
  SignalProcessor processor (cfg, store, std::move (root));
  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 2000;
  processor.Process (signal);
  REQUIRE (observed_base == 100LL);
  signal.timestamp = 3000;
  REQUIRE_THROWS (processor.Process (signal));
  signal.timestamp = 4000;
  processor.Process (signal);
  REQUIRE (observed_base == 100LL);
}

TEST_CASE ("SignalProcessor journal-aware writes retain DB cache ownership "
           "without deep snapshots",
           "[processor][rollback][cache][journal][scaling]")
{
  for (const auto retention : { Retention::Natural, Retention::Durable })
    {
      auto uniq = SQLiteStore::Create (":memory:");
      std::shared_ptr<Store> store (std::move (uniq));
      cortext::testing::InitializeCoreSchema (*store);
      for (long long id = 1; id <= 32; ++id)
        {
          cortext::testing::SeedEmbeddingV2 (
              *store, id, std::vector<float> (256, 0.125f), 1000LL + id);
          cortext::testing::SeedMemoryV2 (
              *store, id, id, "seed/source", "LONG_TERM", 1.0,
              1000LL + id);
        }

      SignalProcessor::Config cfg;
      cortext::testing::RequireEncoder (cfg);
      bool retained = false;
      auto pipeline = std::make_unique<DynamicOperationSet> (
          std::make_unique<CaptureCacheIdentityOp> (&retained));
      SignalProcessor processor (
          cfg, store,
          operations::signal_record_rollback_internal::MarkJournalAware (
              std::move (pipeline)));

      Signal signal;
      signal.embedding = Eigen::VectorXf::Zero (256);
      signal.source_id = "test";
      signal.retention = retention;
      signal.timestamp = 2000;
      const auto first = processor.Process (signal);
      REQUIRE (first.operation_ms.at (
                   "SignalProcessor.snapshot_full_cache_copy_count")
               == 0.0);
      REQUIRE (first.operation_ms.at (
                   "SignalProcessor.snapshot_cache_entry_copy_count")
               == 0.0);
      signal.timestamp = 3000;
      const auto second = processor.Process (signal);
      REQUIRE (second.operation_ms.at (
                   "SignalProcessor.snapshot_cache_entry_copy_count")
               == 0.0);
      REQUIRE (retained);
    }
}

TEST_CASE ("SignalProcessor journal-aware commit failure rebuilds DB caches "
           "and restores volatile stores",
           "[processor][rollback][cache][journal][commit]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  cortext::testing::InitializeCoreSchema (*store);
  cortext::testing::SeedEmbeddingV2 (
      *store, 51LL, std::vector<float> (256, 0.125f), 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 41LL, 51LL, "seed/source",
                                  "LONG_TERM", 1.0, 1000LL);
  cortext::testing::SeedEmbeddingV2 (
      *store, 52LL, std::vector<float> (256, 0.25f), 1100LL);
  cortext::testing::SeedMemoryV2 (*store, 42LL, 52LL, "seed/replacement",
                                  "LONG_TERM", 1.0, 1100LL);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, "
      "edge_type, weight, last_reinforced) "
      "VALUES(42, 41, 'supersedes', 1.0, 1100)", {});
  store->Execute ("CREATE TABLE deferred_parent(id INTEGER PRIMARY KEY)", {});
  store->Execute (
      "CREATE TABLE deferred_child("
      "parent_id INTEGER REFERENCES deferred_parent(id) "
      "DEFERRABLE INITIALLY DEFERRED)",
      {});

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  bool restored = false;
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<ExerciseJournalAwareCommitFailureOp> (&restored));
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::MarkJournalAware (
          std::move (pipeline)));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 2000;
  processor.Process (signal);
  signal.timestamp = 3000;
  REQUIRE_THROWS (processor.Process (signal));
  signal.timestamp = 4000;
  processor.Process (signal);
  REQUIRE (restored);
}

TEST_CASE ("SignalProcessor journal-aware read-only rollback rebuilds DB "
           "caches and restores volatile stores",
           "[processor][rollback][cache][journal][read_only]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  cortext::testing::InitializeCoreSchema (*store);
  cortext::testing::SeedEmbeddingV2 (
      *store, 51LL, std::vector<float> (256, 0.125f), 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 41LL, 51LL, "seed/source",
                                  "LONG_TERM", 1.0, 1000LL);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  bool restored = false;
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<ExerciseJournalAwareReadOnlyRollbackOp> (&restored));
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::MarkJournalAware (
          std::move (pipeline)));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 2000;
  processor.Process (signal);
  signal.retention = Retention::Ephemeral;
  signal.timestamp = 3000;
  processor.Process (signal);
  signal.retention = Retention::Natural;
  signal.timestamp = 4000;
  processor.Process (signal);
  REQUIRE (restored);
}

TEST_CASE ("SignalProcessor rollback preserves pending Natural record storage",
           "[processor][rollback][accumulator][scaling]")
{
  const auto backup_count_before
      = cortext::testing::SignalRecordRollbackBackupCountForTest ();
  const auto copied_count_before
      = cortext::testing::SignalRecordRollbackCopiedRecordCountForTest ();
  for (const std::size_t record_count : { 32U, 512U, 4096U })
    {
      auto uniq = SQLiteStore::Create (":memory:");
      std::shared_ptr<Store> store (std::move (uniq));
      SignalProcessor::Config cfg;
      cortext::testing::RequireEncoder (cfg);
      bool restored_without_copy = false;
      auto pipeline = std::make_unique<DynamicOperationSet> (
          std::make_unique<ExercisePendingUnitRollbackOwnershipOp> (
              record_count, &restored_without_copy));
      SignalProcessor processor (
          cfg, store,
          operations::signal_record_rollback_internal::MarkJournalAware (
              std::move (pipeline)));

      Signal signal;
      signal.embedding = Eigen::VectorXf::Zero (256);
      signal.source_id = "test";
      signal.timestamp = 1000;
      processor.Process (signal);
      signal.timestamp = 2000;
      REQUIRE_THROWS (processor.Process (signal));
      signal.timestamp = 3000;
      processor.Process (signal);

      REQUIRE (restored_without_copy);
    }
  REQUIRE (cortext::testing::SignalRecordRollbackBackupCountForTest ()
           == backup_count_before);
  REQUIRE (cortext::testing::SignalRecordRollbackCopiedRecordCountForTest ()
           == copied_count_before);
}

TEST_CASE ("SignalProcessor lazily backs up records for a failing flush",
           "[processor][rollback][accumulator][flush]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  bool restored = false;
  const auto backup_count_before
      = cortext::testing::SignalRecordRollbackBackupCountForTest ();
  const auto copied_count_before
      = cortext::testing::SignalRecordRollbackCopiedRecordCountForTest ();
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<ExerciseFailingFlushRecordRollbackOp> (&restored));
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::MarkJournalAware (
          std::move (pipeline)));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 1000;
  processor.Process (signal);
  signal.timestamp = 2000;
  REQUIRE_THROWS (processor.Process (signal));
  signal.timestamp = 3000;
  processor.Process (signal);

  REQUIRE (restored);
  REQUIRE (cortext::testing::SignalRecordRollbackBackupCountForTest ()
           == backup_count_before + 1);
  REQUIRE (cortext::testing::SignalRecordRollbackCopiedRecordCountForTest ()
           == copied_count_before + 3);
}

TEST_CASE ("SignalProcessor snapshot setup restores moved ownership on failure",
           "[processor][rollback][accumulator][snapshot_setup]")
{
  for (const int throw_stage : { 1, 2, 3 })
    {
      auto uniq = SQLiteStore::Create (":memory:");
      std::shared_ptr<Store> store (std::move (uniq));
      SignalProcessor::Config cfg;
      cortext::testing::RequireEncoder (cfg);
      bool restored = false;
      auto pipeline = std::make_unique<DynamicOperationSet> (
          std::make_unique<ExerciseSnapshotSetupExceptionSafetyOp> (&restored));
      SignalProcessor processor (
          cfg, store,
          operations::signal_record_rollback_internal::MarkJournalAware (
              std::move (pipeline)));

      Signal signal;
      signal.embedding = Eigen::VectorXf::Zero (256);
      signal.source_id = "test";
      signal.timestamp = 1000;
      processor.Process (signal);
      cortext::testing::SetSignalRecordSnapshotSetupThrowStageForTest (
          throw_stage);
      signal.timestamp = 2000;
      REQUIRE_THROWS (processor.Process (signal));
      signal.timestamp = 3000;
      processor.Process (signal);
      REQUIRE (restored);
    }
}

TEST_CASE ("SignalProcessor journals one erased working-memory slot by ownership",
           "[processor][rollback][working_memory][scaling]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  bool restored = false;
  const auto copied_count_before
      = cortext::testing::SignalRecordRollbackCopiedRecordCountForTest ();
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<ExerciseWorkingMemoryEraseOwnershipOp> (&restored));
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::MarkJournalAware (
          std::move (pipeline)));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 1000;
  processor.Process (signal);
  signal.timestamp = 2000;
  REQUIRE_THROWS (processor.Process (signal));
  signal.timestamp = 3000;
  processor.Process (signal);

  REQUIRE (restored);
  REQUIRE (cortext::testing::SignalRecordRollbackCopiedRecordCountForTest ()
           == copied_count_before);
}

TEST_CASE ("SignalProcessor trims working-memory appends without copying history",
           "[processor][rollback][working_memory][scaling]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  bool restored = false;
  const auto copied_count_before
      = cortext::testing::SignalRecordRollbackCopiedRecordCountForTest ();
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<ExerciseWorkingMemoryAppendOwnershipOp> (&restored));
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::MarkJournalAware (
          std::move (pipeline)));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 1000;
  processor.Process (signal);
  signal.timestamp = 2000;
  REQUIRE_THROWS (processor.Process (signal));
  signal.timestamp = 3000;
  processor.Process (signal);

  REQUIRE (restored);
  REQUIRE (cortext::testing::SignalRecordRollbackCopiedRecordCountForTest ()
           == copied_count_before);
}

TEST_CASE ("SignalProcessor full backup upgrade preserves exact active records",
           "[processor][rollback][accumulator][full_backup]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  bool restored = false;
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<ExerciseActiveToFullBackupUpgradeOp> (&restored));
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::MarkJournalAware (
          std::move (pipeline)));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 1000;
  processor.Process (signal);
  signal.timestamp = 2000;
  REQUIRE_THROWS (processor.Process (signal));
  signal.timestamp = 3000;
  processor.Process (signal);
  REQUIRE (restored);
}

TEST_CASE ("SignalProcessor custom nested source changes roll back automatically",
           "[processor][rollback][nested][custom]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  bool restored = false;
  auto operation
      = std::make_unique<ExerciseNestedDifferentSourceRollbackOp> (&restored);
  auto *operation_ptr = operation.get ();
  auto pipeline
      = std::make_unique<DynamicOperationSet> (std::move (operation));
  SignalProcessor processor (cfg, store, std::move (pipeline));
  operation_ptr->processor = &processor;

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "outer/source";
  signal.timestamp = 1000;
  processor.Process (signal);
  signal.timestamp = 2000;
  REQUIRE_THROWS (processor.Process (signal));
  signal.timestamp = 3000;
  processor.Process (signal);
  REQUIRE (restored);
}

TEST_CASE ("SignalProcessor restores consolidation throughput after rollback",
           "[processor][rollback][consolidation]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  bool restored = false;
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<ExerciseThroughputRollbackOp> (&restored));
  SignalProcessor processor (cfg, store, std::move (pipeline));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 1000;
  processor.Process (signal);
  signal.timestamp = 2000;
  REQUIRE_THROWS (processor.Process (signal));
  signal.timestamp = 3000;
  processor.Process (signal);
  REQUIRE (restored);
}

TEST_CASE ("Ephemeral processing restores consolidation throughput exactly",
           "[processor][ephemeral][consolidation]")
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  bool restored = false;
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<ExerciseThroughputEphemeralOp> (&restored));
  SignalProcessor processor (cfg, store, std::move (pipeline));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 1000;
  processor.Process (signal);
  signal.timestamp = 2000;
  signal.retention = Retention::Ephemeral;
  processor.Process (signal);
  signal.timestamp = 3000;
  signal.retention = Retention::Natural;
  processor.Process (signal);
  REQUIRE (restored);
}

TEST_CASE ("Consolidation throughput registry isolates processor stores and "
           "cleans up on recreation",
           "[processor][consolidation][lifecycle]")
{
  namespace throughput
      = operations::consolidation_throughput_state_internal;
  const std::size_t baseline_size = throughput::RegistrySizeForTest ();
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);

  auto first_store_unique = SQLiteStore::Create (":memory:");
  auto second_store_unique = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> first_store (std::move (first_store_unique));
  std::shared_ptr<Store> second_store (std::move (second_store_unique));
  ProcessorContext *first_address = nullptr;
  ProcessorContext *second_address = nullptr;

  auto first_op = std::make_unique<SetOrCaptureThroughputOp> ();
  first_op->state_to_set = throughput::State { 1.0, 8.0, true };
  first_op->context_address = &first_address;
  auto second_op = std::make_unique<SetOrCaptureThroughputOp> ();
  second_op->state_to_set = throughput::State { 4.0, 14.0, true };
  second_op->context_address = &second_address;
  auto first = std::make_unique<SignalProcessor> (
      cfg, first_store,
      std::make_unique<DynamicOperationSet> (std::move (first_op)));
  auto second = std::make_unique<SignalProcessor> (
      cfg, second_store,
      std::make_unique<DynamicOperationSet> (std::move (second_op)));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 1000;
  first->Process (signal);
  second->Process (signal);
  REQUIRE (first_address != second_address);
  REQUIRE (throughput::Find (*first_address).floor == 1.0);
  REQUIRE (throughput::Find (*first_address).peak == 8.0);
  REQUIRE (throughput::Find (*second_address).floor == 4.0);
  REQUIRE (throughput::Find (*second_address).peak == 14.0);
  REQUIRE (throughput::RegistrySizeForTest () == baseline_size + 2);

  first.reset ();
  second.reset ();
  REQUIRE (throughput::RegistrySizeForTest () == baseline_size);

  auto third_store_unique = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> third_store (std::move (third_store_unique));
  throughput::State fresh_state { -1.0, -1.0 };
  auto third_op = std::make_unique<SetOrCaptureThroughputOp> ();
  third_op->captured = &fresh_state;
  auto third = std::make_unique<SignalProcessor> (
      cfg, third_store,
      std::make_unique<DynamicOperationSet> (std::move (third_op)));
  signal.timestamp = 2000;
  third->Process (signal);
  REQUIRE (fresh_state.floor == 0.0);
  REQUIRE (fresh_state.peak == 0.0);
  third.reset ();
  REQUIRE (throughput::RegistrySizeForTest () == baseline_size);
}

TEST_CASE ("SignalProcessor reuses lifecycle-owned exact rollback storage",
           "[processor][rollback][lifecycle]")
{
  const std::size_t baseline_owner_count
      = cortext::testing::ProcessorRollbackSnapshotOwnerCountForTest ();
  const std::size_t baseline_reuse_count
      = cortext::testing::ProcessorRollbackSnapshotReuseCountForTest ();
  bool restored = false;
  {
    auto uniq = SQLiteStore::Create (":memory:");
    std::shared_ptr<Store> store (std::move (uniq));
    cortext::testing::InitializeCoreSchema (*store);

    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    auto pipeline = std::make_unique<DynamicOperationSet> (
        std::make_unique<ExerciseWholeContextRollbackOp> (&restored));
    SignalProcessor proc (cfg, store, std::move (pipeline));

    Signal signal;
    signal.embedding = Eigen::VectorXf::Zero (256);
    signal.source_id = "test";
    signal.timestamp = 1000;
    proc.Process (signal);
    REQUIRE (
        cortext::testing::ProcessorRollbackSnapshotOwnerCountForTest ()
        == baseline_owner_count + 1);

    signal.timestamp = 2000;
    REQUIRE_THROWS (proc.Process (signal));

    signal.timestamp = 3000;
    proc.Process (signal);
    REQUIRE (restored);
    REQUIRE (
        cortext::testing::ProcessorRollbackSnapshotReuseCountForTest ()
        >= baseline_reuse_count + 1);
  }
  REQUIRE (cortext::testing::ProcessorRollbackSnapshotOwnerCountForTest ()
           == baseline_owner_count);
}

TEST_CASE ("SignalProcessor erases internal caches when destroyed on another "
           "thread",
           "[processor][cache][lifecycle]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  namespace execution_cache
      = operations::execution_cache_sidecar_internal;
  const std::size_t baseline_size = cache::RegistrySizeForTest ();
  const std::size_t baseline_execution_size
      = execution_cache::RegistrySizeForTest ();
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  cortext::testing::InitializeCoreSchema (*store);
  cortext::testing::SeedEmbeddingV2 (
      *store, 51LL, std::vector<float> (256, 0.125f), 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 41LL, 51LL, "seed/source",
                                  "LONG_TERM", 1.0, 1000LL);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<RecordOrderOp> (nullptr, 0));
  auto processor = std::make_unique<SignalProcessor> (
      cfg, store, std::move (pipeline));
  REQUIRE (cache::RegistrySizeForTest () == baseline_size + 1);
  REQUIRE (execution_cache::RegistrySizeForTest ()
           == baseline_execution_size + 1);

  std::thread teardown (
      [owner = std::move (processor)] () mutable { owner.reset (); });
  teardown.join ();
  REQUIRE (cache::RegistrySizeForTest () == baseline_size);
  REQUIRE (execution_cache::RegistrySizeForTest ()
           == baseline_execution_size);
}

TEST_CASE ("SignalProcessor rebuilds historical cache after force commit",
           "[processor][cache][consolidation][lifecycle]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  const std::size_t baseline_size = cache::RegistrySizeForTest ();
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  cortext::testing::InitializeCoreSchema (*store);
  cortext::testing::SeedEmbeddingV2 (
      *store, 51LL, std::vector<float> (256, 0.125f), 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 41LL, 51LL, "seed/source",
                                  "LONG_TERM", 1.0, 1000LL);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<RecordOrderOp> (nullptr, 0));
  SignalProcessor processor (cfg, store, std::move (pipeline));
  REQUIRE (cache::RegistrySizeForTest () == baseline_size + 1);

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 2000;
  signal.force_consolidation = true;
  processor.Process (signal);
  REQUIRE (cache::RegistrySizeForTest () == baseline_size + 1);

  signal.timestamp = 3000;
  signal.force_consolidation = false;
  processor.Process (signal);
  REQUIRE (cache::RegistrySizeForTest () == baseline_size + 1);
}

TEST_CASE ("SignalProcessor restores historical cache after force commit "
           "failure",
           "[processor][rollback][cache][consolidation]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  const std::size_t baseline_size = cache::RegistrySizeForTest ();
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  cortext::testing::InitializeCoreSchema (*store);
  cortext::testing::SeedEmbeddingV2 (
      *store, 51LL, std::vector<float> (256, 0.125f), 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 41LL, 51LL, "seed/source",
                                  "LONG_TERM", 1.0, 1000LL);
  store->Execute ("CREATE TABLE deferred_parent(id INTEGER PRIMARY KEY)", {});
  store->Execute (
      "CREATE TABLE deferred_child("
      "parent_id INTEGER REFERENCES deferred_parent(id) "
      "DEFERRABLE INITIALLY DEFERRED)",
      {});

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<InsertDeferredViolationOp> ());
  SignalProcessor processor (cfg, store, std::move (pipeline));
  REQUIRE (cache::RegistrySizeForTest () == baseline_size + 1);

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "test";
  signal.timestamp = 2000;
  signal.force_consolidation = true;
  REQUIRE_THROWS (processor.Process (signal));
  REQUIRE (cache::RegistrySizeForTest () == baseline_size + 1);
  const auto child_rows
      = store->Execute ("SELECT COUNT(*) AS c FROM deferred_child", {});
  REQUIRE (child_rows.size () == 1);
  REQUIRE (std::any_cast<long long> (child_rows[0].at ("c")) == 0LL);
}
