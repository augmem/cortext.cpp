#include "test_helpers.hpp"
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/clock.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/operations/memory_usage_record.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>
#include "../src/operations/active_signal_embedding_ring_internal.hpp"
#include "../src/operations/bounded_activation_shadow_internal.hpp"
#include "../src/operations/historical_surface_search_cache_internal.hpp"
#include "../src/operations/consolidation_throughput_state_internal.hpp"
#include "../src/operations/signal_record_rollback_internal.hpp"
#include "../src/processor/source_state_eviction_internal.hpp"
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <thread>
#include <tuple>
#include <unordered_set>

using namespace cortext;

namespace cortext::testing
{
std::size_t ProcessorRollbackSnapshotOwnerCountForTest ();
std::size_t ProcessorRollbackSnapshotReuseCountForTest ();
std::size_t SignalRecordRollbackBackupCountForTest ();
std::size_t SignalRecordRollbackCopiedRecordCountForTest ();
void SetSignalRecordSnapshotSetupThrowStageForTest (int stage);
void SetSQLiteCheckpointFailureOnceForTest ();
void SetRetrievalSurfaceReloadFailureOnceForTest ();
std::size_t RetrievalSurfaceReloadCountForTest ();
std::size_t FlushSnapshotCopiedEntryCountForTest ();
std::size_t FullOrphanEmbeddingSweepCountForTest ();
}

namespace cortext::store
{
void DebugApplyCoreMigrationsThroughForTest (Store &store,
                                             int64_t maximum_id);
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

struct ConsolidationSurfaceUpdateOp : IOperation
{
  explicit ConsolidationSurfaceUpdateOp (bool invalidate_cache = false)
      : invalidate_cache (invalidate_cache)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction &tx) const override
  {
    captured = &ctx.GetProcessorContext ();
    if (!ctx.GetSignal ().force_consolidation)
      return;
    auto &p_ctx = ctx.GetProcessorContext ();
    const auto found = p_ctx.retrieval_surface_index.find (1);
    REQUIRE (found != p_ctx.retrieval_surface_index.end ());
    REQUIRE (found->second < p_ctx.retrieval_surface_cache.size ());
    auto surface = p_ctx.retrieval_surface_cache[found->second];
    Eigen::VectorXf embedding = Eigen::VectorXf::Zero (256);
    embedding[0] = 1.0f;
    std::vector<float> values (256, 0.0f);
    values[0] = 1.0f;
    tx.Execute ("UPDATE embeddings SET embedding = ? WHERE embedding_id = 1",
                { values });
    surface.embedding = embedding;
    p_ctx.UpsertRetrievalSurface (std::move (surface));
    if (invalidate_cache)
      {
        operations::historical_surface_search_cache_internal::Erase (p_ctx);
        return;
      }
    operations::historical_surface_search_cache_internal::UpsertCurrent (
        p_ctx,
        { 1, 1, 1001, "LONG_TERM", "seed/1", std::move (embedding), 1 });
  }

  bool invalidate_cache = false;
  mutable ProcessorContext *captured = nullptr;
};

// Moves several indexed memories a short reconsolidation-style distance
// during forced consolidation.
struct ReconsolidationMoveOp : IOperation
{
  explicit ReconsolidationMoveOp (std::vector<long long> moved_ids)
      : moved_ids (std::move (moved_ids))
  {
  }

  static Eigen::VectorXf
  Seeded (std::uint64_t seed)
  {
    Eigen::VectorXf v (256);
    std::uint64_t state = seed * 0x9e3779b97f4a7c15ULL + 0x632be59bd9b4e019ULL;
    for (Eigen::Index dimension = 0; dimension < v.size (); ++dimension)
      {
        state += 0x9e3779b97f4a7c15ULL;
        std::uint64_t mixed = state;
        mixed = (mixed ^ (mixed >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        mixed = (mixed ^ (mixed >> 27U)) * 0x94d049bb133111ebULL;
        mixed ^= mixed >> 31U;
        v[dimension] = static_cast<float> (
                           static_cast<double> (mixed >> 11U)
                           / static_cast<double> (1ULL << 53U))
                       - 0.5f;
      }
    v.normalize ();
    return v;
  }

  static Eigen::VectorXf
  Original (long long memory_id)
  {
    return Seeded (static_cast<std::uint64_t> (memory_id));
  }

  // A short reconsolidation-style blend toward a co-activated direction.
  static Eigen::VectorXf
  Moved (long long memory_id)
  {
    Eigen::VectorXf embedding
        = Original (memory_id)
          + 0.35f * Seeded (static_cast<std::uint64_t> (memory_id) + 1000003U);
    embedding.normalize ();
    return embedding;
  }

  void
  Execute (OperationContext &ctx, Transaction &tx) const override
  {
    captured = &ctx.GetProcessorContext ();
    if (!ctx.GetSignal ().force_consolidation || emitted)
      return;
    auto &p_ctx = ctx.GetProcessorContext ();
    for (const long long memory_id : moved_ids)
      {
        const auto found = p_ctx.retrieval_surface_index.find (memory_id);
        REQUIRE (found != p_ctx.retrieval_surface_index.end ());
        REQUIRE (found->second < p_ctx.retrieval_surface_cache.size ());
        auto surface = p_ctx.retrieval_surface_cache[found->second];
        Eigen::VectorXf embedding = Moved (memory_id);
        tx.Execute (
            "UPDATE embeddings SET embedding = ? WHERE embedding_id = ?",
            { std::vector<float> (embedding.data (),
                                  embedding.data () + embedding.size ()),
              memory_id });
        surface.embedding = embedding;
        p_ctx.UpsertRetrievalSurface (std::move (surface));
        operations::historical_surface_search_cache_internal::UpsertCurrent (
            p_ctx, { memory_id, memory_id, 1000 + memory_id, "LONG_TERM",
                     "seed/" + std::to_string (memory_id),
                     std::move (embedding), memory_id });
      }
    emitted = true;
  }

  std::vector<long long> moved_ids;
  mutable bool emitted = false;
  mutable ProcessorContext *captured = nullptr;
};

struct CustomMutationCacheProbeOp : IOperation
{
  explicit CustomMutationCacheProbeOp (bool *search_invalidated)
      : search_invalidated (search_invalidated)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction &tx) const override
  {
    auto &p_ctx = ctx.GetProcessorContext ();
    tx.Execute (
        "INSERT INTO associations(source_memory_id, target_memory_id, "
        "edge_type, weight, last_reinforced) "
        "VALUES(1, 2, 'similar_to', 0.75, 2000)");
    if (search_invalidated)
      *search_invalidated
          = !operations::historical_surface_search_cache_internal::Find (p_ctx);
  }

  bool *search_invalidated = nullptr;
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
      return;
    if (call_count == 2)
      {
        processor_context.retrieval_surface_cache.clear ();
        processor_context.retrieval_surface_index.clear ();
        processor_context.retrieval_surface_embedding_index.clear ();
        tx.Execute ("INSERT INTO deferred_child(parent_id) VALUES(?)",
                    { 999LL });
        return;
      }
    if (restored)
      {
        *restored
            = processor_context.retrieval_surface_index.contains (41LL)
              && processor_context.retrieval_surface_embedding_index.contains (
                  51LL);
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
      return;
    if (call_count == 2)
      {
        processor_context.retrieval_surface_cache.clear ();
        processor_context.retrieval_surface_index.clear ();
        processor_context.retrieval_surface_embedding_index.clear ();
        return;
      }
    if (restored)
      {
        *restored
            = processor_context.retrieval_surface_index.contains (41LL)
              && processor_context.retrieval_surface_embedding_index.contains (
                  51LL);
      }
  }

  bool *restored = nullptr;
  mutable int call_count = 0;
};

struct PrimeQueryFeedbackOp : IOperation
{
  explicit PrimeQueryFeedbackOp (Eigen::VectorXf embedding)
      : embedding (std::move (embedding))
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.GetProcessorContext ().recent_context_embeddings = { embedding };
    ctx.SetRetrievedMemoryCandidates ({ { 42LL, 52LL, embedding, 1.0 } });
  }

  Eigen::VectorXf embedding;
};

// Simulates an operation that grows the retrieval surface during a read-only
// signal, which the rollback must not keep.
struct GrowRetrievalSurfaceOp : IOperation
{
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &p_ctx = ctx.GetProcessorContext ();
    ProcessorContext::RetrievalSurfaceEntry entry;
    entry.memory_id = 999LL;
    entry.embedding_id = 999LL;
    p_ctx.retrieval_surface_index[entry.memory_id]
        = p_ctx.retrieval_surface_cache.size ();
    p_ctx.retrieval_surface_cache.push_back (std::move (entry));
  }
};

using RetrievalSurfaceCapture
    = std::map<long long, std::tuple<long long, long long, std::string,
                                     std::vector<float>>>;

RetrievalSurfaceCapture
CaptureRetrievalSurface (const ProcessorContext &context)
{
  RetrievalSurfaceCapture capture;
  for (const auto &entry : context.retrieval_surface_cache)
    capture.emplace (
        entry.memory_id,
        std::make_tuple (
            entry.embedding_id, entry.start_ts, entry.source_id,
            std::vector<float> (entry.embedding.data (),
                                entry.embedding.data ()
                                    + entry.embedding.size ())));
  return capture;
}

struct CaptureRetrievalSurfaceOp : IOperation
{
  explicit CaptureRetrievalSurfaceOp (
      std::vector<RetrievalSurfaceCapture> *captures)
      : captures (captures)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    captures->push_back (
        CaptureRetrievalSurface (ctx.GetProcessorContext ()));
  }

  std::vector<RetrievalSurfaceCapture> *captures = nullptr;
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
  explicit ExerciseRollbackCacheRestoreOp (bool *retrieval_restored)
      : retrieval_restored (retrieval_restored)
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
        return;
      }
    if (call_count == 2)
      {
        processor_context.retrieval_surface_cache.clear ();
        processor_context.retrieval_surface_index.clear ();
        processor_context.retrieval_surface_embedding_index.clear ();
        throw std::runtime_error ("force rollback");
      }

    if (retrieval_restored)
      {
        *retrieval_restored
            = processor_context.retrieval_surface_index.contains (41LL)
              && processor_context.retrieval_surface_index.contains (77LL)
              && processor_context.retrieval_surface_embedding_index.contains (
                  51LL)
              && processor_context.retrieval_surface_embedding_index.contains (
                  88LL)
              && processor_context.retrieval_surface_cache.size () == 2
              && processor_context.retrieval_surface_cache[0].memory_id == 41LL
              && processor_context.retrieval_surface_cache[1].memory_id == 77LL;
      }
  }

  bool *retrieval_restored;
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
        foreign.memory_modality = "audio";
        processor_context.accumulator_states["foreign/source"]
            = std::move (foreign);
        processor_context.recent_scores = { 0.25, 0.75 };
        processor_context.memories_since_consolidation = 17;
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
        processor_context.memories_since_consolidation = 99;
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
              && accumulator_it->second.memory_modality == "audio"
              && processor_context.accumulator_states.at ("test")
                         .signals.front ().payload
                     == std::vector<unsigned char> (8, 0x11)
              && !processor_context.accumulator_states.contains (
                  "unexpected/source")
              && processor_context.recent_scores
                     == std::deque<double> { 0.25, 0.75 }
              && processor_context.memories_since_consolidation == 17;
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

TEST_CASE ("SignalProcessor restart restores only exact ring embeddings for "
           "the full knob-derived context window",
           "[processor][recent_context][restart][knobs]")
{
  namespace ring = operations::active_signal_embedding_ring_internal;
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  const int capacity = ring::Capacity (
      cfg.focus, cfg.sensitivity, cfg.stability);
  REQUIRE (capacity == 151);

  const Eigen::VectorXf centroid
      = Eigen::VectorXf::Constant (256, -1.0f);
  cortext::testing::SeedEmbeddingV2 (*store, 1, centroid, 0);
  auto tx = store->Begin ();
  for (long long signal_id = 1; signal_id <= capacity; ++signal_id)
    {
      tx->Execute (
          "INSERT INTO signals(signal_id, source_id, embedding_id, "
          "timestamp, modality, created_at) "
          "VALUES (?, 'opaque/restart', 1, ?, 'text', ?)",
          { signal_id, signal_id, signal_id });
      std::vector<float> exact (256, 0.0f);
      exact[0] = static_cast<float> (signal_id);
      ring::Upsert (*tx, signal_id, exact, signal_id, cfg.focus,
                    cfg.sensitivity, cfg.stability);
    }
  tx->Commit ();

  ProcessorContext *captured = nullptr;
  auto capture = std::make_unique<SetOrCaptureThroughputOp> ();
  capture->context_address = &captured;
  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.timestamp = static_cast<std::uint64_t> (capacity + 1);
  signal.source_id = "opaque/observer";
  signal.modality = "image";
  signal.mimetype = "image/test";
  SignalProcessor restarted (
      cfg, store,
      std::make_unique<DynamicOperationSet> (std::move (capture)));
  restarted.Process (signal);

  REQUIRE (captured != nullptr);
  REQUIRE (captured->recent_context_embeddings.size ()
           == static_cast<std::size_t> (capacity));
  for (int index = 0; index < capacity; ++index)
    {
      REQUIRE (captured->recent_context_embeddings[
                   static_cast<std::size_t> (index)][0]
               == static_cast<float> (index + 1));
    }
}

TEST_CASE ("SignalProcessor backfills the legacy exact context before the "
           "first post-migration ring write",
           "[processor][recent_context][restart][migration]")
{
  namespace ring = operations::active_signal_embedding_ring_internal;
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  const int capacity = ring::Capacity (
      cfg.focus, cfg.sensitivity, cfg.stability);
  REQUIRE (capacity == 151);

  for (long long signal_id = 1; signal_id < capacity; ++signal_id)
    {
      Eigen::VectorXf exact = Eigen::VectorXf::Zero (256);
      exact[0] = static_cast<float> (signal_id);
      cortext::testing::SeedEmbeddingV2 (
          *store, signal_id, exact, signal_id);
      store->Execute (
          "INSERT INTO signals(signal_id, source_id, embedding_id, "
          "timestamp, modality, created_at) "
          "VALUES (?, 'opaque/legacy', ?, ?, 'text', ?)",
          { signal_id, signal_id, signal_id, signal_id });
    }
  REQUIRE (store->Execute (
               "SELECT 1 FROM cortext_active_signal_embeddings LIMIT 1")
               .empty ());

  Eigen::VectorXf aggregate = Eigen::VectorXf::Constant (256, -1.0f);
  cortext::testing::SeedEmbeddingV2 (
      *store, 1000, aggregate, capacity);
  auto tx = store->Begin ();
  ring::EnsureCapacity (*tx, capacity);
  tx->Execute (
      "INSERT INTO signals(signal_id, source_id, embedding_id, timestamp, "
      "modality, created_at) VALUES (?, 'opaque/new', 1000, ?, 'image', ?)",
      { static_cast<long long> (capacity), static_cast<long long> (capacity),
        static_cast<long long> (capacity) });
  std::vector<float> newest_exact (256, 0.0f);
  newest_exact[0] = static_cast<float> (capacity);
  ring::InsertAtCapacity (*tx, capacity, newest_exact, capacity, capacity);
  tx->Commit ();

  ProcessorContext *captured = nullptr;
  auto capture = std::make_unique<SetOrCaptureThroughputOp> ();
  capture->context_address = &captured;
  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.timestamp = static_cast<std::uint64_t> (capacity + 1);
  signal.source_id = "opaque/observer";
  signal.modality = "audio";
  signal.mimetype = "audio/test";
  SignalProcessor restarted (
      cfg, store,
      std::make_unique<DynamicOperationSet> (std::move (capture)));
  restarted.Process (signal);

  REQUIRE (captured != nullptr);
  REQUIRE (captured->recent_context_embeddings.size ()
           == static_cast<std::size_t> (capacity));
  for (int index = 0; index < capacity; ++index)
    REQUIRE (captured->recent_context_embeddings[
                 static_cast<std::size_t> (index)][0]
             == static_cast<float> (index + 1));
}

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

  bool search_invalidated = false;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto root = std::make_unique<DynamicOperationSet> (
      std::make_unique<CustomMutationCacheProbeOp> (
          &search_invalidated));
  SignalProcessor processor (cfg, store, std::move (root));
  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "source-c";
  signal.retention = Retention::Natural;
  signal.timestamp = 2000;
  processor.Process (signal);

  REQUIRE (search_invalidated);
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

TEST_CASE ("Durable checkpoint failure cannot replay the committed signal",
           "[processor][durable][barrier]")
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

TEST_CASE ("Natural consolidation does not borrow the Durable checkpoint",
           "[processor][consolidation][durable][barrier]")
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
  signal.force_consolidation = true;
  signal.retention = Retention::Natural;
  signal.timestamp = 1000;
  const auto natural = processor.Process (signal);
  REQUIRE_FALSE (natural.operation_ms.contains (
      "SignalProcessor.sqlite_wal_checkpoint"));

  signal.retention = Retention::Durable;
  signal.timestamp = 2000;
  const auto durable = processor.Process (signal);
  REQUIRE (durable.operation_ms.contains (
      "SignalProcessor.sqlite_wal_checkpoint"));
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
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<ExerciseRollbackCacheRestoreOp> (
          &retrieval_restored));
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

namespace
{
std::shared_ptr<Store>
MakeReadOnlySurfaceStore (const Eigen::VectorXf &query)
{
  auto uniq = SQLiteStore::Create (":memory:");
  std::shared_ptr<Store> store (std::move (uniq));
  cortext::testing::InitializeCoreSchema (*store);
  Eigen::VectorXf other = Eigen::VectorXf::Zero (256);
  other[1] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (*store, 51LL, other, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 41LL, 51LL, "seed/other",
                                  "LONG_TERM", 1.0, 1000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 52LL, query, 1100LL);
  cortext::testing::SeedMemoryV2 (*store, 42LL, 52LL, "seed/match",
                                  "LONG_TERM", 1.0, 1100LL);
  return store;
}

Signal
MakeReadOnlyQuery (const Eigen::VectorXf &query, double timestamp)
{
  Signal signal;
  signal.embedding = query;
  signal.source_id = "eval/query";
  signal.retention = Retention::Ephemeral;
  signal.timestamp = timestamp;
  return signal;
}

RetrievalSurfaceCapture
FreshRetrievalSurface (const SignalProcessor::Config &cfg,
                       const std::shared_ptr<Store> &store,
                       const Signal &signal)
{
  std::vector<RetrievalSurfaceCapture> fresh;
  SignalProcessor fresh_processor (
      cfg, store,
      std::make_unique<DynamicOperationSet> (
          std::make_unique<CaptureRetrievalSurfaceOp> (&fresh)));
  fresh_processor.Process (signal);
  REQUIRE (fresh.size () == 1);
  return fresh[0];
}
} // namespace

TEST_CASE ("SignalProcessor engine read-only rollback keeps the retrieval "
           "surface without reloading it",
           "[processor][rollback][cache][journal][read_only]")
{
  Eigen::VectorXf query = Eigen::VectorXf::Zero (256);
  query[0] = 1.0f;
  auto store = MakeReadOnlySurfaceStore (query);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  std::vector<RetrievalSurfaceCapture> before;
  std::vector<RetrievalSurfaceCapture> after;
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<CaptureRetrievalSurfaceOp> (&before),
      std::make_unique<PrimeQueryFeedbackOp> (query),
      std::make_unique<operations::RecordMemoryUsage> (),
      std::make_unique<CaptureRetrievalSurfaceOp> (&after));
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::
          MarkEngineOwnedJournalAware (std::move (pipeline)));

  const auto reloads_before
      = cortext::testing::RetrievalSurfaceReloadCountForTest ();
  processor.Process (MakeReadOnlyQuery (query, 2000));
  processor.Process (MakeReadOnlyQuery (query, 3000));
  REQUIRE (cortext::testing::RetrievalSurfaceReloadCountForTest ()
           == reloads_before);

  REQUIRE (before.size () == 2);
  REQUIRE (after.size () == 2);
  REQUIRE (before[0].size () == 2);
  // The kept surface is what the query saw, and what a fresh processor
  // loads from the committed store.
  REQUIRE (after[0] == before[0]);
  REQUIRE (before[1] == before[0]);
  REQUIRE (before[1]
           == FreshRetrievalSurface (cfg, store,
                                     MakeReadOnlyQuery (query, 4000)));
}

TEST_CASE ("SignalProcessor engine read-only rollback reloads a surface the "
           "signal reshaped",
           "[processor][rollback][cache][journal][read_only]")
{
  Eigen::VectorXf query = Eigen::VectorXf::Zero (256);
  query[0] = 1.0f;
  auto store = MakeReadOnlySurfaceStore (query);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  std::vector<RetrievalSurfaceCapture> before;
  std::vector<RetrievalSurfaceCapture> after;
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<CaptureRetrievalSurfaceOp> (&before),
      std::make_unique<GrowRetrievalSurfaceOp> (),
      std::make_unique<CaptureRetrievalSurfaceOp> (&after));
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::
          MarkEngineOwnedJournalAware (std::move (pipeline)));

  processor.Process (MakeReadOnlyQuery (query, 2000));
  const auto reloads_before
      = cortext::testing::RetrievalSurfaceReloadCountForTest ();
  processor.Process (MakeReadOnlyQuery (query, 3000));
  REQUIRE (cortext::testing::RetrievalSurfaceReloadCountForTest ()
           == reloads_before + 1);

  REQUIRE (before.size () == 2);
  REQUIRE (after[1].contains (999LL));
  REQUIRE_FALSE (before[1].contains (999LL));
  REQUIRE (before[1]
           == FreshRetrievalSurface (cfg, store,
                                     MakeReadOnlyQuery (query, 4000)));
}

namespace
{
// Historical surface identities plus current-surface memory ids, as the
// sparse route and supersession search see them.
using HistoricalSurfaceCapture
    = std::pair<std::map<long long, std::vector<long long>>,
                std::vector<long long>>;

struct CaptureHistoricalSurfaceOp : IOperation
{
  explicit CaptureHistoricalSurfaceOp (
      std::vector<HistoricalSurfaceCapture> *captures)
      : captures (captures)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    HistoricalSurfaceCapture capture;
    const auto state = operations::historical_surface_search_cache_internal::
        Find (ctx.GetProcessorContext ());
    if (state)
      {
        for (const auto &entry : state->entries)
          {
            auto &memories = capture.first[entry.embedding_id];
            memories.push_back (entry.memory_id);
            for (const auto &reference : entry.memory_references)
              memories.push_back (reference.memory_id);
            std::sort (memories.begin (), memories.end ());
          }
        for (const auto &entry : state->current_entries)
          capture.second.push_back (entry.memory_id);
        std::sort (capture.second.begin (), capture.second.end ());
      }
    captures->push_back (std::move (capture));
  }

  std::vector<HistoricalSurfaceCapture> *captures = nullptr;
};
} // namespace

TEST_CASE ("SignalProcessor flush keeps the retrieval surface without "
           "reloading it",
           "[processor][flush][cache][retrieval_surface]")
{
  Eigen::VectorXf query = Eigen::VectorXf::Zero (256);
  query[0] = 1.0f;
  auto store = MakeReadOnlySurfaceStore (query);
  // A closed working-memory row with its own embedding, and an orphan
  // embedding: the forced flush prunes both from the store.
  Eigen::VectorXf working = Eigen::VectorXf::Zero (256);
  working[2] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (*store, 53LL, working, 1200LL);
  cortext::testing::SeedMemoryV2 (*store, 43LL, 53LL, "seed/working",
                                  "WORKING", 1.0, 1200LL);
  store->Execute ("UPDATE memories SET end_ts = 1300 WHERE memory_id = 43");
  Eigen::VectorXf orphan = Eigen::VectorXf::Zero (256);
  orphan[3] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (*store, 54LL, orphan, 1250LL);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  std::vector<RetrievalSurfaceCapture> surfaces;
  std::vector<HistoricalSurfaceCapture> historical;
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<CaptureRetrievalSurfaceOp> (&surfaces),
      std::make_unique<CaptureHistoricalSurfaceOp> (&historical));
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::
          MarkEngineOwnedJournalAware (std::move (pipeline)));
  processor.Process (MakeReadOnlyQuery (query, 2000));
  REQUIRE (historical.size () == 1);
  REQUIRE (historical[0].first.count (53LL) == 1);
  REQUIRE (historical[0].first.count (54LL) == 1);

  const auto reloads_before
      = cortext::testing::RetrievalSurfaceReloadCountForTest ();
  processor.Flush ();
  // Flush rewrites only working-memory rows, episode and state rows; it
  // must not pay a store-sized surface reload.
  REQUIRE (cortext::testing::RetrievalSurfaceReloadCountForTest ()
           == reloads_before);
  REQUIRE (store->Execute ("SELECT 1 FROM embeddings WHERE embedding_id IN "
                           "(53, 54)")
               .empty ());

  processor.Process (MakeReadOnlyQuery (query, 3000));
  REQUIRE (surfaces.size () == 2);
  REQUIRE (historical.size () == 2);
  // The pruned working row leaves the surface; nothing else changes.
  REQUIRE (surfaces[0].contains (43LL));
  REQUIRE_FALSE (surfaces[1].contains (43LL));
  auto expected_surface = surfaces[0];
  expected_surface.erase (43LL);
  REQUIRE (surfaces[1] == expected_surface);
  REQUIRE (historical[1].first.count (53LL) == 0);
  REQUIRE (historical[1].first.count (54LL) == 0);

  // The kept caches equal what a fresh processor loads from the store.
  std::vector<RetrievalSurfaceCapture> fresh_surfaces;
  std::vector<HistoricalSurfaceCapture> fresh_historical;
  SignalProcessor fresh (
      cfg, store,
      std::make_unique<DynamicOperationSet> (
          std::make_unique<CaptureRetrievalSurfaceOp> (&fresh_surfaces),
          std::make_unique<CaptureHistoricalSurfaceOp> (&fresh_historical)));
  fresh.Process (MakeReadOnlyQuery (query, 4000));
  REQUIRE (fresh_surfaces.size () == 1);
  REQUIRE (surfaces[1] == fresh_surfaces[0]);
  REQUIRE (historical[1] == fresh_historical[0]);
}

namespace
{
// Gives the processor `count` idle per-source accumulators once.
struct SeedIdleSourcesOp : IOperation
{
  explicit SeedIdleSourcesOp (int count) : count (count) {}

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    if (seeded)
      return;
    seeded = true;
    auto &p_ctx = ctx.GetProcessorContext ();
    for (int index = 0; index < count; ++index)
      {
        AccumulatorState state;
        state.n_signals = 1;
        state.mu_acc = Eigen::VectorXf::Ones (256);
        p_ctx.accumulator_states["idle/" + std::to_string (index)]
            = std::move (state);
      }
  }

  int count = 0;
  mutable bool seeded = false;
};

// On its second call, drops the only reference to an embedding the way an
// engine path does: the row goes, the surfaces forget it, and the embedding
// becomes an orphan candidate.
struct DropEmbeddingReferenceOp : IOperation
{
  DropEmbeddingReferenceOp (long long memory_id, long long embedding_id)
      : memory_id (memory_id), embedding_id (embedding_id)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction &tx) const override
  {
    if (++calls != 2)
      return;
    auto &p_ctx = ctx.GetProcessorContext ();
    tx.Execute ("DELETE FROM current_memory_embeddings WHERE memory_id = ?",
                { memory_id });
    tx.Execute ("DELETE FROM memories WHERE memory_id = ?", { memory_id });
    p_ctx.RemoveRetrievalSurface (memory_id);
    operations::historical_surface_search_cache_internal::RemoveCurrent (
        p_ctx, memory_id);
    p_ctx.orphan_embedding_candidates.push_back (embedding_id);
  }

  long long memory_id = 0;
  long long embedding_id = 0;
  mutable int calls = 0;
};

bool
EmbeddingExists (Store &store, long long embedding_id)
{
  return !store.Execute ("SELECT 1 FROM embeddings WHERE embedding_id = ?",
                         { embedding_id })
              .empty ();
}
} // namespace

TEST_CASE ("SignalProcessor flush work does not grow with the store or the "
           "source count",
           "[processor][flush][cache][scaling]")
{
  Eigen::VectorXf query = Eigen::VectorXf::Zero (256);
  query[0] = 1.0f;
  for (const int count : { 16, 1024 })
    {
      auto uniq = SQLiteStore::Create (":memory:");
      std::shared_ptr<Store> store (std::move (uniq));
      cortext::testing::InitializeCoreSchema (*store);
      for (int index = 0; index < count; ++index)
        {
          Eigen::VectorXf embedding = Eigen::VectorXf::Zero (256);
          embedding[index % 256] = 1.0f;
          embedding[(index / 256 + 1) % 256] += 0.5f;
          cortext::testing::SeedEmbeddingV2 (*store, 5000LL + index,
                                             embedding, 1000LL + index);
          cortext::testing::SeedMemoryV2 (*store, 1000LL + index,
                                          5000LL + index, "seed/ltm",
                                          "LONG_TERM", 1.0, 1000LL + index);
        }

      SignalProcessor::Config cfg;
      cortext::testing::RequireEncoder (cfg);
      std::vector<RetrievalSurfaceCapture> surfaces;
      auto pipeline = std::make_unique<DynamicOperationSet> (
          std::make_unique<SeedIdleSourcesOp> (count),
          std::make_unique<CaptureRetrievalSurfaceOp> (&surfaces));
      SignalProcessor processor (
          cfg, store,
          operations::signal_record_rollback_internal::
              MarkEngineOwnedJournalAware (std::move (pipeline)));
      processor.Process (MakeReadOnlyQuery (query, 2000));
      REQUIRE (surfaces.back ().size () == static_cast<std::size_t> (count));

      const auto copied_before
          = cortext::testing::FlushSnapshotCopiedEntryCountForTest ();
      const auto sweeps_before
          = cortext::testing::FullOrphanEmbeddingSweepCountForTest ();
      const auto reloads_before
          = cortext::testing::RetrievalSurfaceReloadCountForTest ();
      processor.Flush ();
      processor.Flush ();
      processor.Flush ();
      // The rollback snapshot copies neither the surface nor the sources,
      // only the first flush sweeps every embedding, and none reloads.
      REQUIRE (cortext::testing::FlushSnapshotCopiedEntryCountForTest ()
               == copied_before);
      REQUIRE (cortext::testing::FullOrphanEmbeddingSweepCountForTest ()
               == sweeps_before + 1);
      REQUIRE (cortext::testing::RetrievalSurfaceReloadCountForTest ()
               == reloads_before);

      processor.Process (MakeReadOnlyQuery (query, 3000));
      REQUIRE (surfaces.back ()
               == FreshRetrievalSurface (cfg, store,
                                         MakeReadOnlyQuery (query, 4000)));
    }
}

TEST_CASE ("SignalProcessor flush deletes tracked orphan embeddings and a "
           "failed flush reloads the surface",
           "[processor][flush][cache][rollback]")
{
  Eigen::VectorXf query = Eigen::VectorXf::Zero (256);
  query[0] = 1.0f;
  auto store = MakeReadOnlySurfaceStore (query);
  Eigen::VectorXf dropped = Eigen::VectorXf::Zero (256);
  dropped[4] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (*store, 55LL, dropped, 1150LL);
  cortext::testing::SeedMemoryV2 (*store, 44LL, 55LL, "seed/dropped",
                                  "LONG_TERM", 1.0, 1150LL);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  std::vector<RetrievalSurfaceCapture> surfaces;
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<SeedIdleSourcesOp> (8),
      std::make_unique<DropEmbeddingReferenceOp> (44LL, 55LL),
      std::make_unique<CaptureRetrievalSurfaceOp> (&surfaces));
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::
          MarkEngineOwnedJournalAware (std::move (pipeline)));
  processor.Process (MakeReadOnlyQuery (query, 2000));
  REQUIRE (surfaces.back ().contains (44LL));
  processor.Flush ();
  const auto sweeps_after_first
      = cortext::testing::FullOrphanEmbeddingSweepCountForTest ();

  Signal drop = MakeReadOnlyQuery (query, 3000);
  drop.retention = Retention::Natural;
  drop.source_id = "test";
  processor.Process (drop);
  REQUIRE (surfaces.back ().count (44LL) == 0);
  REQUIRE (EmbeddingExists (*store, 55LL));

  store->Execute ("CREATE TRIGGER fail_flush BEFORE INSERT ON episodes "
                  "BEGIN SELECT RAISE(ABORT, 'forced flush failure'); END");
  const auto reloads_before
      = cortext::testing::RetrievalSurfaceReloadCountForTest ();
  REQUIRE_THROWS (processor.Flush ());
  // The rolled-back flush kept the orphan and reloaded the surface from the
  // rolled-back store.
  REQUIRE (EmbeddingExists (*store, 55LL));
  REQUIRE (cortext::testing::RetrievalSurfaceReloadCountForTest ()
           == reloads_before + 1);
  store->Execute ("DROP TRIGGER fail_flush");

  processor.Flush ();
  REQUIRE_FALSE (EmbeddingExists (*store, 55LL));
  REQUIRE (cortext::testing::FullOrphanEmbeddingSweepCountForTest ()
           == sweeps_after_first);

  processor.Process (MakeReadOnlyQuery (query, 4000));
  REQUIRE (surfaces.back ()
           == FreshRetrievalSurface (cfg, store,
                                     MakeReadOnlyQuery (query, 5000)));
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
  auto processor = std::make_unique<SignalProcessor> (
      cfg, store, std::move (pipeline));
  REQUIRE (cache::RegistrySizeForTest () == baseline_size + 1);

  std::thread teardown (
      [owner = std::move (processor)] () mutable { owner.reset (); });
  teardown.join ();
  REQUIRE (cache::RegistrySizeForTest () == baseline_size);
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

TEST_CASE ("Engine consolidation seals an incrementally current sparse route "
           "without reloading the full retrieval surface",
           "[processor][cache][consolidation][hnsw][sqlite]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  cortext::testing::ScopedEnvVar hnsw_route_flag (
      "CORTEXT_HNSW_SPARSE_ROUTE", "1");
  cortext::testing::ScopedEnvVar sqlite_route_flag (
      "CORTEXT_SQLITE_SPARSE_ROUTE", "1");
  cortext::testing::ScopedEnvVar recenter_overlap_profile (
      "CORTEXT_PROFILE_RECENTER_ACTIVATION_OVERLAP", "1");
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  for (long long memory_id = 1; memory_id <= 513; ++memory_id)
    {
      std::vector<float> embedding (256, 0.0f);
      embedding[static_cast<std::size_t> ((memory_id % 255) + 1)] = 1.0f;
      cortext::testing::SeedEmbeddingV2 (
          *store, memory_id, embedding, 1000 + memory_id);
      cortext::testing::SeedMemoryV2 (
          *store, memory_id, memory_id,
          "seed/" + std::to_string (memory_id), "LONG_TERM", 1.0,
          1000 + memory_id);
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto operation = std::make_unique<ConsolidationSurfaceUpdateOp> ();
  auto *capturing_operation = operation.get ();
  auto root = std::make_unique<DynamicOperationSet> (std::move (operation));
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::
          MarkEngineOwnedJournalAware (std::move (root)));

  std::vector<float> active_signal_embedding (256, 0.0f);
  active_signal_embedding[3] = 1.0f;
  cortext::testing::SeedSignalV2 (
      *store, 7001, 3, "opaque/source", 99999);
  {
    auto tx = store->Begin ();
    operations::active_signal_embedding_ring_internal::Upsert (
        *tx, 7001, active_signal_embedding, 99999, cfg.focus,
        cfg.sensitivity, cfg.stability);
    tx->Commit ();
  }

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.embedding[0] = 1.0f;
  signal.source_id = "maintenance";
  signal.timestamp = 100000;
  signal.force_consolidation = true;
  const auto output = processor.Process (signal);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.reload_retrieval_surface_skipped")
           == 1.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.reload_retrieval_surface")
           == 0.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sparse_route_seal_delta_count")
           == 0.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_seal_delta_count")
           == 0.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sparse_route_seal_succeeded")
           == 1.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_seal_succeeded")
           == 1.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_recenter_overlap_profiled")
           == 1.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_recenter_overlap_pair_valid")
           == 1.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_recenter_overlap_failure_code")
           == 0.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_recenter_centroid_source")
           == 2.0);
  const double pre_activated = output.operation_ms.at (
      "SignalProcessor.sqlite_sparse_route_recenter_pre_activated_count");
  REQUIRE (pre_activated > 0.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_recenter_post_activated_count")
           == pre_activated);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_recenter_overlap_count")
           == pre_activated);
  REQUIRE (capturing_operation->captured != nullptr);
  auto state = cache::FindMutable (*capturing_operation->captured);
  REQUIRE (state);
  REQUIRE (state->sparse_route);
  REQUIRE (state->sqlite_sparse_route);
  REQUIRE (state->sqlite_sparse_route->ActivationEntryMemoryId () == 2);

  auto restarted = operations::sparse_retrieval_route_sqlite_internal::Route::
      Open (*store, 256,
            operations::sparse_retrieval_route_sqlite_internal::
                DefaultParameters ());
  REQUIRE (restarted);
  REQUIRE (restarted->RestartRowsLoaded () == 1);
  // The sealed generation persisted the moved vector (memory 1, now on
  // dimension 0) in place.
  {
    const auto rows = store->Execute (
        "SELECT embedding FROM cortext_sparse_route_nodes "
        "WHERE memory_id = 1 AND active = 1");
    REQUIRE (rows.size () == 1);
    const auto bytes = store::BlobFromAny (rows[0].at ("embedding"));
    REQUIRE (bytes.size () == 256 * sizeof (float));
    std::vector<float> persisted (256);
    std::memcpy (persisted.data (), bytes.data (), bytes.size ());
    REQUIRE (persisted[0] == Catch::Approx (1.0f));
  }

  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.timestamp = 100001;
  const auto invalid_pair_output = processor.Process (signal);
  REQUIRE (invalid_pair_output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_recenter_overlap_profiled")
           == 1.0);
  REQUIRE (invalid_pair_output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_recenter_overlap_pair_valid")
           == 0.0);
  REQUIRE (invalid_pair_output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_recenter_overlap_failure_code")
           == 2.0);
  REQUIRE (invalid_pair_output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_recenter_succeeded")
           == 1.0);
  REQUIRE (invalid_pair_output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_recenter_centroid_source")
           == 2.0);
  REQUIRE (invalid_pair_output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_recenter_centroid_member_count")
           == 1.0);
  REQUIRE (state->sqlite_sparse_route->ActivationEntryMemoryId () == 2);
}

TEST_CASE ("Invalidated consolidation surface reloads authority and replaces "
           "the prior sparse route generation",
           "[processor][cache][consolidation][hnsw][sqlite][fallback]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  cortext::testing::ScopedEnvVar hnsw_route_flag (
      "CORTEXT_HNSW_SPARSE_ROUTE", "1");
  cortext::testing::ScopedEnvVar sqlite_route_flag (
      "CORTEXT_SQLITE_SPARSE_ROUTE", "1");
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  for (long long memory_id = 1; memory_id <= 513; ++memory_id)
    {
      std::vector<float> embedding (256, 0.0f);
      embedding[static_cast<std::size_t> ((memory_id % 255) + 1)] = 1.0f;
      cortext::testing::SeedEmbeddingV2 (
          *store, memory_id, embedding, 1000 + memory_id);
      cortext::testing::SeedMemoryV2 (
          *store, memory_id, memory_id,
          "seed/" + std::to_string (memory_id), "LONG_TERM", 1.0,
          1000 + memory_id);
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto operation = std::make_unique<ConsolidationSurfaceUpdateOp> (true);
  auto *capturing_operation = operation.get ();
  auto root = std::make_unique<DynamicOperationSet> (std::move (operation));
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::
          MarkEngineOwnedJournalAware (std::move (root)));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "maintenance";
  signal.timestamp = 100000;
  signal.force_consolidation = true;
  const auto output = processor.Process (signal);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.reload_retrieval_surface_skipped")
           == 0.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.reload_retrieval_surface")
           >= 0.0);
  REQUIRE (capturing_operation->captured != nullptr);
  auto state = cache::Find (*capturing_operation->captured);
  REQUIRE (state);
  REQUIRE (state->sparse_route);
  REQUIRE (state->sqlite_sparse_route);
  auto restarted = operations::sparse_retrieval_route_sqlite_internal::Route::
      Open (*store, 256,
            operations::sparse_retrieval_route_sqlite_internal::
                DefaultParameters ());
  REQUIRE (restarted);
  Eigen::VectorXf query = Eigen::VectorXf::Zero (256);
  query[0] = 1.0f;
  const auto result = restarted->Search (query, 512);
  REQUIRE (result);
  REQUIRE_FALSE (result->empty ());
  REQUIRE (result->front () == 1);
}

TEST_CASE ("Restarted SQLite sparse route relinks moved active topology "
           "without a rebuild",
           "[processor][cache][consolidation][hnsw][sqlite][restart]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  cortext::testing::ScopedEnvVar hnsw_route_flag (
      "CORTEXT_HNSW_SPARSE_ROUTE", "1");
  cortext::testing::ScopedEnvVar sqlite_route_flag (
      "CORTEXT_SQLITE_SPARSE_ROUTE", "1");
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  std::vector<std::pair<long long, Eigen::VectorXf>> route_entries;
  route_entries.reserve (513);
  for (long long memory_id = 1; memory_id <= 513; ++memory_id)
    {
      Eigen::VectorXf vector = Eigen::VectorXf::Zero (256);
      vector[static_cast<Eigen::Index> ((memory_id % 255) + 1)] = 1.0f;
      std::vector<float> embedding (vector.data (),
                                    vector.data () + vector.size ());
      cortext::testing::SeedEmbeddingV2 (
          *store, memory_id, embedding, 1000 + memory_id);
      cortext::testing::SeedMemoryV2 (
          *store, memory_id, memory_id,
          "seed/" + std::to_string (memory_id), "LONG_TERM", 1.0,
          1000 + memory_id);
      route_entries.emplace_back (memory_id, std::move (vector));
    }
  auto hnsw = operations::sparse_retrieval_route_internal::Route::Create (
      256, route_entries,
      operations::sparse_retrieval_route_internal::DefaultParameters ());
  REQUIRE (hnsw);
  auto persisted
      = operations::sparse_retrieval_route_sqlite_internal::Route::Create (
          *store, 256, route_entries, *hnsw,
          operations::sparse_retrieval_route_sqlite_internal::
              DefaultParameters ());
  REQUIRE (persisted);
  hnsw.reset ();
  persisted.reset ();

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto operation = std::make_unique<ConsolidationSurfaceUpdateOp> ();
  auto *capturing_operation = operation.get ();
  auto root = std::make_unique<DynamicOperationSet> (std::move (operation));
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::
          MarkEngineOwnedJournalAware (std::move (root)));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "maintenance";
  signal.timestamp = 100000;
  signal.force_consolidation = true;
  const auto output = processor.Process (signal);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.reload_retrieval_surface_skipped")
           == 1.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_seal_delta_count")
           == 1.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_seal_succeeded")
           == 1.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_seal_failure_code")
           == 0.0);
  REQUIRE (capturing_operation->captured != nullptr);
  const auto state = cache::Find (*capturing_operation->captured);
  REQUIRE (state);
  REQUIRE (state->sqlite_sparse_route);
  REQUIRE_FALSE (state->sqlite_sparse_route_backfill);

  auto restarted = operations::sparse_retrieval_route_sqlite_internal::Route::
      Open (*store, 256,
            operations::sparse_retrieval_route_sqlite_internal::
                DefaultParameters ());
  REQUIRE (restarted);
  REQUIRE (restarted->ActiveCount () == route_entries.size ());
  REQUIRE (store->Execute (
               "SELECT memory_id FROM cortext_sparse_route_dirty")
               .empty ());
  Eigen::VectorXf moved_query = Eigen::VectorXf::Zero (256);
  moved_query[0] = 1.0f;
  const auto moved_result = restarted->Search (moved_query, 512);
  REQUIRE (moved_result);
  REQUIRE_FALSE (moved_result->empty ());
  REQUIRE (moved_result->front () == 1);
}

TEST_CASE ("Engine consolidation relinks reconsolidated route embeddings "
           "and recenters instead of rebuilding",
           "[processor][cache][consolidation][hnsw][sqlite][regression]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  namespace sqlite_route
      = operations::sparse_retrieval_route_sqlite_internal;
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  cortext::testing::ScopedEnvVar hnsw_route_flag (
      "CORTEXT_HNSW_SPARSE_ROUTE", "1");
  cortext::testing::ScopedEnvVar sqlite_route_flag (
      "CORTEXT_SQLITE_SPARSE_ROUTE", "1");
  const auto parameters = sqlite_route::DefaultParameters ();
  const long long store_size
      = static_cast<long long> (parameters.route_capacity) + 88;
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  std::vector<std::pair<long long, Eigen::VectorXf>> route_entries;
  route_entries.reserve (static_cast<std::size_t> (store_size));
  for (long long memory_id = 1; memory_id <= store_size; ++memory_id)
    {
      Eigen::VectorXf vector = ReconsolidationMoveOp::Original (memory_id);
      std::vector<float> embedding (vector.data (),
                                    vector.data () + vector.size ());
      cortext::testing::SeedEmbeddingV2 (
          *store, memory_id, embedding, 1000 + memory_id);
      cortext::testing::SeedMemoryV2 (
          *store, memory_id, memory_id,
          "seed/" + std::to_string (memory_id), "LONG_TERM", 1.0,
          1000 + memory_id);
      route_entries.emplace_back (memory_id, std::move (vector));
    }
  {
    auto hnsw = operations::sparse_retrieval_route_internal::Route::Create (
        256, route_entries, parameters.hnsw);
    REQUIRE (hnsw);
    REQUIRE (sqlite_route::Route::Create (*store, 256, route_entries, *hnsw,
                                          parameters));
  }

  const std::vector<long long> moved_ids{ 3, 40, 97, 150, 201, 254 };
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  // The route recenters on the active-signal ring centroid; make the recent
  // signal the first moved vector so the activation entry follows it.
  {
    const Eigen::VectorXf moved = ReconsolidationMoveOp::Moved (
        moved_ids.front ());
    cortext::testing::SeedSignalV2 (
        *store, 9001, moved_ids.front (), "opaque/source", 99999);
    auto tx = store->Begin ();
    operations::active_signal_embedding_ring_internal::Upsert (
        *tx, 9001,
        std::vector<float> (moved.data (), moved.data () + moved.size ()),
        99999, cfg.focus, cfg.sensitivity, cfg.stability);
    tx->Commit ();
  }
  auto operation = std::make_unique<ReconsolidationMoveOp> (moved_ids);
  auto *capturing_operation = operation.get ();
  auto root = std::make_unique<DynamicOperationSet> (std::move (operation));
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::
          MarkEngineOwnedJournalAware (std::move (root)));

  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.embedding[0] = 1.0f;
  signal.source_id = "maintenance";
  signal.timestamp = 100000;
  signal.force_consolidation = true;
  const auto output = processor.Process (signal);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_seal_delta_count")
           == static_cast<double> (moved_ids.size ()));
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_seal_failure_code")
           == 0.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_seal_succeeded")
           == 1.0);
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_recenter_succeeded")
           == 1.0);
  REQUIRE (capturing_operation->captured != nullptr);
  const auto state = cache::Find (*capturing_operation->captured);
  REQUIRE (state);
  REQUIRE (state->sqlite_sparse_route);
  REQUIRE_FALSE (state->sqlite_sparse_route_backfill);
  REQUIRE_FALSE (state->sqlite_sparse_route->HasDirtyMemoryIds ());
  REQUIRE (state->sqlite_sparse_route->ActivationEntryMemoryId ()
           == moved_ids.front ());

  auto restarted = sqlite_route::Route::Open (*store, 256, parameters);
  REQUIRE (restarted);
  REQUIRE (restarted->ActiveCount ()
           == static_cast<std::size_t> (store_size));
  REQUIRE (restarted->ActivationEntryMemoryId () == moved_ids.front ());
  // The sealed generation persists each moved vector in place rather than
  // being deleted for a backfill rebuild.
  for (const long long memory_id : moved_ids)
    {
      const auto rows = store->Execute (
          "SELECT embedding, active FROM cortext_sparse_route_nodes "
          "WHERE memory_id = ?",
          { memory_id });
      REQUIRE (rows.size () == 1);
      REQUIRE (std::any_cast<long long> (rows[0].at ("active")) == 1);
      const auto bytes = store::BlobFromAny (rows[0].at ("embedding"));
      REQUIRE (bytes.size () == 256 * sizeof (float));
      Eigen::VectorXf persisted (256);
      std::memcpy (persisted.data (), bytes.data (), bytes.size ());
      CHECK ((persisted - ReconsolidationMoveOp::Moved (memory_id)).norm ()
             < 1e-5f);
    }
}

TEST_CASE ("Ordinary sparse-route maintenance seals at its knob-derived "
           "backfill-minus-expansion batch",
           "[processor][cache][hnsw][sqlite][knobs][regression]")
{
  cortext::testing::ScopedEnvVar disable_constructive_recall (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL", "1");
  cortext::testing::ScopedEnvVar sqlite_route_flag (
      "CORTEXT_SQLITE_SPARSE_ROUTE", "1");
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  const auto parameters
      = operations::sparse_retrieval_route_sqlite_internal::
          DefaultParameters ();
  REQUIRE (parameters.backfill_batch_size == 128);
  REQUIRE (parameters.search_expansion_batch == 32);
  const std::size_t ordinary_seal_batch
      = static_cast<std::size_t> (
          operations::sparse_retrieval_knobs_internal::OrdinarySealBatch (
              0.5, 0.5, 0.5));
  REQUIRE (ordinary_seal_batch == 96);

  std::vector<std::pair<long long, Eigen::VectorXf>> route_entries;
  route_entries.reserve (513);
  for (long long memory_id = 1; memory_id <= 513; ++memory_id)
    {
      Eigen::VectorXf vector = Eigen::VectorXf::Zero (256);
      vector[static_cast<Eigen::Index> ((memory_id % 255) + 1)] = 1.0f;
      std::vector<float> embedding (vector.data (),
                                    vector.data () + vector.size ());
      cortext::testing::SeedEmbeddingV2 (
          *store, memory_id, embedding, 1000 + memory_id);
      cortext::testing::SeedMemoryV2 (
          *store, memory_id, memory_id,
          "seed/" + std::to_string (memory_id), "LONG_TERM", 1.0,
          1000 + memory_id);
      route_entries.emplace_back (memory_id, std::move (vector));
    }
  auto hnsw = operations::sparse_retrieval_route_internal::Route::Create (
      256, route_entries,
      operations::sparse_retrieval_route_internal::DefaultParameters ());
  REQUIRE (hnsw);
  auto persisted
      = operations::sparse_retrieval_route_sqlite_internal::Route::Create (
          *store, 256, route_entries, *hnsw, parameters);
  REQUIRE (persisted);
  for (std::size_t index = 0; index < ordinary_seal_batch;
       ++index)
    {
      Eigen::VectorXf revised = Eigen::VectorXf::Zero (256);
      revised[static_cast<Eigen::Index> ((index + 17) % 255)] = 1.0f;
      REQUIRE (persisted->Upsert (
          static_cast<long long> (index + 1), revised));
    }
  REQUIRE (persisted->HasDirtyMemoryIds ());
  persisted.reset ();
  hnsw.reset ();

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  auto root = std::make_unique<DynamicOperationSet> (
      std::make_unique<BoundedShadowNoOp> ());
  SignalProcessor processor (
      cfg, store,
      operations::signal_record_rollback_internal::
          MarkEngineOwnedJournalAware (std::move (root)));
  Signal signal;
  signal.embedding = Eigen::VectorXf::Zero (256);
  signal.source_id = "opaque/ordinary";
  signal.modality = "audio";
  signal.timestamp = 100000;
  const auto output = processor.Process (signal);

  REQUIRE (store->Execute (
               "SELECT memory_id FROM cortext_sparse_route_dirty")
               .empty ());
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_ordinary_seal_delta_count")
           == static_cast<double> (ordinary_seal_batch));
  REQUIRE (output.operation_ms.at (
               "SignalProcessor.sqlite_sparse_route_ordinary_seal_succeeded")
           == 1.0);
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

namespace
{
// Leaves per-source state behind the way accumulation, drift, and pacing do:
// every signal touches its source's AccumulatorState; "open/" sources keep an
// open unit.
struct SourceStateProbeOp : IOperation
{
  ProcessorContext **context_address = nullptr;

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    *context_address = &pctx;
    const auto &signal = ctx.GetSignal ();
    auto &acc = pctx.accumulator_states[signal.source_id];
    acc.drift_accum += 1.0;
    acc.prev_x = signal.embedding;
    if (signal.source_id.rfind ("open/", 0) == 0)
      {
        acc.n_signals += 1;
      }
  }
};
} // namespace

TEST_CASE ("SignalProcessor evicts idle per-source state beyond its budget",
           "[processor][accumulator][memory]")
{
  namespace source_state = cortext::internal::source_state;
  auto store = std::shared_ptr<Store> (SQLiteStore::Create (":memory:"));
  cortext::testing::InitializeCoreSchema (*store);
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);

  ProcessorContext *pctx = nullptr;
  auto probe = std::make_unique<SourceStateProbeOp> ();
  probe->context_address = &pctx;
  SignalProcessor processor (
      cfg, store, std::make_unique<DynamicOperationSet> (std::move (probe)));

  std::uint64_t ts = 1;
  auto process = [&] (const std::string &source_id) {
    Signal signal;
    signal.embedding = Eigen::VectorXf::Constant (256, 0.5f);
    signal.source_id = source_id;
    signal.timestamp = ts++;
    signal.retention = Retention::Durable;
    processor.Process (signal);
  };

  // One open unit, one recurring conversation, and a fresh id per turn (the
  // shape an eval or a per-call host produces).
  process ("open/unit");
  const std::size_t turns = 3 * source_state::kIdleSourceCapacity;
  int active_signals = 0;
  for (std::size_t i = 0; i < turns; ++i)
    {
      process ("turn/" + std::to_string (i));
      if (i % 8 == 0)
        {
          process ("active/conversation");
          ++active_signals;
        }
    }

  REQUIRE (pctx != nullptr);
  const auto &states = pctx->accumulator_states;
  std::size_t idle = 0;
  for (const auto &[id, state] : states)
    {
      if (source_state::IsIdle (state))
        {
          ++idle;
        }
    }
  CHECK (idle <= source_state::kIdleSourceCapacity);
  CHECK (states.size () <= source_state::kIdleSourceCapacity + 1);

  // Open units are never dropped.
  REQUIRE (states.contains ("open/unit"));
  CHECK (states.at ("open/unit").n_signals == 1);
  // A recently active source keeps its state across the churn.
  REQUIRE (states.contains ("active/conversation"));
  CHECK (states.at ("active/conversation").drift_accum
         == static_cast<double> (active_signals));
  // The least recent one-off sources are gone; the newest remain.
  CHECK_FALSE (states.contains ("turn/0"));
  CHECK (states.contains ("turn/" + std::to_string (turns - 1)));
}
