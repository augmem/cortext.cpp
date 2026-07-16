#include <any>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include "../src/operations/historical_surface_search_cache_internal.hpp"
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
        processor_context.index_store["stable"] = { 99LL };
        processor_context.index_reverse[11LL] = "mutated";
        processor_context.procedural_store["stable"][11LL] = 0.0;
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
  proc.Process (signal);

  signal.timestamp = 3000;
  REQUIRE_THROWS (proc.Process (signal));

  signal.timestamp = 4000;
  proc.Process (signal);
  REQUIRE (retrieval_restored);
  REQUIRE (volatile_restored);
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

TEST_CASE ("SignalProcessor erases historical cache when destroyed on another "
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
