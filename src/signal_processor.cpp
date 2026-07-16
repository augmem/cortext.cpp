#include "cortext/core/knobs.hpp"
#include "cortext/core/constants.hpp"
#include "cortext/clock.hpp"
#include "cortext/internal/cancellation.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/sqlite_store.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "cortext/store/schema.hpp"
#include "cortext/store/utils.hpp"
#include "operations/constructive_recall_internal.hpp"
#include "operations/consolidation_throughput_state_internal.hpp"
#include "operations/historical_surface_search_cache_internal.hpp"
#include "working_memory_time_internal.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <any>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cortext
{

namespace
{

constexpr int kClosedWorkingMemoryPruneBatch = 1024;
constexpr int kClosedWorkingMemoryIncrementalPruneBatch = 8;
constexpr int kClosedWorkingMemoryIncrementalPruneIntervalSignals = 8;
constexpr uint64_t kWalPassiveCheckpointIntervalSignals = 256;
constexpr std::uintmax_t kWalPassiveCheckpointMinBytes
    = 256ULL * 1024ULL * 1024ULL;

double
ElapsedMillis (std::chrono::steady_clock::time_point start,
               std::chrono::steady_clock::time_point end)
{
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli> > (
             end - start)
      .count ();
}

inline int64_t
ExtractInt64 (const std::map<std::string, std::any> &row,
              const std::string &key, int64_t default_val);

std::string
Placeholders (std::size_t count)
{
  std::string placeholders;
  placeholders.reserve (count * 3);
  for (std::size_t i = 0; i < count; ++i)
    {
      if (i > 0)
        {
          placeholders += ", ";
        }
      placeholders += "?";
    }
  return placeholders;
}

std::vector<std::any>
MakeParams (const std::vector<long long> &values)
{
  std::vector<std::any> params;
  params.reserve (values.size ());
  for (const auto value : values)
    {
      params.push_back (value);
    }
  return params;
}

std::vector<long long>
ExtractInt64Column (const std::vector<std::map<std::string, std::any>> &rows,
                    const std::string &column)
{
  std::vector<long long> values;
  values.reserve (rows.size ());
  std::unordered_set<long long> seen;
  seen.reserve (rows.size ());
  for (const auto &row : rows)
    {
      const auto value = ExtractInt64 (row, column, 0);
      if (value > 0 && seen.insert (value).second)
        {
          values.push_back (value);
        }
    }
  return values;
}

struct DetachedProcessorCaches
{
  std::vector<ProcessorContext::AssociationCacheEntry> association_cache;
  std::unordered_map<long long, size_t> association_cache_index;
  std::vector<ProcessorContext::RetrievalSurfaceEntry>
      retrieval_surface_cache;
  std::unordered_map<long long, size_t> retrieval_surface_index;
  std::unordered_map<long long, size_t> retrieval_surface_embedding_index;
  std::unordered_map<
      std::string,
      std::vector<ProcessorContext::RetrievalSurfaceSourceIndexEntry>>
      retrieval_surface_source_index;
  std::unordered_set<std::string> retrieval_surface_source_index_dirty;
  ProcessorContext::AssociationFanoutCache association_fanout_cache;
  std::unordered_set<long long> predictive_pre_activation_embedding_ids;
  std::unordered_set<long long> predictive_pre_activation_memory_ids;
  std::unordered_set<long long> retrieval_suppression_embedding_ids;
  std::unordered_set<long long> retrieval_suppression_memory_ids;
  std::unordered_map<std::string, std::vector<long long>> index_store;
  std::unordered_map<long long, std::string> index_reverse;
  std::unordered_map<std::string, std::unordered_map<long long, double>>
      procedural_store;
};

void
ClearRebuildableProcessorCaches (ProcessorContext &ctx)
{
  ctx.association_cache = {};
  ctx.association_cache_index = {};
  ctx.retrieval_surface_cache = {};
  ctx.retrieval_surface_index = {};
  ctx.retrieval_surface_embedding_index = {};
  ctx.retrieval_surface_source_index = {};
  ctx.retrieval_surface_source_index_dirty = {};
  ctx.association_fanout_cache = {};
  ctx.predictive_pre_activation_embedding_ids = {};
  ctx.predictive_pre_activation_memory_ids = {};
  ctx.retrieval_suppression_embedding_ids = {};
  ctx.retrieval_suppression_memory_ids = {};
  ctx.index_store = {};
  ctx.index_reverse = {};
  ctx.procedural_store = {};
}

DetachedProcessorCaches
DetachRebuildableProcessorCaches (ProcessorContext &ctx)
{
  DetachedProcessorCaches caches;
  caches.association_cache = std::move (ctx.association_cache);
  caches.association_cache_index = std::move (ctx.association_cache_index);
  caches.retrieval_surface_cache = std::move (ctx.retrieval_surface_cache);
  caches.retrieval_surface_index = std::move (ctx.retrieval_surface_index);
  caches.retrieval_surface_embedding_index
      = std::move (ctx.retrieval_surface_embedding_index);
  caches.retrieval_surface_source_index
      = std::move (ctx.retrieval_surface_source_index);
  caches.retrieval_surface_source_index_dirty
      = std::move (ctx.retrieval_surface_source_index_dirty);
  caches.association_fanout_cache
      = std::move (ctx.association_fanout_cache);
  caches.predictive_pre_activation_embedding_ids
      = std::move (ctx.predictive_pre_activation_embedding_ids);
  caches.predictive_pre_activation_memory_ids
      = std::move (ctx.predictive_pre_activation_memory_ids);
  caches.retrieval_suppression_embedding_ids
      = std::move (ctx.retrieval_suppression_embedding_ids);
  caches.retrieval_suppression_memory_ids
      = std::move (ctx.retrieval_suppression_memory_ids);
  caches.index_store = std::move (ctx.index_store);
  caches.index_reverse = std::move (ctx.index_reverse);
  caches.procedural_store = std::move (ctx.procedural_store);
  ClearRebuildableProcessorCaches (ctx);
  return caches;
}

void
RestoreRebuildableProcessorCaches (ProcessorContext &ctx,
                                   DetachedProcessorCaches &caches)
{
  ctx.association_cache = std::move (caches.association_cache);
  ctx.association_cache_index = std::move (caches.association_cache_index);
  ctx.retrieval_surface_cache = std::move (caches.retrieval_surface_cache);
  ctx.retrieval_surface_index = std::move (caches.retrieval_surface_index);
  ctx.retrieval_surface_embedding_index
      = std::move (caches.retrieval_surface_embedding_index);
  ctx.retrieval_surface_source_index
      = std::move (caches.retrieval_surface_source_index);
  ctx.retrieval_surface_source_index_dirty
      = std::move (caches.retrieval_surface_source_index_dirty);
  ctx.association_fanout_cache = std::move (caches.association_fanout_cache);
  ctx.predictive_pre_activation_embedding_ids
      = std::move (caches.predictive_pre_activation_embedding_ids);
  ctx.predictive_pre_activation_memory_ids
      = std::move (caches.predictive_pre_activation_memory_ids);
  ctx.retrieval_suppression_embedding_ids
      = std::move (caches.retrieval_suppression_embedding_ids);
  ctx.retrieval_suppression_memory_ids
      = std::move (caches.retrieval_suppression_memory_ids);
  ctx.index_store = std::move (caches.index_store);
  ctx.index_reverse = std::move (caches.index_reverse);
  ctx.procedural_store = std::move (caches.procedural_store);
}

class ScopedProcessorCacheDetach
{
public:
  explicit ScopedProcessorCacheDetach (ProcessorContext &ctx)
      : ctx_ (&ctx), caches_ (DetachRebuildableProcessorCaches (ctx))
  {
  }

  ~ScopedProcessorCacheDetach ()
  {
    if (ctx_)
      {
        RestoreRebuildableProcessorCaches (*ctx_, caches_);
      }
  }

  ScopedProcessorCacheDetach (const ScopedProcessorCacheDetach &) = delete;
  ScopedProcessorCacheDetach &
  operator= (const ScopedProcessorCacheDetach &) = delete;

private:
  ProcessorContext *ctx_ = nullptr;
  DetachedProcessorCaches caches_;
};

struct ProcessorRollbackSnapshot
{
  ProcessorContext context;
  DetachedProcessorCaches caches;
  operations::consolidation_throughput_state_internal::State
      consolidation_throughput;
  bool initialized = false;
};

#if defined(CORTEXT_TESTING)
std::atomic_size_t g_processor_rollback_snapshot_owner_count { 0 };
std::atomic_size_t g_processor_rollback_snapshot_reuse_count { 0 };
#endif

/// Owns the caller-provided operation and the rollback storage for its
/// SignalProcessor. A single handle already requires external synchronization;
/// depth-indexed slots additionally preserve the prior stack-local behavior for
/// synchronous nested Process calls without a global or thread-local registry.
class SnapshotOwningOperation final : public IOperation
{
public:
  class Lease
  {
  public:
    Lease (SnapshotOwningOperation &owner, ProcessorRollbackSnapshot &snapshot)
        : owner_ (&owner), snapshot_ (&snapshot)
    {
    }

    ~Lease ()
    {
      if (owner_)
        {
          owner_->Release ();
        }
    }

    Lease (const Lease &) = delete;
    Lease &operator= (const Lease &) = delete;

    Lease (Lease &&other) noexcept
        : owner_ (std::exchange (other.owner_, nullptr)),
          snapshot_ (std::exchange (other.snapshot_, nullptr))
    {
    }

    Lease &
    operator= (Lease &&) = delete;

    ProcessorRollbackSnapshot &
    Get () const
    {
      return *snapshot_;
    }

  private:
    SnapshotOwningOperation *owner_ = nullptr;
    ProcessorRollbackSnapshot *snapshot_ = nullptr;
  };

  explicit SnapshotOwningOperation (std::unique_ptr<IOperation> operation)
      : operation_ (std::move (operation))
  {
#if defined(CORTEXT_TESTING)
    g_processor_rollback_snapshot_owner_count.fetch_add (
        1, std::memory_order_relaxed);
#endif
  }

  ~SnapshotOwningOperation () override
  {
#if defined(CORTEXT_TESTING)
    g_processor_rollback_snapshot_owner_count.fetch_sub (
        1, std::memory_order_relaxed);
#endif
  }

  Lease
  Acquire ()
  {
    if (active_depth_ == snapshots_.size ())
      {
        snapshots_.push_back (std::make_unique<ProcessorRollbackSnapshot> ());
      }
    auto &snapshot = *snapshots_[active_depth_];
    ++active_depth_;
#if defined(CORTEXT_TESTING)
    if (snapshot.initialized)
      {
        g_processor_rollback_snapshot_reuse_count.fetch_add (
            1, std::memory_order_relaxed);
      }
#endif
    return Lease (*this, snapshot);
  }

  void
  Execute (OperationContext &ctx, Transaction &tx) const override
  {
    operation_->Execute (ctx, tx);
  }

private:
  void
  Release ()
  {
    --active_depth_;
  }

  std::unique_ptr<IOperation> operation_;
  std::vector<std::unique_ptr<ProcessorRollbackSnapshot>> snapshots_;
  std::size_t active_depth_ = 0;
};

void
DeleteStaleWorkingMemoryRows (Transaction &tx,
                              const std::vector<long long> &memory_ids,
                              bool delete_payloads)
{
  if (memory_ids.empty ())
    {
      return;
    }

  const std::string placeholders = Placeholders (memory_ids.size ());
  const auto memory_params = MakeParams (memory_ids);
  std::vector<long long> embedding_ids;
  if (delete_payloads)
    {
      embedding_ids = ExtractInt64Column (
          tx.Execute ("SELECT embedding_id FROM memories "
                      "WHERE memory_id IN (" + placeholders + ") "
                      "  AND kind = 'WORKING' "
                      "  AND embedding_id IS NOT NULL",
                      memory_params),
          "embedding_id");
      auto signal_embedding_ids = ExtractInt64Column (
          tx.Execute ("SELECT embedding_id FROM signals "
                      "WHERE memory_id IN (" + placeholders + ") "
                      "  AND embedding_id IS NOT NULL",
                      memory_params),
          "embedding_id");
      embedding_ids.insert (embedding_ids.end (), signal_embedding_ids.begin (),
                            signal_embedding_ids.end ());
      std::sort (embedding_ids.begin (), embedding_ids.end ());
      embedding_ids.erase (
          std::unique (embedding_ids.begin (), embedding_ids.end ()),
          embedding_ids.end ());
    }

  if (delete_payloads)
    {
      std::vector<std::any> assoc_params;
      assoc_params.reserve (memory_ids.size () * 2);
      for (const auto id : memory_ids)
        {
          assoc_params.push_back (id);
        }
      for (const auto id : memory_ids)
        {
          assoc_params.push_back (id);
        }
      tx.Execute ("DELETE FROM associations "
                  "WHERE source_memory_id IN (" + placeholders + ") "
                  "   OR target_memory_id IN (" + placeholders + ")",
                  assoc_params);
    }
  tx.Execute ("DELETE FROM signals WHERE memory_id IN (" + placeholders + ")",
              memory_params);
  if (delete_payloads)
    {
      tx.Execute ("DELETE FROM current_memory_embeddings "
                  "WHERE memory_id IN (" + placeholders + ")",
                  memory_params);
    }
  tx.Execute ("DELETE FROM memories "
              "WHERE kind = 'WORKING' AND memory_id IN (" + placeholders + ")",
              memory_params);

  if (!embedding_ids.empty ())
    {
      const std::string embedding_placeholders
          = Placeholders (embedding_ids.size ());
      tx.Execute (
          "DELETE FROM embeddings "
          "WHERE embedding_id IN (" + embedding_placeholders + ") "
          "  AND NOT EXISTS ("
          "    SELECT 1 FROM memories m "
          "    WHERE m.embedding_id = embeddings.embedding_id"
          "  ) "
          "  AND NOT EXISTS ("
          "    SELECT 1 FROM signals s "
          "    WHERE s.embedding_id = embeddings.embedding_id"
          "  ) "
          "  AND NOT EXISTS ("
          "    SELECT 1 FROM memory_reconstructions mr "
          "    WHERE mr.embedding_id = embeddings.embedding_id"
          "  )",
          MakeParams (embedding_ids));
    }
}

void
DeleteUnreferencedEmbeddings (Transaction &tx)
{
  tx.Execute (
      "DELETE FROM embeddings "
      "WHERE NOT EXISTS ("
      "    SELECT 1 FROM memories m "
      "    WHERE m.embedding_id = embeddings.embedding_id"
      "  ) "
      "  AND NOT EXISTS ("
      "    SELECT 1 FROM signals s "
      "    WHERE s.embedding_id = embeddings.embedding_id"
      "  ) "
      "  AND NOT EXISTS ("
      "    SELECT 1 FROM memory_reconstructions mr "
      "    WHERE mr.embedding_id = embeddings.embedding_id"
      "  )",
      {});
}

long long
ResolveMemoryIdForRetrievedEmbedding (const OperationContext &op_context,
                                      long long embedding_id)
{
  if (embedding_id <= 0)
    {
      return 0;
    }
  const auto &p_ctx = op_context.GetProcessorContext ();
  auto cache_it = p_ctx.retrieval_surface_embedding_index.find (embedding_id);
  if (cache_it != p_ctx.retrieval_surface_embedding_index.end ())
    {
      const long long memory_id
          = p_ctx.retrieval_surface_cache[cache_it->second].memory_id;
      if (memory_id > 0)
        {
          return memory_id;
        }
    }

  Store *store = op_context.GetStore ();
  if (!store)
    {
      return 0;
    }
  auto memory_rows = store->Execute (
      "SELECT memory_id FROM memories WHERE embedding_id = ? LIMIT 1",
      { embedding_id });
  if (!memory_rows.empty () && memory_rows[0].count ("memory_id") == 1)
    {
      return ExtractInt64 (memory_rows[0], "memory_id", 0);
    }
  auto signal_rows = store->Execute (
      "SELECT memory_id FROM signals WHERE embedding_id = ? LIMIT 1",
      { embedding_id });
  if (!signal_rows.empty () && signal_rows[0].count ("memory_id") == 1)
    {
      return ExtractInt64 (signal_rows[0], "memory_id", 0);
    }
  return 0;
}

void
AssembleOutputMemories (const OperationContext &op_context,
                        SignalProcessor::Output &out)
{
  const auto &cands = op_context.GetRetrievedMemoryEmbeddings ();
  const auto &records = op_context.GetRetrievedMemoryCandidates ();
  out.candidate_memory_ids.reserve (
      records.empty () ? cands.size () : records.size ());
  if (!records.empty ())
    {
      for (const auto &candidate : records)
        {
          if (candidate.memory_id > 0)
            {
              out.candidate_memory_ids.push_back (candidate.memory_id);
            }
        }
    }
  else
    {
      for (const auto &kv : cands)
        {
          const long long memory_id = ResolveMemoryIdForRetrievedEmbedding (
              op_context, kv.first);
          if (memory_id > 0)
            {
              out.candidate_memory_ids.push_back (memory_id);
            }
        }
    }
  for (const auto &e : op_context.GetMemoryUsageEvents ())
    {
      if (e.used)
        {
          out.used_memory_ids.push_back (
              e.memory_id > 0 ? e.memory_id
                              : static_cast<long long> (e.embedding_id));
        }
    }
}

void
CheckpointSQLiteStore (Store *store, bool full, OperationContext *op_context,
                       const char *timing_name)
{
  auto *sqlite_store = dynamic_cast<SQLiteStore *> (store);
  if (!sqlite_store)
    {
      return;
    }
  const auto start = std::chrono::steady_clock::now ();
  try
    {
      sqlite_store->Checkpoint (full);
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "cortext.sqlite_checkpoint_failed",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  if (op_context && timing_name)
    {
      op_context->AddOperationTiming (
          timing_name, ElapsedMillis (start, std::chrono::steady_clock::now ()));
    }
}

void
MaybeRunPassiveWalCheckpoint (Store *store, const Signal &signal,
                              const ProcessorContext &context,
                              OperationContext &op_context)
{
  static const bool enabled = [] {
    const char *value = std::getenv ("CORTEXT_FOREGROUND_WAL_CHECKPOINT");
    return value && (std::strcmp (value, "1") == 0
                     || std::strcmp (value, "true") == 0
                     || std::strcmp (value, "on") == 0);
  } ();
  if (!enabled)
    {
      return;
    }
  if (signal.force_consolidation)
    {
      return;
    }
  if (context.signals_processed == 0)
    {
      return;
    }
  if ((context.signals_processed % kWalPassiveCheckpointIntervalSignals) != 0)
    {
      return;
    }

  auto *sqlite_store = dynamic_cast<SQLiteStore *> (store);
  if (!sqlite_store)
    {
      return;
    }
  if (sqlite_store->WalFileBytes () < kWalPassiveCheckpointMinBytes)
    {
      const auto start = std::chrono::steady_clock::now ();
      op_context.AddOperationTiming (
          "SignalProcessor.sqlite_wal_checkpoint_skip",
          ElapsedMillis (start, std::chrono::steady_clock::now ()));
      return;
    }
  CheckpointSQLiteStore (store, false, &op_context,
                         "SignalProcessor.sqlite_wal_checkpoint_passive");
}

void
AssembleOutputFields (const OperationContext &op_context,
                      SignalProcessor::Output &out)
{
  out.interrupt_allowed = op_context.GetInterruptAllowed ();
  out.interrupt_aborted = op_context.GetInterruptAborted ();
  out.at_boundary = op_context.GetAtBoundary ();
  out.write_decision = op_context.GetWriteDecision ();
  out.threshold_T_dynamic = op_context.GetThresholdTDynamic ();
  out.threshold_hysteresis = op_context.GetThresholdHysteresis ();
  out.effective_focus = op_context.GetEffectiveFocus ();
  out.coherence = op_context.GetCoherence ();
  out.emotion_intensity = op_context.GetEmotionIntensity ();
  out.valence = op_context.GetValence ();
  out.arousal = op_context.GetArousal ();
  out.mni_jaccard = op_context.GetMniJaccard ();
  out.mni_best_mu = op_context.GetMniBestMu ();
  out.mni_dup_thresh = op_context.GetMniDupThresh ();
  out.mni_tau_jaccard_eff = op_context.GetMniTauJaccardEff ();
  out.mni_tau_mu_eff = op_context.GetMniTauMuEff ();
  out.interrupt_gate_has_candidates
      = op_context.GetInterruptGateHasCandidates ();
  out.interrupt_gate_blocked_no_store
      = op_context.GetInterruptGateBlockedNoStore ();
  out.interrupt_gate_rel_pass = op_context.GetInterruptGateRelPass ();
  out.interrupt_gate_novelty_pass = op_context.GetInterruptGateNoveltyPass ();
  out.interrupt_gate_mu_pass = op_context.GetInterruptGateMuPass ();
  out.interrupt_gate_novelty_mu_pass
      = op_context.GetInterruptGateNoveltyMuPass ();
  out.interrupt_gate_dup_pass = op_context.GetInterruptGateDupPass ();
  out.interrupt_gate_boundary_mu_pass
      = op_context.GetInterruptGateBoundaryMuPass ();
  out.interrupt_gate_rel_star = op_context.GetInterruptGateRelStar ();
  out.interrupt_gate_retrieval_thresh
      = op_context.GetInterruptGateRetrievalThresh ();
  out.interrupt_gate_boundary_mult_eff
      = op_context.GetInterruptGateBoundaryMultEff ();
  out.boundary_score = op_context.GetBoundaryScore ();
  out.boundary_type = op_context.GetBoundaryType ();
  out.interrupt_gate_affect_drive = op_context.GetInterruptGateAffectDrive ();
  out.composite_score = op_context.GetCompositeScore ();
  out.serial_position_multiplier = op_context.GetSerialPositionMultiplier ();
  out.metrics = op_context.GetAllMetrics ();
  out.operation_ms = op_context.GetOperationTimings ();
  out.stored_embedding_id = op_context.GetStoredEmbeddingId ();
  out.stored_memory_id = op_context.GetStoredMemoryId ();
  out.stored_signal_id = op_context.GetStoredSignalId ();
  const auto &processor_context = op_context.GetProcessorContext ();
  out.soft_anchor_enabled = processor_context.soft_anchor_enabled;
  out.soft_anchor_state_count = processor_context.soft_anchor_last_state_count;
  out.soft_anchor_link_count = processor_context.soft_anchor_last_link_count;
  out.soft_anchor_create_count = processor_context.soft_anchor_last_create_count;
  out.soft_anchor_update_count = processor_context.soft_anchor_last_update_count;
  out.soft_anchor_none_count = processor_context.soft_anchor_last_none_count;
  out.soft_anchor_last_update_us
      = processor_context.soft_anchor_last_update_us;
  out.soft_anchor_mean_update_us
      = processor_context.soft_anchor_update_count > 0
            ? processor_context.soft_anchor_total_update_us
                  / static_cast<double> (
                      processor_context.soft_anchor_update_count)
            : 0.0;
}

void
ApplyConsolidationHint (const Signal &signal, const SignalProcessor::Config &cfg,
                        const ProcessorContext &ctx,
                        SignalProcessor::Output &out)
{
  if (signal.force_consolidation)
    {
      out.consolidation_state = ConsolidationState::None;
      return;
    }
  const auto rate_state
      = operations::consolidation_throughput_state_internal::Find (ctx);
  out.consolidation_state
      = operations::consolidation_throughput_state_internal::Classify (
          rate_state, ctx.m_rate,
          std::max (0, ctx.memories_since_consolidation), cfg.focus,
          cfg.sensitivity, cfg.stability);
}

const char *
GetMetricName (operations::Metric metric)
{
  switch (metric)
    {
    case operations::Metric::relevance:
      return "relevance";
    case operations::Metric::mismatch:
      return "mismatch";
    case operations::Metric::surprise:
      return "surprise";
    case operations::Metric::rarity:
      return "rarity";
    case operations::Metric::drift:
      return "drift";
    case operations::Metric::contradiction:
      return "contradiction";
    case operations::Metric::utility:
      return "utility";
    case operations::Metric::periphery:
      return "periphery";
    case operations::Metric::coverage:
      return "coverage";
    case operations::Metric::salience:
      return "salience";
    case operations::Metric::valence:
      return "valence";
    case operations::Metric::arousal:
      return "arousal";

    case operations::Metric::focus_spread:
      return "focus_spread";
    case operations::Metric::drift_mag:
      return "drift_mag";
    case operations::Metric::aw_prev:
      return "aw_prev";
    case operations::Metric::rate_prev:
      return "rate_prev";
    case operations::Metric::hys_prev:
      return "hys_prev";
    case operations::Metric::embedding_surprisal:
      return "embedding_surprisal";
    default:
      return "unknown";
    }
}

void
LogMetricTelemetry (const std::unordered_map<operations::Metric, double> &metrics)
{
  for (const auto &kv : metrics)
    {
      const char *metric_name = GetMetricName (kv.first);
      const std::string metric_full_name
          = std::string ("cortext.metric.") + metric_name;
      telemetry::RecordHistogram (metric_full_name, kv.second);
    }
}

void
LogProcessTelemetry (const OperationContext &op_context,
                     const SignalProcessor::Output &out)
{
  telemetry::RecordHistogram ("cortext.threshold_T_dynamic",
                              op_context.GetThresholdTDynamic ());
  telemetry::RecordHistogram ("cortext.threshold_hysteresis",
                              op_context.GetThresholdHysteresis ());
  telemetry::RecordHistogram ("cortext.effective_focus",
                              op_context.GetEffectiveFocus ());
  telemetry::RecordHistogram ("cortext.coherence", op_context.GetCoherence ());
  telemetry::RecordHistogram ("cortext.emotion_intensity",
                              op_context.GetEmotionIntensity ());
  telemetry::RecordHistogram ("cortext.mni_jaccard", op_context.GetMniJaccard ());
  telemetry::RecordHistogram ("cortext.mni_best_mu", op_context.GetMniBestMu ());
  telemetry::RecordHistogram ("cortext.mni_dup_thresh",
                              op_context.GetMniDupThresh ());
  telemetry::RecordHistogram ("cortext.last_weight_sum",
                              op_context.GetLastWeightSum ());
  telemetry::RecordHistogram ("cortext.last_effective_metric_count",
                              static_cast<double> (
                                  op_context.GetLastEffectiveMetricCount ()));
  LogMetricTelemetry (out.metrics);
}

// --- Helpers for vector/matrix conversion ---

inline std::vector<float>
ToFloatVector (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

inline std::vector<char>
SerializeUint64Vector (const std::vector<uint64_t> &values)
{
  std::vector<char> blob (values.size () * sizeof (uint64_t));
  if (!values.empty ())
    {
      std::memcpy (blob.data (), values.data (), blob.size ());
    }
  return blob;
}

inline std::vector<uint64_t>
DeserializeUint64Vector (const std::any &blob)
{
  const uint64_t *data = nullptr;
  size_t count = 0;

  if (blob.type () == typeid (std::vector<char>))
    {
      const auto &vec = std::any_cast<const std::vector<char> &> (blob);
      data = reinterpret_cast<const uint64_t *> (vec.data ());
      count = vec.size () / sizeof (uint64_t);
    }
  else if (blob.type () == typeid (std::vector<unsigned char>))
    {
      const auto &vec
          = std::any_cast<const std::vector<unsigned char> &> (blob);
      data = reinterpret_cast<const uint64_t *> (vec.data ());
      count = vec.size () / sizeof (uint64_t);
    }

  if (!data || count == 0)
    return {};
  return std::vector<uint64_t> (data, data + count);
}

inline Eigen::VectorXf
BlobToEigen (const std::any &blob)
{
  const float *data = nullptr;
  size_t size = 0;

  if (blob.type () == typeid (std::vector<char>))
    {
      const auto &vec = std::any_cast<const std::vector<char> &> (blob);
      data = reinterpret_cast<const float *> (vec.data ());
      size = vec.size () / sizeof (float);
    }
  else if (blob.type () == typeid (std::vector<unsigned char>))
    {
      const auto &vec
          = std::any_cast<const std::vector<unsigned char> &> (blob);
      data = reinterpret_cast<const float *> (vec.data ());
      size = vec.size () / sizeof (float);
    }
  else
    {
      return Eigen::VectorXf ();
    }

  if (size == 0)
    return Eigen::VectorXf ();

  Eigen::VectorXf result (static_cast<Eigen::Index> (size));
  std::memcpy (result.data (), data, size * sizeof (float));
  return result;
}

// Serialize blender P matrix (2D vector of doubles) to flat float vector
inline std::vector<float>
SerializeMatrix (const std::vector<std::vector<double> > &mat)
{
  std::vector<float> out;
  for (const auto &row : mat)
    {
      for (double d : row)
        out.push_back (static_cast<float> (d));
    }
  return out;
}

// Deserialize flat float vector to 2D matrix
inline std::vector<std::vector<double> >
DeserializeMatrix (const std::any &blob, size_t n)
{
  const float *data = nullptr;
  size_t byte_size = 0;

  if (blob.type () == typeid (std::vector<char>))
    {
      const auto &vec = std::any_cast<const std::vector<char> &> (blob);
      data = reinterpret_cast<const float *> (vec.data ());
      byte_size = vec.size ();
    }
  else if (blob.type () == typeid (std::vector<unsigned char>))
    {
      const auto &vec
          = std::any_cast<const std::vector<unsigned char> &> (blob);
      data = reinterpret_cast<const float *> (vec.data ());
      byte_size = vec.size ();
    }
  else
    {
      return {};
    }

  const size_t expected = n * n * sizeof (float);
  if (byte_size != expected)
    return {};

  std::vector<std::vector<double> > result (n, std::vector<double> (n));
  size_t idx = 0;
  for (size_t i = 0; i < n; ++i)
    {
      for (size_t j = 0; j < n; ++j)
        {
          result[i][j] = static_cast<double> (data[idx++]);
        }
    }
  return result;
}

// Helper to extract int64 from std::any
inline int64_t
ExtractInt64 (const std::map<std::string, std::any> &row,
              const std::string &key, int64_t default_val = 0)
{
  auto it = row.find (key);
  if (it == row.end ())
    return default_val;
  const std::any &v = it->second;
  if (v.type () == typeid (long long))
    return static_cast<int64_t> (std::any_cast<long long> (v));
  if (v.type () == typeid (int64_t))
    return std::any_cast<int64_t> (v);
  if (v.type () == typeid (int))
    return static_cast<int64_t> (std::any_cast<int> (v));
  return default_val;
}

// Helper to extract double from std::any
inline double
ExtractDouble (const std::map<std::string, std::any> &row,
               const std::string &key, double default_val = 0.0)
{
  auto it = row.find (key);
  if (it == row.end ())
    return default_val;
  const std::any &v = it->second;
  if (v.type () == typeid (double))
    return std::any_cast<double> (v);
  if (v.type () == typeid (float))
    return static_cast<double> (std::any_cast<float> (v));
  if (v.type () == typeid (long long))
    return static_cast<double> (std::any_cast<long long> (v));
  if (v.type () == typeid (int))
    return static_cast<double> (std::any_cast<int> (v));
  return default_val;
}

inline bool
NearlyEqual (double a, double b)
{
  return std::abs (a - b) <= 1e-12;
}

inline double
ExtractPolicyDouble (const std::map<std::string, std::any> &row,
                     const std::string &key, double legacy_default,
                     double knob_default, bool legacy_policy_defaults)
{
  const double value = ExtractDouble (row, key, knob_default);
  if (legacy_policy_defaults && NearlyEqual (value, legacy_default)
      && !NearlyEqual (knob_default, legacy_default))
    {
      return knob_default;
    }
  return value;
}

inline std::string
ExtractString (const std::map<std::string, std::any> &row,
               const std::string &key,
               const std::string &default_val = std::string ())
{
  auto it = row.find (key);
  if (it == row.end () || !it->second.has_value ())
    return default_val;
  if (it->second.type () == typeid (std::string))
    return std::any_cast<std::string> (it->second);
  return default_val;
}

std::deque<long long>
ParseMemoryIds (const std::string &text)
{
  std::deque<long long> ids;
  std::stringstream stream (text);
  std::string item;
  while (std::getline (stream, item, ','))
    {
      try
        {
          if (!item.empty ())
            {
              ids.push_back (std::stoll (item));
            }
        }
      catch (...)
        {
        }
    }
  return ids;
}

void
SeedKnobDerivedStateDefaults (ProcessorContext &ctx,
                              const SignalProcessor::Config &config)
{
  const double F = core::Clamp (config.focus, 0.0, 1.0);
  const double S = core::Clamp (config.sensitivity, 0.0, 1.0);
  const double T = core::Clamp (config.stability, 0.0, 1.0);
  const auto focus_priors = core::FocusStatePriorsForKnobs (F);
  const auto sensitivity_priors = core::SensitivityStatePriorsForKnobs (S);
  const auto stability_priors = core::StabilityStatePriorsForKnobs (T);

  ctx.weight_relevance_prior = focus_priors.relevance_weight;
  ctx.coverage_gain_floor_prior = focus_priors.coverage_gain_floor;
  ctx.mismatch_weight_prior = focus_priors.mismatch_weight;
  ctx.attention_width_prior = core::Lerp (
      static_cast<double> (core::kAttentionWidthMin),
      static_cast<double> (core::kAttentionWidthMax), 1.0 - F);
  ctx.weight_relevance = ctx.weight_relevance_prior;
  ctx.attention_width = ctx.attention_width_prior;
  ctx.coverage_gain_floor = ctx.coverage_gain_floor_prior;
  ctx.mismatch_weight = ctx.mismatch_weight_prior;

  ctx.base_rate_prior = core::BaseRatePrior (S);
  ctx.weight_novelty_prior = sensitivity_priors.novelty_weight;
  ctx.weight_surprise_prior = sensitivity_priors.surprise_weight;
  ctx.weight_valence_prior = sensitivity_priors.valence_weight;
  ctx.weight_arousal_prior = sensitivity_priors.arousal_weight;
  ctx.weight_emotion_prior = sensitivity_priors.emotion_weight;
  ctx.emotion_gain_prior = sensitivity_priors.emotion_gain;
  ctx.score_gain_prior = sensitivity_priors.score_gain;
  ctx.rate_target_prior = ctx.base_rate_prior;
  ctx.weight_novelty = ctx.weight_novelty_prior;
  ctx.weight_surprise = ctx.weight_surprise_prior;
  ctx.weight_valence = ctx.weight_valence_prior;
  ctx.weight_arousal = ctx.weight_arousal_prior;
  ctx.emotion_gain = ctx.emotion_gain_prior;
  ctx.score_gain = ctx.score_gain_prior;
  ctx.rate_target = ctx.rate_target_prior;

  ctx.hysteresis_band_prior = core::BaseBandPrior (T);
  ctx.half_life_prior = core::BaseHalfLifePrior (T);
  ctx.rate_decay_prior = stability_priors.rate_decay;
  ctx.periphery_half_life_prior
      = core::ClampHalfLife (stability_priors.secondary_half_life_scale
                             * ctx.half_life_prior);
  ctx.salience_half_life_prior
      = core::ClampHalfLife (stability_priors.secondary_half_life_scale
                             * ctx.half_life_prior);
  ctx.drift_weight_prior = stability_priors.drift_weight;
  ctx.half_life = ctx.half_life_prior;
  ctx.rate_decay = ctx.rate_decay_prior;
  ctx.periphery_half_life = ctx.periphery_half_life_prior;
  ctx.salience_half_life = ctx.salience_half_life_prior;
  ctx.drift_weight = ctx.drift_weight_prior;

  ctx.T_dynamic = core::TPrior (F, S, T);
  ctx.T_target = ctx.T_dynamic;
  ctx.hysteresis = ctx.hysteresis_band_prior;
}

// --- State Loading Functions ---
// v2 Schema: Unified state table replaces processor_state + blender tables

bool
LoadState (Store &store, ProcessorContext &ctx,
           const SignalProcessor::Config &config)
{
  try
    {
      auto rows = store.Execute ("SELECT * FROM state WHERE id = 1");
      if (rows.empty ())
        return false; // No persisted state, use defaults

      const auto &row = rows[0];
      SeedKnobDerivedStateDefaults (ctx, config);

      // === Processor state fields ===
      ctx.signals_processed
          = static_cast<int> (ExtractInt64 (row, "signals_processed", 0));
      const bool legacy_policy_defaults
          = NearlyEqual (ExtractDouble (row, "theta_dynamic", 0.2), 0.2)
            && NearlyEqual (ExtractDouble (row, "theta_target", 0.2), 0.2)
            && NearlyEqual (ExtractDouble (row, "weight_relevance", 0.5), 0.5)
            && NearlyEqual (ExtractDouble (row, "coverage_gain_floor", 0.65),
                            0.65)
            && NearlyEqual (ExtractDouble (row, "weight_novelty", 0.3), 0.3)
            && NearlyEqual (ExtractDouble (row, "weight_surprise", 0.2), 0.2)
            && NearlyEqual (ExtractDouble (row, "rate_decay", 0.60), 0.60)
            && NearlyEqual (ExtractDouble (row, "drift_weight", 0.5), 0.5);
      ctx.u_t = ExtractDouble (row, "u_uncertainty", 0.0);
      ctx.outcome_pred = ExtractDouble (row, "outcome_pred", 0.0);
      ctx.neuromod_ach = ExtractDouble (row, "neuromod_ach", 0.0);
      ctx.neuromod_ne = ExtractDouble (row, "neuromod_ne", 0.0);
      ctx.neuromod_da = ExtractDouble (row, "neuromod_da", 0.0);
      ctx.osc_phase = ExtractDouble (row, "osc_phase", 0.0);
      ctx.weight_relevance = ExtractPolicyDouble (
          row, "weight_relevance", 0.5, ctx.weight_relevance,
          true);
      ctx.attention_width = ExtractPolicyDouble (
          row, "attention_width", 1.57, ctx.attention_width,
          true);
      ctx.coverage_gain_floor
          = ExtractPolicyDouble (row, "coverage_gain_floor", 0.65,
                                 ctx.coverage_gain_floor,
                                 true);
      ctx.mismatch_weight = ExtractPolicyDouble (
          row, "mismatch_weight", 0.5, ctx.mismatch_weight,
          true);
      ctx.T_dynamic = ExtractPolicyDouble (
          row, "theta_dynamic", 0.2, ctx.T_dynamic,
          true);
      ctx.T_target = ExtractPolicyDouble (
          row, "theta_target", 0.2, ctx.T_dynamic,
          true);
      ctx.hysteresis = ExtractPolicyDouble (
          row, "hysteresis", 0.05, ctx.hysteresis, true);
      ctx.half_life = ExtractPolicyDouble (
          row, "half_life", 120.0, ctx.half_life, true);
      ctx.rate_target = ExtractPolicyDouble (
          row, "rate_target", 0.2, ctx.rate_target,
          true);
      ctx.delta_half_life_adj
          = ExtractDouble (row, "delta_half_life_adj", 0.0);
      ctx.sustained_influence
          = ExtractDouble (row, "sustained_influence", 0.0);
      ctx.last_signal_timestamp
          = static_cast<uint64_t> (ExtractInt64 (row, "last_signal_timestamp", 0));
      ctx.episode_start_ts
          = static_cast<uint64_t> (ExtractInt64 (row, "episode_start_ts", 0));
      ctx.last_interrupt_tick
          = static_cast<int> (ExtractInt64 (row, "last_interrupt_tick",
                                            ctx.last_interrupt_tick));
      ctx.last_retrieval_ts
          = static_cast<uint64_t> (ExtractInt64 (row, "last_retrieval_ts", 0));
      ctx.last_consolidation_ts
          = static_cast<uint64_t> (ExtractInt64 (row, "last_consolidation_ts", 0));
      ctx.consolidation_count
          = static_cast<int> (ExtractInt64 (row, "consolidation_count", 0));
      ctx.memories_since_consolidation
          = static_cast<int> (
              ExtractInt64 (row, "memories_since_consolidation", 0));
      ctx.is_processing_signal
          = ExtractInt64 (row, "is_processing_signal", 0) != 0;

      ctx.wm_last_accepted
          = ExtractInt64 (row, "wm_last_accepted", 0) != 0;
      ctx.wm_last_chunked
          = ExtractInt64 (row, "wm_last_chunked", 0) != 0;

      // Sensitivity state
      ctx.weight_novelty = ExtractPolicyDouble (
          row, "weight_novelty", 0.3, ctx.weight_novelty,
          true);
      ctx.weight_surprise = ExtractPolicyDouble (
          row, "weight_surprise", 0.2, ctx.weight_surprise,
          true);
      ctx.weight_valence = ExtractPolicyDouble (
          row, "weight_valence", 0.4, ctx.weight_valence,
          true);
      ctx.weight_arousal = ExtractPolicyDouble (
          row, "weight_arousal", 0.0, ctx.weight_arousal,
          true);
      ctx.emotion_gain = ExtractPolicyDouble (
          row, "emotion_gain", 1.0, ctx.emotion_gain,
          true);
      ctx.score_gain = ExtractPolicyDouble (
          row, "score_gain", 1.0, ctx.score_gain, true);

      // Stability state
      ctx.rate_decay = ExtractPolicyDouble (
          row, "rate_decay", 0.60, ctx.rate_decay, true);
      ctx.periphery_half_life
          = ExtractPolicyDouble (row, "periphery_half_life", 120.0,
                                 ctx.periphery_half_life,
                                 true);
      ctx.salience_half_life
          = ExtractPolicyDouble (row, "salience_half_life", 120.0,
                                 ctx.salience_half_life,
                                 true);
      ctx.drift_weight = ExtractPolicyDouble (
          row, "drift_weight", 0.5, ctx.drift_weight, true);
      ctx.retention_ema = ExtractDouble (row, "retention_ema", 0.0);

      // Rate control (Algorithm 8)
      ctx.m_rate = ExtractDouble (row, "m_rate", 0.0);
      operations::consolidation_throughput_state_internal::Reset (
          ctx,
          { ExtractDouble (row, "consolidation_rate_floor", 0.0),
            ExtractDouble (row, "consolidation_rate_peak", 0.0),
            ExtractInt64 (row, "consolidation_rate_initialized", 0) != 0,
            ExtractInt64 (row, "consolidation_rate_armed", 1) != 0 });
      ctx.rho_hat_prev = ExtractDouble (row, "rho_hat_prev", 0.0);
      ctx.rate_ticks = static_cast<int> (ExtractInt64 (row, "rate_ticks", 0));
      ctx.dt_ema = ExtractDouble (row, "dt_ema", 0.0);
      ctx.last_rate_timestamp
          = static_cast<uint64_t> (ExtractInt64 (row, "last_rate_timestamp", 0));
      ctx.reliability = ExtractPolicyDouble (
          row, "reliability", 1.0, 0.0, legacy_policy_defaults);

      // Restore write rate window timestamps if present.
      auto write_rate_it = row.find ("write_rate_timestamps");
      if (write_rate_it != row.end () && write_rate_it->second.has_value ())
        {
          const auto ts = DeserializeUint64Vector (write_rate_it->second);
          if (!ts.empty ())
            {
              ctx.write_rate_window_.SetTimestamps (ts);
            }
        }

      // Emotion state (Algorithm 4)
      ctx.emotion_intensity_ewma
          = ExtractDouble (row, "emotion_intensity", 0.0);
      ctx.valence_ewma = ExtractDouble (row, "valence", 0.5);
      ctx.arousal_ewma = ExtractDouble (row, "arousal", 0.0);
      ctx.flashbulb_rate_ewma
          = ExtractDouble (row, "flashbulb_rate", 0.0);

      // Mood state (Algorithm 4b) - stored as BLOB (48 bytes = 6 doubles)
      auto mood_it = row.find ("mood_vector");
      if (mood_it != row.end () && mood_it->second.has_value ())
        {
          const double *data = nullptr;
          size_t byte_size = 0;
          if (mood_it->second.type () == typeid (std::vector<char>))
            {
              const auto &vec
                  = std::any_cast<const std::vector<char> &> (mood_it->second);
              data = reinterpret_cast<const double *> (vec.data ());
              byte_size = vec.size ();
            }
          else if (mood_it->second.type () == typeid (std::vector<unsigned char>))
            {
              const auto &vec = std::any_cast<const std::vector<unsigned char> &> (
                  mood_it->second);
              data = reinterpret_cast<const double *> (vec.data ());
              byte_size = vec.size ();
            }
          if (data && byte_size >= 6 * sizeof (double))
            {
              for (size_t i = 0; i < 6; ++i)
                ctx.mood_vector[i] = data[i];
            }
        }
      ctx.last_mood_ts
          = static_cast<uint64_t> (ExtractInt64 (row, "last_mood_ts", 0));

      // Embedding prediction error state (Section 3.1.4)
      auto last_emb_it = row.find ("last_embedding");
      if (last_emb_it != row.end () && last_emb_it->second.has_value ())
        {
          Eigen::VectorXf emb = BlobToEigen (last_emb_it->second);
          if (emb.size () > 0)
            ctx.last_embedding = std::move (emb);
        }
      auto pred_ema_it = row.find ("x_pred_ema");
      if (pred_ema_it != row.end () && pred_ema_it->second.has_value ())
        {
          Eigen::VectorXf pred = BlobToEigen (pred_ema_it->second);
          if (pred.size () > 0)
            ctx.x_pred_ema = std::move (pred);
        }

      // === Blender weights (from unified state table) ===
      const double blender_fallback = core::BlendBootstrapFallback (
          config.focus, config.sensitivity, config.stability);
      ctx.blender_state[operations::Metric::relevance]
          = ExtractDouble (row, "w_relevance", blender_fallback);
      ctx.blender_state[operations::Metric::mismatch]
          = ExtractDouble (row, "w_mismatch", blender_fallback);
      ctx.blender_state[operations::Metric::surprise]
          = ExtractDouble (row, "w_surprise", blender_fallback);
      ctx.blender_state[operations::Metric::rarity]
          = ExtractDouble (row, "w_rarity", blender_fallback);
      ctx.blender_state[operations::Metric::drift]
          = ExtractDouble (row, "w_drift", blender_fallback);
      ctx.blender_state[operations::Metric::contradiction]
          = ExtractDouble (row, "w_contradiction", blender_fallback);
      ctx.blender_state[operations::Metric::utility]
          = ExtractDouble (row, "w_utility", blender_fallback);
      ctx.blender_state[operations::Metric::periphery]
          = ExtractDouble (row, "w_periphery", blender_fallback);
      ctx.blender_state[operations::Metric::coverage]
          = ExtractDouble (row, "w_coverage", blender_fallback);
      ctx.blender_state[operations::Metric::salience]
          = ExtractDouble (row, "w_salience", blender_fallback);
      ctx.blender_state[operations::Metric::valence]
          = ExtractDouble (row, "w_valence", blender_fallback);
      ctx.blender_state[operations::Metric::arousal]
          = ExtractDouble (row, "w_arousal", blender_fallback);
      ctx.blender_ready = ExtractInt64 (row, "blender_ready", 0) != 0;
      ctx.blender_update_count
          = static_cast<int> (ExtractInt64 (row, "blender_update_count", 0));

      // === Blender covariance matrix (P_matrix) ===
      auto P_it = row.find ("blender_P_matrix");
      if (P_it != row.end () && P_it->second.has_value ())
        {
          ctx.blender_P = DeserializeMatrix (P_it->second, 12);
        }

      ctx.focus_priors_initialized = true;
      ctx.sensitivity_priors_initialized = true;
      ctx.stability_priors_initialized = true;
      return true;
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load state",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  return false;
}

void
LoadRecentContext (Store &store, ProcessorContext &ctx,
                   const SignalProcessor::Config &config)
{
  try
    {
      const long long keep = static_cast<long long> (
          std::max (1.0, core::NCtx (config.stability)
                             + static_cast<double> (
                                 core::KCtx (config.stability))));
      auto rows = store.Execute (
          "SELECT embedding FROM ("
          "  SELECT s.signal_id, s.embedding_id, e.embedding, s.timestamp "
          "  FROM signals s "
          "  JOIN embeddings e ON s.embedding_id = e.embedding_id "
          "  ORDER BY s.timestamp DESC, s.signal_id DESC "
          "  LIMIT ?"
          ") ORDER BY timestamp ASC, signal_id ASC",
          { keep });
      for (const auto &row : rows)
        {
          auto it = row.find ("embedding");
          if (it != row.end () && it->second.has_value ())
            {
              Eigen::VectorXf emb = BlobToEigen (it->second);
              if (emb.size () > 0)
                {
                  ctx.recent_context_embeddings.push_back (std::move (emb));
                }
            }
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load recent context",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

void
LoadRecentScores (Store &store, ProcessorContext &ctx,
                  const SignalProcessor::Config &config)
{
  try
    {
      const int w = core::WScore (config.stability);
      const long long keep = static_cast<long long> (std::max (
          1, std::max (w, core::RecentScoreHistoryLimit (config.stability))));
      auto rows = store.Execute (
          "SELECT score FROM ("
          "  SELECT signal_id, score, timestamp "
          "  FROM signals "
          "  WHERE score IS NOT NULL "
          "  ORDER BY timestamp DESC, signal_id DESC "
          "  LIMIT ?"
          ") ORDER BY timestamp ASC, signal_id ASC",
          { keep });
      for (const auto &row : rows)
        {
          ctx.recent_scores.push_back (ExtractDouble (row, "score", 0.0));
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load recent scores",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

void
LoadPredictivePreActivationIds (Store &store, ProcessorContext &ctx)
{
  try
    {
      ctx.predictive_pre_activation_embedding_ids.clear ();
      ctx.predictive_pre_activation_memory_ids.clear ();
      auto rows = store.Execute (
          "SELECT memory_id FROM memories "
          "WHERE COALESCE(pre_activation, 0.0) > 1e-6");
      ctx.predictive_pre_activation_memory_ids.reserve (rows.size ());
      for (const auto &row : rows)
        {
          const long long memory_id = ExtractInt64 (row, "memory_id", 0);
          if (memory_id > 0)
            {
              ctx.predictive_pre_activation_memory_ids.insert (memory_id);
            }
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load predictive pre-activation ids",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

void
LoadRetrievalSuppressionIds (Store &store, ProcessorContext &ctx)
{
  try
    {
      ctx.retrieval_suppression_embedding_ids.clear ();
      ctx.retrieval_suppression_memory_ids.clear ();
      auto rows = store.Execute (
          "SELECT memory_id FROM memories "
          "WHERE COALESCE(suppression, 0.0) > 1e-9");
      ctx.retrieval_suppression_memory_ids.reserve (rows.size ());
      for (const auto &row : rows)
        {
          const long long memory_id = ExtractInt64 (row, "memory_id", 0);
          if (memory_id > 0)
            {
              ctx.retrieval_suppression_memory_ids.insert (memory_id);
            }
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load retrieval suppression ids",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

void
LoadHistoricalSurfaceSearchCache (Store &store, ProcessorContext &ctx)
{
  try
    {
      auto embedding_rows = store.Execute (
          "SELECT e.embedding_id, e.embedding, "
          "       COALESCE(m.memory_id, 0) AS memory_id, "
          "       COALESCE(m.start_ts, 0) AS start_ts, "
          "       COALESCE(m.kind, '') AS kind, "
          "       COALESCE(m.source_id, '') AS source_id "
          "FROM embeddings e "
          "LEFT JOIN memories m ON m.embedding_id = e.embedding_id "
          "  AND m.kind = 'LONG_TERM' "
          "ORDER BY e.embedding_id, COALESCE(m.start_ts, 0), m.memory_id");
      std::vector<
          operations::historical_surface_search_cache_internal::Entry>
          search_entries;
      search_entries.reserve (embedding_rows.size ());
      for (const auto &row : embedding_rows)
        {
          auto emb_it = row.find ("embedding");
          if (emb_it == row.end () || !emb_it->second.has_value ())
            {
              continue;
            }
          search_entries.push_back (
              { ExtractInt64 (row, "embedding_id", 0),
                ExtractInt64 (row, "memory_id", 0),
                ExtractInt64 (row, "start_ts", 0),
                ExtractString (row, "kind"), ExtractString (row, "source_id"),
                BlobToEigen (emb_it->second) });
        }
      auto current_rows = store.Execute (
          "SELECT memory_id, embedding_id, embedding "
          "FROM current_memory_embeddings ORDER BY memory_id");
      std::vector<
          operations::historical_surface_search_cache_internal::Entry>
          current_entries;
      current_entries.reserve (current_rows.size ());
      for (const auto &row : current_rows)
        {
          auto emb_it = row.find ("embedding");
          if (emb_it == row.end () || !emb_it->second.has_value ())
            {
              continue;
            }
          current_entries.push_back (
              { ExtractInt64 (row, "embedding_id", 0),
                ExtractInt64 (row, "memory_id", 0), 0, std::string (),
                std::string (), BlobToEigen (emb_it->second) });
        }
      if (search_entries.size () != embedding_rows.size ()
          || current_entries.size () != current_rows.size ()
          || !operations::historical_surface_search_cache_internal::Reset (
              ctx, std::move (search_entries), std::move (current_entries)))
        {
          operations::historical_surface_search_cache_internal::Erase (ctx);
        }
    }
  catch (const std::exception &e)
    {
      operations::historical_surface_search_cache_internal::Erase (ctx);
      telemetry::LogWarn (
          "Failed to load historical surface search cache",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

void
LoadRetrievalSurfaceCache (Store &store, ProcessorContext &ctx)
{
  try
    {
      operations::historical_surface_search_cache_internal::Erase (ctx);
      ctx.retrieval_surface_cache.clear ();
      ctx.retrieval_surface_index.clear ();
      ctx.retrieval_surface_embedding_index.clear ();
      ctx.retrieval_surface_source_index.clear ();
      ctx.retrieval_surface_source_index_dirty.clear ();
      ctx.association_cache.clear ();
      ctx.association_cache_index.clear ();
      ctx.association_fanout_cache = {};

      const long long constructive_recall_enabled
          = operations::constructive_recall::Disabled () ? 0LL : 1LL;
      auto rows = store.Execute (
          "SELECT m.memory_id, m.embedding_id AS base_embedding_id, "
          "       CASE WHEN ?1 != 0 AND latest_r.reconstruction_id IS NOT NULL "
          "            THEN latest_r.embedding_id "
          "            WHEN ?1 != 0 AND cme.memory_id IS NOT NULL "
          "            THEN cme.embedding_id "
          "            ELSE m.embedding_id "
          "       END AS embedding_id, "
          "       CASE WHEN ?1 != 0 AND latest_r.reconstruction_id IS NOT NULL "
          "            THEN latest_e.embedding "
          "            WHEN ?1 != 0 AND cme.memory_id IS NOT NULL "
          "            THEN cme.embedding "
          "            ELSE base_e.embedding END AS embedding, "
          "       base_e.embedding AS base_embedding, "
          "       COALESCE(m.created_at, "
          "                CASE WHEN ?1 != 0 THEN latest_r.created_at END, "
          "                CASE WHEN ?1 != 0 THEN cme.created_at END, "
          "                base_e.created_at, 0) "
          "         AS created_at, "
          "       COALESCE(m.start_ts, 0) AS start_ts, "
          "       COALESCE(m.last_access, 0) AS last_access, "
          "       COALESCE(m.retrieved_count, 0) AS retrieved_count, "
          "       COALESCE(m.used_count, 0) AS used_count, "
          "       COALESCE(NULLIF(m.start_ts, 0), m.created_at, "
          "                CASE WHEN ?1 != 0 THEN latest_r.created_at END, "
          "                CASE WHEN ?1 != 0 THEN cme.created_at END, "
          "                base_e.created_at, 0) AS event_ts, "
          "       m.context, m.source_reliability, "
          "       m.source_contradiction_count, m.emotional_intensity, "
          "       m.s_arousal_avg, m.kind, m.source_id, m.modality, "
          "       COALESCE(m.pre_activation, 0.0) AS pre_activation, "
          "       CASE WHEN (cme.memory_id IS NOT NULL "
          "                  OR m.embedding_id IS NOT NULL) "
          "             AND m.kind = 'LONG_TERM' "
          "            THEN 1 ELSE 0 END AS vector_seed_eligible "
          "FROM memories m "
          "LEFT JOIN current_memory_embeddings cme "
          "  ON cme.memory_id = m.memory_id "
          "LEFT JOIN memory_reconstructions latest_r "
          "  ON latest_r.reconstruction_id = ("
          "    SELECT MAX(mr2.reconstruction_id) "
          "    FROM memory_reconstructions mr2 "
          "    WHERE mr2.memory_id = m.memory_id"
          "  ) "
          "LEFT JOIN embeddings latest_e "
          "  ON latest_e.embedding_id = latest_r.embedding_id "
          "JOIN embeddings base_e ON base_e.embedding_id = m.embedding_id "
          "WHERE cme.memory_id IS NOT NULL "
          "   OR m.kind = 'LABEL' "
          "   OR (m.kind != 'WORKING' "
          "       AND m.kind != 'LABEL' "
          "       AND m.embedding_id IS NOT NULL)",
          { constructive_recall_enabled });

      ctx.retrieval_surface_cache.reserve (rows.size ());
      ctx.retrieval_surface_index.reserve (rows.size ());
      ctx.retrieval_surface_embedding_index.reserve (rows.size ());
      for (const auto &row : rows)
        {
          auto emb_it = row.find ("embedding");
          if (emb_it == row.end () || !emb_it->second.has_value ())
            {
              continue;
            }
          ProcessorContext::RetrievalSurfaceEntry entry;
          entry.memory_id = ExtractInt64 (row, "memory_id", 0);
          entry.embedding_id = ExtractInt64 (row, "embedding_id", 0);
          entry.created_at = ExtractInt64 (row, "created_at", 0);
          entry.start_ts = ExtractInt64 (row, "start_ts", 0);
          entry.last_access = ExtractInt64 (row, "last_access", 0);
          entry.event_ts = ExtractInt64 (row, "event_ts", entry.created_at);
          entry.retrieved_count = ExtractInt64 (row, "retrieved_count", 0);
          entry.used_count = ExtractInt64 (row, "used_count", 0);
          entry.kind = ExtractString (row, "kind");
          entry.source_id = ExtractString (row, "source_id");
          entry.modality = ExtractString (row, "modality");
          entry.source_reliability
              = ExtractDouble (row, "source_reliability", -1.0);
          entry.source_contradiction_count = static_cast<int> (
              ExtractInt64 (row, "source_contradiction_count", 0));
          entry.emotional_intensity
              = ExtractDouble (row, "emotional_intensity", 0.0);
          entry.arousal_avg = ExtractDouble (row, "s_arousal_avg", 0.0);
          entry.pre_activation = ExtractDouble (row, "pre_activation", 0.0);
          entry.vector_seed_eligible
              = ExtractInt64 (row, "vector_seed_eligible", 1) != 0;
          entry.embedding = BlobToEigen (emb_it->second);
          auto ctx_it = row.find ("context");
          if (ctx_it != row.end () && ctx_it->second.has_value ())
            {
              entry.context_embedding = BlobToEigen (ctx_it->second);
            }
          const long long association_memory_id = entry.memory_id;
          const long long association_embedding_id = entry.embedding_id;
          const Eigen::VectorXf association_embedding = entry.embedding;
          const bool association_is_association = entry.kind == "ASSOCIATION";
          ctx.UpsertRetrievalSurface (std::move (entry));
          ctx.UpsertAssociationCache (association_memory_id,
                                      association_embedding_id,
                                      association_embedding,
                                      association_is_association);
        }
      LoadHistoricalSurfaceSearchCache (store, ctx);
      ctx.SortRetrievalSurfaceSourceIndexes ();
      telemetry::LogDebug (
          "cortext.retrieval_surface_cache.load",
          { telemetry::Attribute::Int64 (
              "entry_count",
              static_cast<int64_t> (ctx.retrieval_surface_cache.size ())) });
    }
  catch (const std::exception &e)
    {
      operations::historical_surface_search_cache_internal::Erase (ctx);
      telemetry::LogWarn (
          "Failed to load retrieval surface cache",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

void
LoadObservedRetentionHistory (Store &store, ProcessorContext &ctx,
                              std::uint64_t now_ms, int history_limit)
{
  const int limit = std::max (1, history_limit);
  try
    {
      auto rows = store.Execute (
          "SELECT COALESCE(last_used, last_access) AS last_used "
          "FROM memories "
          "WHERE COALESCE(last_used, last_access) IS NOT NULL "
          "  AND kind != 'WORKING' "
          "ORDER BY last_used ASC "
          "LIMIT ?",
          { static_cast<long long> (limit) });
      for (const auto &row : rows)
        {
          const auto last_used
              = ExtractInt64 (row, "last_used", 0);
          if (last_used <= 0)
            continue;
          const double retention_sec
              = std::max (0.0,
                          static_cast<double> (
                              static_cast<long long> (now_ms) - last_used)
                              / 1000.0);
          ctx.observed_retention_history.push_back (retention_sec);
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load observed retention history",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

// v2 Schema: Load working memory from MEMORIES table (kind='WORKING')
void
LoadWorkingMemory (Store &store, ProcessorContext &ctx,
                   const SignalProcessor::Config &cfg, std::uint64_t now_ms)
{
  try
    {
      const double cost_per_slot
          = core::WMMaintenanceCostPerSlot (cfg.sensitivity, cfg.focus);
      const double strength_floor = core::WMStrengthFloor (
          cfg.focus, cfg.sensitivity, cfg.stability);

      // Load from MEMORIES table with kind='WORKING' and end_ts IS NULL (active)
      auto rows = store.Execute (
          "SELECT memory_id, embedding_id, source_id, strength, last_access, "
          "       strength_updated_at, "
          "       n_signals, s_max, s_avg, s_emotion_max, s_arousal_avg, "
          "       drift_mag, modality, start_ts "
          "FROM memories "
          "WHERE kind = 'WORKING' AND end_ts IS NULL "
          "ORDER BY start_ts ASC");

      int pos = 0;
      for (const auto &row : rows)
        {
          ProcessorContext::WMSlot slot;
          slot.memory_id = ExtractInt64 (row, "memory_id", 0);
          slot.pos_index = pos++;
          slot.strength = ExtractDouble (row, "strength", 0.0);

          // Extract source_id
          auto source_it = row.find ("source_id");
          if (source_it != row.end () && source_it->second.has_value ()
              && source_it->second.type () == typeid (std::string))
            {
              slot.source_id = std::any_cast<std::string> (source_it->second);
            }

          // Extract modality
          auto mod_it = row.find ("modality");
          if (mod_it != row.end () && mod_it->second.has_value ()
              && mod_it->second.type () == typeid (std::string))
            {
              slot.modality = std::any_cast<std::string> (mod_it->second);
            }

          // Load embedding for slot
          const auto embedding_id = ExtractInt64 (row, "embedding_id", 0);
          slot.embedding_id = embedding_id;
          if (embedding_id > 0)
            {
              auto emb_rows = store.Execute (
                  "SELECT embedding FROM embeddings WHERE embedding_id = ?",
                  { embedding_id });
              if (!emb_rows.empty ())
                {
                  auto emb_it = emb_rows[0].find ("embedding");
                  if (emb_it != emb_rows[0].end () && emb_it->second.has_value ())
                    {
                      slot.embedding = BlobToEigen (emb_it->second);
                    }
                }
            }

          // DB stores milliseconds, convert to seconds for last_ts
          const auto ts_ms = ExtractInt64 (row, "last_access", now_ms);
          slot.last_ts = static_cast<double> (ts_ms) / 1000.0;
          const auto persisted_strength_ts_ms
              = ExtractInt64 (row, "strength_updated_at", ts_ms);
          const auto strength_ts_ms = persisted_strength_ts_ms > 0
                                          ? persisted_strength_ts_ms
                                          : ts_ms;
          slot.strength_ts = static_cast<double> (strength_ts_ms) / 1000.0;
          slot.start_ts = ExtractInt64 (row, "start_ts", 0);

          // Apply time-based decay
          const double strength_before = slot.strength;
          const double strength_ts_before = slot.strength_ts;
          const double now_s = static_cast<double> (now_ms) / 1000.0;
          const double elapsed = now_s - slot.strength_ts;
          if (elapsed > 0)
            {
              slot.strength
                  = std::max (strength_floor,
                              slot.strength - cost_per_slot * elapsed);
              slot.strength_ts = now_s;
            }
          const bool metadata_changed
              = slot.strength != strength_before
                || slot.strength_ts != strength_ts_before;

          // Load extended metadata
          slot.n_signals = static_cast<int> (ExtractInt64 (row, "n_signals", 1));
          slot.s_max = ExtractDouble (row, "s_max", 0.0);
          slot.s_avg = ExtractDouble (row, "s_avg", 0.0);
          slot.s_emotion_max = ExtractDouble (row, "s_emotion_max", 0.0);
          slot.s_arousal_avg = ExtractDouble (row, "s_arousal_avg", 0.0);
          slot.drift_acc = ExtractDouble (row, "drift_mag", 0.0);

          // Load signal records for this WM slot (ordered)
          auto sig_rows = store.Execute (
              "SELECT embedding_id, timestamp, modality, mime, blob_id, "
              "       score, serial_position "
              "FROM signals WHERE memory_id = ? "
              "ORDER BY serial_position ASC",
              { slot.memory_id });
          for (const auto &sig_row : sig_rows)
            {
              SignalRecord rec;
              rec.timestamp
                  = static_cast<uint64_t> (ExtractInt64 (sig_row, "timestamp", 0));
              rec.modality = "";
              rec.mime = "";
              auto mod_it = sig_row.find ("modality");
              if (mod_it != sig_row.end () && mod_it->second.has_value ()
                  && mod_it->second.type () == typeid (std::string))
                {
                  rec.modality = std::any_cast<std::string> (mod_it->second);
                }
              auto mime_it = sig_row.find ("mime");
              if (mime_it != sig_row.end () && mime_it->second.has_value ()
                  && mime_it->second.type () == typeid (std::string))
                {
                  rec.mime = std::any_cast<std::string> (mime_it->second);
                }
              rec.score = ExtractDouble (sig_row, "score", 0.0);
              rec.serial_position
                  = static_cast<int> (ExtractInt64 (sig_row, "serial_position", 0));

              auto blob_it = sig_row.find ("blob_id");
              if (blob_it != sig_row.end () && blob_it->second.has_value ())
                {
                  rec.blob_id = store::BlobFromAny (blob_it->second);
                }
              if (!rec.blob_id.empty ())
                {
                  slot.blob_ids.push_back (rec.blob_id);
                  const bool text_record
                      = rec.modality == "text" || rec.mime == "text/plain";
                  if (text_record)
                    {
                      try
                        {
                          auto payload_rows = store.Execute (
                              "SELECT objstore_get(?1) AS payload",
                              { rec.blob_id });
                          if (!payload_rows.empty ()
                              && payload_rows[0].count ("payload") > 0)
                            {
                              const auto payload = store::BlobFromAny (
                                  payload_rows[0].at ("payload"));
                              if (!payload.empty ())
                                {
                                  rec.text_payload.assign (
                                      reinterpret_cast<const char *> (
                                          payload.data ()),
                                      payload.size ());
                                }
                            }
                        }
                      catch (...)
                        {
                        }
                    }
                }

              const auto sig_emb_id
                  = ExtractInt64 (sig_row, "embedding_id", 0);
              if (sig_emb_id > 0)
                {
                  auto sig_emb_rows = store.Execute (
                      "SELECT embedding FROM embeddings WHERE embedding_id = ?",
                      { sig_emb_id });
                  if (!sig_emb_rows.empty ())
                    {
                      auto it_emb = sig_emb_rows[0].find ("embedding");
                      if (it_emb != sig_emb_rows[0].end ()
                          && it_emb->second.has_value ())
                        {
                          rec.embedding = BlobToEigen (it_emb->second);
                        }
                    }
                }
              slot.signal_records.push_back (std::move (rec));
            }

          slot.persisted_signal_record_count = slot.signal_records.size ();
          slot.metadata_dirty = metadata_changed;
          slot.embedding_dirty = false;
          slot.signal_records_dirty = false;
          ctx.wm_slots.push_back (std::move (slot));
        }
      ctx.wm_slots_dirty = false;
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load working memory",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

void
LoadSoftAnchors (Store &store, ProcessorContext &ctx)
{
  try
    {
      auto rows = store.Execute (
          "SELECT anchor_id, status, semantic_centroid, entity_centroid, "
          "       full_centroid, semantic_radius, entity_radius, full_radius, "
          "       source_id, first_step, last_step, first_ts, last_ts, "
          "       last_boundary_id, anchor_strength, support_count, "
          "       contradiction_count, recent_memory_ids "
          "FROM soft_anchors "
          "WHERE status IN ('provisional', 'active', 'durable', 'decayed') "
          "ORDER BY last_step ASC");
      int max_numeric_id = 0;
      for (const auto &row : rows)
        {
          ProcessorContext::SoftAnchorState anchor;
          anchor.anchor_id = ExtractString (row, "anchor_id");
          if (anchor.anchor_id.empty ())
            {
              continue;
            }
          anchor.status = ExtractString (row, "status", "provisional");
          anchor.last_source_id = ExtractString (row, "source_id");
          auto semantic_it = row.find ("semantic_centroid");
          if (semantic_it != row.end () && semantic_it->second.has_value ())
            {
              anchor.semantic_centroid = BlobToEigen (semantic_it->second);
            }
          auto entity_it = row.find ("entity_centroid");
          if (entity_it != row.end () && entity_it->second.has_value ())
            {
              anchor.entity_centroid = BlobToEigen (entity_it->second);
            }
          auto full_it = row.find ("full_centroid");
          if (full_it != row.end () && full_it->second.has_value ())
            {
              anchor.full_centroid = BlobToEigen (full_it->second);
            }
          anchor.semantic_radius = ExtractDouble (row, "semantic_radius", 0.0);
          anchor.entity_radius = ExtractDouble (row, "entity_radius", 0.0);
          anchor.full_radius = ExtractDouble (row, "full_radius", 0.0);
          anchor.first_step
              = static_cast<int> (ExtractInt64 (row, "first_step", 0));
          anchor.last_step
              = static_cast<int> (ExtractInt64 (row, "last_step", 0));
          anchor.first_ts
              = static_cast<uint64_t> (ExtractInt64 (row, "first_ts", 0));
          anchor.last_ts
              = static_cast<uint64_t> (ExtractInt64 (row, "last_ts", 0));
          anchor.last_boundary_id = ExtractInt64 (row, "last_boundary_id", 0);
          anchor.anchor_strength = ExtractDouble (row, "anchor_strength", 0.0);
          anchor.support_count
              = static_cast<int> (ExtractInt64 (row, "support_count", 0));
          anchor.contradiction_count = static_cast<int> (
              ExtractInt64 (row, "contradiction_count", 0));
          anchor.recent_memory_ids = ParseMemoryIds (
              ExtractString (row, "recent_memory_ids"));

          const std::string prefix = "soft_anchor_";
          if (anchor.anchor_id.rfind (prefix, 0) == 0)
            {
              try
                {
                  max_numeric_id = std::max (
                      max_numeric_id,
                      std::stoi (anchor.anchor_id.substr (prefix.size ())));
                }
              catch (...)
                {
                }
            }

          ctx.soft_anchor_states.push_back (std::move (anchor));
        }
      ctx.soft_anchor_next_id = std::max (ctx.soft_anchor_next_id,
                                          max_numeric_id + 1);
      ctx.soft_anchor_last_state_count
          = static_cast<int> (ctx.soft_anchor_states.size ());
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load soft anchors",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

} // namespace

#if defined(CORTEXT_TESTING)
namespace testing
{
std::size_t
ProcessorRollbackSnapshotOwnerCountForTest ()
{
  return g_processor_rollback_snapshot_owner_count.load (
      std::memory_order_relaxed);
}

std::size_t
ProcessorRollbackSnapshotReuseCountForTest ()
{
  return g_processor_rollback_snapshot_reuse_count.load (
      std::memory_order_relaxed);
}
} // namespace testing
#endif

SignalProcessor::SignalProcessor (const Config &config,
                                  std::shared_ptr<Store> store,
                                  std::unique_ptr<IOperation> root_operation,
                                  std::shared_ptr<ObjectStore> object_store)
    : config_ (config),
      clock_ (config.clock ? config.clock : std::make_shared<SystemClock> ()),
      store_ (std::move (store)),
      object_store_ (std::move (object_store)),
      root_operation_ (std::make_unique<SnapshotOwningOperation> (
          std::move (root_operation))),
      context_ (std::make_unique<ProcessorContext> ())
{
  if (!config_.encoder)
    {
      throw std::invalid_argument (
          "SignalProcessor requires a non-null Encoder");
    }
  // Initialize rate observation window capacity derived from Stability knob.
  if (context_)
    {
      const double T = core::Clamp (config_.stability, 0.0, 1.0);
      const int cap = core::WRateSeconds (T);
      context_->write_rate_window_.SetCapacity (
          static_cast<size_t> (std::max (1, cap)));
      context_->recent_ids_lru_.SetCapacity (
          static_cast<size_t> (std::max (
              1, core::RecentRetrievedIdWindow (
                     config_.focus, config_.sensitivity, config_.stability))));
    }
  // Apply schema migrations exactly once during initialization.
  bool loaded_state = false;
  const auto now_ms = clock_->NowMillis ();
  if (store_)
    {
      cortext::store::ApplyMigrations (*store_);
      if (!object_store_)
        {
          object_store_ = std::make_shared<SqlObjectStore> (store_);
        }

      // Load persisted state for algorithm resumption (v2 schema)
      loaded_state = LoadState (*store_, *context_, config_);      // Unified state
      LoadPredictivePreActivationIds (*store_, *context_);
      LoadRetrievalSuppressionIds (*store_, *context_);
      LoadRetrievalSurfaceCache (*store_, *context_);
      LoadRecentContext (*store_, *context_, config_);
      LoadRecentScores (*store_, *context_, config_);
      LoadObservedRetentionHistory (
          *store_, *context_, now_ms,
          core::ObservedRetentionHistoryLimit (
              config_.focus, config_.sensitivity, config_.stability));
      LoadWorkingMemory (*store_, *context_, config_,
                         now_ms);                                  // From MEMORIES
      LoadSoftAnchors (*store_, *context_);
      // Accumulators are volatile staging state. Durable memories and working
      // memory are restored, but unfinished accumulator windows are discarded.
    }

  if (!loaded_state && context_)
    {
      SeedKnobDerivedStateDefaults (*context_, config_);
      context_->last_rate_timestamp = static_cast<uint64_t> (now_ms);
      context_->last_mood_ts = static_cast<uint64_t> (now_ms);
      operations::consolidation_throughput_state_internal::Reset (*context_);
    }
}

SignalProcessor::~SignalProcessor ()
{
  if (context_)
    {
      // SignalProcessor exclusively owns the process-lifetime registry entry.
      // Erasing it prevents a later allocation at the same address from
      // inheriting state from this processor or store.
      operations::historical_surface_search_cache_internal::Erase (*context_);
      operations::consolidation_throughput_state_internal::Erase (*context_);
    }
}

SignalProcessor::Output
SignalProcessor::Process (const Signal &signal)
{
  const auto t0 = std::chrono::steady_clock::now ();
  telemetry::ScopedSpan span ("cortext.process");
  const bool read_only = signal.retention == Retention::Ephemeral;
  const bool maintenance = signal.force_consolidation && !read_only;

  // Create transaction for this signal processing
  const auto tx_begin_start = std::chrono::steady_clock::now ();
  auto tx = store_ ? store_->Begin () : nullptr;
  std::unique_ptr<ObjectTransaction> object_tx;
  if (object_store_ && tx && !maintenance)
    {
      if (dynamic_cast<SqlObjectStore *> (object_store_.get ()) == nullptr)
        {
          object_tx = object_store_->Begin ();
        }
    }
  const auto tx_begin_end = std::chrono::steady_clock::now ();
  const auto snapshot_start = std::chrono::steady_clock::now ();
  auto &snapshot_owner
      = static_cast<SnapshotOwningOperation &> (*root_operation_);
  auto snapshot_lease = snapshot_owner.Acquire ();
  auto &rollback_snapshot = snapshot_lease.Get ();
  {
    auto detached_caches = DetachRebuildableProcessorCaches (*context_);
    try
      {
        rollback_snapshot.context = *context_;
        rollback_snapshot.caches = detached_caches;
        rollback_snapshot.consolidation_throughput
            = operations::consolidation_throughput_state_internal::Find (
                *context_);
        rollback_snapshot.initialized = true;
      }
    catch (...)
      {
        rollback_snapshot.initialized = false;
        RestoreRebuildableProcessorCaches (*context_, detached_caches);
        throw;
      }
    RestoreRebuildableProcessorCaches (*context_, detached_caches);
  }
  const auto snapshot_end = std::chrono::steady_clock::now ();

  OperationContext op_context (signal, *context_, config_, store_.get (),
                               object_tx.get ());
  if (tx)
    {
      op_context.AddOperationTiming ("SignalProcessor.begin_transaction",
                                     ElapsedMillis (tx_begin_start,
                                                    tx_begin_end));
    }
  op_context.AddOperationTiming ("SignalProcessor.snapshot_context",
                                 ElapsedMillis (snapshot_start,
                                                snapshot_end));
  auto rollback_object_tx = [&object_tx] {
    if (object_tx)
      {
        try
          {
            object_tx->Rollback ();
          }
        catch (...)
          {
          }
      }
  };
  auto restore_context_snapshot = [&] {
    operations::historical_surface_search_cache_internal::Erase (*context_);
    *context_ = std::move (rollback_snapshot.context);
    operations::consolidation_throughput_state_internal::Reset (
        *context_, rollback_snapshot.consolidation_throughput);
    RestoreRebuildableProcessorCaches (*context_, rollback_snapshot.caches);
    rollback_snapshot.initialized = false;
    if (store_)
      {
        LoadPredictivePreActivationIds (*store_, *context_);
        LoadRetrievalSuppressionIds (*store_, *context_);
        LoadHistoricalSurfaceSearchCache (*store_, *context_);
      }
  };
  auto restore_read_only_context_snapshot = [&] {
    *context_ = std::move (rollback_snapshot.context);
    operations::consolidation_throughput_state_internal::Reset (
        *context_, rollback_snapshot.consolidation_throughput);
    RestoreRebuildableProcessorCaches (*context_, rollback_snapshot.caches);
    rollback_snapshot.initialized = false;
  };
  std::optional<Output> read_only_output;

  try
    {
      auto op_start = std::chrono::steady_clock::now ();
      if (!maintenance)
        {
          op_context.SetCurrentOperationType ("StartNewEpisode");
          StartNewEpisode (tx.get (), signal.timestamp);
          op_context.AddOperationTiming (
              "SignalProcessor.start_new_episode",
              ElapsedMillis (op_start, std::chrono::steady_clock::now ()));
        }
      if (tx)
        {
          root_operation_->Execute (op_context, *tx);
        }

      if (read_only)
        {
          Output out;
          AssembleOutputMemories (op_context, out);
          AssembleOutputFields (op_context, out);
          rollback_object_tx ();
          if (tx)
            {
              tx->Rollback ();
            }
          restore_read_only_context_snapshot ();
          ApplyConsolidationHint (signal, config_, *context_, out);
          read_only_output = std::move (out);
        }
      else if (!maintenance && op_context.GetWriteDecision ())
        {
          context_->write_rate_window_.Record (signal.timestamp);
        }

      if (!read_only && !maintenance)
        {
          context_->last_signal_timestamp = signal.timestamp;
        }

      span.SetAttribute ("cortext.at_boundary", op_context.GetAtBoundary ());
      span.SetAttribute ("cortext.interrupt_allowed",
                         op_context.GetInterruptAllowed ());
      span.SetAttribute ("cortext.threshold_T_dynamic",
                         op_context.GetThresholdTDynamic ());
      span.SetAttribute ("cortext.threshold_hysteresis",
                         op_context.GetThresholdHysteresis ());
      span.SetAttribute ("cortext.effective_focus",
                         op_context.GetEffectiveFocus ());

      if (!read_only && !maintenance && op_context.ShouldFinalizeEpisode ())
        {
          op_context.SetCurrentOperationType ("FinalizeEpisode");
          op_start = std::chrono::steady_clock::now ();
          FinalizeEpisode (tx.get (), &op_context);
          op_context.AddOperationTiming (
              "SignalProcessor.finalize_episode",
              ElapsedMillis (op_start, std::chrono::steady_clock::now ()));
        }

      if (!read_only && !maintenance)
        {
          context_->signals_processed += 1;
        }

      // Persist state within the same transaction (v2 schema)
      if (tx && !read_only)
        {
          if (maintenance)
            {
              operations::consolidation_throughput_state_internal::Acknowledge (
                  *context_, context_->m_rate);
            }
          op_context.SetCurrentOperationType ("PersistState");
          op_start = std::chrono::steady_clock::now ();
          PersistState (*tx);           // Unified state
          op_context.AddOperationTiming (
              "SignalProcessor.persist_state",
              ElapsedMillis (op_start, std::chrono::steady_clock::now ()));
          if (!maintenance)
            {
              op_context.SetCurrentOperationType ("PersistWorkingMemory");
              op_start = std::chrono::steady_clock::now ();
              PersistWorkingMemory (*tx, false, &op_context);
              op_context.AddOperationTiming (
                  "SignalProcessor.persist_working_memory",
                  ElapsedMillis (op_start, std::chrono::steady_clock::now ()));
            }
          if (object_tx)
            {
              op_context.SetCurrentOperationType ("PersistObjects");
              op_start = std::chrono::steady_clock::now ();
              // External object stores cannot commit atomically with the DB.
              // Commit object content before the root DB commit so a later DB
              // commit failure leaves only content-addressed orphan objects,
              // not DB rows pointing to missing payloads. SqlObjectStore uses
              // the DB transaction fallback and does not need object_tx here.
              object_tx->Commit ();
              op_context.AddOperationTiming (
                  "SignalProcessor.persist_objects",
                  ElapsedMillis (op_start, std::chrono::steady_clock::now ()));
            }
          op_context.SetCurrentOperationType ("CommitTransaction");
          op_start = std::chrono::steady_clock::now ();
          tx->Commit ();
          op_context.AddOperationTiming (
              "SignalProcessor.commit_transaction",
              ElapsedMillis (op_start, std::chrono::steady_clock::now ()));
          if (signal.force_consolidation)
            {
              // Force persistence can prune or rewrite broad embedding sets,
              // so its transaction invalidates the private historical mirror.
              // Rebuild only after the authoritative commit succeeds; a failed
              // commit restores the exact pre-signal snapshot in the catch path.
              LoadRetrievalSurfaceCache (*store_, *context_);
              CheckpointSQLiteStore (
                  store_.get (), true, &op_context,
                  "SignalProcessor.sqlite_wal_checkpoint");
            }
          else
            {
              MaybeRunPassiveWalCheckpoint (store_.get (), signal,
                                            *context_, op_context);
            }
        }
    }
  catch (const internal::CancellationError &)
    {
      if (tx)
        {
          rollback_object_tx ();
          tx->Rollback ();
        }
      restore_context_snapshot ();
      throw;
    }
  catch (const std::exception &e)
    {
      if (tx)
        {
          rollback_object_tx ();
          tx->Rollback ();
        }
      restore_context_snapshot ();
      const std::string op_type = op_context.GetCurrentOperationType ();
      const std::string msg
          = "Process failed in " + op_type + ": " + e.what ();
      telemetry::LogError (
          "cortext.process_failed",
          { telemetry::Attribute::String (
                "cortext.operation_type",
                op_type),
            telemetry::Attribute::String ("cortext.error", e.what ()) });
      throw std::runtime_error (msg);
    }
  catch (...)
    {
      if (tx)
        {
          rollback_object_tx ();
          tx->Rollback ();
        }
      restore_context_snapshot ();
      const std::string op_type = op_context.GetCurrentOperationType ();
      const std::string msg
          = "Process failed in " + op_type + ": unknown error";
      telemetry::LogError (
          "cortext.process_failed",
          { telemetry::Attribute::String (
                "cortext.operation_type",
                op_type),
            telemetry::Attribute::String ("cortext.error", "unknown") });
      throw std::runtime_error (msg);
    }

  Output out;
  if (read_only_output.has_value ())
    {
      out = std::move (*read_only_output);
    }
  else
    {
      AssembleOutputMemories (op_context, out);
      AssembleOutputFields (op_context, out);
      ApplyConsolidationHint (signal, config_, *context_, out);
    }
  const auto t1 = std::chrono::steady_clock::now ();
  const double ms = ElapsedMillis (t0, t1);
  telemetry::RecordHistogram ("cortext.process_duration_ms", ms);
  telemetry::AddCounter ("cortext.signals_processed_total", 1);
  if (out.at_boundary)
    {
      telemetry::AddCounter ("cortext.at_boundary_total", 1);
    }
  if (out.interrupt_allowed)
    {
      telemetry::AddCounter ("cortext.interrupt_allowed_total", 1);
    }
  LogProcessTelemetry (op_context, out);
  span.SetStatusOk ();
  return out;
}

void
SignalProcessor::Flush ()
{
  telemetry::AddCounter ("cortext.flush_total", 1);
  if (!store_)
    {
      return;
    }
  ProcessorContext context_snapshot;
  DetachedProcessorCaches context_cache_snapshot;
  const auto consolidation_throughput_snapshot
      = operations::consolidation_throughput_state_internal::Find (*context_);
  {
    auto detached_caches = DetachRebuildableProcessorCaches (*context_);
    context_snapshot = *context_;
    context_cache_snapshot = detached_caches;
    RestoreRebuildableProcessorCaches (*context_, detached_caches);
  }
  auto restore_context_snapshot = [&] {
    *context_ = std::move (context_snapshot);
    operations::consolidation_throughput_state_internal::Reset (
        *context_, consolidation_throughput_snapshot);
    RestoreRebuildableProcessorCaches (*context_, context_cache_snapshot);
  };
  auto tx = store_->Begin ();
  std::unique_ptr<ObjectTransaction> object_tx;
  if (object_store_)
    {
      if (dynamic_cast<SqlObjectStore *> (object_store_.get ()) == nullptr)
        {
          object_tx = object_store_->Begin ();
        }
    }
  try
    {
      FinalizeEpisode (tx.get (), nullptr);
      PersistState (*tx);               // v2: Unified state
      PersistWorkingMemory (*tx, true); // v2: To MEMORIES
      const auto now_ms = clock_->NowMillis ();
      StartNewEpisode (tx.get (), static_cast<uint64_t> (now_ms));
      if (object_tx)
        {
          object_tx->Commit ();
        }
      tx->Commit ();
    }
  catch (...)
    {
      if (object_tx)
        {
          try
            {
              object_tx->Rollback ();
            }
          catch (...)
            {
            }
        }
      try
        {
          tx->Rollback ();
        }
      catch (...)
        {
        }
      restore_context_snapshot ();
      LoadHistoricalSurfaceSearchCache (*store_, *context_);
      throw;
    }
  LoadRetrievalSurfaceCache (*store_, *context_);
  CheckpointSQLiteStore (store_.get (), true, nullptr, nullptr);
}

void
SignalProcessor::StartNewEpisode (Transaction *tx, uint64_t start_ts)
{
  if (!context_ || start_ts == 0)
    {
      return;
    }
  if (context_->episode_start_ts != 0)
    {
      return;
    }
  context_->episode_start_ts = start_ts;

  if (tx)
    {
      tx->Execute (
          "INSERT OR IGNORE INTO episodes "
          "(episode_id, start_ts, end_ts, boundary_type, centroid, created_at) "
          "VALUES (?, ?, NULL, NULL, NULL, ?)",
          { static_cast<long long> (start_ts),
            static_cast<long long> (start_ts),
            static_cast<long long> (start_ts) });
    }
}

void
SignalProcessor::FinalizeEpisode (Transaction *tx,
                                  const OperationContext *op_context)
{
  telemetry::ScopedSpan span ("cortext.episode.finalize");

  if (context_ && context_->episode_start_ts != 0)
    {
      const uint64_t end_ts
          = op_context ? op_context->GetSignal ().timestamp
                       : clock_->NowMillis ();
      std::optional<std::string> boundary_type;
      std::optional<Eigen::VectorXf> centroid;
      if (op_context)
        {
          boundary_type = op_context->GetBoundaryType ();
          centroid = op_context->GetBoundaryCentroid ();
        }
      else
        {
          boundary_type = std::string ("explicit");
        }

      std::any boundary_type_any
          = boundary_type.has_value () ? std::any (*boundary_type)
                                       : std::any ();
      std::any centroid_any;
      if (centroid.has_value () && centroid->size () > 0)
        {
          centroid_any = ToFloatVector (*centroid);
        }
      else
        {
          centroid_any = std::any ();
        }

      if (tx)
        {
          tx->Execute (
              "UPDATE episodes "
              "SET end_ts = ?, boundary_type = ?, centroid = ? "
              "WHERE episode_id = ?",
              { static_cast<long long> (end_ts), boundary_type_any, centroid_any,
                static_cast<long long> (context_->episode_start_ts) });
        }
    }

  if (context_)
    {
      context_->episode_start_ts = 0;
    }

  // v2 schema: recent context and score windows derive from signals/embeddings
  // at restore time using F/S/T-derived runtime limits. No separate sliding
  // window tables need to be persisted.
  //
  // Note: State and WM are persisted by callers. Accumulators are volatile
  // staging state; unfinished windows are intentionally discarded on restart.

  telemetry::AddCounter ("cortext.episode_commit_total", 1);
  span.SetStatusOk ();

  // Maintain recent_context to last n_ctx(T) plus drift lag for drift metrics.
  if (context_)
    {
      const size_t keep
          = static_cast<size_t> (core::NCtx (config_.stability)
                                 + core::KCtx (config_.stability));
      auto &embs = context_->recent_context_embeddings;
      while (embs.size () > keep)
        {
          embs.pop_front ();
        }
    }
}

// v2 Schema: Unified state persistence to STATE table
void
SignalProcessor::PersistState (Transaction &tx)
{
  if (!context_)
    return;

  const auto wall_now_ms
      = static_cast<long long> (clock_->NowMillis ());
  const auto now_ms = context_->last_signal_timestamp > 0
                          ? static_cast<long long> (
                                context_->last_signal_timestamp)
                          : wall_now_ms;

  // Serialize mood_vector as raw binary BLOB (48 bytes = 6 doubles)
  std::vector<char> mood_blob (6 * sizeof (double));
  std::memcpy (mood_blob.data (), context_->mood_vector.data (),
               6 * sizeof (double));
  const std::vector<char> write_rate_blob = SerializeUint64Vector (
      context_->write_rate_window_.GetTimestamps ());
  const double wm_maintenance_cost
      = core::WMMaintenanceCostPerSlot (config_.sensitivity, config_.focus);
  const int wm_slot_count
      = static_cast<int> (context_->wm_slots.size ());
  const auto consolidation_throughput
      = operations::consolidation_throughput_state_internal::Find (*context_);

  // Get blender weights
  const double blender_fallback = core::BlendBootstrapFallback (
      config_.focus, config_.sensitivity, config_.stability);
  auto get_weight = [this, blender_fallback] (operations::Metric m) {
    auto it = context_->blender_state.find (m);
    return (it != context_->blender_state.end ()) ? it->second
                                                  : blender_fallback;
  };

  double w_relevance = get_weight (operations::Metric::relevance);
  double w_mismatch = get_weight (operations::Metric::mismatch);
  double w_surprise = get_weight (operations::Metric::surprise);
  double w_rarity = get_weight (operations::Metric::rarity);
  double w_drift = get_weight (operations::Metric::drift);
  double w_contradiction = get_weight (operations::Metric::contradiction);
  double w_utility = get_weight (operations::Metric::utility);
  double w_periphery = get_weight (operations::Metric::periphery);
  double w_coverage = get_weight (operations::Metric::coverage);
  double w_salience = get_weight (operations::Metric::salience);
  double w_valence = get_weight (operations::Metric::valence);
  double w_arousal = get_weight (operations::Metric::arousal);

  // Serialize matrices as BLOBs
  std::vector<float> P_blob;
  if (!context_->blender_P.empty ())
    P_blob = SerializeMatrix (context_->blender_P);

  // Insert unified state row
  tx.Execute (
      "INSERT OR REPLACE INTO state "
      "(id, signals_processed, "
      // Threshold state
      "theta_dynamic, theta_target, hysteresis, half_life, "
      // Focus state
      "weight_relevance, attention_width, coverage_gain_floor, mismatch_weight, "
      // Sensitivity state
      "weight_novelty, weight_surprise, weight_valence, weight_arousal, "
      "emotion_gain, score_gain, rate_target, "
      // Emotion state
      "emotion_intensity, valence, arousal, mood_vector, last_mood_ts, "
      "flashbulb_rate, "
      // Stability state
      "rate_decay, periphery_half_life, salience_half_life, drift_weight, retention_ema, "
      // Rate control
      "m_rate, rho_hat_prev, dt_ema, rate_ticks, last_rate_timestamp, reliability, "
      // Uncertainty
      "u_uncertainty, "
      "outcome_pred, neuromod_ach, neuromod_ne, neuromod_da, osc_phase, "
      // Embedding prediction
      "last_embedding, x_pred_ema, delta_half_life_adj, sustained_influence, "
      // Working memory
      "wm_maintenance_cost, wm_slot_count, wm_last_accepted, wm_last_chunked, "
      // Consolidation
      "last_consolidation_ts, consolidation_count, memories_since_consolidation, is_processing_signal, last_retrieval_ts, "
      "consolidation_rate_floor, consolidation_rate_peak, consolidation_rate_initialized, consolidation_rate_armed, "
      // Episode tracking
      "episode_start_ts, last_interrupt_tick, last_signal_timestamp, updated_at, "
      "write_rate_timestamps, "
      // Blender weights
      "w_relevance, w_mismatch, w_surprise, w_rarity, w_drift, w_contradiction, "
      "w_utility, w_periphery, w_coverage, w_salience, w_valence, w_arousal, "
      "blender_ready, blender_update_count, "
      // Blender matrices
      "blender_P_matrix) "
      "VALUES (1, ?, "
      "?, ?, ?, ?, "  // Threshold
      "?, ?, ?, ?, "  // Focus
      "?, ?, ?, ?, ?, ?, ?, "  // Sensitivity
      "?, ?, ?, ?, ?, ?, "  // Emotion
      "?, ?, ?, ?, ?, "  // Stability
      "?, ?, ?, ?, ?, ?, "  // Rate control
      "?, ?, ?, ?, ?, ?, "  // Uncertainty + modulators
      "?, ?, ?, ?, "  // Embedding prediction
      "?, ?, ?, ?, "  // Working memory
      "?, ?, ?, ?, ?, ?, ?, ?, ?, "  // Consolidation
      "?, ?, ?, ?, ?, "  // Episode tracking + write_rate_timestamps
      "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "  // Blender weights (12)
      "?, ?, "  // Blender ready/count
      "?)",  // Blender matrices
      {
        context_->signals_processed,
        // Threshold state
        context_->T_dynamic, context_->T_target, context_->hysteresis,
        context_->half_life,
        // Focus state
        context_->weight_relevance, context_->attention_width,
        context_->coverage_gain_floor, context_->mismatch_weight,
        // Sensitivity state
        context_->weight_novelty, context_->weight_surprise,
        context_->weight_valence, context_->weight_arousal,
        context_->emotion_gain, context_->score_gain, context_->rate_target,
        // Emotion state
        context_->emotion_intensity_ewma, context_->valence_ewma,
        context_->arousal_ewma, mood_blob,
        static_cast<long long> (context_->last_mood_ts),
        context_->flashbulb_rate_ewma,
        // Stability state
        context_->rate_decay, context_->periphery_half_life,
        context_->salience_half_life, context_->drift_weight,
        context_->retention_ema,
        // Rate control
        context_->m_rate, context_->rho_hat_prev, context_->dt_ema,
        context_->rate_ticks,
        static_cast<long long> (context_->last_rate_timestamp),
        context_->reliability,
        // Uncertainty
        context_->u_t,
        context_->outcome_pred,
        context_->neuromod_ach,
        context_->neuromod_ne,
        context_->neuromod_da,
        context_->osc_phase,
        // Embedding prediction
        context_->last_embedding.has_value ()
            ? std::any (ToFloatVector (*context_->last_embedding))
            : std::any (std::vector<float> ()),
        context_->x_pred_ema.has_value ()
            ? std::any (ToFloatVector (*context_->x_pred_ema))
            : std::any (std::vector<float> ()),
        context_->delta_half_life_adj, context_->sustained_influence,
        // Working memory
        wm_maintenance_cost,
        wm_slot_count,
        context_->wm_last_accepted ? 1 : 0,
        context_->wm_last_chunked ? 1 : 0,
        // Consolidation
        static_cast<long long> (context_->last_consolidation_ts),
        context_->consolidation_count,
        context_->memories_since_consolidation,
        context_->is_processing_signal ? 1 : 0,
        static_cast<long long> (context_->last_retrieval_ts),
        consolidation_throughput.floor, consolidation_throughput.peak,
        consolidation_throughput.initialized ? 1 : 0,
        consolidation_throughput.armed ? 1 : 0,
        // Episode tracking
        static_cast<long long> (context_->episode_start_ts),
        context_->last_interrupt_tick,
        static_cast<long long> (context_->last_signal_timestamp), now_ms,
        write_rate_blob.empty () ? std::any (std::vector<char> ())
                                 : std::any (write_rate_blob),
        // Blender weights
        w_relevance, w_mismatch, w_surprise, w_rarity, w_drift, w_contradiction,
        w_utility, w_periphery, w_coverage, w_salience, w_valence, w_arousal,
        // Blender ready/count
        context_->blender_ready ? 1 : 0, context_->blender_update_count,
        // Blender matrices
        P_blob.empty () ? std::any (std::vector<float> ()) : std::any (P_blob)
      });
}

// v2 Schema: Persist working memory to MEMORIES table (kind='WORKING')
void
SignalProcessor::PersistWorkingMemory (Transaction &tx, bool force,
                                       OperationContext *op_context)
{
  if (!context_)
    return;

  if (op_context == nullptr)
    {
      // The process-wide historical registry is not part of the processor
      // snapshot. Rebuild it from authoritative SQL after either commit or
      // rollback of any flush-time embedding mutation.
      operations::historical_surface_search_cache_internal::Erase (*context_);
    }

  auto add_timing
      = [op_context] (const char *name,
                      std::chrono::steady_clock::time_point start) {
          if (!op_context)
            {
              return;
            }
          op_context->AddOperationTiming (
              name, ElapsedMillis (start, std::chrono::steady_clock::now ()));
        };

  if (!force && !context_->wm_slots_dirty)
    {
      bool has_dirty_slot = false;
      for (const auto &slot : context_->wm_slots)
        {
          if (slot.metadata_dirty || slot.embedding_dirty
              || slot.signal_records_dirty || slot.memory_id <= 0)
            {
              has_dirty_slot = true;
              break;
            }
        }
      if (!has_dirty_slot)
        {
          return;
        }
    }

  const auto wall_now_ms
      = static_cast<long long> (clock_->NowMillis ());
  const auto now_ms = context_->last_signal_timestamp > 0
                          ? static_cast<long long> (
                                context_->last_signal_timestamp)
                          : wall_now_ms;

  // Close any stale WM rows not represented by active slots. This only needs
  // to run when the active slot set changed or when a caller forces a flush.
  std::vector<long long> active_ids;
  active_ids.reserve (context_->wm_slots.size ());
  for (const auto &slot : context_->wm_slots)
    {
      if (slot.strength <= 0.0 || slot.embedding.size () == 0)
        {
          continue;
        }
      if (slot.memory_id > 0)
        {
          active_ids.push_back (slot.memory_id);
        }
    }

  if (force || context_->wm_slots_dirty)
    {
      const auto t_close_stale_start = std::chrono::steady_clock::now ();
      if (active_ids.empty ())
        {
          tx.Execute (
              "UPDATE memories SET end_ts = ? "
              "WHERE kind = 'WORKING' AND end_ts IS NULL",
              { now_ms });
        }
      else
        {
          std::string query
              = "UPDATE memories SET end_ts = ? "
                "WHERE kind = 'WORKING' AND end_ts IS NULL "
                "AND memory_id NOT IN ("
                + Placeholders (active_ids.size ()) + ")";
          std::vector<std::any> params;
          params.reserve (1 + active_ids.size ());
          params.push_back (now_ms);
          for (const auto id : active_ids)
            {
              params.push_back (id);
            }
          tx.Execute (query, params);
        }
      add_timing ("SignalProcessor.wm_close_stale",
                  t_close_stale_start);
    }

  const bool should_incremental_prune
      = !force && context_->wm_slots_dirty
        && context_->signals_processed > 0
        && ((context_->signals_processed
             % kClosedWorkingMemoryIncrementalPruneIntervalSignals)
            == 0);
  if (force || should_incremental_prune)
    {
      if (force)
        {
          // Force pruning can delete a broad set of embedding rows. Exact
          // incremental ownership is not available here, so fail closed.
          operations::historical_surface_search_cache_internal::Erase (
              *context_);
        }
      const int prune_batch = force ? kClosedWorkingMemoryPruneBatch
                                    : kClosedWorkingMemoryIncrementalPruneBatch;
      const auto t_prune_select_start = std::chrono::steady_clock::now ();
      auto stale_rows = tx.Execute (
          "SELECT memory_id FROM memories "
          "WHERE kind = 'WORKING' AND end_ts IS NOT NULL "
          "ORDER BY end_ts ASC, memory_id ASC "
          "LIMIT ?",
          { static_cast<long long> (prune_batch) });
      auto stale_ids = ExtractInt64Column (stale_rows, "memory_id");
      add_timing ("SignalProcessor.wm_prune_select",
                  t_prune_select_start);
      const auto t_prune_delete_start = std::chrono::steady_clock::now ();
      DeleteStaleWorkingMemoryRows (tx, stale_ids, force);
      add_timing ("SignalProcessor.wm_prune_delete",
                  t_prune_delete_start);
      if (force)
        {
          const auto t_orphan_delete_start = std::chrono::steady_clock::now ();
          DeleteUnreferencedEmbeddings (tx);
          add_timing ("SignalProcessor.wm_prune_orphan_embeddings",
                      t_orphan_delete_start);
        }
    }

  const double working_source_reliability = core::SourceReliabilityPrior (
      config_.focus, config_.sensitivity, config_.stability);
  const double working_stability = core::MemoryInitialStabilityPolicy (
      config_.focus, config_.sensitivity, config_.stability);

  bool all_slots_clean = true;

  // Upsert current slots as MEMORIES with kind='WORKING'. Clean slots are
  // left untouched; passive maintenance marks changed metadata dirty.
  for (auto &slot : context_->wm_slots)
    {
      std::string failure_stage = "slot_precheck";
      if (slot.strength <= 0.0)
        {
          slot.memory_id = 0;
          slot.embedding_id = 0;
          continue;
        }
      if (slot.embedding.size () == 0)
        {
          slot.memory_id = 0;
          slot.embedding_id = 0;
          continue;
        }

      bool needs_memory_row
          = force || slot.metadata_dirty || slot.embedding_dirty
            || slot.memory_id <= 0;
      bool needs_signal_rows
          = force || slot.signal_records_dirty || slot.memory_id <= 0;
      if (!needs_memory_row && !needs_signal_rows)
        {
          continue;
        }

      try
        {
          const auto ts_ms
              = internal::WorkingMemorySecondsToMillis (slot.last_ts);
          const auto strength_ts_ms = internal::WorkingMemorySecondsToMillis (
              slot.strength_ts > 0.0 ? slot.strength_ts : slot.last_ts);
          const auto slot_created_at
              = slot.start_ts > 0
                    ? static_cast<long long> (slot.start_ts)
                    : static_cast<long long> (ts_ms);
          const auto slot_embedding_created_at
              = ts_ms > 0 ? static_cast<long long> (ts_ms) : now_ms;

          long long embedding_id = slot.embedding_id;
          if (slot.embedding_dirty || embedding_id <= 0)
            {
              failure_stage = "insert_slot_embedding";
              const std::vector<float> emb_vec = ToFloatVector (slot.embedding);
              const auto t_sql_start = std::chrono::steady_clock::now ();
              tx.Execute (
                  "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
                  { emb_vec, slot_embedding_created_at });
              auto emb_rows
                  = tx.Execute ("SELECT last_insert_rowid() AS id", {});
              add_timing ("SignalProcessor.wm_insert_slot_embedding",
                          t_sql_start);
              embedding_id
                  = emb_rows.empty () ? 0 : ExtractInt64 (emb_rows[0], "id", 0);
              if (embedding_id > 0)
                {
                  operations::historical_surface_search_cache_internal::Append (
                      *context_,
                      { embedding_id, 0, 0, std::string (), std::string (),
                        slot.embedding });
                }
            }
          if (embedding_id == 0)
            {
              all_slots_clean = false;
              continue;
            }

          long long memory_id = slot.memory_id;
          if (memory_id > 0)
            {
              if (needs_memory_row)
                {
                  failure_stage = "update_slot_memory";
                  const auto t_sql_start = std::chrono::steady_clock::now ();
                  tx.Execute (
                      "UPDATE memories SET "
                      "embedding_id = ?, source_id = ?, modality = ?, "
                      "start_ts = ?, n_signals = ?, s_max = ?, s_avg = ?, "
                      "s_emotion_max = ?, s_arousal_avg = ?, drift_mag = ?, "
                      "strength = ?, source_reliability = ?, stability = ?, "
                      "last_access = ?, strength_updated_at = ?, end_ts = NULL "
                      "WHERE memory_id = ? AND kind = 'WORKING'",
                      { embedding_id,
                        slot.source_id.empty () ? std::string ("unknown")
                                                : slot.source_id,
                        slot.modality, slot.start_ts, slot.n_signals,
                        slot.s_max, slot.s_avg, slot.s_emotion_max,
                        slot.s_arousal_avg, slot.drift_acc, slot.strength,
                        working_source_reliability, working_stability, ts_ms,
                        strength_ts_ms, memory_id });
                  add_timing ("SignalProcessor.wm_update_slot_memory",
                              t_sql_start);
                }
            }
          else
            {
              failure_stage = "insert_slot_memory";
              const auto t_sql_start = std::chrono::steady_clock::now ();
              tx.Execute (
                  "INSERT INTO memories "
                  "(embedding_id, source_id, kind, modality, start_ts, n_signals, "
                  " s_max, s_avg, s_emotion_max, s_arousal_avg, drift_mag, "
                  " strength, source_reliability, stability, last_access, "
                  " strength_updated_at, created_at) "
                  "VALUES (?, ?, 'WORKING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  { embedding_id,
                    slot.source_id.empty () ? std::string ("unknown")
                                            : slot.source_id,
                    slot.modality, slot.start_ts, slot.n_signals, slot.s_max,
                    slot.s_avg, slot.s_emotion_max, slot.s_arousal_avg,
                    slot.drift_acc, slot.strength, working_source_reliability,
                    working_stability, ts_ms, strength_ts_ms, slot_created_at });

              auto mem_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
              add_timing ("SignalProcessor.wm_insert_slot_memory",
                          t_sql_start);
              memory_id
                  = mem_rows.empty () ? 0 : ExtractInt64 (mem_rows[0], "id", 0);
              if (memory_id == 0)
                {
                  all_slots_clean = false;
                  continue;
                }
              slot.memory_id = memory_id;
              needs_signal_rows = true;
            }
          slot.embedding_id = embedding_id;

          // Keep per-slot signals in sync with the current slot state.
          if (needs_signal_rows)
            {
              std::size_t first_record_to_persist = 0;
              if (!force
                  && slot.persisted_signal_record_count
                         <= slot.signal_records.size ())
                {
                  first_record_to_persist
                      = slot.persisted_signal_record_count;
                }
              else
                {
                  failure_stage = "clear_slot_signals";
                  const auto t_sql_start = std::chrono::steady_clock::now ();
                  tx.Execute ("DELETE FROM signals WHERE memory_id = ?",
                              { memory_id });
                  add_timing ("SignalProcessor.wm_clear_slot_signals",
                              t_sql_start);
                }

              for (std::size_t rec_index = first_record_to_persist;
                   rec_index < slot.signal_records.size (); ++rec_index)
                {
                  const auto &rec = slot.signal_records[rec_index];
                  try
                    {
                      long long signal_embedding_id = embedding_id;
                      if (rec.embedding.size () > 0
                          && rec.embedding.size () == slot.embedding.size ())
                        {
                          failure_stage = "insert_signal_embedding";
                          const std::vector<float> sig_vec
                              = ToFloatVector (rec.embedding);
                          const auto t_sql_start
                              = std::chrono::steady_clock::now ();
                          tx.Execute (
                              "INSERT INTO embeddings (embedding, created_at) "
                              "VALUES (?, ?)",
                              { sig_vec,
                                static_cast<long long> (rec.timestamp) });
                          auto sig_rows
                              = tx.Execute ("SELECT last_insert_rowid() AS id", {});
                          if (!sig_rows.empty ())
                            {
                              signal_embedding_id = ExtractInt64 (
                                  sig_rows[0], "id", embedding_id);
                            }
                          if (signal_embedding_id > 0)
                            {
                              operations::
                                  historical_surface_search_cache_internal::
                                      Append (
                                          *context_,
                                          { signal_embedding_id, 0, 0,
                                            std::string (), std::string (),
                                            rec.embedding });
                            }
                          add_timing (
                              "SignalProcessor.wm_insert_signal_embedding",
                              t_sql_start);
                        }

                      failure_stage = "insert_signal_row";
                      const auto t_sql_start
                          = std::chrono::steady_clock::now ();
                      tx.Execute (
                          "INSERT INTO signals "
                          "(memory_id, source_id, embedding_id, timestamp, "
                          "modality, mime, blob_id, serial_position, score, "
                          "created_at) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          { memory_id,
                            slot.source_id.empty () ? std::string ("unknown")
                                                    : slot.source_id,
                            signal_embedding_id,
                            static_cast<long long> (rec.timestamp),
                            rec.modality, rec.mime,
                            rec.blob_id.empty () ? std::any ()
                                                 : std::any (rec.blob_id),
                            static_cast<long long> (rec.serial_position),
                            rec.score, static_cast<long long> (rec.timestamp) });
                      add_timing ("SignalProcessor.wm_insert_signal_row",
                                  t_sql_start);
                    }
                  catch (const std::exception &)
                    {
                      // Working-memory signal rows are auxiliary. If one record
                      // is malformed, preserve the slot itself and continue.
                      continue;
                    }
                }
              slot.persisted_signal_record_count
                  = slot.signal_records.size ();
            }

          slot.metadata_dirty = false;
          slot.embedding_dirty = false;
          slot.signal_records_dirty = false;
        }
      catch (const std::exception &e)
        {
          all_slots_clean = false;
          telemetry::LogWarn (
              "Failed to persist working-memory slot",
              { telemetry::Attribute::String ("component", "signal_processor"),
                telemetry::Attribute::String ("stage", failure_stage),
                telemetry::Attribute::String (
                    "source_id",
                    slot.source_id.empty () ? std::string ("unknown")
                                            : slot.source_id),
                telemetry::Attribute::Int64 (
                    "memory_id", static_cast<std::int64_t> (slot.memory_id)),
                telemetry::Attribute::Int64 (
                    "slot_embedding_dim",
                    static_cast<std::int64_t> (slot.embedding.size ())),
                telemetry::Attribute::Int64 (
                    "signal_record_count",
                    static_cast<std::int64_t> (slot.signal_records.size ())),
                telemetry::Attribute::String ("error", e.what ()) });
          continue;
        }
    }

  if (all_slots_clean)
    {
      context_->wm_slots_dirty = false;
    }
}

} // namespace cortext
