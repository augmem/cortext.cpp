/// @file memory_system_ablation_bench.cpp
/// @brief 5 system-level memory ablation studies testing component interactions.
///
/// Study 1: Consolidation-Eviction Race
/// Study 2: End-to-End Fact Emergence
/// Study 3: Memory-Fact Co-Decay
/// Study 4: Retrieval-Reinforcement Feedback Loop
/// Study 5: Cross-Consolidation Fact Inheritance

#include "../../src/operations/eviction_ablation.hpp"
#include "../../src/operations/retrieval_debug_state.hpp"
#include "../../src/operations/temporal_retrieval.hpp"
#include "../../src/store/facts.hpp"
#include "include/benchmark_text_encoder.hpp"

#include <cortext/core/knobs.hpp>
#include <cortext/operations/consolidation.hpp>
#include <cortext/operations/consolidation_cluster.hpp>
#include <cortext/operations/consolidation_summarize.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/operations/memory_strength.hpp>
#include <cortext/operations/process_extraction_results.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/signal.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <any>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace
{
std::string g_models_dir = "models";

// =========================================================================
// Shared infrastructure
// =========================================================================

using BenchEncoder = cortext::benchmark::EmbeddingGemmaBenchEncoder;

std::vector<float>
ToFloatVec (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

cortext::Signal
MakeSignal (const Eigen::VectorXf &embedding, std::uint64_t ts,
            const std::string &source_id = "bench")
{
  cortext::Signal s;
  s.embedding = embedding;
  s.timestamp = ts;
  s.source_id = source_id;
  return s;
}

cortext::Signal
MakeConsolidationSignal (const Eigen::VectorXf &embedding, std::uint64_t ts)
{
  auto signal = MakeSignal (embedding, ts, "bench/consolidation");
  signal.consolidation_mode = cortext::ConsolidationMode::Shallow;
  return signal;
}

cortext::ProcessorContext
MakeProcessorContext (double T)
{
  cortext::ProcessorContext pctx;
  pctx.half_life = cortext::core::BaseHalfLifePrior (T);
  // Default: consolidation has never run. Eviction is gated until
  // consolidation sets this to a real timestamp.
  pctx.last_consolidation_ts = 0;
  return pctx;
}

std::string
BuildExtractionSignalText (const cortext::operations::ExtractionResult &result)
{
  std::string text = result.summary_id;
  for (const auto &fact : result.facts)
    {
      if (!text.empty ())
        {
          text += " ";
        }
      text += fact.subject + " " + fact.predicate + " " + fact.object;
    }
  if (text.empty ())
    {
      text = "benchmark extraction result";
    }
  return text;
}

void
SeedMemory (cortext::Store &store, long long id, double strength,
            long long created_at, const Eigen::VectorXf &embedding)
{
  auto vec = ToFloatVec (embedding);
  store.Execute (
      "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { id, vec, created_at });
  store.Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, trace_fast, trace_med, trace_slow, trace_ultra, "
      "use_frequency, stability, connectivity, drift_mag, "
      "influence, sustained_influence, contextual_gain, redundancy, "
      "pre_activation, lability_state, suppression_count, created_at, "
      "last_access) "
      "VALUES(?, ?, 'bench', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, "
      "?, ?, ?, ?, ?, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "
      "?, ?)",
      { id, id, created_at, strength, strength, strength * 0.5,
        strength * 0.2, strength * 0.05, created_at, created_at });
}

void
SeedSummaryMemory (cortext::Store &store, long long embedding_id,
                   long long memory_id, const std::string &summary_id,
                   long long start_ts, const Eigen::VectorXf &embedding)
{
  store.Execute ("DELETE FROM embeddings WHERE embedding_id = ?",
                 { embedding_id });
  store.Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?)",
      { embedding_id, ToFloatVec (embedding), start_ts });
  store.Execute ("DELETE FROM memories WHERE memory_id = ?", { memory_id });
  store.Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, "
      "kind, start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, ?, 'ASSOCIATION', ?, 1, 'text', 0.5, 0.5, 1.0, ?)",
      { memory_id, embedding_id, summary_id, start_ts, start_ts });
}

void
InsertFactAssertion (cortext::Store &store, long long fact_id,
                     const std::string &subject, const std::string &predicate,
                     const std::string &object, long long recorded_at,
                     double confidence, long long summary_memory_id)
{
  store.Execute (
      "INSERT INTO fact_assertions "
      "(fact_id, subject, predicate, object, canonical_subject, "
      "canonical_predicate, "
      "canonical_object, recorded_at_ts, confidence, summary_memory_id, "
      "created_at) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      { fact_id, subject, predicate, object, subject, predicate, object,
        recorded_at, confidence, summary_memory_id, recorded_at });
}

void
LinkFactEvidence (cortext::Store &store, long long fact_id, long long memory_id,
                  const std::string &evidence_type, double support_weight)
{
  store.Execute (
      "INSERT INTO fact_evidence "
      "(fact_id, source_memory_id, evidence_type, support_weight) "
      "VALUES (?, ?, ?, ?)",
      { fact_id, memory_id, evidence_type, support_weight });
}

void
InsertFactCache (cortext::Store &store, long long fact_id,
                 long long embedding_id, const std::string &fact_text,
                 bool is_current, long long recorded_at_ts, long long updated_at)
{
  store.Execute (
      "INSERT INTO fact_cache "
      "(fact_id, embedding_id, fact_text, is_current, recorded_at_ts, "
      "updated_at) "
      "VALUES (?, ?, ?, ?, ?, ?)",
      { fact_id, embedding_id, fact_text, is_current ? 1LL : 0LL,
        recorded_at_ts, updated_at });
}

long long
CountRows (cortext::Store &store, const std::string &table)
{
  auto rows
      = store.Execute ("SELECT COUNT(*) AS cnt FROM " + table, {});
  return std::any_cast<long long> (rows[0].at ("cnt"));
}

long long
CountMemories (cortext::Store &store)
{
  auto rows = store.Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE kind = 'LONG_TERM'", {});
  return std::any_cast<long long> (rows[0].at ("cnt"));
}

long long
CountMemoryById (cortext::Store &store, long long id)
{
  auto rows = store.Execute (
      "SELECT COUNT(*) AS cnt FROM memories WHERE memory_id = ?", { id });
  return std::any_cast<long long> (rows[0].at ("cnt"));
}

double
GetStrength (cortext::Store &store, long long id)
{
  auto rows = store.Execute (
      "SELECT strength FROM memories WHERE memory_id = ?", { id });
  if (rows.empty ())
    return -1.0;
  return std::any_cast<double> (rows[0].at ("strength"));
}

void
RunExtraction (cortext::Store &store, cortext::ProcessorContext &pctx,
               BenchEncoder &encoder,
               cortext::operations::ExtractionResult result,
               std::uint64_t signal_ts, double sensitivity = 0.5,
               double stability = 0.5)
{
  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = sensitivity;
  cfg.stability = stability;
  cfg.encoder = &encoder;

  const auto signal_embedding
      = encoder.EncodeTextEigen (BuildExtractionSignalText (result));
  pctx.pending_extraction_results.push_back (std::move (result));
  cortext::OperationContext ctx (MakeSignal (signal_embedding, signal_ts), pctx,
                                 cfg, &store);
  cortext::operations::ProcessExtractionResults op;
  auto tx = store.Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();
}

void
RunStrengthUpdate (
    cortext::Store &store, BenchEncoder &encoder,
    const cortext::SignalProcessor::Config &cfg,
    const std::vector<cortext::OperationContext::MemoryUsageEvent> &events,
    std::uint64_t signal_ts, cortext::ProcessorContext &pctx)
{
  auto signal = MakeSignal (
      encoder.EncodeTextEigen ("memory system ablation update pulse"),
      signal_ts);
  cortext::OperationContext ctx (signal, pctx, cfg, &store);
  ctx.SetMemoryUsageEvents (events);

  auto tx = store.Begin ();
  cortext::operations::UpdateMemoryStrength op;
  op.Execute (ctx, *tx);
  tx->Commit ();
}

void
MaintainFact (cortext::Store &store, BenchEncoder &encoder, std::uint64_t ts,
              long long fact_id)
{
  auto tx = store.Begin ();
  cortext::store::MaintainFactLifecycle (*tx, &encoder, ts, { fact_id });
  tx->Commit ();
}

class ForceRetrievalGateOp : public cortext::IOperation
{
public:
  void
  Execute (cortext::OperationContext &ctx,
           cortext::Transaction & /*tx*/) const override
  {
    ctx.SetShouldCheckRetrieval (true);
    auto &p_ctx = ctx.GetProcessorContext ();
    if (p_ctx.memory_stream.empty ())
      {
        p_ctx.memory_stream.push_back (ctx.GetSignal ().embedding);
      }
    auto &acc = p_ctx.accumulator_states[ctx.GetSignal ().source_id];
    acc.mu_acc = ctx.GetSignal ().embedding;
    acc.c_t = ctx.GetSignal ().embedding;
  }
};

long long
LookupFactId (cortext::Store &store, const std::string &subject,
              const std::string &predicate, const std::string &object)
{
  auto rows = store.Execute (
      "SELECT fact_id FROM fact_assertions "
      "WHERE canonical_subject = ? AND canonical_predicate = ? "
      "AND canonical_object = ?",
      { subject, predicate, object });
  if (rows.empty ())
    return 0;
  if (rows[0].at ("fact_id").type () == typeid (long long))
    return std::any_cast<long long> (rows[0].at ("fact_id"));
  return std::any_cast<int> (rows[0].at ("fact_id"));
}

double
SafeRatio (double num, double den)
{
  return den <= 0.0 ? 0.0 : num / den;
}

// =========================================================================
// Study 1: Consolidation-Eviction Race
// =========================================================================

struct Study1Metrics
{
  std::string config_name;
  long long candidates_available = 0;
  long long facts_extracted = 0;
  long long evidence_links_valid = 0;
};

Study1Metrics
RunStudy1Config (const std::string &name, int consolidate_at_step)
{
  Study1Metrics metrics;
  metrics.config_name = name;

  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  BenchEncoder encoder (g_models_dir);
  auto pctx = MakeProcessorContext (0.5);

  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.encoder = &encoder;

  // Seed 30 memories at strength=1.0
  constexpr int kCount = 30;
  constexpr long long kIntervalMs = 60000;
  const auto base_emb = encoder.EncodeTextEigen (
      "Alice caregiver Emily evidence cluster");
  for (int i = 1; i <= kCount; ++i)
    {
      SeedMemory (*store, i, 1.0, 0LL, base_emb);
    }

  // Run decay at 60-second intervals, consolidate at the specified step
  constexpr int kTotalSteps = 300;
  bool consolidated = false;

  for (int step = 1; step <= kTotalSteps; ++step)
    {
      const auto ts = static_cast<std::uint64_t> (step * kIntervalMs);

      // Build events for all living memories (all unused)
      std::vector<cortext::OperationContext::MemoryUsageEvent> events;
      for (int i = 1; i <= kCount; ++i)
        {
          if (CountMemoryById (*store, i) > 0)
            events.push_back ({ static_cast<long long> (i), false });
        }
      if (!events.empty ())
        RunStrengthUpdate (*store, encoder, cfg, events, ts, pctx);

      // At the designated step, run consolidation
      if (step == consolidate_at_step && !consolidated)
        {
          consolidated = true;

          // Count available candidates (surviving memories)
          // With consolidation-gated eviction, all memories should still
          // be alive since last_consolidation_ts was 0 (no prior
          // consolidation), preventing any eviction.
          metrics.candidates_available = CountMemories (*store);

          // Set last_consolidation_ts so subsequent decay steps can evict
          // memories that existed before this consolidation.
          pctx.last_consolidation_ts = ts;

          if (metrics.candidates_available > 0)
            {
              // Run ScoreConsolidation + ConsolidationCluster
              auto signal = MakeConsolidationSignal (
                  encoder.EncodeTextEigen ("caregiver consolidation summary"),
                  ts);
              cortext::OperationContext consol_ctx (signal, pctx, cfg,
                                                   store.get ());
              consol_ctx.SetConsolidationShouldStart (true);

              {
                auto tx = store->Begin ();
                cortext::operations::ScoreConsolidation score_op;
                score_op.Execute (consol_ctx, *tx);
                tx->Commit ();
              }
              {
                auto tx = store->Begin ();
                cortext::operations::ConsolidationCluster cluster_op;
                cluster_op.Execute (consol_ctx, *tx);
                tx->Commit ();
              }

              // For each cluster, create a summary memory and inject
              // ExtractionResult
              const auto &clusters = consol_ctx.GetConsolidationClusters ();
              long long summary_base_id = 10000LL;
              for (const auto &cluster : clusters)
                {
                  const long long summary_emb_id = summary_base_id;
                  const long long summary_mem_id = summary_base_id;
                  const std::string summary_id
                      = "summary-race-" + std::to_string (summary_base_id);

                  // Create summary memory
                  Eigen::VectorXf centroid
                      = Eigen::VectorXf::Zero (encoder.Dimension ());
                  if (!cluster.centroid.empty ())
                    {
                      centroid = Eigen::Map<const Eigen::VectorXf> (
                          cluster.centroid.data (),
                          static_cast<Eigen::Index> (cluster.centroid.size ()));
                    }
                  SeedSummaryMemory (*store, summary_emb_id, summary_mem_id,
                                     summary_id, static_cast<long long> (ts),
                                     centroid);

                  // Create derived_from edges
                  for (const auto &emb_id : cluster.embedding_ids)
                    {
                      store->Execute (
                          "INSERT INTO associations(source_memory_id, "
                          "target_memory_id, edge_type) "
                          "VALUES(?, ?, 'derived_from')",
                          { summary_mem_id, emb_id });
                    }

                  // Inject extraction result with a fact
                  cortext::operations::ExtractionResult extraction;
                  extraction.summary_id = summary_id;
                  extraction.facts.push_back (
                      { "alice", "caregiver", "emily", 0.9, std::nullopt,
                        std::nullopt });
                  RunExtraction (*store, pctx, encoder,
                                 std::move (extraction), ts);

                  ++summary_base_id;
                }
            }
        }
    }

  // Count facts extracted
  auto fact_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM fact_assertions", {});
  metrics.facts_extracted
      = std::any_cast<long long> (fact_rows[0].at ("cnt"));

  // Count valid evidence links (where source memory still exists)
  auto evidence_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM fact_evidence fe "
      "JOIN memories m ON fe.source_memory_id = m.memory_id",
      {});
  metrics.evidence_links_valid
      = std::any_cast<long long> (evidence_rows[0].at ("cnt"));

  return metrics;
}

// =========================================================================
// Study 2: End-to-End Fact Emergence
// =========================================================================

struct Study2Metrics
{
  std::string config_name;
  long long summaries_created = 0;
  long long facts_extracted = 0;
  long long fact_predicates = 0;
};

Study2Metrics
RunStudy2Config (const std::string &name, bool do_consolidation,
                 bool do_extraction)
{
  Study2Metrics metrics;
  metrics.config_name = name;

  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  BenchEncoder encoder (g_models_dir);
  auto pctx = MakeProcessorContext (0.5);

  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.0;  // Low F → min_cluster_size=3, enables small groups
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.encoder = &encoder;

  // Seed 20 memories with controlled embeddings
  // IDs 1-5: caregiver cluster
  for (int i = 1; i <= 5; ++i)
    SeedMemory (*store, i, 1.0, 0LL,
                encoder.EncodeTextEigen ("Alice caregiver Emily family support"));
  // IDs 6-10: location cluster
  for (int i = 6; i <= 10; ++i)
    SeedMemory (*store, i, 1.0, 0LL,
                encoder.EncodeTextEigen ("Alice lives on Oak Street in Denver"));
  // IDs 11-15: medication cluster
  for (int i = 11; i <= 15; ++i)
    SeedMemory (
        *store, i, 1.0, 0LL,
        encoder.EncodeTextEigen ("Alice takes metformin for diabetes care"));
  // IDs 16-20: noise
  for (int i = 16; i <= 20; ++i)
    SeedMemory (
        *store, i, 1.0, 0LL,
        encoder.EncodeTextEigen ("Weather report and grocery list unrelated note"));

  if (!do_consolidation)
    {
      return metrics;
    }

  // Force consolidation
  const std::uint64_t ts = 100000ULL;
  auto signal = MakeConsolidationSignal (
      encoder.EncodeTextEigen ("memory system consolidation sweep"), ts);
  cortext::OperationContext consol_ctx (signal, pctx, cfg, store.get ());
  consol_ctx.SetConsolidationShouldStart (true);

  {
    auto tx = store->Begin ();
    cortext::operations::ScoreConsolidation score_op;
    score_op.Execute (consol_ctx, *tx);
    tx->Commit ();
  }
  {
    auto tx = store->Begin ();
    cortext::operations::ConsolidationCluster cluster_op;
    cluster_op.Execute (consol_ctx, *tx);
    tx->Commit ();
  }

  const auto &clusters = consol_ctx.GetConsolidationClusters ();
  metrics.summaries_created = static_cast<long long> (clusters.size ());

  if (!do_extraction)
    {
      return metrics;
    }

  // For each cluster, determine which fact to inject based on membership
  long long summary_base_id = 10000LL;
  for (const auto &cluster : clusters)
    {
      const long long summary_emb_id = summary_base_id;
      const long long summary_mem_id = summary_base_id;
      const std::string summary_id
          = "summary-e2e-" + std::to_string (summary_base_id);

      Eigen::VectorXf centroid = Eigen::VectorXf::Zero (encoder.Dimension ());
      if (!cluster.centroid.empty ())
        {
          centroid = Eigen::Map<const Eigen::VectorXf> (
              cluster.centroid.data (),
              static_cast<Eigen::Index> (cluster.centroid.size ()));
        }
      SeedSummaryMemory (*store, summary_emb_id, summary_mem_id, summary_id,
                         static_cast<long long> (ts), centroid);

      // Create derived_from edges
      for (const auto &emb_id : cluster.embedding_ids)
        {
          store->Execute (
              "INSERT INTO associations(source_memory_id, target_memory_id, "
              "edge_type) VALUES(?, ?, 'derived_from')",
              { summary_mem_id, emb_id });
        }

      // Determine cluster identity by checking which IDs are present
      bool has_caregiver = false;
      bool has_location = false;
      bool has_medication = false;
      for (const auto &emb_id : cluster.embedding_ids)
        {
          if (emb_id >= 1 && emb_id <= 5)
            has_caregiver = true;
          else if (emb_id >= 6 && emb_id <= 10)
            has_location = true;
          else if (emb_id >= 11 && emb_id <= 15)
            has_medication = true;
        }

      cortext::operations::ExtractionResult extraction;
      extraction.summary_id = summary_id;

      if (has_caregiver)
        {
          extraction.facts.push_back (
              { "alice", "caregiver", "emily", 0.9, std::nullopt,
                std::nullopt });
        }
      if (has_location)
        {
          extraction.facts.push_back (
              { "alice", "lives_in", "oak_street", 0.85, std::nullopt,
                std::nullopt });
        }
      if (has_medication)
        {
          extraction.facts.push_back (
              { "alice", "takes_medication", "metformin", 0.95, std::nullopt,
                std::nullopt });
        }

      if (!extraction.facts.empty ())
        {
          RunExtraction (*store, pctx, encoder, std::move (extraction), ts);
        }

      ++summary_base_id;
    }

  // Count facts and distinct predicates
  auto fact_rows
      = store->Execute ("SELECT COUNT(*) AS cnt FROM fact_assertions", {});
  metrics.facts_extracted
      = std::any_cast<long long> (fact_rows[0].at ("cnt"));

  auto pred_rows = store->Execute (
      "SELECT COUNT(DISTINCT canonical_predicate) AS cnt FROM fact_assertions",
      {});
  metrics.fact_predicates
      = std::any_cast<long long> (pred_rows[0].at ("cnt"));

  return metrics;
}

// =========================================================================
// Study 3: Memory-Fact Co-Decay
// =========================================================================

struct Study3Metrics
{
  std::string config_name;
  double phase1_fact_survival = 0.0;
  double phase3_archived_survival = 0.0;
  double unlinked_survival = 0.0;
};

Study3Metrics
RunStudy3Config (const std::string &name, bool fact_floor_enabled,
                 bool archive_facts)
{
  Study3Metrics metrics;
  metrics.config_name = name;

  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  BenchEncoder encoder (g_models_dir);
  auto pctx = MakeProcessorContext (0.5);
  // Allow eviction: pretend consolidation ran before all memories were created.
  pctx.last_consolidation_ts = 1;

  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.encoder = &encoder;

  const auto base_emb = encoder.EncodeTextEigen (
      "Alice memory fact lifecycle benchmark");

  // Seed 10 memories (IDs 1-10)
  for (int i = 1; i <= 10; ++i)
    SeedMemory (*store, i, 1.0, 0LL, base_emb);

  // Facts linked to IDs 1-5:
  // Fact 1: {alice, caregiver, emily} linked to IDs 1-2
  InsertFactAssertion (*store, 1LL, "alice", "caregiver", "emily", 0LL, 0.9,
                       1LL);
  LinkFactEvidence (*store, 1LL, 1LL, "episodic", 1.0);
  LinkFactEvidence (*store, 1LL, 2LL, "episodic", 1.0);

  // Fact 2: {alice, takes_medication, metformin} linked to IDs 3-4
  InsertFactAssertion (*store, 2LL, "alice", "takes_medication", "metformin",
                       0LL, 0.95, 3LL);
  LinkFactEvidence (*store, 2LL, 3LL, "episodic", 1.0);
  LinkFactEvidence (*store, 2LL, 4LL, "episodic", 1.0);

  // Fact 3: {alice, likes, vanilla} linked to ID 5
  InsertFactAssertion (*store, 3LL, "alice", "likes", "vanilla", 0LL, 0.7,
                       5LL);
  LinkFactEvidence (*store, 3LL, 5LL, "episodic", 1.0);

  // IDs 6-10 are unlinked controls

  // Apply eviction ablation override for fact_floor
  cortext::operations::eviction::EvictionAblationOverride eviction_override;
  eviction_override.fact_floor_enabled = fact_floor_enabled;
  cortext::operations::eviction::ScopedEvictionAblationOverride eviction_guard (
      eviction_override);

  constexpr long long kIntervalMs = 60000;

  // Phase 1: 250 decay steps (enough for unlinked to evict at T=0.5)
  for (int step = 1; step <= 250; ++step)
    {
      const auto ts = static_cast<std::uint64_t> (step * kIntervalMs);
      std::vector<cortext::OperationContext::MemoryUsageEvent> events;
      for (int i = 1; i <= 10; ++i)
        {
          if (CountMemoryById (*store, i) > 0)
            events.push_back ({ static_cast<long long> (i), false });
        }
      if (!events.empty ())
        RunStrengthUpdate (*store, encoder, cfg, events, ts, pctx);
    }

  // Measure phase 1 fact-linked survival (IDs 1-5)
  int fact_linked_surviving = 0;
  for (int i = 1; i <= 5; ++i)
    {
      if (CountMemoryById (*store, i) > 0)
        ++fact_linked_surviving;
    }
  metrics.phase1_fact_survival = SafeRatio (fact_linked_surviving, 5);

  // Phase 2: Archive the "likes" fact (fact_id=3) if configured
  if (archive_facts)
    {
      store->Execute (
          "UPDATE fact_assertions SET lifecycle_state = 'archived' "
          "WHERE fact_id = 3",
          {});
    }

  // Phase 3: 250 more decay steps
  for (int step = 251; step <= 500; ++step)
    {
      const auto ts = static_cast<std::uint64_t> (step * kIntervalMs);
      std::vector<cortext::OperationContext::MemoryUsageEvent> events;
      for (int i = 1; i <= 10; ++i)
        {
          if (CountMemoryById (*store, i) > 0)
            events.push_back ({ static_cast<long long> (i), false });
        }
      if (!events.empty ())
        RunStrengthUpdate (*store, encoder, cfg, events, ts, pctx);
    }

  // Measure phase 3 archived fact survival (ID 5, the "likes" memory)
  metrics.phase3_archived_survival
      = (CountMemoryById (*store, 5) > 0) ? 1.0 : 0.0;

  // Measure unlinked survival (IDs 6-10)
  int unlinked_surviving = 0;
  for (int i = 6; i <= 10; ++i)
    {
      if (CountMemoryById (*store, i) > 0)
        ++unlinked_surviving;
    }
  metrics.unlinked_survival = SafeRatio (unlinked_surviving, 5);

  return metrics;
}

// =========================================================================
// Study 4: Retrieval-Reinforcement Feedback Loop
// =========================================================================

struct Study4Metrics
{
  std::string config_name;
  double fact_linked_strength_mean = 0.0;
  double unlinked_strength_mean = 0.0;
  double strength_gap = 0.0;
  double top5_fact_fraction = 0.0;
};

Study4Metrics
RunStudy4Config (const std::string &name, bool fact_layer_enabled)
{
  Study4Metrics metrics;
  metrics.config_name = name;

  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  BenchEncoder encoder (g_models_dir);
  auto pctx = MakeProcessorContext (0.5);

  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.encoder = &encoder;

  // Seed 20 memories: IDs 1-10 fact-linked, IDs 11-20 unlinked
  // Fact-linked memories are slightly LESS similar to the query
  // so only fact boost makes them competitive in retrieval.
  for (int i = 1; i <= 10; ++i)
    {
      SeedMemory (
          *store, i, 1.0, 0LL,
          encoder.EncodeTextEigen (
              // AIST 256-slice: sim ~0.730 to the query, vs ~0.795 for the
              // unlinked text below - linked memories lose on similarity
              // alone, so the fact boost has to earn the inversion.
              "Emily prepares meals and sets out medication reminders each morning"));
    }
  // Unlinked memories are MORE similar to query embedding
  for (int i = 11; i <= 20; ++i)
    {
      SeedMemory (
          *store, i, 1.0, 0LL,
          encoder.EncodeTextEigen (
              "Alice caregiver notes, caregiver schedule, caregiver planning"));
    }

  // Link IDs 1-10 to facts via fact_assertions + fact_evidence + fact_cache
  for (int i = 1; i <= 10; ++i)
    {
      const long long fact_id = static_cast<long long> (i);
      InsertFactAssertion (*store, fact_id, "alice", "caregiver", "emily", 0LL,
                           0.9, static_cast<long long> (i));
      LinkFactEvidence (*store, fact_id, static_cast<long long> (i),
                        "episodic", 1.0);

      // Create fact_cache entry so retrieval KNN can find them
      const std::string fact_text = "alice caregiver emily";
      // Reuse the memory's embedding_id for the fact_cache
      InsertFactCache (*store, fact_id, static_cast<long long> (i), fact_text,
                       true, 0LL, 0LL);
    }

  // Configure retrieval ablation
  cortext::operations::temporal::RetrievalAblationOverride retrieval_override;
  retrieval_override.fact_layer_enabled = fact_layer_enabled;
  cortext::operations::temporal::ScopedRetrievalAblationOverride
      retrieval_guard (retrieval_override);

  constexpr int kCycles = 50;
  constexpr long long kIntervalMs = 60000;

  for (int cycle = 1; cycle <= kCycles; ++cycle)
    {
      const auto ts = static_cast<std::uint64_t> (cycle * kIntervalMs);

      // Run retrieval via SignalProcessor
      auto ops = std::make_unique<cortext::DynamicOperationSet> (
          std::make_unique<ForceRetrievalGateOp> (),
          std::make_unique<
              cortext::operations::GraphAugmentedRetrieveCandidates> ());
      cortext::SignalProcessor processor (cfg, store, std::move (ops));

      cortext::operations::retrieval_debug::ClearLastSelectedEmbeddingOrder ();
      cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
      (void)processor.Process (
          MakeSignal (encoder.EncodeTextEigen ("Alice caregiver"),
                      ts));

      // Read ranked candidates from debug state
      const auto &ranked
          = cortext::operations::retrieval_debug::GetLastRankedCandidates ();

      // Build usage events: top candidates get used=true
      std::vector<cortext::OperationContext::MemoryUsageEvent> events;
      constexpr int kTopK = 5;

      // Mark top-K as used
      std::vector<long long> used_ids;
      for (std::size_t r = 0;
           r < std::min (ranked.size (), static_cast<std::size_t> (kTopK));
           ++r)
        {
          if (ranked[r].memory_id > 0)
            used_ids.push_back (ranked[r].memory_id);
        }

      // Build events for all living memories
      for (int i = 1; i <= 20; ++i)
        {
          if (CountMemoryById (*store, i) == 0)
            continue;
          const long long mid = static_cast<long long> (i);
          const bool used = std::find (used_ids.begin (), used_ids.end (), mid)
                            != used_ids.end ();
          events.push_back (
              { mid, used, used ? 0.5 : std::optional<double>{} });
        }

      if (!events.empty ())
        RunStrengthUpdate (*store, encoder, cfg, events, ts, pctx);
    }

  // Compute metrics
  double fact_sum = 0.0;
  int fact_count = 0;
  double unlinked_sum = 0.0;
  int unlinked_count = 0;

  for (int i = 1; i <= 10; ++i)
    {
      const double s = GetStrength (*store, i);
      if (s >= 0.0)
        {
          fact_sum += s;
          ++fact_count;
        }
    }
  for (int i = 11; i <= 20; ++i)
    {
      const double s = GetStrength (*store, i);
      if (s >= 0.0)
        {
          unlinked_sum += s;
          ++unlinked_count;
        }
    }

  metrics.fact_linked_strength_mean
      = fact_count > 0 ? fact_sum / fact_count : 0.0;
  metrics.unlinked_strength_mean
      = unlinked_count > 0 ? unlinked_sum / unlinked_count : 0.0;
  metrics.strength_gap
      = metrics.fact_linked_strength_mean - metrics.unlinked_strength_mean;

  // Get final retrieval ranking to compute top5_fact_fraction
  {
    auto ops = std::make_unique<cortext::DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<
            cortext::operations::GraphAugmentedRetrieveCandidates> ());
    cortext::SignalProcessor processor (cfg, store, std::move (ops));

    cortext::operations::retrieval_debug::ClearLastSelectedEmbeddingOrder ();
    cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
    const auto final_ts
        = static_cast<std::uint64_t> ((kCycles + 1) * kIntervalMs);
    (void)processor.Process (
        MakeSignal (encoder.EncodeTextEigen ("Alice caregiver"),
                    final_ts));

    const auto &ranked
        = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
    int top5_fact = 0;
    for (std::size_t r = 0; r < std::min (ranked.size (),
                                           static_cast<std::size_t> (5));
         ++r)
      {
        if (ranked[r].memory_id >= 1 && ranked[r].memory_id <= 10)
          ++top5_fact;
      }
    metrics.top5_fact_fraction = SafeRatio (top5_fact, 5);
  }

  return metrics;
}

// =========================================================================
// Study 5: Cross-Consolidation Fact Inheritance
// =========================================================================

struct Study5Metrics
{
  std::string config_name;
  long long fact_count_after = 0;
  long long duplicate_facts = 0;
  long long confirmation_count = 0;
  long long supersession_count = 0;
};

Study5Metrics
RunStudy5Config (const std::string &name, bool seed_existing_fact,
                 bool conflict_object)
{
  Study5Metrics metrics;
  metrics.config_name = name;

  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  BenchEncoder encoder (g_models_dir);
  auto pctx = MakeProcessorContext (0.5);

  const auto base_emb = encoder.EncodeTextEigen (
      "Alice caregiver Emily derived summary inheritance");
  const std::uint64_t ts = 10000ULL;

  // Seed 5 memories (IDs 1-5) that will be linked to existing fact
  for (int i = 1; i <= 5; ++i)
    SeedMemory (*store, i, 1.0, 0LL, base_emb);

  // Seed 5 more memories (IDs 6-10) with no facts
  for (int i = 6; i <= 10; ++i)
    SeedMemory (*store, i, 1.0, 0LL, base_emb);

  // Create a summary memory (kind=ASSOCIATION) linking all 10
  const long long summary_emb_id = 100LL;
  const long long summary_mem_id = 100LL;
  const std::string summary_id = "summary-inherit";
  SeedSummaryMemory (*store, summary_emb_id, summary_mem_id, summary_id,
                     static_cast<long long> (ts), base_emb);

  // Create derived_from edges linking summary to all 10 memories
  for (int i = 1; i <= 10; ++i)
    {
      store->Execute (
          "INSERT INTO associations(source_memory_id, target_memory_id, "
          "edge_type) VALUES(?, ?, 'derived_from')",
          { summary_mem_id, static_cast<long long> (i) });
    }

  if (seed_existing_fact)
    {
      // Create existing fact via extraction so all bookkeeping is done
      const std::string existing_object
          = conflict_object ? "sarah" : "emily";

      // Seed a prior summary for the existing fact
      const long long prior_summary_emb = 200LL;
      const long long prior_summary_mem = 200LL;
      const std::string prior_summary_id = "summary-prior";
      SeedSummaryMemory (*store, prior_summary_emb, prior_summary_mem,
                         prior_summary_id,
                         static_cast<long long> (ts - 5000), base_emb);

      cortext::operations::ExtractionResult prior_extraction;
      prior_extraction.summary_id = prior_summary_id;
      prior_extraction.facts.push_back (
          { "alice", "caregiver", existing_object, 0.9,
            std::nullopt, std::nullopt });
      RunExtraction (*store, pctx, encoder, std::move (prior_extraction),
                     ts - 5000);

      // Link evidence from IDs 1-5
      const long long existing_fact_id
          = LookupFactId (*store, "alice", "caregiver",
                          cortext::store::NormalizeFactTerm (existing_object));
      if (existing_fact_id > 0)
        {
          for (int i = 1; i <= 5; ++i)
            LinkFactEvidence (*store, existing_fact_id,
                              static_cast<long long> (i), "episodic", 1.0);
        }
    }

  // Now inject ExtractionResult with {alice, caregiver, emily} via the
  // new summary
  cortext::operations::ExtractionResult new_extraction;
  new_extraction.summary_id = summary_id;
  new_extraction.facts.push_back (
      { "alice", "caregiver", "emily", 0.9,
        std::nullopt, std::nullopt });
  RunExtraction (*store, pctx, encoder, std::move (new_extraction), ts);

  // Measure results
  // Total fact count for (alice, caregiver, *)
  auto all_facts = store->Execute (
      "SELECT COUNT(*) AS cnt FROM fact_assertions "
      "WHERE canonical_subject = 'alice' AND canonical_predicate = 'caregiver'",
      {});
  metrics.fact_count_after
      = std::any_cast<long long> (all_facts[0].at ("cnt"));

  // Count duplicates: facts with same (subject, predicate, object)
  auto dup_rows = store->Execute (
      "SELECT canonical_object, COUNT(*) AS cnt FROM fact_assertions "
      "WHERE canonical_subject = 'alice' AND canonical_predicate = 'caregiver' "
      "GROUP BY canonical_object HAVING COUNT(*) > 1",
      {});
  metrics.duplicate_facts = static_cast<long long> (dup_rows.size ());

  // Get confirmation_count for the emily fact
  auto confirm_rows = store->Execute (
      "SELECT confirmation_count FROM fact_assertions "
      "WHERE canonical_subject = 'alice' AND canonical_predicate = 'caregiver' "
      "AND canonical_object = 'emily' ORDER BY fact_id DESC LIMIT 1",
      {});
  if (!confirm_rows.empty ())
    {
      const auto &v = confirm_rows[0].at ("confirmation_count");
      if (v.type () == typeid (long long))
        metrics.confirmation_count = std::any_cast<long long> (v);
      else if (v.type () == typeid (int))
        metrics.confirmation_count = std::any_cast<int> (v);
    }

  // Count supersessions
  auto supersede_rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM fact_assertions "
      "WHERE canonical_subject = 'alice' AND canonical_predicate = 'caregiver' "
      "AND superseded_at_ts IS NOT NULL",
      {});
  metrics.supersession_count
      = std::any_cast<long long> (supersede_rows[0].at ("cnt"));

  return metrics;
}

// =========================================================================
// Gate evaluation
// =========================================================================

struct StudyGates
{
  bool study1_pass = false;
  bool study2_pass = false;
  bool study3_pass = false;
  bool study4_pass = false;
  bool study5_pass = false;
};

} // namespace

int
main (int argc, char **argv)
{
  g_models_dir = cortext::benchmark::ParseModelsDirArg (argc, argv);
  BenchEncoder encoder (g_models_dir);
  std::cout << std::fixed << std::setprecision (4);
  std::cout << "encoder_backend=" << encoder.backend_name () << "\n";
  std::cout << "encoder_model_path=" << encoder.resolved_model_path ().string ()
            << "\n";
  StudyGates gates;

  // =====================================================================
  // Study 1: Consolidation-Eviction Race
  // =====================================================================
  std::cout << "=== Study 1: Consolidation-Eviction Race ===\n";

  auto s1_early = RunStudy1Config ("race_early", 50);
  auto s1_default = RunStudy1Config ("race_default", 150);
  auto s1_late = RunStudy1Config ("race_late", 250);

  const auto print_s1 = [] (const Study1Metrics &m) {
    std::cout << "config=" << m.config_name << "\n";
    std::cout << "candidates_available=" << m.candidates_available << "\n";
    std::cout << "facts_extracted=" << m.facts_extracted << "\n";
    std::cout << "evidence_links_valid=" << m.evidence_links_valid << "\n";
    std::cout << "\n";
  };
  print_s1 (s1_early);
  print_s1 (s1_default);
  print_s1 (s1_late);

  // Gates: With consolidation-gated eviction, all configs should find all
  // 30 candidates alive (eviction is blocked until after consolidation).
  // All three should extract facts successfully.
  gates.study1_pass
      = (s1_early.candidates_available == 30)
        && (s1_default.candidates_available == 30)
        && (s1_late.candidates_available == 30)
        && (s1_early.facts_extracted > 0)
        && (s1_late.facts_extracted > 0);

  if (!gates.study1_pass)
    {
      std::cerr << "Study 1 gate detail:"
                << " early_candidates=" << s1_early.candidates_available
                << " default_candidates=" << s1_default.candidates_available
                << " late_candidates=" << s1_late.candidates_available
                << " early_facts=" << s1_early.facts_extracted
                << " late_facts=" << s1_late.facts_extracted << "\n";
    }

  // =====================================================================
  // Study 2: End-to-End Fact Emergence
  // =====================================================================
  std::cout << "=== Study 2: End-to-End Fact Emergence ===\n";

  auto s2_full = RunStudy2Config ("e2e_full", true, true);
  auto s2_no_extraction = RunStudy2Config ("e2e_no_extraction", true, false);
  auto s2_no_consolidation
      = RunStudy2Config ("e2e_no_consolidation", false, false);

  const auto print_s2 = [] (const Study2Metrics &m) {
    std::cout << "config=" << m.config_name << "\n";
    std::cout << "summaries_created=" << m.summaries_created << "\n";
    std::cout << "facts_extracted=" << m.facts_extracted << "\n";
    std::cout << "fact_predicates=" << m.fact_predicates << "\n";
    std::cout << "\n";
  };
  print_s2 (s2_full);
  print_s2 (s2_no_extraction);
  print_s2 (s2_no_consolidation);

  // Gates: e2e_full >= 3 facts; e2e_no_extraction = 0; e2e_no_consolidation
  // = 0
  gates.study2_pass = (s2_full.facts_extracted >= 3)
                      && (s2_no_extraction.facts_extracted == 0)
                      && (s2_no_consolidation.facts_extracted == 0);

  if (!gates.study2_pass)
    {
      std::cerr << "Study 2 gate detail: full_facts="
                << s2_full.facts_extracted
                << " no_extraction_facts="
                << s2_no_extraction.facts_extracted
                << " no_consolidation_facts="
                << s2_no_consolidation.facts_extracted << "\n";
    }

  // =====================================================================
  // Study 3: Memory-Fact Co-Decay
  // =====================================================================
  std::cout << "=== Study 3: Memory-Fact Co-Decay ===\n";

  auto s3_default = RunStudy3Config ("codecay_default", true, true);
  auto s3_no_floor = RunStudy3Config ("codecay_no_floor", false, true);
  auto s3_no_lifecycle = RunStudy3Config ("codecay_no_lifecycle", true, false);

  const auto print_s3 = [] (const Study3Metrics &m) {
    std::cout << "config=" << m.config_name << "\n";
    std::cout << "phase1_fact_survival=" << m.phase1_fact_survival << "\n";
    std::cout << "phase3_archived_survival=" << m.phase3_archived_survival
              << "\n";
    std::cout << "unlinked_survival=" << m.unlinked_survival << "\n";
    std::cout << "\n";
  };
  print_s3 (s3_default);
  print_s3 (s3_no_floor);
  print_s3 (s3_no_lifecycle);

  // Gates:
  // codecay_default: phase1_fact > unlinked
  // codecay_default: phase3_archived <= unlinked for archived fact
  // codecay_no_floor: fact-linked ≈ unlinked (floor disabled)
  gates.study3_pass
      = (s3_default.phase1_fact_survival > s3_default.unlinked_survival)
        && (s3_default.phase3_archived_survival <= s3_default.unlinked_survival
            || s3_default.phase3_archived_survival < 0.01)
        && (std::abs (s3_no_floor.phase1_fact_survival
                      - s3_no_floor.unlinked_survival)
            < 0.5);

  if (!gates.study3_pass)
    {
      std::cerr << "Study 3 gate detail: default_p1_fact="
                << s3_default.phase1_fact_survival
                << " default_unlinked=" << s3_default.unlinked_survival
                << " default_p3_archived="
                << s3_default.phase3_archived_survival
                << " no_floor_p1_fact="
                << s3_no_floor.phase1_fact_survival
                << " no_floor_unlinked=" << s3_no_floor.unlinked_survival
                << "\n";
    }

  // =====================================================================
  // Study 4: Retrieval-Reinforcement Feedback Loop
  // =====================================================================
  std::cout << "=== Study 4: Retrieval-Reinforcement Feedback Loop ===\n";

  auto s4_default = RunStudy4Config ("feedback_default", true);
  auto s4_no_facts = RunStudy4Config ("feedback_no_facts", false);

  const auto print_s4 = [] (const Study4Metrics &m) {
    std::cout << "config=" << m.config_name << "\n";
    std::cout << "fact_linked_strength_mean=" << m.fact_linked_strength_mean
              << "\n";
    std::cout << "unlinked_strength_mean=" << m.unlinked_strength_mean << "\n";
    std::cout << "strength_gap=" << m.strength_gap << "\n";
    std::cout << "top5_fact_fraction=" << m.top5_fact_fraction << "\n";
    std::cout << "\n";
  };
  print_s4 (s4_default);
  print_s4 (s4_no_facts);

  // Gates:
  // feedback_default: strength_gap > 0 (fact-linked stronger)
  // feedback_no_facts: strength_gap <= 0 (unlinked now stronger or equal)
  // The gap should flip sign when facts are disabled.
  gates.study4_pass = (s4_default.strength_gap > 0.0)
                      && (s4_no_facts.strength_gap <= 0.0);

  if (!gates.study4_pass)
    {
      std::cerr << "Study 4 gate detail: default_gap="
                << s4_default.strength_gap
                << " no_facts_gap=" << s4_no_facts.strength_gap << "\n";
    }

  // =====================================================================
  // Study 5: Cross-Consolidation Fact Inheritance
  // =====================================================================
  std::cout << "=== Study 5: Cross-Consolidation Fact Inheritance ===\n";

  auto s5_default = RunStudy5Config ("inherit_default", true, false);
  auto s5_no_existing = RunStudy5Config ("inherit_no_existing", false, false);
  auto s5_conflict = RunStudy5Config ("inherit_conflict", true, true);

  const auto print_s5 = [] (const Study5Metrics &m) {
    std::cout << "config=" << m.config_name << "\n";
    std::cout << "fact_count_after=" << m.fact_count_after << "\n";
    std::cout << "duplicate_facts=" << m.duplicate_facts << "\n";
    std::cout << "confirmation_count=" << m.confirmation_count << "\n";
    std::cout << "supersession_count=" << m.supersession_count << "\n";
    std::cout << "\n";
  };
  print_s5 (s5_default);
  print_s5 (s5_no_existing);
  print_s5 (s5_conflict);

  // Gates:
  // inherit_default: duplicate_facts=0 and confirmation_count increased (>=1)
  // inherit_conflict: supersession_count=1
  gates.study5_pass = (s5_default.duplicate_facts == 0)
                      && (s5_default.confirmation_count >= 1)
                      && (s5_conflict.supersession_count >= 1);

  if (!gates.study5_pass)
    {
      std::cerr << "Study 5 gate detail: default_duplicates="
                << s5_default.duplicate_facts
                << " default_confirmations="
                << s5_default.confirmation_count
                << " conflict_supersessions="
                << s5_conflict.supersession_count << "\n";
    }

  // =====================================================================
  // Summary
  // =====================================================================
  std::cout << "=== Gate Results ===\n";
  std::cout << "study1_gates_pass="
            << (gates.study1_pass ? "true" : "false") << "\n";
  std::cout << "study2_gates_pass="
            << (gates.study2_pass ? "true" : "false") << "\n";
  std::cout << "study3_gates_pass="
            << (gates.study3_pass ? "true" : "false") << "\n";
  std::cout << "study4_gates_pass="
            << (gates.study4_pass ? "true" : "false") << "\n";
  std::cout << "study5_gates_pass="
            << (gates.study5_pass ? "true" : "false") << "\n";

  const bool all_pass = gates.study1_pass && gates.study2_pass
                        && gates.study3_pass && gates.study4_pass
                        && gates.study5_pass;
  std::cout << "all_gates_pass=" << (all_pass ? "true" : "false") << "\n";

  return all_pass ? 0 : 1;
}
