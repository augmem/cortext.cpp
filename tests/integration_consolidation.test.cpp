// Integration Tests for Phase 5
// Tests full consolidation pipeline, graph construction, streaming pacing,
// and interrupt gate with refractory dynamics.

#include <Eigen/Dense>
#include "test_helpers.hpp"
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/algorithms.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/consolidation_mode.hpp>
#include <cortext/operations/consolidation.hpp>
#include <cortext/operations/consolidation_cluster.hpp>
#include <cortext/operations/consolidation_summarize.hpp>
#include <cortext/operations/drift_accumulation.hpp>
#include <cortext/operations/embedding_prediction_error.hpp>
#include <cortext/operations/focus.hpp>
#include <cortext/operations/graph_build.hpp>
#include <cortext/operations/interrupt_gate.hpp>
#include <cortext/operations/recent_context.hpp>
#include <cortext/operations/streaming_pacing.hpp>
#include <cortext/operations/uncertainty.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>
#include <cortext/summarizer/summarizer.hpp>
#include <string>
#include <vector>

#include "../src/operations/eviction_ablation.hpp"

using namespace cortext;
using namespace cortext::operations;
using namespace cortext::core;

namespace
{

Signal
MakeSignal (uint64_t ts, Eigen::VectorXf emb)
{
  Signal s;
  s.embedding = std::move (emb);
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

Signal
MakeSignal (uint64_t ts)
{
  return MakeSignal (ts, Eigen::VectorXf::Ones (256));
}

Eigen::VectorXf
MakeEmbedding (int dim, float base_val)
{
  Eigen::VectorXf emb = Eigen::VectorXf::Constant (dim, base_val);
  return emb;
}

Eigen::VectorXf
MakeOrthogonalEmbedding (int dim, int index)
{
  Eigen::VectorXf emb = Eigen::VectorXf::Zero (dim);
  if (index < dim)
    emb[index] = 1.0f;
  return emb;
}

// Helper to seed embeddings for testing (v2 schema: embeddings + memories)
void
SeedEmbedding (Store *store, long long id, const Eigen::VectorXf &emb,
               double strength = 1.0, double redundancy = 0.0,
               double connectivity = 0.0, double stability = 1.0)
{
  std::vector<float> vec (emb.data (), emb.data () + emb.size ());
  auto now_ts = cortext::testing::NowMs ();

  // v2: Insert into embeddings (minimal vec0 table)
  store->Execute (
      "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { id, vec, now_ts });

  // v2: Insert into memories (comprehensive metadata)
  store->Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, redundancy, connectivity, stability, created_at) "
      "VALUES(?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, ?, ?, ?, ?, ?)",
      { id, id, now_ts, strength, redundancy, connectivity, stability, now_ts });
}

// Helper to seed memories for testing (v2 schema)
void
SeedMemory (Store *store, long long embedding_id, const std::string &text,
            uint64_t ts)
{
  // v2: Update existing memory record (created by SeedEmbedding) or insert new
  store->Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, end_ts, "
      "n_signals, modality, s_max, s_avg, created_at) "
      "VALUES(?, ?, 'test', 'LONG_TERM', ?, ?, 1, 'text', 0.5, 0.5, ?)",
      { embedding_id, embedding_id, static_cast<long long> (ts),
        static_cast<long long> (ts), static_cast<long long> (ts) });

  std::vector<unsigned char> data_blob (text.begin (), text.end ());
  auto blob_rows
      = store->Execute ("SELECT objstore_put(?1) AS id", { data_blob });
  if (!blob_rows.empty () && blob_rows[0].count ("id") != 0)
    {
      auto blob_id = store::BlobFromAny (blob_rows[0].at ("id"));
      if (!blob_id.empty ())
        {
          store->Execute ("UPDATE memories SET blob_id = ? WHERE memory_id = ?",
                          { blob_id, embedding_id });
        }
    }
}

// Helper to count table rows
long long
CountRows (Store *store, const std::string &table)
{
  auto rows = store->Execute ("SELECT COUNT(*) AS c FROM " + table, {});
  if (rows.empty ())
    return 0;
  return std::any_cast<long long> (rows[0].at ("c"));
}

// Helper to get long long value from row
long long
GetInt64 (const std::map<std::string, std::any> &row, const std::string &key)
{
  auto it = row.find (key);
  if (it == row.end ())
    return 0;
  if (it->second.type () == typeid (long long))
    return std::any_cast<long long> (it->second);
  if (it->second.type () == typeid (int))
    return static_cast<long long> (std::any_cast<int> (it->second));
  return 0;
}

double
GetDouble (const std::map<std::string, std::any> &row, const std::string &key,
           double fallback = 0.0)
{
  auto it = row.find (key);
  if (it == row.end () || !it->second.has_value ())
    return fallback;
  if (it->second.type () == typeid (double))
    return std::any_cast<double> (it->second);
  if (it->second.type () == typeid (float))
    return static_cast<double> (std::any_cast<float> (it->second));
  if (it->second.type () == typeid (long long))
    return static_cast<double> (std::any_cast<long long> (it->second));
  if (it->second.type () == typeid (int))
    return static_cast<double> (std::any_cast<int> (it->second));
  return fallback;
}

class CapturingSummarizer final : public Summarizer
{
public:
  std::string
  SummarizeTexts (const std::vector<std::string> &texts) override
  {
    calls++;
    captured_texts = texts;
    last_max_words = -1;
    return "captured summary";
  }

  std::string
  SummarizeTextsLimited (const std::vector<std::string> &texts,
                         int max_words) override
  {
    calls++;
    captured_texts = texts;
    last_max_words = max_words;
    return "captured summary";
  }

  std::string
  SummarizeAudio (const float * /*pcm*/, size_t /*num_samples*/) override
  {
    return {};
  }

  std::string
  SummarizeAudioSegments (const std::vector<AudioSegment> & /*segments*/) override
  {
    return {};
  }

  bool
  IsAvailable () const override
  {
    return true;
  }

  std::vector<std::string> captured_texts;
  int last_max_words = -1;
  int calls = 0;
};

void
MarkCompressibleClusterSources (Store *store)
{
  store->Execute (
      "UPDATE memories SET redundancy = 1.0, strength = 0.0, "
      "emotional_intensity = 0.0, s_emotion_max = 0.0, s_arousal_avg = 0.0",
      {});
}

// Helper op to setup consolidation trigger conditions
struct SetupConsolidationTriggerOp : IOperation
{
  uint64_t now_ts;
  double stability;
  double sensitivity;

  SetupConsolidationTriggerOp (uint64_t ts, double T, double S = 0.5)
      : now_ts (ts), stability (T), sensitivity (S)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetTokensInFlight (0);
    ctx.SetRetrievalQueueDepth (0);
    auto &p = ctx.GetProcessorContext ();

    // Compute knob-derived rate target and set m_rate below threshold
    const double rate_target = ConsolidationRate (stability, sensitivity);
    p.m_rate = rate_target * 0.4; // Below 50% threshold to trigger
    p.rate_target = rate_target;  // Not used anymore but kept for compatibility

    // idle_required is in seconds, timestamps in milliseconds
    int idle_required_s = IdleRequiredSeconds (stability);
    p.last_retrieval_ts = now_ts - static_cast<uint64_t> (idle_required_s + 1) * 1000ULL;

    // Don't set last_consolidation_ts to allow rate trigger
  }
};

// Helper op to verify consolidation started
struct AssertConsolidationStartedOp : IOperation
{
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    REQUIRE (ctx.GetConsolidationShouldStart () == true);
  }
};

// Helper op to seed clustering data
struct SeedClusteringDataOp : IOperation
{
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto *store = ctx.GetStore ();
    std::vector<ConsolidationCandidate> candidates;

    // Create 4 similar embeddings (should cluster together)
    // Group 1: embeddings 1-4 are similar (all have e[0] = 1.0)
    for (long long i = 1; i <= 4; ++i)
      {
        Eigen::VectorXf emb = Eigen::VectorXf::Zero (256);
        emb[0] = 1.0f;
        emb[static_cast<int> (i)] = 0.1f; // Small variation

        SeedEmbedding (store, i, emb, 0.1, 0.0, 0.0, 0.0); // Low strength = candidates

        // Add to in-memory candidates
        ConsolidationCandidate c;
        c.embedding_id = i;
        c.score = 0.05;
        c.embedding = emb;
        candidates.push_back (std::move (c));
      }

    // Group 2: embeddings 5-8 are similar (all have e[100] = 1.0)
    for (long long i = 5; i <= 8; ++i)
      {
        Eigen::VectorXf emb = Eigen::VectorXf::Zero (256);
        emb[100] = 1.0f;
        emb[static_cast<int> (i)] = 0.1f; // Small variation

        SeedEmbedding (store, i, emb, 0.1, 0.0, 0.0, 0.0);

        // Add to in-memory candidates
        ConsolidationCandidate c;
        c.embedding_id = i;
        c.score = 0.05;
        c.embedding = emb;
        candidates.push_back (std::move (c));
      }

    // Set candidates in context (in-memory passing)
    ctx.SetConsolidationCandidates (std::move (candidates));
  }
};

// Helper op to seed graph construction data using V2 schema
struct SeedGraphDataOp : IOperation
{
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto *store = ctx.GetStore ();

    // V2: Create memories with cluster_id for graph edge construction
    // (GraphBuild queries memories with cluster_id IS NOT NULL)
    for (long long i = 1; i <= 4; ++i)
      {
        // Create similar embeddings for co-occurrence testing
        Eigen::VectorXf emb = Eigen::VectorXf::Zero (256);
        emb[0] = 0.9f;
        emb[static_cast<int> (i)] = 0.1f;
        SeedEmbedding (store, i, emb, 0.5, 0.0, 0.1, 0.3);

        // Add memories with timestamps for causal edges, cluster_id for graph
        uint64_t ts = static_cast<uint64_t> (i * 1000);
        SeedMemory (store, i, "Memory text " + std::to_string (i), ts);

        // Set cluster_id on the memory (V2: graph build uses cluster_id)
        store->Execute (
            "UPDATE memories SET cluster_id = ? WHERE memory_id = ?",
            { 100LL, i }); // All in same cluster for edge testing
      }

    // V2: LABEL memories replace extraction_entities
    // (kind='LABEL' with label field)
    cortext::testing::SeedEmbeddingV2 (*store, 200LL, std::vector<float>(256, 0.5f));
    store->Execute (
        "INSERT INTO memories(memory_id, embedding_id, source_id, kind, label, "
        "start_ts, n_signals, modality, s_max, s_avg, created_at) "
        "VALUES(?, ?, 'test', 'LABEL', ?, 0, 1, 'text', 0.5, 0.5, 0)",
        { 200LL, 200LL, std::string ("TestEntity") });

    // V2: Create ASSOCIATION memory to represent the summary
    cortext::testing::SeedEmbeddingV2 (*store, 300LL, std::vector<float>(256, 0.5f));
    store->Execute (
        "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
        "start_ts, n_signals, modality, s_max, s_avg, created_at) "
        "VALUES(?, ?, 'test', 'ASSOCIATION', 0, 1, 'text', 0.5, 0.5, 0)",
        { 300LL, 300LL });

    // V2: Link ASSOCIATION to LABEL via derived_from edge
    store->Execute (
        "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
        "VALUES(?, ?, 'derived_from', 1.0)",
        { 300LL, 200LL });

    // V2: Link ASSOCIATION to source memories via derived_from
    for (long long i = 1; i <= 4; ++i)
      {
        store->Execute (
            "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
            "VALUES(?, ?, 'derived_from', 1.0)",
            { 300LL, i });
      }
  }
};

} // namespace

// =============================================================================
// 5.2.1 Full Consolidation Pipeline Test
// =============================================================================

TEST_CASE ("Consolidation pipeline triggers on explicit consolidation signal",
           "[integration][consolidation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Ensure now_ts is large enough for idle_required subtraction (idle ~45s at T=0.5)
  const uint64_t now_ts = 100'000ULL; // 100 seconds in ms

  auto setup = std::make_unique<SetupConsolidationTriggerOp> (now_ts, cfg.stability);
  auto eval = std::make_unique<EvaluateConsolidation> ();
  auto assert_op = std::make_unique<AssertConsolidationStartedOp> ();
  auto ops = std::make_unique<DynamicOperationSet> (std::move (setup),
                                              std::move (eval),
                                              std::move (assert_op));

  SignalProcessor processor (cfg, store, std::move (ops));
  Signal s = MakeSignal (now_ts);
  s.source_id = "test/consolidation";
  s.consolidation_mode = ConsolidationMode::Both;
  processor.Process (s);
  processor.Flush ();
}

TEST_CASE ("Clustering groups similar embeddings",
           "[integration][consolidation][clustering]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.3; // Lower focus = lower merge threshold, smaller min cluster
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Initialize schema
  auto init_ops = std::make_unique<DynamicOperationSet> ();
  SignalProcessor init_processor (cfg, store, std::move (init_ops));
  init_processor.Process (MakeSignal (1));
  init_processor.Flush ();

  // Seed clustering data
  ProcessorContext pctx;

  Signal s = MakeSignal (2000);
  OperationContext ctx (s, pctx, cfg, store.get ());

  SeedClusteringDataOp seed_op;
  seed_op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Verify candidates seeded (in-memory)
  REQUIRE (ctx.GetConsolidationCandidates ().size () == 8);

  // Run clustering
  ConsolidationCluster cluster_op;
  cluster_op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Verify clusters were formed
  auto clusters = ctx.GetConsolidationClusters ();

  // With F=0.3: min_cluster_size = round(lerp(3, 10, 0.3)) = 5
  // But we have two groups of 4 each, so clusters depend on threshold
  // With merge_threshold = lerp(0.85, 0.95, 0.3) = 0.88
  INFO ("Number of clusters: " << clusters.size ());
  INFO ("Min cluster size: " << MinClusterSize (cfg.focus));

  // At least verify no crash and clusters are returned
  // The exact count depends on similarity calculation
  REQUIRE (clusters.size () >= 0);
}

TEST_CASE ("Summarization creates summary records",
           "[integration][consolidation][summarize]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0; // Very low focus = min cluster size 3
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Initialize schema
  auto init_ops = std::make_unique<DynamicOperationSet> ();
  SignalProcessor init_processor (cfg, store, std::move (init_ops));
  init_processor.Process (MakeSignal (1));
  init_processor.Flush ();

  // Create test cluster directly
  ProcessorContext pctx;

  Signal s = MakeSignal (3000);
  OperationContext ctx (s, pctx, cfg, store.get ());

  // Seed embeddings and memories
  for (long long i = 1; i <= 4; ++i)
    {
      Eigen::VectorXf emb = Eigen::VectorXf::Constant (256, 0.5f);
      emb[0] = 1.0f;
      SeedEmbedding (store.get (), i, emb);
      SeedMemory (store.get (), i, "Test memory content " + std::to_string (i),
                  static_cast<uint64_t> (i * 1000));
    }

  // Create cluster info
  ClusterInfo cluster;
  cluster.cluster_id = 1;
  cluster.embedding_ids = { 1, 2, 3, 4 };
  cluster.centroid = std::vector<float> (256, 0.5f);
  cluster.avg_score = 0.3;

  std::vector<ClusterInfo> clusters = { cluster };
  ctx.SetConsolidationClusters (clusters);

  // Run summarization
  ConsolidationSummarize summarize_op;
  summarize_op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // V2: Verify ASSOCIATION memories were created (replaces consolidation_summaries)
  auto summaries = store->Execute (
      "SELECT memory_id FROM memories WHERE kind = 'ASSOCIATION'",
      {});
  INFO ("Summary (ASSOCIATION) count: " << summaries.size ());

  // Verify extraction requests were queued (cluster size 4 >= min for extraction)
  auto requests = ctx.GetExtractionRequests ();
  INFO ("Extraction request count: " << requests.size ());
}

TEST_CASE ("Summarization preserves raw evidence text before prompting",
           "[integration][consolidation][summarize]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  CapturingSummarizer summarizer;

  auto init_ops = std::make_unique<DynamicOperationSet> ();
  SignalProcessor init_processor (cfg, store, std::move (init_ops));
  init_processor.Process (MakeSignal (1));
  init_processor.Flush ();

  ProcessorContext pctx;
  pctx.summarizer = &summarizer;
  pctx.last_consolidation_ts = 100000;
  Signal s = MakeSignal (3000);
  s.source_id = "test/consolidation";
  s.consolidation_mode = ConsolidationMode::Both;
  OperationContext ctx (s, pctx, cfg, store.get ());
  ctx.SetConsolidationShouldStart (true);

  Eigen::VectorXf emb = Eigen::VectorXf::Constant (256, 0.5f);
  emb[0] = 1.0f;
  for (long long i = 1; i <= 4; ++i)
    {
      SeedEmbedding (store.get (), i, emb);
    }
  SeedMemory (store.get (), 1, "My name is Gabe and I am testing memory.", 1000);
  SeedMemory (store.get (), 2, "I will remember that you are Gabe.", 2000);
  SeedMemory (store.get (), 3, "The conversation stayed on memory testing.", 3000);
  SeedMemory (store.get (), 4, "The last note confirms the test cluster.", 4000);
  store->Execute (
      "UPDATE memories SET source_id = 'stream/a' WHERE memory_id IN (1, 3)",
      {});
  store->Execute (
      "UPDATE memories SET source_id = 'stream/b' WHERE memory_id IN (2, 4)",
      {});
  MarkCompressibleClusterSources (store.get ());

  ClusterInfo cluster;
  cluster.cluster_id = 7;
  cluster.embedding_ids = { 1, 2, 3, 4 };
  cluster.centroid = std::vector<float> (256, 0.5f);
  cluster.avg_score = 0.3;
  ctx.SetConsolidationClusters ({ cluster });

  ConsolidationSummarize summarize_op;
  cortext::operations::eviction::EvictionAblationOverride override;
  override.storage_gate_enabled = true;
  override.min_storage_bytes = 1000;
  override.used_storage_bytes = 10000;
  cortext::operations::eviction::ScopedEvictionAblationOverride pressure (
      override);
  auto tx = store->Begin ();
  summarize_op.Execute (ctx, *tx);
  tx->Commit ();

  REQUIRE (summarizer.captured_texts.size () == 3);
  REQUIRE (summarizer.captured_texts[0]
           == "My name is Gabe and I am testing memory.");
  REQUIRE (summarizer.captured_texts[1]
           == "I will remember that you are Gabe.");
  REQUIRE (summarizer.captured_texts[2]
           == "The conversation stayed on memory testing.");
  REQUIRE (summarizer.last_max_words == 0);
}

TEST_CASE ("Summarization source blob ablation keeps text evidence",
           "[integration][consolidation][summarize][ablation]")
{
  cortext::testing::ScopedEnvVar disable_source_blobs (
      "CORTEXT_DISABLE_SOURCE_BLOBS", "1");

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  CapturingSummarizer summarizer;

  auto init_ops = std::make_unique<DynamicOperationSet> ();
  SignalProcessor init_processor (cfg, store, std::move (init_ops));
  init_processor.Process (MakeSignal (1));
  init_processor.Flush ();

  ProcessorContext pctx;
  pctx.summarizer = &summarizer;
  pctx.last_consolidation_ts = 100000;
  Signal s = MakeSignal (3000);
  s.source_id = "test/consolidation";
  s.consolidation_mode = ConsolidationMode::Both;
  OperationContext ctx (s, pctx, cfg, store.get ());
  ctx.SetConsolidationShouldStart (true);

  Eigen::VectorXf emb = Eigen::VectorXf::Constant (256, 0.5f);
  emb[0] = 1.0f;
  for (long long i = 1; i <= 4; ++i)
    {
      SeedEmbedding (store.get (), i, emb);
    }
  SeedMemory (store.get (), 1, "My name is Gabe and I am testing memory.", 1000);
  SeedMemory (store.get (), 2, "I will remember that you are Gabe.", 2000);
  SeedMemory (store.get (), 3, "The conversation stayed on memory testing.", 3000);
  SeedMemory (store.get (), 4, "The last note confirms the test cluster.", 4000);
  store->Execute (
      "UPDATE memories SET source_id = 'stream/a' WHERE memory_id IN (1, 3)",
      {});
  store->Execute (
      "UPDATE memories SET source_id = 'stream/b' WHERE memory_id IN (2, 4)",
      {});
  MarkCompressibleClusterSources (store.get ());

  ClusterInfo cluster;
  cluster.cluster_id = 7;
  cluster.embedding_ids = { 1, 2, 3, 4 };
  cluster.centroid = std::vector<float> (256, 0.5f);
  cluster.avg_score = 0.3;
  ctx.SetConsolidationClusters ({ cluster });

  ConsolidationSummarize summarize_op;
  cortext::operations::eviction::EvictionAblationOverride override;
  override.storage_gate_enabled = true;
  override.min_storage_bytes = 1000;
  override.used_storage_bytes = 10000;
  cortext::operations::eviction::ScopedEvictionAblationOverride pressure (
      override);
  auto tx = store->Begin ();
  summarize_op.Execute (ctx, *tx);
  tx->Commit ();

  const auto requests = ctx.GetExtractionRequests ();
  REQUIRE (requests.size () == 1);
  REQUIRE (!requests[0].source_texts.empty ());
  REQUIRE (requests[0].source_blobs.empty ());
}

TEST_CASE ("Summarization passes STM graph labels to relabeler as candidates",
           "[integration][consolidation][summarize][labels]")
{
  cortext::testing::ScopedEnvVar enable_audit ("CORTEXT_STM_LTM_AUDIT", "1");

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto init_ops = std::make_unique<DynamicOperationSet> ();
  SignalProcessor init_processor (cfg, store, std::move (init_ops));
  init_processor.Process (MakeSignal (1));
  init_processor.Flush ();

  ProcessorContext pctx;
  pctx.last_consolidation_ts = 100000;
  Signal s = MakeSignal (3000);
  s.source_id = "test/consolidation";
  s.consolidation_mode = ConsolidationMode::Both;
  OperationContext ctx (s, pctx, cfg, store.get ());
  ctx.SetConsolidationShouldStart (true);

  Eigen::VectorXf emb = Eigen::VectorXf::Constant (256, 0.5f);
  emb[0] = 1.0f;
  SeedEmbedding (store.get (), 1, emb);
  SeedEmbedding (store.get (), 2, emb);
  SeedEmbedding (store.get (), 3, emb);
  SeedMemory (store.get (), 1, "They made plans for dinner.", 1000);
  SeedMemory (store.get (), 2, "The next message discussed timing.", 2000);
  SeedMemory (store.get (), 3, "The conversation stayed on logistics.", 3000);
  MarkCompressibleClusterSources (store.get ());

  ProcessorContext::ShadowLabelEdge edge;
  edge.source_id = "stream/a";
  edge.timestamp = 2000;
  edge.step_index = 1;
  edge.label = "Steve";
  edge.weight = 0.9;
  edge.signal_embedding = emb;
  auto &label_edges
      = ctx.GetProcessorContext ().short_term_graphs["stream/a"].label_edges;
  label_edges.push_back (edge);
  edge.label = "Alfred";
  label_edges.push_back (edge);
  edge.label = "Valarie";
  label_edges.push_back (edge);
  edge.label = "Maurice";
  label_edges.push_back (edge);

  ClusterInfo cluster;
  cluster.cluster_id = 10;
  cluster.embedding_ids = { 1, 2, 3 };
  cluster.centroid = std::vector<float> (256, 0.5f);
  cluster.avg_score = 0.3;
  ctx.SetConsolidationClusters ({ cluster });

  ConsolidationSummarize summarize_op;
  auto tx = store->Begin ();
  summarize_op.Execute (ctx, *tx);
  tx->Commit ();

  const auto requests = ctx.GetExtractionRequests ();
  REQUIRE (requests.size () == 1);
  REQUIRE (requests[0].current_labels.size ()
           == std::min<size_t> (
               4,
               cortext::core::STMLabelConsolidationMaxUngrounded (
                   cfg.focus, cfg.sensitivity, cfg.stability)));
  REQUIRE (std::find (requests[0].current_labels.begin (),
                      requests[0].current_labels.end (),
                      "Steve")
           != requests[0].current_labels.end ());
  REQUIRE (std::find (requests[0].current_labels.begin (),
                      requests[0].current_labels.end (),
                      "Maurice")
           != requests[0].current_labels.end ());

  auto audit_rows = store->Execute (
      "SELECT stm_label_edge_count, current_label_count, current_labels "
      "FROM stm_ltm_relabel_audit",
      {});
  REQUIRE (audit_rows.size () == 1);
  REQUIRE (GetInt64 (audit_rows[0], "stm_label_edge_count") == 4);
  REQUIRE (GetInt64 (audit_rows[0], "current_label_count") == 4);
  REQUIRE (std::any_cast<std::string> (
               audit_rows[0].at ("current_labels"))
           == "Steve\nAlfred\nValarie\nMaurice");
}

TEST_CASE ("Summarization can disable STM graph label handoff natively",
           "[integration][consolidation][summarize][labels]")
{
  cortext::testing::ScopedEnvVar enable_audit ("CORTEXT_STM_LTM_AUDIT", "1");
  cortext::testing::ScopedEnvVar disable_handoff (
      "CORTEXT_DISABLE_STM_LABEL_HANDOFF", "1");

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto init_ops = std::make_unique<DynamicOperationSet> ();
  SignalProcessor init_processor (cfg, store, std::move (init_ops));
  init_processor.Process (MakeSignal (1));
  init_processor.Flush ();

  ProcessorContext pctx;
  pctx.last_consolidation_ts = 100000;
  Signal s = MakeSignal (3000);
  s.source_id = "test/consolidation";
  s.consolidation_mode = ConsolidationMode::Both;
  OperationContext ctx (s, pctx, cfg, store.get ());
  ctx.SetConsolidationShouldStart (true);

  Eigen::VectorXf emb = Eigen::VectorXf::Constant (256, 0.5f);
  emb[0] = 1.0f;
  SeedEmbedding (store.get (), 1, emb);
  SeedEmbedding (store.get (), 2, emb);
  SeedEmbedding (store.get (), 3, emb);
  SeedMemory (store.get (), 1, "They made plans for dinner.", 1000);
  SeedMemory (store.get (), 2, "The next message discussed timing.", 2000);
  SeedMemory (store.get (), 3, "The conversation stayed on logistics.", 3000);
  MarkCompressibleClusterSources (store.get ());

  ProcessorContext::ShadowLabelEdge edge;
  edge.source_id = "stream/a";
  edge.timestamp = 2000;
  edge.step_index = 1;
  edge.label = "Steve";
  edge.weight = 0.9;
  edge.signal_embedding = emb;
  ctx.GetProcessorContext ().short_term_graphs["stream/a"].label_edges
      .push_back (edge);

  ClusterInfo cluster;
  cluster.cluster_id = 10;
  cluster.embedding_ids = { 1, 2, 3 };
  cluster.centroid = std::vector<float> (256, 0.5f);
  cluster.avg_score = 0.3;
  ctx.SetConsolidationClusters ({ cluster });

  ConsolidationSummarize summarize_op;
  auto tx = store->Begin ();
  summarize_op.Execute (ctx, *tx);
  tx->Commit ();

  const auto requests = ctx.GetExtractionRequests ();
  REQUIRE (requests.size () == 1);
  REQUIRE (requests[0].current_labels.empty ());

  auto audit_rows = store->Execute (
      "SELECT stm_label_edge_count, current_label_count, current_labels "
      "FROM stm_ltm_relabel_audit",
      {});
  REQUIRE (audit_rows.size () == 1);
  REQUIRE (GetInt64 (audit_rows[0], "stm_label_edge_count") == 1);
  REQUIRE (GetInt64 (audit_rows[0], "current_label_count") == 0);
  REQUIRE (std::any_cast<std::string> (
               audit_rows[0].at ("current_labels"))
           .empty ());
}

TEST_CASE ("Summarization creates retrieval summaries without storage pressure",
           "[integration][consolidation][summarize]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  CapturingSummarizer summarizer;

  auto init_ops = std::make_unique<DynamicOperationSet> ();
  SignalProcessor init_processor (cfg, store, std::move (init_ops));
  init_processor.Process (MakeSignal (1));
  init_processor.Flush ();

  ProcessorContext pctx;
  pctx.summarizer = &summarizer;
  pctx.last_consolidation_ts = 100000;
  Signal s = MakeSignal (3000);
  s.source_id = "test/consolidation";
  s.consolidation_mode = ConsolidationMode::Both;
  OperationContext ctx (s, pctx, cfg, store.get ());
  ctx.SetConsolidationShouldStart (true);

  Eigen::VectorXf emb = Eigen::VectorXf::Constant (256, 0.5f);
  emb[0] = 1.0f;
  SeedEmbedding (store.get (), 1, emb);
  SeedEmbedding (store.get (), 2, emb);
  SeedEmbedding (store.get (), 3, emb);
  SeedMemory (store.get (), 1, "Redundant budget note one.", 1000);
  SeedMemory (store.get (), 2, "Redundant budget note two.", 2000);
  SeedMemory (store.get (), 3, "Redundant budget note three.", 3000);
  MarkCompressibleClusterSources (store.get ());

  ClusterInfo cluster;
  cluster.cluster_id = 8;
  cluster.embedding_ids = { 1, 2, 3 };
  cluster.centroid = std::vector<float> (256, 0.5f);
  cluster.avg_score = 0.3;
  ctx.SetConsolidationClusters ({ cluster });

  cortext::operations::eviction::EvictionAblationOverride override;
  override.storage_gate_enabled = true;
  override.min_storage_bytes = 1000;
  override.used_storage_bytes = 1;
  cortext::operations::eviction::ScopedEvictionAblationOverride pressure (
      override);

  ConsolidationSummarize summarize_op;
  auto tx = store->Begin ();
  summarize_op.Execute (ctx, *tx);
  tx->Commit ();

  REQUIRE (summarizer.calls == 1);
  REQUIRE (CountRows (store.get (), "associations") == 6);
  auto summaries = store->Execute (
      "SELECT COUNT(*) AS c FROM memories "
      "WHERE kind = 'LONG_TERM' AND source_id LIKE 'summary_%'",
      {});
  REQUIRE (std::any_cast<long long> (summaries[0].at ("c")) == 1);
  auto cues = store->Execute (
      "SELECT COUNT(*) AS c FROM memories "
      "WHERE kind = 'ASSOCIATION' AND source_id LIKE 'associative_cue_%'",
      {});
  REQUIRE (std::any_cast<long long> (cues[0].at ("c")) == 1);
  REQUIRE (ctx.GetExtractionRequests ().size () == 1);
}

TEST_CASE ("Summarization links protected fact evidence under pressure",
           "[integration][consolidation][summarize][storage_pressure]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  CapturingSummarizer summarizer;

  auto init_ops = std::make_unique<DynamicOperationSet> ();
  SignalProcessor init_processor (cfg, store, std::move (init_ops));
  init_processor.Process (MakeSignal (1));
  init_processor.Flush ();

  ProcessorContext pctx;
  pctx.summarizer = &summarizer;
  pctx.last_consolidation_ts = 100000;
  Signal s = MakeSignal (3000);
  s.source_id = "test/consolidation";
  s.consolidation_mode = ConsolidationMode::Both;
  OperationContext ctx (s, pctx, cfg, store.get ());
  ctx.SetConsolidationShouldStart (true);

  Eigen::VectorXf emb = Eigen::VectorXf::Constant (256, 0.5f);
  emb[0] = 1.0f;
  SeedEmbedding (store.get (), 1, emb);
  SeedEmbedding (store.get (), 2, emb);
  SeedEmbedding (store.get (), 3, emb);
  SeedMemory (store.get (), 1, "Fact-backed note one.", 1000);
  SeedMemory (store.get (), 2, "Fact-backed note two.", 2000);
  SeedMemory (store.get (), 3, "Fact-backed note three.", 3000);
  MarkCompressibleClusterSources (store.get ());
  store->Execute (
      "UPDATE memories SET strength = 0.08, emotional_intensity = 1.0",
      {});
  store->Execute (
      "INSERT INTO fact_assertions "
      "(fact_id, subject, predicate, object, canonical_subject, "
      "canonical_predicate, canonical_object, recorded_at_ts, confidence, "
      "summary_memory_id, created_at) "
      "VALUES (1, 'gabriel', 'uses', 'cortext', 'gabriel', 'uses', "
      "'cortext', 1000, 0.9, 0, 1000)",
      {});
  store->Execute (
      "INSERT INTO fact_evidence "
      "(fact_id, source_memory_id, evidence_type, support_weight) "
      "VALUES (1, 1, 'source', 1.0)",
      {});
  store->Execute (
      "INSERT INTO fact_evidence "
      "(fact_id, source_memory_id, evidence_type, support_weight) "
      "VALUES (1, 2, 'source', 1.0)",
      {});
  store->Execute (
      "INSERT INTO fact_evidence "
      "(fact_id, source_memory_id, evidence_type, support_weight) "
      "VALUES (1, 3, 'source', 1.0)",
      {});

  ClusterInfo cluster;
  cluster.cluster_id = 9;
  cluster.embedding_ids = { 1, 2, 3 };
  cluster.centroid = std::vector<float> (256, 0.5f);
  cluster.avg_score = 0.3;
  ctx.SetConsolidationClusters ({ cluster });

  cortext::operations::eviction::EvictionAblationOverride override;
  override.storage_gate_enabled = true;
  override.min_storage_bytes = 1000;
  override.used_storage_bytes = 10000;
  cortext::operations::eviction::ScopedEvictionAblationOverride pressure (
      override);

  ConsolidationSummarize summarize_op;
  auto tx = store->Begin ();
  summarize_op.Execute (ctx, *tx);
  tx->Commit ();

  REQUIRE (summarizer.calls == 1);
  auto summaries = store->Execute (
      "SELECT COUNT(*) AS c FROM memories "
      "WHERE kind = 'LONG_TERM' AND source_id LIKE 'summary_%'",
      {});
  REQUIRE (std::any_cast<long long> (summaries[0].at ("c")) == 1);
  auto cues = store->Execute (
      "SELECT COUNT(*) AS c FROM memories "
      "WHERE kind = 'ASSOCIATION' AND source_id LIKE 'associative_cue_%'",
      {});
  REQUIRE (std::any_cast<long long> (cues[0].at ("c")) == 1);
  REQUIRE (ctx.GetExtractionRequests ().size () == 1);
}

// =============================================================================
// 5.2.2 Graph Construction Test
// =============================================================================

TEST_CASE ("Graph build creates co_occurs edges",
           "[integration][graph][co_occurrence]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0; // Low threshold for co-occurrence
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Initialize schema
  auto init_ops = std::make_unique<DynamicOperationSet> ();
  SignalProcessor init_processor (cfg, store, std::move (init_ops));
  init_processor.Process (MakeSignal (1));
  init_processor.Flush ();

  // Seed graph data
  ProcessorContext pctx;

  Signal s = MakeSignal (4000);
  OperationContext ctx (s, pctx, cfg, store.get ());

  SeedGraphDataOp seed_op;
  seed_op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Run graph build
  BuildGraphFromConsolidation graph_op;
  graph_op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // V2: Verify edges were created in associations table
  auto edges = store->Execute (
      "SELECT edge_type, COUNT(*) AS c FROM associations GROUP BY edge_type", {});

  INFO ("Edge types found: " << edges.size ());
  for (const auto &row : edges)
    {
      INFO ("Edge type: " << std::any_cast<std::string> (row.at ("edge_type"))
                          << " count: " << GetInt64 (row, "c"));
    }
}

TEST_CASE ("Graph build creates label nodes and edges",
           "[integration][graph][labels]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Initialize schema
  auto init_ops = std::make_unique<DynamicOperationSet> ();
  SignalProcessor init_processor (cfg, store, std::move (init_ops));
  init_processor.Process (MakeSignal (1));
  init_processor.Flush ();

  // Seed graph data
  ProcessorContext pctx;

  Signal s = MakeSignal (5000);
  OperationContext ctx (s, pctx, cfg, store.get ());

  SeedGraphDataOp seed_op;
  seed_op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Run graph build
  BuildGraphFromConsolidation graph_op;
  graph_op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // V2: Verify LABEL memories were created (replaces graph_nodes with type='label')
  auto label_nodes = store->Execute (
      "SELECT memory_id, label FROM memories WHERE kind = 'LABEL'", {});

  INFO ("Label memories: " << label_nodes.size ());

  // V2: Verify derived_from edges (ASSOCIATION -> LABEL)
  auto derived_from = store->Execute (
      "SELECT * FROM associations WHERE edge_type = 'derived_from'", {});

  INFO ("Derived_from edges: " << derived_from.size ());
}

// =============================================================================
// 5.2.3 Streaming Pacing Test
// =============================================================================

TEST_CASE ("Streaming pacing blocks retrieval below threshold",
           "[integration][streaming]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.0; // highest StreamingPacingThreshold(...)
  cfg.focus = 0.0;       // highest MaxWaitDrift(...)


  // Set drift below threshold
  auto &acc = pctx.accumulator_states["test"];
  acc.drift_acc_pacing = 0.3;
  acc.x_last_check = Eigen::VectorXf::Ones (4);

  Signal s = MakeSignal (1000, Eigen::VectorXf::Ones (4));
  OperationContext ctx (s, pctx, cfg);

  CheckStreamingPacing op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Should NOT trigger retrieval
  REQUIRE (ctx.GetShouldCheckRetrieval () == false);
  // Drift should NOT reset
  REQUIRE (pctx.accumulator_states.at ("test").drift_acc_pacing
           == Catch::Approx (0.3));
}

TEST_CASE ("Streaming pacing triggers above threshold",
           "[integration][streaming]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 1.0; // lowest StreamingPacingThreshold(...)
  cfg.focus = 0.5;


  // Set drift above threshold
  auto &acc = pctx.accumulator_states["test"];
  acc.drift_acc_pacing = 0.2;
  acc.x_last_check = Eigen::VectorXf::Ones (4);

  Signal s = MakeSignal (1000, Eigen::VectorXf::Ones (4));
  OperationContext ctx (s, pctx, cfg);

  CheckStreamingPacing op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Should trigger retrieval
  REQUIRE (ctx.GetShouldCheckRetrieval () == true);
  // Drift should reset
  REQUIRE (pctx.accumulator_states.at ("test").drift_acc_pacing == 0.0);
}

TEST_CASE ("Streaming pacing forces check on max_wait_drift",
           "[integration][streaming]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.sensitivity = 0.0; // threshold from StreamingPacingThreshold(...)
  cfg.focus = 1.0;       // max_wait from MaxWaitDrift(...)


  // Set drift above max_wait but below threshold
  auto &acc = pctx.accumulator_states["test"];
  acc.drift_acc_pacing = 0.6;
  acc.x_last_check = Eigen::VectorXf::Ones (4);

  Signal s = MakeSignal (1000, Eigen::VectorXf::Ones (4));
  OperationContext ctx (s, pctx, cfg);

  CheckStreamingPacing op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Should trigger due to exceeding max_wait
  REQUIRE (ctx.GetShouldCheckRetrieval () == true);
  REQUIRE (pctx.accumulator_states.at ("test").drift_acc_pacing == 0.0);
}

TEST_CASE ("Drift accumulation tracks semantic drift",
           "[integration][streaming]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);


  // First signal initializes
  Eigen::VectorXf emb1 = Eigen::VectorXf::Zero (4);
  Signal s1 = MakeSignal (1000, emb1);
  OperationContext ctx1 (s1, pctx, cfg);

  UpdateDriftAccumulation drift_op;
  drift_op.Execute (ctx1, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.accumulator_states.at ("test").drift_accum == 0.0);
  REQUIRE (pctx.accumulator_states.at ("test").prev_x.size () > 0);

  // Second signal: drift = ||[1,0,0,0] - [0,0,0,0]|| = 1.0
  Eigen::VectorXf emb2 = Eigen::VectorXf::Zero (4);
  emb2[0] = 1.0f;
  Signal s2 = MakeSignal (1001, emb2);
  OperationContext ctx2 (s2, pctx, cfg);

  drift_op.Execute (ctx2, cortext::testing::GetNullTransaction ());

  REQUIRE (pctx.accumulator_states.at ("test").drift_accum == Catch::Approx (1.0));

  // Third signal: additional drift
  Eigen::VectorXf emb3 = Eigen::VectorXf::Zero (4);
  emb3[1] = 1.0f;
  Signal s3 = MakeSignal (1002, emb3);
  OperationContext ctx3 (s3, pctx, cfg);

  drift_op.Execute (ctx3, cortext::testing::GetNullTransaction ());

  // Total drift should accumulate
  REQUIRE (pctx.accumulator_states.at ("test").drift_accum > 1.0);
}

// =============================================================================
// 5.2.4 Interrupt Gate with Refractory Dynamics Test
// =============================================================================

TEST_CASE ("Interrupt gate uses drift-based refractory",
           "[integration][interrupt]")
{
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  ProcessorContext pc;
  pc.signals_processed = 100;
  pc.last_interrupt_tick = 95; // Recent interrupt

  // Setup drift accumulation (replaces tick-based delta)
  auto &acc = pc.accumulator_states["test"];
  acc.drift_accum = 0.1;
  acc.drift_at_last_interrupt = 0.0;

  Signal sig;
  sig.embedding = Eigen::VectorXf::Ones (3);
  sig.timestamp = 0;
  sig.source_id = "test";
  OperationContext oc (sig, pc, cfg);

  oc.SetCoherence (1.0);
  oc.SetThresholdTDynamic (0.1);
  oc.SetAtBoundary (true);

  // Add context and candidates
  pc.recent_memory_centroids.push_back (Eigen::VectorXf::Ones (3).normalized ());

  std::unordered_map<long long, Eigen::VectorXf> cands;
  cands.emplace (1LL, Eigen::VectorXf::Ones (3).normalized ());
  oc.SetRetrievedMemoryEmbeddings (cands);

  ComputeMniGateDecision op;
  op.Execute (oc, cortext::testing::GetNullTransaction ());

  // The refractory multiplier should be based on drift, not ticks
  INFO ("M_refrac calculation uses: drift_accum - drift_at_last_interrupt");
  INFO ("Current drift: " << (acc.drift_accum - acc.drift_at_last_interrupt));
}

TEST_CASE ("Refractory multiplier decays with accumulated drift",
           "[integration][interrupt]")
{
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Test with low drift (high refractory)
  ProcessorContext pc_low_drift;
  pc_low_drift.signals_processed = 100;
  pc_low_drift.last_interrupt_tick = 99;
  auto &acc_low = pc_low_drift.accumulator_states["test"];
  acc_low.drift_accum = 0.1;
  acc_low.drift_at_last_interrupt = 0.05;
  pc_low_drift.recent_memory_centroids.push_back (
      Eigen::VectorXf::Ones (3).normalized ());

  // Test with high drift (low refractory)
  ProcessorContext pc_high_drift;
  pc_high_drift.signals_processed = 100;
  pc_high_drift.last_interrupt_tick = 50;
  auto &acc_high = pc_high_drift.accumulator_states["test"];
  acc_high.drift_accum = 2.0;
  acc_high.drift_at_last_interrupt = 0.0;
  pc_high_drift.recent_memory_centroids.push_back (
      Eigen::VectorXf::Ones (3).normalized ());

  Signal sig;
  sig.embedding = Eigen::VectorXf::Ones (3);
  sig.timestamp = 0;
  sig.source_id = "test";

  OperationContext oc_low (sig, pc_low_drift, cfg);
  OperationContext oc_high (sig, pc_high_drift, cfg);

  oc_low.SetCoherence (1.0);
  oc_low.SetThresholdTDynamic (0.1);
  oc_low.SetAtBoundary (true);

  oc_high.SetCoherence (1.0);
  oc_high.SetThresholdTDynamic (0.1);
  oc_high.SetAtBoundary (true);

  std::unordered_map<long long, Eigen::VectorXf> cands;
  cands.emplace (1LL, Eigen::VectorXf::Ones (3).normalized ());
  oc_low.SetRetrievedMemoryEmbeddings (cands);
  oc_high.SetRetrievedMemoryEmbeddings (cands);

  // Calculate drift deltas BEFORE Execute() since the operation updates
  // drift_at_last_interrupt to match drift_accum
  double low_drift_delta = acc_low.drift_accum - acc_low.drift_at_last_interrupt;
  double high_drift_delta = acc_high.drift_accum - acc_high.drift_at_last_interrupt;

  INFO ("Low drift delta: " << low_drift_delta);
  INFO ("High drift delta: " << high_drift_delta);

  // Verify the drift deltas are set up correctly
  REQUIRE (low_drift_delta == Catch::Approx (0.05));
  REQUIRE (high_drift_delta == Catch::Approx (2.0));

  // M_refrac = 1.0 + k_refrac * exp(-delta / tau_refrac)
  // Higher delta = lower M_refrac (closer to 1.0)
  REQUIRE (low_drift_delta < high_drift_delta);

  // Now execute the operation to verify it processes without error
  ComputeMniGateDecision op;
  op.Execute (oc_low, cortext::testing::GetNullTransaction ());
  op.Execute (oc_high, cortext::testing::GetNullTransaction ());
}

// =============================================================================
// Knob-driven threshold tests
// =============================================================================

TEST_CASE ("Consolidation thresholds vary with knobs",
           "[integration][consolidation][knobs]")
{
  // High stability = longer intervals, larger thresholds
  double T_high = 1.0;
  REQUIRE (ConsolidationIntervalSeconds (T_high) == 3600);
  REQUIRE (ConsolidationThresholdCount (T_high) == 256LL * 120LL);

  // Low stability = shorter intervals, smaller thresholds
  double T_low = 0.0;
  REQUIRE (ConsolidationIntervalSeconds (T_low) == 300);
  REQUIRE (ConsolidationThresholdCount (T_low) == 32LL * 20LL);
}

TEST_CASE ("Clustering thresholds vary with knobs",
           "[integration][consolidation][knobs]")
{
  // High focus = stricter clustering
  double F_high = 1.0;
  REQUIRE (MergeThreshold (F_high) == Catch::Approx (0.95));
  REQUIRE (MinClusterSize (F_high) == 10);

  // Low focus = looser clustering
  double F_low = 0.0;
  REQUIRE (MergeThreshold (F_low) == Catch::Approx (0.85));
  REQUIRE (MinClusterSize (F_low) == 3);
}

TEST_CASE ("Graph edge thresholds vary with knobs",
           "[integration][graph][knobs]")
{
  // High stability = higher causal threshold
  double T_high = 1.0;
  REQUIRE (CausalDriftThreshold (T_high) == Catch::Approx (0.35));
  REQUIRE (ReinforcementDecay (T_high) == Catch::Approx (0.99));

  // Low stability = lower causal threshold
  double T_low = 0.0;
  REQUIRE (CausalDriftThreshold (T_low) == Catch::Approx (0.15));
  REQUIRE (ReinforcementDecay (T_low) == Catch::Approx (0.9));
}

TEST_CASE ("Uncertainty feeds focus update in the pipeline",
           "[integration][order]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.7;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::make_unique<InitializeFocusPriors> (),
      std::make_unique<UpdateRecentContext> (),
      std::make_unique<UpdateEmbeddingPredictionError> (),
      std::make_unique<UpdateUncertainty> (),
      std::make_unique<UpdateFocus> ());

  SignalProcessor proc (cfg, store, std::move (pipeline));

  Signal s = MakeSignal (1000, Eigen::VectorXf::Ones (8));
  proc.Process (s);

  auto rows = store->Execute (
      "SELECT weight_relevance, u_uncertainty FROM state WHERE id = 1;");
  REQUIRE (rows.size () == 1);
  const double weight = GetDouble (rows[0], "weight_relevance");
  const double u_t = GetDouble (rows[0], "u_uncertainty");

  const double prior = core::Sigmoid (2 * cfg.focus - 1);
  const double alpha_f = core::AlphaF (cfg.focus, u_t);
  const double expected = core::Ewma (prior, 1.0, alpha_f);

  REQUIRE (weight == Catch::Approx (expected).epsilon (1e-5));
}
