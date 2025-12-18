#include "cortext/operations/interrupt_gate.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/processor/processor_context.hpp"
#include "cortext/signal.hpp"
#include "cortext/store/sqlite_store.hpp"
#include "test_helpers.hpp"
#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace cortext;
using cortext::operations::ComputeMniGateDecision;

namespace
{

// Embedding dimension expected by schema
constexpr int kEmbeddingDim = 256;

// Signal timestamp used in tests - embeddings created before this are eligible
constexpr uint64_t kTestSignalTimestamp = 1000;

Signal
MakeSignal ()
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (kEmbeddingDim);
  s.timestamp = kTestSignalTimestamp;
  s.source_id = "test";
  return s;
}

// Create a 256D embedding from sparse values and normalize
Eigen::VectorXf
MakeUnit256 (std::initializer_list<float> first_vals)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  int i = 0;
  for (auto f : first_vals)
    {
      if (i < kEmbeddingDim)
        v[i++] = f;
    }
  const float n = v.norm ();
  if (n > 0.0f)
    v /= n;
  return v;
}

// Helper to insert a test embedding with a specific created_at timestamp
void
InsertTestEmbedding (Store &store, long long id, const Eigen::VectorXf &emb,
                     long long created_at = 0)
{
  std::vector<float> emb_data (emb.data (), emb.data () + emb.size ());
  // v2: Insert into embeddings (minimal vec0 table)
  store.Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                 "VALUES (?, ?, ?)",
                 { id, emb_data, created_at });
  // v2: Insert into memories (comprehensive metadata)
  store.Execute ("INSERT INTO memories (memory_id, embedding_id, source_id, kind, "
                 "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
                 "VALUES (?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, 1.0, ?)",
                 { id, id, created_at, created_at });
}

// Creates an in-memory store with initialized schema
std::shared_ptr<Store>
CreateTestStore ()
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  return store;
}

} // namespace

TEST_CASE ("Alg27 allows on MU path", "[operations][interrupt_gate]")
{
  auto store = CreateTestStore ();

  // Setup config (moderate F,S,T)
  SignalProcessor::Config cfg;
  cfg.focus = 0.7;
  cfg.sensitivity = 0.6;
  cfg.stability = 0.5;

  ProcessorContext pc;
  pc.signals_processed = 100; // far from last interrupt
  pc.last_interrupt_tick = 0;

  // Section 8.2: Recent memory centroids (μ_acc) toward +X
  pc.recent_memory_centroids.push_back (MakeUnit256 ({ 1.0f, 0.0f, 0.0f }));

  // No included set to maximize novelty/relevance for this allow case

  // Insert candidate embedding into store (created_at=0, before signal timestamp)
  auto cand_emb = MakeUnit256 ({ 0.95f, 0.05f, 0.0f });
  InsertTestEmbedding (*store, 1LL, cand_emb, 0);

  // Create op-context with store
  auto sig = MakeSignal ();
  OperationContext oc (sig, pc, cfg, store.get ());

  // Coherence high (low penalty), threshold very low to focus on MU/Jaccard
  oc.SetCoherence (1.0);
  oc.SetThresholdTDynamic (0.0);
  oc.SetAtBoundary (true);

  // Candidate strongly aligned with centroid; low redundancy
  std::unordered_map<long long, Eigen::VectorXf> cands;
  cands.emplace (1LL, cand_emb);
  oc.SetRetrievedMemoryEmbeddings (cands);

  ComputeMniGateDecision op;
  auto tx = store->Begin ();
  op.Execute (oc, *tx);

  CAPTURE (oc.GetMniBestMu ());
  CAPTURE (oc.GetMniTauMuEff ());
  CAPTURE (oc.GetMniTauJaccardEff ());
  CAPTURE (oc.GetMniDupThresh ());

  REQUIRE (oc.GetInterruptAllowed () == true);
  REQUIRE (oc.GetMniBestMu () >= oc.GetMniTauMuEff ());
  REQUIRE (oc.GetMniDupThresh () > 0.0);
}

TEST_CASE ("Alg27 duplicate suppression", "[operations][interrupt_gate]")
{
  auto store = CreateTestStore ();

  SignalProcessor::Config cfg;
  cfg.focus = 0.9; // high dup threshold
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  ProcessorContext pc;
  pc.signals_processed = 50;
  pc.last_interrupt_tick = -1000;
  // Section 8.2: Recent memory centroids
  pc.recent_memory_centroids.push_back (MakeUnit256 ({ 1.0f, 0.0f }));

  // Included contains a vector almost identical to candidate
  ProcessorContext::WMSlot slot;
  slot.embedding = MakeUnit256 ({ 0.7f, 0.3f });
  slot.strength = 1.0;
  pc.wm_slots.push_back (slot);

  // Insert candidate embedding into store
  auto cand_emb = MakeUnit256 ({ 0.71f, 0.29f });
  InsertTestEmbedding (*store, 1LL, cand_emb, 0);

  auto sig = MakeSignal ();
  OperationContext oc (sig, pc, cfg, store.get ());
  oc.SetCoherence (1.0);
  oc.SetThresholdTDynamic (0.2);
  oc.SetAtBoundary (true);

  std::unordered_map<long long, Eigen::VectorXf> cands;
  cands.emplace (1LL, cand_emb);
  oc.SetRetrievedMemoryEmbeddings (cands);

  ComputeMniGateDecision op;
  auto tx = store->Begin ();
  op.Execute (oc, *tx);

  CAPTURE (oc.GetMniBestMu ());
  CAPTURE (oc.GetMniTauMuEff ());
  CAPTURE (oc.GetMniTauJaccardEff ());
  CAPTURE (oc.GetMniDupThresh ());

  REQUIRE (oc.GetInterruptAllowed () == false);
  REQUIRE (oc.GetMniBestMu ()
           < oc.GetMniDupThresh ()); // overlap should dominate
}

TEST_CASE ("Alg27 refractory raises thresholds",
           "[operations][interrupt_gate]")
{
  auto store = CreateTestStore ();

  SignalProcessor::Config cfg;
  cfg.focus = 0.9; // tighten dup and increase boundary_mult
  cfg.sensitivity = 0.5;
  cfg.stability = 0.8; // long refractory

  ProcessorContext pc;
  pc.signals_processed = 101;
  pc.last_interrupt_tick = 100; // very recent
  // Section 8.2: Recent memory centroids
  pc.recent_memory_centroids.push_back (MakeUnit256 ({ 0.0f, 1.0f }));

  // Insert candidate embedding into store
  auto cand_emb = MakeUnit256 ({ 0.0f, 1.0f });
  InsertTestEmbedding (*store, 1LL, cand_emb, 0);

  auto sig = MakeSignal ();
  OperationContext oc (sig, pc, cfg, store.get ());
  oc.SetCoherence (1.0);
  oc.SetThresholdTDynamic (0.4); // require higher relevance
  oc.SetAtBoundary (false);      // force boundary multiplier path

  // Candidate aligns well, but refractory should raise thresholds
  std::unordered_map<long long, Eigen::VectorXf> cands;
  cands.emplace (1LL, cand_emb);
  oc.SetRetrievedMemoryEmbeddings (cands);

  ComputeMniGateDecision op;
  auto tx = store->Begin ();
  op.Execute (oc, *tx);

  REQUIRE (oc.GetMniTauMuEff () > 0.0);
  // With Delta small, thresholds elevated; decision may still allow
  REQUIRE (oc.GetInterruptAllowed () == true);
}

TEST_CASE ("Alg27 embedding novelty high for orthogonal candidate",
           "[operations][interrupt_gate]")
{
  auto store = CreateTestStore ();

  // Test that embedding novelty is high (~1.0) when candidate is orthogonal
  // to all embeddings in the context window.
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  ProcessorContext pc;
  pc.signals_processed = 10;
  pc.last_interrupt_tick = -1000;

  // Section 8.2: Context window with two memory centroids in X-Y plane
  pc.recent_memory_centroids.push_back (MakeUnit256 ({ 1.0f, 0.0f, 0.0f }));
  pc.recent_memory_centroids.push_back (MakeUnit256 ({ 0.0f, 1.0f, 0.0f }));

  // Insert candidate embedding into store
  auto cand_emb = MakeUnit256 ({ 0.0f, 0.0f, 1.0f });
  InsertTestEmbedding (*store, 1LL, cand_emb, 0);

  auto sig = MakeSignal ();
  OperationContext oc (sig, pc, cfg, store.get ());
  oc.SetCoherence (1.0);
  oc.SetThresholdTDynamic (0.0);
  oc.SetAtBoundary (true);

  // Candidate orthogonal to context window (in Z direction)
  std::unordered_map<long long, Eigen::VectorXf> cands;
  cands.emplace (1LL, cand_emb);
  oc.SetRetrievedMemoryEmbeddings (cands);

  ComputeMniGateDecision op;
  auto tx = store->Begin ();
  op.Execute (oc, *tx);

  // Embedding novelty = 1 - max(cos) = 1 - 0 = 1.0 (orthogonal)
  // MniJaccard now stores embedding novelty
  REQUIRE (oc.GetMniJaccard () == Catch::Approx (1.0).margin (0.01));
}

TEST_CASE ("Alg27 embedding novelty low for similar candidate",
           "[operations][interrupt_gate]")
{
  auto store = CreateTestStore ();

  // Test that embedding novelty is low (~0.0) when candidate is very
  // similar to an embedding in the context window.
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  ProcessorContext pc;
  pc.signals_processed = 10;
  pc.last_interrupt_tick = -1000;

  // Section 8.2: Context window with one memory centroid
  pc.recent_memory_centroids.push_back (MakeUnit256 ({ 1.0f, 0.0f, 0.0f }));

  // Insert candidate embedding into store
  auto cand_emb = MakeUnit256 ({ 0.99f, 0.1f, 0.0f });
  InsertTestEmbedding (*store, 1LL, cand_emb, 0);

  auto sig = MakeSignal ();
  OperationContext oc (sig, pc, cfg, store.get ());
  oc.SetCoherence (1.0);
  oc.SetThresholdTDynamic (0.0);
  oc.SetAtBoundary (true);

  // Candidate very similar to context (almost same direction)
  std::unordered_map<long long, Eigen::VectorXf> cands;
  cands.emplace (1LL, cand_emb);
  oc.SetRetrievedMemoryEmbeddings (cands);

  ComputeMniGateDecision op;
  auto tx = store->Begin ();
  op.Execute (oc, *tx);

  // Embedding novelty should be close to 0 (high similarity)
  // MniJaccard now stores embedding novelty
  REQUIRE (oc.GetMniJaccard () < 0.1);
}

TEST_CASE ("Alg27 embedding novelty 1.0 when context window empty",
           "[operations][interrupt_gate]")
{
  auto store = CreateTestStore ();

  // Test that embedding novelty defaults to 1.0 when context window is empty.
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  ProcessorContext pc;
  pc.signals_processed = 10;
  pc.last_interrupt_tick = -1000;

  // Empty context window, but provide included vectors for centroid
  ProcessorContext::WMSlot slot;
  slot.embedding = MakeUnit256 ({ 1.0f, 0.0f, 0.0f });
  slot.strength = 1.0;
  pc.wm_slots.push_back (slot);

  // Insert candidate embedding into store
  auto cand_emb = MakeUnit256 ({ 0.5f, 0.5f, 0.0f });
  InsertTestEmbedding (*store, 1LL, cand_emb, 0);

  auto sig = MakeSignal ();
  OperationContext oc (sig, pc, cfg, store.get ());
  oc.SetCoherence (1.0);
  oc.SetThresholdTDynamic (0.0);
  oc.SetAtBoundary (true);

  std::unordered_map<long long, Eigen::VectorXf> cands;
  cands.emplace (1LL, cand_emb);
  oc.SetRetrievedMemoryEmbeddings (cands);

  ComputeMniGateDecision op;
  auto tx = store->Begin ();
  op.Execute (oc, *tx);

  // With empty context window, all candidates are fully novel
  REQUIRE (oc.GetMniJaccard () == Catch::Approx (1.0).margin (0.01));
}

TEST_CASE ("Alg27 Section 8.3.1 Write Exclusion Filter prevents self-triggering",
           "[operations][interrupt_gate]")
{
  auto store = CreateTestStore ();

  // Test that candidates stored during the current signal processing cycle
  // (created_at >= signal.timestamp) are excluded from interrupt consideration.
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  ProcessorContext pc;
  pc.signals_processed = 10;
  pc.last_interrupt_tick = -1000;

  // Section 8.2: Context window uses recent_memory_centroids for centroid
  pc.recent_memory_centroids.push_back (MakeUnit256 ({ 1.0f, 0.0f, 0.0f }));

  // Create candidate embedding that would normally be allowed
  auto cand_emb = MakeUnit256 ({ 0.95f, 0.05f, 0.0f });

  // Insert embedding with created_at = kTestSignalTimestamp (same as signal)
  // This simulates a memory stored during the current signal processing cycle.
  InsertTestEmbedding (*store, 1LL, cand_emb,
                       static_cast<long long> (kTestSignalTimestamp));

  auto sig = MakeSignal ();
  OperationContext oc (sig, pc, cfg, store.get ());
  oc.SetCoherence (1.0);
  oc.SetThresholdTDynamic (0.0);
  oc.SetAtBoundary (true);

  // Even though we set retrieved embeddings, the candidate should be filtered
  // out because its created_at >= signal.timestamp
  std::unordered_map<long long, Eigen::VectorXf> cands;
  cands.emplace (1LL, cand_emb);
  oc.SetRetrievedMemoryEmbeddings (cands);

  ComputeMniGateDecision op;
  auto tx = store->Begin ();
  op.Execute (oc, *tx);

  // Interrupt should NOT be allowed because all candidates were filtered out
  REQUIRE (oc.GetInterruptAllowed () == false);
}

TEST_CASE ("Alg27 Section 8.3.1 mixes eligible and ineligible candidates",
           "[operations][interrupt_gate]")
{
  auto store = CreateTestStore ();

  // Test with a mix of eligible (old) and ineligible (new) candidates.
  // Only the eligible candidate should be considered.
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  ProcessorContext pc;
  pc.signals_processed = 100;
  pc.last_interrupt_tick = 0;

  // Section 8.2: Context centroid from recent memory centroids toward +X
  pc.recent_memory_centroids.push_back (MakeUnit256 ({ 1.0f, 0.0f, 0.0f }));

  // Candidate 1: Old embedding (eligible) - strongly aligned
  auto cand1_emb = MakeUnit256 ({ 0.95f, 0.05f, 0.0f });
  InsertTestEmbedding (*store, 1LL, cand1_emb, 0); // created_at = 0, before signal

  // Candidate 2: New embedding (ineligible) - also strongly aligned
  auto cand2_emb = MakeUnit256 ({ 0.98f, 0.02f, 0.0f });
  InsertTestEmbedding (*store, 2LL, cand2_emb,
                       static_cast<long long> (kTestSignalTimestamp + 100));

  auto sig = MakeSignal ();
  OperationContext oc (sig, pc, cfg, store.get ());
  oc.SetCoherence (1.0);
  oc.SetThresholdTDynamic (0.0);
  oc.SetAtBoundary (true);

  // Include both candidates in retrieved set
  std::unordered_map<long long, Eigen::VectorXf> cands;
  cands.emplace (1LL, cand1_emb);
  cands.emplace (2LL, cand2_emb);
  oc.SetRetrievedMemoryEmbeddings (cands);

  ComputeMniGateDecision op;
  auto tx = store->Begin ();
  op.Execute (oc, *tx);

  // Should still allow interrupt because candidate 1 is eligible
  // (Only candidate 2 was filtered out)
  REQUIRE (oc.GetInterruptAllowed () == true);
}
