#include "cortext/operations/interrupt_gate.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/processor/processor_context.hpp"
#include "cortext/signal.hpp"
#include "test_helpers.hpp"
#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace cortext;
using cortext::operations::ComputeMniGateDecision;

namespace
{

Signal
MakeSignal (int dim)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (dim);
  s.timestamp = 0;
  s.source_id = "test";
  return s;
}

Eigen::VectorXf
MakeUnit (std::initializer_list<float> vals)
{
  Eigen::VectorXf v (static_cast<int> (vals.size ()));
  int i = 0;
  for (auto f : vals)
    v[i++] = f;
  const float n = v.norm ();
  if (n > 0.0f)
    v /= n;
  return v;
}

} // namespace

TEST_CASE ("Alg27 allows on MU path", "[operations][interrupt_gate]")
{
  // Setup config (moderate F,S,T)
  SignalProcessor::Config cfg;
  cfg.focus = 0.7;
  cfg.sensitivity = 0.6;
  cfg.stability = 0.5;

  ProcessorContext pc;
  pc.signals_processed = 100; // far from last interrupt
  pc.last_interrupt_tick = 0;

  // Recent context centroid toward +X
  pc.recent_context_embeddings.push_back (MakeUnit ({ 1.0f, 0.0f, 0.0f }));

  // No included set to maximize novelty/relevance for this allow case

  // Create op-context
  auto sig = MakeSignal (3);
  OperationContext oc (sig, pc, cfg);

  // Coherence high (low penalty), threshold very low to focus on MU/Jaccard
  oc.SetCoherence (1.0);
  oc.SetThresholdTDynamic (0.0);
  oc.SetAtBoundary (true);

  // Candidate strongly aligned with centroid; low redundancy
  std::unordered_map<long long, Eigen::VectorXf> cands;
  cands.emplace (1LL, MakeUnit ({ 0.95f, 0.05f, 0.0f }));
  oc.SetRetrievedMemoryEmbeddings (cands);

  ComputeMniGateDecision op;
  cortext::testing::NullTransaction null_tx;
  op.Execute (oc, null_tx);

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
  SignalProcessor::Config cfg;
  cfg.focus = 0.9; // high dup threshold
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  ProcessorContext pc;
  pc.signals_processed = 50;
  pc.last_interrupt_tick = -1000;
  pc.recent_context_embeddings.push_back (MakeUnit ({ 1.0f, 0.0f }));

  // Included contains a vector almost identical to candidate
  ProcessorContext::WMSlot slot;
  slot.embedding = MakeUnit ({ 0.7f, 0.3f });
  slot.strength = 1.0;
  pc.wm_slots.push_back (slot);

  auto sig = MakeSignal (2);
  OperationContext oc (sig, pc, cfg);
  oc.SetCoherence (1.0);
  oc.SetThresholdTDynamic (0.2);
  oc.SetAtBoundary (true);

  std::unordered_map<long long, Eigen::VectorXf> cands;
  cands.emplace (1LL, MakeUnit ({ 0.71f, 0.29f }));
  oc.SetRetrievedMemoryEmbeddings (cands);

  ComputeMniGateDecision op;
  cortext::testing::NullTransaction null_tx;
  op.Execute (oc, null_tx);

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
  SignalProcessor::Config cfg;
  cfg.focus = 0.9; // tighten dup and increase boundary_mult
  cfg.sensitivity = 0.5;
  cfg.stability = 0.8; // long refractory

  ProcessorContext pc;
  pc.signals_processed = 101;
  pc.last_interrupt_tick = 100; // very recent
  pc.recent_context_embeddings.push_back (MakeUnit ({ 0.0f, 1.0f }));

  auto sig = MakeSignal (2);
  OperationContext oc (sig, pc, cfg);
  oc.SetCoherence (1.0);
  oc.SetThresholdTDynamic (0.4); // require higher relevance
  oc.SetAtBoundary (false);      // force boundary multiplier path

  // Candidate aligns well, but refractory should raise thresholds
  std::unordered_map<long long, Eigen::VectorXf> cands;
  cands.emplace (1LL, MakeUnit ({ 0.0f, 1.0f }));
  oc.SetRetrievedMemoryEmbeddings (cands);

  ComputeMniGateDecision op;
  cortext::testing::NullTransaction null_tx;
  op.Execute (oc, null_tx);

  REQUIRE (oc.GetMniTauMuEff () > 0.0);
  // With Delta small, thresholds elevated; decision may still allow
  REQUIRE (oc.GetInterruptAllowed () == true);
}

TEST_CASE ("Alg27 Jaccard novelty computed vs LRU set",
           "[operations][interrupt_gate]")
{
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  ProcessorContext pc;
  pc.signals_processed = 10;
  pc.last_interrupt_tick = -1000;
  pc.recent_context_embeddings.push_back (MakeUnit ({ 1.0f, 0.0f }));
  // Seed LRU set B = {1,2,3}
  pc.recent_ids_lru_.RecordIds ({ 1LL, 2LL, 3LL });

  auto sig = MakeSignal (2);
  OperationContext oc (sig, pc, cfg);
  oc.SetCoherence (1.0);
  oc.SetThresholdTDynamic (0.0);
  oc.SetAtBoundary (true);

  // Candidate A = {2,3,4,5}
  std::unordered_map<long long, Eigen::VectorXf> cands;
  cands.emplace (2LL, MakeUnit ({ 1.0f, 0.0f }));
  cands.emplace (3LL, MakeUnit ({ 0.0f, 1.0f }));
  cands.emplace (4LL, MakeUnit ({ 1.0f, 0.0f }));
  cands.emplace (5LL, MakeUnit ({ 0.0f, 1.0f }));
  oc.SetRetrievedMemoryEmbeddings (cands);

  ComputeMniGateDecision op;
  cortext::testing::NullTransaction null_tx;
  op.Execute (oc, null_tx);

  // |A∩B| = 2, |A∪B| = 5 => novelty = 1 - 2/5 = 0.6
  REQUIRE (oc.GetMniJaccard () == Catch::Approx (0.6).margin (1e-6));
}
