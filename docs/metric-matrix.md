# Metric Matrix

This document is the ablation-planning map for Cortext's paper-defined metrics.

It is not a prose rehash of the paper. It answers a narrower question:

> For each metric, where is it computed, where is it consumed, what can be toggled independently, and which storage / hydration surfaces are trustworthy for evaluation?

Use this file to define ablation dimensions before building a full factorial or staged sweep.

## Scope

The matrix covers:

- the **12 composite-score input metrics** from the paper
- the **3 derived structural metrics** that feed multiple downstream algorithms
- the main **decision surfaces** built from them
- the **storage / hydration surfaces** that are safe to score during ablations

It does **not** attempt to enumerate the full power set of all possible metric toggles. That would be combinatorially useless. Instead, it enumerates the **atomic application sites** you can factor into a bench.

## Legend

| term | meaning |
|---|---|
| `compute` | the metric is produced here |
| `blend` | the metric contributes to the composite score blender |
| `gate` | the metric changes a threshold / binary decision |
| `rank` | the metric changes retrieval ordering |
| `feedback` | the metric changes a learned or adaptive state |
| `persist` | the metric is written to storage |
| `hydrate` | the metric is available on a hydrated surface |
| `safe` | suitable as a bench dimension by itself |
| `paired` | should usually be tested together with another dimension |

## Surface Truth Table

This table matters for scoring. Not every surface exposes the same metric fidelity.

| surface | what is actually available | safe for ablation scoring? | notes |
|---|---|---|---|
| `OperationContext.metrics` | full live metric map for the current signal | `yes` | source of truth during processing |
| `SignalProcessor::Output.metrics` | full live metric map for the processed signal | `yes` | best public-facing runtime surface for per-step metric scoring |
| `signals` table | all 12 core metrics plus `score`, `threshold_t`, `coherence`, `focus_spread`, `f_effective` | `yes` | source of truth for persisted per-signal metrics |
| `working_memory` hydration | `score <- s_avg`, `salience <- s_max`, `arousal <- s_arousal_avg`, `retrieved_count`, `used_count` | `yes`, but only for those fields | slot summaries only; not a full metric vector |
| `retrieved_memory` hydration | content + counts, but not full stored metric hydration | `no` for persisted metric scoring | use live retrieval debug / output surfaces instead |
| chat chunk diagnostics | live interrupt / retrieval fields such as retrieved memory relevance and composite score | `yes` | best chat-side scoring surface for retrieval / interrupt studies |
| chat working-memory slot panel | slot summaries only | `yes`, but only for slot summary fields | do not treat it as a full metric vector |

## Metric × Operation Matrix

Each row is one real `metric × application site` combination.

| metric | producer | consumer / operation | apply mode | ablation dimension | factorability | paper anchor | code anchor |
|---|---|---|---|---|---|---|---|
| `relevance` | `ComputeMetrics` | `ComputeCompositeScore` | `blend` | `blend.relevance` | `safe` | `§3 Structural Metrics` | `src/operations/metrics.cpp`, `src/operations/blend.cpp` |
| `relevance` | `ComputeMetrics` | `FitMetricWeightsRLS` | `feedback` | `rls.relevance` | `paired` | `§3`, `§2 Core Adaptation` | `src/operations/metrics.cpp`, `src/operations/blend.cpp` |
| `relevance` | `ComputeMetrics` | `PersistSignalMetrics` | `persist` | `persist.relevance` | `safe` | `§10 Implementation` | `src/operations/signal_metrics_persistence.cpp` |
| `mismatch` | `ComputeMetrics` | `ComputeCompositeScore` | `blend` | `blend.mismatch` | `safe` | `§3 Structural Metrics` | `src/operations/metrics.cpp`, `src/operations/blend.cpp` |
| `mismatch` | `ComputeMetrics` | `FitMetricWeightsRLS` | `feedback` | `rls.mismatch` | `paired` | `§3`, `§2` | `src/operations/blend.cpp` |
| `mismatch` | `ComputeMetrics` | `PersistSignalMetrics` | `persist` | `persist.mismatch` | `safe` | `§10` | `src/operations/signal_metrics_persistence.cpp` |
| `surprise` | `ComputeMetrics` | `ComputeCompositeScore` | `blend` | `blend.surprise` | `safe` | `§3 Structural Metrics` | `src/operations/metrics.cpp`, `src/operations/blend.cpp` |
| `surprise` | `ComputeMetrics` | `FitMetricWeightsRLS` | `feedback` | `rls.surprise` | `paired` | `§3`, `§2` | `src/operations/blend.cpp` |
| `surprise` | `ComputeMetrics` | `ApplyPredictivePreActivation` | `feedback` | `predictive.surprise` | `safe` | `§6 Advanced Cognitive` | `src/operations/predictive.cpp` |
| `surprise` | `ComputeMetrics` | `PersistSignalMetrics` | `persist` | `persist.surprise` | `safe` | `§10` | `src/operations/signal_metrics_persistence.cpp` |
| `rarity` | `ComputeMetrics` | `ComputeCompositeScore` | `blend` | `blend.rarity` | `safe` | `§3 Structural Metrics` | `src/operations/metrics.cpp`, `src/operations/blend.cpp` |
| `rarity` | `ComputeMetrics` | `FitMetricWeightsRLS` | `feedback` | `rls.rarity` | `paired` | `§3`, `§2` | `src/operations/blend.cpp` |
| `rarity` | `ComputeMetrics` | `ApplySerialPositionEffects` | `gate` | `serial.rarity` | `safe` | `§6 Serial Position Effects` | `src/operations/serial_position_apply.cpp` |
| `rarity` | `ComputeMetrics` | `UpdateNeuromodulators` | `feedback` | `neuromod.rarity` | `safe` | `§2 Neuromodulators` | `src/operations/neuromodulators.cpp` |
| `rarity` | `ComputeMetrics` | `PersistSignalMetrics` | `persist` | `persist.rarity` | `safe` | `§10` | `src/operations/signal_metrics_persistence.cpp` |
| `drift` | `ComputeMetrics` | `ComputeCompositeScore` | `blend` | `blend.drift` | `safe` | `§3 Structural Metrics` | `src/operations/metrics.cpp`, `src/operations/blend.cpp` |
| `drift` | `ComputeMetrics` | `FitMetricWeightsRLS` | `feedback` | `rls.drift` | `paired` | `§3`, `§2` | `src/operations/blend.cpp` |
| `drift` | `ComputeMetrics` | `PersistSignalMetrics` | `persist` | `persist.drift` | `safe` | `§10` | `src/operations/signal_metrics_persistence.cpp` |
| `contradiction` | `ComputeMetrics` | `ComputeCompositeScore` | `blend` | `blend.contradiction` | `safe` | `§3 Structural Metrics` | `src/operations/metrics.cpp`, `src/operations/blend.cpp` |
| `contradiction` | `ComputeMetrics` | `FitMetricWeightsRLS` | `feedback` | `rls.contradiction` | `paired` | `§3`, `§2` | `src/operations/blend.cpp` |
| `contradiction` | `ComputeMetrics` | `PersistSignalMetrics` | `persist` | `persist.contradiction` | `safe` | `§10` | `src/operations/signal_metrics_persistence.cpp` |
| `utility` | `ComputeMetrics` | `ComputeCompositeScore` | `blend` | `blend.utility` | `safe` | `§3 Structural Metrics` | `src/operations/metrics.cpp`, `src/operations/blend.cpp` |
| `utility` | `ComputeMetrics` | `FitMetricWeightsRLS` | `feedback` | `rls.utility` | `paired` | `§3`, `§2` | `src/operations/blend.cpp` |
| `utility` | `ComputeMetrics` | `PersistSignalMetrics` | `persist` | `persist.utility` | `safe` | `§10` | `src/operations/signal_metrics_persistence.cpp` |
| `periphery` | `ComputeMetrics` | `ComputeCompositeScore` | `blend` | `blend.periphery` | `safe` | `§3 Structural Metrics` | `src/operations/metrics.cpp`, `src/operations/blend.cpp` |
| `periphery` | `ComputeMetrics` | `PersistSignalMetrics` | `persist` | `persist.periphery` | `safe` | `§10` | `src/operations/signal_metrics_persistence.cpp` |
| `coverage` | `ComputeMetrics` | `ComputeCompositeScore` | `blend` | `blend.coverage` | `safe` | `§3 Structural Metrics` | `src/operations/metrics.cpp`, `src/operations/blend.cpp` |
| `coverage` | `ComputeMetrics` | `FitMetricWeightsRLS` | `feedback` | `rls.coverage` | `paired` | `§3`, `§2` | `src/operations/blend.cpp` |
| `coverage` | `ComputeMetrics` | `PersistSignalMetrics` | `persist` | `persist.coverage` | `safe` | `§10` | `src/operations/signal_metrics_persistence.cpp` |
| `coverage` | accumulator state | `ComputeWriteGate` | `gate` | `write_gate.coverage` | `safe` | `§4 Dynamic Thresholding` | `src/operations/write_gate.cpp` |
| `salience` | `ComputeMetrics` | `ComputeCompositeScore` | `blend` | `blend.salience` | `safe` | `§3 Structural Metrics` | `src/operations/metrics.cpp`, `src/operations/blend.cpp` |
| `salience` | `ComputeMetrics` | `GraphAugmentedRetrieveCandidates` | `rank` | `retrieval.salience_affect` | `safe` | `§7 Consolidation`, `§8 Interrupt Gate` | `src/operations/graph_retrieval.cpp` |
| `salience` | `ComputeMetrics` | `ComputeMniGateDecision` | `gate` | `interrupt.salience_affect` | `safe` | `§8 Interrupt Gate` | `src/operations/interrupt_gate.cpp` |
| `salience` | `ComputeMetrics` | `PersistSignalMetrics` | `persist` | `persist.salience` | `safe` | `§10` | `src/operations/signal_metrics_persistence.cpp` |
| `valence` | `ComputeMetrics` | `ComputeCompositeScore` | `blend` | `blend.valence` | `safe` | `§2 Emotional Projection`, `§3 Structural Metrics` | `src/operations/metrics.cpp`, `src/operations/blend.cpp` |
| `valence` | `ComputeMetrics` | `PersistSignalMetrics` | `persist` | `persist.valence` | `safe` | `§10` | `src/operations/signal_metrics_persistence.cpp` |
| `arousal` | `ComputeMetrics` | `ComputeCompositeScore` | `blend` | `blend.arousal` | `safe` | `§2 Emotional Projection`, `§3 Structural Metrics` | `src/operations/metrics.cpp`, `src/operations/blend.cpp` |
| `arousal` | `ComputeMetrics` | `GraphAugmentedRetrieveCandidates` | `rank` | `retrieval.arousal_affect` | `safe` | `§7 Consolidation`, `§8 Interrupt Gate` | `src/operations/graph_retrieval.cpp` |
| `arousal` | `ComputeMetrics` | `ComputeMniGateDecision` | `gate` | `interrupt.arousal_affect` | `safe` | `§8 Interrupt Gate` | `src/operations/interrupt_gate.cpp` |
| `arousal` | `ComputeMetrics` | `UpdateNeuromodulators` | `feedback` | `neuromod.arousal` | `safe` | `§2 Neuromodulators` | `src/operations/neuromodulators.cpp` |
| `arousal` | context output | `ApplySynapticTagging` | `gate` | `tagging.arousal` | `safe` | `§6 Emotional Consolidation` | `src/operations/synaptic_tagging.cpp` |
| `arousal` | `ComputeMetrics` | `PersistSignalMetrics` | `persist` | `persist.arousal` | `safe` | `§10` | `src/operations/signal_metrics_persistence.cpp` |
| `focus_spread` | `ComputeFocusSpread` | `ComputeEffectiveFocus` | `gate` | `effective_focus.focus_spread` | `safe` | `§3 Structural Metrics` | `src/operations/focus_spread.cpp`, `src/operations/effective_focus.cpp` |
| `focus_spread` | `ComputeFocusSpread` | `UpdateUncertainty` | `feedback` | `uncertainty.focus_spread` | `safe` | `§1 Math Foundations` | `src/operations/uncertainty.cpp` |
| `focus_spread` | `ComputeFocusSpread` | `PersistSignalMetrics` | `persist` | `persist.focus_spread` | `safe` | `Appendix field map` | `src/operations/signal_metrics_persistence.cpp` |
| `drift_mag` | `ComputeCoherence` | `ComputeMetrics` | `compute precursor` | `metrics.drift_mag` | `paired` | `§3 Structural Metrics`, `§4 Dynamic Thresholding` | `src/operations/coherence.cpp`, `src/operations/metrics.cpp` |
| `drift_mag` | `ComputeCoherence` | accumulator / boundary logic | `gate` | `boundary.drift_mag` | `safe` | `§4 Dynamic Thresholding` | `src/operations/coherence.cpp`, `src/operations/boundary.cpp`, `src/operations/accumulator.cpp` |
| `embedding_surprisal` | `UpdateEmbeddingPredictionError` | `ComputeMetrics` | `compute precursor` | `metrics.embedding_surprisal` | `paired` | `§3 Structural Metrics` | `src/operations/embedding_prediction_error.cpp`, `src/operations/metrics.cpp` |
| `embedding_surprisal` | `UpdateEmbeddingPredictionError` | `UpdateUncertainty` | `feedback` | `uncertainty.embedding_surprisal` | `safe` | `§1 Math Foundations` | `src/operations/uncertainty.cpp` |
| `embedding_surprisal` | `UpdateEmbeddingPredictionError` | `DetectBoundary` | `gate` | `boundary.embedding_surprisal` | `safe` | `§4 Dynamic Thresholding` | `src/operations/boundary.cpp` |
| `embedding_surprisal` | `UpdateEmbeddingPredictionError` | `UpdateNeuromodulators` | `feedback` | `neuromod.embedding_surprisal` | `safe` | `§2 Neuromodulators` | `src/operations/neuromodulators.cpp` |
| `embedding_surprisal` | `UpdateEmbeddingPredictionError` | `ComputeMniGateDecision` | `gate` | `interrupt.embedding_surprisal` | `safe` | `§8 Interrupt Gate` | `src/operations/interrupt_gate.cpp` |
| `embedding_surprisal` | `UpdateEmbeddingPredictionError` | `ApplySynapticTagging` | `gate` | `tagging.embedding_surprisal` | `safe` | `§6 Emotional Consolidation` | `src/operations/synaptic_tagging.cpp` |

## Decision-Surface Matrix

These are not `Metric` enum entries, but they are the first things you will actually score in most ablation benches.

| surface | produced by | consumed by | ablation dimension | paper anchor | code anchor |
|---|---|---|---|---|---|
| `composite_score` | `ComputeCompositeScore` | `CheckSpikeBypass`, `ComputeWriteGate`, chat / API output | `surface.composite_score` | `§3`, `§4` | `src/operations/blend.cpp`, `src/operations/spike_bypass.cpp`, `src/operations/write_gate.cpp` |
| `threshold_t` | `UpdateThreshold` | `ComputeWriteGate`, chat / API output, signal persistence | `surface.threshold_t` | `§4 Dynamic Thresholding` | `src/operations/threshold.cpp`, `src/operations/write_gate.cpp`, `src/operations/signal_metrics_persistence.cpp` |
| `boundary_score` | `DetectBoundary` | `CheckSpikeBypass`, final boundary / flush decisions, chat diagnostics | `surface.boundary_score` | `§4 Dynamic Thresholding` | `src/operations/boundary.cpp`, `src/operations/spike_bypass.cpp` |
| `write_decision` | `ComputeWriteGate` | `MemoryStorage`, `PersistSignalMetrics`, downstream memory formation | `surface.write_decision` | `§4 Dynamic Thresholding` | `src/operations/write_gate.cpp`, `src/operations/memory_storage.cpp`, `src/operations/signal_metrics_persistence.cpp` |
| `interrupt_gate_affect_drive` | `ComputeMniGateDecision` | interrupt decision, chat diagnostics | `surface.interrupt_affect_drive` | `§8 Interrupt Gate` | `src/operations/interrupt_gate.cpp` |

## Factorial Planning Guidance

| dimension family | recommended benchmark family | reason |
|---|---|---|
| `blend.*` | deterministic signal-level composite-score bench | isolates whether the metric matters before storage / retrieval complexity |
| `persist.*` | storage / hydration audit | proves whether the metric is needed for later inspection or only live decisions |
| `boundary.*`, `write_gate.*`, `surface.threshold_t` | boundary / write deterministic bench | isolates memory-formation effects |
| `interrupt.*`, `retrieval.*`, `surface.interrupt_affect_drive` | interrupt + retrieval bench on short chat and synthetic probes | isolates gating vs ranking |
| `serial.*`, `predictive.*`, `tagging.*`, `neuromod.*` | dedicated deterministic activation bench first, then live-model rerun | these are multi-stage pathways that are easy to misread in noisy corpus runs |
| `uncertainty.*`, `effective_focus.*` | uncertainty / focus deterministic bench | these are upstream modulators and should be proved causally before corpus claims |

## Removal Heuristic

If you are trying to decide what can be removed, use this order:

| removal test | interpretation |
|---|---|
| metric has no direct consumer beyond persistence / display | candidate for deletion or demotion to telemetry |
| metric matters in `blend.*` but nowhere else | keep only if it improves write / retrieval quality enough to justify complexity |
| metric matters only through one downstream operation | prefer a targeted toggle and dedicated bench before removing |
| metric is only visible on untrustworthy hydration surfaces | fix the surface first; do not conclude the metric is useless |
| metric effect collapses in deterministic bench and live rerun | strongest candidate for removal |

## Current High-Value Ablation Dimensions

If the goal is to cut the search space first, start here:

| priority | dimensions |
|---|---|
| `P1` | `blend.relevance`, `blend.surprise`, `blend.utility`, `blend.salience`, `blend.coverage` |
| `P1` | `interrupt.salience_affect`, `interrupt.arousal_affect`, `retrieval.salience_affect`, `retrieval.arousal_affect` |
| `P1` | `boundary.embedding_surprisal`, `write_gate.coverage`, `surface.threshold_t` |
| `P2` | `serial.rarity`, `predictive.surprise`, `neuromod.rarity`, `neuromod.arousal`, `tagging.embedding_surprisal`, `tagging.arousal` |
| `P2` | `effective_focus.focus_spread`, `uncertainty.focus_spread`, `uncertainty.embedding_surprisal` |
| `P3` | all `persist.*` dimensions, which are mostly storage / observability questions rather than behavior questions |

This ordering biases toward metrics that have both:

1. a clear downstream consumer, and
2. a realistic chance of being removable if the effect is weak.
