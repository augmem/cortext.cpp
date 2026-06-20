#pragma once

/// Contract tags: each type names a per-signal value on the
/// OperationContext, used in Operation<Requires<...>, Satisfies<...>>
/// declarations and validated at compile time by OperationSet's chain
/// aggregation (see operation_set.hpp).
///
/// Scope: tags cover the per-signal dataflow through OperationContext.
/// Ambient inputs (Config, Signal, ProcessorContext, Store, transactions) are
/// constructor- or call-injected, not dataflow, and are deliberately untagged.
/// Long-lived ProcessorContext state is likewise unmodeled here; contracts
/// prove per-signal ordering, not cross-signal state evolution.
///
/// Values that are produced together by one operation and consumed as a
/// family share one tag (e.g. BoundaryDecision). A value read before any
/// producer runs (a default read, e.g. StoredEmbeddingId during probe-mode
/// retrieval) is intentionally absent from that consumer's Requires.

namespace cortext::tags
{

// ComputeCoherence
struct AccumulatorWindowState;  ///< accumulator coherence / drift step / eta
struct Coherence;
struct StructuralCoherence;

// Multi-producer metric store (SetMetric/GetMetric/GetAllMetrics)
struct MetricValues;
struct Violation;

// UpdateAccumulator
struct WriteExclusionTs;
struct InterruptAborted;

// UpdateDriftAccumulation
struct DriftAccumSnapshot;

// UpdateSensitivity
struct Arousal;
struct Valence;
struct EmotionIntensity;
struct EmotionProbabilities;
struct SensitivityThresholdDeltas;  ///< emotion / mood / sensitivity deltas

// UpdatePrecisionDelta
struct DeltaThresholdPrecision;

// ComputeEffectiveFocus
struct EffectiveFocus;

// ComputeCompositeScore
struct CompositeScore;

// UpdateThreshold
struct ThresholdState;  ///< T_dynamic + hysteresis

// DetectBoundary (at-boundary, centroid, diagnostics, score, type)
struct BoundaryDecision;
struct FlushRequired;

// CheckSpikeBypass
struct SpikeBypass;

// ComputeWriteGate
struct AccumulatorWriteDecision;
struct WriteDecision;
struct RepresentativeEmbedding;
struct WindowScore;

// MemoryStorage
struct StoredEmbeddingId;
struct StoredMemoryId;
struct StoredSignalId;

// CheckStreamingPacing
struct ShouldCheckRetrieval;

// GraphAugmentedRetrieveCandidates
struct RetrievedMemoryEmbeddings;

// ComputeMniGateDecision
struct InterruptAllowed;
struct SelectedCandidateId;
struct MniGateDiagnostics;  ///< mni_* and interrupt_gate_* telemetry values

// DetectMemoryUsage
struct MemoryUsageEvents;

// ApplySerialPositionEffects -> ApplySerialPositionMultiplier
struct SerialPositionPolicy;  ///< primacy/recency windows, suppression, ...
struct SerialPositionMultiplier;

// ApplyStabilityFeedback
struct DeltaHalfLifeAdjustment;

// Consolidation chain
struct ConsolidationShouldStart;
struct ConsolidationCandidates;
struct ConsolidationClusters;

} // namespace cortext::tags
