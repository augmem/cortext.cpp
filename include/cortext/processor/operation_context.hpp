#pragma once

#include "cortext/operations/extraction.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/processor.hpp" // For SignalProcessor::Config
#include "cortext/processor/processor_context.hpp"
#include "cortext/signal.hpp"
#include <Eigen/Dense>
#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

namespace cortext
{

class Store;

/// @brief Information about a cluster of memories from consolidation.
///
/// Created by ConsolidationCluster and consumed by ConsolidationSummarize.
struct ClusterInfo
{
  int cluster_id;
  std::vector<long long> embedding_ids;
  std::vector<float> centroid;
  double avg_score;
};

/// @brief Represents a candidate memory for consolidation.
struct ConsolidationCandidate
{
  long long embedding_id;
  double score;
  Eigen::VectorXf embedding;
};



/// @brief Contains all the state for a single signal processing run.
///
/// This object is created by the SignalProcessor for each signal and is passed
/// through the entire operation set.
class OperationContext
{
public:
  /// @brief Usage event for a memory row in the feedback table.
  struct MemoryUsageEvent
  {
    long long embedding_id;
    bool used;
    std::optional<double>
        contextual_gain; // Cosine similarity to input; std::nullopt if unknown
  };

  /// @brief Constructs an OperationContext.
  /// @param signal The input signal being processed.
  /// @param context The long-lived context of the SignalProcessor.
  /// @param config The processor's configuration knobs.
  OperationContext (const Signal &signal, ProcessorContext &context,
                    const SignalProcessor::Config &config);

  /// @brief Constructs an OperationContext with an attached Store.
  ///
  /// The store pointer is non-owning and may be null. Operations that need
  /// read access to persisted state (e.g. retrieval) can use it when present.
  OperationContext (const Signal &signal, ProcessorContext &context,
                    const SignalProcessor::Config &config, Store *store);

  // --- Accessors ---

  /// @brief Gets the input signal.
  const Signal &
  GetSignal () const
  {
    return signal_;
  }

  /// @brief Gets the processor's long-lived context.
  ProcessorContext &
  GetProcessorContext ()
  {
    return context_;
  }

  /// @brief Gets the processor's long-lived context (const version).
  const ProcessorContext &
  GetProcessorContext () const
  {
    return context_;
  }

  /// @brief Gets the processor's configuration knobs.
  const SignalProcessor::Config &
  GetConfig () const
  {
    return config_;
  }

  /// @brief Gets the backing store (may be null).
  Store *
  GetStore () const
  {
    return store_;
  }


  // ======================================================================
  // Memory Usage Events API (Algorithms 14, 18 inputs)
  // ======================================================================
  void SetMemoryUsageEvents (std::vector<MemoryUsageEvent> events);
  const std::vector<MemoryUsageEvent> &GetMemoryUsageEvents () const;

  // Provide retrieved memory embeddings for Algorithm 19.
  void
  SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf> embeddings)
  {
    retrieved_memory_embeddings_ = std::move (embeddings);
  }
  const std::unordered_map<long long, Eigen::VectorXf> &
  GetRetrievedMemoryEmbeddings () const
  {
    return retrieved_memory_embeddings_;
  }

  // Track the candidate selected by the interrupt gate (if any).
  void
  SetSelectedCandidateId (std::optional<long long> id)
  {
    selected_candidate_id_ = id;
  }
  std::optional<long long>
  GetSelectedCandidateId () const
  {
    return selected_candidate_id_;
  }

  // ======================================================================
  // Threshold Modulation API (Algorithm 8 inputs/outputs)
  // ======================================================================
  
  void
  SetCompositeScore (std::optional<double> v)
  {
    composite_score_ = v;
  }
  std::optional<double>
  GetCompositeScore () const
  {
    return composite_score_;
  }

  void
  SetDeltaThresholdSensitivity (std::optional<double> v)
  {
    delta_threshold_sensitivity_ = v;
  }
  std::optional<double>
  GetDeltaThresholdSensitivity () const
  {
    return delta_threshold_sensitivity_;
  }

  void
  SetDeltaThresholdPrecision (std::optional<double> v)
  {
    delta_threshold_precision_ = v;
  }
  std::optional<double>
  GetDeltaThresholdPrecision () const
  {
    return delta_threshold_precision_;
  }

  void
  SetDeltaThresholdEmotion (std::optional<double> v)
  {
    delta_threshold_emotion_ = v;
  }
  std::optional<double>
  GetDeltaThresholdEmotion () const
  {
    return delta_threshold_emotion_;
  }

  void
  SetDeltaThresholdMood (std::optional<double> v)
  {
    delta_threshold_mood_ = v;
  }
  std::optional<double>
  GetDeltaThresholdMood () const
  {
    return delta_threshold_mood_;
  }

  // Outputs exposed by operations for downstream consumers/telemetry
  void
  SetThresholdTDynamic (double v)
  {
    threshold_T_dynamic_ = v;
  }
  double
  GetThresholdTDynamic () const
  {
    return threshold_T_dynamic_;
  }

  void
  SetThresholdHysteresis (double v)
  {
    threshold_hysteresis_ = v;
  }
  double
  GetThresholdHysteresis () const
  {
    return threshold_hysteresis_;
  }

  // ======================================================================
  // Boundary & Interrupt Gate API (Algorithm 27)
  // ======================================================================
  
  void
  SetAtBoundary (bool v)
  {
    at_boundary_ = v;
  }
  bool
  GetAtBoundary () const
  {
    return at_boundary_;
  }
  void
  SetInterruptAllowed (bool v)
  {
    interrupt_allowed_ = v;
  }
  bool
  GetInterruptAllowed () const
  {
    return interrupt_allowed_;
  }

  void
  SetInterruptGateHasCandidates (bool v)
  {
    interrupt_gate_has_candidates_ = v;
  }
  bool
  GetInterruptGateHasCandidates () const
  {
    return interrupt_gate_has_candidates_;
  }
  void
  SetInterruptGateBlockedNoStore (bool v)
  {
    interrupt_gate_blocked_no_store_ = v;
  }
  bool
  GetInterruptGateBlockedNoStore () const
  {
    return interrupt_gate_blocked_no_store_;
  }
  void
  SetInterruptGateRelPass (bool v)
  {
    interrupt_gate_rel_pass_ = v;
  }
  bool
  GetInterruptGateRelPass () const
  {
    return interrupt_gate_rel_pass_;
  }
  void
  SetInterruptGateNoveltyPass (bool v)
  {
    interrupt_gate_novelty_pass_ = v;
  }
  bool
  GetInterruptGateNoveltyPass () const
  {
    return interrupt_gate_novelty_pass_;
  }
  void
  SetInterruptGateMuPass (bool v)
  {
    interrupt_gate_mu_pass_ = v;
  }
  bool
  GetInterruptGateMuPass () const
  {
    return interrupt_gate_mu_pass_;
  }
  void
  SetInterruptGateNoveltyMuPass (bool v)
  {
    interrupt_gate_novelty_mu_pass_ = v;
  }
  bool
  GetInterruptGateNoveltyMuPass () const
  {
    return interrupt_gate_novelty_mu_pass_;
  }
  void
  SetInterruptGateDupPass (bool v)
  {
    interrupt_gate_dup_pass_ = v;
  }
  bool
  GetInterruptGateDupPass () const
  {
    return interrupt_gate_dup_pass_;
  }
  void
  SetInterruptGateBoundaryMuPass (bool v)
  {
    interrupt_gate_boundary_mu_pass_ = v;
  }
  bool
  GetInterruptGateBoundaryMuPass () const
  {
    return interrupt_gate_boundary_mu_pass_;
  }
  void
  SetInterruptGateRelStar (double v)
  {
    interrupt_gate_rel_star_ = v;
  }
  double
  GetInterruptGateRelStar () const
  {
    return interrupt_gate_rel_star_;
  }
  void
  SetInterruptGateRetrievalThresh (double v)
  {
    interrupt_gate_retrieval_thresh_ = v;
  }
  double
  GetInterruptGateRetrievalThresh () const
  {
    return interrupt_gate_retrieval_thresh_;
  }
  void
  SetInterruptGateBoundaryMultEff (double v)
  {
    interrupt_gate_boundary_mult_eff_ = v;
  }
  double
  GetInterruptGateBoundaryMultEff () const
  {
    return interrupt_gate_boundary_mult_eff_;
  }
  void
  SetInterruptGateAffectDrive (double v)
  {
    interrupt_gate_affect_drive_ = v;
  }
  double
  GetInterruptGateAffectDrive () const
  {
    return interrupt_gate_affect_drive_;
  }
  // ======================================================================
  // Consolidation Triggers API (Algorithms 28, 28b)
  // ======================================================================
  
  void
  SetTokensInFlight (int v)
  {
    tokens_in_flight_ = v;
  }
  int
  GetTokensInFlight () const
  {
    return tokens_in_flight_;
  }
  void
  SetRetrievalQueueDepth (int v)
  {
    retrieval_queue_depth_ = v;
  }
  int
  GetRetrievalQueueDepth () const
  {
    return retrieval_queue_depth_;
  }
  void
  SetConsolidationShouldStart (bool v)
  {
    consolidation_should_start_ = v;
  }
  bool
  GetConsolidationShouldStart () const
  {
    return consolidation_should_start_;
  }
  // Diagnostics for Algorithm 27 (for testing/telemetry)
  void
  SetMniJaccard (double v)
  {
    mni_jaccard_ = v;
  }
  double
  GetMniJaccard () const
  {
    return mni_jaccard_;
  }
  void
  SetMniBestMu (double v)
  {
    mni_best_mu_ = v;
  }
  double
  GetMniBestMu () const
  {
    return mni_best_mu_;
  }
  void
  SetMniDupThresh (double v)
  {
    mni_dup_thresh_ = v;
  }
  double
  GetMniDupThresh () const
  {
    return mni_dup_thresh_;
  }
  void
  SetMniOverlapStar (double v)
  {
    mni_overlap_star_ = v;
  }
  double
  GetMniOverlapStar () const
  {
    return mni_overlap_star_;
  }
  void
  SetMniTauJaccardEff (double v)
  {
    mni_tau_j_eff_ = v;
  }
  double
  GetMniTauJaccardEff () const
  {
    return mni_tau_j_eff_;
  }
  void
  SetMniTauMuEff (double v)
  {
    mni_tau_m_eff_ = v;
  }
  double
  GetMniTauMuEff () const
  {
    return mni_tau_m_eff_;
  }

  void
  SetCoherence (double v)
  {
    coherence_ = v;
  }
  double
  GetCoherence () const
  {
    return coherence_;
  }

  void
  SetAccumulatorEtaPrev (double v)
  {
    accumulator_eta_prev_ = v;
  }
  std::optional<double>
  GetAccumulatorEtaPrev () const
  {
    return accumulator_eta_prev_;
  }

  // ======================================================================
  // Emotion API (Algorithm 4 outputs for downstream/telemetry)
  // ======================================================================
  
  void
  SetEmotionIntensity (double v)
  {
    emotion_intensity_ = v;
  }
  double
  GetEmotionIntensity () const
  {
    return emotion_intensity_;
  }

  void
  SetValence (double v)
  {
    valence_ = v;
  }
  double
  GetValence () const
  {
    return valence_;
  }

  void
  SetArousal (double v)
  {
    arousal_ = v;
  }
  double
  GetArousal () const
  {
    return arousal_;
  }

  // Emotion probability distribution (Algorithm 4 → Algorithm 4b bridge)
  // Order: [anger, fear, joy, love, sadness, surprise]
  void
  SetEmotionProbabilities (const std::array<double, 6> &probs)
  {
    emotion_probabilities_ = probs;
  }
  const std::array<double, 6> &
  GetEmotionProbabilities () const
  {
    return emotion_probabilities_;
  }

  void
  SetViolation (std::optional<double> v)
  {
    violation_ = v;
  }
  std::optional<double>
  GetViolation () const
  {
    return violation_;
  }

  // ======================================================================
  // Stability Feedback API (Algorithm 17 → Algorithm 6 bridge)
  // ======================================================================
  
  void
  SetDeltaHalfLifeAdjustment (std::optional<double> v)
  {
    delta_half_life_adj_ = v;
  }
  std::optional<double>
  GetDeltaHalfLifeAdjustment () const
  {
    return delta_half_life_adj_;
  }

  // Observed retention (seconds) for Alg 6 input
  void
  SetObservedRetentionSeconds (std::optional<double> v)
  {
    observed_retention_sec_ = v;
  }
  std::optional<double>
  GetObservedRetentionSeconds () const
  {
    return observed_retention_sec_;
  }

  // ======================================================================
  // Metacognitive Monitoring API (Algorithm 25)
  // ======================================================================
  
  void
  SetFeelingOfKnowing (std::optional<double> v)
  {
    feeling_of_knowing_ = v;
  }
  std::optional<double>
  GetFeelingOfKnowing () const
  {
    return feeling_of_knowing_;
  }
  void
  SetMetacogFOKThreshold (double v)
  {
    metacog_fok_threshold_ = v;
  }
  double
  GetMetacogFOKThreshold () const
  {
    return metacog_fok_threshold_;
  }
  void
  SetMetacogTOTDetected (bool v)
  {
    metacog_tot_detected_ = v;
  }
  bool
  GetMetacogTOTDetected () const
  {
    return metacog_tot_detected_;
  }
  void
  SetMetacogUnknownDetected (bool v)
  {
    metacog_unknown_detected_ = v;
  }
  bool
  GetMetacogUnknownDetected () const
  {
    return metacog_unknown_detected_;
  }
  void
  SetMetacogConfidenceDecayRate (double v)
  {
    metacog_confidence_decay_rate_ = v;
  }
  double
  GetMetacogConfidenceDecayRate () const
  {
    return metacog_confidence_decay_rate_;
  }
  void
  SetMetacogStrategySwitchLatencyMs (int v)
  {
    metacog_strategy_switch_latency_ms_ = v;
  }
  int
  GetMetacogStrategySwitchLatencyMs () const
  {
    return metacog_strategy_switch_latency_ms_;
  }
  void
  SetMetacogCertaintyRequirement (double v)
  {
    metacog_certainty_requirement_ = v;
  }
  double
  GetMetacogCertaintyRequirement () const
  {
    return metacog_certainty_requirement_;
  }
  void
  SetMetacogSensitivity (double v)
  {
    metacog_sensitivity_ = v;
  }
  double
  GetMetacogSensitivity () const
  {
    return metacog_sensitivity_;
  }

  // ======================================================================
  // Serial Position Effects API (Algorithm 26)
  // ======================================================================
  
  void
  SetSerialPrimacyWindow (int v)
  {
    serial_primacy_window_ = v;
  }
  int
  GetSerialPrimacyWindow () const
  {
    return serial_primacy_window_;
  }
  void
  SetSerialRecencyWindow (int v)
  {
    serial_recency_window_ = v;
  }
  int
  GetSerialRecencyWindow () const
  {
    return serial_recency_window_;
  }
  void
  SetSerialPrimacyBonus (double v)
  {
    serial_primacy_bonus_ = v;
  }
  double
  GetSerialPrimacyBonus () const
  {
    return serial_primacy_bonus_;
  }
  void
  SetSerialRehearsalCurveDepth (double v)
  {
    serial_rehearsal_curve_depth_ = v;
  }
  double
  GetSerialRehearsalCurveDepth () const
  {
    return serial_rehearsal_curve_depth_;
  }
  void
  SetSerialDistinctivenessThreshold (double v)
  {
    serial_distinctiveness_threshold_ = v;
  }
  double
  GetSerialDistinctivenessThreshold () const
  {
    return serial_distinctiveness_threshold_;
  }
  void
  SetSerialVonRestorffMultiplier (double v)
  {
    serial_von_restorff_multiplier_ = v;
  }
  double
  GetSerialVonRestorffMultiplier () const
  {
    return serial_von_restorff_multiplier_;
  }
  void
  SetSerialMiddleSuppression (double v)
  {
    serial_middle_suppression_ = v;
  }
  double
  GetSerialMiddleSuppression () const
  {
    return serial_middle_suppression_;
  }

  // ======================================================================
  // Effective Focus API (Algorithm 10 stabilizer)
  // ======================================================================
  
  void
  SetEffectiveFocus (double v)
  {
    f_eff_ = v;
  }
  double
  GetEffectiveFocus () const
  {
    return f_eff_;
  }

  // ======================================================================
  // Metrics API (Algorithm 7 and composite score computation)
  // ======================================================================
  // Metrics are stored as normalized values in either [-1,1] or [0,1],
  // depending on the metric definition (see algorithms.md). Consumers that
  // require [0,1] should map negatives via (v+1)/2 when needed.
  
  void
  SetMetric (operations::Metric name, double value_0_to_100)
  {
    // Store as given but clamp to [-1,1] since all metrics are normalized.
    double v = value_0_to_100;
    if (std::isnan (v) || std::isinf (v))
      {
        v = 0.0;
      }
    if (v < -1.0)
      v = -1.0;
    if (v > 1.0)
      v = 1.0;
    metrics_[name] = v;
  }
  std::optional<double>
  GetMetric (operations::Metric name) const
  {
    auto it = metrics_.find (name);
    if (it == metrics_.end ())
      {
        return std::nullopt;
      }
    return it->second;
  }
  const std::unordered_map<operations::Metric, double> &
  GetAllMetrics () const
  {
    return metrics_;
  }

  // Serial position multiplier (Algorithm 26 application)
  void
  SetSerialPositionMultiplier (std::optional<double> v)
  {
    serial_position_multiplier_ = v;
  }
  std::optional<double>
  GetSerialPositionMultiplier () const
  {
    return serial_position_multiplier_;
  }

  // Diagnostics for Algorithm 7 weight handling
  void
  SetLastWeightSum (double v)
  {
    last_weight_sum_ = v;
  }
  double
  GetLastWeightSum () const
  {
    return last_weight_sum_;
  }
  void
  SetLastEffectiveMetricCount (int n)
  {
    last_effective_metric_count_ = n;
  }
  int
  GetLastEffectiveMetricCount () const
  {
    return last_effective_metric_count_;
  }

  // ======================================================================
  // Episode Boundary API (Algorithm 12)
  // ======================================================================

  void
  RequestFinalizeEpisode ()
  {
    should_finalize_episode_ = true;
  }
  bool
  ShouldFinalizeEpisode () const
  {
    return should_finalize_episode_;
  }
  void
  SetBoundaryType (std::optional<std::string> v)
  {
    boundary_type_ = std::move (v);
  }
  std::optional<std::string>
  GetBoundaryType () const
  {
    return boundary_type_;
  }
  void
  SetBoundaryCentroid (std::optional<Eigen::VectorXf> v)
  {
    boundary_centroid_ = std::move (v);
  }
  std::optional<Eigen::VectorXf>
  GetBoundaryCentroid () const
  {
    return boundary_centroid_;
  }

  // ======================================================================
  // Write Gate API (Algorithm 7+8: composite_score > T_dynamic - hysteresis)
  // ======================================================================

  void
  SetWriteDecision (bool v)
  {
    write_decision_ = v;
  }
  bool
  GetWriteDecision () const
  {
    return write_decision_;
  }

  // ======================================================================
  // Memory Storage API (MemoryStorage operation output)
  // ======================================================================

  void
  SetStoredEmbeddingId (std::optional<long long> id)
  {
    stored_embedding_id_ = id;
  }
  std::optional<long long>
  GetStoredEmbeddingId () const
  {
    return stored_embedding_id_;
  }

  // ======================================================================
  // Memory Accumulation API (Section 4.4)
  // ======================================================================

  void
  SetBoundaryScore (double v)
  {
    boundary_score_ = v;
  }
  std::optional<double>
  GetBoundaryScore () const
  {
    return boundary_score_;
  }

  void
  SetFlushRequired (bool v)
  {
    flush_required_ = v;
  }
  bool
  GetFlushRequired () const
  {
    return flush_required_;
  }

  void
  SetSpikeBypass (bool v)
  {
    spike_bypass_ = v;
  }
  bool
  GetSpikeBypass () const
  {
    return spike_bypass_;
  }

  void
  SetWindowScore (double v)
  {
    window_score_ = v;
  }
  std::optional<double>
  GetWindowScore () const
  {
    return window_score_;
  }

  void
  SetAccumulatorWriteDecision (bool v)
  {
    accumulator_write_decision_ = v;
  }
  bool
  GetAccumulatorWriteDecision () const
  {
    return accumulator_write_decision_;
  }

  void
  SetWriteExclusionTs (std::uint64_t v)
  {
    write_exclusion_ts_ = v;
  }
  std::optional<std::uint64_t>
  GetWriteExclusionTs () const
  {
    return write_exclusion_ts_;
  }

  void
  SetRepresentativeEmbedding (Eigen::VectorXf v)
  {
    representative_embedding_ = std::move (v);
  }
  const std::optional<Eigen::VectorXf> &
  GetRepresentativeEmbedding () const
  {
    return representative_embedding_;
  }

  void
  SetAccumulatorCoherence (double v)
  {
    accumulator_coherence_ = v;
  }
  double
  GetAccumulatorCoherence () const
  {
    return accumulator_coherence_;
  }

  void
  SetAccumulatorDriftStep (double v)
  {
    accumulator_drift_step_ = v;
  }
  double
  GetAccumulatorDriftStep () const
  {
    return accumulator_drift_step_;
  }

  // Structural coherence (Section 3.1.1): 1 - var(cos(x_t, context_window))
  void
  SetStructuralCoherence (double v)
  {
    structural_coherence_ = v;
  }
  double
  GetStructuralCoherence () const
  {
    return structural_coherence_;
  }

  // ======================================================================
  // Streaming Pacing API (Section 10.4)
  // ======================================================================

  void
  SetShouldCheckRetrieval (bool v)
  {
    should_check_retrieval_ = v;
  }
  bool
  GetShouldCheckRetrieval () const
  {
    return should_check_retrieval_;
  }

  void
  SetDriftAccumSnapshot (double v)
  {
    drift_accum_snapshot_ = v;
  }
  double
  GetDriftAccumSnapshot () const
  {
    return drift_accum_snapshot_;
  }


  // ======================================================================
  // Consolidation Cluster API (Cluster -> Summarize data passing)
  // ======================================================================

  void
  SetConsolidationClusters (std::vector<ClusterInfo> clusters)
  {
    consolidation_clusters_ = std::move (clusters);
  }
  const std::vector<ClusterInfo> &
  GetConsolidationClusters () const
  {
    return consolidation_clusters_;
  }

  // ======================================================================
  // In-Memory Consolidation Candidates (Phase 2 Refactor)
  // ======================================================================

  void
  SetConsolidationCandidates (std::vector<ConsolidationCandidate> candidates)
  {
    consolidation_candidates_ = std::move (candidates);
  }
  const std::vector<ConsolidationCandidate> &
  GetConsolidationCandidates () const
  {
    return consolidation_candidates_;
  }

  void
  SetExtractionRequests (std::vector<operations::ExtractionRequest> requests)
  {
    extraction_requests_ = std::move (requests);
  }
  const std::vector<operations::ExtractionRequest> &
  GetExtractionRequests () const
  {
    return extraction_requests_;
  }

  void
  SetExtractionCallback (operations::ExtractionCallback *cb)
  {
    extraction_callback_ = cb;
  }
  operations::ExtractionCallback *
  GetExtractionCallback () const
  {
    return extraction_callback_;
  }

  // ======================================================================
  // LLM Components API (OGA/Phi-4)
  // ======================================================================

  /// @brief Gets the extractor (may be null if OGA disabled).
  Extractor *
  GetExtractor () const
  {
    return context_.extractor;
  }

  /// @brief Gets the summarizer (may be null if OGA disabled).
  Summarizer *
  GetSummarizer () const
  {
    return context_.summarizer;
  }

private:
  const Signal &signal_;
  ProcessorContext &context_;
  const SignalProcessor::Config &config_;
  Store *store_ = nullptr;

  // Typed scratch fields
  std::optional<double> composite_score_;
  std::optional<double> delta_threshold_sensitivity_;
  std::optional<double> delta_threshold_precision_;
  std::optional<double> delta_threshold_emotion_;
  std::optional<double> delta_threshold_mood_;
  double threshold_T_dynamic_ = 0.0;
  double threshold_hysteresis_ = 0.0;
  double coherence_ = 0.0;
  std::optional<double> accumulator_eta_prev_;
  double emotion_intensity_ = 0.0;
  double valence_ = 0.5;
  double arousal_ = 0.0;
  std::array<double, 6> emotion_probabilities_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::optional<double> violation_;

  // Algorithm 7 metrics and diagnostics
  std::unordered_map<operations::Metric, double> metrics_;
  double last_weight_sum_ = 0.0;
  int last_effective_metric_count_ = 0;
  double f_eff_ = 0.0;
  bool should_finalize_episode_ = false;
  std::optional<std::string> boundary_type_;
  std::optional<Eigen::VectorXf> boundary_centroid_;

  // Stability feedback bridge and optional observed retention
  std::optional<double> delta_half_life_adj_;
  std::optional<double> observed_retention_sec_;

  // Memory usage events attached to this signal
  std::vector<MemoryUsageEvent> memory_usage_events_;
  // Retrieved memory embeddings attached to this signal
  std::unordered_map<long long, Eigen::VectorXf> retrieved_memory_embeddings_;
  // Selected candidate ID from interrupt gate (if any)
  std::optional<long long> selected_candidate_id_;

  // Metacognitive monitoring (Algorithm 25) fields
  std::optional<double> feeling_of_knowing_;
  double metacog_fok_threshold_ = 0.0;
  bool metacog_tot_detected_ = false;
  bool metacog_unknown_detected_ = false;
  double metacog_confidence_decay_rate_ = 0.0;
  int metacog_strategy_switch_latency_ms_ = 0;
  double metacog_certainty_requirement_ = 0.0;
  double metacog_sensitivity_ = 0.0;

  // Serial Position Effects (Algorithm 26) fields
  int serial_primacy_window_ = 0;
  int serial_recency_window_ = 0;
  double serial_primacy_bonus_ = 0.0;
  double serial_rehearsal_curve_depth_ = 0.0;
  double serial_distinctiveness_threshold_ = 0.0;
  double serial_von_restorff_multiplier_ = 0.0;
  double serial_middle_suppression_ = 0.0;
  std::optional<double> serial_position_multiplier_;

  // Write gate decision (Algorithm 7+8)
  bool write_decision_ = false;

  // Memory storage output (MemoryStorage operation)
  std::optional<long long> stored_embedding_id_;

  // Algorithm 27 fields
  bool at_boundary_ = false;
  bool interrupt_allowed_ = false;
  bool interrupt_gate_has_candidates_ = false;
  bool interrupt_gate_blocked_no_store_ = false;
  bool interrupt_gate_rel_pass_ = false;
  bool interrupt_gate_novelty_pass_ = false;
  bool interrupt_gate_mu_pass_ = false;
  bool interrupt_gate_novelty_mu_pass_ = false;
  bool interrupt_gate_dup_pass_ = false;
  bool interrupt_gate_boundary_mu_pass_ = false;
  double interrupt_gate_rel_star_ = 0.0;
  double interrupt_gate_retrieval_thresh_ = 0.0;
  double interrupt_gate_boundary_mult_eff_ = 0.0;
  double interrupt_gate_affect_drive_ = 0.0;
  double mni_jaccard_ = 0.0;
  double mni_best_mu_ = 0.0;
  double mni_dup_thresh_ = 0.0;
  double mni_overlap_star_ = -1.0;
  double mni_tau_j_eff_ = 0.0;
  double mni_tau_m_eff_ = 0.0;

  // Algorithm 28/28b fields
  int tokens_in_flight_ = 0;
  int retrieval_queue_depth_ = 0;
  bool consolidation_should_start_ = false;

  // Streaming Pacing fields (Section 10.4)
  bool should_check_retrieval_ = true;
  double drift_accum_snapshot_ = 0.0;

  // Consolidation Cluster fields (Section 7.3)
  std::vector<ClusterInfo> consolidation_clusters_;
  // In-Memory Consolidation Candidates
  std::vector<ConsolidationCandidate> consolidation_candidates_;

  std::vector<operations::ExtractionRequest> extraction_requests_;
  operations::ExtractionCallback *extraction_callback_ = nullptr;

  // Memory Accumulation fields (Section 4.4)
  std::optional<double> boundary_score_;
  bool flush_required_ = false;
  bool spike_bypass_ = false;
  std::optional<double> window_score_;
  bool accumulator_write_decision_ = false;
  std::optional<std::uint64_t> write_exclusion_ts_;
  std::optional<Eigen::VectorXf> representative_embedding_;
  double accumulator_coherence_ = 1.0;
  double accumulator_drift_step_ = 0.0;
  double structural_coherence_ = 1.0;  // Section 3.1.1: defaults to high coherence
};

} // namespace cortext
