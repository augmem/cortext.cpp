#include "cortext/cortext.hpp"
#include "cortext/encoder/imagebind.hpp"

#include "cortext/processor.hpp"
#include "cortext/processor/operation_set.hpp"
#include "cortext/signal.hpp"
#include "cortext/store/sqlite_store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include "cortext/operations/consolidation.hpp"
#include "cortext/operations/blend.hpp"
#include "cortext/operations/graph_build.hpp"
#include "cortext/operations/graph_retrieval.hpp"

#include "cortext/operations/embedding_prediction_error.hpp"
#include "cortext/operations/centroids.hpp"
#include "cortext/operations/precision.hpp"
#include "cortext/operations/sensitivity.hpp"
#include "cortext/operations/threshold.hpp"
#include "cortext/operations/uncertainty.hpp"
#include "cortext/processor/operation.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// ConsolidationGate operation declaration
#include "cortext/operations/boundary.hpp"
#include "cortext/operations/coherence.hpp"
#include "cortext/operations/competition.hpp"
#include "cortext/operations/effective_focus.hpp"
#include "cortext/operations/emotion.hpp"
#include "cortext/operations/focus.hpp"
#include "cortext/operations/focus_feedback.hpp"
#include "cortext/operations/focus_spread.hpp"
#include "cortext/operations/influence.hpp"
#include "cortext/operations/interrupt_gate.hpp"
#include "cortext/operations/memory_strength.hpp"
#include "cortext/operations/metacognitive.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/operations/neuromodulators.hpp"
#include "cortext/operations/predictive.hpp"
#include "cortext/operations/recent_context.hpp"
#include "cortext/operations/reconsolidation.hpp"
#include "cortext/operations/sensitivity_feedback.hpp"
#include "cortext/operations/serial_position.hpp"
#include "cortext/operations/serial_position_apply.hpp"
#include "cortext/operations/signal_metrics_persistence.hpp"
#include "cortext/operations/stability.hpp"
#include "cortext/operations/write_gate.hpp"
#include "cortext/operations/memory_storage.hpp"
#include "cortext/operations/stability_feedback.hpp"
#include "cortext/operations/working_memory.hpp"
#include "cortext/operations/detect_memory_usage.hpp"
#include "cortext/operations/drift_accumulation.hpp"
#include "cortext/operations/streaming_pacing.hpp"
#include "cortext/operations/synaptic_tagging.hpp"

// Section 4.4: Memory Accumulation
#include "cortext/operations/accumulator.hpp"
#include "cortext/operations/accumulator_scores.hpp"
#include "cortext/operations/accumulator_reset.hpp"
#include "cortext/operations/coherence.hpp"
#include "cortext/operations/boundary.hpp"
#include "cortext/operations/spike_bypass.hpp"
#include "cortext/operations/write_gate.hpp"

#include "cortext/operations/consolidation_cluster.hpp"
#include "cortext/operations/consolidation_gate.hpp"
#include "cortext/operations/consolidation_summarize.hpp"
#include "cortext/operations/process_extraction_results.hpp"
// Phase 4: Knowledge Graph Enhancement
#include "cortext/operations/emotion_cascade.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include "cortext/extractor/gemma_extractor.hpp"
#include "cortext/summarizer/gemma_summarizer.hpp"

namespace cortext
{

namespace
{

/// @brief Get current timestamp in milliseconds since Unix epoch.
inline std::uint64_t
NowMillis ()
{
  return static_cast<std::uint64_t> (
      std::chrono::duration_cast<std::chrono::milliseconds> (
          std::chrono::system_clock::now ().time_since_epoch ())
          .count ());
}

/// @brief Converts a std::vector<float> to an Eigen::VectorXf using Eigen::Map.
inline Eigen::VectorXf
ToEigen (const std::vector<float> &v)
{
  return Eigen::Map<const Eigen::VectorXf> (v.data (),
                                            static_cast<int> (v.size ()));
}


bool
LoadObjstorePayload (Store *store, const std::vector<unsigned char> &blob_id,
                     std::vector<unsigned char> &out)
{
  if (blob_id.empty () || !store)
    {
      return false;
    }
  try
    {
      auto rows
          = store->Execute ("SELECT objstore_get(?1) AS data", { blob_id });
      if (!rows.empty ())
        {
          const auto it = rows[0].find ("data");
          if (it != rows[0].end ())
            {
              out = store::BlobFromAny (it->second);
              return !out.empty ();
            }
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load objstore payload",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  catch (...)
    {
      telemetry::LogWarn (
          "Failed to load objstore payload (unknown error)",
          { telemetry::Attribute::String ("component", "cortext") });
    }
  return false;
}

/// @brief Load all signal blobs for a memory, ordered by serial_position.
/// @param store The store instance.
/// @param memory_id The memory_id to query signals for.
/// @param out Vector of blobs, one per signal.
/// @return true if any blobs were loaded.
bool
LoadSignalBlobs (Store *store, long long memory_id,
                 std::vector<std::vector<unsigned char>> &out)
{
  if (!store || memory_id <= 0)
    return false;

  try
    {
      auto rows = store->Execute (
          "SELECT blob_id FROM signals "
          "WHERE memory_id = ? AND blob_id IS NOT NULL "
          "ORDER BY serial_position ASC",
          { memory_id });

      for (const auto &row : rows)
        {
          auto it = row.find ("blob_id");
          if (it != row.end () && it->second.has_value ())
            {
              auto blob_id = store::BlobFromAny (it->second);
              if (!blob_id.empty ())
                {
                  std::vector<unsigned char> payload;
                  if (LoadObjstorePayload (store, blob_id, payload))
                    {
                      out.push_back (std::move (payload));
                    }
                }
            }
        }
      return !out.empty ();
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load signal blobs",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::Int64 ("memory_id", memory_id),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  catch (...)
    {
      telemetry::LogWarn (
          "Failed to load signal blobs (unknown error)",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::Int64 ("memory_id", memory_id) });
    }
  return false;
}

void
HydrateMemory (Store *store, long long id, Cortext::Context::Memory &m)
{
  if (!store)
    return;
  try
    {
      // v2: Query uses unified memories table
      // Mimetype is at the signal level - get from first signal
      auto rows = store->Execute (
          "SELECT "
          "  m.memory_id, m.modality, m.source_id, m.start_ts, m.end_ts, "
          "  m.n_signals, m.s_max, m.s_avg, "
          "  m.blob_id, "
          "  COALESCE(m.retrieved_count, 0) AS retrieved_count, "
          "  COALESCE(m.used_count, 0) AS used_count, "
          "  (SELECT s.mime FROM signals s WHERE s.memory_id = m.memory_id "
          "   ORDER BY s.serial_position LIMIT 1) AS signal_mime "
          "FROM memories m "
          "WHERE m.embedding_id = ?",
          { id });

      if (!rows.empty ())
        {
          const auto &row = rows[0];

          auto get_s = [&row] (const char *k) -> std::string {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return {};
            if (it->second.type () == typeid (std::string))
              return std::any_cast<std::string> (it->second);
            return {};
          };

          auto get_ll = [&row] (const char *k) -> long long {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return 0LL;
            if (it->second.type () == typeid (long long))
              return std::any_cast<long long> (it->second);
            if (it->second.type () == typeid (int))
              return static_cast<long long> (std::any_cast<int> (it->second));
            return 0LL;
          };

          [[maybe_unused]] auto get_dbl = [&row] (const char *k, double def = 0.0) -> double {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return def;
            if (it->second.type () == typeid (double))
              return std::any_cast<double> (it->second);
            if (it->second.type () == typeid (float))
              return static_cast<double> (std::any_cast<float> (it->second));
            if (it->second.type () == typeid (int))
              return static_cast<double> (std::any_cast<int> (it->second));
            if (it->second.type () == typeid (long long))
              return static_cast<double> (std::any_cast<long long> (it->second));
            return def;
          };

          [[maybe_unused]] auto get_blob = [&row] (const char *k) {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return std::vector<unsigned char> ();
            return store::BlobFromAny (it->second);
          };

          // v2: Populate from unified memories table
          const long long memory_id = get_ll ("memory_id");
          m.id = memory_id;
          m.modality = get_s ("modality");
          m.mimetype = get_s ("signal_mime");  // Get from first signal
          m.source_id = get_s ("source_id");
          m.timestamp = static_cast<std::uint64_t> (get_ll ("end_ts"));

          // v2: Load content blobs from signals table (ordered by serial_position)
          LoadSignalBlobs (store, memory_id, m.content);
          if (m.content.empty ())
            {
              const auto blob_id = get_blob ("blob_id");
              if (!blob_id.empty ())
                {
                  std::vector<unsigned char> payload;
                  if (LoadObjstorePayload (store, blob_id, payload))
                    {
                      m.content.push_back (std::move (payload));
                    }
                }
            }

          // v2: Counts are inline on memories table
          m.retrieved_count = get_ll ("retrieved_count");
          m.used_count = get_ll ("used_count");

          // v2: signal_metrics are stored on signals table, not readily available here
          // Set defaults for now - metrics come from signal processing context
          m.relevance = 0.0;
          m.mismatch = 0.0;
          m.surprise = 0.0;
          m.rarity = 0.0;
          m.drift = 0.0;
          m.contradiction = 0.0;
          m.utility = 0.0;
          m.periphery = 0.0;
          m.coverage = 0.0;
          m.salience = 0.0;
          m.valence = 0.5;
          m.arousal = 0.0;
          m.composite_score = 0.0;
          m.threshold_t = 0.0;
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to hydrate memory",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::Int64 ("embedding_id", id),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  catch (...)
    {
      telemetry::LogWarn (
          "Failed to hydrate memory (unknown error)",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::Int64 ("embedding_id", id) });
    }
}

void
HydrateWorkingMemoryFromDB (Store *store,
                            std::vector<Cortext::Context::Memory> &out)
{
  if (!store)
    return;

  try
    {
      // Query active WM slots from MEMORIES table with kind='WORKING'
      // Content blobs are loaded separately via LoadSignalBlobs
      auto rows = store->Execute (
          "SELECT m.memory_id, m.source_id, m.modality, m.start_ts, "
          "       m.strength, m.last_access, m.n_signals, "
          "       m.s_max, m.s_avg, "
          "       (SELECT s.mime FROM signals s WHERE s.memory_id = m.memory_id "
          "        ORDER BY s.serial_position LIMIT 1) AS signal_mime "
          "FROM memories m "
          "WHERE m.kind = 'WORKING' AND m.end_ts IS NULL "
          "ORDER BY m.start_ts ASC");

      for (const auto &row : rows)
        {
          Cortext::Context::Memory m;

          // Helper lambdas (same pattern as HydrateMemory)
          auto get_s = [&row] (const char *k) -> std::string {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return {};
            if (it->second.type () == typeid (std::string))
              return std::any_cast<std::string> (it->second);
            return {};
          };

          auto get_ll = [&row] (const char *k) -> long long {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return 0LL;
            if (it->second.type () == typeid (long long))
              return std::any_cast<long long> (it->second);
            if (it->second.type () == typeid (int))
              return static_cast<long long> (std::any_cast<int> (it->second));
            return 0LL;
          };

          auto get_dbl = [&row] (const char *k, double def = 0.0) -> double {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return def;
            if (it->second.type () == typeid (double))
              return std::any_cast<double> (it->second);
            if (it->second.type () == typeid (float))
              return static_cast<double> (std::any_cast<float> (it->second));
            if (it->second.type () == typeid (int))
              return static_cast<double> (std::any_cast<int> (it->second));
            if (it->second.type () == typeid (long long))
              return static_cast<double> (std::any_cast<long long> (it->second));
            return def;
          };

          // Populate Memory struct from WM slot row
          m.id = get_ll ("memory_id");
          m.source_id = get_s ("source_id");
          m.modality = get_s ("modality");
          m.mimetype = get_s ("signal_mime");
          m.timestamp = static_cast<std::uint64_t> (get_ll ("last_access"));

          // Map WM metrics to Memory struct
          m.composite_score = get_dbl ("s_avg", 0.0);
          m.salience = get_dbl ("s_max", 0.0);

          // v2: Load content blobs from signals table (ordered by serial_position)
          LoadSignalBlobs (store, m.id, m.content);

          out.push_back (std::move (m));
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to hydrate working memory",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  catch (...)
    {
      telemetry::LogWarn (
          "Failed to hydrate working memory (unknown error)",
          { telemetry::Attribute::String ("component", "cortext") });
    }
}

} // namespace

struct Cortext::Impl
{
  Config cfg;
  std::string db_path;
  std::string models_dir;

  std::unique_ptr<Encoder> encoder;
  std::shared_ptr<cortext::Store> store;
  std::unique_ptr<cortext::IOperation> pipeline_root;
  std::unique_ptr<cortext::SignalProcessor> processor;

#if !defined(CORTEXT_DISABLE_LITERT)
  // LiteRT-LM components (Gemma extractor and summarizer)
  std::unique_ptr<GemmaExtractor> extractor_instance;
  std::unique_ptr<GemmaSummarizer> summarizer_instance;
#endif

  Impl (const Config &c, std::string db, std::string models)
      : cfg (c), db_path (std::move (db)), models_dir (std::move (models))
  {
    // Store
    auto uniq = cortext::SQLiteStore::Create (db_path.c_str ());
    store = std::shared_ptr<cortext::Store> (std::move (uniq));

    // Encoder (ImageBind-oriented)
    std::filesystem::path imagebind_dir = models_dir;
    const auto has_imagebind = [] (const std::filesystem::path &dir) {
      return std::filesystem::exists (dir / "text_encoder.onnx")
             || std::filesystem::exists (dir / "text_encoder_int8.onnx");
    };
    if (!has_imagebind (imagebind_dir)
        && has_imagebind (imagebind_dir / "imagebind"))
      {
        imagebind_dir /= "imagebind";
      }
    encoder = std::make_unique<ImageBindEncoder> (imagebind_dir.string ());

#if !defined(CORTEXT_DISABLE_LITERT)
    // Create Gemma extractor and summarizer (LiteRT-LM)
    std::filesystem::path gemma_root = models_dir;
    if (!std::filesystem::exists (gemma_root / "gemma3n-e2b-litert")
        && gemma_root.filename () == "imagebind")
      {
        gemma_root = gemma_root.parent_path ();
      }
    gemma_root /= "gemma3n-e2b-litert";
    std::string gemma_path = gemma_root.string ();
    if (std::filesystem::is_directory (gemma_root))
      {
        const std::vector<std::string> candidates = {
          "gemma-3n-E2B-it-int4.litertlm",
          "gemma-3n-E2B-it-int4-Web.litertlm",
          "gemma-3n-E2B-it-int4.mediatek.mt6993.litertlm"
        };
        for (const auto &candidate : candidates)
          {
            const std::filesystem::path p = gemma_root / candidate;
            if (std::filesystem::exists (p))
              {
                gemma_path = p.string ();
                break;
              }
          }
      }
    extractor_instance = std::make_unique<GemmaExtractor> (gemma_path);
    summarizer_instance = std::make_unique<GemmaSummarizer> (gemma_path);
    if (!extractor_instance->IsAvailable () || !summarizer_instance->IsAvailable ())
      {
        throw std::runtime_error (
            "Gemma LiteRT-LM model not available at " + gemma_path);
      }
#endif

    // Default pipeline: full per-signal processing chain.
    using cortext::OperationSet;
    using cortext::operations::ConsolidationCluster;
    using cortext::operations::ConsolidationGate;
    using cortext::operations::ConsolidationSummarize;
    using cortext::operations::EnqueueExtractionJobs;
    using cortext::operations::EvaluateConsolidation;
    using cortext::operations::ProcessExtractionResults;
    using cortext::operations::FitMetricWeightsRLS;
    using cortext::operations::GraphAugmentedRetrieveCandidates;
    using cortext::operations::InitializeEmbeddedCentroids;

    using cortext::operations::ComputeCompositeScore;
    using cortext::operations::ComputeCoherence;
    using cortext::operations::ComputeEffectiveFocus;
    using cortext::operations::ComputeFocusSpread;
    using cortext::operations::ComputeMniGateDecision;
    using cortext::operations::InitializeFocusPriors;
    using cortext::operations::InitializeSensitivityPriors;
    using cortext::operations::InitializeStabilityPriors;
    using cortext::operations::ApplyEmotionalConsolidation;
    using cortext::operations::ApplyFocusFeedback;
    using cortext::operations::ApplyInfluenceFeedback;
    using cortext::operations::ApplyPredictivePreActivation;
    using cortext::operations::ApplyReconsolidation;
    using cortext::operations::ApplyRetrievalCompetition;
    using cortext::operations::ApplySensitivityFeedback;
    using cortext::operations::ApplySerialPositionEffects;
    using cortext::operations::ApplySerialPositionMultiplier;
    using cortext::operations::ApplyStabilityFeedback;
    using cortext::operations::BuildGraphFromConsolidation;
    using cortext::operations::ComputeMetrics;

    using cortext::operations::MetacognitiveMonitoring;
    using cortext::operations::UpdateFocus;
    using cortext::operations::UpdateEmbeddingPredictionError;
    using cortext::operations::UpdateMemoryStrength;
    using cortext::operations::UpdateMood;
    using cortext::operations::UpdateNeuromodulators;
    using cortext::operations::UpdatePrecisionDelta;
    using cortext::operations::UpdateRecentContext;
    using cortext::operations::UpdateRateState;
    using cortext::operations::UpdateSensitivity;
    using cortext::operations::UpdateStability;
    using cortext::operations::UpdateThreshold;
    using cortext::operations::UpdateUncertainty;
    using cortext::operations::WorkingMemory;
    using cortext::operations::PersistSignalMetrics;
    using cortext::operations::MemoryStorage;
    using cortext::operations::DetectMemoryUsage;
    using cortext::operations::UpdateDriftAccumulation;
    using cortext::operations::CheckStreamingPacing;
    // Section 4.4: Memory Accumulation
    using cortext::operations::UpdateAccumulator;
    using cortext::operations::UpdateAccumulatorScores;
    using cortext::operations::ComputeCoherence;
    using cortext::operations::DetectBoundary;
    using cortext::operations::CheckSpikeBypass;
    using cortext::operations::ComputeWriteGate;
    using cortext::operations::ResetAccumulatorAfterFlush;
    using cortext::operations::ApplySynapticTagging;
    // Phase 4: Knowledge Graph Enhancement
    using cortext::operations::PropagateEmotionalCascade;

    pipeline_root = std::make_unique<OperationSet> (


        std::make_unique<InitializeEmbeddedCentroids> (),

        std::make_unique<InitializeFocusPriors> (),
        std::make_unique<InitializeSensitivityPriors> (),
        std::make_unique<InitializeStabilityPriors> (),

        std::make_unique<ComputeCoherence> (),
        // Update accumulator embedding/state before metric computation.
        std::make_unique<UpdateAccumulator> (),
        std::make_unique<UpdateDriftAccumulation> (),
        std::make_unique<ComputeFocusSpread> (),
        std::make_unique<UpdateEmbeddingPredictionError> (),
        std::make_unique<UpdateUncertainty> (),

        std::make_unique<UpdateFocus> (),
        std::make_unique<UpdateSensitivity> (),
        std::make_unique<UpdateMood> (),

        std::make_unique<ComputeEffectiveFocus> (),
        std::make_unique<ComputeMetrics> (),
        std::make_unique<UpdateNeuromodulators> (),
        std::make_unique<FitMetricWeightsRLS> (),
        std::make_unique<ComputeCompositeScore> (),
        std::make_unique<UpdateAccumulatorScores> (),

        std::make_unique<UpdatePrecisionDelta> (),
        std::make_unique<UpdateThreshold> (),
        // Append accumulator centroid to recent context after scoring.
        std::make_unique<UpdateRecentContext> (),

        // Section 4.4: Memory Accumulation (before write gate)
        std::make_unique<DetectBoundary> (),
        std::make_unique<CheckSpikeBypass> (),
        std::make_unique<ComputeWriteGate> (),
        std::make_unique<MemoryStorage> (),
        std::make_unique<ApplySynapticTagging> (),
        std::make_unique<PersistSignalMetrics> (),
        std::make_unique<CheckStreamingPacing> (),
        std::make_unique<GraphAugmentedRetrieveCandidates> (),
        std::make_unique<UpdateRateState> (),
        std::make_unique<ResetAccumulatorAfterFlush> (),

        std::make_unique<ComputeMniGateDecision> (),

        // Derive retrieved/used events after the interrupt gate decision.
        std::make_unique<DetectMemoryUsage> (),

        std::make_unique<ApplyRetrievalCompetition> (),
        std::make_unique<ApplyPredictivePreActivation> (),
        std::make_unique<ApplyReconsolidation> (),

        std::make_unique<ApplyFocusFeedback> (),
        std::make_unique<ApplySensitivityFeedback> (),
        std::make_unique<ApplyStabilityFeedback> (),
        std::make_unique<UpdateStability> (),
        std::make_unique<ApplyInfluenceFeedback> (),

        std::make_unique<ApplySerialPositionEffects> (),
        std::make_unique<ApplySerialPositionMultiplier> (),
        std::make_unique<UpdateMemoryStrength> (),
        std::make_unique<ApplyEmotionalConsolidation> (),
        std::make_unique<WorkingMemory> (),
        std::make_unique<MetacognitiveMonitoring> (),

        std::make_unique<EvaluateConsolidation> (),
        std::make_unique<ConsolidationGate> (),
        // Phase 3: Consolidation pipeline
        std::make_unique<ConsolidationCluster> (),
        std::make_unique<ConsolidationSummarize> (),
        std::make_unique<EnqueueExtractionJobs> (),
        std::make_unique<ProcessExtractionResults> (),
        std::make_unique<BuildGraphFromConsolidation> (),
        std::make_unique<PropagateEmotionalCascade> ());

    cortext::SignalProcessor::Config pcfg;
    pcfg.focus = cfg.focus;
    pcfg.sensitivity = cfg.sensitivity;
    pcfg.stability = cfg.stability;
    pcfg.affect_interrupt = cfg.affect_interrupt;
    pcfg.affect_retrieval = cfg.affect_retrieval;
    pcfg.encoder = encoder.get ();

#if !defined(CORTEXT_DISABLE_LITERT)
    pcfg.extractor = extractor_instance.get ();
    pcfg.summarizer = summarizer_instance.get ();
#endif

    processor = std::make_unique<cortext::SignalProcessor> (
        pcfg, store, std::move (pipeline_root));
  }

  cortext::SignalProcessor::Output
  ProcessEmbedding (const Eigen::VectorXf &embedding, std::uint64_t timestamp,
                    const std::string &source_id)
  {
    cortext::Signal s;
    s.embedding = embedding;
    s.timestamp = timestamp;
    s.source_id = source_id;
    return processor->Process (s);
  }

  Cortext::Context
  HydrateContext (const cortext::SignalProcessor::Output &out)
  {
    Cortext::Context result;
    result.should_interrupt = out.interrupt_allowed;
    result.consolidation_recommended = out.consolidation_recommended;
    result.consolidation_required = out.consolidation_required;
    result.interrupt_gate_has_candidates = out.interrupt_gate_has_candidates;
    result.interrupt_gate_blocked_no_store = out.interrupt_gate_blocked_no_store;
    result.interrupt_gate_rel_pass = out.interrupt_gate_rel_pass;
    result.interrupt_gate_novelty_pass = out.interrupt_gate_novelty_pass;
    result.interrupt_gate_mu_pass = out.interrupt_gate_mu_pass;
    result.interrupt_gate_novelty_mu_pass = out.interrupt_gate_novelty_mu_pass;
    result.interrupt_gate_dup_pass = out.interrupt_gate_dup_pass;
    result.interrupt_gate_boundary_mu_pass = out.interrupt_gate_boundary_mu_pass;
    result.interrupt_gate_rel_star = out.interrupt_gate_rel_star;
    result.interrupt_gate_retrieval_thresh =
        out.interrupt_gate_retrieval_thresh;
    result.interrupt_gate_boundary_mult_eff =
        out.interrupt_gate_boundary_mult_eff;
    result.interrupt_gate_affect_drive = out.interrupt_gate_affect_drive;
    
    // Populate output metrics
    result.output.composite_score = out.composite_score;
    result.output.threshold = out.threshold_T_dynamic;
    result.output.decision = out.write_decision;
    result.output.effective_focus = out.effective_focus;
    result.output.coherence = cfg.focus; // Placeholder - coherence not in Output yet
    result.output.emotion_intensity = out.emotion_intensity;
    result.output.valence = out.valence;
    result.output.arousal = out.arousal;
    
    // Convert metrics map (enum -> int)
    for (const auto& [metric_enum, value] : out.metrics) {
    result.output.metrics[static_cast<int>(metric_enum)] = value;
  }

  // Wire stored_embedding_id from MemoryStorage operation
  result.output.stored_embedding_id = out.stored_embedding_id;
    
    if (!store)
      {
        return result;
      }

    // Hydrate working memory from database (active WM slots)
    HydrateWorkingMemoryFromDB (store.get (), result.working_memory);

    // Hydrate retrieved memory (long-term retrieval results)
    for (const long long id : out.candidate_memory_ids)
      {
        Cortext::Context::Memory m;
        m.id = id;
        HydrateMemory (store.get (), id, m);
        result.retrieved_memory.push_back (std::move (m));
      }
    return result;
  }
};

std::unique_ptr<Cortext>
Cortext::Create (const Config &cfg, const std::string &db_path,
                 const std::string &models_dir)
{
  return std::unique_ptr<Cortext> (new Cortext (cfg, db_path, models_dir));
}

Cortext::Cortext (const Config &cfg, const std::string &db_path,
                  const std::string &models_dir)
    : impl_ (std::make_unique<Impl> (cfg, db_path, models_dir))
{
}

Cortext::~Cortext () = default;

Cortext::Context
Cortext::ProcessText (const std::string &text, const std::string &source_id)
{
  if (!impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
  telemetry::ScopedSpan span ("cortext.api.process_text");
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  impl_->encoder->EncodeText (text, v);
  encode_span.SetStatusOk ();

  // Build signal with payload for MemoryStorage
  cortext::Signal s;
  s.embedding = ToEigen (v);
  s.timestamp = NowMillis ();
  s.source_id = source_id;
  s.payload = std::vector<unsigned char> (text.begin (), text.end ());
  s.modality = "text";
  s.mimetype = "text/plain";

  auto out = impl_->processor->Process (s);
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  Cortext::Context result = impl_->HydrateContext (out);
  hydrate_span.SetStatusOk ();
  return result;
}

Cortext::Context
Cortext::ProcessAudio (const float *pcm, std::size_t num_samples,
                       const std::string &source_id)
{
  if (!impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
  telemetry::ScopedSpan span ("cortext.api.process_audio");
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  impl_->encoder->EncodeAudio (pcm, num_samples, v);
  encode_span.SetStatusOk ();

  // Build signal with payload for MemoryStorage
  cortext::Signal s;
  s.embedding = ToEigen (v);
  s.timestamp = NowMillis ();
  s.source_id = source_id;
  // Store raw PCM bytes (f32le) as payload
  const std::size_t byte_len = num_samples * sizeof (float);
  s.payload = std::vector<unsigned char> (byte_len);
  std::memcpy (s.payload->data (), pcm, byte_len);
  s.modality = "audio";
  s.mimetype = "audio/pcm;format=f32";
  s.sample_rate = 16000; // ImageBind expects 16kHz
  s.num_samples = num_samples;

  auto out = impl_->processor->Process (s);
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  Cortext::Context result = impl_->HydrateContext (out);
  hydrate_span.SetStatusOk ();
  return result;
}

Cortext::Context
Cortext::ProcessImage (const std::uint8_t *data, int width, int height,
                       int channels, const std::string &source_id)
{
  if (!impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
  telemetry::ScopedSpan span ("cortext.api.process_image");
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  impl_->encoder->EncodeImage (data, width, height, channels, v);
  encode_span.SetStatusOk ();

  // Build signal with payload for MemoryStorage
  cortext::Signal s;
  s.embedding = ToEigen (v);
  s.timestamp = NowMillis ();
  s.source_id = source_id;
  // Store raw image bytes as payload
  const std::size_t byte_len
      = static_cast<std::size_t> (width) * static_cast<std::size_t> (height)
        * static_cast<std::size_t> (channels);
  s.payload = std::vector<unsigned char> (byte_len);
  std::memcpy (s.payload->data (), data, byte_len);
  s.modality = "image";
  s.mimetype = "image/raw"; // Raw pixel bytes (RGB/RGBA)
  s.width = width;
  s.height = height;
  s.channels = channels;

  auto out = impl_->processor->Process (s);
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  Cortext::Context result = impl_->HydrateContext (out);
  hydrate_span.SetStatusOk ();
  return result;
}

Cortext::Context
Cortext::Consolidate ()
{
  if (!impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
  telemetry::ScopedSpan span ("cortext.api.consolidate");
  // Drive the pipeline to allow EvaluateConsolidation to emit events
  // and ConsolidationGate to run scoring/jobs when start is signaled.
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  impl_->encoder->EncodeText (std::string (), v);
  encode_span.SetStatusOk ();
  auto out = impl_->ProcessEmbedding (ToEigen (v), NowMillis (),
                                      "cortext/consolidate");
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  Cortext::Context result = impl_->HydrateContext (out);
  hydrate_span.SetStatusOk ();
  return result;
}

void
Cortext::Flush ()
{
  if (!impl_)
    {
      return; // Safe no-op if not initialized
    }
  if (impl_->processor)
    {
      telemetry::ScopedSpan span ("cortext.api.flush");
      impl_->processor->Flush ();
      span.SetStatusOk ();
    }
}

// Static helpers (simple canonical mime strings and content key format).
std::string
Cortext::MakeContentKey (long long embedding_id)
{
  return std::string ("mem/") + std::to_string (embedding_id);
}

std::string
Cortext::MakeAudioMimePcmF32 ()
{
  return "audio/pcm;format=f32";
}

std::string
Cortext::MakeImageMimePng ()
{
  return "image/png";
}

#if defined(CORTEXT_TESTING)
Cortext::Context
Cortext::DebugHydrateForTest (const std::vector<long long> &candidate_ids,
                              const std::vector<long long> &used_ids)
{
  cortext::SignalProcessor::Output out;
  out.candidate_memory_ids = candidate_ids;
  out.used_memory_ids = used_ids;
  return impl_->HydrateContext (out);
}
#endif

} // namespace cortext
