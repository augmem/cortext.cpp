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
#include "cortext/operations/graph_schema.hpp"
#include "cortext/operations/goal_alignment.hpp"
#include "cortext/operations/goal_alignment_fallback.hpp"
#include "cortext/operations/embedding_prediction_error.hpp"
#include "cortext/operations/centroids.hpp"
#include "cortext/operations/precision.hpp"
#include "cortext/operations/sensitivity.hpp"
#include "cortext/operations/threshold.hpp"
#include "cortext/operations/uncertainty.hpp"
#include "cortext/processor/operation.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cstring>
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
#include "cortext/operations/predictive.hpp"
#include "cortext/operations/reconsolidation.hpp"
#include "cortext/operations/sensitivity_feedback.hpp"
#include "cortext/operations/serial_position.hpp"
#include "cortext/operations/serial_position_apply.hpp"
#include "cortext/operations/signal_metrics_persistence.hpp"
#include "cortext/operations/stability.hpp"
#include "cortext/operations/write_gate.hpp"
#include "cortext/operations/memory_storage.hpp"
#include "cortext/operations/stability_feedback.hpp"
#include "cortext/operations/generation_trace.hpp"
#include "cortext/operations/working_memory.hpp"
#include "cortext/operations/detect_memory_usage.hpp"

#include "cortext/operations/consolidation_gate.hpp"
#include "cortext/telemetry/telemetry.hpp"

namespace cortext
{

namespace
{
/// @brief Converts a std::vector<float> to an Eigen::VectorXf using Eigen::Map.
inline Eigen::VectorXf
ToEigen (const std::vector<float> &v)
{
  return Eigen::Map<const Eigen::VectorXf> (v.data (),
                                            static_cast<int> (v.size ()));
}


bool
LoadObjstorePayload (Store *store, const std::vector<unsigned char> &blob_id,
                     std::string &out)
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
              const auto bytes = store::BlobFromAny (it->second);
              if (!bytes.empty ())
                {
                  out.assign (reinterpret_cast<const char *> (bytes.data ()),
                              static_cast<std::size_t> (bytes.size ()));
                  return true;
                }
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

void
HydrateMemory (Store *store, long long id, Cortext::Context::Memory &m)
{
  if (!store)
    return;
  try
    {
      auto rows = store->Execute (
          "SELECT "
          "  mi.modality, mi.mime, mi.source_id, mi.timestamp, mi.blob_id, "
          "  COALESCE(mf.retrieved_count, 0) AS retrieved_count, "
          "  COALESCE(mf.used_count, 0) AS used_count, "
          "  sm.relevance, sm.mismatch, sm.surprise, sm.rarity, sm.drift, "
          "  sm.contradiction, sm.utility, sm.periphery, sm.coverage, "
          "  sm.salience, sm.valence, sm.arousal, sm.composite_score, "
          "  sm.threshold_t "
          "FROM memory_index mi "
          "LEFT JOIN memory_feedback mf ON mi.embedding_id = mf.embedding_id "
          "LEFT JOIN signal_metrics sm ON mi.embedding_id = sm.embedding_id "
          "WHERE mi.embedding_id = ?",
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

          auto get_blob = [&row] (const char *k) {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return std::vector<unsigned char> ();
            return store::BlobFromAny (it->second);
          };

          // Populate from memory_index
          m.modality = get_s ("modality");
          m.mimetype = get_s ("mime");
          m.source_id = get_s ("source_id");
          m.timestamp = static_cast<std::uint64_t> (get_ll ("timestamp"));

          // Load content from objstore if blob_id present
          const auto blob_id_bytes = get_blob ("blob_id");
          if (!blob_id_bytes.empty ())
            {
              std::string payload;
              if (LoadObjstorePayload (store, blob_id_bytes, payload))
                m.content = std::move (payload);
            }

          // Populate from memory_feedback
          m.retrieved_count = get_ll ("retrieved_count");
          m.used_count = get_ll ("used_count");

          // Populate from signal_metrics
          m.metrics.relevance = get_dbl ("relevance");
          m.metrics.mismatch = get_dbl ("mismatch");
          m.metrics.surprise = get_dbl ("surprise");
          m.metrics.rarity = get_dbl ("rarity");
          m.metrics.drift = get_dbl ("drift");
          m.metrics.contradiction = get_dbl ("contradiction");
          m.metrics.utility = get_dbl ("utility");
          m.metrics.periphery = get_dbl ("periphery");
          m.metrics.coverage = get_dbl ("coverage");
          m.metrics.salience = get_dbl ("salience");
          m.metrics.valence = get_dbl ("valence", 0.5);
          m.metrics.arousal = get_dbl ("arousal");
          m.metrics.composite_score = get_dbl ("composite_score");
          m.metrics.threshold_t = get_dbl ("threshold_t");
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

  Impl (const Config &c, std::string db, std::string models)
      : cfg (c), db_path (std::move (db)), models_dir (std::move (models))
  {
    // Store
    auto uniq = cortext::SQLiteStore::Create (db_path.c_str ());
    store = std::shared_ptr<cortext::Store> (std::move (uniq));

    // Encoder stub (ImageBind-oriented)
    encoder = std::make_unique<ImageBindEncoder> (models_dir);

    // Default pipeline: full per-signal processing chain.
    using cortext::OperationSet;
    using cortext::operations::ConsolidationGate;
    using cortext::operations::EvaluateConsolidation;
    using cortext::operations::FitMetricWeightsRLS;
    using cortext::operations::GraphAugmentedRetrieveCandidates;
    using cortext::operations::InitializeEmbeddedCentroids;
    using cortext::operations::ComputeGoalAlignment;
    using cortext::operations::ComputeGoalAlignmentFallback;
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
    using cortext::operations::CheckEpisodeBoundary;
    using cortext::operations::ComputeMetrics;
    using cortext::operations::EnsureGraphSchema;
    using cortext::operations::MetacognitiveMonitoring;
    using cortext::operations::UpdateFocus;
    using cortext::operations::UpdateEmbeddingPredictionError;
    using cortext::operations::UpdateMemoryStrength;
    using cortext::operations::UpdateMood;
    using cortext::operations::UpdatePrecisionDelta;
    using cortext::operations::UpdateSensitivity;
    using cortext::operations::UpdateStability;
    using cortext::operations::UpdateThreshold;
    using cortext::operations::UpdateUncertainty;
    using cortext::operations::WorkingMemory;
    using cortext::operations::PersistSignalMetrics;
    using cortext::operations::RecordGenerationTrace;
    using cortext::operations::ComputeWriteGate;
    using cortext::operations::MemoryStorage;
    using cortext::operations::DetectMemoryUsage;

    pipeline_root = std::make_unique<OperationSet> (
        std::make_unique<EnsureGraphSchema> (),

        std::make_unique<InitializeEmbeddedCentroids> (),

        std::make_unique<InitializeFocusPriors> (),
        std::make_unique<InitializeSensitivityPriors> (),
        std::make_unique<InitializeStabilityPriors> (),

        // Implicit feedback: detect if cached retrievals were "used" in this
        // signal. Must run before feedback operations (ApplyFocusFeedback,
        // ApplySensitivityFeedback, etc.) that consume MemoryUsageEvents.
        std::make_unique<DetectMemoryUsage> (),

        std::make_unique<UpdateFocus> (),
        std::make_unique<UpdateSensitivity> (),
        std::make_unique<UpdateMood> (),

        std::make_unique<ComputeCoherence> (),
        std::make_unique<ComputeFocusSpread> (),
        std::make_unique<ComputeEffectiveFocus> (),
        std::make_unique<CheckEpisodeBoundary> (),

        std::make_unique<UpdateEmbeddingPredictionError> (),
        std::make_unique<UpdateUncertainty> (),
        std::make_unique<ComputeMetrics> (),
        std::make_unique<FitMetricWeightsRLS> (),
        std::make_unique<ComputeCompositeScore> (),

        std::make_unique<UpdatePrecisionDelta> (),
        std::make_unique<UpdateThreshold> (),
        std::make_unique<ComputeWriteGate> (),
        std::make_unique<MemoryStorage> (),
        std::make_unique<PersistSignalMetrics> (),

        std::make_unique<GraphAugmentedRetrieveCandidates> (),
        std::make_unique<ComputeGoalAlignment> (),
        std::make_unique<ComputeGoalAlignmentFallback> (),
        std::make_unique<ComputeMniGateDecision> (),

        std::make_unique<ApplyRetrievalCompetition> (),
        std::make_unique<ApplyPredictivePreActivation> (),
        std::make_unique<ApplyReconsolidation> (),

        std::make_unique<ApplyFocusFeedback> (),
        std::make_unique<ApplySensitivityFeedback> (),
        std::make_unique<ApplyStabilityFeedback> (),
        std::make_unique<UpdateStability> (),
        std::make_unique<ApplyInfluenceFeedback> (),
        std::make_unique<RecordGenerationTrace> (),

        std::make_unique<ApplySerialPositionEffects> (),
        std::make_unique<ApplySerialPositionMultiplier> (),
        std::make_unique<UpdateMemoryStrength> (),
        std::make_unique<ApplyEmotionalConsolidation> (),
        std::make_unique<WorkingMemory> (),
        std::make_unique<MetacognitiveMonitoring> (),

        std::make_unique<EvaluateConsolidation> (),
        std::make_unique<ConsolidationGate> (),
        std::make_unique<BuildGraphFromConsolidation> ());

    cortext::SignalProcessor::Config pcfg;
    pcfg.focus = cfg.focus;
    pcfg.sensitivity = cfg.sensitivity;
    pcfg.stability = cfg.stability;

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
    for (const long long id : out.candidate_memory_ids)
      {
        Cortext::Context::Memory m;
        m.id = id;
        HydrateMemory (store.get (), id, m);
        result.memories.push_back (std::move (m));
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
Cortext::ProcessText (const std::string &text, std::uint64_t timestamp,
                      const std::string &source_id)
{
  telemetry::ScopedSpan span ("cortext.api.process_text");
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  impl_->encoder->EncodeText (text, v);
  encode_span.SetStatusOk ();

  // Build signal with payload for MemoryStorage
  cortext::Signal s;
  s.embedding = ToEigen (v);
  s.timestamp = timestamp;
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
                       std::uint64_t timestamp, const std::string &source_id)
{
  telemetry::ScopedSpan span ("cortext.api.process_audio");
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  impl_->encoder->EncodeAudio (pcm, num_samples, v);
  encode_span.SetStatusOk ();

  // Build signal with payload for MemoryStorage
  cortext::Signal s;
  s.embedding = ToEigen (v);
  s.timestamp = timestamp;
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
                       int channels, std::uint64_t timestamp,
                       const std::string &source_id)
{
  telemetry::ScopedSpan span ("cortext.api.process_image");
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  impl_->encoder->EncodeImage (data, width, height, channels, v);
  encode_span.SetStatusOk ();

  // Build signal with payload for MemoryStorage
  cortext::Signal s;
  s.embedding = ToEigen (v);
  s.timestamp = timestamp;
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
Cortext::Consolidate (std::uint64_t now_timestamp)
{
  telemetry::ScopedSpan span ("cortext.api.consolidate");
  // Drive the pipeline to allow EvaluateConsolidation to emit events
  // and ConsolidationGate to run scoring/jobs when start is signaled.
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  impl_->encoder->EncodeText (std::string (), v);
  encode_span.SetStatusOk ();
  auto out = impl_->ProcessEmbedding (ToEigen (v), now_timestamp,
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
