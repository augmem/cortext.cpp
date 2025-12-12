#include "cortext/cortext.hpp"
#include "cortext/encoder/imagebind.hpp"

#include "cortext/processor.hpp"
#include "cortext/processor/operation_set.hpp"
#include "cortext/signal.hpp"
#include "cortext/store/sqlite_store.hpp"

#include "cortext/operations/consolidation.hpp"
#include "cortext/operations/blend.hpp"
#include "cortext/operations/graph_build.hpp"
#include "cortext/operations/graph_retrieval.hpp"
#include "cortext/operations/graph_schema.hpp"
#include "cortext/operations/goal_alignment.hpp"
#include "cortext/operations/logprob_surprise.hpp"
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
#include "cortext/operations/stability.hpp"
#include "cortext/operations/stability_feedback.hpp"
#include "cortext/operations/working_memory.hpp"

#include "cortext/operations/consolidation_gate.hpp"

namespace cortext
{
namespace
{
inline Eigen::VectorXf
ToEigen (const std::vector<float> &v)
{
  Eigen::VectorXf e (static_cast<int> (v.size ()));
  for (int i = 0; i < e.size (); ++i)
    {
      e[i] = v[static_cast<std::size_t> (i)];
    }
  return e;
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
    using cortext::operations::ComputeGoalAlignment;
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
    using cortext::operations::UpdateLogprobSurprise;
    using cortext::operations::UpdateMemoryStrength;
    using cortext::operations::UpdatePrecisionDelta;
    using cortext::operations::UpdateSensitivity;
    using cortext::operations::UpdateStability;
    using cortext::operations::UpdateThreshold;
    using cortext::operations::UpdateUncertainty;
    using cortext::operations::WorkingMemory;

    pipeline_root = std::make_unique<OperationSet> (
        std::make_unique<EnsureGraphSchema> (),

        std::make_unique<InitializeFocusPriors> (),
        std::make_unique<InitializeSensitivityPriors> (),
        std::make_unique<InitializeStabilityPriors> (),

        std::make_unique<UpdateFocus> (),
        std::make_unique<UpdateSensitivity> (),

        std::make_unique<ComputeCoherence> (),
        std::make_unique<ComputeFocusSpread> (),
        std::make_unique<ComputeEffectiveFocus> (),
        std::make_unique<CheckEpisodeBoundary> (),

        std::make_unique<UpdateLogprobSurprise> (),
        std::make_unique<UpdateUncertainty> (),
        std::make_unique<ComputeMetrics> (),
        std::make_unique<FitMetricWeightsRLS> (),
        std::make_unique<ComputeCompositeScore> (),

        std::make_unique<UpdatePrecisionDelta> (),
        std::make_unique<UpdateThreshold> (),

        std::make_unique<GraphAugmentedRetrieveCandidates> (),
        std::make_unique<ComputeGoalAlignment> (),
        std::make_unique<ComputeMniGateDecision> (),

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
        std::make_unique<BuildGraphFromConsolidation> ());

    cortext::SignalProcessor::Config pcfg;
    pcfg.focus = cfg.focus;
    pcfg.sensitivity = cfg.sensitivity;
    pcfg.stability = cfg.stability;

    processor = std::make_unique<cortext::SignalProcessor> (
        pcfg, store, std::move (pipeline_root));

    // Ensure memory_index schema exists (idempotent).
    try
      {
        store->Execute ("CREATE TABLE IF NOT EXISTS memory_index ("
                        "  embedding_id INTEGER PRIMARY KEY,"
                        "  modality TEXT,"
                        "  mime TEXT,"
                        "  content_key TEXT,"
                        "  source_id TEXT,"
                        "  timestamp INTEGER,"
                        "  width INTEGER,"
                        "  height INTEGER,"
                        "  channels INTEGER,"
                        "  sample_rate INTEGER,"
                        "  num_samples INTEGER,"
                        "  blob_id BLOB"
                        ")");
      }
    catch (...)
      {
        // Ignore schema errors; operations are resilient.
      }
    // Attempt to add blob_id column for pre-existing databases (ignore
    // errors).
    try
      {
        store->Execute ("ALTER TABLE memory_index ADD COLUMN blob_id BLOB",
                        {});
      }
    catch (...)
      {
      }
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

  static std::vector<unsigned char>
  ToUnsignedVector (const std::vector<char> &blob)
  {
    if (blob.empty ())
      {
        return {};
      }
    std::vector<unsigned char> out;
    out.resize (blob.size ());
    std::memcpy (out.data (), blob.data (), blob.size ());
    return out;
  }

  bool
  LoadObjstorePayload (const std::vector<unsigned char> &blob_id,
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
                std::vector<unsigned char> bytes;
                if (it->second.type () == typeid (std::vector<char>))
                  {
                    const auto &blob
                        = std::any_cast<const std::vector<char> &> (
                            it->second);
                    bytes = ToUnsignedVector (blob);
                  }
                else if (it->second.type ()
                         == typeid (std::vector<unsigned char>))
                  {
                    bytes = std::any_cast<const std::vector<unsigned char> &> (
                        it->second);
                  }
                if (!bytes.empty ())
                  {
                    out.assign (reinterpret_cast<const char *> (bytes.data ()),
                                static_cast<std::size_t> (bytes.size ()));
                    return true;
                  }
              }
          }
      }
    catch (...)
      {
      }
    return false;
  }

  Cortext::Context
  HydrateContext (const cortext::SignalProcessor::Output &out)
  {
    Cortext::Context result;
    result.should_interrupt = out.interrupt_allowed;
    if (!store)
      {
        return result;
      }

    for (const long long id : out.candidate_memory_ids)
      {
        Cortext::Context::Memory m;
        m.id = id;

        // memory_index
        try
          {
            auto rows = store->Execute (
                "SELECT modality, mime, source_id, timestamp, blob_id "
                "FROM memory_index WHERE embedding_id = ?",
                { id });
            if (!rows.empty ())
              {
                const auto &row = rows[0];
                auto get_s = [&row] (const char *k) -> std::string {
                  auto it = row.find (k);
                  if (it == row.end ())
                    return {};
                  if (it->second.type () == typeid (std::string))
                    return std::any_cast<std::string> (it->second);
                  return {};
                };
                auto get_ll = [&row] (const char *k) -> long long {
                  auto it = row.find (k);
                  if (it == row.end ())
                    return 0LL;
                  if (it->second.type () == typeid (long long))
                    return std::any_cast<long long> (it->second);
                  if (it->second.type () == typeid (int))
                    return static_cast<long long> (
                        std::any_cast<int> (it->second));
                  return 0LL;
                };
                m.modality = get_s ("modality");
                m.mimetype = get_s ("mime");
                m.source_id = get_s ("source_id");
                m.timestamp
                    = static_cast<std::uint64_t> (get_ll ("timestamp"));
                auto get_blob = [&row] (const char *k) {
                  auto it = row.find (k);
                  if (it == row.end ())
                    {
                      return std::vector<unsigned char> ();
                    }
                  if (it->second.type () == typeid (std::vector<char>))
                    {
                      const auto &blob
                          = std::any_cast<const std::vector<char> &> (
                              it->second);
                      return ToUnsignedVector (blob);
                    }
                  if (it->second.type ()
                      == typeid (std::vector<unsigned char>))
                    {
                      return std::any_cast<
                          const std::vector<unsigned char> &> (it->second);
                    }
                  return std::vector<unsigned char> ();
                };
                const auto blob_id_bytes = get_blob ("blob_id");
                if (!blob_id_bytes.empty ())
                  {
                    std::string payload;
                    if (LoadObjstorePayload (blob_id_bytes, payload))
                      {
                        m.content = std::move (payload);
                      }
                  }
              }
          }
        catch (...)
          {
            // ignore
          }

        // memory_feedback metrics
        try
          {
            auto rows = store->Execute (
                "SELECT retrieved_count, used_count "
                "FROM memory_feedback WHERE embedding_id = ?",
                { id });
            if (!rows.empty ())
              {
                const auto &row = rows[0];
                auto get_ll = [&row] (const char *k) -> long long {
                  auto it = row.find (k);
                  if (it == row.end ())
                    return 0LL;
                  if (it->second.type () == typeid (long long))
                    return std::any_cast<long long> (it->second);
                  if (it->second.type () == typeid (int))
                    return static_cast<long long> (
                        std::any_cast<int> (it->second));
                  return 0LL;
                };
                m.retrieved_count = get_ll ("retrieved_count");
                m.used_count = get_ll ("used_count");
              }
          }
        catch (...)
          {
            // ignore
          }

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
  std::vector<float> v;
  impl_->encoder->EncodeText (text, v);
  auto out = impl_->ProcessEmbedding (ToEigen (v), timestamp, source_id);
  return impl_->HydrateContext (out);
}

Cortext::Context
Cortext::ProcessAudio (const float *pcm, std::size_t num_samples,
                       std::uint64_t timestamp, const std::string &source_id)
{
  std::vector<float> v;
  impl_->encoder->EncodeAudio (pcm, num_samples, v);
  auto out = impl_->ProcessEmbedding (ToEigen (v), timestamp, source_id);
  return impl_->HydrateContext (out);
}

Cortext::Context
Cortext::ProcessImage (const std::uint8_t *data, int width, int height,
                       int channels, std::uint64_t timestamp,
                       const std::string &source_id)
{
  std::vector<float> v;
  impl_->encoder->EncodeImage (data, width, height, channels, v);
  auto out = impl_->ProcessEmbedding (ToEigen (v), timestamp, source_id);
  return impl_->HydrateContext (out);
}

Cortext::Context
Cortext::Consolidate (std::uint64_t now_timestamp)
{
  // Drive the pipeline to allow EvaluateConsolidation to emit events
  // and ConsolidationGate to run scoring/jobs when start is signaled.
  std::vector<float> v;
  impl_->encoder->EncodeText (std::string (), v);
  auto out = impl_->ProcessEmbedding (ToEigen (v), now_timestamp,
                                      "cortext/consolidate");
  return impl_->HydrateContext (out);
}

void
Cortext::Flush ()
{
  if (impl_->processor)
    {
      impl_->processor->Flush ();
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
