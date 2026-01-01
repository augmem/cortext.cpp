#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <any>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cortext/core/algorithms.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/core/utils.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/extractor/extractor.hpp>
#include <cortext/operations/accumulator.hpp>
#include <cortext/operations/accumulator_reset.hpp>
#include <cortext/operations/blend.hpp>
#include <cortext/operations/boundary.hpp>
#include <cortext/operations/centroids.hpp>
#include <cortext/operations/coherence.hpp>
#include <cortext/operations/competition.hpp>
#include <cortext/operations/consolidation.hpp>
#include <cortext/operations/consolidation_cluster.hpp>
#include <cortext/operations/consolidation_gate.hpp>
#include <cortext/operations/consolidation_summarize.hpp>
#include <cortext/operations/detect_memory_usage.hpp>
#include <cortext/operations/drift_accumulation.hpp>
#include <cortext/operations/effective_focus.hpp>
#include <cortext/operations/embedding_prediction_error.hpp>
#include <cortext/operations/emotion.hpp>
#include <cortext/operations/emotion_cascade.hpp>
#include <cortext/operations/focus.hpp>
#include <cortext/operations/focus_feedback.hpp>
#include <cortext/operations/focus_spread.hpp>
#include <cortext/operations/graph_build.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/operations/influence.hpp>
#include <cortext/operations/interrupt_gate.hpp>
#include <cortext/operations/memory_storage.hpp>
#include <cortext/operations/memory_strength.hpp>
#include <cortext/operations/metacognitive.hpp>
#include <cortext/operations/precision.hpp>
#include <cortext/operations/predictive.hpp>
#include <cortext/operations/process_extraction_results.hpp>
#include <cortext/operations/recent_context.hpp>
#include <cortext/operations/reconsolidation.hpp>
#include <cortext/operations/sensitivity.hpp>
#include <cortext/operations/sensitivity_feedback.hpp>
#include <cortext/operations/serial_position.hpp>
#include <cortext/operations/serial_position_apply.hpp>
#include <cortext/operations/signal_metrics_persistence.hpp>
#include <cortext/operations/spike_bypass.hpp>
#include <cortext/operations/stability.hpp>
#include <cortext/operations/stability_feedback.hpp>
#include <cortext/operations/streaming_pacing.hpp>
#include <cortext/operations/threshold.hpp>
#include <cortext/operations/uncertainty.hpp>
#include <cortext/operations/working_memory.hpp>
#include <cortext/operations/write_gate.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/signal.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>
#include <cortext/summarizer/summarizer.hpp>

using namespace cortext;

namespace
{
constexpr int kEmbeddingDim = 256;

struct TopicSpec
{
  std::string keyword;
  std::string label;
  int dim;
};

std::string
ToLowerAscii (const std::string &input)
{
  std::string out;
  out.reserve (input.size ());
  for (unsigned char c : input)
    {
      out.push_back (static_cast<char> (std::tolower (c)));
    }
  return out;
}

bool
ContainsKeyword (const std::string &haystack, const std::string &keyword)
{
  if (keyword.empty ())
    return false;
  return haystack.find (keyword) != std::string::npos;
}

Eigen::VectorXf
Normalize (Eigen::VectorXf v)
{
  const float norm = v.norm ();
  if (norm > 1e-9f)
    {
      v /= norm;
    }
  return v;
}

std::unordered_map<std::string, int>
CountCandidateKinds (Store &store, const std::vector<long long> &ids)
{
  std::unordered_map<std::string, int> counts;
  for (const long long id : ids)
    {
      auto rows = store.Execute (
          "SELECT kind FROM memories WHERE memory_id = ?", { id });
      if (rows.empty ())
        continue;
      auto it = rows[0].find ("kind");
      if (it == rows[0].end () || it->second.type () != typeid (std::string))
        continue;
      const std::string kind = std::any_cast<std::string> (it->second);
      counts[kind]++;
    }
  return counts;
}

class KeywordEncoder final : public Encoder
{
public:
  explicit KeywordEncoder (std::vector<TopicSpec> topics)
      : topics_ (std::move (topics))
  {
  }

  void
  EncodeText (const std::string &text,
              std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    const std::string lower = ToLowerAscii (text);
    bool hit = false;
    for (const auto &topic : topics_)
      {
        if (ContainsKeyword (lower, topic.keyword))
          {
            out_embedding[static_cast<size_t> (topic.dim)] += 1.0f;
            hit = true;
          }
      }

    if (!hit)
      {
        out_embedding[static_cast<size_t> (fallback_dim_)] = 1.0f;
      }

    Eigen::VectorXf v (kEmbeddingDim);
    for (int i = 0; i < kEmbeddingDim; ++i)
      {
        v[i] = out_embedding[static_cast<size_t> (i)];
      }
    v = Normalize (v);
    for (int i = 0; i < kEmbeddingDim; ++i)
      {
        out_embedding[static_cast<size_t> (i)] = v[i];
      }
  }

  void
  EncodeAudio (const float * /*pcm*/, std::size_t /*num_samples*/,
               std::vector<float> &out_embedding) override
  {
    EncodeText ("audio", out_embedding);
  }

  void
  EncodeImage (const std::uint8_t * /*data*/, int /*width*/, int /*height*/,
               int /*channels*/, std::vector<float> &out_embedding) override
  {
    EncodeText ("image", out_embedding);
  }

private:
  std::vector<TopicSpec> topics_;
  int fallback_dim_ = 250;
};

class StubSummarizer final : public Summarizer
{
public:
  std::string
  SummarizeTexts (const std::vector<std::string> &texts) override
  {
    if (texts.empty ())
      return {};
    return texts.front ();
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
};

class KeywordExtractor final : public Extractor
{
public:
  explicit KeywordExtractor (std::vector<TopicSpec> topics)
      : topics_ (std::move (topics))
  {
  }

  operations::ExtractionResult
  ExtractFromText (const std::string &text,
                   const nlohmann::json & /*schema*/) override
  {
    operations::ExtractionResult result;
    const std::string lower = ToLowerAscii (text);
    result.labels.push_back ({ "conversation", 1.0 });
    for (const auto &topic : topics_)
      {
        if (ContainsKeyword (lower, topic.keyword))
          {
            result.labels.push_back ({ topic.label, 0.8 });
            break;
          }
      }
    return result;
  }

  operations::ExtractionResult
  ExtractFromAudio (const float * /*pcm*/, size_t /*num_samples*/,
                    const nlohmann::json & /*schema*/) override
  {
    operations::ExtractionResult result;
    result.labels.push_back ({ "conversation", 0.5 });
    return result;
  }

  bool
  IsAvailable () const override
  {
    return true;
  }

private:
  std::vector<TopicSpec> topics_;
};

class ForceConsolidationOnIdleOp final : public IOperation
{
public:
  explicit ForceConsolidationOnIdleOp (std::string idle_source)
      : idle_source_ (std::move (idle_source))
  {
  }

  void
  Execute (OperationContext &context, Transaction & /*tx*/) const override
  {
    if (context.GetSignal ().source_id == idle_source_)
      {
        context.SetConsolidationShouldStart (true);
        auto &p_ctx = context.GetProcessorContext ();
        p_ctx.last_consolidation_ts = context.GetSignal ().timestamp;
      }
  }

private:
  std::string idle_source_;
};

class ForceRetrievalOnQueryOp final : public IOperation
{
public:
  ForceRetrievalOnQueryOp (std::string source, std::string token)
      : source_id_ (std::move (source)), token_ (std::move (token))
  {
  }

  void
  Execute (OperationContext &context, Transaction & /*tx*/) const override
  {
    const auto &signal = context.GetSignal ();
    if (signal.source_id != source_id_)
      {
        return;
      }
    if (signal.payload.has_value ())
      {
        const std::string text (signal.payload->begin (),
                                signal.payload->end ());
        if (ToLowerAscii (text).find (token_) != std::string::npos)
          {
            auto &acc = context.GetProcessorContext ().accumulator_states[signal.source_id];
            if (acc.n_signals == 0 || acc.mu_acc.size () == 0)
              {
                if (last_embedding_.has_value ()
                    && last_embedding_->size () == signal.embedding.size ())
                  {
                    acc.mu_acc = (*last_embedding_ + signal.embedding) * 0.5f;
                    acc.n_signals = 2;
                  }
                else
                  {
                    acc.mu_acc = signal.embedding;
                    acc.n_signals = 1;
                  }
              }
            context.SetShouldCheckRetrieval (true);
          }
      }
    if (signal.embedding.size () > 0)
      {
        last_embedding_ = signal.embedding;
      }
  }

private:
  std::string source_id_;
  std::string token_;
  mutable std::optional<Eigen::VectorXf> last_embedding_;
};

std::unique_ptr<IOperation>
BuildFullPipeline ()
{
  using cortext::OperationSet;
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
  using cortext::operations::CheckSpikeBypass;
  using cortext::operations::CheckStreamingPacing;
  using cortext::operations::ComputeCoherence;
  using cortext::operations::ComputeCompositeScore;
  using cortext::operations::ComputeEffectiveFocus;
  using cortext::operations::ComputeFocusSpread;
  using cortext::operations::ComputeMetrics;
  using cortext::operations::ComputeMniGateDecision;
  using cortext::operations::ConsolidationCluster;
  using cortext::operations::ConsolidationGate;
  using cortext::operations::ConsolidationSummarize;
  using cortext::operations::DetectBoundary;
  using cortext::operations::DetectMemoryUsage;
  using cortext::operations::EnqueueExtractionJobs;
  using cortext::operations::EvaluateConsolidation;
  using cortext::operations::FitMetricWeightsRLS;
  using cortext::operations::GraphAugmentedRetrieveCandidates;
  using cortext::operations::InitializeEmbeddedCentroids;
  using cortext::operations::InitializeFocusPriors;
  using cortext::operations::InitializeSensitivityPriors;
  using cortext::operations::InitializeStabilityPriors;
  using cortext::operations::MemoryStorage;
  using cortext::operations::MetacognitiveMonitoring;
  using cortext::operations::PersistSignalMetrics;
  using cortext::operations::ProcessExtractionResults;
  using cortext::operations::PropagateEmotionalCascade;
  using cortext::operations::ResetAccumulatorAfterFlush;
  using cortext::operations::UpdateAccumulator;
  using cortext::operations::UpdateDriftAccumulation;
  using cortext::operations::UpdateEmbeddingPredictionError;
  using cortext::operations::UpdateFocus;
  using cortext::operations::UpdateMemoryStrength;
  using cortext::operations::UpdateMood;
  using cortext::operations::UpdatePrecisionDelta;
  using cortext::operations::UpdateRateState;
  using cortext::operations::UpdateRecentContext;
  using cortext::operations::UpdateSensitivity;
  using cortext::operations::UpdateStability;
  using cortext::operations::UpdateThreshold;
  using cortext::operations::UpdateUncertainty;
  using cortext::operations::WorkingMemory;
  using cortext::operations::ComputeWriteGate;

  return std::make_unique<OperationSet> (
      std::make_unique<InitializeEmbeddedCentroids> (),
      std::make_unique<InitializeFocusPriors> (),
      std::make_unique<InitializeSensitivityPriors> (),
      std::make_unique<InitializeStabilityPriors> (),

      std::make_unique<UpdateRecentContext> (),

      std::make_unique<ComputeCoherence> (),
      std::make_unique<UpdateDriftAccumulation> (),
      std::make_unique<ComputeFocusSpread> (),
      std::make_unique<UpdateEmbeddingPredictionError> (),
      std::make_unique<UpdateUncertainty> (),

      std::make_unique<UpdateFocus> (),
      std::make_unique<UpdateSensitivity> (),
      std::make_unique<UpdateMood> (),

      std::make_unique<ComputeEffectiveFocus> (),
      std::make_unique<ComputeMetrics> (),
      std::make_unique<FitMetricWeightsRLS> (),
      std::make_unique<ComputeCompositeScore> (),

      std::make_unique<UpdatePrecisionDelta> (),
      std::make_unique<UpdateThreshold> (),

      std::make_unique<UpdateAccumulator> (),
      std::make_unique<DetectBoundary> (),
      std::make_unique<CheckSpikeBypass> (),
      std::make_unique<ComputeWriteGate> (),
      std::make_unique<MemoryStorage> (),
      std::make_unique<PersistSignalMetrics> (),
      std::make_unique<ResetAccumulatorAfterFlush> (),
      std::make_unique<UpdateRateState> (),

      std::make_unique<CheckStreamingPacing> (),
      std::make_unique<ForceRetrievalOnQueryOp> ("chat", "budget"),
      std::make_unique<GraphAugmentedRetrieveCandidates> (),

      std::make_unique<ComputeMniGateDecision> (),
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
      std::make_unique<ForceConsolidationOnIdleOp> ("cortext/idle"),
      std::make_unique<EvaluateConsolidation> (),
      std::make_unique<ConsolidationGate> (),
      std::make_unique<ConsolidationCluster> (),
      std::make_unique<ConsolidationSummarize> (),
      std::make_unique<EnqueueExtractionJobs> (),
      std::make_unique<ProcessExtractionResults> (),
      std::make_unique<BuildGraphFromConsolidation> (),
      std::make_unique<PropagateEmotionalCascade> ());
}

Signal
MakeTextSignal (KeywordEncoder &encoder, const std::string &text,
                uint64_t ts, const std::string &source_id)
{
  std::vector<float> vec;
  encoder.EncodeText (text, vec);
  Eigen::VectorXf emb (static_cast<Eigen::Index> (vec.size ()));
  for (size_t i = 0; i < vec.size (); ++i)
    {
      emb[static_cast<Eigen::Index> (i)] = vec[i];
    }

  Signal s;
  s.embedding = emb;
  s.timestamp = ts;
  s.source_id = source_id;
  s.payload = std::vector<unsigned char> (text.begin (), text.end ());
  s.modality = "text";
  s.mimetype = "text/plain";
  return s;
}

std::unordered_set<long long>
TopKByCosine (const Eigen::VectorXf &q,
              const std::vector<std::pair<long long, Eigen::VectorXf>> &rows,
              int k)
{
  struct Scored
  {
    long long id;
    double score;
  };
  std::vector<Scored> scored;
  scored.reserve (rows.size ());
  for (const auto &row : rows)
    {
      const Eigen::VectorXf &v = row.second;
      if (v.size () != q.size ())
        continue;
      const double sim = core::CosineSimilarity (q, v);
      scored.push_back (Scored{ row.first, sim });
    }

  std::sort (scored.begin (), scored.end (),
             [] (const Scored &a, const Scored &b) {
               if (a.score != b.score)
                 return a.score > b.score;
               return a.id < b.id;
             });

  if (k < 0)
    k = 0;
  if (static_cast<int> (scored.size ()) > k)
    {
      scored.resize (static_cast<size_t> (k));
    }

  std::unordered_set<long long> ids;
  for (const auto &s : scored)
    {
      ids.insert (s.id);
    }
  return ids;
}

std::vector<std::pair<long long, Eigen::VectorXf>>
LoadEmbeddings (Store &store)
{
  std::vector<std::pair<long long, Eigen::VectorXf>> rows_out;
  auto rows = store.Execute ("SELECT embedding_id, embedding FROM embeddings", {});
  for (const auto &row : rows)
    {
      auto it_id = row.find ("embedding_id");
      auto it_emb = row.find ("embedding");
      if (it_id == row.end () || it_emb == row.end ())
        continue;
      if (it_id->second.type () != typeid (long long))
        continue;
      const long long emb_id = std::any_cast<long long> (it_id->second);
      Eigen::VectorXf v;
      if (!core::DecodeFloatBlob (it_emb->second, kEmbeddingDim, v))
        continue;
      rows_out.emplace_back (emb_id, v);
    }
  return rows_out;
}

std::optional<Eigen::VectorXf>
LookupEmbeddingForId (Store &store,
                      const std::unordered_map<long long, Eigen::VectorXf> &map,
                      long long id)
{
  (void)map;

  auto emb_rows = store.Execute (
      "SELECT embedding FROM embeddings WHERE embedding_id = ?", { id });
  if (!emb_rows.empty () && emb_rows[0].count ("embedding") != 0)
    {
      Eigen::VectorXf v;
      if (core::DecodeFloatBlob (emb_rows[0].at ("embedding"), kEmbeddingDim, v))
        {
          return v;
        }
    }

  auto rows = store.Execute (
      "SELECT e.embedding "
      "FROM memories m "
      "JOIN embeddings e ON m.embedding_id = e.embedding_id "
      "WHERE m.memory_id = ?",
      { id });
  if (rows.empty () || rows[0].count ("embedding") == 0)
    {
      return std::nullopt;
    }

  Eigen::VectorXf v;
  if (!core::DecodeFloatBlob (rows[0].at ("embedding"), kEmbeddingDim, v))
    {
      return std::nullopt;
    }
  return v;
}

} // namespace

TEST_CASE ("Integration: chat memories consolidate and retrieve", "[integration][e2e][chat]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<TopicSpec> topics = {
    { "paris", "travel", 0 },
    { "budget", "finance", 1 },
    { "garden", "garden", 2 },
    { "running", "fitness", 3 },
    { "pasta", "cooking", 4 },
  };

  KeywordEncoder encoder (topics);
  StubSummarizer summarizer;
  KeywordExtractor extractor (topics);

  SignalProcessor::Config cfg;
  cfg.focus = 0.2;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.0;
  cfg.encoder = &encoder;
  cfg.summarizer = &summarizer;
  cfg.extractor = &extractor;

  const int min_cluster_size = core::MinClusterSize (cfg.focus);
  const int label_threshold = core::LabelFrequencyThreshold (cfg.stability);
  REQUIRE (static_cast<int> (topics.size ()) >= label_threshold);
  const int units_per_topic = min_cluster_size + 2;
  const std::string source_id = "chat";

  auto pipeline = BuildFullPipeline ();
  SignalProcessor processor (cfg, store, std::move (pipeline));

  uint64_t ts = 100000;
  const uint64_t gap_ms = 40000;
  for (const auto &topic : topics)
    {
      for (int unit = 0; unit < units_per_topic; ++unit)
        {
          const std::string text1
              = "User: plan " + topic.keyword + " " + std::to_string (unit);
          processor.Process (MakeTextSignal (encoder, text1, ts, source_id));
          ts += 1000;

          const std::string text2
              = "Assistant: details on " + topic.keyword + " "
                + std::to_string (unit);
          processor.Process (MakeTextSignal (encoder, text2, ts, source_id));
          ts += gap_ms;

          const std::string boundary
              = "User: (pause) " + topic.keyword;
          processor.Process (MakeTextSignal (encoder, boundary, ts, source_id));
          ts += 1000;
        }
    }

  const long long expected_memories_min
      = static_cast<long long> (topics.size ()) * min_cluster_size;
  auto mem_rows
      = store->Execute ("SELECT COUNT(*) AS c FROM memories WHERE kind = 'LONG_TERM'", {});
  REQUIRE (!mem_rows.empty ());
  REQUIRE (cortext::testing::GetInt64 (mem_rows[0], "c") >= expected_memories_min);

  // Consolidation tick: force idle gap so EvaluateConsolidation can start.
  ts += 4000000;
  const std::string idle_text = "idle";
  processor.Process (MakeTextSignal (encoder, idle_text, ts, "cortext/idle"));

  auto assoc_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM memories WHERE kind = 'ASSOCIATION'", {});
  REQUIRE (!assoc_rows.empty ());
  REQUIRE (cortext::testing::GetInt64 (assoc_rows[0], "c") >= 1);

  auto label_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM memories "
      "WHERE kind = 'LABEL' AND label = 'conversation'",
      {});
  REQUIRE (!label_rows.empty ());
  REQUIRE (cortext::testing::GetInt64 (label_rows[0], "c") >= 1);

  auto derived_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM associations WHERE edge_type = 'derived_from'",
      {});
  REQUIRE (!derived_rows.empty ());
  REQUIRE (cortext::testing::GetInt64 (derived_rows[0], "c") > 0);

  auto label_edges = store->Execute (
      "SELECT COUNT(*) AS c FROM associations WHERE edge_type = 'has_label'", {});
  REQUIRE (!label_edges.empty ());
  REQUIRE (cortext::testing::GetInt64 (label_edges[0], "c") > 0);

  auto co_occurs = store->Execute (
      "SELECT COUNT(*) AS c FROM associations WHERE edge_type = 'co_occurs'", {});
  REQUIRE (!co_occurs.empty ());
  REQUIRE (cortext::testing::GetInt64 (co_occurs[0], "c") > 0);

  auto clustered = store->Execute (
      "SELECT COUNT(*) AS c FROM memories "
      "WHERE kind = 'LONG_TERM' AND cluster_id IS NOT NULL",
      {});
  REQUIRE (!clustered.empty ());
  REQUIRE (cortext::testing::GetInt64 (clustered[0], "c")
           >= expected_memories_min);

  // Retrieval test: ensure mu_acc (mean of two signals) drives KNN results.
  const std::string query_a = "User: paris plans";
  const std::string query_b = "User: budget review";
  ts += 1000;
  auto out1 = processor.Process (MakeTextSignal (encoder, query_a, ts, source_id));
  (void)out1;
  ts += 400000;
  auto out2 = processor.Process (MakeTextSignal (encoder, query_b, ts, source_id));

  const int k = std::max (1, core::MaxResults (cfg.focus));
  auto embeddings = LoadEmbeddings (*store);
  REQUIRE (!embeddings.empty ());

  std::vector<float> v_a;
  encoder.EncodeText (query_a, v_a);
  Eigen::VectorXf e_a (kEmbeddingDim);
  for (int i = 0; i < kEmbeddingDim; ++i)
    {
      e_a[i] = v_a[static_cast<size_t> (i)];
    }

  std::vector<float> v_b;
  encoder.EncodeText (query_b, v_b);
  Eigen::VectorXf e_b (kEmbeddingDim);
  for (int i = 0; i < kEmbeddingDim; ++i)
    {
      e_b[i] = v_b[static_cast<size_t> (i)];
    }

  const Eigen::VectorXf mu_acc = (e_a + e_b) * 0.5f;

  const auto expected_last = TopKByCosine (e_b, embeddings, k);
  const std::unordered_set<long long> actual_ids (
      out2.candidate_memory_ids.begin (), out2.candidate_memory_ids.end ());

  REQUIRE (!actual_ids.empty ());

  std::unordered_set<long long> combined_ids (out1.candidate_memory_ids.begin (),
                                              out1.candidate_memory_ids.end ());
  combined_ids.insert (out2.candidate_memory_ids.begin (),
                       out2.candidate_memory_ids.end ());
  std::vector<long long> combined_list (combined_ids.begin (),
                                        combined_ids.end ());
  const auto kind_counts = CountCandidateKinds (*store, combined_list);
  REQUIRE (kind_counts.count ("ASSOCIATION") > 0);
  REQUIRE (kind_counts.count ("LABEL") > 0);

  std::unordered_map<long long, Eigen::VectorXf> embedding_map;
  embedding_map.reserve (embeddings.size ());
  for (const auto &row : embeddings)
    {
      embedding_map.emplace (row.first, row.second);
    }

  std::vector<double> mu_scores;
  mu_scores.reserve (embeddings.size ());
  for (const auto &row : embeddings)
    {
      mu_scores.push_back (core::CosineSimilarity (mu_acc, row.second));
    }
  std::sort (mu_scores.begin (), mu_scores.end (),
             [] (double a, double b) { return a > b; });
  const double mu_threshold
      = mu_scores.empty ()
            ? 0.0
            : mu_scores[static_cast<size_t> (
                  std::min (static_cast<int> (mu_scores.size ()) - 1, k - 1))];

  std::size_t count_high = 0;
  std::size_t resolved = 0;
  for (const auto &id : actual_ids)
    {
      auto emb = LookupEmbeddingForId (*store, embedding_map, id);
      if (!emb.has_value ())
        {
          continue;
        }
      ++resolved;
      const double sim = core::CosineSimilarity (mu_acc, *emb);
      if (sim + 1e-6 >= mu_threshold)
        {
          ++count_high;
        }
    }
  REQUIRE (resolved > 0);
  REQUIRE (count_high >= 1);

  double max_last = -1.0;
  for (const auto &row : embeddings)
    {
      max_last = std::max (max_last,
                           core::CosineSimilarity (e_b, row.second));
    }

  bool saw_non_last = false;
  for (const auto &id : actual_ids)
    {
      auto emb = LookupEmbeddingForId (*store, embedding_map, id);
      if (!emb.has_value ())
        continue;
      const double sim_last = core::CosineSimilarity (e_b, *emb);
      if (sim_last + 1e-6 < max_last)
        {
          saw_non_last = true;
          break;
        }
    }
  REQUIRE (saw_non_last);
  REQUIRE (actual_ids != expected_last);
}
