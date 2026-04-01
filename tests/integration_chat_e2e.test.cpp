#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <any>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cortext/cortext.hpp>
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

#include "../src/streaming_text_probe.hpp"

using namespace cortext;

namespace
{
constexpr int kEmbeddingDim = 256;
constexpr std::size_t kStreamingProbeMinChars = 32;
constexpr std::size_t kStreamingProbeMaxChars = 128;

struct TopicSpec
{
  std::string keyword;
  std::string label;
  int dim;
};

struct ScopedTempDb
{
  ScopedTempDb ()
  {
    namespace fs = std::filesystem;
    const auto stamp = std::chrono::high_resolution_clock::now ()
                           .time_since_epoch ()
                           .count ();
    path = (fs::temp_directory_path ()
            / ("cortext_chat_e2e_" + std::to_string (stamp) + ".db"))
               .string ();
  }

  ~ScopedTempDb ()
  {
    std::error_code ec;
    std::filesystem::remove (path, ec);
    std::filesystem::remove (path + "-wal", ec);
    std::filesystem::remove (path + "-shm", ec);
  }

  std::string path;
};

struct ChatMessage
{
  std::string role;
  std::string content;
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

std::string
RepoModelsDir ()
{
  namespace fs = std::filesystem;
  const fs::path root = fs::path (__FILE__).parent_path ().parent_path ();
  return (root / "models").string ();
}

std::string
ExtractTextFromBlobs (const std::vector<std::vector<unsigned char>> &blobs)
{
  std::string result;
  for (const auto &blob : blobs)
    {
      if (!blob.empty ())
        {
          result.append (reinterpret_cast<const char *> (blob.data ()),
                         blob.size ());
        }
    }
  return result;
}

std::string
RoleFromSourceId (const std::string &source_id)
{
  if (source_id == "chat/user")
    return "user";
  if (source_id == "chat/assistant")
    return "assistant";
  return {};
}

std::vector<ChatMessage>
BuildPromptMessages (
    const std::vector<cortext::Cortext::Context::Memory> &working_memory,
    const std::string &latest_user_input)
{
  std::vector<const cortext::Cortext::Context::Memory *> ordered;
  ordered.reserve (working_memory.size ());
  for (const auto &mem : working_memory)
    {
      ordered.push_back (&mem);
    }

  std::sort (ordered.begin (), ordered.end (),
             [] (const auto *a, const auto *b) {
               if (a->timestamp == b->timestamp)
                 return a->id < b->id;
               return a->timestamp < b->timestamp;
             });

  std::vector<ChatMessage> messages;
  messages.reserve (ordered.size () + 1);
  for (const auto *mem : ordered)
    {
      const std::string role = RoleFromSourceId (mem->source_id);
      if (role.empty ())
        {
          continue;
        }

      const std::string text = ExtractTextFromBlobs (mem->content);
      if (text.empty ())
        {
          continue;
        }
      if (role == "user" && !latest_user_input.empty ()
          && text == latest_user_input)
        {
          continue;
        }
      messages.push_back ({ role, text });
    }

  if (!latest_user_input.empty ())
    {
      messages.push_back ({ "user", latest_user_input });
    }
  return messages;
}

std::vector<std::string>
BuildStreamingChunks (const std::string &text, std::size_t chunk_words)
{
  std::vector<std::string> chunks;
  if (text.empty ())
    {
      return chunks;
    }

  if (chunk_words == 0)
    {
      chunk_words = 1;
    }

  std::istringstream input (text);
  std::string word;
  std::string chunk;
  std::size_t words_in_chunk = 0;

  while (input >> word)
    {
      if (!chunk.empty ())
        {
          chunk.push_back (' ');
        }
      chunk += word;
      ++words_in_chunk;
      if (words_in_chunk >= chunk_words)
        {
          chunks.push_back (chunk);
          chunk.clear ();
          words_in_chunk = 0;
        }
    }

  if (!chunk.empty ())
    {
      chunks.push_back (chunk);
    }

  if (chunks.empty ())
    {
      chunks.push_back (text);
    }

  return chunks;
}

std::size_t
TrimmedProbeLength (const std::string &text)
{
  std::size_t end = text.size ();
  while (end > 0
         && std::isspace (static_cast<unsigned char> (text[end - 1]))
         && text[end - 1] != '\n')
    {
      --end;
    }
  return end;
}

bool
HasProbeBoundary (const std::string &text)
{
  if (text.empty ())
    {
      return false;
    }
  if (text.back () == '\n')
    {
      return true;
    }

  std::size_t idx = text.size ();
  while (idx > 0
         && std::isspace (static_cast<unsigned char> (text[idx - 1])))
    {
      --idx;
    }
  if (idx == 0)
    {
      return false;
    }

  switch (text[idx - 1])
    {
    case '.':
    case '!':
    case '?':
    case ':':
    case ';':
      return true;
    default:
      return false;
    }
}

bool
ShouldRunStreamingProbe (const std::string &text, bool force_flush = false)
{
  const std::size_t trimmed = TrimmedProbeLength (text);
  if (trimmed == 0)
    {
      return false;
    }
  if (force_flush)
    {
      return true;
    }
  if (trimmed >= kStreamingProbeMaxChars)
    {
      return true;
    }
  if (trimmed < kStreamingProbeMinChars)
    {
      return false;
    }
  return HasProbeBoundary (text);
}

int
ExtractTurnNumber (const std::string &text)
{
  const std::string marker = "turn ";
  const std::size_t marker_pos = text.find (marker);
  if (marker_pos == std::string::npos)
    {
      return -1;
    }
  std::size_t pos = marker_pos + marker.size ();
  std::size_t end = pos;
  while (end < text.size ()
         && std::isdigit (static_cast<unsigned char> (text[end])))
    {
      ++end;
    }
  if (end == pos)
    {
      return -1;
    }
  return std::stoi (text.substr (pos, end - pos));
}

long long
CountRowsForSource (Store &store, const std::string &table,
                    const std::string &source_id)
{
  const auto rows
      = store.Execute ("SELECT COUNT(*) AS c FROM " + table + " WHERE source_id = ?",
                       { source_id });
  if (rows.empty ())
    {
      return 0;
    }
  return cortext::testing::GetInt64 (rows[0], "c");
}

std::vector<std::string>
GetAssociationSummaryLabels (Store &store, long long min_memory_id = 0)
{
  const auto rows = store.Execute (
      "SELECT memory_id, label FROM memories "
      "WHERE kind = 'ASSOCIATION' AND label IS NOT NULL AND label != '' "
      "  AND memory_id > ? "
      "ORDER BY memory_id ASC",
      { min_memory_id });

  std::vector<std::string> labels;
  labels.reserve (rows.size ());
  for (const auto &row : rows)
    {
      auto it = row.find ("label");
      if (it == row.end () || it->second.type () != typeid (std::string))
        {
          continue;
        }
      labels.push_back (std::any_cast<std::string> (it->second));
    }
  return labels;
}

bool
ContainsCaseInsensitive (const std::string &haystack, const std::string &needle)
{
  return ContainsKeyword (ToLowerAscii (haystack), ToLowerAscii (needle));
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
      = static_cast<long long> (topics.size ()) * 2;
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
           >= static_cast<long long> (topics.size ()));

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

TEST_CASE ("Integration: scripted chat preserves turn-shaped working memory",
           "[integration][e2e][chat][scripted]")
{
  namespace fs = std::filesystem;

  if (!fs::exists (RepoModelsDir ()))
    {
      SKIP ("repo models directory not present");
    }

  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;

  std::unique_ptr<cortext::Cortext> cortext_ctx;
  try
    {
      cortext_ctx = cortext::Cortext::Create (cfg, temp_db.path, RepoModelsDir ());
    }
  catch (...)
    {
      SKIP ("Cortext::Create unavailable in this test environment");
    }
  REQUIRE (cortext_ctx != nullptr);

  auto unique_store = SQLiteStore::Create (temp_db.path);
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  constexpr int kTotalTurns = 100;
  REQUIRE (kTotalTurns % 2 == 0);

  std::uint64_t ts = 1'000'000;

  auto make_user_turn = [] (int turn_number) {
    const int topic = ((turn_number - 1) / 2) + 1;
    return "User turn " + std::to_string (turn_number)
           + ": My name is Gabe. I'm running a long chat-memory test on topic "
           + std::to_string (topic)
           + ". Please remember that this is still the same memory-system "
             "evaluation and keep the recent turn sequence straight.";
  };

  auto make_assistant_turn = [] (int turn_number) {
    const int topic = turn_number / 2;
    return "Assistant turn " + std::to_string (turn_number)
           + ": Understood. You are Gabe, we are still testing the memory "
             "system, and I should preserve the recent user and assistant "
             "turns as separate chat messages. For topic "
           + std::to_string (topic)
           + ", I will answer directly without collapsing the dialogue into "
             "one big role blob.";
  };

  std::vector<ChatMessage> transcript;
  transcript.reserve (kTotalTurns);
  cortext::Cortext::Context latest_ctx;
  std::size_t max_user_prompt_size = 0;

  for (int turn = 1; turn <= kTotalTurns; ++turn)
    {
      const bool is_user = (turn % 2) == 1;
      const std::string role = is_user ? "user" : "assistant";
      const std::string source_id = is_user ? "chat/user" : "chat/assistant";
      const std::string text
          = is_user ? make_user_turn (turn) : make_assistant_turn (turn);

      if (is_user)
        {
          try
            {
              latest_ctx = cortext_ctx->ProcessTextAt (text, source_id, ts++);
            }
          catch (...)
            {
              SKIP ("text processing backend unavailable in this test environment");
            }
          REQUIRE (latest_ctx.output.stored_embedding_id.has_value ());

          const auto prompt = BuildPromptMessages (latest_ctx.working_memory, text);
          REQUIRE_FALSE (prompt.empty ());
          REQUIRE (prompt.back ().role == "user");
          REQUIRE (prompt.back ().content == text);
          max_user_prompt_size = std::max (max_user_prompt_size, prompt.size ());
          if (turn >= 7)
            {
              std::vector<ChatMessage> expected_prompt;
              const std::size_t keep_prior = 3;
              const std::size_t prior_start
                  = transcript.size () > keep_prior
                        ? transcript.size () - keep_prior
                        : 0;
              for (std::size_t i = prior_start; i < transcript.size (); ++i)
                {
                  expected_prompt.push_back (transcript[i]);
                }
              expected_prompt.push_back ({ "user", text });
              std::ostringstream prompt_debug;
              prompt_debug << "turn=" << turn << "\nactual:\n";
              for (const auto &msg : prompt)
                {
                  prompt_debug << msg.role << " :: " << msg.content << "\n";
                }
              prompt_debug << "expected:\n";
              for (const auto &msg : expected_prompt)
                {
                  prompt_debug << msg.role << " :: " << msg.content << "\n";
                }
              INFO (prompt_debug.str ());
              REQUIRE (prompt.size () == expected_prompt.size ());
              for (std::size_t i = 0; i < prompt.size (); ++i)
                {
                  REQUIRE (prompt[i].role == expected_prompt[i].role);
                  REQUIRE (prompt[i].content == expected_prompt[i].content);
                }
            }
        }
      else
        {
          const long long assistant_signals_before_stream
              = CountRowsForSource (*store, "signals", "chat/assistant");
          const auto chunks = BuildStreamingChunks (text, 8);
          cortext::internal::StreamingTextProbeSession probe (*cortext_ctx,
                                                              source_id);
          std::string probe_buffer;
          int probe_calls = 0;
          for (const auto &chunk : chunks)
            {
              if (!probe_buffer.empty ())
                {
                  probe_buffer.push_back (' ');
                }
              probe_buffer += chunk;
              if (!ShouldRunStreamingProbe (probe_buffer))
                {
                  continue;
                }
              const auto probe_ctx = probe.AppendTextChunkAt (probe_buffer, ts++);
              REQUIRE_FALSE (probe_ctx.output.stored_embedding_id.has_value ());
              probe_buffer.clear ();
              ++probe_calls;
            }
          if (!probe_buffer.empty ())
            {
              const auto tail_ctx = probe.CacheTextChunkAt (probe_buffer, ts++);
              REQUIRE_FALSE (tail_ctx.output.stored_embedding_id.has_value ());
            }
          REQUIRE (probe_calls > 0);
          REQUIRE (CountRowsForSource (*store, "signals", "chat/assistant")
                   == assistant_signals_before_stream);

          latest_ctx = probe.FinalizeTextAt (text, ts++);
          REQUIRE (latest_ctx.output.stored_embedding_id.has_value ());
          REQUIRE (CountRowsForSource (*store, "signals", "chat/assistant")
                   > assistant_signals_before_stream);
        }

      transcript.push_back ({ role, text });
    }

  REQUIRE (transcript.size () == static_cast<std::size_t> (kTotalTurns));
  REQUIRE (max_user_prompt_size <= 4);

  const auto final_prompt = BuildPromptMessages (latest_ctx.working_memory, "");
  std::ostringstream final_prompt_debug;
  for (std::size_t i = 0; i < final_prompt.size (); ++i)
    {
      final_prompt_debug << "[" << i << "] " << final_prompt[i].role << " :: "
                         << final_prompt[i].content << "\n";
    }
  INFO ("final working-memory prompt:\n" << final_prompt_debug.str ());
  REQUIRE (final_prompt.size () == 4);
  const std::size_t tail_start
      = transcript.size () > 4 ? transcript.size () - 4 : 0;
  for (std::size_t i = 0; i < final_prompt.size (); ++i)
    {
      const auto &expected = transcript[tail_start + i];
      REQUIRE (final_prompt[i].role == expected.role);
      REQUIRE (final_prompt[i].content == expected.content);
    }

  const auto wm_rows = store->Execute (
      "SELECT memory_id, source_id, n_signals "
      "FROM memories "
      "WHERE kind = 'WORKING' AND end_ts IS NULL "
      "ORDER BY start_ts ASC, memory_id ASC",
      {});
  auto fresh_store = SQLiteStore::Create (temp_db.path);
  const auto fresh_wm_rows = fresh_store->Execute (
      "SELECT memory_id, source_id, n_signals, start_ts, end_ts "
      "FROM memories "
      "WHERE kind = 'WORKING' AND end_ts IS NULL "
      "ORDER BY start_ts ASC, memory_id ASC",
      {});
  std::ostringstream fresh_wm_debug;
  for (const auto &row : fresh_wm_rows)
    {
      const long long memory_id
          = cortext::testing::GetInt64 (row, "memory_id");
      const std::string source_id
          = std::any_cast<std::string> (row.at ("source_id"));
      const long long n_signals
          = cortext::testing::GetInt64 (row, "n_signals");
      const long long start_ts
          = cortext::testing::GetInt64 (row, "start_ts");
      const long long end_ts
          = cortext::testing::GetInt64 (row, "end_ts");
      const auto sig_rows = fresh_store->Execute (
          "SELECT COUNT(*) AS c FROM signals WHERE memory_id = ?",
          { memory_id });
      const long long signal_rows
          = sig_rows.empty () ? 0 : cortext::testing::GetInt64 (sig_rows[0], "c");
      fresh_wm_debug << memory_id << " | " << source_id
                     << " | n_signals=" << n_signals
                     << " | signal_rows=" << signal_rows
                     << " | start_ts=" << start_ts
                     << " | end_ts=" << end_ts << "\n";
    }
  INFO ("fresh working rows:\n" << fresh_wm_debug.str ());
  REQUIRE (wm_rows.size () == 4);
  for (const auto &row : wm_rows)
    {
      const std::string source_id
          = std::any_cast<std::string> (row.at ("source_id"));
      REQUIRE ((source_id == "chat/user" || source_id == "chat/assistant"));
      REQUIRE (cortext::testing::GetInt64 (row, "n_signals") == 1);
    }
}

TEST_CASE (
    "Integration: scripted chat consolidation preserves prompt shape and graph integrity",
    "[integration][e2e][chat][scripted][consolidation]")
{
  namespace fs = std::filesystem;

  if (!fs::exists (RepoModelsDir ()))
    {
      SKIP ("repo models directory not present");
    }

  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  cfg.stability = 0.0;

  std::unique_ptr<cortext::Cortext> cortext_ctx;
  try
    {
      cortext_ctx = cortext::Cortext::Create (cfg, temp_db.path, RepoModelsDir ());
    }
  catch (...)
    {
      SKIP ("Cortext::Create unavailable in this test environment");
    }
  REQUIRE (cortext_ctx != nullptr);

  auto unique_store = SQLiteStore::Create (temp_db.path);
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  constexpr int kTotalTurns = 100;
  constexpr int kConsolidateEveryAssistantTurns = 10;
  REQUIRE (kTotalTurns % 2 == 0);

  std::uint64_t ts = 2'000'000;
  int assistant_turns_seen = 0;
  int consolidation_runs = 0;

  auto make_user_turn = [] (int turn_number) {
    const int topic = ((turn_number - 1) / 2) + 1;
    return "User turn " + std::to_string (turn_number)
           + ": My name is Gabe. I am verifying that long-run chat history "
             "survives periodic consolidation on topic "
           + std::to_string (topic)
           + ". Keep the recent dialogue turns separate and in order.";
  };

  auto make_assistant_turn = [] (int turn_number) {
    const int topic = turn_number / 2;
    return "Assistant turn " + std::to_string (turn_number)
           + ": Acknowledged. We are testing long-run chat memory with "
             "periodic consolidation, and I should preserve each recent user "
             "and assistant turn as a distinct chat message. Topic "
           + std::to_string (topic)
           + " remains part of the same ongoing conversation.";
  };

  std::vector<ChatMessage> transcript;
  transcript.reserve (kTotalTurns);
  cortext::Cortext::Context latest_ctx;
  std::size_t max_user_prompt_size = 0;

  auto assert_prompt_tail = [&transcript] (const std::vector<ChatMessage> &prompt,
                                           const std::string &latest_user_input,
                                           int turn) {
    std::vector<ChatMessage> expected_prompt;
    if (latest_user_input.empty ())
      {
        const std::size_t tail_start
            = transcript.size () > 4 ? transcript.size () - 4 : 0;
        for (std::size_t i = tail_start; i < transcript.size (); ++i)
          {
            expected_prompt.push_back (transcript[i]);
          }
      }
    else
      {
        const std::size_t keep_prior = 3;
        const std::size_t prior_start
            = transcript.size () > keep_prior ? transcript.size () - keep_prior : 0;
        for (std::size_t i = prior_start; i < transcript.size (); ++i)
          {
            expected_prompt.push_back (transcript[i]);
          }
        expected_prompt.push_back ({ "user", latest_user_input });
      }

    std::ostringstream prompt_debug;
    prompt_debug << "turn=" << turn << "\nactual:\n";
    for (const auto &msg : prompt)
      {
        prompt_debug << msg.role << " :: " << msg.content << "\n";
      }
    prompt_debug << "expected:\n";
    for (const auto &msg : expected_prompt)
      {
        prompt_debug << msg.role << " :: " << msg.content << "\n";
      }
    INFO (prompt_debug.str ());
    REQUIRE (prompt.size () == expected_prompt.size ());
    for (std::size_t i = 0; i < prompt.size (); ++i)
      {
        REQUIRE (prompt[i].role == expected_prompt[i].role);
        REQUIRE (prompt[i].content == expected_prompt[i].content);
      }
  };

  for (int turn = 1; turn <= kTotalTurns; ++turn)
    {
      const bool is_user = (turn % 2) == 1;
      const std::string role = is_user ? "user" : "assistant";
      const std::string source_id = is_user ? "chat/user" : "chat/assistant";
      const std::string text
          = is_user ? make_user_turn (turn) : make_assistant_turn (turn);

      if (is_user)
        {
          try
            {
              latest_ctx = cortext_ctx->ProcessTextAt (text, source_id, ts++);
            }
          catch (...)
            {
              SKIP ("text processing backend unavailable in this test environment");
            }
          REQUIRE (latest_ctx.output.stored_embedding_id.has_value ());

          const auto prompt = BuildPromptMessages (latest_ctx.working_memory, text);
          REQUIRE_FALSE (prompt.empty ());
          REQUIRE (prompt.back ().role == "user");
          REQUIRE (prompt.back ().content == text);
          max_user_prompt_size = std::max (max_user_prompt_size, prompt.size ());
          if (turn >= 7)
            {
              assert_prompt_tail (prompt, text, turn);
            }
        }
      else
        {
          const long long assistant_signals_before_stream
              = CountRowsForSource (*store, "signals", "chat/assistant");
          const auto chunks = BuildStreamingChunks (text, 8);
          cortext::internal::StreamingTextProbeSession probe (*cortext_ctx,
                                                              source_id);
          std::string probe_buffer;
          int probe_calls = 0;
          for (const auto &chunk : chunks)
            {
              if (!probe_buffer.empty ())
                {
                  probe_buffer.push_back (' ');
                }
              probe_buffer += chunk;
              if (!ShouldRunStreamingProbe (probe_buffer))
                {
                  continue;
                }
              const auto probe_ctx = probe.AppendTextChunkAt (probe_buffer, ts++);
              REQUIRE_FALSE (probe_ctx.output.stored_embedding_id.has_value ());
              probe_buffer.clear ();
              ++probe_calls;
            }
          if (!probe_buffer.empty ())
            {
              const auto tail_ctx = probe.CacheTextChunkAt (probe_buffer, ts++);
              REQUIRE_FALSE (tail_ctx.output.stored_embedding_id.has_value ());
            }
          REQUIRE (probe_calls > 0);
          REQUIRE (CountRowsForSource (*store, "signals", "chat/assistant")
                   == assistant_signals_before_stream);

          latest_ctx = probe.FinalizeTextAt (text, ts++);
          REQUIRE (latest_ctx.output.stored_embedding_id.has_value ());
          REQUIRE (CountRowsForSource (*store, "signals", "chat/assistant")
                   > assistant_signals_before_stream);

          ++assistant_turns_seen;
        }

      transcript.push_back ({ role, text });

      if (!is_user
          && assistant_turns_seen % kConsolidateEveryAssistantTurns == 0)
        {
          try
            {
              latest_ctx = cortext_ctx->Consolidate (
                  cortext::ConsolidationMode::Shallow);
            }
          catch (...)
            {
              SKIP ("consolidation backend unavailable in this test environment");
            }
          ++consolidation_runs;

          const auto prompt_after_consolidation
              = BuildPromptMessages (latest_ctx.working_memory, "");
          assert_prompt_tail (prompt_after_consolidation, "",
                              turn);
        }
    }

  INFO ("consolidation_runs=" << consolidation_runs);
  REQUIRE (consolidation_runs >= 1);
  REQUIRE (transcript.size () == static_cast<std::size_t> (kTotalTurns));
  REQUIRE (max_user_prompt_size <= 4);

  const auto final_prompt = BuildPromptMessages (latest_ctx.working_memory, "");
  assert_prompt_tail (final_prompt, "", kTotalTurns);

  const auto active_non_chat_wm_rows = store->Execute (
      "SELECT COUNT(*) AS c "
      "FROM memories "
      "WHERE kind = 'WORKING' AND end_ts IS NULL "
      "  AND source_id NOT IN ('chat/user', 'chat/assistant')",
      {});
  REQUIRE_FALSE (active_non_chat_wm_rows.empty ());
  REQUIRE (cortext::testing::GetInt64 (active_non_chat_wm_rows[0], "c") == 0);

  const auto orphan_edges = store->Execute (
      "SELECT COUNT(*) AS c "
      "FROM associations a "
      "LEFT JOIN memories src ON src.memory_id = a.source_memory_id "
      "LEFT JOIN memories dst ON dst.memory_id = a.target_memory_id "
      "WHERE src.memory_id IS NULL OR dst.memory_id IS NULL",
      {});
  const auto orphan_edge_types = store->Execute (
      "SELECT a.edge_type, COUNT(*) AS c "
      "FROM associations a "
      "LEFT JOIN memories src ON src.memory_id = a.source_memory_id "
      "LEFT JOIN memories dst ON dst.memory_id = a.target_memory_id "
      "WHERE src.memory_id IS NULL OR dst.memory_id IS NULL "
      "GROUP BY a.edge_type ORDER BY c DESC",
      {});
  const auto orphan_edge_samples = store->Execute (
      "SELECT a.edge_type, a.source_memory_id, a.target_memory_id, "
      "       COALESCE(src.kind, '<missing>') AS src_kind, "
      "       COALESCE(dst.kind, '<missing>') AS dst_kind "
      "FROM associations a "
      "LEFT JOIN memories src ON src.memory_id = a.source_memory_id "
      "LEFT JOIN memories dst ON dst.memory_id = a.target_memory_id "
      "WHERE src.memory_id IS NULL OR dst.memory_id IS NULL "
      "LIMIT 20",
      {});
  std::ostringstream orphan_debug;
  orphan_debug << "temp_db=" << temp_db.path << "\nby_type:\n";
  for (const auto &row : orphan_edge_types)
    {
      orphan_debug << std::any_cast<std::string> (row.at ("edge_type"))
                   << " -> " << cortext::testing::GetInt64 (row, "c") << "\n";
    }
  orphan_debug << "samples:\n";
  for (const auto &row : orphan_edge_samples)
    {
      orphan_debug << std::any_cast<std::string> (row.at ("edge_type")) << " | "
                   << cortext::testing::GetInt64 (row, "source_memory_id") << " -> "
                   << cortext::testing::GetInt64 (row, "target_memory_id") << " | "
                   << std::any_cast<std::string> (row.at ("src_kind")) << " -> "
                   << std::any_cast<std::string> (row.at ("dst_kind")) << "\n";
    }
  INFO (orphan_debug.str ());
  REQUIRE_FALSE (orphan_edges.empty ());
  REQUIRE (cortext::testing::GetInt64 (orphan_edges[0], "c") == 0);

  const auto derived_edges = store->Execute (
      "SELECT COUNT(*) AS c "
      "FROM associations "
      "WHERE edge_type = 'derived_from'",
      {});
  REQUIRE_FALSE (derived_edges.empty ());
  REQUIRE (cortext::testing::GetInt64 (derived_edges[0], "c") > 0);

  const auto invalid_derived_edges = store->Execute (
      "SELECT COUNT(*) AS c "
      "FROM associations a "
      "JOIN memories src ON src.memory_id = a.source_memory_id "
      "JOIN memories dst ON dst.memory_id = a.target_memory_id "
      "WHERE a.edge_type = 'derived_from' "
      "  AND (src.kind != 'ASSOCIATION' "
      "       OR dst.kind NOT IN ('LONG_TERM', 'ASSOCIATION'))",
      {});
  REQUIRE_FALSE (invalid_derived_edges.empty ());
  REQUIRE (cortext::testing::GetInt64 (invalid_derived_edges[0], "c") == 0);

  const auto invalid_label_edges = store->Execute (
      "SELECT COUNT(*) AS c "
      "FROM associations a "
      "JOIN memories src ON src.memory_id = a.source_memory_id "
      "JOIN memories dst ON dst.memory_id = a.target_memory_id "
      "WHERE a.edge_type = 'has_label' "
      "  AND (src.kind != 'ASSOCIATION' OR dst.kind != 'LABEL')",
      {});
  REQUIRE_FALSE (invalid_label_edges.empty ());
  REQUIRE (cortext::testing::GetInt64 (invalid_label_edges[0], "c") == 0);
}

TEST_CASE (
    "Integration: deep consolidation generates fact-oriented summaries with the real model",
    "[integration][e2e][chat][consolidation][deep][gemma]")
{
  namespace fs = std::filesystem;

  if (!fs::exists (RepoModelsDir ()))
    {
      SKIP ("repo models directory not present");
    }

  ScopedTempDb temp_db;
  cortext::Cortext::Config cfg;
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.0;

  std::unique_ptr<cortext::Cortext> cortext_ctx;
  try
    {
      cortext_ctx = cortext::Cortext::Create (cfg, temp_db.path, RepoModelsDir ());
    }
  catch (...)
    {
      SKIP ("Cortext::Create unavailable in this test environment");
    }
  REQUIRE (cortext_ctx != nullptr);

  auto unique_store = SQLiteStore::Create (temp_db.path);
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  std::uint64_t ts = 3'000'000;

  auto process_turn = [&] (const std::string &source_id, const std::string &text) {
    if (source_id == "chat/user")
      {
        return cortext_ctx->ProcessTextAt (text, source_id, ts++);
      }

    const auto chunks = BuildStreamingChunks (text, 8);
    cortext::internal::StreamingTextProbeSession probe (*cortext_ctx, source_id);
    std::string probe_buffer;
    for (const auto &chunk : chunks)
      {
        if (!probe_buffer.empty ())
          {
            probe_buffer.push_back (' ');
          }
        probe_buffer += chunk;
        if (!ShouldRunStreamingProbe (probe_buffer))
          {
            continue;
          }
        const auto probe_ctx = probe.AppendTextChunkAt (probe_buffer, ts++);
        REQUIRE_FALSE (probe_ctx.output.stored_embedding_id.has_value ());
        probe_buffer.clear ();
      }
    if (!probe_buffer.empty ())
      {
        const auto tail_ctx = probe.CacheTextChunkAt (probe_buffer, ts++);
        REQUIRE_FALSE (tail_ctx.output.stored_embedding_id.has_value ());
      }
    return probe.FinalizeTextAt (text, ts++);
  };

  const std::vector<ChatMessage> emily_block = {
    { "user", "Emily lost her first tooth today at school." },
    { "assistant", "Emily lost her first tooth today, and you want to remember it." },
    { "user", "We celebrated with vanilla ice cream after school." },
    { "assistant", "You celebrated Emily losing her first tooth with vanilla ice cream." },
    { "user", "Tonight Emily is putting the tooth under her pillow for the tooth fairy." },
    { "assistant", "Emily is putting the tooth under her pillow tonight for the tooth fairy." },
  };

  const std::vector<ChatMessage> cortext_block = {
    { "user", "I am building Cortext, a C plus plus memory system for AI assistants." },
    { "assistant", "Cortext is your C plus plus memory system for AI assistants." },
    { "user", "I moved text embeddings to llama dot cpp to speed up the chat app." },
    { "assistant", "You moved text embeddings to llama dot cpp to make the Cortext chat app faster." },
    { "user", "I am profiling the chat application and focusing on consolidation quality next." },
    { "assistant", "You are profiling the Cortext chat application and now focusing on consolidation quality." },
  };

  for (const auto &msg : emily_block)
    {
      const std::string source_id
          = (msg.role == "user") ? "chat/user" : "chat/assistant";
      const auto ctx = process_turn (source_id, msg.content);
      REQUIRE (ctx.output.stored_embedding_id.has_value ());
    }

  cortext::Cortext::Context latest_ctx;
  try
    {
      latest_ctx = cortext_ctx->Consolidate (cortext::ConsolidationMode::Deep);
    }
  catch (...)
    {
      SKIP ("deep consolidation backend unavailable in this test environment");
    }

  const auto first_pass_summaries = GetAssociationSummaryLabels (*store);
  REQUIRE_FALSE (first_pass_summaries.empty ());
  bool first_pass_mentions_emily = false;
  std::cout << "\n[first deep consolidation summaries]\n";
  for (const auto &summary : first_pass_summaries)
    {
      std::cout << "- " << summary << "\n";
      first_pass_mentions_emily
          = first_pass_mentions_emily || ContainsCaseInsensitive (summary, "Emily");
      REQUIRE_FALSE (ContainsCaseInsensitive (summary, "in a conversation"));
      REQUIRE_FALSE (ContainsCaseInsensitive (summary, "this occurred after"));
      REQUIRE_FALSE (ContainsCaseInsensitive (summary, "the user said"));
      REQUIRE_FALSE (ContainsCaseInsensitive (summary, "the assistant asked"));
      REQUIRE_FALSE (ContainsCaseInsensitive (summary, "The user"));
      REQUIRE_FALSE (ContainsCaseInsensitive (summary, "The assistant"));
    }
  REQUIRE (first_pass_mentions_emily);

  long long last_summary_memory_id = 0;
  const auto first_summary_rows = store->Execute (
      "SELECT MAX(memory_id) AS max_id FROM memories WHERE kind = 'ASSOCIATION'",
      {});
  REQUIRE_FALSE (first_summary_rows.empty ());
  last_summary_memory_id
      = cortext::testing::GetInt64 (first_summary_rows[0], "max_id");

  for (const auto &msg : cortext_block)
    {
      const std::string source_id
          = (msg.role == "user") ? "chat/user" : "chat/assistant";
      const auto ctx = process_turn (source_id, msg.content);
      REQUIRE (ctx.output.stored_embedding_id.has_value ());
    }

  latest_ctx = cortext_ctx->Consolidate (cortext::ConsolidationMode::Deep);
  const auto second_pass_summaries
      = GetAssociationSummaryLabels (*store, last_summary_memory_id);
  REQUIRE_FALSE (second_pass_summaries.empty ());

  bool second_pass_mentions_cortext = false;
  bool second_pass_mentions_llama = false;
  std::cout << "\n[second deep consolidation summaries]\n";
  for (const auto &summary : second_pass_summaries)
    {
      std::cout << "- " << summary << "\n";
      second_pass_mentions_cortext
          = second_pass_mentions_cortext
            || ContainsCaseInsensitive (summary, "Cortext");
      second_pass_mentions_llama
          = second_pass_mentions_llama
            || ContainsCaseInsensitive (summary, "llama");
      REQUIRE_FALSE (ContainsCaseInsensitive (summary, "in a conversation"));
      REQUIRE_FALSE (ContainsCaseInsensitive (summary, "this occurred after"));
      REQUIRE_FALSE (ContainsCaseInsensitive (summary, "the user said"));
      REQUIRE_FALSE (ContainsCaseInsensitive (summary, "the assistant asked"));
      REQUIRE_FALSE (ContainsCaseInsensitive (summary, "The user"));
      REQUIRE_FALSE (ContainsCaseInsensitive (summary, "The assistant"));
    }
  const bool second_pass_mentions_target
      = second_pass_mentions_cortext || second_pass_mentions_llama;
  REQUIRE (second_pass_mentions_target);

  const auto second_pass_invalid_sources = store->Execute (
      "SELECT COUNT(*) AS c "
      "FROM associations a "
      "JOIN memories src ON src.memory_id = a.source_memory_id "
      "JOIN memories dst ON dst.memory_id = a.target_memory_id "
      "WHERE a.edge_type = 'derived_from' "
      "  AND src.kind = 'ASSOCIATION' "
      "  AND src.memory_id > ? "
      "  AND dst.kind != 'LONG_TERM'",
      { last_summary_memory_id });
  REQUIRE_FALSE (second_pass_invalid_sources.empty ());
  REQUIRE (cortext::testing::GetInt64 (second_pass_invalid_sources[0], "c")
           == 0);

  const auto second_summary_rows = store->Execute (
      "SELECT MAX(memory_id) AS max_id FROM memories WHERE kind = 'ASSOCIATION'",
      {});
  REQUIRE_FALSE (second_summary_rows.empty ());
  const long long second_summary_max_id
      = cortext::testing::GetInt64 (second_summary_rows[0], "max_id");

  latest_ctx = cortext_ctx->Consolidate (cortext::ConsolidationMode::Deep);
  const auto third_pass_summaries
      = GetAssociationSummaryLabels (*store, second_summary_max_id);
  REQUIRE (third_pass_summaries.empty ());
}
