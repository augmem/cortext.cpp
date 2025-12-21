#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cortext/core/algorithms.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/core/utils.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/accumulator.hpp>
#include <cortext/operations/accumulator_reset.hpp>
#include <cortext/operations/blend.hpp>
#include <cortext/operations/boundary.hpp>
#include <cortext/operations/centroids.hpp>
#include <cortext/operations/coherence.hpp>
#include <cortext/operations/effective_focus.hpp>
#include <cortext/operations/embedding_prediction_error.hpp>
#include <cortext/operations/focus.hpp>
#include <cortext/operations/focus_spread.hpp>
#include <cortext/operations/metrics.hpp>
#include <cortext/operations/precision.hpp>
#include <cortext/operations/recent_context.hpp>
#include <cortext/operations/spike_bypass.hpp>
#include <cortext/operations/sensitivity.hpp>
#include <cortext/operations/stability.hpp>
#include <cortext/operations/threshold.hpp>
#include <cortext/operations/uncertainty.hpp>
#include <cortext/operations/working_memory.hpp>
#include <cortext/operations/write_gate.hpp>
#include <cortext/operations/constants.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/signal.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;

namespace
{
constexpr int kEmbeddingDim = 256;

struct TopicSpec
{
  std::string keyword;
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
        if (lower.find (topic.keyword) != std::string::npos)
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
    const float norm = v.norm ();
    if (norm > 1e-9f)
      {
        v /= norm;
      }
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

double
ComputeMeanCosWindow (const std::deque<Eigen::VectorXf> &ctx,
                      const Eigen::VectorXf &emb, int window)
{
  if (ctx.empty () || emb.size () == 0 || window <= 0)
    {
      return 1.0;
    }
  const int n_total = static_cast<int> (ctx.size ());
  const int start = std::max (0, n_total - window);
  double sum = 0.0;
  int count = 0;
  for (int i = start; i < n_total; ++i)
    {
      const auto &e = ctx[static_cast<size_t> (i)];
      if (e.size () != emb.size ())
        {
          continue;
        }
      sum += core::CosineSimilarity (e, emb);
      ++count;
    }
  if (count <= 0)
    {
      return 1.0;
    }
  return sum / static_cast<double> (count);
}

double
RelevanceToTask (const Eigen::VectorXf &q,
                 const std::deque<Eigen::VectorXf> &task_ctx)
{
  if (q.size () == 0 || task_ctx.empty ())
    {
      return 0.5;
    }
  Eigen::VectorXf mean = Eigen::VectorXf::Zero (q.size ());
  int count = 0;
  for (const auto &e : task_ctx)
    {
      if (e.size () != q.size ())
        {
          continue;
        }
      mean += e;
      ++count;
    }
  if (count <= 0)
    {
      return 0.5;
    }
  mean /= static_cast<float> (count);
  const double cos = core::CosineSimilarity (q, mean);
  return core::Clamp ((cos + 1.0) * 0.5, 0.0, 1.0);
}

class LogWorkingMemoryOp final : public IOperation
{
public:
  void
  Execute (OperationContext &context, Transaction & /*tx*/) const override
  {
    if (!context.GetAccumulatorWriteDecision ())
      {
        return;
      }

    auto &p_ctx = context.GetProcessorContext ();
    const auto &cfg = context.GetConfig ();
    const auto &signal = context.GetSignal ();

    const auto &rep_opt = context.GetRepresentativeEmbedding ();
    if (!rep_opt.has_value () || rep_opt->size () == 0)
      {
        return;
      }

    const Eigen::VectorXf &e_rep = *rep_opt;
    const int ctx_window = static_cast<int> (core::NCtx (cfg.stability));
    const double mean_cos_window
        = ComputeMeanCosWindow (p_ctx.recent_context_embeddings, e_rep,
                                ctx_window);
    const double manifold_complexity
        = core::Clamp ((1.0 - mean_cos_window) * 0.5, 0.0, 1.0);
    const double complexity_penalty
        = manifold_complexity * core::WMComplexityScale (cfg.sensitivity);

    const auto acc_it = p_ctx.accumulator_states.find (signal.source_id);
    if (acc_it == p_ctx.accumulator_states.end ())
      {
        return;
      }
    const auto &acc = acc_it->second;
    const Eigen::VectorXf &task_query
        = (acc.mu_acc.size () > 0) ? acc.mu_acc : e_rep;
    const double relevance = RelevanceToTask (
        task_query, p_ctx.recent_context_embeddings);

    double max_cos = -1.0;
    for (const auto &slot : p_ctx.wm_slots)
      {
        if (slot.embedding.size () != e_rep.size () || slot.embedding.size () == 0)
          continue;
        const double c = core::CosineSimilarity (slot.embedding, e_rep);
        max_cos = std::max (max_cos, c);
      }
    const double novelty_to_set
        = (max_cos <= -1.0)
              ? 1.0
              : core::Clamp ((1.0 - max_cos) * 0.5, 0.0, 1.0);

    const double w_alpha = core::Lerp (0.55, 0.70, cfg.focus);
    const double w_beta = core::Lerp (0.20, 0.35, cfg.focus);
    const double w_gamma = core::Lerp (0.10, 0.30, cfg.sensitivity);
    const double w_sum = std::max (1e-6, w_alpha + w_beta + w_gamma);
    const double alpha = w_alpha / w_sum;
    const double beta = w_beta / w_sum;
    const double gamma = w_gamma / w_sum;

    const double S_window = context.GetWindowScore ().value_or (0.0);
    const double benefit = core::Clamp01 (
        alpha * core::Clamp01 (S_window) + beta * relevance + gamma * novelty_to_set);

    const double gate_threshold = core::WMGateThreshold (cfg.focus);
    const double margin = benefit - gate_threshold;
    const int k = static_cast<int> (p_ctx.wm_slots.size ());
    const int base_capacity
        = std::max (1, core::WMBaseCapacity (cfg.sensitivity, cfg.focus));
    const double over_ratio
        = (k > base_capacity)
              ? (static_cast<double> (k - base_capacity)
                 / static_cast<double> (base_capacity))
              : 0.0;
    const double capacity_pressure = 1.0 + std::pow (over_ratio, 3.0);
    const double cost_per_slot
        = core::WMMaintenanceCostPerSlot (cfg.sensitivity);
    const double raw_cost
        = (cost_per_slot * static_cast<double> (k) + complexity_penalty)
          * capacity_pressure;
    const double cost_total = raw_cost / (1.0 + raw_cost);

    std::cout << std::fixed << std::setprecision (3);
    std::cout << "WM_LOG ts=" << signal.timestamp
              << " source=" << signal.source_id
              << " write=" << (context.GetAccumulatorWriteDecision () ? 1 : 0)
              << " accepted=" << (p_ctx.wm_last_accepted ? 1 : 0)
              << " chunked=" << (p_ctx.wm_last_chunked ? 1 : 0)
              << " slots=" << p_ctx.wm_slots.size ()
              << " S_window=" << S_window
              << " relevance=" << relevance
              << " novelty=" << novelty_to_set
              << " benefit=" << benefit
              << " gate=" << gate_threshold
              << " margin=" << margin
              << " cost=" << cost_total
              << "\n";

    for (size_t i = 0; i < p_ctx.wm_slots.size (); ++i)
      {
        const auto &slot = p_ctx.wm_slots[i];
        std::cout << "  slot[" << i << "] strength=" << slot.strength
                  << " last_ts=" << slot.last_ts
                  << " n_signals=" << slot.n_signals
                  << " s_max=" << slot.s_max
                  << " s_avg=" << slot.s_avg
                  << " source=" << slot.source_id
                  << "\n";
      }
  }
};

class PreWorkingMemoryLogOp final : public IOperation
{
public:
  void
  Execute (OperationContext &context, Transaction & /*tx*/) const override
  {
    if (!context.GetAccumulatorWriteDecision ())
      {
        return;
      }

    auto &p_ctx = context.GetProcessorContext ();
    const auto &cfg = context.GetConfig ();
    const auto &signal = context.GetSignal ();

    const auto &rep_opt = context.GetRepresentativeEmbedding ();
    if (!rep_opt.has_value () || rep_opt->size () == 0)
      {
        return;
      }
    const Eigen::VectorXf &e_rep = *rep_opt;

    const int ctx_window = static_cast<int> (core::NCtx (cfg.stability));
    const double mean_cos_window
        = ComputeMeanCosWindow (p_ctx.recent_context_embeddings, e_rep,
                                ctx_window);
    const double manifold_complexity
        = core::Clamp ((1.0 - mean_cos_window) * 0.5, 0.0, 1.0);
    const double complexity_penalty
        = manifold_complexity * core::WMComplexityScale (cfg.sensitivity);

    const auto acc_it = p_ctx.accumulator_states.find (signal.source_id);
    if (acc_it == p_ctx.accumulator_states.end ())
      {
        return;
      }
    const auto &acc = acc_it->second;
    const Eigen::VectorXf &task_query
        = (acc.mu_acc.size () > 0) ? acc.mu_acc : e_rep;
    const double relevance = RelevanceToTask (
        task_query, p_ctx.recent_context_embeddings);

    double max_cos = -1.0;
    for (const auto &slot : p_ctx.wm_slots)
      {
        if (slot.embedding.size () != e_rep.size () || slot.embedding.size () == 0)
          continue;
        const double c = core::CosineSimilarity (slot.embedding, e_rep);
        max_cos = std::max (max_cos, c);
      }
    const double novelty_to_set
        = (max_cos <= -1.0)
              ? 1.0
              : core::Clamp ((1.0 - max_cos) * 0.5, 0.0, 1.0);

    const double w_alpha = core::Lerp (0.55, 0.70, cfg.focus);
    const double w_beta = core::Lerp (0.20, 0.35, cfg.focus);
    const double w_gamma = core::Lerp (0.10, 0.30, cfg.sensitivity);
    const double w_sum = std::max (1e-6, w_alpha + w_beta + w_gamma);
    const double alpha = w_alpha / w_sum;
    const double beta = w_beta / w_sum;
    const double gamma = w_gamma / w_sum;

    const double S_window = context.GetWindowScore ().value_or (0.0);
    const double benefit = core::Clamp01 (
        alpha * core::Clamp01 (S_window) + beta * relevance + gamma * novelty_to_set);

    const double gate_threshold = core::WMGateThreshold (cfg.focus);
    const double margin = benefit - gate_threshold;
    const int k = static_cast<int> (p_ctx.wm_slots.size ());
    const int base_capacity
        = std::max (1, core::WMBaseCapacity (cfg.sensitivity, cfg.focus));
    const double over_ratio
        = (k > base_capacity)
              ? (static_cast<double> (k - base_capacity)
                 / static_cast<double> (base_capacity))
              : 0.0;
    const double capacity_pressure = 1.0 + std::pow (over_ratio, 3.0);
    const double cost_per_slot
        = core::WMMaintenanceCostPerSlot (cfg.sensitivity);
    const double raw_cost
        = (cost_per_slot * static_cast<double> (k) + complexity_penalty)
          * capacity_pressure;
    const double cost_total = raw_cost / (1.0 + raw_cost);

    int best_idx = -1;
    double best_sim = -1.0;
    for (int i = 0; i < static_cast<int> (p_ctx.wm_slots.size ()); ++i)
      {
        const auto &slot = p_ctx.wm_slots[static_cast<size_t> (i)];
        if (!slot.source_id.empty () && slot.source_id != signal.source_id)
          {
            continue;
          }
        if (slot.embedding.size () != e_rep.size ())
          {
            continue;
          }
        const double sim = core::Clamp (
            core::CosineSimilarity (slot.embedding, e_rep),
            operations::constants::kNormalizedMin,
            operations::constants::kNormalizedMax);
        if (sim > best_sim)
          {
            best_sim = sim;
            best_idx = i;
          }
      }

    const double chunk_threshold = core::WMChunkingThreshold (cfg.focus);
    const double rehearsal_threshold = core::WMRehearsalThreshold (cfg.focus);

    std::cout << std::fixed << std::setprecision (3);
    std::cout << "WM_PRE ts=" << signal.timestamp
              << " source=" << signal.source_id
              << " boundary=" << context.GetBoundaryScore ().value_or (0.0)
              << " flush=" << (context.GetFlushRequired () ? 1 : 0)
              << " spike_bypass=" << (context.GetSpikeBypass () ? 1 : 0)
              << " write=" << (context.GetAccumulatorWriteDecision () ? 1 : 0)
              << " slots=" << p_ctx.wm_slots.size ()
              << " S_window=" << S_window
              << " relevance=" << relevance
              << " novelty=" << novelty_to_set
              << " benefit=" << benefit
              << " gate=" << gate_threshold
              << " margin=" << margin
              << " cost=" << cost_total
              << " best_sim=" << best_sim
              << " best_idx=" << best_idx
              << " chunk_thr=" << chunk_threshold
              << " rehearse_thr=" << rehearsal_threshold
              << " base_cap=" << base_capacity
              << " cap_pressure=" << capacity_pressure
              << "\n";
  }
};

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

std::unique_ptr<IOperation>
BuildWorkingMemoryPipeline ()
{
  using cortext::OperationSet;
  using cortext::operations::CheckSpikeBypass;
  using cortext::operations::ComputeCoherence;
  using cortext::operations::ComputeCompositeScore;
  using cortext::operations::ComputeEffectiveFocus;
  using cortext::operations::ComputeFocusSpread;
  using cortext::operations::ComputeMetrics;
  using cortext::operations::DetectBoundary;
  using cortext::operations::InitializeEmbeddedCentroids;
  using cortext::operations::InitializeFocusPriors;
  using cortext::operations::InitializeSensitivityPriors;
  using cortext::operations::InitializeStabilityPriors;
  using cortext::operations::ResetAccumulatorAfterFlush;
  using cortext::operations::UpdateAccumulator;
  using cortext::operations::UpdateEmbeddingPredictionError;
  using cortext::operations::UpdateFocus;
  using cortext::operations::UpdateMood;
  using cortext::operations::UpdatePrecisionDelta;
  using cortext::operations::UpdateRecentContext;
  using cortext::operations::UpdateSensitivity;
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
      std::make_unique<UpdateEmbeddingPredictionError> (),
      std::make_unique<UpdateUncertainty> (),
      std::make_unique<ComputeFocusSpread> (),
      std::make_unique<UpdateFocus> (),
      std::make_unique<UpdateSensitivity> (),
      std::make_unique<UpdateMood> (),
      std::make_unique<ComputeEffectiveFocus> (),
      std::make_unique<ComputeMetrics> (),
      std::make_unique<ComputeCompositeScore> (),
      std::make_unique<UpdatePrecisionDelta> (),
      std::make_unique<UpdateThreshold> (),
      std::make_unique<UpdateAccumulator> (),
      std::make_unique<DetectBoundary> (),
      std::make_unique<CheckSpikeBypass> (),
      std::make_unique<ComputeWriteGate> (),
      std::make_unique<PreWorkingMemoryLogOp> (),
      std::make_unique<WorkingMemory> (),
      std::make_unique<LogWorkingMemoryOp> (),
      std::make_unique<ResetAccumulatorAfterFlush> ());
}

} // namespace

TEST_CASE ("Manual: working memory logs", "[manual][working_memory]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<TopicSpec> topics = {
    { "paris", 0 },
    { "budget", 1 },
  };

  KeywordEncoder encoder (topics);
  SignalProcessor::Config cfg;
  cfg.focus = 0.3;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.encoder = &encoder;

  auto pipeline = BuildWorkingMemoryPipeline ();
  SignalProcessor processor (cfg, store, std::move (pipeline));

  const std::string source_id = "chat";
  uint64_t ts = 100000;
  const uint64_t gap_ms = 45000;

  processor.Process (MakeTextSignal (encoder, "User: paris plan", ts, source_id));
  ts += 1000;
  processor.Process (MakeTextSignal (encoder, "Assistant: paris details", ts, source_id));
  ts += gap_ms;
  processor.Process (MakeTextSignal (encoder, "User: paris wrap", ts, source_id));
  ts += 1000;

  processor.Process (MakeTextSignal (encoder, "User: paris follow", ts, source_id));
  ts += 1000;
  processor.Process (MakeTextSignal (encoder, "Assistant: paris more", ts, source_id));
  ts += gap_ms;
  processor.Process (MakeTextSignal (encoder, "User: paris close", ts, source_id));
  ts += 1000;

  processor.Process (MakeTextSignal (encoder, "User: budget start", ts, source_id));
  ts += 1000;
  processor.Process (MakeTextSignal (encoder, "Assistant: budget tips", ts, source_id));
  ts += gap_ms;
  processor.Process (MakeTextSignal (encoder, "User: budget close", ts, source_id));

  SUCCEED ("Manual WM log emitted");
}
