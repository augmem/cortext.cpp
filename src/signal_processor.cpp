#include "cortext/core/knobs.hpp"
#include "cortext/processor.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include <any>
#include <map>
#include <vector>

namespace cortext
{

SignalProcessor::SignalProcessor (const Config &config,
                                  std::shared_ptr<Store> store,
                                  std::unique_ptr<IOperation> root_operation)
    : config_ (config), store_ (std::move (store)),
      root_operation_ (std::move (root_operation)),
      context_ (std::make_unique<ProcessorContext> ())
{
  // Initialize rate observation window capacity derived from Stability knob.
  if (context_)
    {
      const double T = core::Clamp (config_.stability, 0.0, 1.0);
      const int cap
          = static_cast<int> (std::round (core::Lerp (10.0, 60.0, T)));
      context_->write_rate_window_.SetCapacity (
          static_cast<size_t> (std::max (1, cap)));
    }
  StartNewEpisode ();
}

SignalProcessor::~SignalProcessor () = default;

SignalProcessor::Output
SignalProcessor::Process (const Signal &signal)
{
  OperationContext op_context (signal, *context_, config_, write_buffer_,
                               store_.get ());

  // Record timestamp for observed write-rate window before ops execute.
  context_->write_rate_window_.Record (signal.timestamp);

  // Compute optional observed retention (avg age of active memories)
  // active = strength >= periphery_cutoff(T) AND last_used > 0
  if (store_)
    {
      const double cutoff = core::PeripheryCutoff (config_.stability);
      const uint64_t now_ts = signal.timestamp;
      try
        {
          const std::vector<std::map<std::string, std::any> > rows
              = store_->Execute ("SELECT last_used FROM memory_feedback "
                                 "WHERE strength >= ? AND last_used > 0",
                                 { cutoff });
          if (!rows.empty ())
            {
              double sum_age = 0.0;
              int count = 0;
              for (const auto &row : rows)
                {
                  auto it = row.find ("last_used");
                  if (it == row.end ())
                    continue;
                  const std::any &v = it->second;
                  uint64_t last_used_ts = 0;
                  if (v.type () == typeid (long long))
                    {
                      last_used_ts = static_cast<uint64_t> (
                          std::any_cast<long long> (v));
                    }
                  else if (v.type () == typeid (int64_t))
                    {
                      last_used_ts
                          = static_cast<uint64_t> (std::any_cast<int64_t> (v));
                    }
                  else if (v.type () == typeid (int))
                    {
                      last_used_ts
                          = static_cast<uint64_t> (std::any_cast<int> (v));
                    }
                  else
                    {
                      continue;
                    }
                  if (now_ts > last_used_ts)
                    {
                      sum_age += static_cast<double> (now_ts - last_used_ts);
                      count += 1;
                    }
                }
              if (count > 0)
                {
                  op_context.SetObservedRetentionSeconds (
                      sum_age / static_cast<double> (count));
                }
            }
        }
      catch (...)
        {
          // If the store does not have the table yet or other error, skip.
        }
    }

  root_operation_->Execute (op_context);

  // Honor Algorithm 12 boundary request from operations.
  if (op_context.ShouldFinalizeEpisode ())
    {
      FinalizeEpisode ();
      StartNewEpisode ();
    }

  // Increment maturity counters post-processing (current signal observed).
  context_->signals_processed += 1;

  // Assemble Output from OperationContext (no additional computation).
  Output out;

  // Memory candidates and usage
  {
    const auto &cands = op_context.GetRetrievedMemoryEmbeddings ();
    out.candidate_memory_ids.reserve (cands.size ());
    for (const auto &kv : cands)
      {
        out.candidate_memory_ids.push_back (kv.first);
      }
    for (const auto &e : op_context.GetMemoryUsageEvents ())
      {
        if (e.used)
          {
            out.used_memory_ids.push_back (
                static_cast<long long> (e.embedding_id));
          }
      }
  }

  // Gate decision and boundary
  out.interrupt_allowed = op_context.GetInterruptAllowed ();
  out.at_boundary = op_context.GetAtBoundary ();

  // Thresholds and stabilizers
  out.threshold_T_dynamic = op_context.GetThresholdTDynamic ();
  out.threshold_hysteresis = op_context.GetThresholdHysteresis ();
  out.effective_focus = op_context.GetEffectiveFocus ();

  // Emotion projections
  out.emotion_intensity = op_context.GetEmotionIntensity ();
  out.valence = op_context.GetValence ();
  out.arousal = op_context.GetArousal ();

  // MNI diagnostics
  out.mni_jaccard = op_context.GetMniJaccard ();
  out.mni_best_mu = op_context.GetMniBestMu ();
  out.mni_dup_thresh = op_context.GetMniDupThresh ();
  out.mni_tau_jaccard_eff = op_context.GetMniTauJaccardEff ();
  out.mni_tau_mu_eff = op_context.GetMniTauMuEff ();

  // Optional composites
  out.composite_score = op_context.GetCompositeScore ();
  out.serial_position_multiplier = op_context.GetSerialPositionMultiplier ();

  // Metrics (Algorithm 7)
  out.metrics = op_context.GetAllMetrics ();

  return out;
}

void
SignalProcessor::Flush ()
{
  FinalizeEpisode ();
  StartNewEpisode ();
}

void
SignalProcessor::StartNewEpisode ()
{
  if (store_)
    {
      episode_transaction_ = store_->Begin ();
    }
}

void
SignalProcessor::FinalizeEpisode ()
{
  if (!episode_transaction_)
    {
      return;
    }

  for (const auto &instruction : write_buffer_)
    {
      episode_transaction_->Execute (instruction.query, instruction.params);
    }
  write_buffer_.clear ();

  episode_transaction_->Commit ();

  // Maintain recent_context to last n_ctx(T) after boundary (Alg 12).
  const size_t keep = static_cast<size_t> (core::NCtx (config_.stability));
  auto &embs = context_->recent_context_embeddings;
  while (embs.size () > keep)
    {
      embs.pop_front ();
    }
}

} // namespace cortext
