#include "cortext/core/knobs.hpp"
#include "cortext/processor.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "cortext/store/schema.hpp"
#include <chrono>
#include <any>
#include <map>
#include <vector>

namespace cortext
{

namespace
{

void
ComputeObservedRetention (Store *store, const SignalProcessor::Config &config,
                          const Signal &signal, OperationContext &op_context)
{
  if (!store)
    {
      return;
    }
  const double cutoff = core::PeripheryCutoff (config.stability);
  const uint64_t now_ts = signal.timestamp;
  try
    {
      const std::vector<std::map<std::string, std::any> > rows
          = store->Execute ("SELECT last_used FROM memory_feedback "
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
      telemetry::LogWarn (
          "Observed retention query failed; skipping",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("db.system", "sqlite"),
            telemetry::Attribute::String ("db.operation", "SELECT") });
    }
}

void
AssembleOutputMemories (const OperationContext &op_context,
                        SignalProcessor::Output &out)
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

void
AssembleOutputFields (const OperationContext &op_context,
                      SignalProcessor::Output &out)
{
  out.interrupt_allowed = op_context.GetInterruptAllowed ();
  out.at_boundary = op_context.GetAtBoundary ();
  out.threshold_T_dynamic = op_context.GetThresholdTDynamic ();
  out.threshold_hysteresis = op_context.GetThresholdHysteresis ();
  out.effective_focus = op_context.GetEffectiveFocus ();
  out.emotion_intensity = op_context.GetEmotionIntensity ();
  out.valence = op_context.GetValence ();
  out.arousal = op_context.GetArousal ();
  out.mni_jaccard = op_context.GetMniJaccard ();
  out.mni_best_mu = op_context.GetMniBestMu ();
  out.mni_dup_thresh = op_context.GetMniDupThresh ();
  out.mni_tau_jaccard_eff = op_context.GetMniTauJaccardEff ();
  out.mni_tau_mu_eff = op_context.GetMniTauMuEff ();
  out.composite_score = op_context.GetCompositeScore ();
  out.serial_position_multiplier = op_context.GetSerialPositionMultiplier ();
  out.metrics = op_context.GetAllMetrics ();
}

const char *
GetMetricName (operations::Metric metric)
{
  switch (metric)
    {
    case operations::Metric::relevance:
      return "relevance";
    case operations::Metric::mismatch:
      return "mismatch";
    case operations::Metric::surprise:
      return "surprise";
    case operations::Metric::rarity:
      return "rarity";
    case operations::Metric::drift:
      return "drift";
    case operations::Metric::contradiction:
      return "contradiction";
    case operations::Metric::utility:
      return "utility";
    case operations::Metric::periphery:
      return "periphery";
    case operations::Metric::coverage:
      return "coverage";
    case operations::Metric::salience:
      return "salience";
    case operations::Metric::valence:
      return "valence";
    case operations::Metric::arousal:
      return "arousal";
    case operations::Metric::goal_alignment:
      return "goal_alignment";
    case operations::Metric::focus_spread:
      return "focus_spread";
    case operations::Metric::drift_mag:
      return "drift_mag";
    case operations::Metric::aw_prev:
      return "aw_prev";
    case operations::Metric::rate_prev:
      return "rate_prev";
    case operations::Metric::hys_prev:
      return "hys_prev";
    default:
      return "unknown";
    }
}

void
LogMetricTelemetry (const std::unordered_map<operations::Metric, double> &metrics)
{
  for (const auto &kv : metrics)
    {
      const char *metric_name = GetMetricName (kv.first);
      const std::string metric_full_name
          = std::string ("cortext.metric.") + metric_name;
      telemetry::RecordHistogram (metric_full_name, kv.second);
    }
}

void
LogProcessTelemetry (const OperationContext &op_context,
                     const SignalProcessor::Output &out)
{
  telemetry::RecordHistogram ("cortext.threshold_T_dynamic",
                              op_context.GetThresholdTDynamic ());
  telemetry::RecordHistogram ("cortext.threshold_hysteresis",
                              op_context.GetThresholdHysteresis ());
  telemetry::RecordHistogram ("cortext.effective_focus",
                              op_context.GetEffectiveFocus ());
  telemetry::RecordHistogram ("cortext.coherence", op_context.GetCoherence ());
  telemetry::RecordHistogram ("cortext.emotion_intensity",
                              op_context.GetEmotionIntensity ());
  telemetry::RecordHistogram ("cortext.mni_jaccard", op_context.GetMniJaccard ());
  telemetry::RecordHistogram ("cortext.mni_best_mu", op_context.GetMniBestMu ());
  telemetry::RecordHistogram ("cortext.mni_dup_thresh",
                              op_context.GetMniDupThresh ());
  telemetry::RecordHistogram ("cortext.last_weight_sum",
                              op_context.GetLastWeightSum ());
  telemetry::RecordHistogram ("cortext.last_effective_metric_count",
                              static_cast<double> (
                                  op_context.GetLastEffectiveMetricCount ()));
  LogMetricTelemetry (out.metrics);
}

} // namespace

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
  // Apply schema migrations exactly once during initialization.
  if (store_)
  {
      cortext::store::SchemaRegistry registry;
      if (root_operation_)
      {
          root_operation_->CollectSchema(registry);
      }
      cortext::store::ApplyMigrations(*store_, registry);
  }

  StartNewEpisode ();
}

SignalProcessor::~SignalProcessor () = default;

SignalProcessor::Output
SignalProcessor::Process (const Signal &signal)
{
  const auto t0 = std::chrono::steady_clock::now ();
  telemetry::ScopedSpan span ("cortext.process");
  OperationContext op_context (signal, *context_, config_, write_buffer_,
                               store_.get ());
  context_->write_rate_window_.Record (signal.timestamp);
  ComputeObservedRetention (store_.get (), config_, signal, op_context);
  root_operation_->Execute (op_context);
  span.SetAttribute ("cortext.at_boundary", op_context.GetAtBoundary ());
  span.SetAttribute ("cortext.interrupt_allowed",
                     op_context.GetInterruptAllowed ());
  span.SetAttribute ("cortext.threshold_T_dynamic",
                     op_context.GetThresholdTDynamic ());
  span.SetAttribute ("cortext.threshold_hysteresis",
                     op_context.GetThresholdHysteresis ());
  span.SetAttribute ("cortext.effective_focus", op_context.GetEffectiveFocus ());
  if (op_context.ShouldFinalizeEpisode ())
    {
      FinalizeEpisode ();
      StartNewEpisode ();
    }
  context_->signals_processed += 1;
  Output out;
  AssembleOutputMemories (op_context, out);
  AssembleOutputFields (op_context, out);
  const auto t1 = std::chrono::steady_clock::now ();
  const double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli> > (t1 - t0).count ();
  telemetry::RecordHistogram ("cortext.process_duration_ms", ms);
  telemetry::AddCounter ("cortext.signals_processed_total", 1);
  if (out.at_boundary)
    {
      telemetry::AddCounter ("cortext.at_boundary_total", 1);
    }
  if (out.interrupt_allowed)
    {
      telemetry::AddCounter ("cortext.interrupt_allowed_total", 1);
    }
  LogProcessTelemetry (op_context, out);
  span.SetStatusOk ();
  return out;
}

void
SignalProcessor::Flush ()
{
  telemetry::AddCounter ("cortext.flush_total", 1);
  FinalizeEpisode ();
  StartNewEpisode ();
}

void
SignalProcessor::StartNewEpisode ()
{
  // Keep the episode transaction closed during signal processing to avoid
  // holding a long-lived read transaction that blocks other writers. We open
  // a short-lived transaction only when finalizing buffered writes.
  episode_transaction_.reset ();
}

void
SignalProcessor::FinalizeEpisode ()
{
  if (!store_)
    {
      return;
    }

  telemetry::ScopedSpan span ("cortext.episode.finalize");

  if (write_buffer_.empty ())
    {
      span.SetStatusOk ();
      return;
    }

  episode_transaction_ = store_->Begin ();
  for (const auto &instruction : write_buffer_)
    {
      episode_transaction_->Execute (instruction.query, instruction.params);
    }
  write_buffer_.clear ();

  episode_transaction_->Commit ();
  episode_transaction_.reset ();
  telemetry::AddCounter ("cortext.episode_commit_total", 1);
  span.SetStatusOk ();

  // Maintain recent_context to last n_ctx(T) after boundary (Alg 12).
  const size_t keep = static_cast<size_t> (core::NCtx (config_.stability));
  auto &embs = context_->recent_context_embeddings;
  while (embs.size () > keep)
    {
      embs.pop_front ();
    }
}

} // namespace cortext
