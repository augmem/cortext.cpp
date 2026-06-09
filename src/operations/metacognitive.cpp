#include "cortext/operations/metacognitive.hpp"

#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "eviction_ablation.hpp"
#include "storage_pressure.hpp"
#include "temporal_retrieval.hpp"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <string>

namespace cortext::operations
{

namespace
{
bool
EnvFlag (const char *name)
{
  const char *value = std::getenv (name);
  if (!value)
    {
      return false;
    }
  std::string s (value);
  std::transform (s.begin (), s.end (), s.begin (),
                  [] (unsigned char c) { return static_cast<char> (std::tolower (c)); });
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

double
ResolveResurfacingDecayScale (
    Transaction &tx,
    const temporal::RetrievalAblationOverride &retrieval_override,
    double stability)
{
  const auto mode = temporal::ResolveResurfacingDecayMode (retrieval_override);
  if (mode == temporal::ResurfacingDecayMode::TimeOnly)
    {
      return 1.0;
    }

  const auto eviction_override = eviction::GetEvictionAblationOverride ();
  const auto pressure_state
      = pressure::ComputeStoragePressureState (tx, eviction_override);
  switch (mode)
    {
    case temporal::ResurfacingDecayMode::PressureGate:
      return pressure::GateScale (
          pressure_state, core::RetrievalPressureGateLowScale (stability));
    case temporal::ResurfacingDecayMode::PressureRamp:
      return pressure::RampScale (
          pressure_state, core::RetrievalPressureRampLowScale (stability));
    case temporal::ResurfacingDecayMode::TimeOnly:
      break;
    }
  return 1.0;
}
} // namespace

void
MetacognitiveMonitoring::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  const double F = core::Clamp01 (cfg.focus);
  const double S = core::Clamp01 (cfg.sensitivity);
  const double T = core::Clamp01 (cfg.stability);

  double retrieval_strength = 0.0;
  const auto &usage_events = context.GetMemoryUsageEvents ();
  for (const auto &event : usage_events)
    {
      if (!event.contextual_gain.has_value ())
        {
          continue;
        }
      retrieval_strength = std::max (
          retrieval_strength, core::Map01 (*event.contextual_gain));
    }
  const double retrieval = core::Clamp01 (retrieval_strength);

  const double FOK = core::Clamp01 (
      context.GetFeelingOfKnowing ().value_or (1.0 - p_ctx.u_t));

  // Derived thresholds and params
  const double fok_threshold = core::FOKThreshold (F);
  const double tot_fok_cut = core::TOTFokCutoff (F);
  const double tot_ret_cut = core::TOTRetrievalCutoff (F);
  const double unknown_cut = core::UnknownThreshold (F);
  const double conf_decay = core::ConfidenceDecayRate (T);
  const int switch_latency_ms = core::StrategySwitchLatencyMs (S);
  const double meta_sens = core::MetacognitiveSensitivity (F, S);
  const std::uint64_t now_ts = context.GetSignal ().timestamp;
  const bool disable_conf_decay
      = EnvFlag ("CORTEXT_DISABLE_METACOG_CONFIDENCE_DECAY");
  const auto retrieval_override = temporal::GetRetrievalAblationOverride ();
  const auto resurfacing_decay_mode
      = temporal::ResolveResurfacingDecayMode (retrieval_override);
  const double resurfacing_decay_scale
      = ResolveResurfacingDecayScale (tx, retrieval_override, T);
  const double certainty_req
      = core::CertaintyRequirement (resurfacing_decay_scale);
  double delta_t_s = 0.0;
  if (p_ctx.last_signal_timestamp > 0 && now_ts > p_ctx.last_signal_timestamp)
    {
      delta_t_s
          = static_cast<double> (now_ts - p_ctx.last_signal_timestamp) / 1000.0;
    }
  const double effective_delta_t_s = delta_t_s * resurfacing_decay_scale;
  const double decay_factor
      = disable_conf_decay ? 1.0 : std::exp (-conf_decay * effective_delta_t_s);
  const double retained_confidence
      = core::Clamp01 (p_ctx.metacognitive_confidence * decay_factor);
  const double confidence_alpha
      = core::MetacognitiveConfidenceAlpha (F, S, T);
  const double confidence = core::Clamp01 (
      std::max (FOK, core::Ewma (retained_confidence, FOK, confidence_alpha)));

  context.SetFeelingOfKnowing (FOK);
  p_ctx.fok_state = FOK;
  p_ctx.retrieval_strength = retrieval;
  p_ctx.metacognitive_confidence = confidence;

  // Decisions
  const bool tot_detected
      = (confidence > tot_fok_cut) && (retrieval < tot_ret_cut);
  const bool unknown_detected
      = (retrieval < unknown_cut) && (confidence < certainty_req);

  if (p_ctx.metacognitive_mode_expires_at > 0
      && now_ts >= p_ctx.metacognitive_mode_expires_at)
    {
      p_ctx.metacognitive_mode = ProcessorContext::MetacognitiveMode::Normal;
      p_ctx.metacognitive_mode_expires_at = 0;
      p_ctx.metacognitive_certainty_satisfied = false;
    }

  if (tot_detected)
    {
      p_ctx.metacognitive_mode
          = ProcessorContext::MetacognitiveMode::TotRecovery;
      p_ctx.metacognitive_mode_expires_at
          = now_ts + static_cast<std::uint64_t> (switch_latency_ms);
      p_ctx.metacognitive_certainty_satisfied = false;
      p_ctx.metacognitive_tot_trigger_count++;
    }
  else if (unknown_detected)
    {
      p_ctx.metacognitive_mode
          = ProcessorContext::MetacognitiveMode::UnknownCaution;
      p_ctx.metacognitive_mode_expires_at
          = now_ts + static_cast<std::uint64_t> (switch_latency_ms);
      p_ctx.metacognitive_certainty_satisfied = false;
      p_ctx.metacognitive_unknown_trigger_count++;
    }
  else
    {
      if (p_ctx.metacognitive_mode
              == ProcessorContext::MetacognitiveMode::TotRecovery
          && retrieval >= tot_ret_cut)
        {
          p_ctx.metacognitive_mode
              = ProcessorContext::MetacognitiveMode::Normal;
          p_ctx.metacognitive_mode_expires_at = 0;
          p_ctx.metacognitive_certainty_satisfied = false;
        }
      if (p_ctx.metacognitive_mode
              == ProcessorContext::MetacognitiveMode::UnknownCaution
          && p_ctx.metacognitive_certainty_satisfied)
        {
          p_ctx.metacognitive_mode
              = ProcessorContext::MetacognitiveMode::Normal;
          p_ctx.metacognitive_mode_expires_at = 0;
          p_ctx.metacognitive_certainty_satisfied = false;
        }
    }

  // Expose outputs
  context.SetMetacogFOKThreshold (fok_threshold);
  context.SetMetacogTOTDetected (tot_detected);
  context.SetMetacogUnknownDetected (unknown_detected);
  context.SetMetacogConfidenceDecayRate (conf_decay);
  context.SetMetacogStrategySwitchLatencyMs (switch_latency_ms);
  context.SetMetacogCertaintyRequirement (certainty_req);
  context.SetMetacogSensitivity (meta_sens);

  telemetry::LogDebug ("cortext.metacognitive",
                       { telemetry::Attribute::Double ("fok_score", FOK),
                         telemetry::Attribute::Double (
                             "metacognitive_confidence", confidence),
                         telemetry::Attribute::Double (
                             "metacognitive_retained_confidence",
                             retained_confidence),
                         telemetry::Attribute::Double ("confidence_decay_rate",
                                                      conf_decay),
                         telemetry::Attribute::Double ("confidence_decay_factor",
                                                      decay_factor),
                         telemetry::Attribute::Double ("delta_t_s", delta_t_s),
                         telemetry::Attribute::Double (
                             "effective_delta_t_s", effective_delta_t_s),
                         telemetry::Attribute::String (
                             "resurfacing_decay_mode",
                             temporal::ToString (resurfacing_decay_mode)),
                         telemetry::Attribute::Double (
                             "resurfacing_decay_scale",
                             resurfacing_decay_scale),
                         telemetry::Attribute::Double ("retrieval_strength",
                                                      retrieval),
                         telemetry::Attribute::Bool ("tot_state", tot_detected),
                         telemetry::Attribute::Bool ("unknown_state",
                                                     unknown_detected),
                         telemetry::Attribute::Int64 (
                             "metacognitive_mode",
                             static_cast<int64_t> (p_ctx.metacognitive_mode)),
                         telemetry::Attribute::Int64 (
                             "metacognitive_mode_expires_at",
                             static_cast<int64_t> (
                                 p_ctx.metacognitive_mode_expires_at)) });
}

} // namespace cortext::operations
