#include "meta_learning_internal.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/utils.hpp"
#include "../experimental_env.hpp"

#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cortext::operations::meta_learning
{

namespace
{
enum class Family
{
  AttentionWidth,
  RateTarget,
  HysteresisBand
};

struct FamilySpec
{
  const char *name = "";
  double domain_min = 0.0;
  double domain_max = 1.0;
  double alpha_f = 0.0;
  double alpha_s = 0.0;
  double alpha_t = 0.0;
  double beta = 0.0;
  double a = 0.0;
  double b = 1.0;
};

struct CoeffRow
{
  std::string family;
  double alpha_f = 0.0;
  double alpha_s = 0.0;
  double alpha_t = 0.0;
  double beta = 0.0;
  double a = 0.0;
  double b = 1.0;
  long long update_count = 0;
  long long updated_at = 0;
};

constexpr double kCoeffClamp = 8.0;
constexpr double kMinLearningMagnitude = 0.05;
constexpr double kCoeffLearningRate = 0.08;
constexpr double kEndpointLearningRate = 0.03;

template <typename Executor>
std::vector<std::map<std::string, std::any>>
Exec (Executor &executor, const std::string &query,
      const std::vector<std::any> &params = {})
{
  return executor.Execute (query, params);
}

double
GetDouble (const std::map<std::string, std::any> &row, const char *key,
           double fallback)
{
  auto it = row.find (key);
  if (it == row.end () || !it->second.has_value ())
    {
      return fallback;
    }
  return store::AnyToDouble (it->second, fallback);
}

long long
GetInt64 (const std::map<std::string, std::any> &row, const char *key,
          long long fallback)
{
  auto it = row.find (key);
  if (it == row.end () || !it->second.has_value ())
    {
      return fallback;
    }
  if (it->second.type () == typeid (long long))
    {
      return std::any_cast<long long> (it->second);
    }
  if (it->second.type () == typeid (int))
    {
      return static_cast<long long> (std::any_cast<int> (it->second));
    }
  return fallback;
}

FamilySpec
GetFamilySpec (Family family)
{
  switch (family)
    {
    case Family::AttentionWidth:
      return { "focus_attention_width",
               static_cast<double> (core::kAttentionWidthMin),
               static_cast<double> (core::kAttentionWidthMax),
               6.0,
               0.0,
               0.0,
               -3.0,
               static_cast<double> (core::kAttentionWidthMax),
               static_cast<double> (core::kAttentionWidthMin) };
    case Family::RateTarget:
      return { "sensitivity_rate_target",
               0.2,
               5.0,
               0.0,
               6.0,
               0.0,
               -3.0,
               0.2,
               5.0 };
    case Family::HysteresisBand:
      return { "stability_hysteresis_band",
               constants::kHysteresisBandMin,
               constants::kHysteresisBandMax,
               0.0,
               0.0,
               6.0,
               -3.0,
               0.02,
               0.25 };
    }
  return {};
}

CoeffRow
DefaultRow (Family family)
{
  const auto spec = GetFamilySpec (family);
  return { spec.name, spec.alpha_f, spec.alpha_s, spec.alpha_t, spec.beta,
           spec.a,    spec.b,       0,            0 };
}

double
LegacyValue (Family family, const SignalProcessor::Config &config)
{
  switch (family)
    {
    case Family::AttentionWidth:
      return core::Lerp (static_cast<double> (core::kAttentionWidthMin),
                         static_cast<double> (core::kAttentionWidthMax),
                         1.0 - config.focus);
    case Family::RateTarget:
      return core::BaseRatePrior (config.sensitivity);
    case Family::HysteresisBand:
      return core::BaseBandPrior (config.stability);
    }
  return 0.0;
}

double
Evaluate (const CoeffRow &row, const SignalProcessor::Config &config)
{
  const double z = row.alpha_f * core::Clamp (config.focus, 0.0, 1.0)
                   + row.alpha_s * core::Clamp (config.sensitivity, 0.0, 1.0)
                   + row.alpha_t * core::Clamp (config.stability, 0.0, 1.0)
                   + row.beta;
  const double kappa = core::Sigmoid (z);
  return core::Lerp (row.a, row.b, kappa);
}

double
Normalize (double value, double domain_min, double domain_max)
{
  const double denom = std::max (1e-9, domain_max - domain_min);
  return core::Clamp ((value - domain_min) / denom, 0.0, 1.0);
}

template <typename Executor>
std::optional<CoeffRow>
LoadRowIfExists (Executor &executor, Family family)
{
  const auto spec = GetFamilySpec (family);
  auto rows = Exec (
      executor,
      "SELECT family, alpha_f, alpha_s, alpha_t, beta, a, b, "
      "       update_count, updated_at "
      "FROM meta_learning_coeffs WHERE family = ?",
      { std::string (spec.name) });
  if (rows.empty ())
    {
      return std::nullopt;
    }

  CoeffRow row = DefaultRow (family);
  const auto &src = rows[0];
  row.family = spec.name;
  row.alpha_f = core::Clamp (GetDouble (src, "alpha_f", row.alpha_f),
                             -kCoeffClamp, kCoeffClamp);
  row.alpha_s = core::Clamp (GetDouble (src, "alpha_s", row.alpha_s),
                             -kCoeffClamp, kCoeffClamp);
  row.alpha_t = core::Clamp (GetDouble (src, "alpha_t", row.alpha_t),
                             -kCoeffClamp, kCoeffClamp);
  row.beta = core::Clamp (GetDouble (src, "beta", row.beta), -kCoeffClamp,
                          kCoeffClamp);
  row.a = core::Clamp (GetDouble (src, "a", row.a), spec.domain_min,
                       spec.domain_max);
  row.b = core::Clamp (GetDouble (src, "b", row.b), spec.domain_min,
                       spec.domain_max);
  row.update_count = GetInt64 (src, "update_count", 0);
  row.updated_at = GetInt64 (src, "updated_at", 0);
  return row;
}

template <typename Executor>
void
PersistRow (Executor &executor, const CoeffRow &row)
{
  Exec (executor,
        "INSERT INTO meta_learning_coeffs("
        "family, alpha_f, alpha_s, alpha_t, beta, a, b, update_count, updated_at) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(family) DO UPDATE SET "
        "alpha_f=excluded.alpha_f, alpha_s=excluded.alpha_s, "
        "alpha_t=excluded.alpha_t, beta=excluded.beta, a=excluded.a, "
        "b=excluded.b, update_count=excluded.update_count, "
        "updated_at=excluded.updated_at",
        { row.family, row.alpha_f, row.alpha_s, row.alpha_t, row.beta, row.a,
          row.b,      row.update_count, row.updated_at });
}

bool
InformativeOutcome (const OperationContext &context)
{
  const auto &pctx = context.GetProcessorContext ();
  return context.GetWriteDecision () || pctx.last_used_flag > 0.0
         || std::abs (pctx.delta_reward) > 1e-3;
}

double
OutcomeDirection (const OperationContext &context)
{
  const auto &pctx = context.GetProcessorContext ();
  const double acceptance = context.GetWriteDecision () ? 1.0 : 0.0;
  const double retrieval_use = core::Clamp (pctx.last_used_flag, 0.0, 1.0);
  const double certainty = core::Clamp (1.0 - pctx.u_t, 0.0, 1.0);
  const double reward = core::Clamp (0.5 + 0.5 * pctx.delta_reward, 0.0, 1.0);
  const double success = core::Clamp (0.30 * acceptance + 0.30 * retrieval_use
                                          + 0.25 * certainty + 0.15 * reward,
                                      0.0, 1.0);
  return core::Clamp (2.0 * success - 1.0, -1.0, 1.0);
}

double
ObservedValueForFamily (Family family, const OperationContext &context)
{
  const auto &pctx = context.GetProcessorContext ();
  switch (family)
    {
    case Family::AttentionWidth:
      return core::Clamp (pctx.attention_width,
                          static_cast<double> (core::kAttentionWidthMin),
                          static_cast<double> (core::kAttentionWidthMax));
    case Family::RateTarget:
      {
        const double observed = (pctx.rho_hat_prev > 0.0) ? pctx.rho_hat_prev
                                                          : pctx.rate_target;
        return core::Clamp (observed, 0.2, 5.0);
      }
    case Family::HysteresisBand:
      return core::Clamp (pctx.hysteresis, constants::kHysteresisBandMin,
                          constants::kHysteresisBandMax);
    }
  return 0.0;
}

void
ApplyLearnedPriors (Family family, double value, ProcessorContext &pctx)
{
  switch (family)
    {
    case Family::AttentionWidth:
      pctx.attention_width_prior = value;
      break;
    case Family::RateTarget:
      pctx.base_rate_prior = value;
      pctx.rate_target_prior = value;
      pctx.rate_target = value;
      break;
    case Family::HysteresisBand:
      pctx.hysteresis_band_prior = value;
      pctx.hysteresis = value;
      break;
    }
}

void
UpdateRowForFamily (CoeffRow &row, const FamilySpec &spec,
                    const SignalProcessor::Config &config,
                    double observed_value, double direction)
{
  const double magnitude = std::abs (direction);
  if (magnitude < kMinLearningMagnitude)
    {
      return;
    }

  const double z = row.alpha_f * core::Clamp (config.focus, 0.0, 1.0)
                   + row.alpha_s * core::Clamp (config.sensitivity, 0.0, 1.0)
                   + row.alpha_t * core::Clamp (config.stability, 0.0, 1.0)
                   + row.beta;
  const double kappa = core::Sigmoid (z);
  const double predicted = core::Clamp (
      core::Lerp (row.a, row.b, kappa), spec.domain_min, spec.domain_max);

  const double error_norm
      = Normalize (observed_value, spec.domain_min, spec.domain_max)
        - Normalize (predicted, spec.domain_min, spec.domain_max);
  const double signed_error = direction * error_norm;
  const double sig_grad = kappa * (1.0 - kappa);
  const double span = spec.domain_max - spec.domain_min;
  const double coeff_lr = kCoeffLearningRate * magnitude;
  const double endpoint_lr = kEndpointLearningRate * magnitude;

  row.alpha_f = core::Clamp (
      row.alpha_f
          + coeff_lr * signed_error * (row.b - row.a) * sig_grad * config.focus,
      -kCoeffClamp, kCoeffClamp);
  row.alpha_s = core::Clamp (
      row.alpha_s + coeff_lr * signed_error * (row.b - row.a) * sig_grad
                        * config.sensitivity,
      -kCoeffClamp, kCoeffClamp);
  row.alpha_t = core::Clamp (
      row.alpha_t + coeff_lr * signed_error * (row.b - row.a) * sig_grad
                        * config.stability,
      -kCoeffClamp, kCoeffClamp);
  row.beta = core::Clamp (
      row.beta + coeff_lr * signed_error * (row.b - row.a) * sig_grad,
      -kCoeffClamp, kCoeffClamp);

  row.a = core::Clamp (
      row.a + endpoint_lr * signed_error * (1.0 - kappa) * span,
      spec.domain_min, spec.domain_max);
  row.b = core::Clamp (row.b + endpoint_lr * signed_error * kappa * span,
                       spec.domain_min, spec.domain_max);
}

template <typename Executor>
double
ResolveFamilyValue (Executor &executor, Family family,
                    const SignalProcessor::Config &config)
{
  const auto spec = GetFamilySpec (family);
  const auto row = LoadRowIfExists (executor, family);
  if (!row.has_value ())
    {
      return core::Clamp (LegacyValue (family, config), spec.domain_min,
                          spec.domain_max);
    }
  return core::Clamp (Evaluate (*row, config), spec.domain_min, spec.domain_max);
}

template <typename Executor>
CoeffRow
LoadOrCreateLearnableRow (Executor &executor, Family family,
                          const SignalProcessor::Config &config)
{
  if (const auto existing = LoadRowIfExists (executor, family); existing.has_value ())
    {
      return *existing;
    }

  auto row = DefaultRow (family);
  const double seed = LegacyValue (family, config);
  const auto spec = GetFamilySpec (family);
  const double epsilon = 0.01 * (spec.domain_max - spec.domain_min);
  row.a = core::Clamp (seed - 0.5 * epsilon, spec.domain_min, spec.domain_max);
  row.b = core::Clamp (seed + 0.5 * epsilon, spec.domain_min, spec.domain_max);
  return row;
}

} // namespace

bool
Disabled ()
{
  return internal::experimental_env::Flag ("CORTEXT_DISABLE_META_LEARNING");
}

double
ResolveAttentionWidthPrior (Transaction &tx, const SignalProcessor::Config &config)
{
  if (Disabled ())
    {
      return core::Lerp (static_cast<double> (core::kAttentionWidthMin),
                         static_cast<double> (core::kAttentionWidthMax),
                         1.0 - config.focus);
    }
  return ResolveFamilyValue (tx, Family::AttentionWidth, config);
}

double
ResolveRateTargetPrior (Transaction &tx, const SignalProcessor::Config &config)
{
  if (Disabled ())
    {
      return core::BaseRatePrior (config.sensitivity);
    }
  return ResolveFamilyValue (tx, Family::RateTarget, config);
}

double
ResolveHysteresisBandPrior (Transaction &tx,
                            const SignalProcessor::Config &config)
{
  if (Disabled ())
    {
      return core::BaseBandPrior (config.stability);
    }
  return ResolveFamilyValue (tx, Family::HysteresisBand, config);
}

void
LearnFromOutcome (OperationContext &context, Transaction &tx)
{
  if (Disabled () || !InformativeOutcome (context))
    {
      return;
    }

  const auto &config = context.GetConfig ();
  auto &pctx = context.GetProcessorContext ();
  const double direction = OutcomeDirection (context);
  if (std::abs (direction) < kMinLearningMagnitude)
    {
      return;
    }

  const std::array<Family, 3> families = {
    Family::AttentionWidth,
    Family::RateTarget,
    Family::HysteresisBand
  };

  for (const Family family : families)
    {
      const auto spec = GetFamilySpec (family);
      auto row = LoadOrCreateLearnableRow (tx, family, config);
      const double observed_value = ObservedValueForFamily (family, context);
      UpdateRowForFamily (row, spec, config, observed_value, direction);
      row.update_count += 1;
      row.updated_at = static_cast<long long> (context.GetSignal ().timestamp);
      PersistRow (tx, row);
      ApplyLearnedPriors (family,
                          core::Clamp (Evaluate (row, config), spec.domain_min,
                                       spec.domain_max),
                          pctx);
    }
}

} // namespace cortext::operations::meta_learning

namespace cortext::operations
{

void
ApplyMetaLearning::Execute (OperationContext &context, Transaction &tx) const
{
  meta_learning::LearnFromOutcome (context, tx);
}

} // namespace cortext::operations
