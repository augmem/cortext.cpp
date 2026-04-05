#include "../../src/operations/meta_learning_internal.hpp"

#include <cortext/core/constants.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/focus.hpp>
#include <cortext/operations/sensitivity.hpp>
#include <cortext/operations/stability.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <any>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace
{
class BenchEncoder : public cortext::Encoder
{
public:
  void
  EncodeText (const std::string &, std::vector<float> &out_embedding) override
  {
    out_embedding.assign (256, 0.0f);
    out_embedding[0] = 1.0f;
  }

  void
  EncodeAudio (const float *, std::size_t, std::vector<float> &out_embedding) override
  {
    out_embedding.assign (256, 0.0f);
    out_embedding[0] = 1.0f;
  }

  void
  EncodeImage (const std::uint8_t *, int, int, int,
               std::vector<float> &out_embedding) override
  {
    out_embedding.assign (256, 0.0f);
    out_embedding[0] = 1.0f;
  }
};

BenchEncoder &
GetBenchEncoder ()
{
  static BenchEncoder encoder;
  return encoder;
}

class ScopedEnvVar
{
public:
  explicit ScopedEnvVar (const char *name) : name_ (name)
  {
    const char *existing = std::getenv (name);
    if (existing)
      {
        had_value_ = true;
        old_value_ = existing;
      }
    unsetenv (name_);
  }

  ScopedEnvVar (const char *name, const std::string &value) : ScopedEnvVar (name)
  {
    setenv (name_, value.c_str (), 1);
  }

  ~ScopedEnvVar ()
  {
    if (had_value_)
      {
        setenv (name_, old_value_.c_str (), 1);
      }
    else
      {
        unsetenv (name_);
      }
  }

private:
  const char *name_;
  bool had_value_ = false;
  std::string old_value_;
};

struct StudyResult
{
  double attention_width_prior = 0.0;
  double rate_target_prior = 0.0;
  double hysteresis_band_prior = 0.0;
  long long update_count = 0;
};

cortext::Signal
MakeSignal (std::uint64_t ts)
{
  cortext::Signal s;
  s.embedding = Eigen::VectorXf::Zero (256);
  s.embedding[0] = 1.0f;
  s.timestamp = ts;
  s.source_id = "bench";
  return s;
}

StudyResult
RunStudy (bool enabled)
{
  ScopedEnvVar disable ("CORTEXT_DISABLE_META_LEARNING",
                        enabled ? "0" : "1");

  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  cortext::ProcessorContext pctx;
  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto tx = store->Begin ();
  cortext::OperationContext init_ctx (MakeSignal (1000), pctx, cfg, store.get ());
  cortext::operations::InitializeFocusPriors focus;
  cortext::operations::InitializeSensitivityPriors sensitivity;
  cortext::operations::InitializeStabilityPriors stability;
  focus.Execute (init_ctx, *tx);
  sensitivity.Execute (init_ctx, *tx);
  stability.Execute (init_ctx, *tx);

  cortext::operations::ApplyMetaLearning meta;
  for (int i = 0; i < 8; ++i)
    {
      cortext::OperationContext ctx (
          MakeSignal (2000 + static_cast<std::uint64_t> (i) * 1000), pctx, cfg,
          store.get ());
      pctx.attention_width = static_cast<double> (cortext::core::kAttentionWidthMin);
      pctx.rate_target = 4.8;
      pctx.rho_hat_prev = 4.8;
      pctx.hysteresis = 0.22;
      pctx.u_t = 0.1;
      pctx.last_used_flag = 1.0;
      pctx.delta_reward = 0.6;
      ctx.SetWriteDecision (true);
      meta.Execute (ctx, *tx);
    }

  StudyResult out;
  out.attention_width_prior = pctx.attention_width_prior;
  out.rate_target_prior = pctx.rate_target_prior;
  out.hysteresis_band_prior = pctx.hysteresis_band_prior;
  auto rows = tx->Execute (
      "SELECT COALESCE(SUM(update_count), 0) AS updates FROM meta_learning_coeffs");
  if (!rows.empty ())
    {
      const auto it = rows[0].find ("updates");
      if (it != rows[0].end () && it->second.type () == typeid (long long))
        {
          out.update_count = std::any_cast<long long> (it->second);
        }
      else if (it != rows[0].end () && it->second.type () == typeid (int))
        {
          out.update_count = static_cast<long long> (std::any_cast<int> (it->second));
        }
    }
  return out;
}

} // namespace

int
main ()
{
  const StudyResult on = RunStudy (true);
  const StudyResult off = RunStudy (false);

  std::cout << "meta_learning_attention_width_prior_on="
            << on.attention_width_prior << "\n";
  std::cout << "meta_learning_attention_width_prior_off="
            << off.attention_width_prior << "\n";
  std::cout << "meta_learning_rate_target_prior_on="
            << on.rate_target_prior << "\n";
  std::cout << "meta_learning_rate_target_prior_off="
            << off.rate_target_prior << "\n";
  std::cout << "meta_learning_hysteresis_band_prior_on="
            << on.hysteresis_band_prior << "\n";
  std::cout << "meta_learning_hysteresis_band_prior_off="
            << off.hysteresis_band_prior << "\n";
  std::cout << "meta_learning_update_count_on=" << on.update_count << "\n";
  std::cout << "meta_learning_update_count_off=" << off.update_count << "\n";

  const bool attention_ok = on.attention_width_prior < off.attention_width_prior;
  const bool rate_ok = on.rate_target_prior > off.rate_target_prior;
  const bool hysteresis_ok = on.hysteresis_band_prior > off.hysteresis_band_prior;
  const bool update_ok = on.update_count == 24 && off.update_count == 0;

  std::cout << "meta_learning_attention_shift_ok=" << (attention_ok ? 1 : 0)
            << "\n";
  std::cout << "meta_learning_rate_shift_ok=" << (rate_ok ? 1 : 0) << "\n";
  std::cout << "meta_learning_hysteresis_shift_ok=" << (hysteresis_ok ? 1 : 0)
            << "\n";
  std::cout << "meta_learning_update_count_ok=" << (update_ok ? 1 : 0)
            << "\n";

  return (attention_ok && rate_ok && hysteresis_ok && update_ok) ? 0 : 1;
}
