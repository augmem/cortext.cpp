/// @file
/// @brief C API implementation for Cortext.
///
/// Return codes convention:
///   0 = success
///   1 = invalid parameters (NULL pointer, invalid handle)
///   2 = internal error (exception caught during processing)
#include "cortext/capi.h"
#include "cortext/cortext.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef CORTEXT_VERSION
#define CORTEXT_VERSION "0.0.0-dev"
#endif

namespace
{
thread_local std::string g_last_error;

void
clear_last_error ()
{
  g_last_error.clear ();
}

void
set_last_error (std::string message)
{
  g_last_error = std::move (message);
}

cortext::Cortext::Config
default_config ()
{
  return cortext::Cortext::Config{};
}

cortext::Cortext::Config
config_from_c (const cortext_config *cfg)
{
  auto cpp_cfg = default_config ();
  if (!cfg)
    {
      return cpp_cfg;
    }

  cpp_cfg.focus = cfg->focus;
  cpp_cfg.sensitivity = cfg->sensitivity;
  cpp_cfg.stability = cfg->stability;
  cpp_cfg.affect_interrupt = cfg->affect_interrupt != 0;
  cpp_cfg.affect_retrieval = cfg->affect_retrieval != 0;
  cpp_cfg.reinforcement_enabled = cfg->reinforcement_enabled != 0;
  cpp_cfg.procedural_enabled = cfg->procedural_enabled != 0;
  cpp_cfg.sequential_edges_enabled = cfg->sequential_edges_enabled != 0;
  cpp_cfg.label_bank_path = cfg->label_bank_path ? cfg->label_bank_path : "";
  return cpp_cfg;
}

char *
copy_string_result (const std::string &value)
{
  auto *buf = static_cast<char *> (std::malloc (value.size () + 1));
  if (!buf)
    {
      set_last_error ("out of memory while allocating string result");
      return nullptr;
    }
  std::memcpy (buf, value.c_str (), value.size () + 1);
  return buf;
}

std::string
encode_base64 (const std::vector<unsigned char> &data)
{
  static constexpr char kAlphabet[]
      = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  std::string out;
  out.reserve (((data.size () + 2) / 3) * 4);

  for (size_t i = 0; i < data.size (); i += 3)
    {
      const unsigned int octet_a = data[i];
      const unsigned int octet_b = (i + 1 < data.size ()) ? data[i + 1] : 0;
      const unsigned int octet_c = (i + 2 < data.size ()) ? data[i + 2] : 0;
      const unsigned int triple = (octet_a << 16) | (octet_b << 8) | octet_c;

      out.push_back (kAlphabet[(triple >> 18) & 0x3f]);
      out.push_back (kAlphabet[(triple >> 12) & 0x3f]);
      out.push_back ((i + 1 < data.size ()) ? kAlphabet[(triple >> 6) & 0x3f]
                                            : '=');
      out.push_back ((i + 2 < data.size ()) ? kAlphabet[triple & 0x3f] : '=');
    }

  return out;
}

nlohmann::json
memory_to_json (const cortext::Cortext::Context::Memory &memory)
{
  nlohmann::json content = nlohmann::json::array ();
  for (const auto &blob : memory.content)
    {
      content.push_back ({
          { "size_bytes", blob.size () },
          { "base64", encode_base64 (blob) },
      });
    }

  return {
    { "source_id", memory.source_id },
    { "id", memory.id },
    { "timestamp", memory.timestamp },
    { "retrieved_count", memory.retrieved_count },
    { "used_count", memory.used_count },
    { "modality", memory.modality },
    { "mimetype", memory.mimetype },
    { "content", std::move (content) },
    { "relevance", memory.relevance },
    { "mismatch", memory.mismatch },
    { "surprise", memory.surprise },
    { "rarity", memory.rarity },
    { "drift", memory.drift },
    { "contradiction", memory.contradiction },
    { "utility", memory.utility },
    { "periphery", memory.periphery },
    { "coverage", memory.coverage },
    { "salience", memory.salience },
    { "valence", memory.valence },
    { "arousal", memory.arousal },
    { "composite_score", memory.composite_score },
    { "threshold_t", memory.threshold_t },
  };
}

nlohmann::json
context_to_json (const cortext::Cortext::Context &ctx)
{
  nlohmann::json working_memory = nlohmann::json::array ();
  for (const auto &memory : ctx.working_memory)
    {
      working_memory.push_back (memory_to_json (memory));
    }

  nlohmann::json retrieved_memory = nlohmann::json::array ();
  for (const auto &memory : ctx.retrieved_memory)
    {
      retrieved_memory.push_back (memory_to_json (memory));
    }

  nlohmann::json metrics = nlohmann::json::object ();
  for (const auto &[metric_id, value] : ctx.output.metrics)
    {
      metrics[std::to_string (metric_id)] = value;
    }

  nlohmann::json operation_ms = nlohmann::json::object ();
  for (const auto &[name, value] : ctx.output.operation_ms)
    {
      operation_ms[name] = value;
    }

  nlohmann::json output = {
    { "effective_focus", ctx.output.effective_focus },
    { "coherence", ctx.output.coherence },
    { "emotion_intensity", ctx.output.emotion_intensity },
    { "valence", ctx.output.valence },
    { "arousal", ctx.output.arousal },
    { "metrics", std::move (metrics) },
    { "operation_ms", std::move (operation_ms) },
    { "composite_score",
      ctx.output.composite_score.has_value ()
          ? nlohmann::json (*ctx.output.composite_score)
          : nlohmann::json (nullptr) },
    { "threshold",
      ctx.output.threshold.has_value () ? nlohmann::json (*ctx.output.threshold)
                                        : nlohmann::json (nullptr) },
    { "decision",
      ctx.output.decision.has_value () ? nlohmann::json (*ctx.output.decision)
                                       : nlohmann::json (nullptr) },
    { "stored_embedding_id",
      ctx.output.stored_embedding_id.has_value ()
          ? nlohmann::json (*ctx.output.stored_embedding_id)
          : nlohmann::json (nullptr) },
  };

  return {
    { "working_memory", std::move (working_memory) },
    { "retrieved_memory", std::move (retrieved_memory) },
    { "should_interrupt", ctx.should_interrupt },
    { "interrupt_aborted", ctx.interrupt_aborted },
    { "at_boundary", ctx.at_boundary },
    { "consolidation_recommended", ctx.consolidation_recommended },
    { "consolidation_required", ctx.consolidation_required },
    { "interrupt_gate_has_candidates", ctx.interrupt_gate_has_candidates },
    { "interrupt_gate_blocked_no_store", ctx.interrupt_gate_blocked_no_store },
    { "interrupt_gate_rel_pass", ctx.interrupt_gate_rel_pass },
    { "interrupt_gate_novelty_pass", ctx.interrupt_gate_novelty_pass },
    { "interrupt_gate_mu_pass", ctx.interrupt_gate_mu_pass },
    { "interrupt_gate_novelty_mu_pass", ctx.interrupt_gate_novelty_mu_pass },
    { "interrupt_gate_dup_pass", ctx.interrupt_gate_dup_pass },
    { "interrupt_gate_boundary_mu_pass", ctx.interrupt_gate_boundary_mu_pass },
    { "interrupt_gate_rel_star", ctx.interrupt_gate_rel_star },
    { "interrupt_gate_retrieval_thresh", ctx.interrupt_gate_retrieval_thresh },
    { "interrupt_gate_boundary_mult_eff", ctx.interrupt_gate_boundary_mult_eff },
    { "interrupt_gate_affect_drive", ctx.interrupt_gate_affect_drive },
    { "boundary_score",
      ctx.boundary_score.has_value () ? nlohmann::json (*ctx.boundary_score)
                                      : nlohmann::json (nullptr) },
    { "output", std::move (output) },
    { "encode_ms", ctx.encode_ms },
    { "process_ms", ctx.process_ms },
    { "hydrate_ms", ctx.hydrate_ms },
    { "total_ms", ctx.total_ms },
  };
}

char *
context_result_to_json (const cortext::Cortext::Context &ctx)
{
  clear_last_error ();
  return copy_string_result (context_to_json (ctx).dump ());
}

cortext::ConsolidationMode
parse_mode_or_default (int mode)
{
  if (mode == CORTEXT_CONSOLIDATE_SHALLOW)
    {
      return cortext::ConsolidationMode::Shallow;
    }
  if (mode == CORTEXT_CONSOLIDATE_DEEP)
    {
      return cortext::ConsolidationMode::Deep;
    }
  return cortext::ConsolidationMode::Both;
}

template <typename Fn>
int
invoke_status_only (Fn &&fn)
{
  try
    {
      clear_last_error ();
      fn ();
      return 0;
    }
  catch (const std::exception &ex)
    {
      set_last_error (ex.what ());
      return 2;
    }
  catch (...)
    {
      set_last_error ("internal error");
      return 2;
    }
}

template <typename Fn>
char *
invoke_json (Fn &&fn)
{
  try
    {
      return context_result_to_json (fn ());
    }
  catch (const std::exception &ex)
    {
      set_last_error (ex.what ());
      return nullptr;
    }
  catch (...)
    {
      set_last_error ("internal error");
      return nullptr;
    }
}
} // namespace

extern "C"
{

  static cortext::Cortext *
  cast_handle (cortext_handle h)
  {
    return reinterpret_cast<cortext::Cortext *> (h);
  }

  cortext_handle
  cortext_create (double focus, double sensitivity, double stability,
                  const char *db_path)
  {
    if (!db_path)
      {
        set_last_error ("db_path must not be NULL");
        return nullptr;
      }

    cortext_config cfg{};
    cortext_config_init (&cfg);
    cfg.focus = focus;
    cfg.sensitivity = sensitivity;
    cfg.stability = stability;
    return cortext_create_with_config (&cfg, db_path, "models/imagebind");
  }

  void
  cortext_config_init (cortext_config *cfg)
  {
    if (!cfg)
      {
        set_last_error ("config pointer must not be NULL");
        return;
      }

    const auto defaults = default_config ();
    cfg->struct_size = sizeof (cortext_config);
    cfg->focus = defaults.focus;
    cfg->sensitivity = defaults.sensitivity;
    cfg->stability = defaults.stability;
    cfg->affect_interrupt = defaults.affect_interrupt ? 1 : 0;
    cfg->affect_retrieval = defaults.affect_retrieval ? 1 : 0;
    cfg->reinforcement_enabled = defaults.reinforcement_enabled ? 1 : 0;
    cfg->procedural_enabled = defaults.procedural_enabled ? 1 : 0;
    cfg->sequential_edges_enabled = defaults.sequential_edges_enabled ? 1 : 0;
    cfg->label_bank_path = nullptr;
    clear_last_error ();
  }

  cortext_handle
  cortext_create_with_config (const cortext_config *cfg, const char *db_path,
                              const char *models_dir)
  {
    if (!db_path)
      {
        set_last_error ("db_path must not be NULL");
        return nullptr;
      }

    try
      {
        auto inst = cortext::Cortext::Create (
            config_from_c (cfg), std::string (db_path),
            std::string (models_dir ? models_dir : "models/imagebind"));
        clear_last_error ();
        return reinterpret_cast<cortext_handle> (inst.release ());
      }
    catch (const std::exception &ex)
      {
        set_last_error (ex.what ());
        return nullptr;
      }
    catch (...)
      {
        set_last_error ("internal error");
        return nullptr;
      }
  }

  cortext_handle
  cortext_create_with_models (double focus, double sensitivity,
                              double stability, const char *db_path,
                              const char *models_dir)
  {
    if (!db_path || !models_dir)
      {
        set_last_error ("db_path and models_dir must not be NULL");
        return nullptr;
      }

    cortext_config cfg{};
    cortext_config_init (&cfg);
    cfg.focus = focus;
    cfg.sensitivity = sensitivity;
    cfg.stability = stability;
    return cortext_create_with_config (&cfg, db_path, models_dir);
  }

  void
  cortext_free (cortext_handle h)
  {
    if (!h)
      {
        return;
      }
    auto *p = cast_handle (h);
    delete p;
  }

  int
  cortext_process_text (cortext_handle h, const char *text,
                        const char *source_id)
  {
    auto *p = cast_handle (h);
    if (!p || !text || !source_id)
      {
        set_last_error ("handle, text, and source_id must all be non-NULL");
        return 1;
      }

    return invoke_status_only (
        [&] { (void)p->ProcessText (std::string (text), std::string (source_id)); });
  }

  int
  cortext_process_audio (cortext_handle h, const float *pcm,
                         size_t num_samples, const char *source_id)
  {
    auto *p = cast_handle (h);
    if (!p || !pcm || !source_id)
      {
        set_last_error ("handle, pcm, and source_id must all be non-NULL");
        return 1;
      }

    return invoke_status_only (
        [&] { (void)p->ProcessAudio (pcm, num_samples, std::string (source_id)); });
  }

  int
  cortext_process_image (cortext_handle h, const uint8_t *data, int width,
                         int height, int channels, const char *source_id)
  {
    auto *p = cast_handle (h);
    if (!p || !data || !source_id)
      {
        set_last_error ("handle, data, and source_id must all be non-NULL");
        return 1;
      }

    return invoke_status_only ([&] {
      (void)p->ProcessImage (data, width, height, channels,
                             std::string (source_id));
    });
  }

  int
  cortext_consolidate (cortext_handle h)
  {
    auto *p = cast_handle (h);
    if (!p)
      {
        set_last_error ("handle must not be NULL");
        return 1;
      }

    return invoke_status_only ([&] { (void)p->Consolidate (); });
  }

  int
  cortext_consolidate_mode (cortext_handle h, int mode)
  {
    auto *p = cast_handle (h);
    if (!p)
      {
        set_last_error ("handle must not be NULL");
        return 1;
      }

    return invoke_status_only (
        [&] { (void)p->Consolidate (parse_mode_or_default (mode)); });
  }

  int
  cortext_flush (cortext_handle h)
  {
    auto *p = cast_handle (h);
    if (!p)
      {
        set_last_error ("handle must not be NULL");
        return 1;
      }

    return invoke_status_only ([&] { p->Flush (); });
  }

  const char *
  cortext_version (void)
  {
    return CORTEXT_VERSION;
  }

  const char *
  cortext_last_error (void)
  {
    return g_last_error.empty () ? nullptr : g_last_error.c_str ();
  }

  void
  cortext_string_free (char *value)
  {
    std::free (value);
  }

  char *
  cortext_process_text_json (cortext_handle h, const char *text,
                             const char *source_id)
  {
    auto *p = cast_handle (h);
    if (!p || !text || !source_id)
      {
        set_last_error ("handle, text, and source_id must all be non-NULL");
        return nullptr;
      }

    return invoke_json (
        [&] { return p->ProcessText (std::string (text), std::string (source_id)); });
  }

  char *
  cortext_process_audio_json (cortext_handle h, const float *pcm,
                              size_t num_samples, const char *source_id)
  {
    auto *p = cast_handle (h);
    if (!p || !pcm || !source_id)
      {
        set_last_error ("handle, pcm, and source_id must all be non-NULL");
        return nullptr;
      }

    return invoke_json (
        [&] { return p->ProcessAudio (pcm, num_samples, std::string (source_id)); });
  }

  char *
  cortext_process_image_json (cortext_handle h, const uint8_t *data, int width,
                              int height, int channels,
                              const char *source_id)
  {
    auto *p = cast_handle (h);
    if (!p || !data || !source_id)
      {
        set_last_error ("handle, data, and source_id must all be non-NULL");
        return nullptr;
      }

    return invoke_json ([&] {
      return p->ProcessImage (data, width, height, channels,
                              std::string (source_id));
    });
  }

  char *
  cortext_consolidate_json (cortext_handle h)
  {
    auto *p = cast_handle (h);
    if (!p)
      {
        set_last_error ("handle must not be NULL");
        return nullptr;
      }

    return invoke_json ([&] { return p->Consolidate (); });
  }

  char *
  cortext_consolidate_mode_json (cortext_handle h, int mode)
  {
    auto *p = cast_handle (h);
    if (!p)
      {
        set_last_error ("handle must not be NULL");
        return nullptr;
      }

    return invoke_json (
        [&] { return p->Consolidate (parse_mode_or_default (mode)); });
  }

} // extern "C"
