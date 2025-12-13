/// @file
/// @brief C API implementation for Cortext.
///
/// Return codes convention:
///   0 = success
///   1 = invalid parameters (NULL pointer, invalid handle)
///   2 = internal error (exception caught during processing)
#include "cortext/capi.h"
#include "cortext/cortext.hpp"

#include <memory>
#include <new>
#include <string>

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
        return nullptr;
      }
    cortext::Cortext::Config cfg;
    cfg.focus = focus;
    cfg.sensitivity = sensitivity;
    cfg.stability = stability;
    auto inst = cortext::Cortext::Create (cfg, std::string (db_path),
                                          std::string ("models/imagebind"));
    return reinterpret_cast<cortext_handle> (inst.release ());
  }

  cortext_handle
  cortext_create_with_models (double focus, double sensitivity,
                              double stability, const char *db_path,
                              const char *models_dir)
  {
    if (!db_path || !models_dir)
      {
        return nullptr;
      }
    cortext::Cortext::Config cfg;
    cfg.focus = focus;
    cfg.sensitivity = sensitivity;
    cfg.stability = stability;
    auto inst = cortext::Cortext::Create (cfg, std::string (db_path),
                                          std::string (models_dir));
    return reinterpret_cast<cortext_handle> (inst.release ());
  }

  void
  cortext_free (cortext_handle h)
  {
    auto *p = cast_handle (h);
    delete p;
  }

  int
  cortext_process_text (cortext_handle h, const char *text, uint64_t timestamp,
                        const char *source_id)
  {
    auto *p = cast_handle (h);
    if (!p || !text || !source_id)
      {
        return 1;
      }
    try
      {
        (void)p->ProcessText (std::string (text), timestamp,
                              std::string (source_id));
        return 0;
      }
    catch (...)
      {
        return 2;
      }
  }

  int
  cortext_process_audio (cortext_handle h, const float *pcm,
                         size_t num_samples, uint64_t timestamp,
                         const char *source_id)
  {
    auto *p = cast_handle (h);
    if (!p || !pcm || !source_id)
      {
        return 1;
      }
    try
      {
        (void)p->ProcessAudio (pcm, num_samples, timestamp,
                               std::string (source_id));
        return 0;
      }
    catch (...)
      {
        return 2;
      }
  }

  int
  cortext_process_image (cortext_handle h, const uint8_t *data, int width,
                         int height, int channels, uint64_t timestamp,
                         const char *source_id)
  {
    auto *p = cast_handle (h);
    if (!p || !data || !source_id)
      {
        return 1;
      }
    try
      {
        (void)p->ProcessImage (data, width, height, channels, timestamp,
                               std::string (source_id));
        return 0;
      }
    catch (...)
      {
        return 2;
      }
  }

  int
  cortext_consolidate (cortext_handle h, uint64_t now_timestamp)
  {
    auto *p = cast_handle (h);
    if (!p)
      {
        return 1;
      }
    try
      {
        (void)p->Consolidate (now_timestamp);
        return 0;
      }
    catch (...)
      {
        return 2;
      }
  }

  int
  cortext_flush (cortext_handle h)
  {
    auto *p = cast_handle (h);
    if (!p)
      {
        return 1;
      }
    try
      {
        p->Flush ();
        return 0;
      }
    catch (...)
      {
        return 2;
      }
  }

} // extern "C"
