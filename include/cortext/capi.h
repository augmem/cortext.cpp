#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef void *cortext_handle;

  cortext_handle cortext_create (double focus, double sensitivity,
                                 double stability, const char *db_path);

  cortext_handle cortext_create_with_models (double focus, double sensitivity,
                                             double stability,
                                             const char *db_path,
                                             const char *models_dir);

  void cortext_free (cortext_handle h);

  int cortext_process_text (cortext_handle h, const char *text,
                            uint64_t timestamp, const char *source_id);

  int cortext_process_audio (cortext_handle h, const float *pcm,
                             size_t num_samples, uint64_t timestamp,
                             const char *source_id);

  int cortext_process_image (cortext_handle h, const uint8_t *data, int width,
                             int height, int channels, uint64_t timestamp,
                             const char *source_id);

  int cortext_consolidate (cortext_handle h, uint64_t now_timestamp);

  int cortext_flush (cortext_handle h);

#ifdef __cplusplus
}
#endif
