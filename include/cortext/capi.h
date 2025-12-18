/// @file
/// @brief C API for the Cortext multimodal memory system.
///
/// This API provides a C-compatible interface for creating and managing
/// Cortext instances. All functions are thread-safe with respect to different
/// handles, but a single handle must not be used concurrently from multiple
/// threads without external synchronization.
///
/// Example usage:
/// @code{.c}
///   cortext_handle ctx = cortext_create(0.5, 0.5, 0.5, "memory.db");
///   if (!ctx) {
///     // Handle creation failure
///     return 1;
///   }
///   int result = cortext_process_text(ctx, "Hello", "user");
///   if (result != 0) {
///     // Handle processing error
///   }
///   cortext_flush(ctx);
///   cortext_free(ctx);
/// @endcode
#pragma once
#include <stddef.h>
#include <stdint.h>

// Define CORTEXT_EXPORT for C code (C++ uses cortext/export.hpp)
#ifndef CORTEXT_EXPORT
#if defined(_WIN32)
#if defined(CORTEXT_BUILDING_SHARED)
#define CORTEXT_EXPORT __declspec(dllexport)
#elif defined(CORTEXT_USING_SHARED)
#define CORTEXT_EXPORT __declspec(dllimport)
#else
#define CORTEXT_EXPORT
#endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#define CORTEXT_EXPORT __attribute__((visibility("default")))
#else
#define CORTEXT_EXPORT
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /// @brief Opaque handle to a Cortext instance.
  ///
  /// This handle must be freed with cortext_free() when no longer needed.
  typedef void *cortext_handle;

  /// @brief Creates a Cortext instance with default models directory.
  /// @param focus Focus knob value in [0.0, 1.0].
  /// @param sensitivity Sensitivity knob value in [0.0, 1.0].
  /// @param stability Stability knob value in [0.0, 1.0].
  /// @param db_path Path to SQLite database file (e.g., "memory.db" or ":memory:").
  /// @return Handle to the created instance, or NULL on failure.
  ///
  /// The returned handle must be freed with cortext_free().
  /// Default models directory is "models/imagebind".
  CORTEXT_EXPORT cortext_handle cortext_create (double focus, double sensitivity,
                                                double stability,
                                                const char *db_path);

  /// @brief Creates a Cortext instance with custom models directory.
  /// @param focus Focus knob value in [0.0, 1.0].
  /// @param sensitivity Sensitivity knob value in [0.0, 1.0].
  /// @param stability Stability knob value in [0.0, 1.0].
  /// @param db_path Path to SQLite database file (e.g., "memory.db" or ":memory:").
  /// @param models_dir Path to directory containing ImageBind ONNX models.
  /// @return Handle to the created instance, or NULL on failure.
  ///
  /// The returned handle must be freed with cortext_free().
  CORTEXT_EXPORT cortext_handle
  cortext_create_with_models (double focus, double sensitivity, double stability,
                              const char *db_path, const char *models_dir);

  /// @brief Frees a Cortext instance and releases all resources.
  /// @param h Handle to the instance to free. Safe to pass NULL.
  ///
  /// After calling this function, the handle is invalid and must not be used.
  CORTEXT_EXPORT void cortext_free (cortext_handle h);

  /// @brief Processes text input through the memory system.
  /// @param h Handle to a Cortext instance.
  /// @param text Input text string (must be non-NULL).
  /// @param source_id Source identifier string (must be non-NULL).
  /// @return 0 on success, 1 if invalid parameters, 2 on internal error.
  ///
  /// This function encodes the text and processes it through the signal
  /// pipeline. Timestamps are generated internally in milliseconds.
  /// Any retrieved memories are buffered until cortext_flush().
  CORTEXT_EXPORT int cortext_process_text (cortext_handle h, const char *text,
                                           const char *source_id);

  /// @brief Processes audio input through the memory system.
  /// @param h Handle to a Cortext instance.
  /// @param pcm PCM audio samples (float32, must be non-NULL).
  /// @param num_samples Number of PCM samples.
  /// @param source_id Source identifier string (must be non-NULL).
  /// @return 0 on success, 1 if invalid parameters, 2 on internal error.
  ///
  /// Audio must be 16kHz mono float32 PCM. Timestamps are generated internally.
  /// Any retrieved memories are buffered until cortext_flush().
  CORTEXT_EXPORT int cortext_process_audio (cortext_handle h, const float *pcm,
                                            size_t num_samples,
                                            const char *source_id);

  /// @brief Processes image input through the memory system.
  /// @param h Handle to a Cortext instance.
  /// @param data Raw image data (RGB or RGBA, must be non-NULL).
  /// @param width Image width in pixels.
  /// @param height Image height in pixels.
  /// @param channels Number of channels (3 for RGB, 4 for RGBA).
  /// @param source_id Source identifier string (must be non-NULL).
  /// @return 0 on success, 1 if invalid parameters, 2 on internal error.
  ///
  /// Image data is expected in row-major order (height × width × channels).
  /// Timestamps are generated internally. Any retrieved memories are
  /// buffered until cortext_flush().
  CORTEXT_EXPORT int cortext_process_image (cortext_handle h,
                                            const uint8_t *data, int width,
                                            int height, int channels,
                                            const char *source_id);

  /// @brief Triggers consolidation evaluation.
  /// @param h Handle to a Cortext instance.
  /// @return 0 on success, 1 if invalid handle, 2 on internal error.
  ///
  /// This evaluates whether background consolidation should start based on
  /// system conditions. Changes are buffered until cortext_flush().
  CORTEXT_EXPORT int cortext_consolidate (cortext_handle h);

  /// @brief Commits all buffered database writes.
  /// @param h Handle to a Cortext instance.
  /// @return 0 on success, 1 if invalid handle, 2 on internal error.
  ///
  /// This commits the current episode transaction and starts a new episode.
  /// Call after processing a batch of signals or before querying results.
  CORTEXT_EXPORT int cortext_flush (cortext_handle h);

#ifdef __cplusplus
}
#endif
