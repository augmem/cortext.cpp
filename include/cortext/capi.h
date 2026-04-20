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

  /// @brief Binding-friendly configuration for cortext_create_with_config.
  typedef struct cortext_config
  {
    size_t struct_size;
    double focus;
    double sensitivity;
    double stability;
    int affect_interrupt;
    int affect_retrieval;
    int reinforcement_enabled;
    int procedural_enabled;
    int sequential_edges_enabled;
    const char *label_bank_path;
  } cortext_config;

  /// @brief Consolidation mode for cortext_consolidate_mode.
  typedef enum cortext_consolidation_mode
  {
    CORTEXT_CONSOLIDATE_SHALLOW = 0,
    CORTEXT_CONSOLIDATE_DEEP = 1,
    CORTEXT_CONSOLIDATE_BOTH = 2
  } cortext_consolidation_mode;

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

  /// @brief Initializes a cortext_config struct with the library defaults.
  CORTEXT_EXPORT void cortext_config_init (cortext_config *cfg);

  /// @brief Creates a Cortext instance from a binding-friendly config struct.
  /// @param cfg Optional configuration. NULL uses the library defaults.
  /// @param db_path Path to SQLite database file.
  /// @param models_dir Optional path to model assets. NULL uses the default
  ///   model root.
  CORTEXT_EXPORT cortext_handle
  cortext_create_with_config (const cortext_config *cfg, const char *db_path,
                              const char *models_dir);

  /// @brief Creates a Cortext instance with custom models directory.
  /// @param focus Focus knob value in [0.0, 1.0].
  /// @param sensitivity Sensitivity knob value in [0.0, 1.0].
  /// @param stability Stability knob value in [0.0, 1.0].
  /// @param db_path Path to SQLite database file (e.g., "memory.db" or ":memory:").
  /// @param models_dir Path to the local model root directory.
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

  /// @brief Triggers consolidation with explicit mode.
  /// @param h Handle to a Cortext instance.
  /// @param mode Consolidation mode (shallow, deep, or both).
  /// @return 0 on success, 1 if invalid handle, 2 on internal error.
  ///
  /// Shallow runs embedding-only labeling/graphing; deep runs the configured
  /// local summarization/extraction backend (Gemma/LiteRT-LM, LFM2/llama.cpp,
  /// or the mixed Gemma+LFM2 path). Both defaults to the full deep path.
  CORTEXT_EXPORT int cortext_consolidate_mode (cortext_handle h, int mode);

  /// @brief Commits all buffered database writes.
  /// @param h Handle to a Cortext instance.
  /// @return 0 on success, 1 if invalid handle, 2 on internal error.
  ///
  /// This commits the current episode transaction and starts a new episode.
  /// Call after processing a batch of signals or before querying results.
  CORTEXT_EXPORT int cortext_flush (cortext_handle h);

  /// @brief Returns the Cortext library version string.
  CORTEXT_EXPORT const char *cortext_version (void);

  /// @brief Returns the last error produced by this thread's C API calls.
  CORTEXT_EXPORT const char *cortext_last_error (void);

  /// @brief Frees strings returned by JSON-returning C API helpers.
  CORTEXT_EXPORT void cortext_string_free (char *value);

  /// @brief Processes text input and returns the resulting Context as JSON.
  CORTEXT_EXPORT char *cortext_process_text_json (cortext_handle h,
                                                  const char *text,
                                                  const char *source_id);

  /// @brief Processes audio input and returns the resulting Context as JSON.
  CORTEXT_EXPORT char *cortext_process_audio_json (cortext_handle h,
                                                   const float *pcm,
                                                   size_t num_samples,
                                                   const char *source_id);

  /// @brief Processes image input and returns the resulting Context as JSON.
  CORTEXT_EXPORT char *cortext_process_image_json (cortext_handle h,
                                                   const uint8_t *data,
                                                   int width, int height,
                                                   int channels,
                                                   const char *source_id);

  /// @brief Triggers consolidation and returns the resulting Context as JSON.
  CORTEXT_EXPORT char *cortext_consolidate_json (cortext_handle h);

  /// @brief Triggers consolidation with explicit mode and returns JSON.
  CORTEXT_EXPORT char *cortext_consolidate_mode_json (cortext_handle h,
                                                      int mode);

#ifdef __cplusplus
}
#endif
