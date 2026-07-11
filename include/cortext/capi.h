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

  /// @brief Value types used by external DB provider callbacks.
  typedef enum cortext_db_value_type
  {
    CORTEXT_DB_VALUE_NULL = 0,
    CORTEXT_DB_VALUE_INT64 = 1,
    CORTEXT_DB_VALUE_DOUBLE = 2,
    CORTEXT_DB_VALUE_TEXT = 3,
    CORTEXT_DB_VALUE_BLOB = 4
  } cortext_db_value_type;

  /// @brief Typed SQL value passed between Cortext and external DB providers.
  typedef struct cortext_db_value
  {
    int type;
    int64_t int64_value;
    double double_value;
    const char *text_value;
    const uint8_t *blob_data;
    size_t blob_size;
  } cortext_db_value;

  /// @brief One named result cell returned by external DB providers.
  typedef struct cortext_db_cell
  {
    const char *column_name;
    cortext_db_value value;
  } cortext_db_cell;

  /// @brief One result row returned by external DB providers.
  typedef struct cortext_db_row
  {
    const cortext_db_cell *cells;
    size_t cell_count;
  } cortext_db_row;

  /// @brief Full result set returned by external DB providers.
  typedef struct cortext_db_result
  {
    const cortext_db_row *rows;
    size_t row_count;
  } cortext_db_result;

  typedef int (*cortext_db_execute_fn) (
      void *user_data, void *transaction, const char *query,
      const cortext_db_value *params, size_t param_count,
      cortext_db_result *out_result, const char **out_error);

  typedef int (*cortext_db_begin_fn) (void *user_data,
                                      void *parent_transaction,
                                      void **out_transaction,
                                      const char **out_error);

  typedef int (*cortext_db_transaction_fn) (void *user_data,
                                            void *transaction,
                                            const char **out_error);

  typedef void (*cortext_db_release_transaction_fn) (void *user_data,
                                                     void *transaction);

  typedef void (*cortext_db_release_result_fn) (void *user_data,
                                                cortext_db_result *result);

  /// @brief Synchronous SQL callback table for externally owned DB providers.
  typedef struct cortext_db_callbacks
  {
    size_t struct_size;
    cortext_db_execute_fn execute;
    cortext_db_begin_fn begin;
    cortext_db_transaction_fn commit;
    cortext_db_transaction_fn rollback;
    cortext_db_release_transaction_fn release_transaction;
    cortext_db_release_result_fn release_result;
  } cortext_db_callbacks;

  typedef int (*cortext_object_begin_fn) (void *user_data,
                                          void *parent_transaction,
                                          void **out_transaction,
                                          const char **out_error);

  typedef int (*cortext_object_put_fn) (
      void *user_data, void *transaction, const uint8_t *id, size_t id_size,
      const uint8_t *data, size_t data_size, const char **out_error);

  typedef int (*cortext_object_get_fn) (
      void *user_data, void *transaction, const uint8_t *id, size_t id_size,
      const uint8_t **out_data, size_t *out_data_size,
      const char **out_error);

  typedef int (*cortext_object_exists_fn) (
      void *user_data, void *transaction, const uint8_t *id, size_t id_size,
      int *out_exists, const char **out_error);

  typedef int (*cortext_object_delete_fn) (
      void *user_data, void *transaction, const uint8_t *id, size_t id_size,
      int *out_deleted, const char **out_error);

  typedef int (*cortext_object_transaction_fn) (void *user_data,
                                                void *transaction,
                                                const char **out_error);

  typedef void (*cortext_object_release_transaction_fn) (void *user_data,
                                                        void *transaction);

  typedef void (*cortext_object_release_blob_fn) (void *user_data,
                                                 const uint8_t *data,
                                                 size_t data_size);

  /// @brief Synchronous object storage callback table for external providers.
  typedef struct cortext_object_callbacks
  {
    size_t struct_size;
    cortext_object_begin_fn begin;
    cortext_object_put_fn put;
    cortext_object_get_fn get;
    cortext_object_exists_fn exists;
    cortext_object_delete_fn delete_object;
    cortext_object_transaction_fn commit;
    cortext_object_transaction_fn rollback;
    cortext_object_release_transaction_fn release_transaction;
    cortext_object_release_blob_fn release_blob;
  } cortext_object_callbacks;

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
    int signal_filter_audio_enabled;
    int signal_filter_image_enabled;
    int signal_filter_text_enabled;
  } cortext_config;

  /// @brief Optional source media payload persisted for audio/image signals.
  ///
  /// The canonical process input is still passed through pcm/data arguments.
  /// When this struct carries bytes, those bytes are stored as the durable
  /// signal/memory payload using mimetype. Passing NULL, or a struct with
  /// size == 0, means process without storing an audio/image payload. Media
  /// bytes are copied during the call.
  typedef struct cortext_media
  {
    const uint8_t *data;
    size_t size;
    const char *mimetype;
  } cortext_media;

  /// @brief Retention policy for process ingress (matches cortext::Retention).
  ///
  /// CORTEXT_RETENTION_NATURAL (0): default when omitted; episode algorithms
  /// decide boundary and write acceptance.
  /// CORTEXT_RETENTION_DURABLE (1): force turn boundary + force write.
  /// CORTEXT_RETENTION_BOUNDARY (2): force turn boundary only.
  /// CORTEXT_RETENTION_EPHEMERAL (3): never store; force retrieval boundary.
  typedef enum cortext_retention
  {
    CORTEXT_RETENTION_NATURAL = 0,
    CORTEXT_RETENTION_DURABLE = 1,
    CORTEXT_RETENTION_BOUNDARY = 2,
    CORTEXT_RETENTION_EPHEMERAL = 3
  } cortext_retention;

  /// @brief Options for JSON process result payloads.
  ///
  /// Existing JSON process calls include the processed signal embedding by
  /// default. Pass include_embedding = 0 through the _with_options variants to
  /// omit that vector from the returned JSON payload.
  ///
  /// retention defaults to CORTEXT_RETENTION_NATURAL. Older clients that only
  /// set struct_size + include_embedding keep Natural (false/false forces).
  typedef struct cortext_process_json_options
  {
    size_t struct_size;
    int include_embedding;
    int retention; ///< cortext_retention; Natural when field absent/unknown
    int reserved;  ///< Must be zero; distinguishes this layout from v1.1.10.
  } cortext_process_json_options;

  /// @brief Initializes process JSON options to current defaults.
  ///
  /// Set struct_size to sizeof(cortext_process_json_options) before calling
  /// this function to initialize the retention field. A zero struct_size uses
  /// the legacy prefix for ABI-safe initialization.
  CORTEXT_EXPORT void cortext_process_json_options_init (
      cortext_process_json_options *options);

  /// @brief Creates a Cortext instance.
  /// @param focus Focus knob value in [0.0, 1.0].
  /// @param sensitivity Sensitivity knob value in [0.0, 1.0].
  /// @param stability Stability knob value in [0.0, 1.0].
  /// @param db_path Path to SQLite database file (e.g., "memory.db" or ":memory:").
  /// @return Handle to the created instance, or NULL on failure.
  ///
  /// The returned handle must be freed with cortext_free().
  CORTEXT_EXPORT cortext_handle cortext_create (double focus, double sensitivity,
                                                double stability,
                                                const char *db_path);

  /// @brief Initializes a cortext_config struct with the library defaults.
  CORTEXT_EXPORT void cortext_config_init (cortext_config *cfg);

  /// @brief Creates a Cortext instance from a binding-friendly config struct.
  /// @param cfg Optional configuration. NULL uses the library defaults.
  /// @param db_path Path to SQLite database file.
  CORTEXT_EXPORT cortext_handle
  cortext_create_with_config (const cortext_config *cfg, const char *db_path);

  /// @brief Creates a Cortext instance using an externally owned DB provider.
  /// @param cfg Optional configuration. NULL uses the library defaults.
  /// @param callbacks Required synchronous DB callback table.
  /// @param user_data Opaque pointer passed to every callback.
  /// @return Handle to the created instance, or NULL on failure.
  ///
  /// Cortext does not close the provider or take ownership of user_data. The
  /// provider must stay alive until cortext_free() returns. Until object
  /// storage is extracted into its own interface, providers must support the
  /// same SQL-level blob functions Cortext currently uses for payload storage.
  CORTEXT_EXPORT cortext_handle cortext_create_with_store_callbacks (
      const cortext_config *cfg, const cortext_db_callbacks *callbacks,
      void *user_data);

  /// @brief Creates a Cortext instance using external DB and object providers.
  CORTEXT_EXPORT cortext_handle
  cortext_create_with_store_and_object_callbacks (
      const cortext_config *cfg, const cortext_db_callbacks *db_callbacks,
      void *db_user_data, const cortext_object_callbacks *object_callbacks,
      void *object_user_data);

  /// @brief Creates a Cortext instance with SQLite DB and external object store.
  CORTEXT_EXPORT cortext_handle
  cortext_create_with_config_and_object_callbacks (
      const cortext_config *cfg, const char *db_path,
      const cortext_object_callbacks *object_callbacks, void *object_user_data);

  /// @brief Frees a Cortext instance and releases all resources.
  /// @param h Handle to the instance to free. Safe to pass NULL.
  ///
  /// After calling this function, the handle is invalid and must not be used.
  CORTEXT_EXPORT void cortext_free (cortext_handle h);

  /// @brief Processes text input through the memory system.
  /// @param h Handle to a Cortext instance.
  /// @param text Input text string (must be non-NULL).
  /// @param source_id Opaque source stream identifier for provenance and exact
  /// same-source grouping (must be non-NULL).
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
  /// @param source_id Opaque source stream identifier for provenance and exact
  /// same-source grouping (must be non-NULL).
  /// @return 0 on success, 1 if invalid parameters, 2 on internal error.
  ///
  /// Audio must be 16kHz mono float32 PCM. Timestamps are generated internally.
  /// Any retrieved memories are buffered until cortext_flush().
  CORTEXT_EXPORT int cortext_process_audio (cortext_handle h, const float *pcm,
                                            size_t num_samples,
                                            const char *source_id);

  /// @brief Processes audio while storing caller-provided source media bytes.
  ///
  /// PCM remains 16kHz mono float32 and is used for processing. If media is
  /// non-NULL and media->size > 0, media->data is stored using media->mimetype
  /// instead of storing the canonical PCM buffer.
  CORTEXT_EXPORT int cortext_process_audio_with_media (
      cortext_handle h, const float *pcm, size_t num_samples,
      const char *source_id, const cortext_media *media);

  /// @brief Processes image input through the memory system.
  /// @param h Handle to a Cortext instance.
  /// @param data Raw image data (RGB or RGBA, must be non-NULL).
  /// @param width Image width in pixels.
  /// @param height Image height in pixels.
  /// @param channels Number of channels (3 for RGB, 4 for RGBA).
  /// @param source_id Opaque source stream identifier for provenance and exact
  /// same-source grouping (must be non-NULL).
  /// @return 0 on success, 1 if invalid parameters, 2 on internal error.
  ///
  /// Image data is expected in row-major order (height × width × channels).
  /// Timestamps are generated internally. Any retrieved memories are
  /// buffered until cortext_flush().
  CORTEXT_EXPORT int cortext_process_image (cortext_handle h,
                                            const uint8_t *data, int width,
                                            int height, int channels,
                                            const char *source_id);

  /// @brief Processes image pixels while storing caller-provided source media.
  ///
  /// Pixel data remains the canonical processing input. If media is non-NULL
  /// and media->size > 0, media->data is stored using media->mimetype instead
  /// of storing the canonical raw pixel buffer.
  CORTEXT_EXPORT int cortext_process_image_with_media (
      cortext_handle h, const uint8_t *data, int width, int height,
      int channels, const char *source_id, const cortext_media *media);

  /// @brief Encodes text and returns an embedding JSON object.
  ///
  /// The returned JSON has shape `{ "embedding": number[], "dimension": n }`.
  /// It is an embedding-only call: no storage, retrieval, consolidation, or
  /// processor-state mutation occurs. Free the returned string with
  /// cortext_string_free().
  CORTEXT_EXPORT char *cortext_embed_text_json (cortext_handle h,
                                                const char *text);

  /// @brief Encodes audio PCM and returns an embedding JSON object.
  ///
  /// Audio uses the same public input contract as cortext_process_audio: 16 kHz
  /// mono float32 PCM. Free the returned string with cortext_string_free().
  CORTEXT_EXPORT char *cortext_embed_audio_json (cortext_handle h,
                                                 const float *pcm,
                                                 size_t num_samples);

  /// @brief Encodes image pixels and returns an embedding JSON object.
  ///
  /// Image data uses the same public input contract as cortext_process_image.
  /// Free the returned string with cortext_string_free().
  CORTEXT_EXPORT char *cortext_embed_image_json (cortext_handle h,
                                                 const uint8_t *data,
                                                 int width, int height,
                                                 int channels);

  /// @brief Runs one explicit shallow consolidation pass.
  /// @param h Handle to a Cortext instance.
  /// @return 0 on success, 1 if invalid handle, 2 on internal error.
  ///
  /// This replays stored embeddings through the v1 graph/label consolidation
  /// path. Changes are buffered until cortext_flush().
  CORTEXT_EXPORT int cortext_consolidate (cortext_handle h);

  /// @brief Commits all buffered database writes.
  /// @param h Handle to a Cortext instance.
  /// @return 0 on success, 1 if invalid handle, 2 on internal error.
  ///
  /// This commits the current episode transaction and starts a new episode.
  /// Call after processing a batch of signals or before querying results.
  CORTEXT_EXPORT int cortext_flush (cortext_handle h);

  /// @brief Resets volatile in-memory processor state.
  /// @param h Handle to a Cortext instance.
  /// @return 0 on success, 1 if invalid handle, 2 on internal error.
  ///
  /// Durable memories, signals, embeddings, and object-store content are
  /// retained. The handle remains valid and reloads persisted adaptive state
  /// from the same backing store.
  CORTEXT_EXPORT int cortext_reset (cortext_handle h);

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

  /// @brief Processes text input and returns configurable Context JSON.
  CORTEXT_EXPORT char *cortext_process_text_json_with_options (
      cortext_handle h, const char *text, const char *source_id,
      const cortext_process_json_options *options);

  /// @brief Processes audio input and returns the resulting Context as JSON.
  CORTEXT_EXPORT char *cortext_process_audio_json (cortext_handle h,
                                                   const float *pcm,
                                                   size_t num_samples,
                                                   const char *source_id);

  /// @brief Processes audio input and returns configurable Context JSON.
  CORTEXT_EXPORT char *cortext_process_audio_json_with_options (
      cortext_handle h, const float *pcm, size_t num_samples,
      const char *source_id, const cortext_process_json_options *options);

  /// @brief Processes audio with source media and returns Context JSON.
  CORTEXT_EXPORT char *cortext_process_audio_with_media_json (
      cortext_handle h, const float *pcm, size_t num_samples,
      const char *source_id, const cortext_media *media);

  /// @brief Processes audio with source media and returns configurable Context
  /// JSON.
  CORTEXT_EXPORT char *cortext_process_audio_with_media_json_with_options (
      cortext_handle h, const float *pcm, size_t num_samples,
      const char *source_id, const cortext_media *media,
      const cortext_process_json_options *options);

  /// @brief Processes image input and returns the resulting Context as JSON.
  CORTEXT_EXPORT char *cortext_process_image_json (cortext_handle h,
                                                   const uint8_t *data,
                                                   int width, int height,
                                                   int channels,
                                                   const char *source_id);

  /// @brief Processes image input and returns configurable Context JSON.
  CORTEXT_EXPORT char *cortext_process_image_json_with_options (
      cortext_handle h, const uint8_t *data, int width, int height,
      int channels, const char *source_id,
      const cortext_process_json_options *options);

  /// @brief Processes image pixels with source media and returns Context JSON.
  CORTEXT_EXPORT char *cortext_process_image_with_media_json (
      cortext_handle h, const uint8_t *data, int width, int height,
      int channels, const char *source_id, const cortext_media *media);

  /// @brief Processes image pixels with source media and returns configurable
  /// Context JSON.
  CORTEXT_EXPORT char *cortext_process_image_with_media_json_with_options (
      cortext_handle h, const uint8_t *data, int width, int height,
      int channels, const char *source_id, const cortext_media *media,
      const cortext_process_json_options *options);

  /// @brief Runs shallow consolidation and returns the resulting Context as JSON.
  CORTEXT_EXPORT char *cortext_consolidate_json (cortext_handle h);

#ifdef __cplusplus
}
#endif
