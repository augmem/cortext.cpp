/// @file
/// @brief Browser WebAssembly module anchor and JS-friendly process wrappers.
///
/// The C API functions are provided by src/capi.cpp and exported from the
/// Emscripten link step. This translation unit gives CMake a concrete
/// executable target while the linker uses --no-entry, and adds
/// cortext_wasm_process_*_json wrappers that take the per-call process options
/// as scalar arguments, so JavaScript never writes the
/// cortext_process_json_options struct layout itself.
///
/// Shared trailing arguments of every wrapper:
/// - include_embedding: nonzero keeps the signal embedding in the JSON.
/// - retention: cortext_retention value (0 Natural .. 3 Ephemeral).
/// - evidence_budget_bytes: 0 = unset (whole memories); see
///   cortext_process_json_options.evidence_budget_bytes.
/// - has_reference_ms / reference_ms / as_of: the query reference time
///   (cortext_process_json_options); has_reference_ms = 0 (or omitted by an
///   older caller) leaves it unset. reference_ms is a double so JavaScript can
///   pass a millisecond timestamp as a plain number; a negative, non-integral
///   or above-2^53 value leaves it unset.
///
/// Media arguments (media_data, media_size, media_mimetype): size 0 stores
/// the canonical input, exactly like the C API without media; non-zero size
/// needs non-NULL data and a non-empty MIME type (cortext_media).

#if !defined(__EMSCRIPTEN__)
#error "src/wasm/module.cpp is only for Emscripten builds"
#endif

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "cortext/capi.h"

namespace
{

cortext_process_json_options
MakeOptions (int include_embedding, int retention,
             unsigned int evidence_budget_bytes, int has_reference_ms,
             double reference_ms, int as_of)
{
  cortext_process_json_options options{};
  options.struct_size = sizeof (options);
  cortext_process_json_options_init (&options);
  options.include_embedding = include_embedding != 0 ? 1 : 0;
  options.retention = retention;
  options.evidence_budget_bytes = evidence_budget_bytes;
  if (has_reference_ms != 0 && reference_ms >= 0.0
      && reference_ms <= 9007199254740991.0
      && std::floor (reference_ms) == reference_ms)
    {
      options.has_reference_ms = 1;
      options.reference_ms = static_cast<uint64_t> (reference_ms);
      options.as_of = as_of != 0 ? 1 : 0;
    }
  return options;
}

cortext_media
MakeMedia (const uint8_t *data, unsigned int size, const char *mimetype)
{
  cortext_media media{};
  media.data = data;
  media.size = size;
  media.mimetype = mimetype;
  return media;
}

} // namespace

extern "C" void
cortext_wasm_module_anchor ()
{
}

extern "C" char *
cortext_wasm_process_text_json (cortext_handle handle, const char *text,
                                const char *source_id, int include_embedding,
                                int retention,
                                unsigned int evidence_budget_bytes,
                                int has_reference_ms, double reference_ms,
                                int as_of)
{
  const auto options
      = MakeOptions (include_embedding, retention, evidence_budget_bytes,
                     has_reference_ms, reference_ms, as_of);
  return cortext_process_text_json_with_options (handle, text, source_id,
                                                 &options);
}

/// Text with caller media: the text is embedded; non-empty media bytes are
/// stored as the payload with media_mimetype (text only when text/*).
extern "C" char *
cortext_wasm_process_text_with_media_json (
    cortext_handle handle, const char *text, const char *source_id,
    const uint8_t *media_data, unsigned int media_size,
    const char *media_mimetype, int include_embedding, int retention,
    unsigned int evidence_budget_bytes, int has_reference_ms,
    double reference_ms, int as_of)
{
  const auto options
      = MakeOptions (include_embedding, retention, evidence_budget_bytes,
                     has_reference_ms, reference_ms, as_of);
  const auto media = MakeMedia (media_data, media_size, media_mimetype);
  return cortext_process_text_with_media_json_with_options (
      handle, text, source_id, &media, &options);
}

/// Audio: 16 kHz mono float32 PCM, optional caller media.
extern "C" char *
cortext_wasm_process_audio_json (
    cortext_handle handle, const float *pcm, unsigned int num_samples,
    const char *source_id, const uint8_t *media_data,
    unsigned int media_size, const char *media_mimetype,
    int include_embedding, int retention, unsigned int evidence_budget_bytes,
    int has_reference_ms, double reference_ms, int as_of)
{
  const auto options
      = MakeOptions (include_embedding, retention, evidence_budget_bytes,
                     has_reference_ms, reference_ms, as_of);
  const auto media = MakeMedia (media_data, media_size, media_mimetype);
  return cortext_process_audio_with_media_json_with_options (
      handle, pcm, num_samples, source_id, &media, &options);
}

/// Image: row-major pixels with explicit width/height/channels, optional
/// caller media.
extern "C" char *
cortext_wasm_process_image_json (
    cortext_handle handle, const uint8_t *data, int width, int height,
    int channels, const char *source_id, const uint8_t *media_data,
    unsigned int media_size, const char *media_mimetype,
    int include_embedding, int retention, unsigned int evidence_budget_bytes,
    int has_reference_ms, double reference_ms, int as_of)
{
  const auto options
      = MakeOptions (include_embedding, retention, evidence_budget_bytes,
                     has_reference_ms, reference_ms, as_of);
  const auto media = MakeMedia (media_data, media_size, media_mimetype);
  return cortext_process_image_with_media_json_with_options (
      handle, data, width, height, channels, source_id, &media, &options);
}
