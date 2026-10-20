/// @file
/// @brief Browser WebAssembly module anchor.
///
/// The C API functions are provided by src/capi.cpp and exported from the
/// Emscripten link step. This translation unit gives CMake a concrete
/// executable target while the linker uses --no-entry.

#if !defined(__EMSCRIPTEN__)
#error "src/wasm/module.cpp is only for Emscripten builds"
#endif

#include "cortext/capi.h"

extern "C" void
cortext_wasm_module_anchor ()
{
}

/// evidence_budget_bytes: 0 = unset (whole memories); see
/// cortext_process_json_options.evidence_budget_bytes.
/// has_reference_ms / reference_ms / as_of: the query reference time
/// (cortext_process_json_options); has_reference_ms = 0 (or omitted by an
/// older caller) leaves it unset. reference_ms is a double so JavaScript can
/// pass a millisecond timestamp as a plain number.
extern "C" char *
cortext_wasm_process_text_json (cortext_handle handle, const char *text,
                                const char *source_id, int include_embedding,
                                int retention,
                                unsigned int evidence_budget_bytes,
                                int has_reference_ms, double reference_ms,
                                int as_of)
{
  cortext_process_json_options options{};
  options.struct_size = sizeof (options);
  cortext_process_json_options_init (&options);
  options.include_embedding = include_embedding != 0 ? 1 : 0;
  options.retention = retention;
  options.evidence_budget_bytes = evidence_budget_bytes;
  if (has_reference_ms != 0 && reference_ms >= 0.0
      && reference_ms <= 9007199254740991.0)
    {
      options.has_reference_ms = 1;
      options.reference_ms = static_cast<uint64_t> (reference_ms);
      options.as_of = as_of != 0 ? 1 : 0;
    }
  return cortext_process_text_json_with_options (handle, text, source_id,
                                                 &options);
}
