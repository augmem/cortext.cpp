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

extern "C" char *
cortext_wasm_process_text_json (cortext_handle handle, const char *text,
                                const char *source_id, int include_embedding,
                                int retention)
{
  cortext_process_json_options options{};
  options.struct_size = sizeof (options);
  cortext_process_json_options_init (&options);
  options.include_embedding = include_embedding != 0 ? 1 : 0;
  options.retention = retention;
  return cortext_process_text_json_with_options (handle, text, source_id,
                                                 &options);
}
