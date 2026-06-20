/// @file
/// @brief Browser WebAssembly module anchor.
///
/// The C API functions are provided by src/capi.cpp and exported from the
/// Emscripten link step. This translation unit gives CMake a concrete
/// executable target while the linker uses --no-entry.

#if !defined(__EMSCRIPTEN__)
#error "src/wasm/module.cpp is only for Emscripten builds"
#endif

extern "C" void
cortext_wasm_module_anchor ()
{
}
