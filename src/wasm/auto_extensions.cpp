/// @file
/// @brief Automatic SQLite extension registration for WASM builds.
///
/// This module uses the __attribute__((constructor)) feature (supported by
/// Emscripten) to register statically embedded SQLite extensions (vec, graph)
/// once at module initialization time. The constructor attribute ensures that
/// sqlite3_initialize() and sqlite3_auto_extension() are called before any
/// user code runs, avoiding the need for manual initialization calls.
///
/// This approach is safe for WASM and matches the global-state guidance in
/// cpp-rules: the registration is idempotent and happens exactly once per
/// WebAssembly module instance.
// src/wasm/auto_extensions.cpp
#include <sqlite3.h>

extern "C"
{
#if defined(CORTEXT_EMBED_VEC)
  int sqlite3_vec_init (sqlite3 *, char **, const sqlite3_api_routines *);
#endif
#if defined(CORTEXT_EMBED_GRAPH)
  int sqlite3_graph_init (sqlite3 *, char **, const sqlite3_api_routines *);
#endif
}

#if defined(__EMSCRIPTEN__)
__attribute__ ((constructor)) static void
cortext_wasm_register_extensions ()
{
  sqlite3_initialize ();
#if defined(CORTEXT_EMBED_VEC)
  sqlite3_auto_extension ((void (*) (void))sqlite3_vec_init);
#endif
#if defined(CORTEXT_EMBED_GRAPH)
  sqlite3_auto_extension ((void (*) (void))sqlite3_graph_init);
#endif
}
#endif
