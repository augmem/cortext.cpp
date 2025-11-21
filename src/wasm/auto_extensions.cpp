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
