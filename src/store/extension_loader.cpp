// src/store/extension_loader.cpp
#include "cortext/store/extension_loader.hpp"

#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#if defined(CORTEXT_EMBED_OBJSTORE)
#include "objstore/objstore.h"
#endif

namespace cortext
{

extern "C"
{
#if defined(CORTEXT_EMBED_VEC)
  int sqlite3_vec_init (sqlite3 *, char **, const sqlite3_api_routines *);
#endif
}

/// @brief Registers statically linked SQLite extensions for process-wide use.
///
/// Platform-specific behavior:
/// - **Apple platforms**: sqlite3_auto_extension() is deprecated, so this only
///   initializes SQLite. Actual extension registration happens per-connection
///   via RegisterBuiltInExtensionsOnDb().
/// - **Other platforms**: Uses sqlite3_auto_extension() to register built-in
///   extensions globally so all new connections inherit them.
/// - **WASM**: Handled separately in src/wasm/auto_extensions.cpp via constructor
///   attribute.
///
/// Thread-safety: Uses std::call_once to ensure one-time initialization.
void
RegisterBuiltInExtensions ()
{
  static std::once_flag once;
  std::call_once (once, [] () {
#if defined(__APPLE__)
    // sqlite3_auto_extension is deprecated on Apple platforms; rely on
    // per-connection registration via RegisterBuiltInExtensionsOnDb instead.
    sqlite3_initialize ();
#else
#if defined(CORTEXT_EMBED_VEC)
    sqlite3_auto_extension ((void (*) (void))sqlite3_vec_init);
#endif
#endif
  });
}

static void
enableLoadExtension (sqlite3 *db, int onOff)
{
  if (!db)
    return;
  // Prefer sqlite3_db_config per SQLite docs
  sqlite3_db_config (db, SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, onOff,
                     nullptr);
}

static void
tryLoad (sqlite3 *db, const std::string &path, const char *entry)
{
  if (!db || path.empty ())
    return;
  // Fallback to SQL function to avoid relying on sqlite3_load_extension symbol
  // presence. Ignore errors; extension loading may be disabled on some
  // platforms.
  sqlite3_stmt *stmt = nullptr;
  std::string sql = "SELECT load_extension(?1";
  if (entry && *entry)
    sql += ", ?2";
  sql += ")";
  if (sqlite3_prepare_v2 (db, sql.c_str (), -1, &stmt, nullptr) == SQLITE_OK)
    {
      sqlite3_bind_text (stmt, 1, path.c_str (), -1, SQLITE_TRANSIENT);
      if (entry && *entry)
        {
          sqlite3_bind_text (stmt, 2, entry, -1, SQLITE_TRANSIENT);
        }
      (void)sqlite3_step (stmt);
    }
  if (stmt)
    sqlite3_finalize (stmt);
}

void
TryLoadDynamicExtensions (sqlite3 *db)
{
#if defined(__EMSCRIPTEN__)
  (void)db;
  return;
#else
#if !defined(CORTEXT_ENABLE_DYNAMIC_EXTENSIONS)
  (void)db;
  return;
#else
  if (!db)
    return;

  enableLoadExtension (db, 1);

  // Environment overrides
  const char *vecEnv = std::getenv ("SQLITE_VEC_PATH");
  if (vecEnv && *vecEnv)
    {
      tryLoad (db, std::string (vecEnv), "sqlite3_vec_init");
    }
  else
    {
      // Common relative locations
#if defined(__APPLE__)
      const std::vector<std::string> vecCandidates = {
        "third_party/sqlite-vec/dist/vec0.dylib",
        "deps/sqlite-vec/vec0.dylib",
      };
#elif defined(_WIN32)
      const std::vector<std::string> vecCandidates = {
        "third_party/sqlite-vec/dist/vec0.dll",
        "deps/sqlite-vec/vec0.dll",
      };
#else
      const std::vector<std::string> vecCandidates = {
        "third_party/sqlite-vec/dist/vec0.so",
        "deps/sqlite-vec/vec0.so",
      };
#endif
      for (const auto &p : vecCandidates)
        {
          tryLoad (db, p, "sqlite3_vec_init");
        }
    }

  enableLoadExtension (db, 0);
#endif // CORTEXT_ENABLE_DYNAMIC_EXTENSIONS
#endif // __EMSCRIPTEN__
}

void
RegisterBuiltInExtensionsOnDb (sqlite3 *db)
{
  if (!db)
    return;
#if defined(CORTEXT_EMBED_VEC)
  (void)sqlite3_vec_init (db, nullptr, nullptr);
#endif
#if defined(CORTEXT_EMBED_OBJSTORE)
  objstore_config cfg{};
  cfg.backend = OBJSTORE_BACKEND_AUTO;
  (void)objstore_register (db, &cfg);
#endif
}

} // namespace cortext
