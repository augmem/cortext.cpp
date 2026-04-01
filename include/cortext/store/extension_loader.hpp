/// @file
/// @brief SQLite extension loader for vec and objstore modules.
#pragma once

#include <sqlite3.h>

namespace cortext
{

/// @brief Registers statically linked SQLite extensions globally (process-wide).
///
/// This function is idempotent and safe to call multiple times. On non-Apple
/// platforms, it uses sqlite3_auto_extension() to register built-in
/// extensions so they are automatically loaded on new connections. On Apple
/// platforms, sqlite3_auto_extension() is deprecated; use
/// RegisterBuiltInExtensionsOnDb() per-connection instead.
///
/// Safe for WASM builds (no-op when extensions are unavailable).
void RegisterBuiltInExtensions ();

/// @brief Attempts to dynamically load SQLite extensions from well-known paths.
/// @param db SQLite connection handle (must be non-NULL).
///
/// This function tries to load sqlite-vec from standard locations or the
/// SQLITE_VEC_PATH environment variable. Always a no-op in WASM builds.
/// Never throws; failures are silently ignored to allow graceful degradation
/// when extensions are unavailable.
void TryLoadDynamicExtensions (sqlite3 *db);

/// @brief Registers built-in extensions on a specific SQLite connection.
/// @param db SQLite connection handle (must be non-NULL).
///
/// This function calls the init functions for statically embedded extensions
/// (vec, objstore) on the given connection. Safe to call multiple times;
/// underlying init functions are idempotent. Required on Apple platforms where
/// sqlite3_auto_extension() is unavailable.
void RegisterBuiltInExtensionsOnDb (sqlite3 *db);

} // namespace cortext
