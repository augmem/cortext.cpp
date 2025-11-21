// include/cortext/store/extension_loader.hpp
#pragma once

#include <sqlite3.h>

namespace cortext
{

// Registers any built-in (statically linked) SQLite extensions, if present.
// Safe to call multiple times; performs one-time registration per process.
void RegisterBuiltInExtensions ();

// Attempts to dynamically load supported extensions (sqlite-vec, sqlite-graph)
// from well-known paths or environment overrides. No-op for WASM builds.
// Never throws; failures are silently ignored.
void TryLoadDynamicExtensions (sqlite3 *db);

// Registers built-in extensions (vec/graph/objstore when embedded) on a
// specific sqlite3* connection. Safe to call multiple times; underlying init
// functions are idempotent.
void RegisterBuiltInExtensionsOnDb (sqlite3 *db);

} // namespace cortext
