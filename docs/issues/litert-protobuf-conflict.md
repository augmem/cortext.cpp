# Issue: LiteRT-LM and OpenTelemetry OTLP Protobuf Symbol Conflict

## Summary

Cortext users cannot use OpenTelemetry OTLP exporters (gRPC or HTTP) when LiteRT-LM is enabled. The application crashes with a bus error during protobuf static initialization due to symbol conflicts between LiteRT's bundled protobuf and system protobuf.

## Symptoms

```
$ ./cortext_chat
zsh: bus error ./cortext_chat
```

Crash occurs in `google::protobuf::internal::ThreadSafeArena::GetSerialArenaFallback` before `main()` is reached.

## Root Cause

**Two incompatible protobuf implementations are loaded into the same process:**

1. **LiteRT-LM** bundles protobuf ~v5.x statically inside `liblitert_lm_cpp.dylib`
   - 3,324 protobuf symbols exported (verified via `nm -gU`)
   - Symbols like `google::protobuf::FileDescriptorTables`, `google::protobuf::internal::ShutdownData`

2. **OpenTelemetry OTLP** links against system protobuf v29.x (homebrew)
   - Uses protobuf for OTLP wire format serialization
   - Both gRPC and HTTP exporters depend on `opentelemetry_proto` which requires protobuf

When the dynamic linker loads both libraries, protobuf's global state (registries, arenas, thread-local storage) gets corrupted because two different implementations are fighting over the same symbols.

## Why Simple Fixes Don't Work

| Approach | Why It Fails |
|----------|--------------|
| **Hide symbols post-build** | `llvm-objcopy --localize-symbols` doesn't support Mach-O. `nmedit` requires listing symbols to KEEP (inverted logic). Fragile across versions. |
| **Use OTLP HTTP with JSON** | OTLP HTTP still links `opentelemetry_proto` internally for message structures, even when using JSON wire format. |
| **Disable OTLP** | Not acceptable - users need observability with standard backends (Uptrace, Jaeger, etc.) |
| **Disable LiteRT** | Not acceptable - needed for on-device memory consolidation |

## Proposed Solution: Dynamic Loading with RTLD_LOCAL

Load LiteRT-LM at runtime using `dlopen()` with `RTLD_LOCAL` flag, which keeps its symbols isolated from the global namespace.

```cpp
// Symbols stay private to LiteRT - no conflicts with system protobuf
void* handle = dlopen("liblitert_lm_cpp.dylib", RTLD_LOCAL | RTLD_NOW);
```

### Why This Works

- `RTLD_LOCAL` prevents LiteRT's symbols from being visible to other libraries
- System protobuf (used by OTLP) loads normally with its own symbols
- Each library uses its own protobuf implementation without conflicts
- Standard pattern used by plugin systems everywhere

### Implementation Plan

1. **Create `LiteRTLoader` class** (`src/litert/loader.hpp`)
   - Handles `dlopen`/`dlsym` lifecycle
   - Provides type-safe wrappers for LiteRT C++ API
   - Lazy loading on first use

2. **Define function pointer types** for LiteRT API surface we use:
   - `litert::lm::Engine` creation/destruction
   - `litert::lm::Session` management
   - Generation/inference methods

3. **Update CMake** to not link `litert_lm` directly
   - Remove `target_link_libraries(cortext PUBLIC litert_lm)`
   - Keep include paths for headers
   - Set RPATH so library can be found at runtime

4. **Update cortext wrapper** (`src/extractor/gemma_extractor.cpp`, `src/summarizer/gemma_summarizer.cpp`)
   - Use `LiteRTLoader::Instance()` instead of direct calls
   - Graceful fallback if library not found

### API Sketch

```cpp
// include/cortext/litert/loader.hpp
namespace cortext::litert {

class LiteRTLoader {
public:
  static LiteRTLoader& Instance();

  // Returns false if library not found or incompatible
  bool Load(const std::string& library_path = "");
  bool IsLoaded() const;
  void Unload();

  // Wrapped LiteRT API
  std::unique_ptr<Engine> CreateEngine(const EngineConfig& config);

private:
  void* handle_ = nullptr;

  // Function pointers populated by Load()
  void* (*create_engine_)(const void*) = nullptr;
  void (*destroy_engine_)(void*) = nullptr;
  // ... etc
};

} // namespace cortext::litert
```

### Platform Support

| Platform | Dynamic Loading | Notes |
|----------|-----------------|-------|
| macOS | `dlopen`/`dlsym` | Full support |
| Linux | `dlopen`/`dlsym` | Full support |
| Windows | `LoadLibrary`/`GetProcAddress` | Needs wrapper |
| WASM | N/A | LiteRT already disabled for WASM |

## Alternatives Considered

### 1. Rebuild LiteRT with System Protobuf

Modify LiteRT's Bazel WORKSPACE to use external protobuf matching system version.

**Rejected because:**
- Complex Bazel changes
- Breaks with upstream LiteRT updates
- Version pinning nightmare (must match homebrew/system exactly)

### 2. Process Isolation

Run LiteRT in a separate subprocess, communicate via IPC.

**Deferred because:**
- Higher implementation complexity
- IPC overhead for latency-sensitive operations
- Could revisit if RTLD_LOCAL has issues

### 3. Namespace Protobuf in LiteRT

Rebuild LiteRT with protobuf in a different namespace (`litert::protobuf`).

**Rejected because:**
- Requires forking LiteRT
- Massive build system changes
- Unmaintainable long-term

### 4. Static Linking with Symbol Hiding

Link LiteRT statically, use `-unexported_symbols_list` at final link.

**Rejected because:**
- LiteRT builds as shared library only (Bazel config)
- Would require significant Bazel modifications

## Testing Plan

1. **Unit test**: `LiteRTLoader` loads/unloads correctly
2. **Integration test**: LiteRT inference works through loader
3. **Conflict test**: OTLP gRPC exports while LiteRT is active
4. **Stress test**: Concurrent OTLP export + LiteRT generation

## Migration Path

1. Implement `LiteRTLoader` behind feature flag
2. Test with existing LiteRT-dependent code
3. Remove direct linking, enable by default
4. Update documentation

## References

- [dlopen(3) man page](https://man7.org/linux/man-pages/man3/dlopen.3.html)
- [RTLD_LOCAL behavior](https://stackoverflow.com/questions/34073051/when-we-are-supposed-to-use-rtld-local)
- [OpenTelemetry C++ OTLP](https://github.com/open-telemetry/opentelemetry-cpp/tree/main/exporters/otlp)
- [LiteRT-LM repository](https://github.com/google-ai-edge/LiteRT)
