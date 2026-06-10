# Bindings

The repository now ships thin wrappers over the C ABI for:

- `bindings/python`: Python via `ctypes`
- `bindings/go`: Go via `cgo`
- `bindings/javascript`: Node.js plus TypeScript declarations
- `bindings/dart`: Dart via `dart:ffi`

All four can load the shared library from the Zig output directory:

```bash
zig build -Dshared=true -Dllama=false
```

The existing repository-local CMake FFI preset remains supported:

```bash
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext
```

For the Node addon, use the Node-enabled variant:

```bash
cmake --preset ffi-release-node
cmake --build --preset ffi-release-node --target cortext_node
```

## Inference providers

The config now takes optional `summarizer_provider_uri` / `extractor_provider_uri`
strings to route the Summarizer/Extractor roles to an external inference
provider (e.g. an Ollama server). Unset/empty keeps local model
auto-discovery; a URI that cannot be resolved and verified fails creation.

```python
cortext.Cortext(config=cortext.Config(
    summarizer_provider_uri="ollama://127.0.0.1:11435/gemma4:e2b",
    extractor_provider_uri="ollama://127.0.0.1:11435/gemma4:e2b"))
```

```go
cortext.New(":memory:", "", &cortext.Config{
	SummarizerProviderURI: "ollama://127.0.0.1:11435/gemma4:e2b",
	ExtractorProviderURI:  "ollama://127.0.0.1:11435/gemma4:e2b"})
```

```js
new Cortext({
  summarizerProviderUri: "ollama://127.0.0.1:11435/gemma4:e2b",
  extractorProviderUri: "ollama://127.0.0.1:11435/gemma4:e2b",
});
```

```dart
Cortext(config: Config(
  summarizerProviderUri: 'ollama://127.0.0.1:11435/gemma4:e2b',
  extractorProviderUri: 'ollama://127.0.0.1:11435/gemma4:e2b'));
```
