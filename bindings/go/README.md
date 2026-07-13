# cortext Go Binding

The Go package uses `cgo` over the Cortext C ABI.

## Build

From the repository root:

```bash
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext
```

Then:

```bash
cd bindings/go
go test .
```

The package links against `build/ffi-release/libcortext.*` or
`zig-out/lib/libcortext.*`.

## Use

```go
package main

import (
	"fmt"

	"github.com/augmem/cortext/bindings/go"
)

func main() {
	engine, err := cortext.New(":memory:", nil)
	if err != nil {
		panic(err)
	}
	defer engine.Close()

	ctx, err := engine.ProcessText("Bailey likes tennis balls.", "chat/main")
	if err != nil {
		panic(err)
	}
	fmt.Println(ctx["should_interrupt"])

	embedding, err := engine.EmbedText("embed without storing")
	if err != nil {
		panic(err)
	}
	fmt.Println(len(embedding))
}
```

## API

<!-- public Go binding API and Config semantics matching bindings/go/cortext.go -->

- `Config`: Focus/Sensitivity/Stability, mechanism toggles, and signal-filter
  toggles. Fields are optional pointers, so omitted values retain native
  defaults while explicit zero/false values remain representable. Use
  `cortext.Ptr(value)` in struct literals, or pass `nil` to `New` for all
  native defaults.
- `ProcessText`, `ProcessAudio`, `ProcessImage`: decoded JSON context maps.
- `ProcessTextJSON`, `ProcessAudioJSON`, `ProcessImageJSON`: raw JSON bytes.
- `EmbedText`, `EmbedAudio`, `EmbedImage`: embed-only helpers.
- `Consolidate`, `Flush`, `Reset`, `Close`.

Audio input is 16 kHz mono float32 PCM. Image input is row-major RGB/RGBA bytes
with explicit dimensions; undersized buffers are rejected before entering C.
