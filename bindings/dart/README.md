# cortext-dart

The Dart package uses `dart:ffi` over the Cortext C ABI.

## Build

From the repository root:

```bash
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext
```

Then:

```bash
cd bindings/dart
dart pub get
dart test
```

The package searches `CORTEXT_LIBRARY_PATH`, `build/ffi-release`,
`build/ffi-release/lib`, `zig-out/lib`, and `install/lib`.

## Use

```dart
import 'dart:typed_data';

import 'package:cortext/cortext.dart';

void main() {
  final engine = Cortext(dbPath: ':memory:', modelsDir: 'models');

  final ctx = engine.processText('Bailey likes tennis balls.', 'chat/main');
  print(ctx['should_interrupt']);

  final embedding = engine.embedText('embed without storing');
  print(embedding.length);

  final audio = Float32List(16000);
  engine.processAudio(audio, 'mic/main');

  engine.consolidate();
  engine.flush();
  engine.reset();
  engine.close();
}
```

## API

- `Config`: Focus/Sensitivity/Stability, mechanism toggles, and signal-filter
  toggles.
- `processText`, `processAudio`, `processImage`: decoded JSON maps.
- `processTextJson`, `processAudioJson`, `processImageJson`: raw JSON strings.
- `embedText`, `embedAudio`, `embedImage`: embed-only helpers.
- `consolidate`, `flush`, `reset`, `close`.

Audio input is 16 kHz mono float32 PCM. Image input is row-major RGB/RGBA bytes
with explicit dimensions.
