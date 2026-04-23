# cortext-dart

Build the shared library from the repository root:

```bash
zig build -Dshared=true -Dllama=false

# or the legacy CMake path:
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext
```

Then use the Dart package from the repository:

```bash
cd bindings/dart
dart pub get
dart test
```

By default the package looks for the shared library in:

- `CORTEXT_LIBRARY_PATH`
- `zig-out/lib/`
- `build/ffi-release/`
- `build/ffi-release/lib/`
- `install/lib/`

Example:

```dart
import 'dart:typed_data';

import 'package:cortext/cortext.dart';

void main() {
  final cortext = Cortext(dbPath: ':memory:');
  final context = cortext.processText('hello world', 'user');
  print(context['should_interrupt']);

  final audio = Float32List(16000);
  cortext.processAudio(audio, 'mic');
  cortext.close();
}
```
