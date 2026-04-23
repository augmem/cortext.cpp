import 'dart:io';

import 'package:cortext/cortext.dart';
import 'package:test/test.dart';

void main() {
  test('candidate library paths include Zig and ffi-release output directories', () {
    final paths = CortextLibrary.candidateLibraryPaths(
      repoRoot: '/tmp/cortext',
    );
    expect(
      paths,
      contains(
        '/tmp/cortext${Platform.pathSeparator}zig-out${Platform.pathSeparator}lib${Platform.pathSeparator}${_expectedLibraryName()}',
      ),
    );
    expect(
      paths,
      contains(
        '/tmp/cortext${Platform.pathSeparator}build${Platform.pathSeparator}ffi-release${Platform.pathSeparator}${_expectedLibraryName()}',
      ),
    );
  });
}

String _expectedLibraryName() {
  if (Platform.isMacOS) {
    return 'libcortext.dylib';
  }
  if (Platform.isWindows) {
    return 'cortext.dll';
  }
  return 'libcortext.so';
}
