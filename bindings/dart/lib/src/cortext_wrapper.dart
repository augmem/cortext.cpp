import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:meta/meta.dart';

import 'cortext_bindings.dart';

enum ConsolidationMode {
  shallow(0),
  deep(1),
  both(2);

  const ConsolidationMode(this.value);

  final int value;
}

final class CortextError implements Exception {
  const CortextError(this.message);

  final String message;

  @override
  String toString() => 'CortextError: $message';
}

@immutable
final class Config {
  const Config({
    this.focus = 0.5,
    this.sensitivity = 0.5,
    this.stability = 0.5,
    this.affectInterrupt = true,
    this.affectRetrieval = true,
    this.reinforcementEnabled = true,
    this.proceduralEnabled = true,
    this.sequentialEdgesEnabled = true,
    this.labelBankPath,
    this.summarizerProviderUri,
    this.extractorProviderUri,
  });

  final double focus;
  final double sensitivity;
  final double stability;
  final bool affectInterrupt;
  final bool affectRetrieval;
  final bool reinforcementEnabled;
  final bool proceduralEnabled;
  final bool sequentialEdgesEnabled;
  final String? labelBankPath;

  /// Inference-provider URI for the Summarizer role, e.g.
  /// `ollama://127.0.0.1:11435/gemma4:e2b`. Null/empty keeps local model
  /// auto-discovery; an unresolvable URI makes construction throw.
  final String? summarizerProviderUri;

  /// Inference-provider URI for the Extractor role; same semantics as
  /// [summarizerProviderUri].
  final String? extractorProviderUri;
}

final class CortextLibrary {
  CortextLibrary._(this.dynamicLibrary)
    : bindings = CortextBindings(dynamicLibrary);

  final DynamicLibrary dynamicLibrary;
  final CortextBindings bindings;

  static CortextLibrary open({String? libraryPath}) {
    if (libraryPath != null) {
      return CortextLibrary._(DynamicLibrary.open(libraryPath));
    }

    final envPath = Platform.environment['CORTEXT_LIBRARY_PATH'];
    if (envPath != null && envPath.isNotEmpty) {
      return CortextLibrary._(DynamicLibrary.open(envPath));
    }

    for (final candidate in candidateLibraryPaths()) {
      if (File(candidate).existsSync()) {
        return CortextLibrary._(DynamicLibrary.open(candidate));
      }
    }

    for (final name in _libraryBasenamesForPlatform()) {
      try {
        return CortextLibrary._(DynamicLibrary.open(name));
      } on ArgumentError {
        continue;
      }
    }

    throw CortextError(
      'Could not locate the Cortext shared library. '
      'Build it with `zig build -Dshared=true -Dllama=false` or '
      '`cmake --preset ffi-release && cmake --build --preset ffi-release --target cortext`, '
      'or set CORTEXT_LIBRARY_PATH.',
    );
  }

  String version() {
    final raw = bindings.cortext_version();
    return raw == nullptr ? '' : raw.cast<Utf8>().toDartString();
  }

  String lastError() {
    final raw = bindings.cortext_last_error();
    return raw == nullptr ? '' : raw.cast<Utf8>().toDartString();
  }

  @visibleForTesting
  static List<String> candidateLibraryPaths({String? repoRoot}) {
    final roots = repoRoot != null
        ? <String>[repoRoot]
        : _candidateSearchRoots();
    return <String>[
      for (final root in roots)
        for (final directory in <String>[
          '$root/zig-out/lib',
          '$root/build/ffi-release',
          '$root/build/ffi-release/lib',
          '$root/install/lib',
        ])
          for (final name in _libraryBasenamesForPlatform()) '$directory/$name',
    ];
  }

  static List<String> _candidateSearchRoots() {
    final roots = <String>{};
    void addAncestors(String startingPath) {
      var current = Directory(startingPath).absolute;
      while (true) {
        roots.add(current.path);
        final parent = current.parent;
        if (parent.path == current.path) {
          break;
        }
        current = parent;
      }
    }

    addAncestors(Directory.current.path);
    if (Platform.script.scheme == 'file') {
      addAncestors(File.fromUri(Platform.script).parent.path);
    }
    return roots.toList(growable: false);
  }

  static List<String> _libraryBasenamesForPlatform() {
    if (Platform.isMacOS) {
      return const <String>['libcortext.dylib'];
    }
    if (Platform.isWindows) {
      return const <String>['cortext.dll', 'libcortext.dll'];
    }
    return const <String>['libcortext.so'];
  }
}

final class Cortext {
  Cortext({
    this.dbPath = ':memory:',
    this.modelsDir,
    this.config,
    String? libraryPath,
    CortextLibrary? library,
  }) : _library = library ?? CortextLibrary.open(libraryPath: libraryPath) {
    final nativeConfig = calloc<cortext_config>();
    Pointer<Utf8>? labelBankPathPointer;
    Pointer<Utf8>? summarizerProviderUriPointer;
    Pointer<Utf8>? extractorProviderUriPointer;
    Pointer<Utf8>? dbPathPointer;
    Pointer<Utf8>? modelsDirPointer;

    try {
      _library.bindings.cortext_config_init(nativeConfig);
      final configValue = config;
      if (configValue != null) {
        nativeConfig.ref
          ..focus = configValue.focus
          ..sensitivity = configValue.sensitivity
          ..stability = configValue.stability
          ..affect_interrupt = _boolToInt(configValue.affectInterrupt)
          ..affect_retrieval = _boolToInt(configValue.affectRetrieval)
          ..reinforcement_enabled = _boolToInt(configValue.reinforcementEnabled)
          ..procedural_enabled = _boolToInt(configValue.proceduralEnabled)
          ..sequential_edges_enabled = _boolToInt(
            configValue.sequentialEdgesEnabled,
          );

        final labelBankPath = configValue.labelBankPath;
        if (labelBankPath != null) {
          labelBankPathPointer = labelBankPath.toNativeUtf8();
          nativeConfig.ref.label_bank_path = labelBankPathPointer.cast();
        }

        final summarizerProviderUri = configValue.summarizerProviderUri;
        if (summarizerProviderUri != null) {
          summarizerProviderUriPointer = summarizerProviderUri.toNativeUtf8();
          nativeConfig.ref.summarizer_provider_uri = summarizerProviderUriPointer
              .cast();
        }

        final extractorProviderUri = configValue.extractorProviderUri;
        if (extractorProviderUri != null) {
          extractorProviderUriPointer = extractorProviderUri.toNativeUtf8();
          nativeConfig.ref.extractor_provider_uri = extractorProviderUriPointer
              .cast();
        }
      }

      dbPathPointer = dbPath.toNativeUtf8();
      if (modelsDir != null) {
        modelsDirPointer = modelsDir!.toNativeUtf8();
      }

      _handle = _library.bindings.cortext_create_with_config(
        nativeConfig,
        dbPathPointer.cast(),
        modelsDirPointer?.cast() ?? nullptr,
      );
      if (_handle == nullptr) {
        _throwLastError('cortext_create_with_config failed');
      }
    } finally {
      calloc.free(nativeConfig);
      if (labelBankPathPointer != null) {
        malloc.free(labelBankPathPointer);
      }
      if (summarizerProviderUriPointer != null) {
        malloc.free(summarizerProviderUriPointer);
      }
      if (extractorProviderUriPointer != null) {
        malloc.free(extractorProviderUriPointer);
      }
      if (dbPathPointer != null) {
        malloc.free(dbPathPointer);
      }
      if (modelsDirPointer != null) {
        malloc.free(modelsDirPointer);
      }
    }
  }

  final String dbPath;
  final String? modelsDir;
  final Config? config;
  final CortextLibrary _library;
  Pointer<Void> _handle = nullptr;

  String version() => _library.version();

  String lastError() => _library.lastError();

  void close() {
    if (_handle == nullptr) {
      return;
    }
    _library.bindings.cortext_free(_handle);
    _handle = nullptr;
  }

  String processTextJson(String text, String sourceId) {
    _ensureOpen();
    return _withTwoStrings(
      text,
      sourceId,
      (textPointer, sourcePointer) =>
          _library.bindings.cortext_process_text_json(
            _handle,
            textPointer.cast(),
            sourcePointer.cast(),
          ),
    );
  }

  Map<String, dynamic> processText(String text, String sourceId) {
    return _decodeJsonObject(processTextJson(text, sourceId));
  }

  String processAudioJson(Float32List pcm, String sourceId) {
    _ensureOpen();
    final sourcePointer = sourceId.toNativeUtf8();
    Pointer<Float> pcmPointer = nullptr;
    try {
      if (pcm.isNotEmpty) {
        pcmPointer = malloc<Float>(pcm.length);
        pcmPointer.asTypedList(pcm.length).setAll(0, pcm);
      }
      final raw = _library.bindings.cortext_process_audio_json(
        _handle,
        pcmPointer,
        pcm.length,
        sourcePointer.cast(),
      );
      return _takeJsonString(raw);
    } finally {
      malloc.free(sourcePointer);
      if (pcmPointer != nullptr) {
        malloc.free(pcmPointer);
      }
    }
  }

  Map<String, dynamic> processAudio(Float32List pcm, String sourceId) {
    return _decodeJsonObject(processAudioJson(pcm, sourceId));
  }

  String processImageJson(
    Uint8List data,
    int width,
    int height,
    int channels,
    String sourceId,
  ) {
    _ensureOpen();
    final sourcePointer = sourceId.toNativeUtf8();
    Pointer<Uint8> dataPointer = nullptr;
    try {
      if (data.isNotEmpty) {
        dataPointer = malloc<Uint8>(data.length);
        dataPointer.asTypedList(data.length).setAll(0, data);
      }
      final raw = _library.bindings.cortext_process_image_json(
        _handle,
        dataPointer,
        width,
        height,
        channels,
        sourcePointer.cast(),
      );
      return _takeJsonString(raw);
    } finally {
      malloc.free(sourcePointer);
      if (dataPointer != nullptr) {
        malloc.free(dataPointer);
      }
    }
  }

  Map<String, dynamic> processImage(
    Uint8List data,
    int width,
    int height,
    int channels,
    String sourceId,
  ) {
    return _decodeJsonObject(
      processImageJson(data, width, height, channels, sourceId),
    );
  }

  String consolidateJson() {
    _ensureOpen();
    return _takeJsonString(_library.bindings.cortext_consolidate_json(_handle));
  }

  Map<String, dynamic> consolidate() {
    return _decodeJsonObject(consolidateJson());
  }

  String consolidateModeJson(ConsolidationMode mode) {
    _ensureOpen();
    return _takeJsonString(
      _library.bindings.cortext_consolidate_mode_json(_handle, mode.value),
    );
  }

  Map<String, dynamic> consolidateMode(ConsolidationMode mode) {
    return _decodeJsonObject(consolidateModeJson(mode));
  }

  void flush() {
    _ensureOpen();
    final status = _library.bindings.cortext_flush(_handle);
    if (status != 0) {
      _throwLastError('cortext_flush failed');
    }
  }

  String _withTwoStrings(
    String first,
    String second,
    Pointer<Char> Function(Pointer<Utf8> first, Pointer<Utf8> second) call,
  ) {
    final firstPointer = first.toNativeUtf8();
    final secondPointer = second.toNativeUtf8();
    try {
      return _takeJsonString(call(firstPointer, secondPointer));
    } finally {
      malloc.free(firstPointer);
      malloc.free(secondPointer);
    }
  }

  String _takeJsonString(Pointer<Char> raw) {
    if (raw == nullptr) {
      _throwLastError('cortext call failed');
    }
    try {
      return raw.cast<Utf8>().toDartString();
    } finally {
      _library.bindings.cortext_string_free(raw);
    }
  }

  Map<String, dynamic> _decodeJsonObject(String payload) {
    final decoded = jsonDecode(payload);
    if (decoded is! Map<String, dynamic>) {
      throw const CortextError('Expected a JSON object from cortext');
    }
    return decoded;
  }

  Never _throwLastError(String prefix) {
    final detail = lastError();
    throw CortextError(detail.isEmpty ? prefix : '$prefix: $detail');
  }

  void _ensureOpen() {
    if (_handle == nullptr) {
      throw const CortextError('Cortext handle is closed');
    }
  }

  int _boolToInt(bool value) => value ? 1 : 0;
}
