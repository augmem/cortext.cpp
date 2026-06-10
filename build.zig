const std = @import("std");

const sqlite_header_template = @embedFile("third_party/sqlite/src/sqlite.h.in");

const cortext_cpp_sources = &.{
    "src/store.cpp",
    "src/store/extension_loader.cpp",
    "src/store/object_store.cpp",
    "src/store/schema.cpp",
    "src/store/facts.cpp",
    "src/internal/cancellation.cpp",
    "src/processor/operation_context.cpp",
    "src/signal_processor.cpp",
    "src/cortext.cpp",
    "src/core/thread_config.cpp",
    "src/capi.cpp",
    "src/telemetry/telemetry.cpp",
    "src/data/centroids.cpp",
    "src/data/embedded_centroid_vectors.cpp",
    "src/operations/centroids.cpp",
    "src/operations/focus.cpp",
    "src/operations/recent_context.cpp",
    "src/operations/sensitivity.cpp",
    "src/operations/uncertainty.cpp",
    "src/operations/threshold.cpp",
    "src/operations/coherence.cpp",
    "src/operations/stability.cpp",
    "src/operations/blend.cpp",
    "src/operations/metrics.cpp",
    "src/operations/effective_focus.cpp",
    "src/operations/focus_spread.cpp",
    "src/operations/boundary.cpp",
    "src/operations/memory_strength.cpp",
    "src/operations/focus_feedback.cpp",
    "src/operations/sensitivity_feedback.cpp",
    "src/operations/stability_feedback.cpp",
    "src/operations/influence.cpp",
    "src/operations/reconsolidation.cpp",
    "src/operations/competition.cpp",
    "src/operations/predictive.cpp",
    "src/operations/emotion.cpp",
    "src/operations/working_memory.cpp",
    "src/operations/metacognitive.cpp",
    "src/operations/serial_position.cpp",
    "src/operations/serial_position_apply.cpp",
    "src/operations/interrupt_gate.cpp",
    "src/operations/consolidation.cpp",
    "src/operations/consolidation_cluster.cpp",
    "src/operations/consolidation_gate.cpp",
    "src/operations/consolidation_shallow.cpp",
    "src/operations/label_bank.cpp",
    "src/operations/consolidation_summarize.cpp",
    "src/operations/process_extraction_results.cpp",
    "src/operations/embedding_prediction_error.cpp",
    "src/operations/precision.cpp",
    "src/operations/constructive_recall_internal.cpp",
    "src/operations/meta_learning_internal.cpp",
    "src/operations/graph_build.cpp",
    "src/operations/graph_retrieval.cpp",
    "src/operations/temporal_retrieval.cpp",
    "src/operations/eviction_ablation.cpp",
    "src/operations/retrieval_debug_state.cpp",
    "src/operations/neuromodulators.cpp",
    "src/operations/emotion_cascade.cpp",
    "src/operations/signal_metrics_persistence.cpp",
    "src/operations/write_gate.cpp",
    "src/operations/memory_storage.cpp",
    "src/operations/detect_memory_usage.cpp",
    "src/operations/synaptic_tagging.cpp",
    "src/operations/drift_accumulation.cpp",
    "src/operations/streaming_pacing.cpp",
    "src/operations/accumulator_scores.cpp",
    "src/operations/accumulator.cpp",
    "src/operations/accumulator_reset.cpp",
    "src/operations/spike_bypass.cpp",
    "src/extractor/gemma_extractor.cpp",
    "src/summarizer/gemma_summarizer.cpp",
    "src/deep_llm/deep_llm_factory.cpp",
    "src/generator/json_decoder.cpp",
    "src/audio/gemma_audio.cpp",
};

const objstore_c_sources = &.{
    "third_party/sqlite-objstore/src/objstore.c",
    "third_party/sqlite-objstore/src/backend_registry.c",
    "third_party/sqlite-objstore/src/backend_fs_common.c",
    "third_party/sqlite-objstore/src/backend_portable.c",
    "third_party/sqlite-objstore/src/objstore_vtab.c",
    "third_party/sqlite-objstore/src/objstore_txn.c",
    "third_party/sqlite-objstore/src/object_manager.c",
    "third_party/sqlite-objstore/src/blake3_hash.c",
    "third_party/sqlite-objstore/third_party/blake3/blake3.c",
    "third_party/sqlite-objstore/third_party/blake3/blake3_dispatch.c",
    "third_party/sqlite-objstore/third_party/blake3/blake3_portable.c",
    "third_party/sqlite-objstore/src/backend_opfs.c",
    "third_party/sqlite-objstore/src/backend_vfs.c",
    "third_party/sqlite-objstore/src/backend_file.c",
    "third_party/sqlite-objstore/src/backend_sqlite.c",
};

const sqlite_vec_header =
    \\#ifndef SQLITE_VEC_H
    \\#define SQLITE_VEC_H
    \\
    \\#ifndef SQLITE_CORE
    \\#include "sqlite3ext.h"
    \\#else
    \\#include "sqlite3.h"
    \\#endif
    \\
    \\#ifdef SQLITE_VEC_STATIC
    \\  #define SQLITE_VEC_API
    \\#else
    \\  #ifdef _WIN32
    \\    #define SQLITE_VEC_API __declspec(dllexport)
    \\  #else
    \\    #define SQLITE_VEC_API
    \\  #endif
    \\#endif
    \\
    \\#define SQLITE_VEC_VERSION "v0.0.0"
    \\#define SQLITE_VEC_DATE ""
    \\#define SQLITE_VEC_SOURCE ""
    \\#define SQLITE_VEC_VERSION_MAJOR 0
    \\#define SQLITE_VEC_VERSION_MINOR 0
    \\#define SQLITE_VEC_VERSION_PATCH 0
    \\
    \\#ifdef __cplusplus
    \\extern "C" {
    \\#endif
    \\
    \\SQLITE_VEC_API int sqlite3_vec_init(sqlite3 *db, char **pzErrMsg,
    \\                  const sqlite3_api_routines *pApi);
    \\
    \\#ifdef __cplusplus
    \\}  /* end of the 'extern "C"' block */
    \\#endif
    \\
    \\#endif /* ifndef SQLITE_VEC_H */
    \\
;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const shared = b.option(bool, "shared", "Build libcortext as a shared library") orelse true;
    const link_sqlite = b.option(bool, "link-sqlite", "Link against host/target sqlite3") orelse target.query.isNative();
    const enable_llama = b.option(bool, "llama", "Enable llama.cpp support using prebuilt target-compatible libraries") orelse false;
    const llama_include = b.option([]const u8, "llama_include", "Directory containing llama.h");
    const llama_lib = b.option([]const u8, "llama_lib", "Path to libllama");
    const ggml_include = b.option([]const u8, "ggml_include", "Directory containing ggml-backend.h");
    const ggml_lib = b.option([]const u8, "ggml_lib", "Path to libggml");
    const ggml_base_lib = b.option([]const u8, "ggml_base_lib", "Optional path to libggml-base");
    const ggml_cpu_lib = b.option([]const u8, "ggml_cpu_lib", "Optional path to libggml-cpu");
    const ggml_blas_lib = b.option([]const u8, "ggml_blas_lib", "Optional path to libggml-blas");

    const eigen = b.dependency("eigen", .{});
    const nlohmann_json = b.dependency("nlohmann_json", .{});

    const mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });

    const lib = b.addLibrary(.{
        .name = "cortext",
        .linkage = if (shared) .dynamic else .static,
        .root_module = mod,
        .version = .{ .major = 0, .minor = 1, .patch = 0 },
    });

    mod.addIncludePath(b.path("include"));
    mod.addIncludePath(b.path("src"));
    mod.addIncludePath(b.path("third_party/sqlite/src"));
    mod.addIncludePath(b.path("third_party/sqlite-objstore/include"));
    mod.addIncludePath(b.path("third_party/sqlite-objstore/third_party/blake3"));
    mod.addIncludePath(eigen.path(""));
    mod.addIncludePath(nlohmann_json.path("include"));

    const generated = b.addWriteFiles();
    _ = generated.add("sqlite3.h", sqliteHeader(b));
    _ = generated.add("third_party/sqlite-vec/sqlite-vec.h", sqlite_vec_header);
    mod.addIncludePath(generated.getDirectory());
    mod.addIncludePath(generated.getDirectory().path(b, "third_party/sqlite-vec"));

    mod.addCMacro("SQLITE_CORE", "1");
    mod.addCMacro("CORTEXT_EMBED_VEC", "1");
    mod.addCMacro("SQLITE_VEC_OMIT_FS", "1");
    mod.addCMacro("CORTEXT_EMBED_OBJSTORE", "1");
    mod.addCMacro("CORTEXT_DISABLE_OPENTELEMETRY", "1");
    mod.addCMacro("CORTEXT_DISABLE_OGA", "1");
    mod.addCMacro("CORTEXT_DISABLE_LITERT", "1");
    mod.addCMacro("CORTEXT_DISABLE_SHERPA_ONNX", "1");
    mod.addCMacro("BLAKE3_NO_SSE2", "1");
    mod.addCMacro("BLAKE3_NO_SSE41", "1");
    mod.addCMacro("BLAKE3_NO_AVX2", "1");
    mod.addCMacro("BLAKE3_NO_AVX512", "1");
    if (shared) {
        mod.addCMacro("CORTEXT_BUILDING_SHARED", "1");
    }

    if (enable_llama) {
        const include_dir = llama_include orelse fail(b, "-Dllama=true requires -Dllama_include=/path/to/include");
        const library = llama_lib orelse fail(b, "-Dllama=true requires -Dllama_lib=/path/to/libllama");
        const ggml_include_dir = ggml_include orelse include_dir;

        mod.addIncludePath(.{ .cwd_relative = include_dir });
        mod.addIncludePath(.{ .cwd_relative = ggml_include_dir });
        mod.addCMacro("CORTEXT_ENABLE_LLAMA_CPP", "1");
        mod.addCMacro("CORTEXT_LLAMA_HEADER_PATH", b.fmt("\"{s}/llama.h\"", .{include_dir}));
        mod.addCMacro("CORTEXT_GGML_BACKEND_HEADER_PATH", b.fmt("\"{s}/ggml-backend.h\"", .{ggml_include_dir}));
        mod.addObjectFile(.{ .cwd_relative = library });
        if (ggml_lib) |path| mod.addObjectFile(.{ .cwd_relative = path });
        if (ggml_base_lib) |path| mod.addObjectFile(.{ .cwd_relative = path });
        if (ggml_cpu_lib) |path| mod.addObjectFile(.{ .cwd_relative = path });
        if (ggml_blas_lib) |path| {
            mod.addObjectFile(.{ .cwd_relative = path });
            if (target.result.os.tag == .macos) {
                mod.linkFramework("Accelerate", .{});
            }
        }
    }

    const cxx_flags = &.{
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
    };
    mod.addCSourceFiles(.{ .files = cortext_cpp_sources, .flags = cxx_flags });
    if (pathExists("src/operations/storage_pressure.cpp")) {
        mod.addCSourceFiles(.{ .files = &.{"src/operations/storage_pressure.cpp"}, .flags = cxx_flags });
    }
    mod.addCSourceFiles(.{ .files = &.{"third_party/sqlite-vec/sqlite-vec.c"}, .flags = &.{"-w"} });
    mod.addCSourceFiles(.{ .files = objstore_c_sources, .flags = &.{"-w"} });
    if (target.result.cpu.arch.isAARCH64()) {
        mod.addCSourceFiles(.{ .files = &.{"third_party/sqlite-objstore/third_party/blake3/blake3_neon.c"}, .flags = &.{"-w"} });
    }

    if (link_sqlite) {
        mod.linkSystemLibrary("sqlite3", .{});
    }
    if (target.result.os.tag != .windows) {
        mod.linkSystemLibrary("m", .{});
        mod.linkSystemLibrary("pthread", .{});
    }
    if (target.result.os.tag == .linux) {
        mod.linkSystemLibrary("dl", .{});
    }

    b.installDirectory(.{
        .source_dir = b.path("include"),
        .install_dir = .header,
        .install_subdir = "",
        .include_extensions = &.{ ".h", ".hpp" },
    });
    b.installArtifact(lib);

    const check_step = b.step("check", "Build libcortext without installing");
    check_step.dependOn(&lib.step);
}

fn fail(b: *std.Build, comptime message: []const u8) noreturn {
    std.log.err(message, .{});
    b.invalid_user_input = true;
    @panic(message);
}

fn sqliteHeader(b: *std.Build) []const u8 {
    var out: []u8 = b.allocator.dupe(u8, sqlite_header_template) catch @panic("OOM");
    out = replaceAll(b, out, "--VERS--", "3.51.0");
    out = replaceAll(b, out, "--VERSION-NUMBER--", "3051000");
    out = replaceAll(b, out, "--SOURCE-ID--", "cortext-third-party-sqlite");
    out = replaceAll(b, out, "--SCM-BRANCH--", "cortext");
    out = replaceAll(b, out, "--SCM-TAGS--", "");
    out = replaceAll(b, out, "--SCM-DATETIME--", "");
    return b.fmt("{s}\n#endif /* SQLITE3_H */\n", .{out});
}

fn replaceAll(b: *std.Build, input: []u8, needle: []const u8, replacement: []const u8) []u8 {
    return std.mem.replaceOwned(u8, b.allocator, input, needle, replacement) catch @panic("OOM");
}

fn pathExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}
