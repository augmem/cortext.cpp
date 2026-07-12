const std = @import("std");

const cortext_version = "1.2.0";

const sqlite_header_template = @embedFile("third_party/sqlite/src/sqlite.h.in");

const cortext_cpp_sources = &.{
    "src/store.cpp",
    "src/store/extension_loader.cpp",
    "src/store/object_store.cpp",
    "src/store/schema.cpp",
    "src/internal/cancellation.cpp",
    "src/processor/operation_context.cpp",
    "src/signal_processor.cpp",
    "src/signal_filter.cpp",
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
    "src/operations/serial_position.cpp",
    "src/operations/serial_position_apply.cpp",
    "src/operations/interrupt_gate.cpp",
    "src/operations/consolidation.cpp",
    "src/operations/consolidation_cluster.cpp",
    "src/operations/consolidation_gate.cpp",
    "src/operations/consolidation_shallow.cpp",
    "src/operations/embedding_prediction_error.cpp",
    "src/operations/precision.cpp",
    "src/operations/constructive_recall_internal.cpp",
    "src/operations/meta_learning_internal.cpp",
    "src/operations/graph_build.cpp",
    "src/operations/graph_retrieval.cpp",
    "src/operations/eviction_policy_override.cpp",
    "src/operations/storage_pressure.cpp",
    "src/operations/retrieval_trace_state.cpp",
    "src/operations/neuromodulators.cpp",
    "src/operations/emotion_cascade.cpp",
    "src/operations/signal_metrics_persistence.cpp",
    "src/operations/write_gate.cpp",
    "src/operations/memory_storage.cpp",
    "src/operations/detect_memory_usage.cpp",
    "src/operations/synaptic_tagging.cpp",
    "src/operations/drift_accumulation.cpp",
    "src/operations/streaming_pacing.cpp",
    "src/operations/soft_anchor.cpp",
    "src/operations/accumulator_scores.cpp",
    "src/operations/accumulator.cpp",
    "src/operations/accumulator_reset.cpp",
    "src/operations/spike_bypass.cpp",
    "src/models/embedding_model_pin.cpp",
    "src/models/aist_gguf_encoder.cpp",
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

const ggml_base_sources = &.{
    "src/ggml.c",
    "src/ggml.cpp",
    "src/ggml-alloc.c",
    "src/ggml-backend.cpp",
    "src/ggml-backend-reg.cpp",
    "src/ggml-opt.cpp",
    "src/ggml-threading.cpp",
    "src/ggml-quants.c",
    "src/gguf.cpp",
};

const ggml_cpu_sources = &.{
    "src/ggml-cpu/ggml-cpu.c",
    "src/ggml-cpu/ggml-cpu.cpp",
    "src/ggml-cpu/repack.cpp",
    "src/ggml-cpu/hbm.cpp",
    "src/ggml-cpu/quants.c",
    "src/ggml-cpu/traits.cpp",
    "src/ggml-cpu/amx/amx.cpp",
    "src/ggml-cpu/amx/mmq.cpp",
    "src/ggml-cpu/binary-ops.cpp",
    "src/ggml-cpu/unary-ops.cpp",
    "src/ggml-cpu/vec.cpp",
    "src/ggml-cpu/ops.cpp",
};

const ggml_cpu_x86_sources = &.{
    "src/ggml-cpu/arch/x86/quants.c",
    "src/ggml-cpu/arch/x86/repack.cpp",
};

const ggml_cpu_arm_sources = &.{
    "src/ggml-cpu/arch/arm/quants.c",
    "src/ggml-cpu/arch/arm/repack.cpp",
};

const sqlite_c_sources = &.{
    "third_party/sqlite/src/alter.c",
    "third_party/sqlite/src/analyze.c",
    "third_party/sqlite/src/attach.c",
    "third_party/sqlite/src/auth.c",
    "third_party/sqlite/src/backup.c",
    "third_party/sqlite/src/bitvec.c",
    "third_party/sqlite/src/btmutex.c",
    "third_party/sqlite/src/btree.c",
    "third_party/sqlite/src/build.c",
    "third_party/sqlite/src/callback.c",
    "third_party/sqlite/src/carray.c",
    "third_party/sqlite/src/complete.c",
    "third_party/sqlite/src/date.c",
    "third_party/sqlite/src/dbpage.c",
    "third_party/sqlite/src/dbstat.c",
    "third_party/sqlite/src/delete.c",
    "third_party/sqlite/src/expr.c",
    "third_party/sqlite/src/fault.c",
    "third_party/sqlite/src/fkey.c",
    "third_party/sqlite/src/func.c",
    "third_party/sqlite/src/global.c",
    "third_party/sqlite/src/hash.c",
    "third_party/sqlite/src/insert.c",
    "third_party/sqlite/src/json.c",
    "third_party/sqlite/src/legacy.c",
    "third_party/sqlite/src/loadext.c",
    "third_party/sqlite/src/main.c",
    "third_party/sqlite/src/malloc.c",
    "third_party/sqlite/src/mem0.c",
    "third_party/sqlite/src/mem1.c",
    "third_party/sqlite/src/mem2.c",
    "third_party/sqlite/src/mem3.c",
    "third_party/sqlite/src/mem5.c",
    "third_party/sqlite/src/memdb.c",
    "third_party/sqlite/src/memjournal.c",
    "third_party/sqlite/src/mutex.c",
    "third_party/sqlite/src/mutex_noop.c",
    "third_party/sqlite/src/mutex_unix.c",
    "third_party/sqlite/src/mutex_w32.c",
    "third_party/sqlite/src/notify.c",
    "third_party/sqlite/src/os.c",
    "third_party/sqlite/src/os_kv.c",
    "third_party/sqlite/src/os_unix.c",
    "third_party/sqlite/src/os_win.c",
    "third_party/sqlite/src/pager.c",
    "third_party/sqlite/src/pcache.c",
    "third_party/sqlite/src/pcache1.c",
    "third_party/sqlite/src/pragma.c",
    "third_party/sqlite/src/prepare.c",
    "third_party/sqlite/src/printf.c",
    "third_party/sqlite/src/random.c",
    "third_party/sqlite/src/resolve.c",
    "third_party/sqlite/src/rowset.c",
    "third_party/sqlite/src/select.c",
    "third_party/sqlite/src/status.c",
    "third_party/sqlite/src/table.c",
    "third_party/sqlite/src/threads.c",
    "third_party/sqlite/src/tokenize.c",
    "third_party/sqlite/src/treeview.c",
    "third_party/sqlite/src/trigger.c",
    "third_party/sqlite/src/update.c",
    "third_party/sqlite/src/upsert.c",
    "third_party/sqlite/src/utf.c",
    "third_party/sqlite/src/util.c",
    "third_party/sqlite/src/vacuum.c",
    "third_party/sqlite/src/vdbe.c",
    "third_party/sqlite/src/vdbeapi.c",
    "third_party/sqlite/src/vdbeaux.c",
    "third_party/sqlite/src/vdbeblob.c",
    "third_party/sqlite/src/vdbemem.c",
    "third_party/sqlite/src/vdbesort.c",
    "third_party/sqlite/src/vdbetrace.c",
    "third_party/sqlite/src/vdbevtab.c",
    "third_party/sqlite/src/vtab.c",
    "third_party/sqlite/src/wal.c",
    "third_party/sqlite/src/walker.c",
    "third_party/sqlite/src/where.c",
    "third_party/sqlite/src/wherecode.c",
    "third_party/sqlite/src/whereexpr.c",
    "third_party/sqlite/src/window.c",
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

const sqlite_ctime_source =
    \\/* Generated by cortext Zig fallback when tclsh is unavailable. */
    \\#ifndef SQLITE_OMIT_COMPILEOPTION_DIAGS
    \\const char **sqlite3CompileOptions(int *pnOpt){
    \\  static const char *azCompileOpt[] = { 0 };
    \\  if( pnOpt ) *pnOpt = 0;
    \\  return azCompileOpt;
    \\}
    \\#endif
    \\
;

const sqlite_opcodes_source =
    \\/* Generated by cortext Zig fallback when tclsh is unavailable. */
    \\#if !defined(SQLITE_OMIT_EXPLAIN) || defined(SQLITE_ENABLE_BYTECODE_VTAB)
    \\const char *sqlite3OpcodeName(int i){
    \\  (void)i;
    \\  return "OP_Unknown";
    \\}
    \\#endif
    \\
;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const shared = b.option(bool, "shared", "Build libcortext as a shared library") orelse true;
    const build_cli = b.option(bool, "cli", "Build and install the cortext_cli executable") orelse true;
    const build_node_addon = b.option(bool, "node-addon", "Build and install the Node-API addon") orelse false;
    const node_include = b.option([]const u8, "node-include", "Directory containing node_api.h for Node-API addon builds");
    const node_lib = b.option([]const u8, "node-lib", "Windows node.exe import library for Node-API addon builds");
    const unsupported_text_only = b.option(bool, "unsupported-text-only", "Allow unsupported builds without audio/image GGML kernel support") orelse false;
    const enable_ggml = b.option(bool, "ggml", "Enable GGML audio/image kernel support") orelse !unsupported_text_only;
    const fetch_aist_model = b.option(bool, "fetch-aist-model", "Download the required AIST GGUF model during the default build") orelse true;
    const aist_model_quant = b.option([]const u8, "aist-model-quant", "AIST quantization to download: q8_0, q5_1, or all") orelse "q8_0";
    const model_assets_dir = b.option([]const u8, "model-assets-dir", "Directory where Cortext build-time model assets are stored") orelse "models";
    const ggml_include = b.option([]const u8, "ggml_include", "Directory containing ggml.h and ggml-backend.h");
    const ggml_lib = b.option([]const u8, "ggml_lib", "Path to libggml");
    const ggml_base_lib = b.option([]const u8, "ggml_base_lib", "Path to libggml-base");
    const ggml_cpu_lib = b.option([]const u8, "ggml_cpu_lib", "Path to libggml-cpu");
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
        .version = .{ .major = 1, .minor = 2, .patch = 0 },
    });

    if (fetch_aist_model) {
        if (!std.mem.eql(u8, aist_model_quant, "q8_0") and
            !std.mem.eql(u8, aist_model_quant, "q5_1") and
            !std.mem.eql(u8, aist_model_quant, "all"))
        {
            fail(b, "-Daist-model-quant must be one of: q8_0, q5_1, all");
        }
        const aist_model = b.addSystemCommand(&.{
            "python3",
            "scripts/download_aist_model.py",
            "--output-dir",
            model_assets_dir,
            "--quant",
            aist_model_quant,
        });
        lib.step.dependOn(&aist_model.step);
    }

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
    const ctime_c = generated.add("ctime.c", sqlite_ctime_source);
    const opcodes_c = generated.add("opcodes.c", sqlite_opcodes_source);
    mod.addIncludePath(generated.getDirectory());
    mod.addIncludePath(generated.getDirectory().path(b, "third_party/sqlite-vec"));

    const lemon_mod = b.createModule(.{
        .target = b.graph.host,
        .optimize = .ReleaseFast,
        .link_libc = true,
    });
    lemon_mod.addCSourceFile(.{
        .file = b.path("third_party/sqlite/tool/lemon.c"),
        .flags = &.{"-w"},
    });
    const lemon = b.addExecutable(.{
        .name = "sqlite_lemon",
        .root_module = lemon_mod,
    });
    const keyword_mod = b.createModule(.{
        .target = b.graph.host,
        .optimize = .ReleaseFast,
        .link_libc = true,
    });
    keyword_mod.addCSourceFile(.{
        .file = b.path("third_party/sqlite/tool/mkkeywordhash.c"),
        .flags = &.{"-w"},
    });
    const mkkeywordhash = b.addExecutable(.{
        .name = "sqlite_mkkeywordhash",
        .root_module = keyword_mod,
    });
    const parse_gen = b.addSystemCommand(&.{
        "sh",
        "-c",
        "out_dir=$1; lemon=$(realpath \"$2\"); parse_y=$3; lempar=$4; parse_c=$5; parse_h=$6; cp \"$parse_y\" \"$out_dir/parse.y\" && cd \"$out_dir\" && \"$lemon\" -T\"$lempar\" parse.y && cp parse.c \"$parse_c\" && cp parse.h \"$parse_h\"",
        "sqlite-parse-gen",
    });
    _ = parse_gen.addOutputDirectoryArg("sqlite-parse");
    parse_gen.addArtifactArg(lemon);
    parse_gen.addFileArg(b.path("third_party/sqlite/src/parse.y"));
    parse_gen.addFileArg(b.path("third_party/sqlite/tool/lempar.c"));
    const parse_c = parse_gen.addOutputFileArg("parse.c");
    const parse_h = parse_gen.addOutputFileArg("parse.h");
    mod.addIncludePath(parse_h.dirname());

    const opcode_gen = b.addSystemCommand(&.{"python3"});
    opcode_gen.addFileArg(b.path("third_party/sqlite/tool/mkopcodeh.py"));
    opcode_gen.addFileArg(parse_h);
    opcode_gen.addFileArg(b.path("third_party/sqlite/src/vdbe.c"));
    const opcodes_h = opcode_gen.addOutputFileArg("opcodes.h");
    mod.addIncludePath(opcodes_h.dirname());

    const pragma_gen = b.addSystemCommand(&.{"python3"});
    pragma_gen.addFileArg(b.path("third_party/sqlite/tool/mkpragmatab.py"));
    const pragma_h = pragma_gen.addOutputFileArg("pragma.h");
    pragma_gen.addFileArg(b.path("third_party/sqlite/tool/mkpragmatab.tcl"));
    mod.addIncludePath(pragma_h.dirname());

    const keyword_gen = b.addSystemCommand(&.{"python3"});
    keyword_gen.addFileArg(b.path("third_party/sqlite/tool/run_to_file.py"));
    const keyword_h = keyword_gen.addOutputFileArg("keywordhash.h");
    keyword_gen.addArtifactArg(mkkeywordhash);
    mod.addIncludePath(keyword_h.dirname());

    mod.addCMacro("SQLITE_CORE", "1");
    mod.addCMacro("SQLITE_THREADSAFE", "1");
    mod.addCMacro("SQLITE_ENABLE_JSON1", "1");
    mod.addCMacro("CORTEXT_EMBED_VEC", "1");
    mod.addCMacro("SQLITE_VEC_OMIT_FS", "1");
    mod.addCMacro("CORTEXT_EMBED_OBJSTORE", "1");
    mod.addCMacro("CORTEXT_DISABLE_OPENTELEMETRY", "1");
    mod.addCMacro("CORTEXT_VERSION", b.fmt("\"{s}\"", .{cortext_version}));
    mod.addCMacro("BLAKE3_NO_SSE2", "1");
    mod.addCMacro("BLAKE3_NO_SSE41", "1");
    mod.addCMacro("BLAKE3_NO_AVX2", "1");
    mod.addCMacro("BLAKE3_NO_AVX512", "1");
    if (shared) {
        mod.addCMacro("CORTEXT_BUILDING_SHARED", "1");
    }
    if (target.result.os.tag == .macos) {
        mod.addCMacro("_DARWIN_C_SOURCE", "1");
        mod.addCMacro("EIGEN_ALLOCA", "__builtin_alloca");
    }
    if (target.result.os.tag == .windows and
        (target.result.cpu.arch == .x86_64 or target.result.cpu.arch == .x86))
    {
        mod.addCMacro("EIGEN_DONT_VECTORIZE", "1");
    }

    if (!enable_ggml and !unsupported_text_only) {
        fail(b, "-Dggml=false requires -Dunsupported-text-only=true");
    }

    if (enable_ggml) {
        const use_prebuilt_ggml = ggml_include != null or ggml_lib != null or
            ggml_base_lib != null or ggml_cpu_lib != null or ggml_blas_lib != null;

        mod.addCMacro("CORTEXT_ENABLE_GGML", "1");
        if (use_prebuilt_ggml) {
            const ggml_include_dir = ggml_include orelse fail(b, "prebuilt GGML requires -Dggml_include=/path/to/include");
            const ggml_library = ggml_lib orelse fail(b, "prebuilt GGML requires -Dggml_lib=/path/to/libggml");
            const ggml_base_library = ggml_base_lib orelse fail(b, "prebuilt GGML requires -Dggml_base_lib=/path/to/libggml-base");
            const ggml_cpu_library = ggml_cpu_lib orelse fail(b, "prebuilt GGML requires -Dggml_cpu_lib=/path/to/libggml-cpu");

            mod.addIncludePath(.{ .cwd_relative = ggml_include_dir });
            mod.addCMacro("CORTEXT_GGML_BACKEND_HEADER_PATH", b.fmt("\"{s}/ggml-backend.h\"", .{ggml_include_dir}));
            mod.addObjectFile(.{ .cwd_relative = ggml_library });
            mod.addObjectFile(.{ .cwd_relative = ggml_base_library });
            mod.addObjectFile(.{ .cwd_relative = ggml_cpu_library });
            if (ggml_blas_lib) |path| {
                mod.addObjectFile(.{ .cwd_relative = path });
                if (target.result.os.tag == .macos) {
                    mod.linkFramework("Accelerate", .{});
                }
            }
        } else {
            const ggml = b.dependency("ggml", .{});
            mod.addIncludePath(ggml.path("include"));
            mod.addIncludePath(ggml.path("src"));
            mod.addIncludePath(ggml.path("src/ggml-cpu"));
            mod.addCMacro("CORTEXT_GGML_BACKEND_HEADER_PATH", "\"ggml-backend.h\"");
            mod.addCMacro("GGML_USE_CPU", "1");
            mod.addCMacro("GGML_VERSION", "\"0.9.5\"");
            mod.addCMacro("GGML_COMMIT", "\"ebc3a0f4\"");
            mod.addCMacro("GGML_SCHED_MAX_COPIES", "4");
            mod.addCMacro("_GNU_SOURCE", "1");
            mod.addCMacro("_XOPEN_SOURCE", "600");
            mod.addCMacro("GGML_USE_CPU_REPACK", "1");
            if (target.result.cpu.arch == .x86_64 or target.result.cpu.arch == .x86) {
                // Zig's native x86 feature detection can define AVX-512 BF16
                // without making the corresponding intrinsics available to
                // its C frontend. Build the portable GGML path instead.
                addGgmlSources(mod, ggml, ggml_base_sources, &.{
                    "-w",
                    "-D_GLIBCXX_ASSERTIONS",
                    "-mno-avx512bf16",
                    "-U__AVX512BF16__",
                });
            } else {
                addGgmlSources(mod, ggml, ggml_base_sources, &.{
                    "-w",
                    "-D_GLIBCXX_ASSERTIONS",
                });
            }
            addGgmlSources(mod, ggml, ggml_cpu_sources, &.{
                "-w",
                "-D_GLIBCXX_ASSERTIONS",
            });
            if (target.result.cpu.arch == .x86_64 or target.result.cpu.arch == .x86) {
                addGgmlSources(mod, ggml, ggml_cpu_x86_sources, &.{
                    "-w",
                    "-D_GLIBCXX_ASSERTIONS",
                });
            }
            if (target.result.cpu.arch.isArm() or target.result.cpu.arch.isAARCH64()) {
                addGgmlSources(mod, ggml, ggml_cpu_arm_sources, &.{
                    "-w",
                    "-D_GLIBCXX_ASSERTIONS",
                });
            }
        }
        mod.addCMacro("CORTEXT_ENABLE_AUDIO", "1");
        mod.addCMacro("CORTEXT_ENABLE_IMAGE", "1");
        mod.addCMacro("CORTEXT_REQUIRE_AIST_GGML_KERNELS", "1");
    } else {
        mod.addCMacro("CORTEXT_DISABLE_AUDIO", "1");
        mod.addCMacro("CORTEXT_DISABLE_IMAGE", "1");
    }

    const cxx_flags = &.{
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
    };
    mod.addCSourceFiles(.{ .files = cortext_cpp_sources, .flags = cxx_flags });
    mod.addCSourceFiles(.{ .files = sqlite_c_sources, .flags = &.{"-w"} });
    mod.addCSourceFile(.{ .file = parse_c, .flags = &.{"-w"} });
    mod.addCSourceFile(.{ .file = ctime_c, .flags = &.{"-w"} });
    mod.addCSourceFile(.{ .file = opcodes_c, .flags = &.{"-w"} });
    mod.addCSourceFiles(.{ .files = &.{"third_party/sqlite-vec/sqlite-vec.c"}, .flags = &.{"-w"} });
    mod.addCSourceFiles(.{ .files = objstore_c_sources, .flags = &.{"-w"} });
    if (target.result.cpu.arch.isAARCH64()) {
        mod.addCSourceFiles(.{ .files = &.{"third_party/sqlite-objstore/third_party/blake3/blake3_neon.c"}, .flags = &.{"-w"} });
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

    if (build_cli) {
        const cli_mod = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .link_libcpp = true,
        });
        cli_mod.addIncludePath(b.path("include"));
        cli_mod.addCSourceFile(.{
            .file = b.path("tools/cli/main.cpp"),
            .flags = cxx_flags,
        });
        const cli = b.addExecutable(.{
            .name = "cortext_cli",
            .root_module = cli_mod,
        });
        cli.linkLibrary(lib);
        b.installArtifact(cli);
        check_step.dependOn(&cli.step);
    }

    if (build_node_addon) {
        const node_include_dir = node_include orelse fail(b, "-Dnode-addon=true requires -Dnode-include=/path/to/node/include");
        const addon_mod = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .link_libcpp = true,
        });
        addon_mod.addIncludePath(b.path("include"));
        addon_mod.addIncludePath(.{ .cwd_relative = node_include_dir });
        addon_mod.addCMacro("NAPI_VERSION", "8");
        addon_mod.addCSourceFile(.{
            .file = b.path("bindings/javascript/src/addon.cpp"),
            .flags = cxx_flags,
        });
        if (target.result.os.tag == .windows) {
            const node_import_lib = node_lib orelse fail(b, "Windows -Dnode-addon=true requires -Dnode-lib=/path/to/node.lib");
            addon_mod.addObjectFile(.{ .cwd_relative = node_import_lib });
        }
        const addon = b.addLibrary(.{
            .name = "cortext_node",
            .linkage = .dynamic,
            .root_module = addon_mod,
        });
        addon.linkLibrary(lib);
        if (target.result.os.tag != .windows) {
            addon.linker_allow_shlib_undefined = true;
        }
        b.installArtifact(addon);
        check_step.dependOn(&addon.step);
    }
}

fn fail(b: *std.Build, comptime message: []const u8) noreturn {
    std.log.err(message, .{});
    b.invalid_user_input = true;
    @panic(message);
}

fn addGgmlSources(
    mod: *std.Build.Module,
    dependency: *std.Build.Dependency,
    files: []const []const u8,
    common_flags: []const []const u8,
) void {
    for (files) |file| {
        mod.addCSourceFile(.{
            .file = dependency.path(file),
            .flags = common_flags,
        });
    }
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
