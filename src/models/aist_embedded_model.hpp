#pragma once

#include <filesystem>
#include <optional>

namespace cortext
{

/// @brief True when this build linked Git AIST shards + vocab into the library.
bool HasEmbeddedAistModel ();

/// @brief Assemble linked shards into a full GGUF in the process model cache.
///
/// Git stores only under-100 MiB parts (no LFS). Those parts are `.incbin`'d
/// into `libcortext` at build time. On first use this concatenates them,
/// verifies per-shard and whole-file SHA-256, writes the full `.gguf` + vocab
/// under the cache, and returns the GGUF path. No network download and no
/// `CORTEXT_AIST_MODEL_PATH` are required for default shared-library use.
///
/// When the build has no embedded shards, returns nullopt.
std::optional<std::filesystem::path> MaterializeEmbeddedAistModel ();

/// @brief Assemble the linked late-interaction reranker shards
/// (`AIST-87M-LI.gguf`) into the process model cache and return its path.
///
/// The reranker ships with the AIST model as Git shards linked into
/// `libcortext`; callers that supply their own AIST model path (bindings,
/// `CORTEXT_AIST_MODEL_PATH`) still get it from here when no file sits beside
/// that model. Returns nullopt when the build has no embedded shards.
std::optional<std::filesystem::path> MaterializeEmbeddedLateInteractionModel ();

/// @brief Path to the materialized vocab.txt next to the assembled model cache.
std::optional<std::filesystem::path> MaterializeEmbeddedAistVocab ();

} // namespace cortext
