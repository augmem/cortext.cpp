#pragma once

#include <cstdint>

namespace cortext::core
{

/// @brief Get configured thread count for embedding operations.
///
/// Reads CORTEXT_EMBED_THREADS environment variable. If not set or invalid,
/// defaults to 1/3rd of available CPU cores (minimum 1).
///
/// @return Thread count for embedding operations
uint32_t GetEmbedThreadCount ();

/// @brief Get configured thread count for inference operations.
///
/// Reads CORTEXT_INFER_THREADS environment variable. If not set or invalid,
/// defaults to 1/3rd of available CPU cores (minimum 1).
///
/// @return Thread count for inference operations
uint32_t GetInferThreadCount ();

/// @brief Compute default thread count based on hardware.
///
/// Returns 1/3rd of hardware_concurrency(), with a minimum of 1.
///
/// @return Default thread count
uint32_t DefaultThreadCount ();

} // namespace cortext::core
