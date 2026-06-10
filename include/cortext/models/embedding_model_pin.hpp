#pragma once

#include <filesystem>
#include <string>

namespace cortext::models
{

/// @brief Identity pin of the embedding model that produced precomputed vectors.
///
/// Format: "<backend_name>:b3-<blake3/16hex of model file>:dim<N>".
/// Every derived artifact (live database embeddings, label bank, embedded
/// centroids) is stamped with the fingerprint of the encoder that produced
/// it; the runtime hard-fails on mismatch instead of silently comparing
/// vectors across spaces.
std::string
ComputeEmbeddingModelPin (const std::string &backend_name,
                                  const std::filesystem::path &model_path,
                                  int dim);

} // namespace cortext::models
