#pragma once

#include "cortext/store/store.hpp"
#include <Eigen/Dense>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cortext::operations::constructive_recall
{

struct ReconstructionRecord
{
  long long reconstruction_id = 0;
  long long memory_id = 0;
  long long embedding_id = 0;
  std::vector<unsigned char> blob_id;
  double uncertainty = 0.0;
  long long created_at = 0;
  std::string trigger;
};

bool Disabled ();

std::optional<ReconstructionRecord>
LoadLatestReconstruction (Store *store, long long memory_id);

std::optional<ReconstructionRecord>
LoadLatestReconstruction (Transaction &tx, long long memory_id);

std::vector<unsigned char>
LoadCurrentBlobId (Store *store, long long memory_id);

std::vector<unsigned char>
LoadCurrentBlobId (Transaction &tx, long long memory_id);

std::optional<Eigen::VectorXf>
LoadCurrentEmbedding (Store *store, long long memory_id,
                      long long base_embedding_id, int expected_dim);

std::optional<Eigen::VectorXf>
LoadCurrentEmbedding (Transaction &tx, long long memory_id,
                      long long base_embedding_id, int expected_dim);

long long
AppendReconstructionWithEmbeddingId (Transaction &tx, long long memory_id,
                                     long long embedding_id,
                                     const std::vector<unsigned char> &blob_id,
                                     long long created_at, double uncertainty,
                                     std::string_view trigger,
                                     double source_confidence = 1.0,
                                     double context_similarity = 0.0);

long long
AppendReconstructionWithEmbedding (Transaction &tx, long long memory_id,
                                   const Eigen::VectorXf &embedding,
                                   const std::vector<unsigned char> &blob_id,
                                   long long created_at, double uncertainty,
                                   std::string_view trigger,
                                   double source_confidence = 1.0,
                                   double context_similarity = 0.0);

} // namespace cortext::operations::constructive_recall
