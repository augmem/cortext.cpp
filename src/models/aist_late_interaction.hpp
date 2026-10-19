#pragma once

#include "cortext/encoder/encoder.hpp"
#include "cortext/signal.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cortext::internal
{

/// @brief Learned late-interaction reranker over frozen AIST position features
/// (tools/reranker/li.py is the training definition; this mirrors it).
///
/// Every item, of any modality, becomes a set of unit vectors in one
/// late-interaction space: its AIST position features (text tokens, image
/// patches) through a per-tower linear map, plus its AIST shared-space
/// retrieval vector (the stored 256-d view) through one global map. Stored
/// signals keep K k-means codes of their positions (int8, ~K*dim bytes);
/// queries keep every position. A memory scores by MaxSim between the query
/// set and each member set (gated weighted mean of per-query-vector maxima),
/// a soft maximum over members, and a rank prior.
class LateInteractionModel
{
public:
  /// @brief Loads the reranker GGUF; throws std::runtime_error when the file
  /// is missing or malformed.
  static std::unique_ptr<LateInteractionModel>
  Load (const std::filesystem::path &path);

  int CodesPerMember () const { return codes_per_member_; }
  int Dim () const { return dim_; }
  /// @brief How many head-of-pool candidates queries rerank.
  int Depth () const { return depth_; }
  int MaxMembers () const { return max_members_; }

  /// @brief Memory-side set: k-means codes of the item positions, mapped,
  /// normalized, int8. Empty set when the tower has no trained map.
  LateInteractionSet MemorySet (const EncoderPositions &positions) const;

  /// @brief Query-side set: every position (capped), mapped, normalized, with
  /// gate logits. Empty set when the tower has no trained map.
  LateInteractionSet QuerySet (const EncoderPositions &positions) const;

  /// @brief Storage blob: [u8 version=1][u8 count][u16 dim LE][int8 codes].
  static std::vector<unsigned char> Serialize (const LateInteractionSet &set);
  static bool Deserialize (const unsigned char *data, std::size_t size,
                           LateInteractionSet &out);

  /// @brief One stored member: its code set (may be empty) and its stored
  /// 256-d retrieval vector (may be empty).
  struct Member
  {
    LateInteractionSet codes;
    Eigen::VectorXf global;
  };

  /// @brief Scores memories in pool order (index = pool rank). A memory with
  /// no members scores below every memory that has one.
  std::vector<float>
  Score (const LateInteractionSet &query, const Eigen::VectorXf &query_global,
         const std::vector<std::vector<Member>> &memories) const;

private:
  struct TowerMap
  {
    int in = 0;
    Eigen::MatrixXf weight;  // [dim x in]
  };

  const TowerMap *MapFor (const std::string &tower) const;
  Eigen::MatrixXf MapRows (const TowerMap &map, const float *rows, int count,
                           int stride) const;
  Eigen::VectorXf GlobalVector (const Eigen::VectorXf &global) const;

  int dim_ = 0;
  int codes_per_member_ = 16;
  int text_cap_ = 256;
  int query_cap_ = 128;
  int kmeans_iters_ = 8;
  int depth_ = 64;
  int max_members_ = 8;
  std::optional<TowerMap> text_;
  std::optional<TowerMap> image_;
  std::optional<TowerMap> audio_;
  Eigen::MatrixXf global_;  // [dim x 256]
  Eigen::VectorXf gate_w_;  // [dim]
  float gate_b_ = 0.0F;
  float global_bias_ = 0.0F;
  float scale_ = 1.0F;
  float member_tau_ = 0.05F;
  float rank_w_ = 0.0F;
};

/// @brief Resolves the reranker GGUF: CORTEXT_LI_MODEL_PATH, else
/// AIST-87M-LI.gguf beside the AIST model, else the shards linked into the
/// library (assembled into the model cache); nullopt when none exists.
std::optional<std::filesystem::path>
ResolveLateInteractionModelPath (const std::filesystem::path &aist_model_path);

} // namespace cortext::internal
