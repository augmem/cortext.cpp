/// @file
#pragma once

#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>

namespace cortext::data
{

/// @brief Affect centroids for valence, arousal, goal alignment, and violations.
///
/// All centroids are 256-dimensional L2-normalized vectors.
struct AffectCentroids
{
  Eigen::VectorXf valence_positive;  ///< Positive valence centroid (256-dim)
  Eigen::VectorXf valence_negative;  ///< Negative valence centroid (256-dim)
  Eigen::VectorXf arousal_high;      ///< High arousal centroid (256-dim)
  Eigen::VectorXf arousal_low;       ///< Low arousal centroid (256-dim)
  Eigen::VectorXf goal_aligned;      ///< Goal-aligned centroid (256-dim)
  Eigen::VectorXf goal_unaligned;    ///< Goal-unaligned centroid (256-dim)
  Eigen::VectorXf violation_high;    ///< High violation centroid (256-dim)
  Eigen::VectorXf violation_low;     ///< Low violation centroid (256-dim)

  /// @brief Compute valence score in [-1, 1].
  /// @param emb Input embedding (256-dim, L2-normalized)
  /// @return Valence score: 1 = positive, -1 = negative, 0 = neutral
  float ComputeValence (const Eigen::VectorXf &emb) const;

  /// @brief Compute arousal score in [0, 1].
  /// @param emb Input embedding (256-dim, L2-normalized)
  /// @return Arousal score: 1 = high arousal, 0 = low arousal
  float ComputeArousal (const Eigen::VectorXf &emb) const;

  /// @brief Compute goal alignment score in [0, 1].
  /// @param emb Input embedding (256-dim, L2-normalized)
  /// @return Goal alignment: 1 = aligned, 0 = unaligned
  float ComputeGoalAlignment (const Eigen::VectorXf &emb) const;

  /// @brief Compute violation score in [0, 1].
  /// @param emb Input embedding (256-dim, L2-normalized)
  /// @return Violation score: 1 = high violation, 0 = low violation
  float ComputeViolation (const Eigen::VectorXf &emb) const;
};

/// @brief Emotion centroids for classification and scoring.
///
/// Supports multiple emotion labels (anger, joy, fear, sadness, etc.) with
/// 256-dimensional L2-normalized centroids.
struct EmotionCentroids
{
  /// Map from emotion label to centroid vector
  std::unordered_map<std::string, Eigen::VectorXf> centroids;

  /// @brief Classify emotion by finding nearest centroid.
  /// @param emb Input embedding (256-dim, L2-normalized)
  /// @return Emotion label with highest cosine similarity
  std::string ClassifyEmotion (const Eigen::VectorXf &emb) const;

  /// @brief Get similarity score for a specific emotion.
  /// @param emb Input embedding (256-dim, L2-normalized)
  /// @param emotion Emotion label (e.g., "anger", "joy")
  /// @return Cosine similarity in [0, 1], or 0.0 if emotion not found
  float GetEmotionScore (const Eigen::VectorXf &emb,
                         const std::string &emotion) const;

  /// @brief Get all emotion scores sorted by similarity.
  /// @param emb Input embedding (256-dim, L2-normalized)
  /// @return Vector of (emotion, score) pairs, sorted descending by score
  std::vector<std::pair<std::string, float> >
  GetTopEmotions (const Eigen::VectorXf &emb, std::size_t top_k = 5) const;
};

/// @brief Unified centroids bundle for operations.
///
/// Contains affect centroids and the fixed-order emotion centroids used by
/// Algorithm 4 in operations::UpdateSensitivity.
struct Centroids
{
  AffectCentroids affect;
  std::vector<Eigen::VectorXf> emotion_centroids;       ///< 6-emotion fixed order
  std::vector<Eigen::VectorXf> audio_emotion_centroids; ///< 7-emotion for audio
};

/// @brief Load baked centroid vectors compiled into the library.
///
/// The returned centroids contain:
/// - Affect centroids (8 vectors)
/// - Emotion centroids for Algorithm 4 (6 vectors, fixed order)
///
/// @return Centroid vectors copied into owning Eigen containers.
Centroids GetEmbeddedCentroids ();

} // namespace cortext::data


