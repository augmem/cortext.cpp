// tests/centroids.test.cpp
#include <catch2/catch_test_macros.hpp>

#include "cortext/data/centroids.hpp"
#include <Eigen/Dense>
#include <cmath>

using namespace cortext::data;

TEST_CASE ("AffectCentroids: ComputeValence returns values in [-1, 1]",
           "[augmem][affect]")
{
  AffectCentroids centroids;

  // Create mock centroids (normalized)
  centroids.valence_positive = Eigen::VectorXf::Ones (256);
  centroids.valence_positive.normalize ();

  centroids.valence_negative = -Eigen::VectorXf::Ones (256);
  centroids.valence_negative.normalize ();

  // Test with positive-like embedding
  Eigen::VectorXf emb_pos = Eigen::VectorXf::Ones (256);
  emb_pos.normalize ();
  float valence_pos = centroids.ComputeValence (emb_pos);
  REQUIRE (valence_pos >= -1.0f);
  REQUIRE (valence_pos <= 1.0f);
  REQUIRE (valence_pos > 0.0f); // Should be positive

  // Test with negative-like embedding
  Eigen::VectorXf emb_neg = -Eigen::VectorXf::Ones (256);
  emb_neg.normalize ();
  float valence_neg = centroids.ComputeValence (emb_neg);
  REQUIRE (valence_neg >= -1.0f);
  REQUIRE (valence_neg <= 1.0f);
  REQUIRE (valence_neg < 0.0f); // Should be negative
}

TEST_CASE ("AffectCentroids: ComputeArousal returns values in [0, 1]",
           "[augmem][affect]")
{
  AffectCentroids centroids;

  centroids.arousal_high = Eigen::VectorXf::Ones (256);
  centroids.arousal_high.normalize ();

  centroids.arousal_low = -Eigen::VectorXf::Ones (256);
  centroids.arousal_low.normalize ();

  // Test with high arousal embedding
  Eigen::VectorXf emb_high = Eigen::VectorXf::Ones (256);
  emb_high.normalize ();
  float arousal_high = centroids.ComputeArousal (emb_high);
  REQUIRE (arousal_high >= 0.0f);
  REQUIRE (arousal_high <= 1.0f);
  REQUIRE (arousal_high > 0.5f); // Should be high

  // Test with low arousal embedding
  Eigen::VectorXf emb_low = -Eigen::VectorXf::Ones (256);
  emb_low.normalize ();
  float arousal_low = centroids.ComputeArousal (emb_low);
  REQUIRE (arousal_low >= 0.0f);
  REQUIRE (arousal_low <= 1.0f);
  REQUIRE (arousal_low < 0.5f); // Should be low
}

TEST_CASE ("AffectCentroids: ComputeGoalAlignment returns values in [0, 1]",
           "[augmem][affect]")
{
  AffectCentroids centroids;

  centroids.goal_aligned = Eigen::VectorXf::Ones (256);
  centroids.goal_aligned.normalize ();

  centroids.goal_unaligned = -Eigen::VectorXf::Ones (256);
  centroids.goal_unaligned.normalize ();

  Eigen::VectorXf emb_aligned = Eigen::VectorXf::Ones (256);
  emb_aligned.normalize ();
  float goal = centroids.ComputeGoalAlignment (emb_aligned);

  REQUIRE (goal >= 0.0f);
  REQUIRE (goal <= 1.0f);
  REQUIRE (goal > 0.5f); // Should be aligned
}

TEST_CASE ("AffectCentroids: ComputeViolation returns values in [0, 1]",
           "[augmem][affect]")
{
  AffectCentroids centroids;

  centroids.violation_high = Eigen::VectorXf::Ones (256);
  centroids.violation_high.normalize ();

  centroids.violation_low = -Eigen::VectorXf::Ones (256);
  centroids.violation_low.normalize ();

  Eigen::VectorXf emb = Eigen::VectorXf::Ones (256);
  emb.normalize ();
  float violation = centroids.ComputeViolation (emb);

  REQUIRE (violation >= 0.0f);
  REQUIRE (violation <= 1.0f);
}

TEST_CASE ("EmotionCentroids: ClassifyEmotion returns best match",
           "[augmem][emotion]")
{
  EmotionCentroids centroids;

  // Create mock emotion centroids
  Eigen::VectorXf anger_centroid = Eigen::VectorXf::Ones (256);
  anger_centroid.normalize ();
  centroids.centroids["anger"] = anger_centroid;

  Eigen::VectorXf joy_centroid = -Eigen::VectorXf::Ones (256);
  joy_centroid.normalize ();
  centroids.centroids["joy"] = joy_centroid;

  // Test with anger-like embedding
  Eigen::VectorXf emb_anger = Eigen::VectorXf::Ones (256);
  emb_anger.normalize ();
  std::string emotion = centroids.ClassifyEmotion (emb_anger);
  REQUIRE (emotion == "anger");

  // Test with joy-like embedding
  Eigen::VectorXf emb_joy = -Eigen::VectorXf::Ones (256);
  emb_joy.normalize ();
  emotion = centroids.ClassifyEmotion (emb_joy);
  REQUIRE (emotion == "joy");
}

TEST_CASE ("EmotionCentroids: ClassifyEmotion returns empty for no centroids",
           "[augmem][emotion]")
{
  EmotionCentroids centroids; // Empty

  Eigen::VectorXf emb = Eigen::VectorXf::Ones (256);
  emb.normalize ();
  std::string emotion = centroids.ClassifyEmotion (emb);
  REQUIRE (emotion == "");
}

TEST_CASE ("EmotionCentroids: GetEmotionScore returns normalized scores",
           "[augmem][emotion]")
{
  EmotionCentroids centroids;

  Eigen::VectorXf anger_centroid = Eigen::VectorXf::Ones (256);
  anger_centroid.normalize ();
  centroids.centroids["anger"] = anger_centroid;

  Eigen::VectorXf emb = Eigen::VectorXf::Ones (256);
  emb.normalize ();

  float score = centroids.GetEmotionScore (emb, "anger");
  REQUIRE (score >= 0.0f);
  REQUIRE (score <= 1.0f);

  // Unknown emotion should return 0
  float unknown_score = centroids.GetEmotionScore (emb, "unknown");
  REQUIRE (unknown_score == 0.0f);
}

TEST_CASE ("EmotionCentroids: GetTopEmotions returns sorted results",
           "[augmem][emotion]")
{
  EmotionCentroids centroids;

  // Create multiple emotion centroids
  for (int i = 0; i < 5; ++i)
    {
      Eigen::VectorXf centroid = Eigen::VectorXf::Random (256);
      centroid.normalize ();
      centroids.centroids["emotion_" + std::to_string (i)] = centroid;
    }

  Eigen::VectorXf emb = Eigen::VectorXf::Random (256);
  emb.normalize ();

  auto top3 = centroids.GetTopEmotions (emb, 3);

  REQUIRE (top3.size () <= 3);

  // Check scores are sorted descending
  for (std::size_t i = 1; i < top3.size (); ++i)
    {
      REQUIRE (top3[i - 1].second >= top3[i].second);
    }

  // Check all scores are in [0, 1]
  for (const auto &[emotion, score] : top3)
    {
      REQUIRE (score >= 0.0f);
      REQUIRE (score <= 1.0f);
    }
}

TEST_CASE ("EmotionCentroids: GetTopEmotions handles empty centroids",
           "[augmem][emotion]")
{
  EmotionCentroids centroids; // Empty

  Eigen::VectorXf emb = Eigen::VectorXf::Ones (256);
  emb.normalize ();

  auto top = centroids.GetTopEmotions (emb, 5);
  REQUIRE (top.empty ());
}

TEST_CASE ("EmotionCentroids: GetTopEmotions respects top_k limit",
           "[augmem][emotion]")
{
  EmotionCentroids centroids;

  // Create 10 emotions
  for (int i = 0; i < 10; ++i)
    {
      Eigen::VectorXf centroid = Eigen::VectorXf::Random (256);
      centroid.normalize ();
      centroids.centroids["emotion_" + std::to_string (i)] = centroid;
    }

  Eigen::VectorXf emb = Eigen::VectorXf::Random (256);
  emb.normalize ();

  auto top5 = centroids.GetTopEmotions (emb, 5);
  REQUIRE (top5.size () == 5);

  auto top3 = centroids.GetTopEmotions (emb, 3);
  REQUIRE (top3.size () == 3);
}

// Note: Full integration tests with Initialize() would require:
// - A valid encoder models directory
// - local encoder backend enabled
// - Sufficient time for centroid computation
// These tests focus on the core computation logic that can run without external dependencies.
