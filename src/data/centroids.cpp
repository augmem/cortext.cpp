/// @file
#include "cortext/data/centroids.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cortext::data
{

float
AffectCentroids::ComputeValence (const Eigen::VectorXf &emb) const
{
  float sim_pos = emb.dot (valence_positive);
  float sim_neg = emb.dot (valence_negative);
  return std::clamp (sim_pos - sim_neg, -1.0f, 1.0f);
}

float
AffectCentroids::ComputeArousal (const Eigen::VectorXf &emb) const
{
  float sim_high = emb.dot (arousal_high);
  float sim_low = emb.dot (arousal_low);
  return std::clamp ((sim_high - sim_low + 1.0f) / 2.0f, 0.0f, 1.0f);
}

float
AffectCentroids::ComputeGoalAlignment (const Eigen::VectorXf &emb) const
{
  float sim_aligned = emb.dot (goal_aligned);
  float sim_unaligned = emb.dot (goal_unaligned);
  return std::clamp ((sim_aligned - sim_unaligned + 1.0f) / 2.0f, 0.0f, 1.0f);
}

float
AffectCentroids::ComputeViolation (const Eigen::VectorXf &emb) const
{
  float sim_high = emb.dot (violation_high);
  float sim_low = emb.dot (violation_low);
  return std::clamp ((sim_high - sim_low + 1.0f) / 2.0f, 0.0f, 1.0f);
}

std::string
EmotionCentroids::ClassifyEmotion (const Eigen::VectorXf &emb) const
{
  if (centroids.empty ())
    {
      return "";
    }
  std::string best_emotion;
  float best_sim = -std::numeric_limits<float>::infinity ();
  for (const auto &[emotion, centroid] : centroids)
    {
      float sim = emb.dot (centroid);
      if (sim > best_sim)
        {
          best_sim = sim;
          best_emotion = emotion;
        }
    }
  return best_emotion;
}

float
EmotionCentroids::GetEmotionScore (const Eigen::VectorXf &emb,
                                  const std::string &emotion) const
{
  auto it = centroids.find (emotion);
  if (it == centroids.end ())
    {
      return 0.0f;
    }
  float sim = emb.dot (it->second);
  return std::clamp ((sim + 1.0f) / 2.0f, 0.0f, 1.0f);
}

std::vector<std::pair<std::string, float> >
EmotionCentroids::GetTopEmotions (const Eigen::VectorXf &emb,
                                 std::size_t top_k) const
{
  std::vector<std::pair<std::string, float> > scores;
  scores.reserve (centroids.size ());
  for (const auto &[emotion, centroid] : centroids)
    {
      float sim = emb.dot (centroid);
      float score = std::clamp ((sim + 1.0f) / 2.0f, 0.0f, 1.0f);
      scores.emplace_back (emotion, score);
    }
  std::partial_sort (
      scores.begin (), scores.begin () + std::min (top_k, scores.size ()),
      scores.end (),
      [] (const auto &a, const auto &b) { return a.second > b.second; });
  if (scores.size () > top_k)
    {
      scores.resize (top_k);
    }
  return scores;
}

} // namespace cortext::data


