/// @file
#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Parameters for emotional cascade propagation derived from knobs.
struct EmotionCascadeParams
{
  int cascade_radius;    ///< Maximum graph hops for propagation
  double cascade_decay;  ///< Decay factor per hop (0.3-0.7)

  /// @brief Derive parameters from knobs.
  static EmotionCascadeParams FromKnobs (double F, double S, double T);
};

/// @brief Propagates emotional tags through the knowledge graph.
///
/// When high-intensity emotional events are stored, this operation propagates
/// the emotional influence to semantically connected memories via graph
/// traversal. Implements Section 8.7 of the paper:
///
/// ```
/// for each hop k in [1, cascade_radius]:
///   neighbors = graph.traverse(source, depth=k)
///   for neighbor in neighbors:
///     propagated_intensity = source_intensity * cascade_decay^k
///     update emotional_tag(neighbor, propagated_intensity)
/// ```
///
/// Higher Sensitivity (S) increases cascade radius (more hops) and slows
/// decay (emotional influence persists longer through the graph).
class PropagateEmotionalCascade : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
