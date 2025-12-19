/// @file
#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Parameters for concept node detection derived from knobs.
struct ConceptDetectionParams
{
  int min_episodes;              ///< Minimum episodes for concept detection
  int entity_frequency_threshold; ///< Minimum entity mentions for concept

  /// @brief Derive parameters from knobs.
  static ConceptDetectionParams FromKnobs (double F, double S, double T);
};

/// @brief Detects concept nodes from recurrent entities across episodes.
///
/// Concept nodes represent emergent centroids of recurrent topics that appear
/// across multiple episodes. This operation:
/// 1. Queries LABEL memories (entities) that appear in N+ distinct episodes
/// 2. Filters by entity frequency threshold
/// 3. Creates concept LABEL memories with computed centroids
/// 4. Links concepts to related entities with 'generalizes' edges in ASSOCIATIONS
/// 5. Inserts concept embeddings for vector search
///
/// V2 schema: Uses MEMORIES (kind='LABEL') and ASSOCIATIONS instead of
/// legacy graph_nodes/graph_edges tables.
class DetectConceptNodes : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
