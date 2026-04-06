#pragma once

#include <vector>

namespace cortext::operations::retrieval_debug
{

struct RankedCandidate
{
  long long embedding_id = 0;
  long long memory_id = 0;
  double score = 0.0;
  double relevance = 0.0;
  double proc_score = 0.0;
  double predictive_bonus = 0.0;
  double pre_activation = 0.0;
  double fact_boost = 0.0;
  double fact_stale_penalty = 0.0;
  int linked_fact_count = 0;
};

void ClearLastSelectedEmbeddingOrder ();
void SetLastSelectedEmbeddingOrder (const std::vector<long long> &embedding_ids);
const std::vector<long long> &GetLastSelectedEmbeddingOrder ();
void ClearLastRankedCandidates ();
void SetLastRankedCandidates (const std::vector<RankedCandidate> &candidates);
const std::vector<RankedCandidate> &GetLastRankedCandidates ();

} // namespace cortext::operations::retrieval_debug
