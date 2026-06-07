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
  double label_graph_boost = 0.0;
  int label_match_count = 0;
  double durable_source_boost = 0.0;
  int durable_source_count = 0;
};

struct RetrievalSummary
{
  bool fact_layer_enabled = false;
  int fact_seed_count = 0;
  int candidate_fact_link_memory_count = 0;
  int candidate_fact_link_row_count = 0;
  int selected_fact_linked_count = 0;
  int text_query_token_count = 0;
  int text_query_wm_slots = 0;
  int text_query_wm_chars = 0;
  int fact_text_candidate_count = 0;
  int fact_text_rejected_low_score_count = 0;
  int fact_text_match_count = 0;
  double fact_text_best_score = 0.0;
};

void ClearLastSelectedEmbeddingOrder ();
void SetLastSelectedEmbeddingOrder (const std::vector<long long> &embedding_ids);
const std::vector<long long> &GetLastSelectedEmbeddingOrder ();
void ClearLastRankedCandidates ();
void SetLastRankedCandidates (const std::vector<RankedCandidate> &candidates);
const std::vector<RankedCandidate> &GetLastRankedCandidates ();
void ClearLastRetrievalSummary ();
void SetLastRetrievalSummary (const RetrievalSummary &summary);
RetrievalSummary GetLastRetrievalSummary ();

} // namespace cortext::operations::retrieval_debug
