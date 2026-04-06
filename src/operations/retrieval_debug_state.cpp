#include "retrieval_debug_state.hpp"

namespace cortext::operations::retrieval_debug
{

namespace
{

thread_local std::vector<long long> g_last_selected_embedding_order;
thread_local std::vector<RankedCandidate> g_last_ranked_candidates;

} // namespace

void
ClearLastSelectedEmbeddingOrder ()
{
  g_last_selected_embedding_order.clear ();
}

void
SetLastSelectedEmbeddingOrder (const std::vector<long long> &embedding_ids)
{
  g_last_selected_embedding_order = embedding_ids;
}

const std::vector<long long> &
GetLastSelectedEmbeddingOrder ()
{
  return g_last_selected_embedding_order;
}

void
ClearLastRankedCandidates ()
{
  g_last_ranked_candidates.clear ();
}

void
SetLastRankedCandidates (const std::vector<RankedCandidate> &candidates)
{
  g_last_ranked_candidates = candidates;
}

const std::vector<RankedCandidate> &
GetLastRankedCandidates ()
{
  return g_last_ranked_candidates;
}

} // namespace cortext::operations::retrieval_debug
