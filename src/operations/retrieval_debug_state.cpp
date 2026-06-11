#include "retrieval_debug_state.hpp"

namespace cortext::operations::retrieval_debug
{

namespace
{

thread_local std::vector<long long> g_last_selected_embedding_order;
thread_local std::vector<RankedCandidate> g_last_ranked_candidates;
thread_local std::vector<RejectedCandidate> g_last_rejected_candidates;
thread_local std::vector<EvidencePacket> g_last_evidence_packets;
thread_local RetrievalSummary g_last_retrieval_summary;

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

void
ClearLastRejectedCandidates ()
{
  g_last_rejected_candidates.clear ();
}

void
SetLastRejectedCandidates (const std::vector<RejectedCandidate> &candidates)
{
  g_last_rejected_candidates = candidates;
}

const std::vector<RejectedCandidate> &
GetLastRejectedCandidates ()
{
  return g_last_rejected_candidates;
}

void
ClearLastEvidencePackets ()
{
  g_last_evidence_packets.clear ();
}

void
SetLastEvidencePackets (const std::vector<EvidencePacket> &packets)
{
  g_last_evidence_packets = packets;
}

const std::vector<EvidencePacket> &
GetLastEvidencePackets ()
{
  return g_last_evidence_packets;
}

void
ClearLastRetrievalSummary ()
{
  g_last_retrieval_summary = {};
}

void
SetLastRetrievalSummary (const RetrievalSummary &summary)
{
  g_last_retrieval_summary = summary;
}

RetrievalSummary
GetLastRetrievalSummary ()
{
  return g_last_retrieval_summary;
}

} // namespace cortext::operations::retrieval_debug
