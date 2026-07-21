#include "retrieval_trace_state.hpp"

namespace cortext::operations::retrieval_trace
{

namespace
{

thread_local std::vector<long long> g_last_selected_embedding_order;
thread_local std::vector<RankedCandidate> g_last_ranked_candidates;
thread_local std::vector<RankedCandidate> g_last_seed_candidates;
thread_local std::vector<RankedCandidate> g_last_exact_seed_candidates;
thread_local std::vector<RejectedCandidate> g_last_rejected_candidates;
thread_local std::vector<EvidencePacket> g_last_evidence_packets;
thread_local RetrievalSummary g_last_retrieval_summary;
thread_local std::vector<SurfaceMutation> g_surface_mutations;
thread_local bool g_surface_mutation_capture_enabled = false;
#ifdef CORTEXT_TESTING
thread_local std::size_t g_last_family_exact_comparison_count = 0;
thread_local std::size_t g_last_sql_fallback_query_count = 0;
thread_local std::size_t g_last_sql_fallback_materialized_row_count = 0;
#endif
thread_local bool g_capture_enabled = true;

} // namespace

void
SetSurfaceMutationCaptureEnabled (bool enabled)
{
  g_surface_mutation_capture_enabled = enabled;
  if (!enabled)
    g_surface_mutations.clear ();
}

bool
SurfaceMutationCaptureEnabled ()
{
  return g_surface_mutation_capture_enabled;
}

void
ClearSurfaceMutations ()
{
  g_surface_mutations.clear ();
}

void
RecordSurfaceUpsert (long long memory_id, long long embedding_id,
                     std::vector<float> embedding)
{
  if (!g_surface_mutation_capture_enabled || memory_id <= 0
      || embedding_id <= 0 || embedding.empty ())
    return;
  g_surface_mutations.push_back (
      { SurfaceMutation::Action::Upsert, memory_id, embedding_id,
        std::move (embedding) });
}

void
RecordSurfaceRemove (long long memory_id)
{
  if (!g_surface_mutation_capture_enabled || memory_id <= 0)
    return;
  g_surface_mutations.push_back (
      { SurfaceMutation::Action::Remove, memory_id, 0, {} });
}

const std::vector<SurfaceMutation> &
GetSurfaceMutations ()
{
  return g_surface_mutations;
}

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
ClearLastSeedCandidates ()
{
  g_last_seed_candidates.clear ();
}

void
SetLastSeedCandidates (const std::vector<RankedCandidate> &candidates)
{
  g_last_seed_candidates = candidates;
}

const std::vector<RankedCandidate> &
GetLastSeedCandidates ()
{
  return g_last_seed_candidates;
}

void
ClearLastExactSeedCandidates ()
{
  g_last_exact_seed_candidates.clear ();
}

void
SetLastExactSeedCandidates (
    const std::vector<RankedCandidate> &candidates)
{
  g_last_exact_seed_candidates = candidates;
}

const std::vector<RankedCandidate> &
GetLastExactSeedCandidates ()
{
  return g_last_exact_seed_candidates;
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

void
ClearLastFamilyExactComparisonCount ()
{
#ifdef CORTEXT_TESTING
  g_last_family_exact_comparison_count = 0;
#endif
}

void
IncrementLastFamilyExactComparisonCount ()
{
#ifdef CORTEXT_TESTING
  ++g_last_family_exact_comparison_count;
#endif
}

std::size_t
GetLastFamilyExactComparisonCount ()
{
#ifdef CORTEXT_TESTING
  return g_last_family_exact_comparison_count;
#else
  return 0;
#endif
}

void
ClearLastSqlFallbackQueryCount ()
{
#ifdef CORTEXT_TESTING
  g_last_sql_fallback_query_count = 0;
#endif
}

void
IncrementLastSqlFallbackQueryCount ()
{
#ifdef CORTEXT_TESTING
  ++g_last_sql_fallback_query_count;
#endif
}

std::size_t
GetLastSqlFallbackQueryCount ()
{
#ifdef CORTEXT_TESTING
  return g_last_sql_fallback_query_count;
#else
  return 0;
#endif
}

void
ClearLastSqlFallbackMaterializedRowCount ()
{
#ifdef CORTEXT_TESTING
  g_last_sql_fallback_materialized_row_count = 0;
#endif
}

void
SetLastSqlFallbackMaterializedRowCount (std::size_t count)
{
#ifdef CORTEXT_TESTING
  g_last_sql_fallback_materialized_row_count = count;
#else
  (void)count;
#endif
}

std::size_t
GetLastSqlFallbackMaterializedRowCount ()
{
#ifdef CORTEXT_TESTING
  return g_last_sql_fallback_materialized_row_count;
#else
  return 0;
#endif
}

bool
CaptureEnabled ()
{
  return g_capture_enabled;
}

ScopedCapture::ScopedCapture (bool enabled)
    : previous_ (g_capture_enabled)
{
  g_capture_enabled = enabled;
}

ScopedCapture::~ScopedCapture ()
{
  g_capture_enabled = previous_;
}

} // namespace cortext::operations::retrieval_trace
