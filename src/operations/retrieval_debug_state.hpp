#pragma once

#include <string>
#include <vector>

namespace cortext::operations::retrieval_debug
{

struct ActivationLedger
{
  double base_level = 0.0;
  double spreading_activation = 0.0;
  double partial_match_penalty = 0.0;
  double recent_inhibition = 0.0;
  double utility = 0.0;
  double exploration_noise = 0.0;
  double activation_total = 0.0;
};

struct RankedCandidate
{
  long long embedding_id = 0;
  long long memory_id = 0;
  double score = 0.0;
  double relevance = 0.0;
  double proc_score = 0.0;
  double predictive_bonus = 0.0;
  double pre_activation = 0.0;
  int label_match_count = 0;
  double durable_source_boost = 0.0;
  int durable_source_count = 0;
  double temporal_score = 0.0;
  double evidence_confidence = 0.0;
  double evidence_weight = 0.0;
  int evidence_source_diversity = 0;
  double evidence_contradiction_mass = 0.0;
  ActivationLedger activation;
};

struct RejectedCandidate
{
  RankedCandidate candidate;
  std::string reason;
  std::string stage;
  double observed = 0.0;
  double threshold = 0.0;
};

struct EvidencePacketMember
{
  int rank = 0;
  long long embedding_id = 0;
  long long memory_id = 0;
  double weight = 0.0;
  double score = 0.0;
  ActivationLedger activation;
  double evidence_confidence = 0.0;
  double evidence_weight = 0.0;
  int evidence_source_diversity = 0;
  double evidence_contradiction_mass = 0.0;
};

struct EvidencePacket
{
  int packet_id = 0;
  std::string consumer;
  std::string reason;
  double tie_margin = 0.0;
  double temperature = 0.0;
  double score_span = 0.0;
  double activation_total = 0.0;
  double evidence_confidence = 0.0;
  double evidence_weight = 0.0;
  int evidence_source_diversity = 0;
  double evidence_contradiction_mass = 0.0;
  std::vector<EvidencePacketMember> members;
};

struct RetrievalSummary
{
  int text_query_token_count = 0;
  int text_query_wm_slots = 0;
  int text_query_wm_chars = 0;
  int rejected_candidate_count = 0;
  int rejected_filter_count = 0;
  int rejected_selection_count = 0;
  int evidence_packet_count = 0;
  int evidence_packet_member_count = 0;
  double evidence_packet_confidence_mean = 0.0;
};

void ClearLastSelectedEmbeddingOrder ();
void SetLastSelectedEmbeddingOrder (const std::vector<long long> &embedding_ids);
const std::vector<long long> &GetLastSelectedEmbeddingOrder ();
void ClearLastRankedCandidates ();
void SetLastRankedCandidates (const std::vector<RankedCandidate> &candidates);
const std::vector<RankedCandidate> &GetLastRankedCandidates ();
void ClearLastRejectedCandidates ();
void
SetLastRejectedCandidates (const std::vector<RejectedCandidate> &candidates);
const std::vector<RejectedCandidate> &GetLastRejectedCandidates ();
void ClearLastEvidencePackets ();
void SetLastEvidencePackets (const std::vector<EvidencePacket> &packets);
const std::vector<EvidencePacket> &GetLastEvidencePackets ();
void ClearLastRetrievalSummary ();
void SetLastRetrievalSummary (const RetrievalSummary &summary);
RetrievalSummary GetLastRetrievalSummary ();

bool CaptureEnabled ();

class ScopedCapture
{
public:
  explicit ScopedCapture (bool enabled);
  ~ScopedCapture ();

  ScopedCapture (const ScopedCapture &) = delete;
  ScopedCapture &operator= (const ScopedCapture &) = delete;

private:
  bool previous_;
};

} // namespace cortext::operations::retrieval_debug
