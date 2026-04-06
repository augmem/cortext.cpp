#pragma once

#include "cortext/store/store.hpp"
#include "cortext/encoder/encoder.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cortext::store
{

enum class FactQueryMode
{
  Current,
  ValidAt,
  KnownAt
};

enum class FactLifecycleState
{
  Active,
  Weak,
  Archived
};

struct FactLifecycleOptions
{
  bool support_decay_enabled = true;
  bool allow_delete = true;
  bool compress_repeated_confirmations = true;
  bool preserve_high_severity_history = true;
};

class ScopedFactLifecycleOptions
{
public:
  explicit ScopedFactLifecycleOptions (const FactLifecycleOptions &options);
  ~ScopedFactLifecycleOptions ();

  ScopedFactLifecycleOptions (const ScopedFactLifecycleOptions &) = delete;
  ScopedFactLifecycleOptions &
  operator= (const ScopedFactLifecycleOptions &) = delete;

private:
  FactLifecycleOptions previous_;
};

struct FactRecord
{
  long long fact_id = 0;
  std::string subject;
  std::string predicate;
  std::string object;
  std::string canonical_subject;
  std::string canonical_predicate;
  std::string canonical_object;
  std::optional<long long> valid_start_ts;
  std::optional<long long> valid_end_ts;
  long long recorded_at_ts = 0;
  std::optional<long long> superseded_at_ts;
  double confidence = 0.5;
  long long summary_memory_id = 0;
  long long embedding_id = 0;
  std::string fact_text;
  bool is_current = false;
  int evidence_count = 0;
  double support_mass = 0.0;
  int source_diversity = 0;
  double contradiction_mass = 0.0;
  int confirmation_count = 0;
  int challenge_count = 0;
  int compressed_support_count = 0;
  long long last_confirmation_ts = 0;
  std::optional<long long> last_challenge_ts;
  std::string severity_class = "medium";
  FactLifecycleState lifecycle_state = FactLifecycleState::Active;
  std::optional<long long> archived_at;
  long long last_maintenance_ts = 0;
};

struct FactQuery
{
  FactQueryMode mode = FactQueryMode::Current;
  std::optional<std::string> canonical_subject;
  std::optional<std::string> canonical_predicate;
  std::uint64_t timestamp = 0;
};

struct FactScore
{
  double current_boost = 0.0;
  double temporal_match = 0.0;
  double evidence_support = 0.0;
  double routine_support = 0.0;
  double recency_support = 0.0;
  double lifecycle_support = 0.0;
  double supersession_penalty = 0.0;
  double combined = 0.0;
};

FactLifecycleOptions GetFactLifecycleOptions ();
void SetFactLifecycleOptions (const FactLifecycleOptions &options);
void ClearFactLifecycleOptions ();
const char *ToString (FactLifecycleState state);
FactLifecycleState ParseFactLifecycleState (const std::string &raw);
std::string PredicateSeverityClass (const std::string &canonical_predicate);
std::string NormalizeFactTerm (const std::string &raw);
std::string NormalizeFactPredicate (const std::string &raw);
std::string BuildFactText (const std::string &subject,
                           const std::string &predicate,
                           const std::string &object);
void RefreshFactCache (Transaction &tx, Encoder *encoder, long long fact_id,
                       std::uint64_t now_ts);
void MaintainFactLifecycle (Transaction &tx, Encoder *encoder,
                            std::uint64_t now_ts,
                            const std::vector<long long> &priority_fact_ids
                            = {});
std::vector<FactRecord> QueryFacts (Transaction &tx, const FactQuery &query);
FactScore ScoreFactRecord (const FactRecord &record, FactQueryMode mode,
                           std::uint64_t timestamp);

} // namespace cortext::store
