#include "facts.hpp"

#include "../operations/temporal_retrieval.hpp"

#include "cortext/core/algorithms.hpp"

#include <algorithm>
#include <any>
#include <chrono>
#include <cctype>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace cortext::store
{

namespace
{

thread_local FactLifecycleOptions g_fact_lifecycle_options;

long long
NowMs ()
{
  return std::chrono::duration_cast<std::chrono::milliseconds> (
             std::chrono::system_clock::now ().time_since_epoch ())
      .count ();
}

long long
ExtractInt64 (const std::map<std::string, std::any> &row, const char *field)
{
  auto it = row.find (field);
  if (it == row.end () || !it->second.has_value ())
    {
      return 0;
    }
  if (it->second.type () == typeid (long long))
    {
      return std::any_cast<long long> (it->second);
    }
  if (it->second.type () == typeid (int))
    {
      return std::any_cast<int> (it->second);
    }
  return 0;
}

double
ExtractDouble (const std::map<std::string, std::any> &row, const char *field,
               double fallback = 0.0)
{
  auto it = row.find (field);
  if (it == row.end () || !it->second.has_value ())
    {
      return fallback;
    }
  if (it->second.type () == typeid (double))
    {
      return std::any_cast<double> (it->second);
    }
  if (it->second.type () == typeid (float))
    {
      return static_cast<double> (std::any_cast<float> (it->second));
    }
  if (it->second.type () == typeid (long long))
    {
      return static_cast<double> (std::any_cast<long long> (it->second));
    }
  if (it->second.type () == typeid (int))
    {
      return static_cast<double> (std::any_cast<int> (it->second));
    }
  return fallback;
}

std::string
ExtractString (const std::map<std::string, std::any> &row, const char *field)
{
  auto it = row.find (field);
  if (it == row.end () || !it->second.has_value ()
      || it->second.type () != typeid (std::string))
    {
      return {};
    }
  return std::any_cast<std::string> (it->second);
}

std::optional<long long>
NullableInt64 (const std::map<std::string, std::any> &row, const char *field)
{
  auto it = row.find (field);
  if (it == row.end () || !it->second.has_value ())
    {
      return std::nullopt;
    }
  if (it->second.type () == typeid (long long))
    {
      return std::any_cast<long long> (it->second);
    }
  if (it->second.type () == typeid (int))
    {
      return static_cast<long long> (std::any_cast<int> (it->second));
    }
  return std::nullopt;
}

std::any
NullableAny (const std::optional<long long> &value)
{
  if (value.has_value ())
    {
      return std::any (*value);
    }
  return std::any ();
}

bool
MatchesCurrent (const FactRecord &record, long long ts)
{
  const bool valid = (!record.valid_start_ts.has_value ()
                      || *record.valid_start_ts <= ts)
                     && (!record.valid_end_ts.has_value ()
                         || *record.valid_end_ts > ts);
  const bool known = record.recorded_at_ts <= ts
                     && (!record.superseded_at_ts.has_value ()
                         || *record.superseded_at_ts > ts);
  return valid && known
         && record.lifecycle_state != FactLifecycleState::Archived;
}

bool
MatchesValidAt (const FactRecord &record, long long ts)
{
  return (!record.valid_start_ts.has_value () || *record.valid_start_ts <= ts)
         && (!record.valid_end_ts.has_value () || *record.valid_end_ts > ts);
}

bool
MatchesKnownAt (const FactRecord &record, long long ts)
{
  return record.recorded_at_ts <= ts
         && (!record.superseded_at_ts.has_value ()
             || *record.superseded_at_ts > ts);
}

double
PredicateRoutineAffinity (const std::string &canonical_predicate)
{
  const auto contains = [&canonical_predicate] (const char *needle) {
    return canonical_predicate.find (needle) != std::string::npos;
  };
  if (contains ("schedule") || contains ("routine") || contains ("home")
      || contains ("household") || contains ("works_at"))
    {
      return 1.0;
    }
  if (contains ("prefer") || contains ("likes") || contains ("favorite"))
    {
      return 0.8;
    }
  if (contains ("medication") || contains ("caregiver")
      || contains ("lives_in") || contains ("location"))
    {
      return 0.35;
    }
  return 0.15;
}

std::string
PredicateSeverityClassImpl (const std::string &canonical_predicate)
{
  const auto contains = [&canonical_predicate] (const char *needle) {
    return canonical_predicate.find (needle) != std::string::npos;
  };
  if (contains ("medication") || contains ("caregiver") || contains ("location")
      || contains ("schedule") || contains ("appointment")
      || contains ("safety") || contains ("destination")
      || contains ("pickup") || contains ("lives_in"))
    {
      return "high";
    }
  if (contains ("routine") || contains ("household") || contains ("home")
      || contains ("social") || contains ("works_at"))
    {
      return "medium";
    }
  return "low";
}

double
SeverityDecayWindow (const std::string &severity_class)
{
  if (severity_class == "high")
    {
      return 18000.0;
    }
  if (severity_class == "medium")
    {
      return 12000.0;
    }
  return 7000.0;
}

double
DeletionGraceWindow (const std::string &severity_class)
{
  if (severity_class == "high")
    {
      return 16000.0;
    }
  if (severity_class == "medium")
    {
      return 9000.0;
    }
  return 4500.0;
}

double
LifecycleMultiplier (FactLifecycleState state, FactQueryMode mode)
{
  switch (state)
    {
    case FactLifecycleState::Active:
      return 1.0;
    case FactLifecycleState::Weak:
      return mode == FactQueryMode::Current ? 0.62 : 0.84;
    case FactLifecycleState::Archived:
      return mode == FactQueryMode::Current ? 0.0 : 0.85;
    }
  return 1.0;
}

double
EvidenceSupportValue (const std::string &evidence_type, double support_weight)
{
  double type_weight = 0.70;
  if (evidence_type == "episodic")
    {
      type_weight = 1.0;
    }
  else if (evidence_type == "summary")
    {
      type_weight = 0.88;
    }
  return core::Clamp (type_weight * core::Clamp (support_weight, 0.20, 1.0),
                      0.0, 1.0);
}

double
ComputeRoutineSupport (const FactRecord &record, long long ts)
{
  const long long anchor_ts
      = record.valid_start_ts.value_or (record.recorded_at_ts);
  const long long duration = std::max (0LL, ts - anchor_ts);
  const double duration_support
      = std::clamp (static_cast<double> (duration) / 8000.0, 0.0, 1.0);
  const double evidence_support
      = std::clamp (0.55 * record.support_mass
                        + 0.45
                              * std::clamp (
                                  static_cast<double> (record.confirmation_count)
                                      / 6.0,
                                  0.0, 1.0),
                    0.0, 1.0);
  return std::clamp (0.55 * duration_support + 0.45 * evidence_support, 0.0,
                     1.0);
}

double
ComputeRecencySupport (const FactRecord &record, long long ts)
{
  const long long anchor
      = record.last_confirmation_ts > 0 ? record.last_confirmation_ts
                                        : record.recorded_at_ts;
  const long long age = std::max (0LL, ts - anchor);
  return std::clamp (1.0 - static_cast<double> (age) / 8000.0, 0.0, 1.0);
}

void
DeleteFactRow (Transaction &tx, long long fact_id)
{
  if (fact_id <= 0)
    {
      return;
    }

  long long embedding_id = 0;
  auto cache_rows
      = tx.Execute ("SELECT embedding_id FROM fact_cache WHERE fact_id = ?",
                    { fact_id });
  if (!cache_rows.empty ())
    {
      embedding_id = ExtractInt64 (cache_rows[0], "embedding_id");
    }

  tx.Execute ("DELETE FROM fact_evidence WHERE fact_id = ?", { fact_id });
  tx.Execute ("DELETE FROM fact_cache WHERE fact_id = ?", { fact_id });
  tx.Execute ("DELETE FROM fact_assertions WHERE fact_id = ?", { fact_id });
  if (embedding_id > 0)
    {
      tx.Execute ("DELETE FROM embeddings WHERE embedding_id = ?",
                  { embedding_id });
    }
}

void
RecomputeFactLifecycle (Transaction &tx, Encoder *encoder, long long fact_id,
                        std::uint64_t now_ts)
{
  if (fact_id <= 0)
    {
      return;
    }

  auto rows = tx.Execute (
      "SELECT fact_id, canonical_subject, canonical_predicate, "
      "canonical_object, valid_start_ts, valid_end_ts, recorded_at_ts, "
      "superseded_at_ts, confidence, confirmation_count, challenge_count, "
      "compressed_support_count, last_confirmation_ts, last_challenge_ts, "
      "COALESCE(severity_class, 'medium') AS severity_class, "
      "COALESCE(lifecycle_state, 'active') AS lifecycle_state, archived_at "
      "FROM fact_assertions WHERE fact_id = ?",
      { fact_id });
  if (rows.empty ())
    {
      return;
    }

  const auto &row = rows[0];
  const std::string canonical_subject
      = ExtractString (row, "canonical_subject");
  const std::string canonical_predicate
      = ExtractString (row, "canonical_predicate");
  const std::string canonical_object
      = ExtractString (row, "canonical_object");
  const auto valid_end_ts = NullableInt64 (row, "valid_end_ts");
  const auto superseded_at_ts = NullableInt64 (row, "superseded_at_ts");
  const double confidence
      = core::Clamp (ExtractDouble (row, "confidence", 0.5), 0.0, 1.0);
  const int confirmation_count
      = static_cast<int> (ExtractInt64 (row, "confirmation_count"));
  const int compressed_support_count
      = static_cast<int> (ExtractInt64 (row, "compressed_support_count"));
  const long long recorded_at_ts = ExtractInt64 (row, "recorded_at_ts");
  const long long last_confirmation_ts
      = std::max (recorded_at_ts, ExtractInt64 (row, "last_confirmation_ts"));
  const auto previous_archived_at = NullableInt64 (row, "archived_at");

  std::string severity_class = ExtractString (row, "severity_class");
  if (severity_class.empty ())
    {
      severity_class = PredicateSeverityClassImpl (canonical_predicate);
    }

  const auto options = GetFactLifecycleOptions ();

  double live_support_mass = 0.0;
  std::unordered_set<long long> live_sources;
  auto evidence_rows = tx.Execute (
      "SELECT fe.source_memory_id, fe.evidence_type, fe.support_weight, "
      "m.memory_id AS live_memory_id "
      "FROM fact_evidence fe "
      "LEFT JOIN memories m ON m.memory_id = fe.source_memory_id "
      "WHERE fe.fact_id = ?",
      { fact_id });
  for (const auto &evidence_row : evidence_rows)
    {
      const long long live_memory_id
          = ExtractInt64 (evidence_row, "live_memory_id");
      if (live_memory_id <= 0)
        {
          continue;
        }
      live_sources.insert (live_memory_id);
      live_support_mass += EvidenceSupportValue (
          ExtractString (evidence_row, "evidence_type"),
          ExtractDouble (evidence_row, "support_weight", 1.0));
    }

  double contradiction_mass = 0.0;
  auto conflicting_rows = tx.Execute (
      "SELECT confidence FROM fact_assertions "
      "WHERE canonical_subject = ? "
      "  AND canonical_predicate = ? "
      "  AND canonical_object <> ? "
      "  AND fact_id <> ? "
      "  AND COALESCE(lifecycle_state, 'active') != 'archived' "
      "  AND (superseded_at_ts IS NULL OR superseded_at_ts > ?)",
      { canonical_subject, canonical_predicate, canonical_object, fact_id,
        static_cast<long long> (now_ts) });
  for (const auto &conflict_row : conflicting_rows)
    {
      contradiction_mass += ExtractDouble (conflict_row, "confidence", 0.0);
    }

  const double evidence_norm = core::Clamp (live_support_mass / 2.2, 0.0, 1.0);
  const double diversity_norm
      = core::Clamp (static_cast<double> (live_sources.size ()) / 3.0, 0.0,
                     1.0);
  const int repeated_count
      = std::max (confirmation_count, compressed_support_count + 1);
  const double repeated_bonus
      = options.compress_repeated_confirmations
            ? core::Clamp (
                  std::log1p (static_cast<double> (std::max (0, repeated_count - 1)))
                      / std::log1p (20.0),
                  0.0, 1.0)
            : core::Clamp (
                  static_cast<double> (std::max (0, repeated_count - 1)) / 6.0,
                  0.0, 1.0);
  double recency_factor = 1.0;
  if (options.support_decay_enabled)
    {
      const long long age
          = std::max (0LL,
                      static_cast<long long> (now_ts) - last_confirmation_ts);
      const double floor = severity_class == "high" ? 0.30 : 0.0;
      recency_factor
          = core::Clamp (1.0 - static_cast<double> (age)
                                   / SeverityDecayWindow (severity_class),
                         floor, 1.0);
    }

  const double contradiction_norm
      = core::Clamp (contradiction_mass / 1.6, 0.0, 1.0);
  double support_score
      = (0.38 * confidence + 0.30 * evidence_norm + 0.17 * diversity_norm
         + 0.15 * repeated_bonus)
        * (0.45 + 0.55 * recency_factor)
        - 0.18 * contradiction_norm;
  const bool open_current
      = !valid_end_ts.has_value () && !superseded_at_ts.has_value ();
  if (live_sources.empty () && severity_class == "high"
      && options.preserve_high_severity_history && open_current)
    {
      support_score = std::max (support_score, 0.22);
    }
  support_score = core::Clamp (support_score, 0.0, 1.0);

  FactLifecycleState lifecycle_state = FactLifecycleState::Archived;
  const double active_threshold = severity_class == "high" ? 0.45 : 0.52;
  const double weak_threshold = severity_class == "low" ? 0.22 : 0.18;
  if (support_score >= active_threshold)
    {
      lifecycle_state = FactLifecycleState::Active;
    }
  else if (support_score >= weak_threshold)
    {
      lifecycle_state = FactLifecycleState::Weak;
    }
  if (severity_class == "high" && options.preserve_high_severity_history
      && lifecycle_state == FactLifecycleState::Archived && open_current)
    {
      lifecycle_state = FactLifecycleState::Weak;
    }
  if (open_current && live_sources.empty () && severity_class != "low")
    {
      lifecycle_state = FactLifecycleState::Weak;
    }

  std::optional<long long> archived_at = previous_archived_at;
  if (lifecycle_state == FactLifecycleState::Archived)
    {
      if (!archived_at.has_value ())
        {
          archived_at = static_cast<long long> (now_ts);
        }
    }
  else
    {
      archived_at = std::nullopt;
    }

  bool should_delete = false;
  if (options.allow_delete && archived_at.has_value ())
    {
      const long long archived_age
          = std::max (0LL, static_cast<long long> (now_ts) - *archived_at);
      const bool low_value = severity_class == "low";
      const bool superseded_or_closed
          = superseded_at_ts.has_value () || valid_end_ts.has_value ();
      const bool allow_high_delete
          = !options.preserve_high_severity_history && severity_class == "high";
      const double delete_threshold
          = (low_value || allow_high_delete) ? 0.18 : 0.08;
      if ((low_value || allow_high_delete)
          && archived_age
                 >= static_cast<long long> (DeletionGraceWindow (severity_class))
          && (superseded_or_closed || live_sources.empty ())
          && support_score <= delete_threshold)
        {
          should_delete = true;
        }
    }

  if (should_delete)
    {
      DeleteFactRow (tx, fact_id);
      return;
    }

  tx.Execute (
      "UPDATE fact_assertions "
      "SET support_mass = ?, "
      "    source_diversity = ?, "
      "    contradiction_mass = ?, "
      "    severity_class = ?, "
      "    lifecycle_state = ?, "
      "    archived_at = ?, "
      "    last_maintenance_ts = ? "
      "WHERE fact_id = ?",
      { support_score,
        static_cast<long long> (live_sources.size ()),
        contradiction_mass,
        severity_class,
        std::string (ToString (lifecycle_state)),
        NullableAny (archived_at),
        static_cast<long long> (now_ts),
        fact_id });
  RefreshFactCache (tx, encoder, fact_id, now_ts);
}

std::vector<long long>
BuildLifecycleCandidates (Transaction &tx,
                          const std::vector<long long> &priority_fact_ids)
{
  constexpr std::size_t kSweepLimit = 256;
  std::vector<long long> candidates;
  candidates.reserve (kSweepLimit);
  std::unordered_set<long long> seen;

  for (const long long fact_id : priority_fact_ids)
    {
      if (fact_id > 0 && seen.insert (fact_id).second)
        {
          candidates.push_back (fact_id);
        }
    }

  auto rows = tx.Execute (
      "SELECT fact_id FROM fact_assertions "
      "ORDER BY CASE COALESCE(lifecycle_state, 'active') "
      "           WHEN 'archived' THEN 0 "
      "           WHEN 'weak' THEN 1 "
      "           ELSE 2 END, "
      "         COALESCE(last_maintenance_ts, 0) ASC, "
      "         recorded_at_ts ASC "
      "LIMIT ?",
      { static_cast<long long> (kSweepLimit) });
  for (const auto &row : rows)
    {
      const long long fact_id = ExtractInt64 (row, "fact_id");
      if (fact_id > 0 && seen.insert (fact_id).second)
        {
          candidates.push_back (fact_id);
        }
    }

  return candidates;
}

} // namespace

ScopedFactLifecycleOptions::ScopedFactLifecycleOptions (
    const FactLifecycleOptions &options)
    : previous_ (GetFactLifecycleOptions ())
{
  SetFactLifecycleOptions (options);
}

ScopedFactLifecycleOptions::~ScopedFactLifecycleOptions ()
{
  g_fact_lifecycle_options = previous_;
}

FactLifecycleOptions
GetFactLifecycleOptions ()
{
  return g_fact_lifecycle_options;
}

void
SetFactLifecycleOptions (const FactLifecycleOptions &options)
{
  g_fact_lifecycle_options = options;
}

void
ClearFactLifecycleOptions ()
{
  g_fact_lifecycle_options = {};
}

const char *
ToString (FactLifecycleState state)
{
  switch (state)
    {
    case FactLifecycleState::Active:
      return "active";
    case FactLifecycleState::Weak:
      return "weak";
    case FactLifecycleState::Archived:
      return "archived";
    }
  return "active";
}

FactLifecycleState
ParseFactLifecycleState (const std::string &raw)
{
  if (raw == "weak")
    {
      return FactLifecycleState::Weak;
    }
  if (raw == "archived")
    {
      return FactLifecycleState::Archived;
    }
  return FactLifecycleState::Active;
}

std::string
PredicateSeverityClass (const std::string &canonical_predicate)
{
  return PredicateSeverityClassImpl (canonical_predicate);
}

std::string
NormalizeFactTerm (const std::string &raw)
{
  std::string norm;
  norm.reserve (raw.size ());
  bool previous_sep = false;
  for (unsigned char c : raw)
    {
      if (std::isalnum (c))
        {
          norm.push_back (static_cast<char> (std::tolower (c)));
          previous_sep = false;
        }
      else if (!norm.empty () && !previous_sep)
        {
          norm.push_back ('_');
          previous_sep = true;
        }
    }
  while (!norm.empty () && norm.back () == '_')
    {
      norm.pop_back ();
    }
  return norm;
}

std::string
NormalizeFactPredicate (const std::string &raw)
{
  return NormalizeFactTerm (raw);
}

std::string
BuildFactText (const std::string &subject, const std::string &predicate,
               const std::string &object)
{
  return subject + " " + predicate + " " + object;
}

void
RefreshFactCache (Transaction &tx, Encoder *encoder, long long fact_id,
                  std::uint64_t now_ts)
{
  if (fact_id <= 0)
    {
      return;
    }

  auto rows = tx.Execute (
      "SELECT subject, predicate, object, valid_start_ts, valid_end_ts, "
      "recorded_at_ts, superseded_at_ts, "
      "COALESCE(lifecycle_state, 'active') AS lifecycle_state, "
      "COALESCE(support_mass, 0.0) AS support_mass "
      "FROM fact_assertions WHERE fact_id = ?",
      { fact_id });
  if (rows.empty ())
    {
      return;
    }

  const auto &row = rows[0];
  const std::string subject = ExtractString (row, "subject");
  const std::string predicate = ExtractString (row, "predicate");
  const std::string object = ExtractString (row, "object");
  const auto valid_start_ts = NullableInt64 (row, "valid_start_ts");
  const auto valid_end_ts = NullableInt64 (row, "valid_end_ts");
  const long long recorded_at_ts = ExtractInt64 (row, "recorded_at_ts");
  const auto superseded_at_ts = NullableInt64 (row, "superseded_at_ts");
  const FactLifecycleState lifecycle_state
      = ParseFactLifecycleState (ExtractString (row, "lifecycle_state"));
  const double support_mass = ExtractDouble (row, "support_mass", 0.0);
  const std::string fact_text = BuildFactText (subject, predicate, object);

  long long embedding_id = 0;
  auto cache_rows
      = tx.Execute ("SELECT embedding_id FROM fact_cache WHERE fact_id = ?",
                    { fact_id });
  if (!cache_rows.empty ())
    {
      embedding_id = ExtractInt64 (cache_rows[0], "embedding_id");
    }

  if (embedding_id <= 0 && encoder != nullptr)
    {
      std::vector<float> embedding;
      try
        {
          encoder->EncodeText (fact_text, embedding);
        }
      catch (const std::exception &)
        {
          embedding.clear ();
        }
      if (!embedding.empty ())
        {
          tx.Execute ("INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
                      { embedding, static_cast<long long> (now_ts) });
          auto emb_rows
              = tx.Execute ("SELECT last_insert_rowid() AS id", {});
          if (!emb_rows.empty ())
            {
              embedding_id = ExtractInt64 (emb_rows[0], "id");
            }
        }
    }

  const long long now = static_cast<long long> (now_ts);
  const bool is_current
      = lifecycle_state != FactLifecycleState::Archived
        && (!valid_start_ts.has_value () || *valid_start_ts <= now)
        && (!valid_end_ts.has_value () || *valid_end_ts > now)
        && recorded_at_ts <= now
        && (!superseded_at_ts.has_value () || *superseded_at_ts > now);

  tx.Execute (
      "INSERT OR REPLACE INTO fact_cache "
      "(fact_id, embedding_id, fact_text, is_current, valid_start_ts, "
      " valid_end_ts, recorded_at_ts, superseded_at_ts, lifecycle_state, "
      " support_mass, updated_at) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      { fact_id,
        embedding_id > 0 ? std::any (embedding_id) : std::any (),
        fact_text,
        is_current ? 1 : 0,
        NullableAny (valid_start_ts),
        NullableAny (valid_end_ts),
        recorded_at_ts,
        NullableAny (superseded_at_ts),
        std::string (ToString (lifecycle_state)),
        support_mass,
        static_cast<long long> (now_ts) });
}

void
MaintainFactLifecycle (Transaction &tx, Encoder *encoder, std::uint64_t now_ts,
                       const std::vector<long long> &priority_fact_ids)
{
  const auto candidates = BuildLifecycleCandidates (tx, priority_fact_ids);
  for (const long long fact_id : candidates)
    {
      RecomputeFactLifecycle (tx, encoder, fact_id, now_ts);
    }
}

std::vector<FactRecord>
QueryFacts (Transaction &tx, const FactQuery &query)
{
  const long long ts
      = query.timestamp == 0 ? NowMs ()
                             : static_cast<long long> (query.timestamp);
  std::string sql =
      "SELECT fa.fact_id, fa.subject, fa.predicate, fa.object, "
      "fa.canonical_subject, fa.canonical_predicate, fa.canonical_object, "
      "fa.valid_start_ts, fa.valid_end_ts, fa.recorded_at_ts, "
      "fa.superseded_at_ts, fa.confidence, fa.summary_memory_id, "
      "COALESCE(fc.embedding_id, 0) AS embedding_id, "
      "COALESCE(fc.fact_text, '') AS fact_text, "
      "COALESCE(fc.is_current, 0) AS is_current, "
      "(SELECT COUNT(*) FROM fact_evidence fe "
      " LEFT JOIN memories m ON m.memory_id = fe.source_memory_id "
      " WHERE fe.fact_id = fa.fact_id AND m.memory_id IS NOT NULL) "
      "  AS evidence_count, "
      "COALESCE(fa.support_mass, 0.0) AS support_mass, "
      "COALESCE(fa.source_diversity, 0) AS source_diversity, "
      "COALESCE(fa.contradiction_mass, 0.0) AS contradiction_mass, "
      "COALESCE(fa.confirmation_count, 0) AS confirmation_count, "
      "COALESCE(fa.challenge_count, 0) AS challenge_count, "
      "COALESCE(fa.compressed_support_count, 0) AS compressed_support_count, "
      "COALESCE(fa.last_confirmation_ts, 0) AS last_confirmation_ts, "
      "fa.last_challenge_ts, "
      "COALESCE(fa.severity_class, 'medium') AS severity_class, "
      "COALESCE(fa.lifecycle_state, 'active') AS lifecycle_state, "
      "fa.archived_at, "
      "COALESCE(fa.last_maintenance_ts, 0) AS last_maintenance_ts "
      "FROM fact_assertions fa "
      "LEFT JOIN fact_cache fc ON fc.fact_id = fa.fact_id "
      "WHERE 1=1";
  std::vector<std::any> params;

  if (query.canonical_subject.has_value ())
    {
      sql += " AND fa.canonical_subject = ?";
      params.push_back (*query.canonical_subject);
    }
  if (query.canonical_predicate.has_value ())
    {
      sql += " AND fa.canonical_predicate = ?";
      params.push_back (*query.canonical_predicate);
    }

  switch (query.mode)
    {
    case FactQueryMode::Current:
      sql += " AND COALESCE(fa.lifecycle_state, 'active') != 'archived'"
             " AND (fa.valid_start_ts IS NULL OR fa.valid_start_ts <= ?)"
             " AND (fa.valid_end_ts IS NULL OR fa.valid_end_ts > ?)"
             " AND fa.recorded_at_ts <= ?"
             " AND (fa.superseded_at_ts IS NULL OR fa.superseded_at_ts > ?)";
      params.push_back (ts);
      params.push_back (ts);
      params.push_back (ts);
      params.push_back (ts);
      break;
    case FactQueryMode::ValidAt:
      sql += " AND (fa.valid_start_ts IS NULL OR fa.valid_start_ts <= ?)"
             " AND (fa.valid_end_ts IS NULL OR fa.valid_end_ts > ?)";
      params.push_back (ts);
      params.push_back (ts);
      break;
    case FactQueryMode::KnownAt:
      sql += " AND fa.recorded_at_ts <= ?"
             " AND (fa.superseded_at_ts IS NULL OR fa.superseded_at_ts > ?)";
      params.push_back (ts);
      params.push_back (ts);
      break;
    }

  sql += " ORDER BY fa.canonical_subject ASC, fa.canonical_predicate ASC, "
         "fa.recorded_at_ts DESC";

  std::vector<FactRecord> out;
  auto rows = tx.Execute (sql, params);
  out.reserve (rows.size ());
  for (const auto &row : rows)
    {
      FactRecord record;
      record.fact_id = ExtractInt64 (row, "fact_id");
      record.subject = ExtractString (row, "subject");
      record.predicate = ExtractString (row, "predicate");
      record.object = ExtractString (row, "object");
      record.canonical_subject = ExtractString (row, "canonical_subject");
      record.canonical_predicate
          = ExtractString (row, "canonical_predicate");
      record.canonical_object = ExtractString (row, "canonical_object");
      record.valid_start_ts = NullableInt64 (row, "valid_start_ts");
      record.valid_end_ts = NullableInt64 (row, "valid_end_ts");
      record.recorded_at_ts = ExtractInt64 (row, "recorded_at_ts");
      record.superseded_at_ts = NullableInt64 (row, "superseded_at_ts");
      record.confidence = ExtractDouble (row, "confidence", 0.5);
      record.summary_memory_id = ExtractInt64 (row, "summary_memory_id");
      record.embedding_id = ExtractInt64 (row, "embedding_id");
      record.fact_text = ExtractString (row, "fact_text");
      record.is_current = ExtractInt64 (row, "is_current") != 0;
      record.evidence_count
          = static_cast<int> (ExtractInt64 (row, "evidence_count"));
      record.support_mass = ExtractDouble (row, "support_mass", 0.0);
      record.source_diversity
          = static_cast<int> (ExtractInt64 (row, "source_diversity"));
      record.contradiction_mass
          = ExtractDouble (row, "contradiction_mass", 0.0);
      record.confirmation_count
          = static_cast<int> (ExtractInt64 (row, "confirmation_count"));
      record.challenge_count
          = static_cast<int> (ExtractInt64 (row, "challenge_count"));
      record.compressed_support_count
          = static_cast<int> (ExtractInt64 (row, "compressed_support_count"));
      record.last_confirmation_ts
          = ExtractInt64 (row, "last_confirmation_ts");
      record.last_challenge_ts = NullableInt64 (row, "last_challenge_ts");
      record.severity_class = ExtractString (row, "severity_class");
      record.lifecycle_state
          = ParseFactLifecycleState (ExtractString (row, "lifecycle_state"));
      record.archived_at = NullableInt64 (row, "archived_at");
      record.last_maintenance_ts
          = ExtractInt64 (row, "last_maintenance_ts");
      out.push_back (std::move (record));
    }
  return out;
}

FactScore
ScoreFactRecord (const FactRecord &record, FactQueryMode mode,
                 std::uint64_t timestamp)
{
  const long long ts
      = timestamp == 0 ? NowMs () : static_cast<long long> (timestamp);

  FactScore score;
  score.current_boost = (record.is_current
                         && record.lifecycle_state != FactLifecycleState::Archived)
                            ? 1.0
                            : 0.0;
  switch (mode)
    {
    case FactQueryMode::Current:
      score.temporal_match = MatchesCurrent (record, ts) ? 1.0 : 0.0;
      break;
    case FactQueryMode::ValidAt:
      score.temporal_match = MatchesValidAt (record, ts) ? 1.0 : 0.0;
      break;
    case FactQueryMode::KnownAt:
      score.temporal_match = MatchesKnownAt (record, ts) ? 1.0 : 0.0;
      break;
    }

  score.evidence_support
      = core::Clamp (0.70 * record.support_mass
                         + 0.30
                               * std::clamp (
                                   static_cast<double> (record.evidence_count)
                                       / 5.0,
                                   0.0, 1.0),
                     0.0, 1.0);
  score.routine_support = ComputeRoutineSupport (record, ts);
  score.recency_support = ComputeRecencySupport (record, ts);
  score.lifecycle_support = LifecycleMultiplier (record.lifecycle_state, mode);
  score.supersession_penalty
      = record.superseded_at_ts.has_value () ? 1.0 : 0.0;
  const double routine_affinity
      = PredicateRoutineAffinity (record.canonical_predicate);
  const double base = 0.40 * score.temporal_match
                      + 0.22 * std::clamp (record.confidence, 0.0, 1.0)
                      + 0.24 * score.evidence_support
                      + 0.10 * score.current_boost
                      + 0.04
                            * std::clamp (
                                static_cast<double> (record.source_diversity)
                                    / 3.0,
                                0.0, 1.0)
                      - 0.18 * score.supersession_penalty;
  double routine_recency_adjust = 0.0;
  const auto override = operations::temporal::GetRetrievalAblationOverride ();
  switch (override.routine_recency_mode.value_or (
      operations::temporal::RoutineRecencyMode::Balanced))
    {
    case operations::temporal::RoutineRecencyMode::Balanced:
      break;
    case operations::temporal::RoutineRecencyMode::RoutineBiased:
      routine_recency_adjust
          = 0.30 * routine_affinity * score.routine_support
            - 0.12 * score.recency_support;
      break;
    case operations::temporal::RoutineRecencyMode::RecencyBiased:
      routine_recency_adjust
          = 0.34 * score.recency_support
            - 0.16 * routine_affinity * score.routine_support;
      break;
    }
  score.combined = std::clamp (
      (base + routine_recency_adjust) * score.lifecycle_support, 0.0, 1.0);
  return score;
}

} // namespace cortext::store
