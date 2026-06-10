#include "../../src/operations/retrieval_debug_state.hpp"
#include "../../src/operations/temporal_retrieval.hpp"
#include "../../src/store/facts.hpp"
#include "include/benchmark_text_encoder.hpp"

#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/operations/process_extraction_results.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/signal.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <any>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace
{
std::string g_models_dir = "models";

using BenchEncoder = cortext::benchmark::EmbeddingGemmaBenchEncoder;

std::vector<float>
ToFloatVec (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

cortext::Signal
MakeSignal (const Eigen::VectorXf &embedding, std::uint64_t ts)
{
  cortext::Signal s;
  s.embedding = embedding;
  s.timestamp = ts;
  s.source_id = "bench";
  return s;
}

class ForceRetrievalGateOp : public cortext::IOperation
{
public:
  void
  Execute (cortext::OperationContext &ctx,
           cortext::Transaction & /*tx*/) const override
  {
    ctx.SetShouldCheckRetrieval (true);
    auto &p_ctx = ctx.GetProcessorContext ();
    if (p_ctx.memory_stream.empty ())
      {
        p_ctx.memory_stream.push_back (ctx.GetSignal ().embedding);
      }
    auto &acc = p_ctx.accumulator_states[ctx.GetSignal ().source_id];
    acc.mu_acc = ctx.GetSignal ().embedding;
    acc.c_t = ctx.GetSignal ().embedding;
  }
};

std::string
BuildExtractionSignalText (const cortext::operations::ExtractionResult &result)
{
  std::string text = result.summary_id;
  for (const auto &fact : result.facts)
    {
      if (!text.empty ())
        {
          text += " ";
        }
      text += fact.subject + " " + fact.predicate + " " + fact.object;
    }
  if (text.empty ())
    {
      text = "bitemporal ablation extraction";
    }
  return text;
}

void
SeedMemoryWithEmbedding (cortext::Store &store, long long memory_id,
                         long long embedding_id, const Eigen::VectorXf &embedding,
                         const std::string &source_id, long long start_ts)
{
  store.Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?)",
      { embedding_id, ToFloatVec (embedding), start_ts });
  store.Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
      "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, ?, 'ASSOCIATION', ?, 1, 'text', 0.5, 0.5, 1.0, ?)",
      { memory_id, embedding_id, source_id, start_ts, start_ts });
}

void
SeedEmbeddingOnly (cortext::Store &store, long long embedding_id,
                   const Eigen::VectorXf &embedding, long long created_at)
{
  store.Execute ("INSERT INTO embeddings(embedding_id, embedding, created_at) "
                 "VALUES (?, ?, ?)",
                 { embedding_id, ToFloatVec (embedding), created_at });
}

void
InsertFactAssertion (cortext::Store &store, long long fact_id,
                     const std::string &subject, const std::string &predicate,
                     const std::string &object,
                     const std::string &canonical_subject,
                     const std::string &canonical_predicate,
                     const std::string &canonical_object,
                     std::optional<long long> valid_start_ts,
                     std::optional<long long> valid_end_ts,
                     long long recorded_at_ts,
                     std::optional<long long> superseded_at_ts,
                     double confidence, long long summary_memory_id)
{
  store.Execute (
      "INSERT INTO fact_assertions "
      "(fact_id, subject, predicate, object, canonical_subject, "
      " canonical_predicate, canonical_object, valid_start_ts, valid_end_ts, "
      " recorded_at_ts, superseded_at_ts, confidence, summary_memory_id, "
      " created_at) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      { fact_id,
        subject,
        predicate,
        object,
        canonical_subject,
        canonical_predicate,
        canonical_object,
        valid_start_ts.has_value () ? std::any (*valid_start_ts) : std::any (),
        valid_end_ts.has_value () ? std::any (*valid_end_ts) : std::any (),
        recorded_at_ts,
        superseded_at_ts.has_value () ? std::any (*superseded_at_ts)
                                      : std::any (),
        confidence,
        summary_memory_id,
        recorded_at_ts });
}

void
InsertFactCache (cortext::Store &store, long long fact_id, long long embedding_id,
                 const std::string &fact_text, bool is_current,
                 std::optional<long long> valid_start_ts,
                 std::optional<long long> valid_end_ts,
                 long long recorded_at_ts,
                 std::optional<long long> superseded_at_ts,
                 long long updated_at)
{
  store.Execute (
      "INSERT INTO fact_cache "
      "(fact_id, embedding_id, fact_text, is_current, valid_start_ts, "
      " valid_end_ts, recorded_at_ts, superseded_at_ts, updated_at) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
      { fact_id,
        embedding_id,
        fact_text,
        is_current ? 1LL : 0LL,
        valid_start_ts.has_value () ? std::any (*valid_start_ts) : std::any (),
        valid_end_ts.has_value () ? std::any (*valid_end_ts) : std::any (),
        recorded_at_ts,
        superseded_at_ts.has_value () ? std::any (*superseded_at_ts)
                                      : std::any (),
        updated_at });
}

void
LinkFactEvidence (cortext::Store &store, long long fact_id, long long memory_id,
                  const std::string &evidence_type, double support_weight)
{
  store.Execute (
      "INSERT INTO fact_evidence "
      "(fact_id, source_memory_id, evidence_type, support_weight) "
      "VALUES (?, ?, ?, ?)",
      { fact_id, memory_id, evidence_type, support_weight });
}

void
RunExtraction (cortext::Store &store, cortext::ProcessorContext &pctx,
               BenchEncoder &encoder, cortext::operations::ExtractionResult result,
               std::uint64_t signal_ts, double sensitivity = 0.5,
               double stability = 0.5)
{
  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = sensitivity;
  cfg.stability = stability;
  cfg.encoder = &encoder;

  const auto signal_embedding
      = encoder.EncodeTextEigen (BuildExtractionSignalText (result));
  pctx.pending_extraction_results.push_back (std::move (result));
  cortext::OperationContext ctx (MakeSignal (signal_embedding, signal_ts),
                                 pctx, cfg, &store);
  cortext::operations::ProcessExtractionResults op;
  auto tx = store.Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();
}

enum class ScenarioId
{
  CaregiverTransition,
  DelayedLocation,
  ConflictingNoiseGuard,
  ProvenanceSchedule,
  RepeatedLocationChanges,
  RoutineException,
  OldHomeCurrentHome,
  UpdatedPreference,
  MedicationChange,
  WeakScheduleChange
};

struct AblationConfig
{
  std::string name;
  double focus = 1.0;
  double sensitivity = 0.5;
  double stability = 0.5;
  bool representative_subset_only = false;
  cortext::operations::temporal::RetrievalAblationOverride override;
};

struct QueryExpectation
{
  cortext::operations::temporal::RetrievalMode retrieval_mode
      = cortext::operations::temporal::RetrievalMode::Current;
  cortext::store::FactQueryMode fact_query_mode
      = cortext::store::FactQueryMode::Current;
  std::uint64_t signal_ts = 0;
  std::uint64_t query_ts = 0;
  std::uint64_t fallback_timestamp = 0;
  std::string expected_object;
  long long expected_top_memory = 0;
  std::vector<long long> stale_memory_ids;
  std::vector<long long> supported_memory_ids;
  std::vector<long long> background_memory_ids;
  std::size_t background_topn = 0;
};

struct ScenarioSpec
{
  ScenarioId id;
  std::string name;
  double severity_weight = 1.0;
  bool representative_subset = false;
  std::function<void(std::shared_ptr<cortext::Store>, cortext::ProcessorContext &,
                     BenchEncoder &)>
      setup;
  std::vector<QueryExpectation> queries;
  std::vector<long long> expected_current_sequence;
  bool check_erroneous_supersession = false;
};

const char *
ScenarioPredicate (ScenarioId id)
{
  switch (id)
    {
    case ScenarioId::CaregiverTransition:
    case ScenarioId::ConflictingNoiseGuard:
      return "caregiver";
    case ScenarioId::DelayedLocation:
    case ScenarioId::RepeatedLocationChanges:
      return "lives_in";
    case ScenarioId::RoutineException:
    case ScenarioId::ProvenanceSchedule:
    case ScenarioId::WeakScheduleChange:
      return "schedule";
    case ScenarioId::OldHomeCurrentHome:
      return "home";
    case ScenarioId::UpdatedPreference:
      return "likes";
    case ScenarioId::MedicationChange:
      return "takes_medication";
    }
  return "schedule";
}

std::string
BuildQueryText (ScenarioId id,
                cortext::operations::temporal::RetrievalMode mode)
{
  switch (id)
    {
    case ScenarioId::CaregiverTransition:
    case ScenarioId::ConflictingNoiseGuard:
      switch (mode)
        {
        case cortext::operations::temporal::RetrievalMode::Current:
          return "Who is Alice's caregiver right now?";
        case cortext::operations::temporal::RetrievalMode::ValidAt:
          return "Who was Alice's caregiver at that earlier time?";
        case cortext::operations::temporal::RetrievalMode::KnownAt:
          return "Who did we believe Alice's caregiver was at that time?";
        }
      break;
    case ScenarioId::DelayedLocation:
    case ScenarioId::RepeatedLocationChanges:
      switch (mode)
        {
        case cortext::operations::temporal::RetrievalMode::Current:
          return "Where does Alice live now?";
        case cortext::operations::temporal::RetrievalMode::ValidAt:
          return "Where was Alice living then?";
        case cortext::operations::temporal::RetrievalMode::KnownAt:
          return "Where did we think Alice lived at that point?";
        }
      break;
    case ScenarioId::ProvenanceSchedule:
    case ScenarioId::RoutineException:
    case ScenarioId::WeakScheduleChange:
      switch (mode)
        {
        case cortext::operations::temporal::RetrievalMode::Current:
          return "What appointment does Alice have today at noon?";
        case cortext::operations::temporal::RetrievalMode::ValidAt:
          return "What appointment was Alice scheduled for then?";
        case cortext::operations::temporal::RetrievalMode::KnownAt:
          return "What appointment did we believe Alice had then?";
        }
      break;
    case ScenarioId::OldHomeCurrentHome:
      switch (mode)
        {
        case cortext::operations::temporal::RetrievalMode::Current:
          return "What is Alice's current home?";
        case cortext::operations::temporal::RetrievalMode::ValidAt:
          return "What was Alice's home then?";
        case cortext::operations::temporal::RetrievalMode::KnownAt:
          return "What home did we believe Alice had then?";
        }
      break;
    case ScenarioId::UpdatedPreference:
      switch (mode)
        {
        case cortext::operations::temporal::RetrievalMode::Current:
          return "What does Alice like now?";
        case cortext::operations::temporal::RetrievalMode::ValidAt:
          return "What did Alice like earlier?";
        case cortext::operations::temporal::RetrievalMode::KnownAt:
          return "What did we believe Alice liked at that time?";
        }
      break;
    case ScenarioId::MedicationChange:
      switch (mode)
        {
        case cortext::operations::temporal::RetrievalMode::Current:
          return "What medication is Alice taking now?";
        case cortext::operations::temporal::RetrievalMode::ValidAt:
          return "What medication was Alice taking then?";
        case cortext::operations::temporal::RetrievalMode::KnownAt:
          return "What medication did we think Alice was taking then?";
        }
      break;
    }
  return "What is the current fact about Alice?";
}

std::vector<long long>
RunRetrieval (std::shared_ptr<cortext::Store> store, BenchEncoder &encoder,
              const std::string &query_text,
              std::uint64_t signal_ts,
              cortext::operations::temporal::RetrievalMode mode,
              std::optional<std::uint64_t> retrieval_ts,
              const AblationConfig &config)
{
  cortext::SignalProcessor::Config cfg;
  cfg.focus = config.focus;
  cfg.sensitivity = config.sensitivity;
  cfg.stability = config.stability;
  cfg.encoder = &encoder;

  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  cortext::operations::temporal::ScopedRetrievalOverride temporal_guard (
      mode, retrieval_ts);
  cortext::operations::temporal::ScopedRetrievalAblationOverride ablation_guard (
      config.override);
  cortext::operations::retrieval_debug::ClearLastSelectedEmbeddingOrder ();
  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  (void)processor.Process (
      MakeSignal (encoder.EncodeTextEigen (query_text), signal_ts));

  std::vector<long long> memory_ids;
  const auto &ranked
      = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  memory_ids.reserve (ranked.size ());
  for (const auto &candidate : ranked)
    {
      if (candidate.memory_id > 0)
        {
          memory_ids.push_back (candidate.memory_id);
        }
    }
  if (!memory_ids.empty ())
    {
      return memory_ids;
    }
  return cortext::operations::retrieval_debug::GetLastSelectedEmbeddingOrder ();
}

std::string
QueryFirstFactObject (cortext::Store &store, cortext::store::FactQueryMode mode,
                      const std::string &subject, const std::string &predicate,
                      std::uint64_t ts, std::uint64_t fallback_timestamp,
                      const AblationConfig &config)
{
  cortext::operations::temporal::ScopedRetrievalAblationOverride guard (
      config.override);
  auto tx = store.Begin ();
  const auto rows = cortext::store::QueryFacts (
      *tx,
      { cortext::operations::temporal::ResolveFactQueryMode (mode),
        std::string (subject),
        std::string (predicate),
        cortext::operations::temporal::ResolveFactQueryTimestamp (
            mode, ts, fallback_timestamp) });
  if (rows.empty ())
    {
      return {};
    }
  return rows.front ().object;
}

bool
TopEquals (const std::vector<long long> &order, long long expected)
{
  return !order.empty () && order.front () == expected;
}

bool
TopIn (const std::vector<long long> &order, const std::vector<long long> &allowed)
{
  if (order.empty ())
    {
      return false;
    }
  return std::find (allowed.begin (), allowed.end (), order.front ())
         != allowed.end ();
}

bool
ContainsAnyTopN (const std::vector<long long> &order,
                 const std::vector<long long> &targets, std::size_t n)
{
  const std::size_t limit = std::min (order.size (), n);
  for (std::size_t i = 0; i < limit; ++i)
    {
      if (std::find (targets.begin (), targets.end (), order[i])
          != targets.end ())
        {
          return true;
        }
    }
  return false;
}

int
CountUnexpectedFlips (const std::vector<long long> &states,
                      const std::vector<long long> &expected)
{
  int flips = 0;
  for (std::size_t i = 0; i < std::min (states.size (), expected.size ()); ++i)
    {
      if (states[i] != expected[i])
        {
          ++flips;
        }
    }
  return flips;
}

std::string
FormatRankedCandidates ()
{
  std::ostringstream out;
  const auto &ranked
      = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  for (const auto &candidate : ranked)
    {
      out << " [mem=" << candidate.memory_id << ", emb="
          << candidate.embedding_id << ", score=" << candidate.score
          << ", rel=" << candidate.relevance << ", boost="
          << candidate.fact_boost << ", stale="
          << candidate.fact_stale_penalty << ", links="
          << candidate.linked_fact_count << "]";
    }
  return out.str ();
}

struct AggregateMetrics
{
  std::string name;
  double current_weight_total = 0.0;
  double current_weight_pass = 0.0;
  double historical_weight_total = 0.0;
  double historical_weight_pass = 0.0;
  double belief_weight_total = 0.0;
  double belief_weight_pass = 0.0;
  double retrieval_current_weight_total = 0.0;
  double retrieval_current_weight_pass = 0.0;
  double stale_weight_total = 0.0;
  double stale_weight_intrusions = 0.0;
  double unsupported_weight_total = 0.0;
  double unsupported_weight_hits = 0.0;
  double routine_weight_total = 0.0;
  double routine_weight_pass = 0.0;
  double clarification_weight_total = 0.0;
  double clarification_weight_pass = 0.0;
  double contradiction_weight_total = 0.0;
  double contradiction_weight_hits = 0.0;
  double correction_failure_weight_total = 0.0;
  double correction_failure_weight_hits = 0.0;
  double outdated_first_weight_total = 0.0;
  double outdated_first_weight_hits = 0.0;
  int current_query_checks = 0;
  int current_query_passes = 0;
  int valid_query_checks = 0;
  int valid_query_passes = 0;
  int known_query_checks = 0;
  int known_query_passes = 0;
  int erroneous_supersession_count = 0;
  int unexpected_flip_count = 0;
  int correction_event_count = 0;
  int correction_steps_total = 0;
  int retrieval_runs = 0;
  double retrieval_ms_sum = 0.0;
  int scenario_count = 0;
  int passed_scenarios = 0;
  std::vector<std::string> failures;
};

double
SafeRatio (double num, double den)
{
  return den <= 0.0 ? 0.0 : num / den;
}

void
NoteFailure (AggregateMetrics &metrics, const std::string &name,
             const std::string &detail)
{
  std::ostringstream out;
  out << name << ": " << detail;
  metrics.failures.push_back (out.str ());
}

std::string
CurrentObjectForSequence (const AblationConfig &config,
                          const std::shared_ptr<cortext::Store> &store,
                          ScenarioId id, std::uint64_t ts)
{
  switch (id)
    {
    case ScenarioId::CaregiverTransition:
    case ScenarioId::ConflictingNoiseGuard:
      return QueryFirstFactObject (*store, cortext::store::FactQueryMode::Current,
                                   "alice", "caregiver", ts, ts, config);
    case ScenarioId::DelayedLocation:
    case ScenarioId::RepeatedLocationChanges:
      return QueryFirstFactObject (*store, cortext::store::FactQueryMode::Current,
                                   "alice", "lives_in", ts, ts, config);
    case ScenarioId::RoutineException:
    case ScenarioId::ProvenanceSchedule:
    case ScenarioId::WeakScheduleChange:
      return QueryFirstFactObject (*store, cortext::store::FactQueryMode::Current,
                                   "alice", "schedule", ts, ts, config);
    case ScenarioId::OldHomeCurrentHome:
      return QueryFirstFactObject (*store, cortext::store::FactQueryMode::Current,
                                   "alice", "home", ts, ts, config);
    case ScenarioId::UpdatedPreference:
      return QueryFirstFactObject (*store, cortext::store::FactQueryMode::Current,
                                   "alice", "likes", ts, ts, config);
    case ScenarioId::MedicationChange:
      return QueryFirstFactObject (
          *store, cortext::store::FactQueryMode::Current, "alice",
          "takes_medication", ts, ts, config);
    }
  return {};
}

void
AccumulateCorrectionSteps (AggregateMetrics &metrics,
                           const std::vector<std::string> &actual,
                           const std::vector<std::string> &expected)
{
  for (std::size_t i = 1; i < std::min (actual.size (), expected.size ()); ++i)
    {
      if (expected[i] == expected[i - 1])
        {
          continue;
        }
      metrics.correction_event_count++;
      int steps = 0;
      bool corrected = (actual[i] == expected[i]);
      if (!corrected)
        {
          steps = 1;
          for (std::size_t j = i + 1; j < std::min (actual.size (), expected.size ());
               ++j)
            {
              if (actual[j] == expected[j])
                {
                  steps = static_cast<int> (j - i);
                  corrected = true;
                  break;
                }
            }
        }
      if (!corrected)
        {
          steps = static_cast<int> (expected.size () - i);
        }
      metrics.correction_steps_total += steps;
    }
}

std::vector<ScenarioSpec>
BuildScenarios ()
{
  std::vector<ScenarioSpec> scenarios;

  scenarios.push_back (
      { ScenarioId::CaregiverTransition,
        "caregiver_transition",
        3.0,
        true,
        [] (std::shared_ptr<cortext::Store> store,
            cortext::ProcessorContext & /*pctx*/, BenchEncoder &encoder) {
          SeedMemoryWithEmbedding (
              *store, 1LL, 1LL,
              encoder.EncodeTextEigen ("Alice's caregiver was Sarah."),
              "caregiver-sarah", 1000LL);
          SeedMemoryWithEmbedding (
              *store, 2LL, 2LL,
              encoder.EncodeTextEigen ("Alice's caregiver is Emily now."),
              "caregiver-emily", 5000LL);
          SeedEmbeddingOnly (
              *store, 201LL,
              encoder.EncodeTextEigen ("Alice caregiver Emily"), 7000LL);
          InsertFactAssertion (*store, 100LL, "Alice", "caregiver", "Sarah",
                               "alice", "caregiver", "sarah", 1000LL, 5000LL,
                               3000LL, 7000LL, 0.80, 1LL);
          InsertFactAssertion (*store, 101LL, "Alice", "caregiver", "Emily",
                               "alice", "caregiver", "emily", 5000LL,
                               std::nullopt, 7000LL, std::nullopt, 0.92, 2LL);
          InsertFactCache (*store, 100LL, 1LL, "Alice caregiver Sarah", false,
                           1000LL, 5000LL, 3000LL, 7000LL, 7000LL);
          InsertFactCache (*store, 101LL, 201LL, "Alice caregiver Emily", true,
                           5000LL, std::nullopt, 7000LL, std::nullopt, 8000LL);
          LinkFactEvidence (*store, 100LL, 1LL, "summary", 1.0);
          LinkFactEvidence (*store, 101LL, 2LL, "summary", 1.0);
        },
        {
            { cortext::operations::temporal::RetrievalMode::Current,
              cortext::store::FactQueryMode::Current, 8000ULL, 8000ULL, 8000ULL,
              "Emily", 2LL, { 1LL }, { 2LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::ValidAt,
              cortext::store::FactQueryMode::ValidAt, 8000ULL, 4000ULL, 8000ULL,
              "Sarah", 1LL, {}, { 1LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::KnownAt,
              cortext::store::FactQueryMode::KnownAt, 8000ULL, 6000ULL, 8000ULL,
              "Sarah", 1LL, {}, { 1LL }, {}, 0 },
        },
        { 2LL },
        false } );

  scenarios.push_back (
      { ScenarioId::DelayedLocation,
        "delayed_location",
        3.0,
        true,
        [] (std::shared_ptr<cortext::Store> store,
            cortext::ProcessorContext &pctx, BenchEncoder &encoder) {
          SeedMemoryWithEmbedding (
              *store, 20LL, 10LL,
              encoder.EncodeTextEigen ("Alice used to live in Austin."),
              "summary-1", 1000LL);
          SeedMemoryWithEmbedding (
              *store, 21LL, 11LL,
              encoder.EncodeTextEigen ("Alice now lives in Denver."),
              "summary-2", 5000LL);

          cortext::operations::ExtractionResult first;
          first.summary_id = "summary-1";
          first.facts.push_back (
              { "Alice", "lives_in", "Austin", 0.80, std::uint64_t (1000),
                std::nullopt });
          RunExtraction (*store, pctx, encoder, std::move (first), 3000ULL);

          cortext::operations::ExtractionResult second;
          second.summary_id = "summary-2";
          second.facts.push_back (
              { "Alice", "lives_in", "Denver", 0.92, std::uint64_t (5000),
                std::nullopt });
          RunExtraction (*store, pctx, encoder, std::move (second), 7000ULL);
        },
        {
            { cortext::operations::temporal::RetrievalMode::Current,
              cortext::store::FactQueryMode::Current, 8000ULL, 8000ULL, 8000ULL,
              "Denver", 21LL, { 20LL }, { 21LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::ValidAt,
              cortext::store::FactQueryMode::ValidAt, 8000ULL, 4000ULL, 8000ULL,
              "Austin", 20LL, {}, { 20LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::KnownAt,
              cortext::store::FactQueryMode::KnownAt, 8000ULL, 6000ULL, 8000ULL,
              "Austin", 20LL, {}, { 20LL }, {}, 0 },
        },
        { 21LL },
        false } );

  scenarios.push_back (
      { ScenarioId::ConflictingNoiseGuard,
        "conflicting_noise_guard",
        3.0,
        false,
        [] (std::shared_ptr<cortext::Store> store,
            cortext::ProcessorContext &pctx, BenchEncoder &encoder) {
          SeedMemoryWithEmbedding (
              *store, 30LL, 30LL,
              encoder.EncodeTextEigen ("Alice's caregiver is Sarah."),
              "summary-1", 1000LL);
          SeedMemoryWithEmbedding (
              *store, 31LL, 31LL,
              encoder.EncodeTextEigen (
                  "Someone speculated that Alice's caregiver might be Emily."),
              "summary-2", 5000LL);

          cortext::operations::ExtractionResult first;
          first.summary_id = "summary-1";
          first.facts.push_back (
              { "Alice", "caregiver", "Sarah", 0.95, std::uint64_t (1000),
                std::nullopt });
          RunExtraction (*store, pctx, encoder, std::move (first), 3000ULL,
                         0.10, 0.80);

          cortext::operations::ExtractionResult noisy;
          noisy.summary_id = "summary-2";
          noisy.facts.push_back (
              { "Alice", "caregiver", "Emily", 0.35, std::uint64_t (5000),
                std::nullopt });
          RunExtraction (*store, pctx, encoder, std::move (noisy), 7000ULL,
                         0.10, 0.80);
        },
        {
            { cortext::operations::temporal::RetrievalMode::Current,
              cortext::store::FactQueryMode::Current, 8000ULL, 8000ULL, 8000ULL,
              "Sarah", 30LL, { 31LL }, { 30LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::KnownAt,
              cortext::store::FactQueryMode::KnownAt, 8000ULL, 8000ULL, 8000ULL,
              "Sarah", 30LL, {}, { 30LL }, {}, 0 },
        },
        { 30LL },
        true } );

  scenarios.push_back (
      { ScenarioId::ProvenanceSchedule,
        "provenance_schedule",
        3.0,
        false,
        [] (std::shared_ptr<cortext::Store> store,
            cortext::ProcessorContext & /*pctx*/, BenchEncoder &encoder) {
          SeedMemoryWithEmbedding (
              *store, 40LL, 40LL,
              encoder.EncodeTextEigen ("Alice has a doctor appointment today."),
              "linked-memory", 5000LL);
          SeedMemoryWithEmbedding (
              *store, 41LL, 41LL,
              encoder.EncodeTextEigen (
                  "Alice talked about doctor appointments in general."),
              "semantic-distractor", 5000LL);
          SeedEmbeddingOnly (
              *store, 202LL,
              encoder.EncodeTextEigen ("Alice schedule Doctor today"), 7000LL);
          InsertFactAssertion (*store, 200LL, "Alice", "schedule",
                               "Doctor today", "alice", "schedule",
                               "doctor_today", 5000LL, std::nullopt, 7000LL,
                               std::nullopt, 0.95, 40LL);
          InsertFactCache (*store, 200LL, 202LL, "Alice schedule Doctor today",
                           true, 5000LL, std::nullopt, 7000LL, std::nullopt,
                           8000LL);
          LinkFactEvidence (*store, 200LL, 40LL, "summary", 1.0);
        },
        {
            { cortext::operations::temporal::RetrievalMode::Current,
              cortext::store::FactQueryMode::Current, 8000ULL, 8000ULL, 8000ULL,
              "Doctor today", 40LL, {}, { 40LL }, {}, 0 },
        },
        {},
        false } );

  scenarios.push_back (
      { ScenarioId::RepeatedLocationChanges,
        "repeated_location_changes",
        3.0,
        true,
        [] (std::shared_ptr<cortext::Store> store,
            cortext::ProcessorContext & /*pctx*/, BenchEncoder &encoder) {
          SeedMemoryWithEmbedding (
              *store, 50LL, 50LL,
              encoder.EncodeTextEigen ("Alice lived in Austin before."),
              "austin-old", 1000LL);
          SeedMemoryWithEmbedding (
              *store, 51LL, 51LL,
              encoder.EncodeTextEigen ("Alice later lived in Denver."),
              "denver-mid", 5000LL);
          SeedMemoryWithEmbedding (
              *store, 52LL, 52LL,
              encoder.EncodeTextEigen ("Alice lives in Austin again now."),
              "austin-current", 9000LL);
          SeedEmbeddingOnly (
              *store, 203LL,
              encoder.EncodeTextEigen ("Alice lives_in Austin"), 11000LL);
          InsertFactAssertion (*store, 300LL, "Alice", "lives_in", "Austin",
                               "alice", "lives_in", "austin", 1000LL, 5000LL,
                               3000LL, 7000LL, 0.90, 50LL);
          InsertFactAssertion (*store, 301LL, "Alice", "lives_in", "Denver",
                               "alice", "lives_in", "denver", 5000LL, 9000LL,
                               7000LL, 11000LL, 0.94, 51LL);
          InsertFactAssertion (*store, 302LL, "Alice", "lives_in", "Austin",
                               "alice", "lives_in", "austin", 9000LL,
                               std::nullopt, 11000LL, std::nullopt, 0.97,
                               52LL);
          InsertFactCache (*store, 300LL, 50LL, "Alice lives_in Austin", false,
                           1000LL, 5000LL, 3000LL, 7000LL, 12000LL);
          InsertFactCache (*store, 301LL, 51LL, "Alice lives_in Denver", false,
                           5000LL, 9000LL, 7000LL, 11000LL, 12000LL);
          InsertFactCache (*store, 302LL, 203LL, "Alice lives_in Austin", true,
                           9000LL, std::nullopt, 11000LL, std::nullopt,
                           12000LL);
          LinkFactEvidence (*store, 300LL, 50LL, "summary", 1.0);
          LinkFactEvidence (*store, 301LL, 51LL, "summary", 1.0);
          LinkFactEvidence (*store, 302LL, 52LL, "summary", 1.0);
        },
        {
            { cortext::operations::temporal::RetrievalMode::Current,
              cortext::store::FactQueryMode::Current, 4000ULL, 4000ULL, 4000ULL,
              "Austin", 50LL, {}, { 50LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::Current,
              cortext::store::FactQueryMode::Current, 8000ULL, 8000ULL, 8000ULL,
              "Denver", 51LL, {}, { 51LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::Current,
              cortext::store::FactQueryMode::Current, 12000ULL, 12000ULL,
              12000ULL, "Austin", 52LL, { 50LL, 51LL }, { 52LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::ValidAt,
              cortext::store::FactQueryMode::ValidAt, 12000ULL, 8000ULL, 12000ULL,
              "Denver", 51LL, {}, { 51LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::KnownAt,
              cortext::store::FactQueryMode::KnownAt, 12000ULL, 8000ULL, 12000ULL,
              "Denver", 51LL, {}, { 51LL }, {}, 0 },
        },
        { 50LL, 51LL, 52LL },
        false } );

  scenarios.push_back (
      { ScenarioId::RoutineException,
        "routine_exception",
        2.0,
        true,
        [] (std::shared_ptr<cortext::Store> store,
            cortext::ProcessorContext & /*pctx*/, BenchEncoder &encoder) {
          SeedMemoryWithEmbedding (
              *store, 60LL, 60LL,
              encoder.EncodeTextEigen ("Alice usually has lunch at noon."),
              "usual-routine", 1000LL);
          SeedMemoryWithEmbedding (
              *store, 61LL, 61LL,
              encoder.EncodeTextEigen (
                  "Today Alice has a doctor appointment at noon."),
              "today-exception", 5000LL);
          SeedMemoryWithEmbedding (
              *store, 62LL, 62LL,
              encoder.EncodeTextEigen ("Alice often walks after lunch."),
              "other-routine-1", 3000LL);
          SeedMemoryWithEmbedding (
              *store, 63LL, 63LL,
              encoder.EncodeTextEigen ("Alice usually reads at home in the afternoon."),
              "other-routine-2", 3500LL);
          SeedEmbeddingOnly (
              *store, 204LL,
              encoder.EncodeTextEigen ("Alice schedule Doctor at noon"), 7000LL);
          InsertFactAssertion (*store, 400LL, "Alice", "schedule",
                               "Lunch at noon", "alice", "schedule",
                               "lunch_at_noon", 1000LL, std::nullopt, 2000LL,
                               std::nullopt, 0.82, 60LL);
          InsertFactAssertion (*store, 401LL, "Alice", "schedule",
                               "Doctor at noon", "alice", "schedule",
                               "doctor_at_noon", 5000LL, 6000LL, 5000LL,
                               std::nullopt, 0.93, 61LL);
          InsertFactCache (*store, 400LL, 60LL, "Alice schedule Lunch at noon",
                           true, 1000LL, std::nullopt, 2000LL, std::nullopt,
                           7000LL);
          InsertFactCache (*store, 401LL, 204LL, "Alice schedule Doctor at noon",
                           true, 5000LL, 6000LL, 5000LL, std::nullopt, 7000LL);
          LinkFactEvidence (*store, 400LL, 60LL, "summary", 1.0);
          LinkFactEvidence (*store, 401LL, 61LL, "summary", 1.0);
        },
        {
            { cortext::operations::temporal::RetrievalMode::Current,
              cortext::store::FactQueryMode::Current, 5500ULL, 5500ULL, 5500ULL,
              "Doctor at noon", 61LL, { 60LL }, { 61LL }, { 60LL }, 3 },
            { cortext::operations::temporal::RetrievalMode::ValidAt,
              cortext::store::FactQueryMode::ValidAt, 3000ULL, 3000ULL, 5500ULL,
              "Lunch at noon", 60LL, {}, { 60LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::KnownAt,
              cortext::store::FactQueryMode::KnownAt, 5500ULL, 5500ULL, 5500ULL,
              "Doctor at noon", 61LL, {}, { 61LL }, {}, 0 },
        },
        { 61LL },
        false } );

  scenarios.push_back (
      { ScenarioId::OldHomeCurrentHome,
        "old_home_current_home",
        2.0,
        false,
        [] (std::shared_ptr<cortext::Store> store,
            cortext::ProcessorContext & /*pctx*/, BenchEncoder &encoder) {
          SeedMemoryWithEmbedding (
              *store, 70LL, 70LL,
              encoder.EncodeTextEigen ("Alice's old home was Chicago."),
              "old-home", 1000LL);
          SeedMemoryWithEmbedding (
              *store, 71LL, 71LL,
              encoder.EncodeTextEigen ("Alice's current home is Boston."),
              "current-home", 5000LL);
          SeedEmbeddingOnly (
              *store, 205LL,
              encoder.EncodeTextEigen ("Alice home Boston"), 7000LL);
          InsertFactAssertion (*store, 500LL, "Alice", "home", "Chicago",
                               "alice", "home", "chicago", 1000LL, 5000LL,
                               2000LL, 7000LL, 0.86, 70LL);
          InsertFactAssertion (*store, 501LL, "Alice", "home", "Boston",
                               "alice", "home", "boston", 5000LL,
                               std::nullopt, 7000LL, std::nullopt, 0.90, 71LL);
          InsertFactCache (*store, 500LL, 70LL, "Alice home Chicago", false,
                           1000LL, 5000LL, 2000LL, 7000LL, 8000LL);
          InsertFactCache (*store, 501LL, 205LL, "Alice home Boston", true,
                           5000LL, std::nullopt, 7000LL, std::nullopt, 8000LL);
          LinkFactEvidence (*store, 500LL, 70LL, "summary", 1.0);
          LinkFactEvidence (*store, 501LL, 71LL, "summary", 1.0);
        },
        {
            { cortext::operations::temporal::RetrievalMode::Current,
              cortext::store::FactQueryMode::Current, 8000ULL, 8000ULL, 8000ULL,
              "Boston", 71LL, { 70LL }, { 71LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::ValidAt,
              cortext::store::FactQueryMode::ValidAt, 8000ULL, 4000ULL, 8000ULL,
              "Chicago", 70LL, {}, { 70LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::KnownAt,
              cortext::store::FactQueryMode::KnownAt, 8000ULL, 6000ULL, 8000ULL,
              "Chicago", 70LL, {}, { 70LL }, {}, 0 },
        },
        { 71LL },
        false } );

  scenarios.push_back (
      { ScenarioId::UpdatedPreference,
        "updated_preference",
        1.0,
        false,
        [] (std::shared_ptr<cortext::Store> store,
            cortext::ProcessorContext & /*pctx*/, BenchEncoder &encoder) {
          SeedMemoryWithEmbedding (
              *store, 80LL, 80LL,
              encoder.EncodeTextEigen ("Alice used to like coffee."),
              "coffee-old", 1000LL);
          SeedMemoryWithEmbedding (
              *store, 81LL, 81LL,
              encoder.EncodeTextEigen ("Alice likes tea now."),
              "tea-current", 7000LL);
          SeedEmbeddingOnly (
              *store, 206LL,
              encoder.EncodeTextEigen ("Alice likes Tea"), 9000LL);
          InsertFactAssertion (*store, 600LL, "Alice", "likes", "Coffee",
                               "alice", "likes", "coffee", 1000LL, 7000LL,
                               2000LL, 9000LL, 0.72, 80LL);
          InsertFactAssertion (*store, 601LL, "Alice", "likes", "Tea", "alice",
                               "likes", "tea", 7000LL, std::nullopt, 9000LL,
                               std::nullopt, 0.80, 81LL);
          InsertFactCache (*store, 600LL, 80LL, "Alice likes Coffee", false,
                           1000LL, 7000LL, 2000LL, 9000LL, 10000LL);
          InsertFactCache (*store, 601LL, 206LL, "Alice likes Tea", true,
                           7000LL, std::nullopt, 9000LL, std::nullopt, 10000LL);
          LinkFactEvidence (*store, 600LL, 80LL, "summary", 1.0);
          LinkFactEvidence (*store, 601LL, 81LL, "summary", 1.0);
        },
        {
            { cortext::operations::temporal::RetrievalMode::Current,
              cortext::store::FactQueryMode::Current, 10000ULL, 10000ULL,
              10000ULL, "Tea", 81LL, { 80LL }, { 81LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::ValidAt,
              cortext::store::FactQueryMode::ValidAt, 10000ULL, 5000ULL,
              10000ULL, "Coffee", 80LL, {}, { 80LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::KnownAt,
              cortext::store::FactQueryMode::KnownAt, 10000ULL, 8000ULL,
              10000ULL, "Coffee", 80LL, {}, { 80LL }, {}, 0 },
        },
        { 81LL },
        false } );

  scenarios.push_back (
      { ScenarioId::MedicationChange,
        "medication_change",
        3.0,
        true,
        [] (std::shared_ptr<cortext::Store> store,
            cortext::ProcessorContext & /*pctx*/, BenchEncoder &encoder) {
          SeedMemoryWithEmbedding (
              *store, 90LL, 90LL,
              encoder.EncodeTextEigen ("Alice used to take Donepezil."),
              "donepezil-old", 1000LL);
          SeedMemoryWithEmbedding (
              *store, 91LL, 91LL,
              encoder.EncodeTextEigen ("Alice now takes Memantine."),
              "memantine-current", 6000LL);
          SeedEmbeddingOnly (
              *store, 207LL,
              encoder.EncodeTextEigen ("Alice takes_medication Memantine"),
              9000LL);
          InsertFactAssertion (*store, 700LL, "Alice", "takes_medication",
                               "Donepezil", "alice", "takes_medication",
                               "donepezil", 1000LL, 6000LL, 3000LL, 9000LL,
                               0.90, 90LL);
          InsertFactAssertion (*store, 701LL, "Alice", "takes_medication",
                               "Memantine", "alice", "takes_medication",
                               "memantine", 6000LL, std::nullopt, 9000LL,
                               std::nullopt, 0.95, 91LL);
          InsertFactCache (*store, 700LL, 90LL, "Alice takes_medication Donepezil",
                           false, 1000LL, 6000LL, 3000LL, 9000LL, 10000LL);
          InsertFactCache (*store, 701LL, 207LL, "Alice takes_medication Memantine",
                           true, 6000LL, std::nullopt, 9000LL, std::nullopt,
                           10000LL);
          LinkFactEvidence (*store, 700LL, 90LL, "summary", 1.0);
          LinkFactEvidence (*store, 701LL, 91LL, "summary", 1.0);
        },
        {
            { cortext::operations::temporal::RetrievalMode::Current,
              cortext::store::FactQueryMode::Current, 10000ULL, 10000ULL,
              10000ULL, "Memantine", 91LL, { 90LL }, { 91LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::ValidAt,
              cortext::store::FactQueryMode::ValidAt, 10000ULL, 5000ULL,
              10000ULL, "Donepezil", 90LL, {}, { 90LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::KnownAt,
              cortext::store::FactQueryMode::KnownAt, 10000ULL, 8000ULL,
              10000ULL, "Donepezil", 90LL, {}, { 90LL }, {}, 0 },
        },
        { 91LL },
        false } );

  scenarios.push_back (
      { ScenarioId::WeakScheduleChange,
        "weak_schedule_change",
        2.0,
        true,
        [] (std::shared_ptr<cortext::Store> store,
            cortext::ProcessorContext & /*pctx*/, BenchEncoder &encoder) {
          SeedMemoryWithEmbedding (
              *store, 100LL, 100LL,
              encoder.EncodeTextEigen ("Alice usually has lunch at noon."),
              "routine-lunch", 1000LL);
          SeedMemoryWithEmbedding (
              *store, 101LL, 101LL,
              encoder.EncodeTextEigen (
                  "Alice has a doctor appointment at noon today."),
              "doctor-exception", 5000LL);
          SeedMemoryWithEmbedding (
              *store, 102LL, 102LL,
              encoder.EncodeTextEigen ("Alice has a noon reminder for medication."),
              "filler-1", 4500LL);
          SeedMemoryWithEmbedding (
              *store, 103LL, 103LL,
              encoder.EncodeTextEigen ("Alice has lunch plans later this week."),
              "filler-2", 4600LL);
          SeedEmbeddingOnly (
              *store, 208LL,
              encoder.EncodeTextEigen ("Alice schedule Doctor at noon"), 7000LL);
          InsertFactAssertion (*store, 800LL, "Alice", "schedule",
                               "Lunch at noon", "alice", "schedule",
                               "lunch_at_noon", 1000LL, std::nullopt, 2000LL,
                               std::nullopt, 0.88, 100LL);
          InsertFactAssertion (*store, 801LL, "Alice", "schedule",
                               "Doctor at noon", "alice", "schedule",
                               "doctor_at_noon", 5000LL, 6000LL, 5000LL,
                               std::nullopt, 0.84, 101LL);
          InsertFactCache (*store, 800LL, 100LL, "Alice schedule Lunch at noon",
                           true, 1000LL, std::nullopt, 2000LL, std::nullopt,
                           7000LL);
          InsertFactCache (*store, 801LL, 208LL, "Alice schedule Doctor at noon",
                           true, 5000LL, 6000LL, 5000LL, std::nullopt, 7000LL);
          LinkFactEvidence (*store, 800LL, 100LL, "summary", 1.0);
          LinkFactEvidence (*store, 801LL, 101LL, "summary", 1.0);
        },
        {
            { cortext::operations::temporal::RetrievalMode::Current,
              cortext::store::FactQueryMode::Current, 5500ULL, 5500ULL, 5500ULL,
              "Doctor at noon", 0LL, { 100LL }, { 101LL }, { 100LL }, 3 },
            { cortext::operations::temporal::RetrievalMode::ValidAt,
              cortext::store::FactQueryMode::ValidAt, 3000ULL, 3000ULL, 5500ULL,
              "Lunch at noon", 100LL, {}, { 100LL }, {}, 0 },
            { cortext::operations::temporal::RetrievalMode::KnownAt,
              cortext::store::FactQueryMode::KnownAt, 5500ULL, 5500ULL, 5500ULL,
              "Doctor at noon", 0LL, {}, { 101LL }, {}, 0 },
        },
        { 100LL },
        false } );

  return scenarios;
}

AggregateMetrics
EvaluateConfig (const AblationConfig &config)
{
  AggregateMetrics metrics;
  metrics.name = config.name;
  BenchEncoder encoder (g_models_dir);
  const auto scenarios = BuildScenarios ();

  auto run_timed = [&] (const std::shared_ptr<cortext::Store> &store,
                        ScenarioId scenario_id,
                        cortext::operations::temporal::RetrievalMode mode,
                        std::uint64_t signal_ts,
                        std::optional<std::uint64_t> retrieval_ts) {
    const auto start = std::chrono::steady_clock::now ();
    const auto order = RunRetrieval (store, encoder,
                                     BuildQueryText (scenario_id, mode),
                                     signal_ts, mode, retrieval_ts, config);
    const auto end = std::chrono::steady_clock::now ();
    metrics.retrieval_runs++;
    metrics.retrieval_ms_sum
        += std::chrono::duration_cast<std::chrono::duration<double, std::milli>> (
               end - start)
               .count ();
    return order;
  };

  for (const auto &scenario : scenarios)
    {
      if (config.representative_subset_only && !scenario.representative_subset)
        {
          continue;
        }

      auto unique_store = cortext::SQLiteStore::Create (":memory:");
      auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
      cortext::store::ApplyMigrations (*store);
      cortext::ProcessorContext pctx;
      scenario.setup (store, pctx, encoder);

      bool scenario_pass = true;
      std::vector<long long> current_top_sequence;
      std::vector<std::string> current_object_sequence;

      for (const auto &query_spec : scenario.queries)
        {
          const std::string actual_object = QueryFirstFactObject (
              *store, query_spec.fact_query_mode, "alice",
              ScenarioPredicate (scenario.id), query_spec.query_ts,
              query_spec.fallback_timestamp, config);

          const auto order = run_timed (store, scenario.id,
                                        query_spec.retrieval_mode,
                                        query_spec.signal_ts,
                                        query_spec.query_ts);
          const bool object_ok = actual_object == query_spec.expected_object;
          const bool retrieval_ok
              = query_spec.expected_top_memory <= 0
                || TopEquals (order, query_spec.expected_top_memory);
          const bool query_pass = object_ok && retrieval_ok;
          scenario_pass = scenario_pass && query_pass;

          switch (query_spec.retrieval_mode)
            {
            case cortext::operations::temporal::RetrievalMode::Current:
              metrics.current_query_checks++;
              metrics.current_query_passes += query_pass ? 1 : 0;
              metrics.current_weight_total += scenario.severity_weight;
              metrics.current_weight_pass += query_pass ? scenario.severity_weight
                                                        : 0.0;
              metrics.retrieval_current_weight_total += scenario.severity_weight;
              metrics.retrieval_current_weight_pass
                  += retrieval_ok ? scenario.severity_weight : 0.0;
              metrics.contradiction_weight_total += scenario.severity_weight;
              metrics.contradiction_weight_hits
                  += object_ok ? 0.0 : scenario.severity_weight;
              metrics.correction_failure_weight_total += scenario.severity_weight;
              metrics.correction_failure_weight_hits
                  += object_ok ? 0.0 : scenario.severity_weight;
              metrics.outdated_first_weight_total += scenario.severity_weight;
              metrics.outdated_first_weight_hits
                  += TopIn (order, query_spec.stale_memory_ids)
                         ? scenario.severity_weight
                         : 0.0;
              metrics.stale_weight_total += scenario.severity_weight;
              metrics.stale_weight_intrusions
                  += TopIn (order, query_spec.stale_memory_ids)
                         ? scenario.severity_weight
                         : 0.0;
              metrics.unsupported_weight_total += scenario.severity_weight;
              metrics.unsupported_weight_hits
                  += (!query_spec.supported_memory_ids.empty ()
                          && !TopIn (order, query_spec.supported_memory_ids))
                         ? scenario.severity_weight
                         : 0.0;
              if (!query_spec.background_memory_ids.empty ())
                {
                  const bool has_background = ContainsAnyTopN (
                      order, query_spec.background_memory_ids,
                      query_spec.background_topn);
                  metrics.routine_weight_total += scenario.severity_weight;
                  metrics.routine_weight_pass
                      += has_background ? scenario.severity_weight : 0.0;
                  metrics.clarification_weight_total += scenario.severity_weight;
                  metrics.clarification_weight_pass
                      += (object_ok && has_background)
                             ? scenario.severity_weight
                             : 0.0;
                }
              if (query_spec.expected_top_memory > 0)
                {
                  current_top_sequence.push_back (
                      order.empty () ? 0LL : order.front ());
                  current_object_sequence.push_back (actual_object);
                }
              break;
            case cortext::operations::temporal::RetrievalMode::ValidAt:
              metrics.valid_query_checks++;
              metrics.valid_query_passes += query_pass ? 1 : 0;
              metrics.historical_weight_total += scenario.severity_weight;
              metrics.historical_weight_pass
                  += query_pass ? scenario.severity_weight : 0.0;
              break;
            case cortext::operations::temporal::RetrievalMode::KnownAt:
              metrics.known_query_checks++;
              metrics.known_query_passes += query_pass ? 1 : 0;
              metrics.belief_weight_total += scenario.severity_weight;
              metrics.belief_weight_pass
                  += query_pass ? scenario.severity_weight : 0.0;
              break;
            }
        }

      if (!scenario.expected_current_sequence.empty ())
        {
          metrics.unexpected_flip_count += CountUnexpectedFlips (
              current_top_sequence, scenario.expected_current_sequence);
        }

      if (!current_object_sequence.empty ())
        {
          std::vector<std::string> expected_current_objects;
          switch (scenario.id)
            {
            case ScenarioId::CaregiverTransition:
              expected_current_objects = { "Sarah", "Emily" };
              break;
            case ScenarioId::DelayedLocation:
              expected_current_objects = { "Denver" };
              break;
            case ScenarioId::ConflictingNoiseGuard:
              expected_current_objects = { "Sarah" };
              break;
            case ScenarioId::RepeatedLocationChanges:
              expected_current_objects = { "Austin", "Denver", "Austin" };
              break;
            case ScenarioId::RoutineException:
            case ScenarioId::WeakScheduleChange:
              expected_current_objects = { "Doctor at noon" };
              break;
            case ScenarioId::OldHomeCurrentHome:
              expected_current_objects = { "Chicago", "Boston" };
              break;
            case ScenarioId::UpdatedPreference:
              expected_current_objects = { "Coffee", "Tea" };
              break;
            case ScenarioId::MedicationChange:
              expected_current_objects = { "Donepezil", "Memantine" };
              break;
            case ScenarioId::ProvenanceSchedule:
              expected_current_objects = { "Doctor today" };
              break;
            }
          AccumulateCorrectionSteps (metrics, current_object_sequence,
                                     expected_current_objects);
        }

      if (scenario.check_erroneous_supersession)
        {
          const auto rows = store->Execute (
              "SELECT canonical_object, superseded_at_ts "
              "FROM fact_assertions "
              "WHERE canonical_subject = 'alice' "
              "ORDER BY recorded_at_ts ASC",
              {});
          if (rows.size () < 2
              || rows[0].at ("superseded_at_ts").type ()
                     != typeid (std::nullptr_t))
            {
              metrics.erroneous_supersession_count++;
              scenario_pass = false;
            }
        }

      if (metrics.unexpected_flip_count > 0
          && scenario.id == ScenarioId::RepeatedLocationChanges)
        {
          scenario_pass = false;
        }

      if (!scenario_pass)
        {
          NoteFailure (metrics, scenario.name, FormatRankedCandidates ());
        }
      metrics.scenario_count++;
      metrics.passed_scenarios += scenario_pass ? 1 : 0;
    }

  return metrics;
}

struct RoutineRecencyProbe
{
  double routine_score = 0.0;
  double current_score = 0.0;
  long long top_memory = 0LL;
};

RoutineRecencyProbe
EvaluateRoutineRecencyProbe (const AblationConfig &config)
{
  RoutineRecencyProbe probe;
  BenchEncoder encoder (g_models_dir);
  cortext::ProcessorContext pctx;
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  const auto scenarios = BuildScenarios ();
  const auto it = std::find_if (
      scenarios.begin (), scenarios.end (), [] (const ScenarioSpec &scenario) {
        return scenario.id == ScenarioId::WeakScheduleChange;
      });
  if (it == scenarios.end ())
    {
      return probe;
    }

  it->setup (store, pctx, encoder);
  {
    cortext::operations::temporal::ScopedRetrievalAblationOverride guard (
        config.override);
    auto tx = store->Begin ();
    cortext::store::FactQuery query;
    query.mode = cortext::store::FactQueryMode::Current;
    query.canonical_subject = "alice";
    query.canonical_predicate = "schedule";
    query.timestamp = 5500ULL;
    const auto facts = cortext::store::QueryFacts (*tx, query);
    for (const auto &fact : facts)
      {
        const auto score
            = cortext::store::ScoreFactRecord (fact, query.mode, query.timestamp);
        if (fact.canonical_object == "lunch_at_noon")
          {
            probe.routine_score = score.combined;
          }
        else if (fact.canonical_object == "doctor_at_noon")
          {
            probe.current_score = score.combined;
          }
      }
  }

  const auto order = RunRetrieval (
      store, encoder,
      BuildQueryText (ScenarioId::WeakScheduleChange,
                      cortext::operations::temporal::RetrievalMode::Current),
      5500ULL,
                                   cortext::operations::temporal::RetrievalMode::Current,
                                   5500ULL, config);
  probe.top_memory = order.empty () ? 0LL : order.front ();
  return probe;
}

void
PrintMetrics (const AggregateMetrics &metrics)
{
  std::cout << "config=" << metrics.name << "\n";
  std::cout << "scenario_count=" << metrics.scenario_count << "\n";
  std::cout << "pass_count=" << metrics.passed_scenarios << "\n";
  std::cout << "query_mode_current_accuracy="
            << SafeRatio (metrics.current_query_passes,
                          metrics.current_query_checks)
            << "\n";
  std::cout << "query_mode_valid_at_accuracy="
            << SafeRatio (metrics.valid_query_passes, metrics.valid_query_checks)
            << "\n";
  std::cout << "query_mode_known_at_accuracy="
            << SafeRatio (metrics.known_query_passes, metrics.known_query_checks)
            << "\n";
  std::cout << "weighted_current_accuracy="
            << SafeRatio (metrics.current_weight_pass,
                          metrics.current_weight_total)
            << "\n";
  std::cout << "weighted_historical_accuracy="
            << SafeRatio (metrics.historical_weight_pass,
                          metrics.historical_weight_total)
            << "\n";
  std::cout << "weighted_belief_accuracy="
            << SafeRatio (metrics.belief_weight_pass,
                          metrics.belief_weight_total)
            << "\n";
  std::cout << "weighted_current_retrieval_accuracy="
            << SafeRatio (metrics.retrieval_current_weight_pass,
                          metrics.retrieval_current_weight_total)
            << "\n";
  std::cout << "weighted_stale_intrusion_rate="
            << SafeRatio (metrics.stale_weight_intrusions,
                          metrics.stale_weight_total)
            << "\n";
  std::cout << "outdated_first_retrieval_rate="
            << SafeRatio (metrics.outdated_first_weight_hits,
                          metrics.outdated_first_weight_total)
            << "\n";
  std::cout << "contradiction_rate="
            << SafeRatio (metrics.contradiction_weight_hits,
                          metrics.contradiction_weight_total)
            << "\n";
  std::cout << "correction_failure_rate="
            << SafeRatio (metrics.correction_failure_weight_hits,
                          metrics.correction_failure_weight_total)
            << "\n";
  std::cout << "unsupported_top_hit_rate="
            << SafeRatio (metrics.unsupported_weight_hits,
                          metrics.unsupported_weight_total)
            << "\n";
  std::cout << "routine_persistence_rate="
            << SafeRatio (metrics.routine_weight_pass,
                          metrics.routine_weight_total)
            << "\n";
  std::cout << "current_vs_usual_clarification_rate="
            << SafeRatio (metrics.clarification_weight_pass,
                          metrics.clarification_weight_total)
            << "\n";
  std::cout << "erroneous_supersession_count="
            << metrics.erroneous_supersession_count << "\n";
  std::cout << "unexpected_flip_count=" << metrics.unexpected_flip_count
            << "\n";
  std::cout << "mean_time_to_correct_updates="
            << SafeRatio (metrics.correction_steps_total,
                          metrics.correction_event_count)
            << "\n";
  std::cout << "mean_retrieval_ms="
            << SafeRatio (metrics.retrieval_ms_sum, metrics.retrieval_runs)
            << "\n";
  for (const auto &failure : metrics.failures)
    {
      std::cout << "failure=" << failure << "\n";
    }
}

} // namespace

int
main (int argc, char **argv)
{
  g_models_dir = cortext::benchmark::ParseModelsDirArg (argc, argv);
  BenchEncoder encoder_probe (g_models_dir);
  std::cout << "encoder_backend=" << encoder_probe.backend_name () << "\n";
  std::cout << "encoder_model_path="
            << encoder_probe.resolved_model_path ().string () << "\n";

  using cortext::operations::temporal::FactBoostStrength;
  using cortext::operations::temporal::ProvenanceMode;
  using cortext::operations::temporal::RoutineRecencyMode;
  using cortext::operations::temporal::StalePenaltyStrength;

  std::vector<AblationConfig> configs;
  configs.push_back ({ "default", 1.0, 0.5, 0.5, false, {} });

  AblationConfig fact_off{ "fact_off" };
  fact_off.override.fact_layer_enabled = false;
  configs.push_back (fact_off);

  AblationConfig current_only{ "current_only" };
  current_only.override.history_enabled = false;
  configs.push_back (current_only);

  AblationConfig boost_off{ "boost_off" };
  boost_off.override.fact_boost_strength = FactBoostStrength::Off;
  configs.push_back (boost_off);

  AblationConfig boost_weak{ "boost_weak" };
  boost_weak.override.fact_boost_strength = FactBoostStrength::Weak;
  configs.push_back (boost_weak);

  AblationConfig boost_strong{ "boost_strong" };
  boost_strong.override.fact_boost_strength = FactBoostStrength::Strong;
  configs.push_back (boost_strong);

  AblationConfig stale_off{ "stale_off" };
  stale_off.override.stale_penalty_strength = StalePenaltyStrength::Off;
  configs.push_back (stale_off);

  AblationConfig stale_moderate{ "stale_moderate" };
  stale_moderate.override.stale_penalty_strength
      = StalePenaltyStrength::Moderate;
  configs.push_back (stale_moderate);

  AblationConfig stale_strong{ "stale_strong" };
  stale_strong.override.stale_penalty_strength = StalePenaltyStrength::Strong;
  configs.push_back (stale_strong);

  AblationConfig provenance_any{ "provenance_any" };
  provenance_any.override.provenance_mode = ProvenanceMode::AnyFactMatch;
  configs.push_back (provenance_any);

  AblationConfig routine_biased{ "routine_biased" };
  routine_biased.override.routine_recency_mode
      = RoutineRecencyMode::RoutineBiased;
  configs.push_back (routine_biased);

  AblationConfig recency_biased{ "recency_biased" };
  recency_biased.override.routine_recency_mode
      = RoutineRecencyMode::RecencyBiased;
  configs.push_back (recency_biased);

  const std::vector<double> sweep_values{ 0.2, 0.5, 0.9 };
  for (double focus : sweep_values)
    {
      for (double sensitivity : sweep_values)
        {
          for (double stability : sweep_values)
            {
              std::ostringstream name;
              name << "fst_f" << static_cast<int> (focus * 10.0) << "_s"
                   << static_cast<int> (sensitivity * 10.0) << "_t"
                   << static_cast<int> (stability * 10.0);
              configs.push_back ({ name.str (), focus, sensitivity, stability,
                                   true, {} });
            }
        }
    }

  std::vector<AggregateMetrics> results;
  results.reserve (configs.size ());
  for (const auto &config : configs)
    {
      auto metrics = EvaluateConfig (config);
      PrintMetrics (metrics);
      results.push_back (metrics);
    }

  const auto find = [&] (const std::string &name) -> const AggregateMetrics & {
    for (const auto &metrics : results)
      {
        if (metrics.name == name)
          {
            return metrics;
          }
      }
    return results.front ();
  };

  const auto &default_metrics = find ("default");
  const auto &fact_off_metrics = find ("fact_off");
  const auto &current_only_metrics = find ("current_only");
  const auto &provenance_any_metrics = find ("provenance_any");
  const auto &stale_off_metrics = find ("stale_off");
  const auto &stale_strong_metrics = find ("stale_strong");
  const auto &boost_off_metrics = find ("boost_off");
  const auto &boost_strong_metrics = find ("boost_strong");
  const auto &routine_biased_metrics = find ("routine_biased");
  const auto &recency_biased_metrics = find ("recency_biased");
  const auto routine_probe = EvaluateRoutineRecencyProbe (routine_biased);
  const auto recency_probe = EvaluateRoutineRecencyProbe (recency_biased);

  const double default_current
      = SafeRatio (default_metrics.current_weight_pass,
                   default_metrics.current_weight_total);
  const double fact_off_current
      = SafeRatio (fact_off_metrics.current_weight_pass,
                   fact_off_metrics.current_weight_total);
  const double default_historical
      = SafeRatio (default_metrics.historical_weight_pass,
                   default_metrics.historical_weight_total);
  const double current_only_historical
      = SafeRatio (current_only_metrics.historical_weight_pass,
                   current_only_metrics.historical_weight_total);
  const double default_belief
      = SafeRatio (default_metrics.belief_weight_pass,
                   default_metrics.belief_weight_total);
  const double current_only_belief
      = SafeRatio (current_only_metrics.belief_weight_pass,
                   current_only_metrics.belief_weight_total);
  const double default_stale
      = SafeRatio (default_metrics.stale_weight_intrusions,
                   default_metrics.stale_weight_total);
  const double fact_off_stale
      = SafeRatio (fact_off_metrics.stale_weight_intrusions,
                   fact_off_metrics.stale_weight_total);
  const double stale_off_rate
      = SafeRatio (stale_off_metrics.stale_weight_intrusions,
                   stale_off_metrics.stale_weight_total);
  const double stale_strong_rate
      = SafeRatio (stale_strong_metrics.stale_weight_intrusions,
                   stale_strong_metrics.stale_weight_total);
  const double default_unsupported
      = SafeRatio (default_metrics.unsupported_weight_hits,
                   default_metrics.unsupported_weight_total);
  const double provenance_any_unsupported
      = SafeRatio (provenance_any_metrics.unsupported_weight_hits,
                   provenance_any_metrics.unsupported_weight_total);
  const double boost_off_current
      = SafeRatio (boost_off_metrics.current_weight_pass,
                   boost_off_metrics.current_weight_total);
  const double boost_strong_current
      = SafeRatio (boost_strong_metrics.current_weight_pass,
                   boost_strong_metrics.current_weight_total);
  const double routine_persistence_routine
      = SafeRatio (routine_biased_metrics.routine_weight_pass,
                   routine_biased_metrics.routine_weight_total);
  const double routine_persistence_recency
      = SafeRatio (recency_biased_metrics.routine_weight_pass,
                   recency_biased_metrics.routine_weight_total);
  const double correction_failure_routine
      = SafeRatio (routine_biased_metrics.correction_failure_weight_hits,
                   routine_biased_metrics.correction_failure_weight_total);
  const double correction_failure_recency
      = SafeRatio (recency_biased_metrics.correction_failure_weight_hits,
                   recency_biased_metrics.correction_failure_weight_total);
  const double routine_preference_gap
      = routine_probe.routine_score - routine_probe.current_score;
  const double recency_preference_gap
      = recency_probe.routine_score - recency_probe.current_score;

  std::cout << "comparison.default_vs_fact_off.current_accuracy_delta="
            << (default_current - fact_off_current) << "\n";
  std::cout << "comparison.default_vs_fact_off.stale_intrusion_delta="
            << (fact_off_stale - default_stale) << "\n";
  std::cout << "comparison.default_vs_current_only.historical_accuracy_delta="
            << (default_historical - current_only_historical) << "\n";
  std::cout << "comparison.default_vs_current_only.belief_accuracy_delta="
            << (default_belief - current_only_belief) << "\n";
  std::cout << "comparison.provenance_any.unsupported_delta="
            << (provenance_any_unsupported - default_unsupported) << "\n";
  std::cout << "comparison.stale_strong_vs_off.intrusion_delta="
            << (stale_off_rate - stale_strong_rate) << "\n";
  std::cout << "comparison.boost_strong_vs_off.current_accuracy_delta="
            << (boost_strong_current - boost_off_current) << "\n";
  std::cout << "comparison.routine_vs_recency.persistence_delta="
            << (routine_persistence_routine - routine_persistence_recency)
            << "\n";
  std::cout << "comparison.recency_vs_routine.correction_failure_delta="
            << (correction_failure_routine - correction_failure_recency)
            << "\n";
  std::cout << "comparison.routine_vs_recency.preference_gap_delta="
            << (routine_preference_gap - recency_preference_gap) << "\n";
  std::cout << "comparison.routine_bias.prefers_routine="
            << (routine_preference_gap > 0.0 ? 1 : 0) << "\n";
  std::cout << "comparison.recency_bias.prefers_current="
            << (recency_preference_gap < 0.0 ? 1 : 0) << "\n";
  std::cout << "comparison.routine_bias.top_memory="
            << routine_probe.top_memory << "\n";
  std::cout << "comparison.recency_bias.top_memory="
            << recency_probe.top_memory << "\n";

  const bool gates_pass
      = default_metrics.passed_scenarios == default_metrics.scenario_count
        && default_current > fact_off_current
        && fact_off_stale > default_stale
        && default_historical > current_only_historical
        && default_belief > current_only_belief
        && provenance_any_unsupported > default_unsupported
        && stale_off_rate > stale_strong_rate
        && boost_strong_current >= boost_off_current
        && routine_preference_gap > 0.0
        && recency_preference_gap < 0.0
        && routine_preference_gap > recency_preference_gap
        && default_metrics.erroneous_supersession_count == 0
        && default_metrics.unexpected_flip_count == 0;

  std::cout << "gates_pass=" << (gates_pass ? 1 : 0) << "\n";
  return gates_pass ? 0 : 1;
}
