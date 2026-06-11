#include "../../src/operations/retrieval_debug_state.hpp"
#include "include/benchmark_text_encoder.hpp"

#include <cortext/core/sparse.hpp>
#include <cortext/models/aist_gguf_encoder.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>

#include <Eigen/Dense>

#include <any>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace
{
using BenchEncoder = cortext::benchmark::EmbeddingGemmaBenchEncoder;

class ScopedEnvVar
{
public:
  explicit ScopedEnvVar (const char *name) : name_ (name)
  {
    const char *existing = std::getenv (name);
    if (existing != nullptr)
      {
        had_value_ = true;
        old_value_ = existing;
      }
    unsetenv (name_);
  }

  ScopedEnvVar (const char *name, const std::string &value)
      : ScopedEnvVar (name)
  {
    setenv (name_, value.c_str (), 1);
  }

  ~ScopedEnvVar ()
  {
    if (had_value_)
      {
        setenv (name_, old_value_.c_str (), 1);
      }
    else
      {
        unsetenv (name_);
      }
  }

private:
  const char *name_;
  bool had_value_ = false;
  std::string old_value_;
};

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

class SeedProceduralStoreOp : public cortext::IOperation
{
public:
  SeedProceduralStoreOp (long long memory_id, double score)
      : memory_id_ (memory_id), score_ (score)
  {
  }

  void
  Execute (cortext::OperationContext &ctx,
           cortext::Transaction & /*tx*/) const override
  {
    const int key_size = cortext::core::SparseKeySize (
        ctx.GetConfig ().focus, ctx.GetConfig ().sensitivity,
        ctx.GetConfig ().stability);
    const std::string key
        = cortext::core::SparseKey (ctx.GetSignal ().embedding, key_size);
    ctx.GetProcessorContext ().procedural_store[key][memory_id_] = score_;
  }

private:
  long long memory_id_ = 0;
  double score_ = 0.0;
};

struct RankInfo
{
  bool selected = false;
  int rank = std::numeric_limits<int>::max ();
  double score = 0.0;
  double base_level = 0.0;
  double partial_match_penalty = 0.0;
  double recent_inhibition = 0.0;
  double utility = 0.0;
};

struct RetrievalRun
{
  std::vector<cortext::operations::retrieval_debug::RankedCandidate> ranked;
  std::vector<cortext::operations::retrieval_debug::EvidencePacket>
      evidence_packets;
  RankInfo target;
  RankInfo comparison;
  long long evidence_blend_reconstructions = 0;
  double evidence_blend_source_confidence = 0.0;
  double evidence_packet_confidence = 0.0;
};

struct StudyResult
{
  std::string name;
  bool passed = false;
};

std::vector<float>
ToFloatVec (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

Eigen::VectorXf
ToEigen (const std::vector<float> &values)
{
  Eigen::VectorXf out (static_cast<Eigen::Index> (values.size ()));
  for (std::size_t i = 0; i < values.size (); ++i)
    {
      out (static_cast<Eigen::Index> (i)) = values[i];
    }
  return out;
}

double
AnyToDouble (const std::any &value)
{
  if (!value.has_value ())
    {
      return 0.0;
    }
  if (value.type () == typeid (double))
    {
      return std::any_cast<double> (value);
    }
  if (value.type () == typeid (float))
    {
      return static_cast<double> (std::any_cast<float> (value));
    }
  if (value.type () == typeid (long long))
    {
      return static_cast<double> (std::any_cast<long long> (value));
    }
  if (value.type () == typeid (int))
    {
      return static_cast<double> (std::any_cast<int> (value));
    }
  return 0.0;
}

Eigen::VectorXf
EncodeTextEigen256 (BenchEncoder &encoder, const std::string &text)
{
  std::vector<float> full;
  encoder.EncodeText (text, full);
  return ToEigen (cortext::TruncateAistMatryoshka (full, 256));
}

cortext::Signal
MakeSignal (const Eigen::VectorXf &embedding, std::uint64_t ts)
{
  cortext::Signal signal;
  signal.embedding = embedding;
  signal.timestamp = ts;
  signal.source_id = "actr-real-encoder-bench";
  return signal;
}

void
SeedTextMemory (cortext::Store &store, BenchEncoder &encoder, long long id,
                const std::string &text, long long created_at,
                long long last_access = 0, long long retrieved_count = 0,
                long long used_count = 0,
                const std::string &source_id = "actr-real-encoder-bench",
                const std::string &modality = "text",
                long long source_contradiction_count = 0)
{
  const Eigen::VectorXf embedding = EncodeTextEigen256 (encoder, text);
  store.Execute (
      "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { id, ToFloatVec (embedding), created_at });
  store.Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, created_at, last_access, retrieved_count, "
      "used_count, source_reliability, source_contradiction_count) "
      "VALUES(?, ?, ?, 'LONG_TERM', ?, 1, ?, 0.5, 0.5, 1.0, ?, ?, ?, ?, "
      "1.0, ?)",
      { id, id, source_id, created_at, modality, created_at, last_access,
        retrieved_count, used_count, source_contradiction_count });
}

void
SeedFactEvidencePacket (cortext::Store &store, long long fact_id,
                        long long embedding_id, const Eigen::VectorXf &embedding,
                        const std::vector<long long> &source_memory_ids,
                        double confidence, long long ts)
{
  store.Execute (
      "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { embedding_id, ToFloatVec (embedding), ts });
  store.Execute (
      "INSERT INTO fact_assertions("
      "fact_id, subject, predicate, object, canonical_subject, "
      "canonical_predicate, canonical_object, valid_start_ts, recorded_at_ts, "
      "confidence, summary_memory_id, created_at, support_mass, "
      "source_diversity, contradiction_mass, confirmation_count, "
      "compressed_support_count, last_confirmation_ts, lifecycle_state) "
      "VALUES(?, 'Nadia', 'owns', 'cobalt rollback', 'nadia', 'owns', "
      "'cobalt rollback', ?, ?, ?, ?, ?, 0.92, ?, 0.0, ?, ?, ?, 'active')",
      { fact_id,
        ts,
        ts,
        confidence,
        source_memory_ids.empty () ? 0LL : source_memory_ids.front (),
        ts,
        static_cast<long long> (source_memory_ids.size ()),
        static_cast<long long> (source_memory_ids.size ()),
        static_cast<long long> (source_memory_ids.size ()),
        ts });
  store.Execute (
      "INSERT INTO fact_cache(fact_id, embedding_id, fact_text, is_current, "
      "valid_start_ts, recorded_at_ts, updated_at) "
      "VALUES(?, ?, 'Nadia owns cobalt rollback', 1, ?, ?, ?)",
      { fact_id, embedding_id, ts, ts, ts });
  for (const long long memory_id : source_memory_ids)
    {
      store.Execute (
          "INSERT OR REPLACE INTO fact_evidence("
          "fact_id, source_memory_id, evidence_type, support_weight) "
          "VALUES(?, ?, 'episodic', 1.0)",
          { fact_id, memory_id });
    }
}

std::shared_ptr<cortext::Store>
CreateStore ()
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);
  return store;
}

RankInfo
FindRank (const RetrievalRun &run, long long memory_id)
{
  RankInfo info;
  for (int i = 0; i < static_cast<int> (run.ranked.size ()); ++i)
    {
      const auto &candidate = run.ranked[static_cast<std::size_t> (i)];
      if (candidate.memory_id != memory_id)
        {
          continue;
        }
      info.selected = true;
      info.rank = i + 1;
      info.score = candidate.score;
      info.base_level = candidate.activation.base_level;
      info.partial_match_penalty = candidate.activation.partial_match_penalty;
      info.recent_inhibition = candidate.activation.recent_inhibition;
      info.utility = candidate.activation.utility;
      return info;
    }
  return info;
}

RetrievalRun
RunRetrieval (const std::shared_ptr<cortext::Store> &store,
              BenchEncoder &encoder, const std::string &query,
              std::uint64_t timestamp, long long target_id,
              long long comparison_id, bool enable_base_level,
              bool enable_recent_inhibition, bool enable_procedural,
              bool enable_partial_matching = false,
              std::optional<long long> procedural_memory_id = std::nullopt,
              bool enable_evidence_blending = false,
              bool disable_facts = true,
              bool enable_evidence_confidence = false,
              bool disable_source_confidence_filter = false)
{
  std::vector<std::unique_ptr<ScopedEnvVar>> guards;
  guards.push_back (std::make_unique<ScopedEnvVar> (
      "CORTEXT_DISABLE_SOURCE_SEED_GRAPH_EXPANSION", "1"));
  guards.push_back (disable_source_confidence_filter
                        ? std::make_unique<ScopedEnvVar> (
                              "CORTEXT_DISABLE_SOURCE_CONF", "1")
                        : std::make_unique<ScopedEnvVar> (
                              "CORTEXT_DISABLE_SOURCE_CONF"));
  guards.push_back (std::make_unique<ScopedEnvVar> (
      "CORTEXT_DISABLE_DURABLE_SOURCE_SET_RETRIEVAL", "1"));
  guards.push_back (std::make_unique<ScopedEnvVar> (
      "CORTEXT_DISABLE_PRECONSOLIDATED_LABEL_GRAPH", "1"));
  guards.push_back (std::make_unique<ScopedEnvVar> (
      "CORTEXT_DISABLE_TEMPORAL_RETRIEVAL", "1"));
  guards.push_back (
      disable_facts
          ? std::make_unique<ScopedEnvVar> ("CORTEXT_DISABLE_FACTS", "1")
          : std::make_unique<ScopedEnvVar> ("CORTEXT_DISABLE_FACTS"));
  guards.push_back (enable_base_level
                        ? std::make_unique<ScopedEnvVar> (
                              "CORTEXT_ENABLE_BASE_LEVEL_AVAILABILITY", "1")
                        : std::make_unique<ScopedEnvVar> (
                              "CORTEXT_ENABLE_BASE_LEVEL_AVAILABILITY"));
  guards.push_back (enable_recent_inhibition
                        ? std::make_unique<ScopedEnvVar> (
                              "CORTEXT_ENABLE_RECENT_RETRIEVAL_INHIBITION",
                              "1")
                        : std::make_unique<ScopedEnvVar> (
                              "CORTEXT_ENABLE_RECENT_RETRIEVAL_INHIBITION"));
  guards.push_back (enable_procedural
                        ? std::make_unique<ScopedEnvVar> (
                              "CORTEXT_DISABLE_PROCEDURAL_PROACTIVE_RETRIEVAL")
                        : std::make_unique<ScopedEnvVar> (
                              "CORTEXT_DISABLE_PROCEDURAL_PROACTIVE_RETRIEVAL",
                              "1"));
  guards.push_back (enable_partial_matching
                        ? std::make_unique<ScopedEnvVar> (
                              "CORTEXT_ENABLE_PARTIAL_MATCHING_PENALTY", "1")
                        : std::make_unique<ScopedEnvVar> (
                              "CORTEXT_ENABLE_PARTIAL_MATCHING_PENALTY"));
  guards.push_back (enable_evidence_blending
                        ? std::make_unique<ScopedEnvVar> (
                              "CORTEXT_ENABLE_EVIDENCE_BLENDING", "1")
                        : std::make_unique<ScopedEnvVar> (
                              "CORTEXT_ENABLE_EVIDENCE_BLENDING"));
  guards.push_back (enable_evidence_confidence
                        ? std::make_unique<ScopedEnvVar> (
                              "CORTEXT_ENABLE_EVIDENCE_CONFIDENCE", "1")
                        : std::make_unique<ScopedEnvVar> (
                              "CORTEXT_ENABLE_EVIDENCE_CONFIDENCE"));
  guards.push_back (std::make_unique<ScopedEnvVar> (
      "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL"));

  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &encoder;
  cfg.focus = 0.8;
  cfg.sensitivity = 0.7;
  cfg.stability = 0.5;
  cfg.procedural_enabled = true;

  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> ());
  if (procedural_memory_id.has_value ())
    {
      ops->Add (
          std::make_unique<SeedProceduralStoreOp> (*procedural_memory_id, 1.0));
    }
  ops->Add (
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());

  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (EncodeTextEigen256 (encoder, query),
                                 timestamp));
  processor.Flush ();

  RetrievalRun run;
  run.ranked
      = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  run.evidence_packets
      = cortext::operations::retrieval_debug::GetLastEvidencePackets ();
  run.target = FindRank (run, target_id);
  run.comparison = FindRank (run, comparison_id);
  auto rows = store->Execute (
      "SELECT COUNT(*) AS cnt FROM memory_reconstructions "
      "WHERE trigger = 'evidence_blend'",
      {});
  if (!rows.empty ())
    {
      run.evidence_blend_reconstructions
          = cortext::store::AnyToLongLong (rows[0].at ("cnt")).value_or (0);
    }
  rows = store->Execute (
      "SELECT COALESCE(AVG(source_confidence), 0.0) AS avg_conf "
      "FROM memory_reconstructions WHERE trigger = 'evidence_blend'",
      {});
  if (!rows.empty () && rows[0].count ("avg_conf"))
    {
      run.evidence_blend_source_confidence
          = AnyToDouble (rows[0].at ("avg_conf"));
    }
  if (!run.evidence_packets.empty ())
    {
      run.evidence_packet_confidence
          = run.evidence_packets.front ().evidence_confidence;
    }
  return run;
}

void
PrintStudy (const std::string &name, const RetrievalRun &off,
            const RetrievalRun &on, const StudyResult &result)
{
  std::cout << "study=" << name
            << " off_rank="
            << (off.target.selected ? off.target.rank : 0)
            << " on_rank=" << (on.target.selected ? on.target.rank : 0)
            << " off_comparison_rank="
            << (off.comparison.selected ? off.comparison.rank : 0)
            << " on_comparison_rank="
            << (on.comparison.selected ? on.comparison.rank : 0)
            << " off_score=" << std::fixed << std::setprecision (6)
            << off.target.score << " on_score=" << on.target.score
            << " off_comparison_score=" << off.comparison.score
            << " on_comparison_score=" << on.comparison.score
            << " on_base_level=" << on.target.base_level
            << " on_partial_match_penalty="
            << on.comparison.partial_match_penalty
            << " on_recent_inhibition=" << on.comparison.recent_inhibition
            << " on_utility=" << on.target.utility
            << " on_evidence_packets=" << on.evidence_packets.size ()
            << " on_evidence_blend_reconstructions="
            << on.evidence_blend_reconstructions
            << " off_evidence_packet_confidence="
            << off.evidence_packet_confidence
            << " on_evidence_packet_confidence="
            << on.evidence_packet_confidence
            << " off_evidence_blend_source_confidence="
            << off.evidence_blend_source_confidence
            << " on_evidence_blend_source_confidence="
            << on.evidence_blend_source_confidence
            << " passed=" << (result.passed ? 1 : 0) << "\n";
}

StudyResult
RunBaseLevelStudy (BenchEncoder &encoder)
{
  constexpr long long kTarget = 100;
  constexpr long long kComparison = 101;
  constexpr long long kNow = 1'000'000;
  auto store = CreateStore ();
  SeedTextMemory (
      *store, encoder, kTarget,
      "Nadia's websocket deploy rollback checklist covers database dry-run, "
      "canary health checks, and paging the build captain.",
      100'000, kNow - 600'000, 32, 18);
  SeedTextMemory (
      *store, encoder, kComparison,
      "The websocket deploy dashboard tracks canary health checks and service "
      "latency during rollout.",
      110'000);
  SeedTextMemory (*store, encoder, 102,
                  "A design review about icon colors and empty-state copy.",
                  120'000);

  const std::string query
      = "websocket deploy canary health checks and rollback checklist";
  const RetrievalRun off = RunRetrieval (
      store, encoder, query, kNow, kTarget, kComparison, false, false, false);
  const RetrievalRun on = RunRetrieval (
      store, encoder, query, kNow, kTarget, kComparison, true, false, false);

  StudyResult result{ "base_level_availability", false };
  result.passed = off.target.selected
                  && off.comparison.selected
                  && on.target.selected
                  && on.comparison.selected
                  && off.comparison.rank == 1
                  && off.target.rank == 2
                  && on.target.rank == 1
                  && on.comparison.rank == 2
                  && on.target.base_level > off.target.base_level + 0.005;
  PrintStudy (result.name, off, on, result);
  return result;
}

StudyResult
RunRecentInhibitionStudy (BenchEncoder &encoder)
{
  constexpr long long kTarget = 200;
  constexpr long long kComparison = 201;
  constexpr long long kNow = 2'000'000;
  auto store = CreateStore ();
  SeedTextMemory (
      *store, encoder, kComparison,
      "Atlas deploy follow-up checklist covers migration dry-run and canary "
      "rollout.",
      1'000'000, kNow - 20'000, 8, 2);
  SeedTextMemory (
      *store, encoder, kTarget,
      "Nadia owns the rollback note after the Atlas launch.",
      1'010'000);
  SeedTextMemory (
      *store, encoder, 202,
      "The lunch plan moved from Thai food to sandwiches near the office.",
      1'020'000);

  const std::string query = "Atlas deploy follow-up checklist";
  const RetrievalRun off = RunRetrieval (
      store, encoder, query, kNow, kTarget, kComparison, false, false, false);
  const RetrievalRun on = RunRetrieval (
      store, encoder, query, kNow, kTarget, kComparison, false, true, false);

  StudyResult result{ "recent_retrieval_inhibition", false };
  result.passed
      = on.target.selected
        && on.comparison.selected
        && on.comparison.recent_inhibition < -0.001
        && off.comparison.selected
        && off.comparison.rank == 1
        && on.target.rank == 1
        && on.comparison.rank > on.target.rank
        && on.comparison.score < off.comparison.score - 0.001;
  PrintStudy (result.name, off, on, result);
  return result;
}

StudyResult
RunProceduralStudy (BenchEncoder &encoder)
{
  constexpr long long kTarget = 300;
  constexpr long long kComparison = 301;
  constexpr long long kNow = 3'000'000;
  auto store = CreateStore ();
  SeedTextMemory (
      *store, encoder, kTarget,
      "The learned cobalt routine starts in the migration notebook, tags Alex "
      "as build captain, then verifies rollback ownership.",
      1'900'000);
  SeedTextMemory (
      *store, encoder, kComparison,
      "Release checklist compiles tests, pushes artifacts, and checks the "
      "deployment dashboard.",
      1'910'000);
  SeedTextMemory (
      *store, encoder, 302,
      "The onboarding checklist covers laptop setup and repository access.",
      1'920'000);

  const std::string query = "release checklist";
  const RetrievalRun off = RunRetrieval (
      store, encoder, query, kNow, kTarget, kComparison, false, false, false,
      false, kTarget);
  const RetrievalRun on = RunRetrieval (
      store, encoder, query, kNow, kTarget, kComparison, false, false, true,
      false, kTarget);

  StudyResult result{ "procedural_utility", false };
  result.passed = off.target.selected
                  && on.target.selected
                  && on.target.utility > 0.01
                  && off.comparison.selected
                  && on.comparison.selected
                  && off.comparison.rank == 1
                  && off.target.rank == 2
                  && on.target.rank == 1
                  && on.comparison.rank == 2;
  PrintStudy (result.name, off, on, result);
  return result;
}

StudyResult
RunPartialMatchingStudy (BenchEncoder &encoder)
{
  constexpr long long kTarget = 400;
  constexpr long long kComparison = 401;
  constexpr long long kNow = 4'000'000;
  auto store = CreateStore ();
  SeedTextMemory (
      *store, encoder, kComparison,
      "Launch rollback checklist for canary health check ownership.",
      2'900'000, 0, 0, 0, "other-audio-source", "audio", 1);
  SeedTextMemory (
      *store, encoder, kTarget,
      "Nadia's launch rollback note records the canary health check owner.",
      2'910'000);
  SeedTextMemory (
      *store, encoder, 402,
      "The grocery list mentions lemons, tea, and a replacement light bulb.",
      2'920'000);

  const std::string query = "launch rollback checklist canary health check";
  const RetrievalRun off = RunRetrieval (
      store, encoder, query, kNow, kTarget, kComparison, false, false, false,
      false);
  const RetrievalRun on = RunRetrieval (
      store, encoder, query, kNow, kTarget, kComparison, false, false, false,
      true);

  StudyResult result{ "partial_matching_penalty", false };
  result.passed
      = off.target.selected
        && off.comparison.selected
        && on.target.selected
        && on.comparison.selected
        && off.comparison.rank == 1
        && off.target.rank == 2
        && on.target.rank == 1
        && on.comparison.rank == 2
        && on.comparison.partial_match_penalty < -0.001
        && on.target.partial_match_penalty
               > on.comparison.partial_match_penalty;
  PrintStudy (result.name, off, on, result);
  return result;
}

StudyResult
RunEvidenceBlendingStudy (BenchEncoder &encoder)
{
  constexpr long long kTarget = 500;
  constexpr long long kComparison = 501;
  constexpr long long kNow = 5'000'000;
  auto make_store = [&] {
    auto store = CreateStore ();
    SeedTextMemory (
        *store, encoder, kTarget,
        "Nadia documented the cobalt migration rollback checklist with canary "
        "health checks and build captain handoff.",
        3'900'000);
    SeedTextMemory (
        *store, encoder, kComparison,
        "Nadia documented the cobalt migration rollback checklist with canary "
        "health checks and release captain handoff.",
        3'910'000);
    SeedTextMemory (
        *store, encoder, 502,
        "The weekend plan mentions trail shoes, coffee, and a book pickup.",
        3'920'000);
    return store;
  };

  const std::string query
      = "cobalt migration rollback checklist canary health checks build "
        "captain handoff";
  auto off_store = make_store ();
  const RetrievalRun off = RunRetrieval (
      off_store, encoder, query, kNow, kTarget, kComparison, false, false, false,
      false, std::nullopt, false);
  auto on_store = make_store ();
  const RetrievalRun on = RunRetrieval (
      on_store, encoder, query, kNow, kTarget, kComparison, false, false, false,
      false, std::nullopt, true);

  bool packet_has_target = false;
  bool packet_has_comparison = false;
  double packet_weight_sum = 0.0;
  bool normalized_packet = false;
  if (!on.evidence_packets.empty ())
    {
      const auto &packet = on.evidence_packets.front ();
      for (const auto &member : packet.members)
        {
          packet_weight_sum += member.weight;
          packet_has_target = packet_has_target || member.memory_id == kTarget;
          packet_has_comparison
              = packet_has_comparison || member.memory_id == kComparison;
        }
      normalized_packet = std::abs (packet_weight_sum - 1.0) <= 1e-6
                          && packet.members.size () >= 2
                          && packet.score_span <= packet.tie_margin;
    }

  StudyResult result{ "evidence_blending", false };
  result.passed
      = off.target.selected
        && off.comparison.selected
        && on.target.selected
        && on.comparison.selected
        && off.target.rank == on.target.rank
        && off.comparison.rank == on.comparison.rank
        && std::abs (off.target.score - on.target.score) <= 1e-9
        && std::abs (off.comparison.score - on.comparison.score) <= 1e-9
        && off.evidence_packets.empty ()
        && !on.evidence_packets.empty ()
        && packet_has_target
        && packet_has_comparison
        && normalized_packet
        && on.evidence_blend_reconstructions > off.evidence_blend_reconstructions;
  PrintStudy (result.name, off, on, result);
  return result;
}

StudyResult
RunEvidenceWeightedConfidenceStudy (BenchEncoder &encoder)
{
  constexpr long long kTarget = 600;
  constexpr long long kComparison = 601;
  constexpr long long kNow = 6'000'000;
  const std::string query
      = "cobalt migration rollback checklist canary health checks build "
        "captain handoff";
  const Eigen::VectorXf fact_embedding = EncodeTextEigen256 (
      encoder, "Nadia owns cobalt migration rollback checklist build captain "
               "handoff");

  auto make_store = [&] {
    auto store = CreateStore ();
    SeedTextMemory (
        *store, encoder, kTarget,
        "Nadia documented the cobalt migration rollback checklist with canary "
        "health checks and build captain handoff.",
        4'900'000);
    SeedTextMemory (
        *store, encoder, kComparison,
        "Nadia documented the cobalt migration rollback checklist with canary "
        "health checks and release captain handoff.",
        4'910'000);
    SeedTextMemory (
        *store, encoder, 602,
        "The weekend plan mentions trail shoes, coffee, and a book pickup.",
        4'920'000);
    store->Execute (
        "UPDATE memories SET source_reliability = 0.30 "
        "WHERE memory_id IN (?, ?)",
        { kTarget, kComparison });
    SeedFactEvidencePacket (*store, 960LL, 961LL, fact_embedding,
                            { kTarget, kComparison }, 0.93, 4'930'000);
    return store;
  };

  auto off_store = make_store ();
  const RetrievalRun off = RunRetrieval (
      off_store, encoder, query, kNow, kTarget, kComparison, false, false, false,
      false, std::nullopt, true, false, false, true);
  auto on_store = make_store ();
  const RetrievalRun on = RunRetrieval (
      on_store, encoder, query, kNow, kTarget, kComparison, false, false, false,
      false, std::nullopt, true, false, true, true);

  StudyResult result{ "evidence_weighted_confidence", false };
  result.passed
      = off.target.selected
        && off.comparison.selected
        && on.target.selected
        && on.comparison.selected
        && off.target.rank == on.target.rank
        && off.comparison.rank == on.comparison.rank
        && std::abs (off.target.score - on.target.score) <= 1e-9
        && std::abs (off.comparison.score - on.comparison.score) <= 1e-9
        && !off.evidence_packets.empty ()
        && !on.evidence_packets.empty ()
        && off.evidence_packet_confidence == 0.0
        && on.evidence_packet_confidence > 0.60
        && on.evidence_blend_source_confidence
               > off.evidence_blend_source_confidence + 0.15;
  PrintStudy (result.name, off, on, result);
  return result;
}

} // namespace

int
main (int argc, char **argv)
{
  try
    {
      const std::string models_dir
          = cortext::benchmark::ParseModelsDirArg (argc, argv);
      BenchEncoder encoder (models_dir);
      std::cout << "encoder_backend=" << encoder.backend_name ()
                << " model=" << encoder.resolved_model_path ().string () << "\n";

      const std::vector<StudyResult> results{
        RunBaseLevelStudy (encoder),
        RunRecentInhibitionStudy (encoder),
	        RunProceduralStudy (encoder),
	        RunPartialMatchingStudy (encoder),
	        RunEvidenceBlendingStudy (encoder),
	        RunEvidenceWeightedConfidenceStudy (encoder),
	      };

      int passed = 0;
      for (const auto &result : results)
        {
          if (result.passed)
            {
              ++passed;
            }
        }
      std::cout << "summary=" << passed << "/" << results.size () << " passed\n";
      return passed == static_cast<int> (results.size ()) ? 0 : 1;
    }
  catch (const std::exception &e)
    {
      std::cerr << "actr_retrieval_ablation_bench failed: " << e.what ()
                << "\n";
      return 2;
    }
}
