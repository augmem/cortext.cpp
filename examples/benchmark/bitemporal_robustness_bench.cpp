#include "../../src/operations/retrieval_debug_state.hpp"
#include "../../src/operations/temporal_retrieval.hpp"
#include "../../src/store/facts.hpp"

#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/operations/process_extraction_results.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/signal.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <any>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace
{

constexpr int kEmbeddingDim = 256;

Eigen::VectorXf
MakeVec (float first, float second = 0.0f)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = first;
  v[1] = second;
  const float norm = v.norm ();
  if (norm > 1e-9f)
    {
      v /= norm;
    }
  return v;
}

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

class BenchEncoder final : public cortext::Encoder
{
public:
  void
  EncodeText (const std::string & /*text*/,
              std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }

  void
  EncodeAudio (const float * /*pcm*/, std::size_t /*num_samples*/,
               std::vector<float> &out_embedding) override
  {
    EncodeText ("", out_embedding);
  }

  void
  EncodeImage (const std::uint8_t * /*data*/, int /*width*/, int /*height*/,
               int /*channels*/, std::vector<float> &out_embedding) override
  {
    EncodeText ("", out_embedding);
  }
};

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

  pctx.pending_extraction_results.push_back (std::move (result));
  cortext::OperationContext ctx (MakeSignal (MakeVec (1.0f, 0.0f), signal_ts),
                                 pctx, cfg, &store);
  cortext::operations::ProcessExtractionResults op;
  auto tx = store.Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();
}

std::vector<long long>
RunRetrieval (std::shared_ptr<cortext::Store> store, const Eigen::VectorXf &query,
              std::uint64_t signal_ts,
              cortext::operations::temporal::RetrievalMode mode,
              std::optional<std::uint64_t> retrieval_ts, double focus = 1.0,
              double sensitivity = 0.5, double stability = 0.5)
{
  cortext::SignalProcessor::Config cfg;
  cfg.focus = focus;
  cfg.sensitivity = sensitivity;
  cfg.stability = stability;
  static BenchEncoder encoder;
  cfg.encoder = &encoder;

  auto ops = std::make_unique<cortext::OperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  cortext::operations::temporal::ScopedRetrievalOverride guard (mode,
                                                                retrieval_ts);
  cortext::operations::retrieval_debug::ClearLastSelectedEmbeddingOrder ();
  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  (void)processor.Process (MakeSignal (query, signal_ts));
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
                      std::uint64_t ts)
{
  auto tx = store.Begin ();
  const auto rows = cortext::store::QueryFacts (
      *tx, { mode, std::string (subject), std::string (predicate), ts });
  if (rows.empty ())
    {
      return {};
    }
  return rows.front ().object;
}

int
CountStateFlips (const std::vector<std::string> &states)
{
  int flips = 0;
  for (size_t i = 1; i < states.size (); ++i)
    {
      if (!states[i - 1].empty () && !states[i].empty ()
          && states[i - 1] != states[i])
        {
          flips++;
        }
    }
  return flips;
}

struct ScenarioMetrics
{
  std::string name;
  double severity_weight = 1.0;
  int current_checks = 0;
  int current_passes = 0;
  int historical_checks = 0;
  int historical_passes = 0;
  int belief_checks = 0;
  int belief_passes = 0;
  int retrieval_checks = 0;
  int stale_intrusions = 0;
  int erroneous_supersessions = 0;
  int unexpected_flips = 0;
  int correction_events = 0;
  int correction_steps_total = 0;
  double retrieval_ms_sum = 0.0;
  bool passed = false;
  std::string retrieval_trace;
};

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

} // namespace

int
main ()
{
  std::vector<ScenarioMetrics> scenarios;
  BenchEncoder encoder;
  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);

  {
    ScenarioMetrics m;
    m.name = "delayed_knowledge_location";
    m.severity_weight = 3.0;
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
    cortext::store::ApplyMigrations (*store);
    cortext::ProcessorContext pctx;
    SeedMemoryWithEmbedding (*store, 20LL, 10LL, MakeVec (0.82f, 0.57f),
                             "summary-1", 1000LL);
    SeedMemoryWithEmbedding (*store, 21LL, 11LL, MakeVec (0.97f, 0.24f),
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

    m.current_checks++;
    m.current_passes += QueryFirstFactObject (
                            *store, cortext::store::FactQueryMode::Current,
                            "alice", "lives_in", 8000ULL)
                        == "Denver";
    m.historical_checks++;
    m.historical_passes += QueryFirstFactObject (
                               *store, cortext::store::FactQueryMode::ValidAt,
                               "alice", "lives_in", 4000ULL)
                           == "Austin";
    m.belief_checks++;
    m.belief_passes += QueryFirstFactObject (
                           *store, cortext::store::FactQueryMode::KnownAt,
                           "alice", "lives_in", 6000ULL)
                       == "Austin";

    const auto start = std::chrono::steady_clock::now ();
    const auto order = RunRetrieval (
        store, query, 8000ULL,
        cortext::operations::temporal::RetrievalMode::Current, 8000ULL);
    const auto end = std::chrono::steady_clock::now ();
    m.retrieval_checks++;
    m.stale_intrusions += (order.empty () || order.front () != 21LL) ? 1 : 0;
    if (m.stale_intrusions > 0)
      {
        m.retrieval_trace = FormatRankedCandidates ();
      }
    m.retrieval_ms_sum
        += std::chrono::duration_cast<std::chrono::duration<double, std::milli>> (
               end - start)
               .count ();
    m.passed = m.current_passes == m.current_checks
               && m.historical_passes == m.historical_checks
               && m.belief_passes == m.belief_checks
               && m.stale_intrusions == 0;
    scenarios.push_back (m);
  }

  {
    ScenarioMetrics m;
    m.name = "conflicting_noise_guard_caregiver";
    m.severity_weight = 3.0;
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
    cortext::store::ApplyMigrations (*store);
    cortext::ProcessorContext pctx;
    SeedMemoryWithEmbedding (*store, 20LL, 10LL, MakeVec (0.82f, 0.58f),
                             "summary-1", 1000LL);
    SeedMemoryWithEmbedding (*store, 21LL, 11LL, MakeVec (0.98f, 0.19f),
                             "summary-2", 5000LL);

    std::vector<std::string> states;
    cortext::operations::ExtractionResult first;
    first.summary_id = "summary-1";
    first.facts.push_back (
        { "Alice", "caregiver", "Sarah", 0.95, std::uint64_t (1000),
          std::nullopt });
    RunExtraction (*store, pctx, encoder, std::move (first), 3000ULL, 0.10,
                   0.80);
    states.push_back (QueryFirstFactObject (
        *store, cortext::store::FactQueryMode::Current, "alice", "caregiver",
        3000ULL));

    cortext::operations::ExtractionResult noisy;
    noisy.summary_id = "summary-2";
    noisy.facts.push_back (
        { "Alice", "caregiver", "Emily", 0.35, std::uint64_t (5000),
          std::nullopt });
    RunExtraction (*store, pctx, encoder, std::move (noisy), 7000ULL, 0.10,
                   0.80);
    states.push_back (QueryFirstFactObject (
        *store, cortext::store::FactQueryMode::Current, "alice", "caregiver",
        8000ULL));

    m.current_checks++;
    m.current_passes += states.back () == "Sarah";

    const auto rows = store->Execute (
        "SELECT canonical_object, superseded_at_ts "
        "FROM fact_assertions "
        "WHERE canonical_subject = 'alice' AND canonical_predicate = 'caregiver' "
        "ORDER BY recorded_at_ts ASC",
        {});
    m.erroneous_supersessions
        += (rows.size () < 2
            || rows[0].at ("superseded_at_ts").type () != typeid (std::nullptr_t))
               ? 1
               : 0;
    m.unexpected_flips += CountStateFlips (states);

    const auto start = std::chrono::steady_clock::now ();
    const auto order = RunRetrieval (
        store, query, 8000ULL,
        cortext::operations::temporal::RetrievalMode::Current, 8000ULL);
    const auto end = std::chrono::steady_clock::now ();
    m.retrieval_checks++;
    m.stale_intrusions += (order.empty () || order.front () != 20LL) ? 1 : 0;
    if (m.stale_intrusions > 0)
      {
        m.retrieval_trace = FormatRankedCandidates ();
      }
    m.retrieval_ms_sum
        += std::chrono::duration_cast<std::chrono::duration<double, std::milli>> (
               end - start)
               .count ();
    m.passed = m.current_passes == m.current_checks
               && m.erroneous_supersessions == 0 && m.unexpected_flips == 0
               && m.stale_intrusions == 0;
    scenarios.push_back (m);
  }

  {
    ScenarioMetrics m;
    m.name = "mixed_delayed_conflict_caregiver";
    m.severity_weight = 3.0;
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
    cortext::store::ApplyMigrations (*store);
    cortext::ProcessorContext pctx;
    SeedMemoryWithEmbedding (*store, 20LL, 10LL, MakeVec (0.82f, 0.58f),
                             "summary-1", 1000LL);
    SeedMemoryWithEmbedding (*store, 21LL, 11LL, MakeVec (0.98f, 0.19f),
                             "summary-2", 5000LL);
    SeedMemoryWithEmbedding (*store, 22LL, 12LL, MakeVec (0.80f, 0.60f),
                             "summary-3", 9000LL);

    std::vector<std::string> states;
    cortext::operations::ExtractionResult first;
    first.summary_id = "summary-1";
    first.facts.push_back (
        { "Alice", "caregiver", "Sarah", 0.95, std::uint64_t (1000),
          std::nullopt });
    RunExtraction (*store, pctx, encoder, std::move (first), 3000ULL, 0.10,
                   0.80);
    states.push_back (QueryFirstFactObject (
        *store, cortext::store::FactQueryMode::Current, "alice", "caregiver",
        3000ULL));

    cortext::operations::ExtractionResult noisy;
    noisy.summary_id = "summary-2";
    noisy.facts.push_back (
        { "Alice", "caregiver", "Emily", 0.35, std::uint64_t (5000),
          std::nullopt });
    RunExtraction (*store, pctx, encoder, std::move (noisy), 7000ULL, 0.10,
                   0.80);
    states.push_back (QueryFirstFactObject (
        *store, cortext::store::FactQueryMode::Current, "alice", "caregiver",
        8000ULL));

    cortext::operations::ExtractionResult correction;
    correction.summary_id = "summary-3";
    correction.facts.push_back (
        { "Alice", "caregiver", "Emily", 0.96, std::uint64_t (9000),
          std::nullopt });
    RunExtraction (*store, pctx, encoder, std::move (correction), 9000ULL, 0.10,
                   0.80);
    states.push_back (QueryFirstFactObject (
        *store, cortext::store::FactQueryMode::Current, "alice", "caregiver",
        10000ULL));

    m.current_checks++;
    m.current_passes += states.back () == "Emily";
    m.belief_checks++;
    m.belief_passes += QueryFirstFactObject (
                           *store, cortext::store::FactQueryMode::KnownAt,
                           "alice", "caregiver", 8000ULL)
                       == "Sarah";
    m.correction_events = 1;
    m.correction_steps_total += states.back () == "Emily" ? 0 : 1;
    m.unexpected_flips += std::max (0, CountStateFlips (states) - 1);

    const auto start = std::chrono::steady_clock::now ();
    const auto order = RunRetrieval (
        store, query, 10000ULL,
        cortext::operations::temporal::RetrievalMode::Current, 10000ULL);
    const auto end = std::chrono::steady_clock::now ();
    m.retrieval_checks++;
    m.stale_intrusions += (order.empty () || order.front () != 22LL) ? 1 : 0;
    if (m.stale_intrusions > 0)
      {
        m.retrieval_trace = FormatRankedCandidates ();
      }
    m.retrieval_ms_sum
        += std::chrono::duration_cast<std::chrono::duration<double, std::milli>> (
               end - start)
               .count ();
    m.passed = m.current_passes == m.current_checks
               && m.belief_passes == m.belief_checks
               && m.correction_steps_total == 0 && m.unexpected_flips == 0
               && m.stale_intrusions == 0;
    scenarios.push_back (m);
  }

  {
    ScenarioMetrics m;
    m.name = "repeated_legitimate_location_changes";
    m.severity_weight = 3.0;
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
    cortext::store::ApplyMigrations (*store);
    cortext::ProcessorContext pctx;
    SeedMemoryWithEmbedding (*store, 20LL, 10LL, MakeVec (0.88f, 0.47f),
                             "summary-1", 1000LL);
    SeedMemoryWithEmbedding (*store, 21LL, 11LL, MakeVec (0.84f, 0.54f),
                             "summary-2", 5000LL);
    SeedMemoryWithEmbedding (*store, 22LL, 12LL, MakeVec (0.79f, 0.61f),
                             "summary-3", 9000LL);

    std::vector<std::string> states;
    cortext::operations::ExtractionResult first;
    first.summary_id = "summary-1";
    first.facts.push_back (
        { "Alice", "lives_in", "Austin", 0.90, std::uint64_t (1000),
          std::nullopt });
    RunExtraction (*store, pctx, encoder, std::move (first), 3000ULL);
    states.push_back (QueryFirstFactObject (
        *store, cortext::store::FactQueryMode::Current, "alice", "lives_in",
        3000ULL));

    cortext::operations::ExtractionResult second;
    second.summary_id = "summary-2";
    second.facts.push_back (
        { "Alice", "lives_in", "Denver", 0.94, std::uint64_t (5000),
          std::nullopt });
    RunExtraction (*store, pctx, encoder, std::move (second), 7000ULL);
    states.push_back (QueryFirstFactObject (
        *store, cortext::store::FactQueryMode::Current, "alice", "lives_in",
        8000ULL));

    cortext::operations::ExtractionResult third;
    third.summary_id = "summary-3";
    third.facts.push_back (
        { "Alice", "lives_in", "Austin", 0.97, std::uint64_t (9000),
          std::nullopt });
    RunExtraction (*store, pctx, encoder, std::move (third), 11000ULL);
    states.push_back (QueryFirstFactObject (
        *store, cortext::store::FactQueryMode::Current, "alice", "lives_in",
        12000ULL));

    m.current_checks++;
    m.current_passes += states.back () == "Austin";
    m.historical_checks++;
    m.historical_passes += QueryFirstFactObject (
                               *store, cortext::store::FactQueryMode::ValidAt,
                               "alice", "lives_in", 8000ULL)
                           == "Denver";
    m.unexpected_flips += std::max (0, CountStateFlips (states) - 2);

    const auto start = std::chrono::steady_clock::now ();
    const auto order = RunRetrieval (
        store, query, 12000ULL,
        cortext::operations::temporal::RetrievalMode::Current, 12000ULL);
    const auto end = std::chrono::steady_clock::now ();
    m.retrieval_checks++;
    m.stale_intrusions += (order.empty () || order.front () != 22LL) ? 1 : 0;
    if (m.stale_intrusions > 0)
      {
        m.retrieval_trace = FormatRankedCandidates ();
      }
    m.retrieval_ms_sum
        += std::chrono::duration_cast<std::chrono::duration<double, std::milli>> (
               end - start)
               .count ();
    m.passed = m.current_passes == m.current_checks
               && m.historical_passes == m.historical_checks
               && m.unexpected_flips == 0 && m.stale_intrusions == 0;
    scenarios.push_back (m);
  }

  {
    ScenarioMetrics m;
    m.name = "stable_routine_confirmations";
    m.severity_weight = 2.0;
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
    cortext::store::ApplyMigrations (*store);
    cortext::ProcessorContext pctx;
    SeedMemoryWithEmbedding (*store, 20LL, 10LL, MakeVec (0.90f, 0.43f),
                             "summary-1", 1000LL);
    SeedMemoryWithEmbedding (*store, 21LL, 11LL, MakeVec (0.89f, 0.45f),
                             "summary-2", 3000LL);
    SeedMemoryWithEmbedding (*store, 22LL, 12LL, MakeVec (0.88f, 0.47f),
                             "summary-3", 5000LL);

    std::vector<std::string> states;
    for (int i = 0; i < 3; ++i)
      {
        cortext::operations::ExtractionResult result;
        result.summary_id = "summary-" + std::to_string (i + 1);
        result.facts.push_back (
            { "Alice", "routine", "usual choir", 0.90 - 0.05 * i,
              std::nullopt, std::nullopt });
        RunExtraction (*store, pctx, encoder, std::move (result),
                       static_cast<std::uint64_t> (3000 + 2000 * i));
        states.push_back (QueryFirstFactObject (
            *store, cortext::store::FactQueryMode::Current, "alice", "routine",
            static_cast<std::uint64_t> (3000 + 2000 * i)));
      }

    m.current_checks++;
    m.current_passes += states.back () == "usual choir";
    m.unexpected_flips += CountStateFlips (states);

    auto tx = store->Begin ();
    const auto facts = cortext::store::QueryFacts (
        *tx, { cortext::store::FactQueryMode::Current, std::string ("alice"),
               std::string ("routine"), 8000ULL });
    m.passed = m.current_passes == m.current_checks
               && m.unexpected_flips == 0 && facts.size () == 1
               && !facts.empty () && facts.front ().evidence_count == 3;
    if (!m.passed)
      {
        std::ostringstream detail;
        detail << "facts=" << facts.size ();
        if (!facts.empty ())
          {
            detail << " evidence_count=" << facts.front ().evidence_count
                   << " object=" << facts.front ().object;
          }
        const auto rows = store->Execute (
            "SELECT fact_id, canonical_object, confidence, valid_start_ts, "
            "recorded_at_ts, superseded_at_ts, summary_memory_id "
            "FROM fact_assertions "
            "WHERE canonical_subject = 'alice' AND canonical_predicate = 'routine' "
            "ORDER BY recorded_at_ts ASC",
            {});
        auto any_i64 = [] (const std::map<std::string, std::any> &row,
                           const char *field,
                           long long fallback = -1LL) {
          auto it = row.find (field);
          if (it == row.end () || !it->second.has_value ())
            {
              return fallback;
            }
          if (it->second.type () == typeid (long long))
            {
              return std::any_cast<long long> (it->second);
            }
          if (it->second.type () == typeid (int))
            {
              return static_cast<long long> (std::any_cast<int> (it->second));
            }
          return fallback;
        };
        auto any_double = [] (const std::map<std::string, std::any> &row,
                              const char *field, double fallback = -1.0) {
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
          return fallback;
        };
        auto any_string = [] (const std::map<std::string, std::any> &row,
                              const char *field) {
          auto it = row.find (field);
          if (it == row.end () || !it->second.has_value ()
              || it->second.type () != typeid (std::string))
            {
              return std::string ();
            }
          return std::any_cast<std::string> (it->second);
        };
        detail << " rows=";
        for (const auto &row : rows)
          {
            detail << "{fact_id=" << any_i64 (row, "fact_id")
                   << ", object=" << any_string (row, "canonical_object")
                   << ", conf=" << any_double (row, "confidence")
                   << ", valid_start=" << any_i64 (row, "valid_start_ts")
                   << ", recorded=" << any_i64 (row, "recorded_at_ts")
                   << ", superseded=" << any_i64 (row, "superseded_at_ts")
                   << ", summary=" << any_i64 (row, "summary_memory_id")
                   << "}";
          }
        detail << " states=";
        for (const auto &state : states)
          {
            detail << "[" << state << "]";
          }
        m.retrieval_trace = detail.str ();
      }
    scenarios.push_back (m);
  }

  {
    ScenarioMetrics m;
    m.name = "routine_exception_and_return";
    m.severity_weight = 2.0;
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
    cortext::store::ApplyMigrations (*store);
    cortext::ProcessorContext pctx;
    SeedMemoryWithEmbedding (*store, 20LL, 10LL, MakeVec (0.91f, 0.41f),
                             "summary-1", 1000LL);
    SeedMemoryWithEmbedding (*store, 21LL, 11LL, MakeVec (0.98f, 0.18f),
                             "summary-2", 5000LL);
    SeedMemoryWithEmbedding (*store, 22LL, 12LL, MakeVec (0.92f, 0.39f),
                             "summary-3", 9000LL);

    std::vector<std::string> states;
    cortext::operations::ExtractionResult usual;
    usual.summary_id = "summary-1";
    usual.facts.push_back (
        { "Alice", "routine", "usual breakfast", 0.85, std::uint64_t (1000),
          std::nullopt });
    RunExtraction (*store, pctx, encoder, std::move (usual), 3000ULL);
    states.push_back (QueryFirstFactObject (
        *store, cortext::store::FactQueryMode::Current, "alice", "routine",
        3000ULL));

    cortext::operations::ExtractionResult exception;
    exception.summary_id = "summary-2";
    exception.facts.push_back (
        { "Alice", "routine", "fasting morning", 0.92, std::uint64_t (5000),
          std::nullopt });
    RunExtraction (*store, pctx, encoder, std::move (exception), 7000ULL);
    states.push_back (QueryFirstFactObject (
        *store, cortext::store::FactQueryMode::Current, "alice", "routine",
        8000ULL));

    cortext::operations::ExtractionResult back_to_usual;
    back_to_usual.summary_id = "summary-3";
    back_to_usual.facts.push_back (
        { "Alice", "routine", "usual breakfast", 0.95, std::uint64_t (9000),
          std::nullopt });
    RunExtraction (*store, pctx, encoder, std::move (back_to_usual), 11000ULL);
    states.push_back (QueryFirstFactObject (
        *store, cortext::store::FactQueryMode::Current, "alice", "routine",
        12000ULL));

    m.current_checks++;
    m.current_passes += states.back () == "usual breakfast";
    m.historical_checks++;
    m.historical_passes += QueryFirstFactObject (
                               *store, cortext::store::FactQueryMode::ValidAt,
                               "alice", "routine", 8000ULL)
                           == "fasting morning";
    m.unexpected_flips += std::max (0, CountStateFlips (states) - 2);

    const auto start = std::chrono::steady_clock::now ();
    const auto order = RunRetrieval (
        store, query, 12000ULL,
        cortext::operations::temporal::RetrievalMode::Current, 12000ULL);
    const auto end = std::chrono::steady_clock::now ();
    m.retrieval_checks++;
    m.stale_intrusions += (order.empty () || order.front () != 22LL) ? 1 : 0;
    if (m.stale_intrusions > 0)
      {
        m.retrieval_trace = FormatRankedCandidates ();
      }
    m.retrieval_ms_sum
        += std::chrono::duration_cast<std::chrono::duration<double, std::milli>> (
               end - start)
               .count ();
    m.passed = m.current_passes == m.current_checks
               && m.historical_passes == m.historical_checks
               && m.unexpected_flips == 0 && m.stale_intrusions == 0;
    scenarios.push_back (m);
  }

  double weighted_current_total = 0.0;
  double weighted_current_pass = 0.0;
  double weighted_historical_total = 0.0;
  double weighted_historical_pass = 0.0;
  double weighted_belief_total = 0.0;
  double weighted_belief_pass = 0.0;
  double weighted_retrieval_total = 0.0;
  double weighted_stale_intrusions = 0.0;
  int total_erroneous_supersessions = 0;
  int total_unexpected_flips = 0;
  int total_correction_events = 0;
  int total_correction_steps = 0;
  double total_retrieval_ms = 0.0;
  int total_retrieval_calls = 0;
  int pass_count = 0;

  for (const auto &scenario : scenarios)
    {
      if (scenario.current_checks > 0)
        {
          weighted_current_total
              += scenario.severity_weight * scenario.current_checks;
          weighted_current_pass
              += scenario.severity_weight * scenario.current_passes;
        }
      if (scenario.historical_checks > 0)
        {
          weighted_historical_total
              += scenario.severity_weight * scenario.historical_checks;
          weighted_historical_pass
              += scenario.severity_weight * scenario.historical_passes;
        }
      if (scenario.belief_checks > 0)
        {
          weighted_belief_total += scenario.severity_weight * scenario.belief_checks;
          weighted_belief_pass += scenario.severity_weight * scenario.belief_passes;
        }
      if (scenario.retrieval_checks > 0)
        {
          weighted_retrieval_total
              += scenario.severity_weight * scenario.retrieval_checks;
          weighted_stale_intrusions
              += scenario.severity_weight * scenario.stale_intrusions;
          total_retrieval_ms += scenario.retrieval_ms_sum;
          total_retrieval_calls += scenario.retrieval_checks;
        }
      total_erroneous_supersessions += scenario.erroneous_supersessions;
      total_unexpected_flips += scenario.unexpected_flips;
      total_correction_events += scenario.correction_events;
      total_correction_steps += scenario.correction_steps_total;
      pass_count += scenario.passed ? 1 : 0;

      std::cout << scenario.name << ": pass=" << (scenario.passed ? 1 : 0)
                << " current=" << scenario.current_passes << "/"
                << scenario.current_checks << " historical="
                << scenario.historical_passes << "/" << scenario.historical_checks
                << " belief=" << scenario.belief_passes << "/"
                << scenario.belief_checks << " stale_intrusions="
                << scenario.stale_intrusions
                << " erroneous_supersessions="
                << scenario.erroneous_supersessions
                << " unexpected_flips=" << scenario.unexpected_flips;
      if (!scenario.retrieval_trace.empty ())
        {
          std::cout << " trace=" << scenario.retrieval_trace;
        }
      std::cout << "\n";
    }

  const double weighted_current_accuracy
      = weighted_current_total <= 0.0 ? 1.0
                                      : weighted_current_pass
                                            / weighted_current_total;
  const double weighted_historical_accuracy
      = weighted_historical_total <= 0.0 ? 1.0
                                         : weighted_historical_pass
                                               / weighted_historical_total;
  const double weighted_belief_accuracy
      = weighted_belief_total <= 0.0 ? 1.0
                                     : weighted_belief_pass / weighted_belief_total;
  const double weighted_stale_intrusion_rate
      = weighted_retrieval_total <= 0.0 ? 0.0
                                        : weighted_stale_intrusions
                                              / weighted_retrieval_total;
  const double mean_time_to_correct_updates
      = total_correction_events == 0
            ? 0.0
            : static_cast<double> (total_correction_steps)
                  / static_cast<double> (total_correction_events);
  const double mean_retrieval_ms
      = total_retrieval_calls == 0
            ? 0.0
            : total_retrieval_ms / static_cast<double> (total_retrieval_calls);

  std::cout << "scenario_count=" << scenarios.size () << "\n";
  std::cout << "pass_count=" << pass_count << "\n";
  std::cout << "weighted_current_accuracy=" << weighted_current_accuracy << "\n";
  std::cout << "weighted_historical_accuracy=" << weighted_historical_accuracy
            << "\n";
  std::cout << "weighted_belief_accuracy=" << weighted_belief_accuracy << "\n";
  std::cout << "weighted_stale_intrusion_rate="
            << weighted_stale_intrusion_rate << "\n";
  std::cout << "erroneous_supersession_count="
            << total_erroneous_supersessions << "\n";
  std::cout << "unexpected_flip_count=" << total_unexpected_flips << "\n";
  std::cout << "mean_time_to_correct_updates="
            << mean_time_to_correct_updates << "\n";
  std::cout << "mean_retrieval_ms=" << mean_retrieval_ms << "\n";

  const bool gates_pass
      = pass_count == static_cast<int> (scenarios.size ())
        && weighted_current_accuracy == 1.0
        && weighted_historical_accuracy == 1.0
        && weighted_belief_accuracy == 1.0
        && weighted_stale_intrusion_rate == 0.0
        && total_erroneous_supersessions == 0 && total_unexpected_flips == 0
        && mean_time_to_correct_updates == 0.0;
  std::cout << "gates_pass=" << (gates_pass ? 1 : 0) << "\n";

  return gates_pass ? 0 : 1;
}
