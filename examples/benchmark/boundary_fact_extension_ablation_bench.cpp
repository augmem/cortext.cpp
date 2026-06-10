#include "../../src/operations/retrieval_debug_state.hpp"
#include "../../src/operations/temporal_retrieval.hpp"

#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/boundary.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/accumulator_state.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <any>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace
{
constexpr int kEmbeddingDim = 256;

enum FamilyBit : int
{
  kBoundarySurprisal = 0,
  kBoundaryNatural,
  kFactLayer,
  kFactHistory,
  kFactStalePenalty,
  kFactProvenance,
  kFamilyCount
};

struct FamilyInfo
{
  const char *name;
  const char *slug;
};

constexpr std::array<FamilyInfo, kFamilyCount> kFamilies = {
  FamilyInfo{ "boundary_surprisal", "boundary_surprisal" },
  FamilyInfo{ "boundary_natural", "boundary_natural" },
  FamilyInfo{ "fact_layer", "fact_layer" },
  FamilyInfo{ "fact_history", "fact_history" },
  FamilyInfo{ "fact_stale_penalty", "fact_stale_penalty" },
  FamilyInfo{ "fact_provenance", "fact_provenance" },
};

class BenchEncoder : public cortext::Encoder
{
public:
  void EncodeText (const std::string &, std::vector<float> &out) override
  {
    out.assign (kEmbeddingDim, 0.0f);
    out[0] = 1.0f;
  }
  void EncodeAudio (const float *, std::size_t, std::vector<float> &out) override
  {
    out.assign (kEmbeddingDim, 0.0f);
    out[0] = 1.0f;
  }
  void EncodeImage (const std::uint8_t *, int, int, int,
                    std::vector<float> &out) override
  {
    out.assign (kEmbeddingDim, 0.0f);
    out[0] = 1.0f;
  }
};

BenchEncoder &
GetBenchEncoder ()
{
  static BenchEncoder encoder;
  return encoder;
}

class ScopedEnvVar
{
public:
  explicit ScopedEnvVar (const char *name) : name_ (name)
  {
    const char *existing = std::getenv (name);
    if (existing)
      {
        had_value_ = true;
        old_value_ = existing;
      }
    unsetenv (name_);
  }
  ScopedEnvVar (const char *name, const std::string &value) : ScopedEnvVar (name)
  {
    setenv (name_, value.c_str (), 1);
  }
  ~ScopedEnvVar ()
  {
    if (had_value_)
      setenv (name_, old_value_.c_str (), 1);
    else
      unsetenv (name_);
  }

private:
  const char *name_;
  bool had_value_ = false;
  std::string old_value_;
};

class NullTransaction : public cortext::Transaction
{
public:
  std::unique_ptr<cortext::Transaction> Begin () override
  {
    return std::make_unique<NullTransaction> ();
  }
  std::vector<std::map<std::string, std::any>>
  Execute (const std::string &, const std::vector<std::any> & = {}) override
  {
    return {};
  }
  void Commit () override {}
  void Rollback () override {}
};

bool
FamilyEnabled (std::uint32_t mask, int bit)
{
  return (mask & (1u << bit)) != 0u;
}

class ScopedMaskEnv
{
public:
  explicit ScopedMaskEnv (std::uint32_t mask)
  {
    configure ("CORTEXT_BOUNDARY_DISABLE_SURPRISAL",
               FamilyEnabled (mask, kBoundarySurprisal));
    configure ("CORTEXT_BOUNDARY_DISABLE_NATURAL",
               FamilyEnabled (mask, kBoundaryNatural));
  }

private:
  void configure (const char *name, bool enabled)
  {
    if (enabled)
      guards_.push_back (std::make_unique<ScopedEnvVar> (name));
    else
      guards_.push_back (std::make_unique<ScopedEnvVar> (name, "1"));
  }
  std::vector<std::unique_ptr<ScopedEnvVar>> guards_;
};

Eigen::VectorXf
MakeVec (float first, float second = 0.0f)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = first;
  v[1] = second;
  const float n = v.norm ();
  if (n > 1e-9f)
    v /= n;
  return v;
}

std::vector<float>
ToFloatVec (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

cortext::Signal
MakeSignal (const Eigen::VectorXf &emb, std::uint64_t ts)
{
  cortext::Signal s;
  s.embedding = emb;
  s.timestamp = ts;
  s.source_id = "bench";
  return s;
}

std::shared_ptr<cortext::Store>
CreateStore ()
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);
  return store;
}

std::shared_ptr<cortext::Store>
GetScenarioStore (int /*slot*/)
{
  return CreateStore ();
}

void
SeedMemoryWithEmbedding (cortext::Store &store, long long memory_id,
                         long long embedding_id, const Eigen::VectorXf &embedding,
                         const std::string &source_id, long long start_ts)
{
  store.Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) VALUES (?, ?, ?)",
      { embedding_id, ToFloatVec (embedding), start_ts });
  store.Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, start_ts, "
      "n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, ?, 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, 1.0, ?)",
      { memory_id, embedding_id, source_id, start_ts, start_ts });
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
      "canonical_predicate, canonical_object, valid_start_ts, valid_end_ts, "
      "recorded_at_ts, superseded_at_ts, confidence, summary_memory_id, created_at) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      { fact_id, subject, predicate, object, canonical_subject,
        canonical_predicate, canonical_object,
        valid_start_ts.has_value () ? std::any (*valid_start_ts) : std::any (),
        valid_end_ts.has_value () ? std::any (*valid_end_ts) : std::any (),
        recorded_at_ts,
        superseded_at_ts.has_value () ? std::any (*superseded_at_ts) : std::any (),
        confidence, summary_memory_id, recorded_at_ts });
}

void
InsertFactCache (cortext::Store &store, long long fact_id, long long embedding_id,
                 const std::string &fact_text, bool is_current,
                 std::optional<long long> valid_start_ts,
                 std::optional<long long> valid_end_ts, long long recorded_at_ts,
                 std::optional<long long> superseded_at_ts, long long updated_at)
{
  store.Execute (
      "INSERT INTO fact_cache "
      "(fact_id, embedding_id, fact_text, is_current, valid_start_ts, valid_end_ts, "
      "recorded_at_ts, superseded_at_ts, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
      { fact_id, embedding_id, fact_text, is_current ? 1LL : 0LL,
        valid_start_ts.has_value () ? std::any (*valid_start_ts) : std::any (),
        valid_end_ts.has_value () ? std::any (*valid_end_ts) : std::any (),
        recorded_at_ts,
        superseded_at_ts.has_value () ? std::any (*superseded_at_ts) : std::any (),
        updated_at });
}

void
LinkFactEvidence (cortext::Store &store, long long fact_id, long long source_memory_id,
                  const std::string &evidence_type, double support_weight)
{
  store.Execute (
      "INSERT INTO fact_evidence (fact_id, source_memory_id, evidence_type, support_weight) "
      "VALUES (?, ?, ?, ?)",
      { fact_id, source_memory_id, evidence_type, support_weight });
}

class ForceRetrievalGateOp : public cortext::IOperation
{
public:
  void Execute (cortext::OperationContext &ctx,
                cortext::Transaction & /*tx*/) const override
  {
    ctx.SetShouldCheckRetrieval (true);
    auto &p = ctx.GetProcessorContext ();
    if (p.memory_stream.empty ())
      p.memory_stream.push_back (ctx.GetSignal ().embedding);
    auto &acc = p.accumulator_states[ctx.GetSignal ().source_id];
    acc.mu_acc = ctx.GetSignal ().embedding;
    acc.c_t = ctx.GetSignal ().embedding;
  }
};

std::vector<long long>
RunFactRetrieval (std::shared_ptr<cortext::Store> store, const Eigen::VectorXf &query,
                  std::uint64_t signal_ts,
                  cortext::operations::temporal::RetrievalMode mode,
                  std::optional<std::uint64_t> retrieval_ts,
                  const cortext::operations::temporal::RetrievalAblationOverride &override)
{
  cortext::SignalProcessor::Config cfg;
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.encoder = &GetBenchEncoder ();
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));
  cortext::operations::temporal::ScopedRetrievalOverride guard (mode, retrieval_ts);
  cortext::operations::temporal::ScopedRetrievalAblationOverride ablation_guard (override);
  cortext::operations::retrieval_debug::ClearLastSelectedEmbeddingOrder ();
  (void)processor.Process (MakeSignal (query, signal_ts));
  return cortext::operations::retrieval_debug::GetLastSelectedEmbeddingOrder ();
}

double
RunBoundarySurprisalScenario (std::uint32_t mask)
{
  ScopedMaskEnv guards (mask);
  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;
  cortext::ProcessorContext pctx;
  cortext::AccumulatorState state;
  state.Reset (MakeVec (1.0f, 0.0f), 1000);
  state.n_signals = 6;
  state.eta_acc = 0.05;
  state.last_signal_ts = 1500;
  pctx.accumulator_states["bench"] = std::move (state);
  cortext::OperationContext ctx (MakeSignal (MakeVec (1.0f, 0.0f), 2500), pctx, cfg);
  ctx.SetAccumulatorDriftStep (0.0);
  ctx.SetAccumulatorCoherence (1.0);
  ctx.SetMetric (cortext::operations::Metric::embedding_surprisal, 1.0);
  cortext::operations::DetectBoundary op;
  NullTransaction tx;
  op.Execute (ctx, tx);
  return ctx.GetFlushRequired () ? 1.0 : 0.0;
}

double
RunBoundaryNaturalScenario (std::uint32_t mask)
{
  ScopedMaskEnv guards (mask);
  cortext::SignalProcessor::Config cfg;
  cfg.encoder = &GetBenchEncoder ();
  cfg.focus = 0.6;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.3;
  cortext::ProcessorContext pctx;
  cortext::AccumulatorState state;
  state.Reset (MakeVec (1.0f, 0.0f), 1000);
  state.n_signals = 6;
  state.eta_acc = 0.05;
  state.last_signal_ts = 1800;
  state.coherence_prev = 1.0;
  pctx.accumulator_states["bench"] = std::move (state);
  cortext::OperationContext ctx (MakeSignal (MakeVec (0.2f, 0.98f), 2400), pctx, cfg);
  ctx.SetAccumulatorDriftStep (0.8);
  ctx.SetAccumulatorCoherence (0.1);
  ctx.SetMetric (cortext::operations::Metric::embedding_surprisal, 0.9);
  cortext::operations::DetectBoundary op;
  NullTransaction tx;
  op.Execute (ctx, tx);
  return ctx.GetFlushRequired () ? 1.0 : 0.0;
}

double
RunFactLayerScenario (std::uint32_t mask)
{
  auto store = GetScenarioStore (kFactLayer);
  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.80f, 0.60f), "linked", 5000LL);
  SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.99f, 0.12f), "distractor", 5000LL);
  store->Execute ("INSERT INTO embeddings(embedding_id, embedding, created_at) VALUES (?, ?, ?)",
                  { 104LL, ToFloatVec (query), 7000LL });
  InsertFactAssertion (*store, 400LL, "Alice", "caregiver", "Emily", "alice", "caregiver",
                       "emily", 5000LL, std::nullopt, 7000LL, std::nullopt, 0.95, 1LL);
  InsertFactCache (*store, 400LL, 104LL, "Alice caregiver Emily", true, 5000LL,
                   std::nullopt, 7000LL, std::nullopt, 8000LL);
  LinkFactEvidence (*store, 400LL, 1LL, "summary", 1.0);
  cortext::operations::temporal::RetrievalAblationOverride ov;
  ov.fact_layer_enabled = FamilyEnabled (mask, kFactLayer);
  ov.history_enabled = true;
  ov.stale_penalty_strength = cortext::operations::temporal::StalePenaltyStrength::Moderate;
  ov.provenance_mode = cortext::operations::temporal::ProvenanceMode::DirectLinkOnly;
  const auto order = RunFactRetrieval (store, query, 8000ULL,
                                       cortext::operations::temporal::RetrievalMode::Current,
                                       8000ULL, ov);
  return (!order.empty () && order.front () == 1LL) ? 1.0 : 0.0;
}

double
RunFactHistoryScenario (std::uint32_t mask)
{
  auto store = GetScenarioStore (kFactHistory);
  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.82f, 0.57f), "austin", 1000LL);
  SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.97f, 0.24f), "denver", 5000LL);
  store->Execute ("INSERT INTO embeddings(embedding_id, embedding, created_at) VALUES (?, ?, ?)",
                  { 102LL, ToFloatVec (query), 7000LL });
  InsertFactAssertion (*store, 200LL, "Alice", "lives_in", "Austin", "alice", "lives_in",
                       "austin", 1000LL, 5000LL, 2000LL, 7000LL, 0.8, 1LL);
  InsertFactAssertion (*store, 201LL, "Alice", "lives_in", "Denver", "alice", "lives_in",
                       "denver", 5000LL, std::nullopt, 7000LL, std::nullopt, 0.9, 2LL);
  InsertFactCache (*store, 200LL, 102LL, "Alice lives_in Austin", false, 1000LL, 5000LL,
                   2000LL, 7000LL, 7000LL);
  InsertFactCache (*store, 201LL, 2LL, "Alice lives_in Denver", true, 5000LL,
                   std::nullopt, 7000LL, std::nullopt, 8000LL);
  LinkFactEvidence (*store, 200LL, 1LL, "summary", 1.0);
  LinkFactEvidence (*store, 201LL, 2LL, "summary", 1.0);
  cortext::operations::temporal::RetrievalAblationOverride ov;
  ov.fact_layer_enabled = true;
  ov.history_enabled = FamilyEnabled (mask, kFactHistory);
  const auto order = RunFactRetrieval (store, query, 8000ULL,
                                       cortext::operations::temporal::RetrievalMode::ValidAt,
                                       4000ULL, ov);
  return (!order.empty () && order.front () == 1LL) ? 1.0 : 0.0;
}

double
RunFactStalePenaltyScenario (std::uint32_t mask)
{
  auto store = GetScenarioStore (kFactStalePenalty);
  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.99f, 0.10f), "sarah", 1000LL);
  SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.72f, 0.69f), "emily", 5000LL);
  store->Execute ("INSERT INTO embeddings(embedding_id, embedding, created_at) VALUES (?, ?, ?)",
                  { 110LL, ToFloatVec (query), 7000LL });
  InsertFactAssertion (*store, 405LL, "Alice", "caregiver", "Sarah", "alice", "caregiver",
                       "sarah", 1000LL, 5000LL, 3000LL, 7000LL, 0.8, 1LL);
  InsertFactAssertion (*store, 406LL, "Alice", "caregiver", "Emily", "alice", "caregiver",
                       "emily", 5000LL, std::nullopt, 7000LL, std::nullopt, 0.9, 2LL);
  InsertFactCache (*store, 405LL, 1LL, "Alice caregiver Sarah", false, 1000LL, 5000LL,
                   3000LL, 7000LL, 7000LL);
  InsertFactCache (*store, 406LL, 110LL, "Alice caregiver Emily", true, 5000LL,
                   std::nullopt, 7000LL, std::nullopt, 8000LL);
  LinkFactEvidence (*store, 405LL, 1LL, "summary", 1.0);
  LinkFactEvidence (*store, 406LL, 2LL, "summary", 1.0);
  cortext::operations::temporal::RetrievalAblationOverride ov;
  ov.fact_layer_enabled = true;
  ov.history_enabled = true;
  ov.stale_penalty_strength = FamilyEnabled (mask, kFactStalePenalty)
                                  ? cortext::operations::temporal::StalePenaltyStrength::Strong
                                  : cortext::operations::temporal::StalePenaltyStrength::Off;
  const auto order = RunFactRetrieval (store, query, 8000ULL,
                                       cortext::operations::temporal::RetrievalMode::Current,
                                       8000ULL, ov);
  return (!order.empty () && order.front () == 2LL) ? 1.0 : 0.0;
}

double
RunFactProvenanceScenario (std::uint32_t mask)
{
  auto store = GetScenarioStore (kFactProvenance);
  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.80f, 0.60f), "linked", 5000LL);
  SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.99f, 0.12f), "distractor", 5000LL);
  store->Execute ("INSERT INTO embeddings(embedding_id, embedding, created_at) VALUES (?, ?, ?)",
                  { 109LL, ToFloatVec (query), 7000LL });
  InsertFactAssertion (*store, 404LL, "Alice", "caregiver", "Emily", "alice", "caregiver",
                       "emily", 5000LL, std::nullopt, 7000LL, std::nullopt, 0.95, 1LL);
  InsertFactCache (*store, 404LL, 109LL, "Alice caregiver Emily", true, 5000LL,
                   std::nullopt, 7000LL, std::nullopt, 8000LL);
  LinkFactEvidence (*store, 404LL, 1LL, "summary", 1.0);
  cortext::operations::temporal::RetrievalAblationOverride ov;
  ov.fact_layer_enabled = true;
  ov.history_enabled = true;
  ov.provenance_mode = FamilyEnabled (mask, kFactProvenance)
                           ? cortext::operations::temporal::ProvenanceMode::DirectLinkOnly
                           : cortext::operations::temporal::ProvenanceMode::AnyFactMatch;
  const auto order = RunFactRetrieval (store, query, 8000ULL,
                                       cortext::operations::temporal::RetrievalMode::Current,
                                       8000ULL, ov);
  return (!order.empty () && order.front () == 1LL) ? 1.0 : 0.0;
}

struct ComboResult
{
  std::uint32_t mask = 0;
  std::array<double, kFamilyCount> scores{};
  double total = 0.0;
};

ComboResult
RunSuite (std::uint32_t mask)
{
  ComboResult out;
  out.mask = mask;
  out.scores[kBoundarySurprisal] = RunBoundarySurprisalScenario (mask);
  out.scores[kBoundaryNatural] = RunBoundaryNaturalScenario (mask);
  out.scores[kFactLayer] = RunFactLayerScenario (mask);
  out.scores[kFactHistory] = RunFactHistoryScenario (mask);
  out.scores[kFactStalePenalty] = RunFactStalePenaltyScenario (mask);
  out.scores[kFactProvenance] = RunFactProvenanceScenario (mask);
  for (double x : out.scores)
    out.total += x;
  return out;
}

int
BitCount (std::uint32_t mask)
{
  int c = 0;
  while (mask)
    {
      c += static_cast<int> (mask & 1u);
      mask >>= 1u;
    }
  return c;
}

std::string
MaskSummary (std::uint32_t mask)
{
  std::ostringstream out;
  bool first = true;
  for (int i = 0; i < kFamilyCount; ++i)
    {
      if (!FamilyEnabled (mask, i))
        continue;
      if (!first)
        out << ",";
      first = false;
      out << kFamilies[static_cast<std::size_t> (i)].slug;
    }
  return first ? "none" : out.str ();
}

std::string
DisabledSummary (std::uint32_t mask)
{
  std::ostringstream out;
  bool first = true;
  for (int i = 0; i < kFamilyCount; ++i)
    {
      if (FamilyEnabled (mask, i))
        continue;
      if (!first)
        out << ",";
      first = false;
      out << kFamilies[static_cast<std::size_t> (i)].slug;
    }
  return first ? "none" : out.str ();
}

} // namespace

int
main ()
{
  const std::uint32_t full_mask = (1u << kFamilyCount) - 1u;
  const std::uint32_t combo_count = (1u << kFamilyCount);
  std::vector<ComboResult> results;
  results.reserve (combo_count);
  for (std::uint32_t mask = 0; mask < combo_count; ++mask)
    results.push_back (RunSuite (mask));

  double best_score = 0.0;
  double full_score = 0.0;
  for (const auto &r : results)
    {
      best_score = std::max (best_score, r.total);
      if (r.mask == full_mask)
        full_score = r.total;
    }

  std::cout << std::fixed << std::setprecision (6);
  std::cout << "boundary_fact_family_count=" << kFamilyCount << "\n";
  std::cout << "boundary_fact_combo_count=" << combo_count << "\n";
  std::cout << "boundary_fact_full_mask_score=" << full_score << "\n";
  std::cout << "boundary_fact_best_score=" << best_score << "\n";
  std::cout << "boundary_fact_full_mask_is_best=" << (full_score == best_score ? 1 : 0) << "\n";
  for (const auto &r : results)
    {
      if (r.mask != full_mask)
        {
          continue;
        }
      for (int bit = 0; bit < kFamilyCount; ++bit)
        {
          std::cout << "boundary_fact_full_mask_component_"
                    << kFamilies[static_cast<std::size_t> (bit)].slug << "="
                    << r.scores[static_cast<std::size_t> (bit)] << "\n";
        }
      break;
    }
  std::cout << "\n| family | mean_marginal | max_score_without | essential_for_best |\n";
  std::cout << "|---|---:|---:|---:|\n";
  for (int bit = 0; bit < kFamilyCount; ++bit)
    {
      double sum = 0.0;
      int count = 0;
      double max_without = -1e9;
      for (const auto &r : results)
        {
          if (FamilyEnabled (r.mask, bit))
            continue;
          const auto neighbor = r.mask | (1u << bit);
          sum += results[neighbor].total - r.total;
          count++;
          max_without = std::max (max_without, r.total);
        }
      const double mean = count > 0 ? sum / static_cast<double> (count) : 0.0;
      std::cout << "| `" << kFamilies[static_cast<std::size_t> (bit)].name << "` | "
                << mean << " | " << max_without << " | "
                << (max_without < best_score ? 1 : 0) << " |\n";
    }
  std::vector<const ComboResult *> sorted;
  for (const auto &r : results)
    sorted.push_back (&r);
  std::sort (sorted.begin (), sorted.end (), [] (const ComboResult *a, const ComboResult *b) {
    if (a->total != b->total)
      return a->total > b->total;
    return BitCount (a->mask) < BitCount (b->mask);
  });
  std::cout << "\n| rank | score | bits_on | enabled | disabled |\n";
  std::cout << "|---|---:|---:|---|---|\n";
  for (std::size_t i = 0; i < std::min<std::size_t> (8, sorted.size ()); ++i)
    {
      std::cout << "| " << (i + 1) << " | " << sorted[i]->total << " | "
                << BitCount (sorted[i]->mask) << " | " << MaskSummary (sorted[i]->mask)
                << " | " << DisabledSummary (sorted[i]->mask) << " |\n";
    }
  return 0;
}
