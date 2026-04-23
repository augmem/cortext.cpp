#include "../../src/store/facts.hpp"

#include <cortext/core/knobs.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/process_extraction_results.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/signal.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <any>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace
{

constexpr int kEmbeddingDim = 256;
constexpr double kFocus = 0.5;
constexpr double kSensitivity = 0.5;
constexpr double kInitialConfidence = 0.90;

struct CorrectionEvent
{
  std::string object;
  double confidence = 0.0;
  std::uint64_t summary_start_ts = 0;
  std::uint64_t signal_ts = 0;
};

struct Scenario
{
  std::string name;
  double stability = 0.5;
  std::vector<CorrectionEvent> corrections;
};

struct ScenarioResult
{
  std::string name;
  double required_confidence = 0.0;
  std::string current_object;
  std::string historical_object;
  long long pizza_superseded = 0;
  long long soup_open = 0;
  long long soup_archived = 0;
  long long soup_latest_confirmation_count = 0;
  bool passed = false;
};

cortext::Signal
MakeSignal (std::uint64_t ts)
{
  cortext::Signal signal;
  signal.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  signal.timestamp = ts;
  signal.source_id = "bench";
  return signal;
}

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
SeedSummary (cortext::Store &store, long long embedding_id, long long memory_id,
             const std::string &summary_id, long long start_ts)
{
  std::vector<float> embedding (kEmbeddingDim, 0.0f);
  embedding[0] = 1.0f;
  store.Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?)",
      { embedding_id, embedding, start_ts });
  store.Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, start_ts, "
      "n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, ?, 'ASSOCIATION', ?, 1, 'text', 0.5, 0.5, 1.0, ?)",
      { memory_id, embedding_id, summary_id, start_ts, start_ts });
}

void
RunExtraction (cortext::Store &store, cortext::ProcessorContext &pctx,
               BenchEncoder &encoder,
               cortext::operations::ExtractionResult result,
               std::uint64_t signal_ts, double stability)
{
  cortext::SignalProcessor::Config cfg;
  cfg.focus = kFocus;
  cfg.sensitivity = kSensitivity;
  cfg.stability = stability;
  cfg.encoder = &encoder;

  pctx.pending_extraction_results.push_back (std::move (result));
  cortext::OperationContext ctx (MakeSignal (signal_ts), pctx, cfg, &store);
  cortext::operations::ProcessExtractionResults op;
  auto tx = store.Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();
}

std::string
CurrentObjectAt (cortext::Store &store, std::uint64_t ts)
{
  auto tx = store.Begin ();
  const auto facts = cortext::store::QueryFacts (
      *tx, { cortext::store::FactQueryMode::Current, std::string ("alice"),
             std::string ("favorite_food"), ts });
  if (facts.empty ())
    {
      return {};
    }
  return facts.front ().object;
}

std::string
ValidObjectAt (cortext::Store &store, std::uint64_t ts)
{
  auto tx = store.Begin ();
  const auto facts = cortext::store::QueryFacts (
      *tx, { cortext::store::FactQueryMode::ValidAt, std::string ("alice"),
             std::string ("favorite_food"), ts });
  if (facts.empty ())
    {
      return {};
    }
  return facts.front ().object;
}

long long
CountFacts (cortext::Store &store, const std::string &canonical_object,
            const char *predicate_suffix)
{
  auto rows = store.Execute (
      std::string ("SELECT COUNT(*) AS c FROM fact_assertions "
                   "WHERE canonical_subject = 'alice' "
                   "  AND canonical_predicate = 'favorite_food' "
                   "  AND canonical_object = ? "
                   "  AND ")
          + predicate_suffix,
      { canonical_object });
  if (rows.empty ())
    {
      return 0;
    }
  const auto &value = rows[0].at ("c");
  if (value.type () == typeid (long long))
    {
      return std::any_cast<long long> (value);
    }
  return static_cast<long long> (std::any_cast<int> (value));
}

long long
LatestConfirmationCount (cortext::Store &store, const std::string &canonical_object)
{
  auto rows = store.Execute (
      "SELECT confirmation_count FROM fact_assertions "
      "WHERE canonical_subject = 'alice' "
      "  AND canonical_predicate = 'favorite_food' "
      "  AND canonical_object = ? "
      "ORDER BY recorded_at_ts DESC LIMIT 1",
      { canonical_object });
  if (rows.empty ())
    {
      return 0;
    }
  const auto &value = rows[0].at ("confirmation_count");
  if (value.type () == typeid (long long))
    {
      return std::any_cast<long long> (value);
    }
  return static_cast<long long> (std::any_cast<int> (value));
}

double
RequiredSupersessionConfidence (double existing_confidence, double sensitivity,
                                double stability)
{
  const double s = cortext::core::SensitivityBias (sensitivity);
  const double t = cortext::core::Clamp (stability, 0.0, 1.0);
  const double margin = cortext::core::Lerp (0.18, 0.03, s)
                        * cortext::core::Lerp (1.10, 0.90, t);
  return cortext::core::Clamp (
      existing_confidence
          + margin
                * (1.0 - cortext::core::Clamp (existing_confidence, 0.0, 1.0)),
      0.0, 1.0);
}

ScenarioResult
RunScenario (const Scenario &scenario)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  BenchEncoder encoder;
  cortext::ProcessorContext pctx;

  SeedSummary (*store, 10000LL, 20000LL, "summary-initial", 1000LL);
  cortext::operations::ExtractionResult initial;
  initial.summary_id = "summary-initial";
  initial.facts.push_back (
      { "Alice", "favorite_food", "Pizza", kInitialConfidence,
        std::nullopt, std::nullopt });
  RunExtraction (*store, pctx, encoder, std::move (initial), 3000ULL,
                 scenario.stability);

  for (std::size_t i = 0; i < scenario.corrections.size (); ++i)
    {
      const auto &correction = scenario.corrections[i];
      const long long embedding_id = 10100LL + 100LL * static_cast<long long> (i);
      const long long memory_id = 20100LL + 100LL * static_cast<long long> (i);
      const std::string summary_id
          = "summary-correction-" + std::to_string (i + 1);
      SeedSummary (*store, embedding_id, memory_id, summary_id,
                   static_cast<long long> (correction.summary_start_ts));

      cortext::operations::ExtractionResult update;
      update.summary_id = summary_id;
      update.facts.push_back (
          { "Alice", "favorite_food", correction.object, correction.confidence,
            std::nullopt, std::nullopt });
      RunExtraction (*store, pctx, encoder, std::move (update),
                     correction.signal_ts, scenario.stability);
    }

  ScenarioResult result;
  result.name = scenario.name;
  result.required_confidence
      = RequiredSupersessionConfidence (kInitialConfidence, kSensitivity,
                                        scenario.stability);
  result.current_object = CurrentObjectAt (*store, 20000ULL);
  result.historical_object = ValidObjectAt (*store, 4000ULL);
  result.pizza_superseded = CountFacts (
      *store, "pizza", "superseded_at_ts IS NOT NULL");
  result.soup_open = CountFacts (
      *store, "soup", "superseded_at_ts IS NULL");
  result.soup_archived = CountFacts (
      *store, "soup", "superseded_at_ts IS NOT NULL");
  result.soup_latest_confirmation_count
      = LatestConfirmationCount (*store, "soup");

  if (scenario.name == "single_weak_contradiction")
    {
      result.passed = result.current_object == "Pizza"
                      && result.historical_object == "Pizza"
                      && result.pizza_superseded == 0
                      && result.soup_open == 0
                      && result.soup_archived == 1;
    }
  else if (scenario.name == "single_strong_contradiction")
    {
      result.passed = result.current_object == "Soup"
                      && result.historical_object == "Pizza"
                      && result.pizza_superseded == 1
                      && result.soup_open == 1
                      && result.soup_archived == 0;
    }
  else if (scenario.name == "strong_contradiction_plus_confirmation")
    {
      result.passed = result.current_object == "Soup"
                      && result.historical_object == "Pizza"
                      && result.pizza_superseded == 1
                      && result.soup_open == 1
                      && result.soup_archived == 0
                      && result.soup_latest_confirmation_count >= 2;
    }
  else if (scenario.name == "repeated_moderate_contradictions")
    {
      result.passed = result.current_object == "Pizza"
                      && result.historical_object == "Pizza"
                      && result.pizza_superseded == 0
                      && result.soup_open == 0
                      && result.soup_archived == 3;
    }

  return result;
}

} // namespace

int
main ()
{
  std::cout << std::fixed << std::setprecision (6);

  const std::vector<Scenario> scenarios = {
    { "single_weak_contradiction",
      0.5,
      { { "Soup", 0.55, 6000ULL, 8000ULL } } },
    { "single_strong_contradiction",
      0.5,
      { { "Soup", 0.96, 6000ULL, 8000ULL } } },
    { "strong_contradiction_plus_confirmation",
      0.5,
      { { "Soup", 0.96, 6000ULL, 8000ULL },
        { "Soup", 0.88, 9000ULL, 11000ULL } } },
    { "repeated_moderate_contradictions",
      0.5,
      { { "Soup", 0.72, 6000ULL, 8000ULL },
        { "Soup", 0.72, 9000ULL, 11000ULL },
        { "Soup", 0.72, 12000ULL, 14000ULL } } },
  };

  bool ok = true;
  for (const auto &scenario : scenarios)
    {
      const auto result = RunScenario (scenario);
      std::cout << "scenario=" << result.name
                << " required_confidence=" << result.required_confidence
                << " current=" << result.current_object
                << " historical=" << result.historical_object
                << " pizza_superseded=" << result.pizza_superseded
                << " soup_open=" << result.soup_open
                << " soup_archived=" << result.soup_archived
                << " soup_confirmation_count="
                << result.soup_latest_confirmation_count
                << " passed=" << (result.passed ? 1 : 0) << "\n";
      ok = ok && result.passed;
    }

  std::cout << "gates_pass=" << (ok ? 1 : 0) << "\n";
  return ok ? 0 : 1;
}
