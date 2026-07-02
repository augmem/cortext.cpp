// tests/operations_embedding_prediction_error.test.cpp
#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/operations/embedding_prediction_error.hpp>
#include <cortext/operations/metrics.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;
using cortext::operations::Metric;
using cortext::operations::UpdateEmbeddingPredictionError;

namespace
{

static Signal
MakeSignal (const Eigen::VectorXf &embedding, uint64_t ts = 1)
{
  Signal s;
  s.embedding = embedding;
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

} // namespace

TEST_CASE ("Embedding Prediction Error - first signal initializes state",
           "[operations][embedding_prediction_error]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto op = std::make_unique<UpdateEmbeddingPredictionError> ();
  auto ops = std::make_unique<DynamicOperationSet> (std::move (op));

  SignalProcessor processor (cfg, store, std::move (ops));

  Eigen::VectorXf emb = Eigen::VectorXf::Random (256);
  auto out = processor.Process (MakeSignal (emb, 1));

  // First signal: predictor initializes, surprisal is zero
  auto it = out.metrics.find (Metric::embedding_surprisal);
  REQUIRE (it != out.metrics.end ());
  CHECK (it->second == Catch::Approx (0.0));
}

TEST_CASE ("Embedding Prediction Error - identical embeddings yield low "
           "surprisal",
           "[operations][embedding_prediction_error]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto op1 = std::make_unique<UpdateEmbeddingPredictionError> ();
  auto ops = std::make_unique<DynamicOperationSet> (std::move (op1));

  SignalProcessor processor (cfg, store, std::move (ops));

  Eigen::VectorXf emb = Eigen::VectorXf::Ones (256).normalized ();

  // First signal (initializes state)
  processor.Process (MakeSignal (emb, 1));

  // Second signal (identical to expectation)
  auto out = processor.Process (MakeSignal (emb, 2));

  auto it = out.metrics.find (Metric::embedding_surprisal);
  REQUIRE (it != out.metrics.end ());
  // Low surprisal because actual matches predicted (both are the same)
  CHECK (it->second < 0.25);
}

TEST_CASE ("Embedding Prediction Error - orthogonal shift yields high surprisal",
           "[operations][embedding_prediction_error]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto op = std::make_unique<UpdateEmbeddingPredictionError> ();
  auto ops = std::make_unique<DynamicOperationSet> (std::move (op));

  SignalProcessor processor (cfg, store, std::move (ops));

  // Signal 1: unit vector along +x
  Eigen::VectorXf emb1 = Eigen::VectorXf::Zero (256);
  emb1 (0) = 1.0f;
  processor.Process (MakeSignal (emb1, 1));

  // Signal 2: orthogonal shift to +y
  Eigen::VectorXf emb2 = Eigen::VectorXf::Zero (256);
  emb2 (1) = 1.0f;
  auto out = processor.Process (MakeSignal (emb2, 2));

  auto it = out.metrics.find (Metric::embedding_surprisal);
  REQUIRE (it != out.metrics.end ());
  // Orthogonal shift should be highly surprising
  CHECK (it->second > 0.9);
}

TEST_CASE ("Embedding Prediction Error - EMA adapts to repeated input",
           "[operations][embedding_prediction_error]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto op = std::make_unique<UpdateEmbeddingPredictionError> ();
  auto ops = std::make_unique<DynamicOperationSet> (std::move (op));

  SignalProcessor processor (cfg, store, std::move (ops));

  // Signal 1: start position
  Eigen::VectorXf emb1 = Eigen::VectorXf::Zero (256);
  emb1 (0) = 1.0f;
  processor.Process (MakeSignal (emb1, 1));

  // Signal 2: shift to +y (high surprise)
  Eigen::VectorXf emb2 = Eigen::VectorXf::Zero (256);
  emb2 (1) = 1.0f;
  auto out2 = processor.Process (MakeSignal (emb2, 2));

  // Signal 3: repeat +y (EMA should adapt, reducing surprise)
  Eigen::VectorXf emb3 = Eigen::VectorXf::Zero (256);
  emb3 (1) = 1.0f;
  auto out3 = processor.Process (MakeSignal (emb3, 3));

  auto it2 = out2.metrics.find (Metric::embedding_surprisal);
  auto it3 = out3.metrics.find (Metric::embedding_surprisal);

  REQUIRE (it2 != out2.metrics.end ());
  REQUIRE (it3 != out3.metrics.end ());

  // Repeat should be less surprising than the first shift
  CHECK (it3->second < it2->second);
}

TEST_CASE ("Embedding Prediction Error - dimension mismatch resets state",
           "[operations][embedding_prediction_error]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto op = std::make_unique<UpdateEmbeddingPredictionError> ();
  auto ops = std::make_unique<DynamicOperationSet> (std::move (op));

  SignalProcessor processor (cfg, store, std::move (ops));

  // Signal 1: 256 dimensions
  Eigen::VectorXf emb1 = Eigen::VectorXf::Random (256);
  processor.Process (MakeSignal (emb1, 1));

  // Signal 2: 128 dimensions (different size - state should reset)
  Eigen::VectorXf emb2 = Eigen::VectorXf::Random (128);
  auto out = processor.Process (MakeSignal (emb2, 2));

  // After dimension mismatch, state is reset with zero surprisal
  auto it = out.metrics.find (Metric::embedding_surprisal);
  REQUIRE (it != out.metrics.end ());
  CHECK (it->second == Catch::Approx (0.0));
}

TEST_CASE ("Embedding Prediction Error - empty embedding is no-op",
           "[operations][embedding_prediction_error]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto op = std::make_unique<UpdateEmbeddingPredictionError> ();
  auto ops = std::make_unique<DynamicOperationSet> (std::move (op));

  SignalProcessor processor (cfg, store, std::move (ops));

  // Signal with empty embedding
  Eigen::VectorXf empty_emb;
  auto out = processor.Process (MakeSignal (empty_emb, 1));

  // No metric should be set for empty embedding
  auto it = out.metrics.find (Metric::embedding_surprisal);
  REQUIRE (it == out.metrics.end ());
}

TEST_CASE ("Embedding Prediction Error tracks SSE history for ΔSSE",
           "[operations][embedding_prediction_error]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  UpdateEmbeddingPredictionError op;

  Eigen::VectorXf emb1 = Eigen::VectorXf::Zero (8);
  emb1 (0) = 1.0f;
  auto signal1 = MakeSignal (emb1, 1);
  OperationContext ctx1 (signal1, pctx, cfg);
  op.Execute (ctx1, cortext::testing::GetNullTransaction ());
  REQUIRE (!pctx.prediction_error_sse.has_value ());

  Eigen::VectorXf emb2 = Eigen::VectorXf::Zero (8);
  emb2 (0) = 2.0f;
  auto signal2 = MakeSignal (emb2, 2);
  OperationContext ctx2 (signal2, pctx, cfg);
  op.Execute (ctx2, cortext::testing::GetNullTransaction ());
  REQUIRE (pctx.prediction_error_sse.has_value ());
  REQUIRE (!pctx.prediction_error_sse_prev.has_value ());

  Eigen::VectorXf emb3 = Eigen::VectorXf::Zero (8);
  emb3 (0) = 3.0f;
  auto signal3 = MakeSignal (emb3, 3);
  OperationContext ctx3 (signal3, pctx, cfg);
  op.Execute (ctx3, cortext::testing::GetNullTransaction ());
  REQUIRE (pctx.prediction_error_sse.has_value ());
  REQUIRE (pctx.prediction_error_sse_prev.has_value ());
}
