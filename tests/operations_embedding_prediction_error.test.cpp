// tests/operations_embedding_prediction_error.test.cpp
#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
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
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto op = std::make_unique<UpdateEmbeddingPredictionError> ();
  auto ops = std::make_unique<OperationSet> (std::move (op));

  SignalProcessor processor (cfg, store, std::move (ops));

  Eigen::VectorXf emb = Eigen::VectorXf::Random (256);
  auto out = processor.Process (MakeSignal (emb, 1));

  // First signal: no metric should be set (no previous for prediction)
  auto it = out.metrics.find (Metric::embedding_surprisal);
  REQUIRE (it == out.metrics.end ());
}

TEST_CASE ("Embedding Prediction Error - identical embeddings yield low "
           "surprisal",
           "[operations][embedding_prediction_error]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto op1 = std::make_unique<UpdateEmbeddingPredictionError> ();
  auto ops = std::make_unique<OperationSet> (std::move (op1));

  SignalProcessor processor (cfg, store, std::move (ops));

  Eigen::VectorXf emb = Eigen::VectorXf::Ones (256).normalized ();

  // First signal (initializes state)
  processor.Process (MakeSignal (emb, 1));

  // Second signal (identical - zero delta means prediction is previous
  // embedding)
  auto out = processor.Process (MakeSignal (emb, 2));

  auto it = out.metrics.find (Metric::embedding_surprisal);
  REQUIRE (it != out.metrics.end ());
  // Low surprisal because actual matches predicted (both are the same)
  CHECK (it->second < 0.1);
}

TEST_CASE ("Embedding Prediction Error - consistent direction yields moderate "
           "surprisal initially",
           "[operations][embedding_prediction_error]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto op = std::make_unique<UpdateEmbeddingPredictionError> ();
  auto ops = std::make_unique<OperationSet> (std::move (op));

  SignalProcessor processor (cfg, store, std::move (ops));

  // Signal 1: unit vector along first axis
  Eigen::VectorXf emb1 = Eigen::VectorXf::Zero (256);
  emb1 (0) = 1.0f;
  processor.Process (MakeSignal (emb1, 1));

  // Signal 2: moved +x direction
  Eigen::VectorXf emb2 = Eigen::VectorXf::Zero (256);
  emb2 (0) = 2.0f;
  processor.Process (MakeSignal (emb2, 2));

  // Signal 3: continue +x (follows the trend that was just established)
  Eigen::VectorXf emb3 = Eigen::VectorXf::Zero (256);
  emb3 (0) = 3.0f;
  auto out = processor.Process (MakeSignal (emb3, 3));

  auto it = out.metrics.find (Metric::embedding_surprisal);
  REQUIRE (it != out.metrics.end ());
  // Following trend should have reasonable surprisal (not too high)
  CHECK (it->second < 0.5);
}

TEST_CASE ("Embedding Prediction Error - direction reversal yields higher "
           "surprisal",
           "[operations][embedding_prediction_error]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto op = std::make_unique<UpdateEmbeddingPredictionError> ();
  auto ops = std::make_unique<OperationSet> (std::move (op));

  SignalProcessor processor (cfg, store, std::move (ops));

  // Signal 1: start position
  Eigen::VectorXf emb1 = Eigen::VectorXf::Zero (256);
  emb1 (0) = 1.0f;
  processor.Process (MakeSignal (emb1, 1));

  // Signal 2: move +x
  Eigen::VectorXf emb2 = Eigen::VectorXf::Zero (256);
  emb2 (0) = 2.0f;
  auto out2 = processor.Process (MakeSignal (emb2, 2));

  // Signal 3: reverse to -x (opposite of established trend)
  Eigen::VectorXf emb3 = Eigen::VectorXf::Zero (256);
  emb3 (0) = -1.0f;
  auto out3 = processor.Process (MakeSignal (emb3, 3));

  auto it2 = out2.metrics.find (Metric::embedding_surprisal);
  auto it3 = out3.metrics.find (Metric::embedding_surprisal);

  REQUIRE (it2 != out2.metrics.end ());
  REQUIRE (it3 != out3.metrics.end ());

  // Reversal should be more surprising than continuing forward
  CHECK (it3->second > it2->second);
}

TEST_CASE ("Embedding Prediction Error - dimension mismatch resets state",
           "[operations][embedding_prediction_error]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto op = std::make_unique<UpdateEmbeddingPredictionError> ();
  auto ops = std::make_unique<OperationSet> (std::move (op));

  SignalProcessor processor (cfg, store, std::move (ops));

  // Signal 1: 256 dimensions
  Eigen::VectorXf emb1 = Eigen::VectorXf::Random (256);
  processor.Process (MakeSignal (emb1, 1));

  // Signal 2: 128 dimensions (different size - state should reset)
  Eigen::VectorXf emb2 = Eigen::VectorXf::Random (128);
  auto out = processor.Process (MakeSignal (emb2, 2));

  // After dimension mismatch, state is reset and no metric is produced
  auto it = out.metrics.find (Metric::embedding_surprisal);
  REQUIRE (it == out.metrics.end ());
}

TEST_CASE ("Embedding Prediction Error - empty embedding is no-op",
           "[operations][embedding_prediction_error]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto op = std::make_unique<UpdateEmbeddingPredictionError> ();
  auto ops = std::make_unique<OperationSet> (std::move (op));

  SignalProcessor processor (cfg, store, std::move (ops));

  // Signal with empty embedding
  Eigen::VectorXf empty_emb;
  auto out = processor.Process (MakeSignal (empty_emb, 1));

  // No metric should be set for empty embedding
  auto it = out.metrics.find (Metric::embedding_surprisal);
  REQUIRE (it == out.metrics.end ());
}
