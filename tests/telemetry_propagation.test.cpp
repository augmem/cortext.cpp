#include <catch2/catch_test_macros.hpp>

#include <cortext/processor/operation_set.hpp>
#include <cortext/processor.hpp>
#include <cortext/signal.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/telemetry/telemetry.hpp>

namespace {

class NoopOperation : public cortext::IOperation {
public:
  void Execute(cortext::OperationContext& context) const override {
    (void)context;
    // Intentionally empty; span propagation is tested via compilation/runtime.
  }
};

} // namespace

TEST_CASE("telemetry: OperationSet span propagation is safe") {
  // This should be safe even when OTEL exporters are disabled at runtime.
  (void)cortext::telemetry::InitializeFromEnv();

  auto store = cortext::SQLiteStore::Create(":memory:");

  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto ops = std::make_unique<cortext::OperationSet>(
      std::make_unique<NoopOperation>(),
      std::make_unique<NoopOperation>());

  cortext::SignalProcessor processor(cfg, std::move(store), std::move(ops));

  cortext::Signal s;
  s.embedding = Eigen::VectorXf::Ones(4);
  s.timestamp = 1;
  s.source_id = "test";

  const auto out = processor.Process(s);
  (void)out;
  processor.Flush();

  SUCCEED();
}


