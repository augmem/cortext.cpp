#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "../examples/chat/metrics_state.hpp"

TEST_CASE("Usage accumulator tracks exact and estimated attempts",
          "[chat][metrics]") {
  chat::UsageAccumulator accumulator;

  SECTION("exact usage stays exact") {
    chat::AccumulateUsage(
        accumulator, chat::StreamingUsage{120, 34, 154}, 0, 0);

    REQUIRE(accumulator.prompt_tokens == 120);
    REQUIRE(accumulator.completion_tokens == 34);
    REQUIRE(accumulator.total_tokens == 154);
    REQUIRE(chat::GetUsageAccuracy(accumulator) == chat::UsageAccuracy::Exact);
  }

  SECTION("missing usage falls back to character estimates") {
    chat::AccumulateUsage(accumulator, std::nullopt, 41, 9);

    REQUIRE(accumulator.prompt_tokens == 11);
    REQUIRE(accumulator.completion_tokens == 3);
    REQUIRE(accumulator.total_tokens == 14);
    REQUIRE(chat::GetUsageAccuracy(accumulator)
            == chat::UsageAccuracy::Estimated);
  }

  SECTION("mixed attempts stay labeled mixed") {
    chat::AccumulateUsage(
        accumulator, chat::StreamingUsage{80, 20, 100}, 0, 0);
    chat::AccumulateUsage(accumulator, std::nullopt, 16, 16);

    REQUIRE(accumulator.prompt_tokens == 84);
    REQUIRE(accumulator.completion_tokens == 24);
    REQUIRE(accumulator.total_tokens == 108);
    REQUIRE(chat::GetUsageAccuracy(accumulator) == chat::UsageAccuracy::Mixed);
    REQUIRE(std::string(chat::UsageAccuracyLabel(chat::UsageAccuracy::Mixed))
            == "mixed");
  }
}

TEST_CASE("Interrupt rate helper handles empty and populated histories",
          "[chat][metrics]") {
  REQUIRE(chat::ComputeInterruptRate(0, 0) == 0.0);
  REQUIRE(chat::ComputeInterruptRate(5, 2) == Catch::Approx(0.4));
}
