#include <catch2/catch_test_macros.hpp>

#include <cortext/telemetry/telemetry.hpp>

#include <cstdint>

TEST_CASE ("telemetry: noop by default without SDK providers")
{
  cortext::telemetry::ScopedSpan span ("cortext.test.noop");
  CHECK_FALSE (span.IsRecording ());
  cortext::telemetry::AddCounter ("cortext.test.counter", 1);
  cortext::telemetry::RecordHistogram ("cortext.test.histogram", 0.25);
  cortext::telemetry::LogInfo (
      "hello",
      { cortext::telemetry::Attribute::String ("component", "test") });
  SUCCEED ();
}


