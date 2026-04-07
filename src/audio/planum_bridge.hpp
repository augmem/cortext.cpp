#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "cortext/cortext.hpp"
#include "planum/contract/perception_event.hpp"

namespace cortext::audio
{

struct PlanumBridgeResult
{
  std::string source_id;
  std::uint64_t timestamp = 0;
  bool routed_to_text = false;
};

class PlanumBridgeTarget
{
public:
  virtual ~PlanumBridgeTarget () = default;

  virtual Cortext::Context
  ProcessTextAt (const std::string &text, const std::string &source_id,
                 std::uint64_t timestamp)
      = 0;

  virtual Cortext::Context
  ProcessAudio (const float *pcm, std::size_t num_samples,
                const std::string &source_id)
      = 0;
};

class CortextPlanumBridgeTarget final : public PlanumBridgeTarget
{
public:
  explicit CortextPlanumBridgeTarget (Cortext &cortext) noexcept;

  Cortext::Context
  ProcessTextAt (const std::string &text, const std::string &source_id,
                 std::uint64_t timestamp) override;

  Cortext::Context
  ProcessAudio (const float *pcm, std::size_t num_samples,
                const std::string &source_id) override;

private:
  Cortext *cortext_ = nullptr;
};

class PlanumBridge
{
public:
  explicit PlanumBridge (PlanumBridgeTarget &target) noexcept;

  [[nodiscard]] std::optional<PlanumBridgeResult>
  Accept (const planum::contract::PerceptionEvent &event) const;

  [[nodiscard]] static bool
  ShouldRouteToText (const planum::contract::PerceptionEvent &event) noexcept;

  [[nodiscard]] static std::uint64_t
  DeriveTimestamp (const planum::contract::PerceptionEvent &event) noexcept;

  [[nodiscard]] static std::string
  DeriveSourceId (const planum::contract::PerceptionEvent &event);

private:
  PlanumBridgeTarget *target_ = nullptr;
};

} // namespace cortext::audio
