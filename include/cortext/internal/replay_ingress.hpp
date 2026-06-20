#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "cortext/cortext.hpp"
#include "cortext/retention.hpp"

namespace cortext::internal
{

/// @brief Internal replay-only ingress helpers for historical benchmark data.
///
/// These helpers preserve source event timestamps for non-text modalities
/// without widening Cortext's public application API.
class ReplayIngress
{
public:
  static Cortext::Context ProcessTextAt (
      Cortext &cortext, const std::string &text,
      const std::string &source_id, std::uint64_t timestamp,
      Retention retention = Retention::Durable,
      bool hydrate_context = true);

  static Cortext::Context ProcessAudioAt (
      Cortext &cortext, const float *pcm, std::size_t num_samples,
      const std::string &source_id, std::uint64_t timestamp,
      Retention retention = Retention::Durable);

  static Cortext::Context ProcessImageAt (
      Cortext &cortext, const std::uint8_t *data, int width, int height,
      int channels, const std::string &source_id, std::uint64_t timestamp,
      Retention retention = Retention::Durable);

  static Cortext::Context ConsolidateAt (Cortext &cortext,
                                         std::uint64_t timestamp);
};

} // namespace cortext::internal
