#pragma once

#include "cortext/cortext.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace cortext::internal
{

/// Non-public helper for streaming assistant probe checks.
///
/// This session owns an ephemeral processor that encodes only newly arrived
/// assistant chunks. It never performs durable writes. Completed replies are
/// committed once through the normal durable processor, optionally reusing the
/// embedding accumulated from prior streamed chunks.
class StreamingTextProbeSession
{
public:
  StreamingTextProbeSession (Cortext &cortext, std::string source_id);
  ~StreamingTextProbeSession ();

  StreamingTextProbeSession (const StreamingTextProbeSession &) = delete;
  StreamingTextProbeSession &
  operator= (const StreamingTextProbeSession &)
      = delete;
  StreamingTextProbeSession (StreamingTextProbeSession &&) noexcept;
  StreamingTextProbeSession &
  operator= (StreamingTextProbeSession &&) noexcept;

  Cortext::Context AppendTextChunk (const std::string &text);
  Cortext::Context AppendTextChunkAt (const std::string &text,
                                      std::uint64_t timestamp);
  Cortext::Context CacheTextChunk (const std::string &text);
  Cortext::Context CacheTextChunkAt (const std::string &text,
                                     std::uint64_t timestamp);
  Cortext::Context FinalizeText (const std::string &text);
  Cortext::Context FinalizeTextAt (const std::string &text,
                                   std::uint64_t timestamp);

  void Reset ();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext::internal
