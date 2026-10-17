#pragma once

// Shared definition of conversational contiguity: which LONG_TERM memory is
// the turn adjacent to another. Context hydration attaches adjacent turns to
// retrieved anchors (cortext.cpp), and graph retrieval skips candidates that
// hydration will attach anyway (graph_retrieval.cpp).

#include <string>

namespace cortext::internal::contiguity
{

// Adjacent turns are the next/previous LONG_TERM rows in write order, looked
// up within this many memory ids.
constexpr long long kIdScanSpan = 16;
// ... of the same source stream, written within this gap.
constexpr long long kMaxGapMs = 30LL * 60LL * 1000LL;

// Source stream of a source id: everything before its last '/' (the id
// itself when it has none).
inline std::string
StreamOf (const std::string &source_id)
{
  const auto slash = source_id.rfind ('/');
  return slash == std::string::npos ? source_id : source_id.substr (0, slash);
}

} // namespace cortext::internal::contiguity
