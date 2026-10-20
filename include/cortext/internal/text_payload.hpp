#pragma once

#include <string_view>

namespace cortext::internal
{

/// True when a signal payload is text the engine may read as text (keyword
/// index, standing memories, working-memory text, lexical query terms). The
/// MIME type decides: text/* is text, any other MIME type is an opaque payload
/// whatever the signal's modality (for example caller Media stored for a text
/// signal). An empty MIME type falls back to the modality for rows written
/// without one.
inline bool
IsTextPayload (std::string_view modality, std::string_view mime)
{
  return mime.empty () ? modality == "text" : mime.substr (0, 5) == "text/";
}

} // namespace cortext::internal
