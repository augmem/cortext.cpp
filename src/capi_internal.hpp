#pragma once

#include <cortext/cortext.hpp>
#include <string>

namespace cortext::internal
{

#if defined(CORTEXT_TESTING)
std::string ContextToJsonForTest (const Cortext::Context &context,
                                  bool include_embedding = true);
#endif

} // namespace cortext::internal
