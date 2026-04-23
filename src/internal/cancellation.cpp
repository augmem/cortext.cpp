#include "cortext/internal/cancellation.hpp"

namespace cortext::internal::detail
{

thread_local std::vector<cortext::StopToken> stop_token_stack;

} // namespace cortext::internal::detail
