#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <optional>
#include <string>

namespace cortext::internal::experimental_env
{

namespace detail
{
#if defined(CORTEXT_EXPERIMENT_HOOKS)
inline std::optional<std::string>
Get (const char *name)
{
  const char *value = std::getenv (name);
  if (value == nullptr || *value == '\0')
    {
      return std::nullopt;
    }
  return std::string (value);
}

inline std::string
Lower (std::string value)
{
  std::transform (value.begin (), value.end (), value.begin (),
                  [] (unsigned char c) {
                    return static_cast<char> (std::tolower (c));
                  });
  return value;
}
#endif
} // namespace detail

inline bool
Flag (const char *name)
{
#if defined(CORTEXT_EXPERIMENT_HOOKS)
  const auto value = detail::Get (name);
  if (!value)
    {
      return false;
    }
  const std::string lower = detail::Lower (*value);
  return lower == "1" || lower == "true" || lower == "yes"
         || lower == "on";
#else
  (void)name;
  return false;
#endif
}

inline bool
Bool (const char *name, bool fallback)
{
#if defined(CORTEXT_EXPERIMENT_HOOKS)
  const auto value = detail::Get (name);
  if (!value)
    {
      return fallback;
    }
  const std::string lower = detail::Lower (*value);
  return lower == "1" || lower == "true" || lower == "yes"
         || lower == "on";
#else
  (void)name;
  return fallback;
#endif
}

inline std::optional<int>
Int (const char *name)
{
#if defined(CORTEXT_EXPERIMENT_HOOKS)
  const auto value = detail::Get (name);
  if (!value)
    {
      return std::nullopt;
    }
  char *end = nullptr;
  const long parsed = std::strtol (value->c_str (), &end, 10);
  if (end == value->c_str ())
    {
      return std::nullopt;
    }
  return static_cast<int> (parsed);
#else
  (void)name;
  return std::nullopt;
#endif
}

inline std::optional<long long>
Int64 (const char *name)
{
#if defined(CORTEXT_EXPERIMENT_HOOKS)
  const auto value = detail::Get (name);
  if (!value)
    {
      return std::nullopt;
    }
  char *end = nullptr;
  const long long parsed = std::strtoll (value->c_str (), &end, 10);
  if (end == value->c_str ())
    {
      return std::nullopt;
    }
  return parsed;
#else
  (void)name;
  return std::nullopt;
#endif
}

} // namespace cortext::internal::experimental_env
