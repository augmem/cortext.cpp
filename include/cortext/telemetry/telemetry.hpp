#pragma once

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>

namespace cortext::telemetry
{

/// @brief A low-overhead attribute representation for spans/logs.
///
/// This is intentionally small and avoids any OpenTelemetry SDK types in the
/// public headers so downstream consumers do not need OpenTelemetry includes.
struct Attribute
{
  enum class Type
  {
    kBool,
    kInt64,
    kDouble,
    kString,
  };

  std::string_view key;
  Type type = Type::kString;

  bool bool_value = false;
  std::int64_t int64_value = 0;
  double double_value = 0.0;
  std::string_view string_value;

  static Attribute
  Bool (std::string_view k, bool v)
  {
    Attribute a;
    a.key = k;
    a.type = Type::kBool;
    a.bool_value = v;
    return a;
  }

  static Attribute
  Int64 (std::string_view k, std::int64_t v)
  {
    Attribute a;
    a.key = k;
    a.type = Type::kInt64;
    a.int64_value = v;
    return a;
  }

  static Attribute
  Double (std::string_view k, double v)
  {
    Attribute a;
    a.key = k;
    a.type = Type::kDouble;
    a.double_value = v;
    return a;
  }

  static Attribute
  String (std::string_view k, std::string_view v)
  {
    Attribute a;
    a.key = k;
    a.type = Type::kString;
    a.string_value = v;
    return a;
  }
};

/// @brief An RAII span that also activates the span in the current thread
/// context for automatic propagation.
class ScopedSpan
{
public:
  explicit ScopedSpan (std::string_view name);
  ScopedSpan (std::string_view name, std::initializer_list<Attribute> attrs);
  ~ScopedSpan ();

  ScopedSpan (ScopedSpan &&other) noexcept;
  ScopedSpan &operator= (ScopedSpan &&other) noexcept;

  ScopedSpan (const ScopedSpan &) = delete;
  ScopedSpan &operator= (const ScopedSpan &) = delete;

  bool IsRecording () const;

  void SetAttribute (std::string_view key, bool value);
  void SetAttribute (std::string_view key, std::int64_t value);
  void SetAttribute (std::string_view key, double value);
  void SetAttribute (std::string_view key, std::string_view value);

  void AddEvent (std::string_view name);

  void SetStatusOk ();
  void SetStatusError (std::string_view message);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// @brief Adds delta to a named counter with no attributes.
void AddCounter (std::string_view name, std::uint64_t delta);

/// @brief Records a value into a named histogram with no attributes.
void RecordHistogram (std::string_view name, double value);

/// @brief Emits a structured log entry (no attributes by default).
void LogError (std::string_view message);
void LogWarn (std::string_view message);
void LogInfo (std::string_view message);
void LogDebug (std::string_view message);
void LogTrace (std::string_view message);

void LogError (std::string_view message, std::initializer_list<Attribute> attrs);
void LogWarn (std::string_view message, std::initializer_list<Attribute> attrs);
void LogInfo (std::string_view message, std::initializer_list<Attribute> attrs);
void LogDebug (std::string_view message, std::initializer_list<Attribute> attrs);
void LogTrace (std::string_view message, std::initializer_list<Attribute> attrs);

} // namespace cortext::telemetry


