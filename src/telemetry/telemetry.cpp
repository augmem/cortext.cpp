#include "cortext/telemetry/telemetry.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <array>
#include <algorithm>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__EMSCRIPTEN__)

namespace cortext::telemetry
{

struct ScopedSpan::Impl
{
};

ScopedSpan::ScopedSpan (std::string_view) : impl_ (nullptr)
{
}

ScopedSpan::ScopedSpan (std::string_view, std::initializer_list<Attribute>)
    : impl_ (nullptr)
{
}

ScopedSpan::~ScopedSpan () = default;

ScopedSpan::ScopedSpan (ScopedSpan &&other) noexcept : impl_ (nullptr)
{
  impl_ = std::move (other.impl_);
}

ScopedSpan &
ScopedSpan::operator= (ScopedSpan &&other) noexcept
{
  if (this != &other)
    {
      impl_ = std::move (other.impl_);
    }
  return *this;
}

bool
ScopedSpan::IsRecording () const
{
  return false;
}

void
ScopedSpan::SetAttribute (std::string_view, bool)
{
}

void
ScopedSpan::SetAttribute (std::string_view, std::int64_t)
{
}

void
ScopedSpan::SetAttribute (std::string_view, double)
{
}

void
ScopedSpan::SetAttribute (std::string_view, std::string_view)
{
}

void
ScopedSpan::AddEvent (std::string_view)
{
}

void
ScopedSpan::SetStatusOk ()
{
}

void
ScopedSpan::SetStatusError (std::string_view)
{
}

void
AddCounter (std::string_view, std::uint64_t)
{
}

void
RecordHistogram (std::string_view, double)
{
}

void
LogError (std::string_view)
{
}

void
LogWarn (std::string_view)
{
}

void
LogInfo (std::string_view)
{
}

void
LogError (std::string_view, std::initializer_list<Attribute>)
{
}

void
LogWarn (std::string_view, std::initializer_list<Attribute>)
{
}

void
LogInfo (std::string_view, std::initializer_list<Attribute>)
{
}

} // namespace cortext::telemetry

#else

#include <opentelemetry/common/attribute_value.h>
#include <opentelemetry/context/runtime_context.h>
#include <opentelemetry/logs/logger.h>
#include <opentelemetry/logs/provider.h>
#include <opentelemetry/metrics/meter.h>
#include <opentelemetry/metrics/provider.h>
#include <opentelemetry/nostd/shared_ptr.h>
#include <opentelemetry/nostd/string_view.h>
#include <opentelemetry/trace/provider.h>
#include <opentelemetry/trace/scope.h>
#include <opentelemetry/trace/span.h>
#include <opentelemetry/trace/tracer.h>

namespace cortext::telemetry
{

namespace trace_api = opentelemetry::trace;
namespace metrics_api = opentelemetry::metrics;
namespace logs_api = opentelemetry::logs;

namespace
{

struct TelemetryState
{
  std::mutex mu;
  opentelemetry::nostd::shared_ptr<metrics_api::MeterProvider> meter_provider;
  opentelemetry::nostd::shared_ptr<metrics_api::Meter> meter;
  std::unordered_map<std::string, opentelemetry::nostd::shared_ptr<
                                   metrics_api::Counter<std::uint64_t> > >
      counters_u64;
  std::unordered_map<std::string, opentelemetry::nostd::shared_ptr<
                                   metrics_api::Histogram<double> > >
      histograms_f64;
};

TelemetryState &
GetState ()
{
  static TelemetryState state;
  return state;
}

opentelemetry::nostd::shared_ptr<metrics_api::Meter>
GetOrCreateMeterLocked (TelemetryState &state)
{
  const opentelemetry::nostd::shared_ptr<metrics_api::MeterProvider>
      current_provider = metrics_api::Provider::GetMeterProvider ();
  if (state.meter_provider != current_provider)
    {
      state.meter_provider = current_provider;
      state.meter = current_provider->GetMeter ("cortext");
      state.counters_u64.clear ();
      state.histograms_f64.clear ();
    }
  if (!state.meter)
    {
      state.meter = current_provider->GetMeter ("cortext");
    }
  return state.meter;
}

struct StableStringPool
{
  static constexpr std::size_t kMaxStrings = 64;
  static constexpr std::size_t kMaxBytesPerString = 2048;
  std::array<std::string, kMaxStrings> slots;
  std::size_t next_slot = 0;
};

opentelemetry::nostd::string_view
MakeStableStringView (std::string_view input)
{
  thread_local StableStringPool pool;
  const std::size_t slot = pool.next_slot;
  pool.next_slot = (pool.next_slot + 1) % StableStringPool::kMaxStrings;
  std::string &dest = pool.slots[slot];
  const std::size_t copy_len
      = std::min (input.size (), StableStringPool::kMaxBytesPerString);
  dest.assign (input.data (), copy_len);
  return opentelemetry::nostd::string_view (dest.data (), dest.size ());
}

std::vector<std::pair<opentelemetry::nostd::string_view,
                      opentelemetry::common::AttributeValue> >
BuildAttributes (std::initializer_list<Attribute> attrs)
{
  std::vector<std::pair<opentelemetry::nostd::string_view,
                        opentelemetry::common::AttributeValue> >
      kvs;
  kvs.reserve (attrs.size ());
  for (const auto &a : attrs)
    {
      const opentelemetry::nostd::string_view k (a.key.data (), a.key.size ());
      switch (a.type)
        {
        case Attribute::Type::kBool:
          kvs.emplace_back (k, a.bool_value);
          break;
        case Attribute::Type::kInt64:
          kvs.emplace_back (k, a.int64_value);
          break;
        case Attribute::Type::kDouble:
          kvs.emplace_back (k, a.double_value);
          break;
        case Attribute::Type::kString:
        default:
          kvs.emplace_back (k, MakeStableStringView (a.string_value));
          break;
        }
    }
  return kvs;
}

} // namespace

struct ScopedSpan::Impl
{
  opentelemetry::nostd::shared_ptr<trace_api::Span> span;
  std::unique_ptr<trace_api::Scope> scope;
};

ScopedSpan::ScopedSpan (std::string_view name) : impl_ (nullptr)
{
  const opentelemetry::nostd::shared_ptr<trace_api::TracerProvider> provider
      = trace_api::Provider::GetTracerProvider ();
  const opentelemetry::nostd::shared_ptr<trace_api::Tracer> tracer
      = provider->GetTracer ("cortext");
  impl_ = std::make_unique<ScopedSpan::Impl> ();
  impl_->span = tracer->StartSpan (std::string (name));
  impl_->scope = std::make_unique<trace_api::Scope> (impl_->span);
}

ScopedSpan::ScopedSpan (std::string_view name,
                        std::initializer_list<Attribute> attrs)
    : ScopedSpan (name)
{
  for (const auto &a : attrs)
    {
      switch (a.type)
        {
        case Attribute::Type::kBool:
          SetAttribute (a.key, a.bool_value);
          break;
        case Attribute::Type::kInt64:
          SetAttribute (a.key, a.int64_value);
          break;
        case Attribute::Type::kDouble:
          SetAttribute (a.key, a.double_value);
          break;
        case Attribute::Type::kString:
        default:
          SetAttribute (a.key, a.string_value);
          break;
        }
    }
}

ScopedSpan::~ScopedSpan ()
{
  if (impl_ && impl_->span)
    {
      impl_->span->End ();
    }
}

ScopedSpan::ScopedSpan (ScopedSpan &&other) noexcept : impl_ (nullptr)
{
  impl_ = std::move (other.impl_);
}

ScopedSpan &
ScopedSpan::operator= (ScopedSpan &&other) noexcept
{
  if (this != &other)
    {
      impl_ = std::move (other.impl_);
    }
  return *this;
}

bool
ScopedSpan::IsRecording () const
{
  return impl_ && impl_->span && impl_->span->IsRecording ();
}

void
ScopedSpan::SetAttribute (std::string_view key, bool value)
{
  if (!IsRecording ())
    {
      return;
    }
  impl_->span->SetAttribute (std::string (key), value);
}

void
ScopedSpan::SetAttribute (std::string_view key, std::int64_t value)
{
  if (!IsRecording ())
    {
      return;
    }
  impl_->span->SetAttribute (std::string (key), value);
}

void
ScopedSpan::SetAttribute (std::string_view key, double value)
{
  if (!IsRecording ())
    {
      return;
    }
  impl_->span->SetAttribute (std::string (key), value);
}

void
ScopedSpan::SetAttribute (std::string_view key, std::string_view value)
{
  if (!IsRecording ())
    {
      return;
    }
  impl_->span->SetAttribute (std::string (key), std::string (value));
}

void
ScopedSpan::AddEvent (std::string_view name)
{
  if (!IsRecording ())
    {
      return;
    }
  impl_->span->AddEvent (std::string (name));
}

void
ScopedSpan::SetStatusOk ()
{
  if (!IsRecording ())
    {
      return;
    }
  impl_->span->SetStatus (trace_api::StatusCode::kOk);
}

void
ScopedSpan::SetStatusError (std::string_view message)
{
  if (!IsRecording ())
    {
      return;
    }
  impl_->span->SetStatus (trace_api::StatusCode::kError, std::string (message));
}

void
AddCounter (std::string_view name, std::uint64_t delta)
{
  TelemetryState &state = GetState ();
  std::lock_guard<std::mutex> lock (state.mu);
  const opentelemetry::nostd::shared_ptr<metrics_api::Meter> meter
      = GetOrCreateMeterLocked (state);
  if (!meter)
    {
      return;
    }
  auto it = state.counters_u64.find (std::string (name));
  if (it == state.counters_u64.end ())
    {
      opentelemetry::nostd::shared_ptr<metrics_api::Counter<std::uint64_t> > c
          = meter->CreateUInt64Counter (std::string (name));
      it = state.counters_u64.emplace (std::string (name), std::move (c)).first;
    }
  it->second->Add (delta, opentelemetry::context::RuntimeContext::GetCurrent ());
}

void
RecordHistogram (std::string_view name, double value)
{
  TelemetryState &state = GetState ();
  std::lock_guard<std::mutex> lock (state.mu);
  const opentelemetry::nostd::shared_ptr<metrics_api::Meter> meter
      = GetOrCreateMeterLocked (state);
  if (!meter)
    {
      return;
    }
  auto it = state.histograms_f64.find (std::string (name));
  if (it == state.histograms_f64.end ())
    {
      opentelemetry::nostd::shared_ptr<metrics_api::Histogram<double> > h
          = meter->CreateDoubleHistogram (std::string (name));
      it = state.histograms_f64.emplace (std::string (name), std::move (h))
               .first;
    }
  it->second->Record (value,
                      opentelemetry::context::RuntimeContext::GetCurrent ());
}

void
LogError (std::string_view message)
{
  LogError (message, {});
}

void
LogWarn (std::string_view message)
{
  LogWarn (message, {});
}

void
LogInfo (std::string_view message)
{
  LogInfo (message, {});
}

void
LogError (std::string_view message, std::initializer_list<Attribute> attrs)
{
  ScopedSpan span ("cortext.log");
  span.SetAttribute ("log.severity", std::string_view ("error"));
  span.SetAttribute ("log.message", message);
  for (const auto &a : attrs)
    {
      switch (a.type)
        {
        case Attribute::Type::kBool:
          span.SetAttribute (a.key, a.bool_value);
          break;
        case Attribute::Type::kInt64:
          span.SetAttribute (a.key, a.int64_value);
          break;
        case Attribute::Type::kDouble:
          span.SetAttribute (a.key, a.double_value);
          break;
        case Attribute::Type::kString:
        default:
          span.SetAttribute (a.key, a.string_value);
          break;
        }
    }
  span.AddEvent ("cortext.log.error");
  const opentelemetry::nostd::shared_ptr<logs_api::LoggerProvider> provider
      = logs_api::Provider::GetLoggerProvider ();
  const opentelemetry::nostd::shared_ptr<logs_api::Logger> logger
      = provider->GetLogger ("cortext");
  const auto kvs = BuildAttributes (attrs);
  logger->EmitLogRecord (logs_api::Severity::kError,
                         MakeStableStringView (message),
                         kvs);
}

void
LogWarn (std::string_view message, std::initializer_list<Attribute> attrs)
{
  ScopedSpan span ("cortext.log");
  span.SetAttribute ("log.severity", std::string_view ("warn"));
  span.SetAttribute ("log.message", message);
  for (const auto &a : attrs)
    {
      switch (a.type)
        {
        case Attribute::Type::kBool:
          span.SetAttribute (a.key, a.bool_value);
          break;
        case Attribute::Type::kInt64:
          span.SetAttribute (a.key, a.int64_value);
          break;
        case Attribute::Type::kDouble:
          span.SetAttribute (a.key, a.double_value);
          break;
        case Attribute::Type::kString:
        default:
          span.SetAttribute (a.key, a.string_value);
          break;
        }
    }
  span.AddEvent ("cortext.log.warn");
  const opentelemetry::nostd::shared_ptr<logs_api::LoggerProvider> provider
      = logs_api::Provider::GetLoggerProvider ();
  const opentelemetry::nostd::shared_ptr<logs_api::Logger> logger
      = provider->GetLogger ("cortext");
  const auto kvs = BuildAttributes (attrs);
  logger->EmitLogRecord (logs_api::Severity::kWarn,
                         MakeStableStringView (message),
                         kvs);
}

void
LogInfo (std::string_view message, std::initializer_list<Attribute> attrs)
{
  ScopedSpan span ("cortext.log");
  span.SetAttribute ("log.severity", std::string_view ("info"));
  span.SetAttribute ("log.message", message);
  for (const auto &a : attrs)
    {
      switch (a.type)
        {
        case Attribute::Type::kBool:
          span.SetAttribute (a.key, a.bool_value);
          break;
        case Attribute::Type::kInt64:
          span.SetAttribute (a.key, a.int64_value);
          break;
        case Attribute::Type::kDouble:
          span.SetAttribute (a.key, a.double_value);
          break;
        case Attribute::Type::kString:
        default:
          span.SetAttribute (a.key, a.string_value);
          break;
        }
    }
  span.AddEvent ("cortext.log.info");
  const opentelemetry::nostd::shared_ptr<logs_api::LoggerProvider> provider
      = logs_api::Provider::GetLoggerProvider ();
  const opentelemetry::nostd::shared_ptr<logs_api::Logger> logger
      = provider->GetLogger ("cortext");
  const auto kvs = BuildAttributes (attrs);
  logger->EmitLogRecord (logs_api::Severity::kInfo,
                         MakeStableStringView (message),
                         kvs);
}

} // namespace cortext::telemetry

#endif


