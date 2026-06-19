#pragma once

#include "eviction_ablation.hpp"

namespace cortext
{
struct ProcessorContext;
class Transaction;
}

namespace cortext::operations::pressure
{

struct StoragePressureState
{
  bool active = false;
  long long used_bytes = 0;
  long long threshold_bytes = 0;
};

StoragePressureState
ComputeStoragePressureState (Transaction &tx,
                             const eviction::EvictionAblationOverride &override);

StoragePressureState
ComputeCachedStoragePressureState (
    ProcessorContext &ctx, Transaction &tx,
    const eviction::EvictionAblationOverride &override,
    int refresh_interval_signals = 64);

double
GateScale (const StoragePressureState &state,
           double low_pressure_scale);

double
RampScale (const StoragePressureState &state,
           double low_pressure_scale);

} // namespace cortext::operations::pressure
