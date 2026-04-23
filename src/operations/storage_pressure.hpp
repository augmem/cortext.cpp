#pragma once

#include "eviction_ablation.hpp"

namespace cortext
{
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

double
GateScale (const StoragePressureState &state,
           double low_pressure_scale = 0.2);

double
RampScale (const StoragePressureState &state,
           double low_pressure_scale = 0.2);

} // namespace cortext::operations::pressure
