#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cortext::operations::sparse_retrieval_knobs_internal
{

inline double
ClampUnit (double value)
{
  if (!std::isfinite (value))
    throw std::invalid_argument ("Cortext knob must be finite");
  return std::clamp (value, 0.0, 1.0);
}

inline int
RouteCapacity (double focus, double sensitivity, double stability)
{
  const double f = ClampUnit (focus);
  const double s = ClampUnit (sensitivity);
  const double t = ClampUnit (stability);
  return static_cast<int> (
      std::lround (256.0 + 256.0 * f + 128.0 * s + 128.0 * t));
}

inline int
BackfillBatchSize (double focus, double sensitivity, double stability)
{
  const double f = ClampUnit (focus);
  const double s = ClampUnit (sensitivity);
  const double t = ClampUnit (stability);
  return static_cast<int> (
      std::lround (64.0 + 64.0 * f + 32.0 * s + 32.0 * t));
}

inline int
ActivationIdentityTarget (double focus, double sensitivity,
                          double stability)
{
  return 2 * RouteCapacity (focus, sensitivity, stability)
         + 2 * BackfillBatchSize (focus, sensitivity, stability);
}

// Signal-level payload fallback is part of the same bounded retrieval surface.
// A memory may retain arbitrarily many authoritative signal rows in SQLite,
// but one hydration returns at most B recent rows.
inline int
HydrationFallbackSignalLimit (double focus, double sensitivity,
                              double stability)
{
  return BackfillBatchSize (focus, sensitivity, stability);
}

// Family collapse is data dependent inside the fixed candidate envelope, but
// exact cosine checks must never become another store-sized scan.
inline int
FamilyExactComparisonLimit (double focus, double sensitivity,
                            double stability)
{
  return 2 * RouteCapacity (focus, sensitivity, stability);
}

inline int
GraphNeighborCount (double focus, double sensitivity, double stability)
{
  return std::max (
      8, BackfillBatchSize (focus, sensitivity, stability) / 2);
}

inline int
SearchExpansionBatch (double focus, double sensitivity, double stability)
{
  return std::max (
      8, BackfillBatchSize (focus, sensitivity, stability) / 4);
}

inline int
PublicQueryNodeBudget (double focus, double sensitivity, double stability)
{
  return 5 * RouteCapacity (focus, sensitivity, stability);
}

inline int
PublicQueryEffort (double focus, double sensitivity, double stability)
{
  return 5 * RouteCapacity (focus, sensitivity, stability);
}

// The retained retrieval lifecycle pays a larger, still fixed SQLite HNSW
// envelope so consolidation can reset work without falling below the measured
// identity-quality boundary.  The ramp step reuses the independently derived
// reciprocal-update unit; no packet age, modality, or source label participates.
inline int
RetrievalSawtoothFloor (double focus, double sensitivity, double stability)
{
  return 8 * RouteCapacity (focus, sensitivity, stability);
}

inline int
RetrievalSawtoothPeak (double focus, double sensitivity, double stability)
{
  return 9 * RouteCapacity (focus, sensitivity, stability);
}

inline int
RetrievalSawtoothStep (double focus, double sensitivity, double stability)
{
  return std::max (
      2, BackfillBatchSize (focus, sensitivity, stability) / 16);
}

inline int
OrdinarySealBatch (double focus, double sensitivity, double stability)
{
  return BackfillBatchSize (focus, sensitivity, stability)
         - SearchExpansionBatch (focus, sensitivity, stability);
}

inline int
GraphLevelZeroLinks (double focus, double sensitivity, double stability)
{
  return std::max (16, RouteCapacity (focus, sensitivity, stability) / 4);
}

inline int
MaximumLevel (double focus, double sensitivity, double stability)
{
  int remaining
      = std::max (1, RouteCapacity (focus, sensitivity, stability) - 1);
  int levels = 0;
  while (remaining > 0)
    {
      ++levels;
      remaining >>= 1;
    }
  return std::max (1, levels);
}

inline int
ReciprocalUpdateCount (double focus, double sensitivity, double stability)
{
  return std::max (
      2, BackfillBatchSize (focus, sensitivity, stability) / 16);
}

inline int
ConstructionEffort (double focus, double sensitivity, double stability)
{
  return std::max (
      32,
      static_cast<int> (std::lround (
          static_cast<double> (
              RouteCapacity (focus, sensitivity, stability))
          * 25.0 / 64.0)));
}

inline int
QueryEffort (double focus, double sensitivity, double stability)
{
  const int route_capacity
      = RouteCapacity (focus, sensitivity, stability);
  return std::max (
      route_capacity,
      static_cast<int> (
          std::lround (static_cast<double> (route_capacity) * 5.0 / 2.0)));
}

} // namespace cortext::operations::sparse_retrieval_knobs_internal
