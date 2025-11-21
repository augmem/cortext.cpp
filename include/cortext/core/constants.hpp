#pragma once

namespace cortext::core
{

// Mathematical constants
constexpr float kPi = 3.14159265358979323846f;
constexpr float kE = 2.71828182845904523536f;

// Attention width bounds used across focus-related operations
// Keep consistent with prior usage (π/10 .. π)
constexpr float kAttentionWidthMin = 0.1f * kPi;
constexpr float kAttentionWidthMax = kPi;

} // namespace cortext::core
