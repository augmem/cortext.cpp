#pragma once

namespace cortext
{

/// @brief Controls storage eligibility and turn-edge policy for a signal.
///
/// When retention is omitted from Process* APIs, the default is Natural:
/// episode algorithms decide the boundary and write. Durable keeps the
/// historical chat-turn convenience of an explicit boundary and accepted write.
///
/// - Natural (default): may store; boundary and write algorithms decide.
///   Episode algorithms (accumulator + detect_boundary + write_gate) decide.
/// - Durable: may store; explicitly closes and commits the current unit.
///   Explicit turn: close unit and prefer committing it now.
/// - Boundary: may store; explicitly closes the current unit.
///   Close the unit at this signal; write still requires S_window.
/// - Ephemeral: never store; explicitly closes the retrieval turn.
///   Live context and retrieval still run; the write path must not persist.
enum class Retention
{
  // Zero-initialized signals must take the gated, non-forcing path.
  Natural = 0,
  Durable = 1,
  Boundary = 2,
  Ephemeral = 3,
};

/// @brief Whether retention creates an explicit turn boundary.
constexpr bool
RetentionForcesBoundary (Retention retention)
{
  return retention == Retention::Durable || retention == Retention::Boundary
         || retention == Retention::Ephemeral;
}

/// @brief Whether retention accepts a write without accumulator evidence.
constexpr bool
RetentionForcesWrite (Retention retention)
{
  return retention == Retention::Durable;
}

} // namespace cortext
