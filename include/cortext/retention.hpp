#pragma once

namespace cortext
{

/// @brief Controls storage eligibility and turn-edge policy for a signal.
///
/// When the retention argument is omitted from Process* APIs, the default is
/// Natural (no forced edge, no forced write). Durable keeps the historical
/// chat-turn convenience: force_boundary + force_write.
///
/// - Natural (default): may store; force_boundary=false, force_write=false.
///   Episode algorithms (accumulator + detect_boundary + write_gate) decide.
/// - Durable: may store; force_boundary=true, force_write=true.
///   Explicit turn: close unit and prefer committing it now.
/// - Boundary: may store; force_boundary=true, force_write=false.
///   Close the unit at this signal; write still requires S_window.
/// - Ephemeral: never store; force_boundary=true for retrieval pacing.
///   Live context and retrieval still run; the write path must not persist.
enum class Retention
{
  Natural,
  Durable,
  Boundary,
  Ephemeral,
};

} // namespace cortext
