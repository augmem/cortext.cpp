#pragma once

namespace cortext
{

/// @brief Controls whether a processed signal may become durable memory.
enum class Retention
{
  Durable,
  Ephemeral,
};

} // namespace cortext
