#pragma once

#include <Eigen/Dense>
#include <optional>
#include <string>
#include <vector>

#include "cortext/retention.hpp"

namespace cortext
{

/// @brief Represents a single unit of data to be processed.
///
/// A signal is a modality-agnostic representation of an event, typically
/// containing an embedding vector and relevant metadata. When a payload is
/// provided, the MemoryStorage operation will persist it to objstore if the
/// write gate decision passes.
struct Signal
{
  // The universal representation of the input data.
  Eigen::VectorXf embedding;

  // Optional higher-dimensional signal representation for engine-side
  // continuity policies such as SoftAnchor. Retrieval/storage may consume a
  // compact embedding while ingress policies consume this richer signal.
  std::optional<Eigen::VectorXf> soft_anchor_embedding;

  // Core metadata.
  uint64_t timestamp = 0;
  // Opaque provenance/grouping key. Core code may compare it for exact stream
  // equality, but must not parse application roles or modalities from it.
  std::string source_id;
  bool force_boundary = false;
  bool force_write = false;
  // Internal maintenance hint. When set, the signal drives consolidation
  // without giving behavioral meaning to the opaque source_id string.
  bool force_consolidation = false;
  // Durable by default. Ephemeral signals update live processing state and
  // retrieval context, but the write path must not persist them as memories.
  Retention retention = Retention::Durable;

  // Payload for storage (persisted to objstore when write gate passes).
  std::optional<std::vector<unsigned char>> payload;
  std::string modality;  // "text" | "audio" | "image"
  std::string mimetype;  // "text/plain" | "audio/pcm;format=f32" | "image/png"

  // Image-specific metadata (when modality == "image").
  int width = 0;
  int height = 0;
  int channels = 0;

  // Audio-specific metadata (when modality == "audio").
  int sample_rate = 0;
  std::size_t num_samples = 0;

};

} // namespace cortext
