#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cortext
{

// Forward declaration for ProcessorOutput fields
namespace operations
{
enum class Metric;
}

/// @brief High-level entrypoint for Cortext with tri-modal process stubs.
class Cortext
{
public:
  /// @brief Simplified output metrics from processor.
  struct ProcessorOutput
  {
    std::optional<double> composite_score;
    std::optional<double> threshold;
    std::optional<bool> decision;
    double effective_focus = 0.0;
    double coherence = 0.0;
    double emotion_intensity = 0.0;
    double valence = 0.5;
    double arousal = 0.0;
    std::unordered_map<int, double> metrics; // Metric enum cast to int -> value
  };
  
  /// @brief Context returned from processing calls, containing hydrated
  /// memories.
  struct Context
  {
    struct Memory
    {
      std::string source_id;
      long long id = 0;
      std::uint64_t timestamp = 0;
      long long retrieved_count = 0;
      long long used_count = 0;
      std::string content;
      std::string modality;
      std::string mimetype;
    };

    std::vector<Memory> memories;
    bool should_interrupt = false;
    ProcessorOutput output;
  };

  /// @brief Three-knob configuration preserved across the system.
  struct Config
  {
    double focus = 0.5;
    double sensitivity = 0.5;
    double stability = 0.5;
  };

  /// @brief Factory to create a Cortext instance.
  /// @param cfg Three-knob configuration.
  /// @param db_path SQLite database path (e.g. ":memory:" or a file path).
  /// @param models_dir Models directory (e.g. "models/imagebind").
  /// @return A unique_ptr to a Cortext instance.
  static std::unique_ptr<Cortext> Create (const Config &cfg,
                                          const std::string &db_path,
                                          const std::string &models_dir);

  /// @brief Process text input (stub - no embeddings yet).
  Context ProcessText (const std::string &text, std::uint64_t timestamp,
                       const std::string &source_id);

  /// @brief Process audio PCM input (stub - no embeddings yet).
  Context ProcessAudio (const float *pcm, std::size_t num_samples,
                        std::uint64_t timestamp, const std::string &source_id);

  /// @brief Process image input (stub - no embeddings yet).
  Context ProcessImage (const std::uint8_t *data, int width, int height,
                        int channels, std::uint64_t timestamp,
                        const std::string &source_id);

  /// @brief Attempt to trigger consolidation if conditions allow.
  Context Consolidate (std::uint64_t now_timestamp);

  /// @brief Flush/commit any pending episode writes.
  void Flush ();

#if defined(CORTEXT_TESTING)
  /// @brief Test-only helper to hydrate arbitrary memory ids.
  Context DebugHydrateForTest (const std::vector<long long> &candidate_ids,
                               const std::vector<long long> &used_ids);
#endif

  // Content persistence API (best-effort; no-op if id unavailable).
  // These helpers allow callers to persist raw content once a memory id
  // exists. Implemented using sqlite-objstore for payloads and SQLite
  // memory_index for metadata.
  static std::string MakeContentKey (long long embedding_id);
  static std::string MakeAudioMimePcmF32 ();
  static std::string MakeImageMimePng ();

  ~Cortext ();

  Cortext (const Cortext &) = delete;
  Cortext &operator= (const Cortext &) = delete;
  Cortext (Cortext &&) = delete;
  Cortext &operator= (Cortext &&) = delete;

private:
  struct Impl;
  explicit Cortext (const Config &cfg, const std::string &db_path,
                    const std::string &models_dir);
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext
