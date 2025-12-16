#include "cortext/generator/gemma_text_generator.hpp"

#include "cortext/generator/constrained_sampler.hpp"
#include "cortext/generator/decoder_runner.hpp"
#include "cortext/generator/gemma_tokenizer.hpp"
#include "cortext/generator/json_decoder.hpp"
#include "cortext/generator/kv_cache.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace cortext
{

namespace
{

// Model configuration (gemma-3n-E2B-it)
constexpr int kNumLayers = 30;
constexpr int kNumKVHeads = 2;   // Grouped-query attention
constexpr int kHeadDim = 256;
constexpr int kHiddenDim = 2048; // Embedding dimension
constexpr int64_t kEosTokenId = 1;
constexpr int64_t kBosTokenId = 2;

/// @brief Apply softmax to logits.
std::vector<float>
Softmax (const std::vector<float> &logits)
{
  std::vector<float> result (logits.size ());

  // Find max for numerical stability
  float max_val = *std::max_element (logits.begin (), logits.end ());

  // Compute exp and sum
  float sum = 0.0f;
  for (std::size_t i = 0; i < logits.size (); ++i)
    {
      result[i] = std::exp (logits[i] - max_val);
      sum += result[i];
    }

  // Normalize
  if (sum > 0.0f)
    {
      for (std::size_t i = 0; i < result.size (); ++i)
        {
          result[i] /= sum;
        }
    }

  return result;
}

/// @brief Sample token from logits with optional temperature, top-k, top-p.
int64_t
SampleToken (const std::vector<float> &logits, float temperature, int top_k,
             float top_p, std::mt19937 &rng)
{
  // Greedy decoding
  if (temperature <= 0.0f)
    {
      return static_cast<int64_t> (
          std::distance (logits.begin (),
                         std::max_element (logits.begin (), logits.end ())));
    }

  // Scale by temperature
  std::vector<float> scaled (logits.size ());
  float temp = std::max (1e-6f, temperature);
  for (std::size_t i = 0; i < logits.size (); ++i)
    {
      scaled[i] = logits[i] / temp;
    }

  // Apply top-k filtering
  if (top_k > 0 && top_k < static_cast<int> (scaled.size ()))
    {
      // Find k-th largest value
      std::vector<std::size_t> indices (scaled.size ());
      std::iota (indices.begin (), indices.end (), 0);
      std::partial_sort (
          indices.begin (), indices.begin () + top_k, indices.end (),
          [&scaled] (std::size_t a, std::size_t b) {
            return scaled[a] > scaled[b];
          });

      // Keep only top-k, set rest to -inf
      std::vector<float> masked (scaled.size (),
                                 -std::numeric_limits<float>::infinity ());
      for (int i = 0; i < top_k; ++i)
        {
          masked[indices[i]] = scaled[indices[i]];
        }
      scaled = std::move (masked);
    }

  // Apply top-p (nucleus) filtering
  if (top_p > 0.0f && top_p < 1.0f)
    {
      // Sort indices by logit value descending
      std::vector<std::size_t> indices (scaled.size ());
      std::iota (indices.begin (), indices.end (), 0);
      std::sort (indices.begin (), indices.end (),
                 [&scaled] (std::size_t a, std::size_t b) {
                   return scaled[a] > scaled[b];
                 });

      // Compute probabilities of sorted logits
      std::vector<float> sorted_logits (scaled.size ());
      for (std::size_t i = 0; i < indices.size (); ++i)
        {
          sorted_logits[i] = scaled[indices[i]];
        }
      auto probs = Softmax (sorted_logits);

      // Find cutoff where cumulative probability exceeds top_p
      float cumsum = 0.0f;
      std::size_t cutoff = 0;
      for (std::size_t i = 0; i < probs.size (); ++i)
        {
          cumsum += probs[i];
          if (cumsum >= top_p)
            {
              cutoff = i + 1;
              break;
            }
        }
      if (cutoff == 0)
        cutoff = 1;

      // Keep only tokens within nucleus
      std::vector<float> masked (scaled.size (),
                                 -std::numeric_limits<float>::infinity ());
      for (std::size_t i = 0; i < cutoff; ++i)
        {
          masked[indices[i]] = scaled[indices[i]];
        }
      scaled = std::move (masked);
    }

  // Convert to probabilities
  auto probs = Softmax (scaled);

  // Check if any probabilities are valid
  bool any_valid = false;
  for (float p : probs)
    {
      if (std::isfinite (p) && p > 0.0f)
        {
          any_valid = true;
          break;
        }
    }

  if (!any_valid)
    {
      // Fall back to greedy
      return static_cast<int64_t> (
          std::distance (logits.begin (),
                         std::max_element (logits.begin (), logits.end ())));
    }

  // Sample from distribution
  std::discrete_distribution<std::size_t> dist (probs.begin (), probs.end ());
  return static_cast<int64_t> (dist (rng));
}

} // namespace

// ============================================================================
// GemmaTextGenerator Implementation
// ============================================================================

struct GemmaTextGenerator::Impl
{
  std::string models_dir;
  std::unique_ptr<GemmaTokenizer> tokenizer;
  std::unique_ptr<DecoderRunner> decoder;
  std::unique_ptr<EmbedRunner> embedder;

  // Random number generator for sampling
  std::mt19937 rng{ std::random_device{}() };

  // Last generation stats
  std::size_t last_token_count = 0;

  bool available = false;

  void
  LoadModels (const std::string &dir)
  {
    models_dir = dir;

    // Load tokenizer
    std::filesystem::path tokenizer_path
        = std::filesystem::path (dir) / "tokenizer.json";
    if (!std::filesystem::exists (tokenizer_path))
      {
        throw std::runtime_error ("Tokenizer not found: "
                                  + tokenizer_path.string ());
      }
    tokenizer = std::make_unique<GemmaTokenizer> (tokenizer_path.string ());

    // Load ONNX models (conditional on CORTEXT_ENABLE_GEMMA_ORT)
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
    decoder = std::make_unique<DecoderRunner> (dir);
    embedder = std::make_unique<EmbedRunner> (dir);
    available = decoder->IsAvailable () && embedder->IsAvailable ();
#else
    available = false;
#endif
  }

  /// @brief Embed token IDs to get input embeddings and per-layer inputs.
  /// Uses EmbedStep for single-token decode steps (more efficient).
  EmbedOutput
  EmbedTokens (const std::vector<int64_t> &token_ids)
  {
    if (!embedder || !embedder->IsAvailable ())
      {
        throw std::runtime_error ("Embedder not available");
      }

    // Use optimized single-token path for decode steps
    if (token_ids.size () == 1)
      {
        return embedder->EmbedStep (token_ids[0]);
      }

    // Use prefill path for multi-token sequences
    std::vector<std::size_t> shape = { 1, token_ids.size () };
    return embedder->EmbedPrefill (token_ids, shape);
  }

  /// @brief Run decoder with embeddings and KV cache.
  DecoderOutput
  RunDecoder (const EmbedOutput &embed_output,
              const std::vector<int64_t> &position_ids, KVCache &kv_cache)
  {
    if (!decoder || !decoder->IsAvailable ())
      {
        throw std::runtime_error ("Decoder not available");
      }

    // Get past KV from cache
    auto past_kv = kv_cache.PastInputs ();

    // Position IDs shape
    std::vector<std::size_t> position_shape = { 1, position_ids.size () };

    // KV cache shape for past inputs
    std::vector<std::size_t> past_kv_shape
        = { 1, static_cast<std::size_t> (kNumKVHeads), kv_cache.Length (),
            static_cast<std::size_t> (kHeadDim) };

    return decoder->Run (embed_output.inputs_embeds, embed_output.embeds_shape,
                         embed_output.per_layer_inputs,
                         embed_output.per_layer_shape, position_ids,
                         position_shape, past_kv, past_kv_shape);
  }

  /// @brief Generate JSON with schema constraints.
  nlohmann::json
  GenerateJSONImpl (const std::vector<int64_t> &input_tokens,
                    const nlohmann::json &schema, int max_tokens,
                    float temperature, int top_k, float top_p)
  {
    if (!available)
      {
        throw std::runtime_error ("Generator not available");
      }

    // Initialize parser and sampler
    StreamingJSONParser parser (schema);
    SchemaConstrainedSampler sampler (*tokenizer, parser);

    // Initialize KV cache
    std::size_t capacity
        = input_tokens.size () + static_cast<std::size_t> (max_tokens) + 8;
    KVSpec spec{ kNumLayers, kNumKVHeads, kHeadDim, 1 };
    KVCache kv_cache (spec, capacity);

    // Track generated tokens
    std::vector<int64_t> generated_tokens;

    // Current input for next step
    std::vector<int64_t> current_input = input_tokens;

    // Generation loop
    for (int step = 0; step < max_tokens; ++step)
      {
        // Embed tokens (returns both embeddings and per_layer_inputs)
        auto embed_output = EmbedTokens (current_input);

        // Create position IDs
        std::vector<int64_t> position_ids (current_input.size ());
        std::size_t pos_start = kv_cache.Length ();
        for (std::size_t i = 0; i < current_input.size (); ++i)
          {
            position_ids[i] = static_cast<int64_t> (pos_start + i);
          }

        // Run decoder
        auto output = RunDecoder (embed_output, position_ids, kv_cache);

        // Update KV cache
        if (step == 0)
          {
            // First step: initialize from prefill output
            kv_cache.InitFromPresent (output.present_kv, output.kv_shape);
          }
        else
          {
            // Subsequent steps: append single token's KV
            kv_cache.AppendFromPresent (output.present_kv);
          }

        // Get logits for last position
        // logits_shape is [batch, seq_len, vocab_size]
        std::size_t vocab_size = output.logits_shape.back ();
        std::size_t seq_len = output.logits_shape[1];
        std::size_t last_pos_offset = (seq_len - 1) * vocab_size;

        std::vector<float> logits (output.logits.begin () + last_pos_offset,
                                   output.logits.begin () + last_pos_offset
                                       + vocab_size);

        // Apply schema constraints
        auto constrained_logits = sampler.ConstrainLogits (logits);

        // Sample next token
        int64_t next_token
            = SampleToken (constrained_logits, temperature, top_k, top_p, rng);
        generated_tokens.push_back (next_token);

        // Decode token and update parser
        std::string token_text = tokenizer->Decode ({ next_token });
        ParserStatus status = parser.AddToken (token_text);

        // Check completion
        if (status == ParserStatus::kComplete)
          {
            break;
          }
        else if (status == ParserStatus::kInvalid)
          {
            throw std::runtime_error (
                "Invalid JSON state during generation. Buffer: "
                + parser.Buffer ());
          }

        // Check EOS
        if (next_token == tokenizer->EosTokenId ())
          {
            break;
          }

        // Prepare next input (single token)
        current_input = { next_token };
      }

    // Store stats
    last_token_count = generated_tokens.size ();

    return parser.GetParsedObject ();
  }

  /// @brief Generate unconstrained text.
  std::string
  GenerateTextImpl (const std::vector<int64_t> &input_tokens,
                    const GenerationConfig &config)
  {
    if (!available)
      {
        throw std::runtime_error ("Generator not available");
      }

    // Initialize KV cache
    std::size_t capacity = input_tokens.size ()
                           + static_cast<std::size_t> (config.max_new_tokens)
                           + 8;
    KVSpec spec{ kNumLayers, kNumKVHeads, kHeadDim, 1 };
    KVCache kv_cache (spec, capacity);

    // Track generated tokens
    std::vector<int64_t> generated_tokens;

    // Current input for next step
    std::vector<int64_t> current_input = input_tokens;

    // Generation loop
    for (int step = 0; step < config.max_new_tokens; ++step)
      {
        // Embed tokens (returns both embeddings and per_layer_inputs)
        auto embed_output = EmbedTokens (current_input);

        // Create position IDs
        std::vector<int64_t> position_ids (current_input.size ());
        std::size_t pos_start = kv_cache.Length ();
        for (std::size_t i = 0; i < current_input.size (); ++i)
          {
            position_ids[i] = static_cast<int64_t> (pos_start + i);
          }

        // Run decoder
        auto output = RunDecoder (embed_output, position_ids, kv_cache);

        // Update KV cache
        if (step == 0)
          {
            kv_cache.InitFromPresent (output.present_kv, output.kv_shape);
          }
        else
          {
            kv_cache.AppendFromPresent (output.present_kv);
          }

        // Get logits for last position
        std::size_t vocab_size = output.logits_shape.back ();
        std::size_t seq_len = output.logits_shape[1];
        std::size_t last_pos_offset = (seq_len - 1) * vocab_size;

        std::vector<float> logits (output.logits.begin () + last_pos_offset,
                                   output.logits.begin () + last_pos_offset
                                       + vocab_size);

        // Sample next token
        int top_k = config.do_sample ? config.top_k : 0;
        float top_p = config.do_sample ? config.top_p : 0.0f;
        float temp = config.do_sample ? config.temperature : 0.0f;

        int64_t next_token = SampleToken (logits, temp, top_k, top_p, rng);
        generated_tokens.push_back (next_token);

        // Check EOS
        if (next_token == tokenizer->EosTokenId ())
          {
            break;
          }

        // Prepare next input
        current_input = { next_token };
      }

    // Store stats
    last_token_count = generated_tokens.size ();

    // Decode all generated tokens
    return tokenizer->Decode (generated_tokens);
  }
};

GemmaTextGenerator::GemmaTextGenerator (const std::string &models_dir)
    : impl_ (std::make_unique<Impl> ())
{
  impl_->LoadModels (models_dir);
}

GemmaTextGenerator::~GemmaTextGenerator () = default;
GemmaTextGenerator::GemmaTextGenerator (GemmaTextGenerator &&) noexcept
    = default;
GemmaTextGenerator &
GemmaTextGenerator::operator= (GemmaTextGenerator &&) noexcept
    = default;

std::string
GemmaTextGenerator::Generate (const std::string &prompt,
                              const GenerationConfig &config)
{
  // Tokenize prompt (with BOS)
  auto tokens = impl_->tokenizer->Encode (prompt, true);
  return impl_->GenerateTextImpl (tokens, config);
}

std::string
GemmaTextGenerator::GenerateMultimodal (
    const std::vector<GenerationInput> &inputs, const GenerationConfig &config)
{
  // For now, concatenate text inputs and ignore non-text modalities
  // Full multimodal support requires multimodal_encoder implementation
  std::string combined_text;
  for (const auto &input : inputs)
    {
      if (input.modality == "text")
        {
          combined_text += input.text;
        }
      else
        {
          // TODO: Encode image/audio with multimodal encoder
          throw std::runtime_error ("Multimodal input not yet supported: "
                                    + input.modality);
        }
    }

  return Generate (combined_text, config);
}

bool
GemmaTextGenerator::IsAvailable () const
{
  return impl_->available;
}

nlohmann::json
GemmaTextGenerator::GenerateJSON (const std::string &prompt,
                                  const nlohmann::json &schema, int max_tokens,
                                  float temperature, int top_k, float top_p)
{
  // Tokenize prompt (with BOS)
  auto tokens = impl_->tokenizer->Encode (prompt, true);
  return impl_->GenerateJSONImpl (tokens, schema, max_tokens, temperature,
                                  top_k, top_p);
}

nlohmann::json
GemmaTextGenerator::GenerateJSONMultimodal (
    const std::vector<GenerationInput> &inputs, const nlohmann::json &schema,
    int max_tokens, float temperature, int top_k, float top_p)
{
  // For now, concatenate text inputs
  std::string combined_text;
  for (const auto &input : inputs)
    {
      if (input.modality == "text")
        {
          combined_text += input.text;
        }
      else
        {
          throw std::runtime_error ("Multimodal input not yet supported: "
                                    + input.modality);
        }
    }

  return GenerateJSON (combined_text, schema, max_tokens, temperature, top_k,
                       top_p);
}

std::size_t
GemmaTextGenerator::LastTokenCount () const
{
  return impl_->last_token_count;
}

int64_t
GemmaTextGenerator::EosTokenId () const
{
  return impl_->tokenizer ? impl_->tokenizer->EosTokenId () : kEosTokenId;
}

} // namespace cortext
