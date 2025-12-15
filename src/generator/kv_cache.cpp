#include "cortext/generator/kv_cache.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace cortext
{

KVCache::KVCache (KVSpec spec, std::size_t capacity)
    : spec_ (spec), capacity_ (std::max (capacity, std::size_t{ 1 }))
{
  const std::size_t B = static_cast<std::size_t> (spec_.batch_size);
  const std::size_t H = static_cast<std::size_t> (spec_.num_kv_heads);
  const std::size_t D = static_cast<std::size_t> (spec_.head_dim);
  const std::size_t buffer_size = B * H * capacity_ * D;

  keys_.reserve (spec_.num_layers);
  values_.reserve (spec_.num_layers);

  for (int layer = 0; layer < spec_.num_layers; ++layer)
    {
      keys_.emplace_back (buffer_size, 0.0f);
      values_.emplace_back (buffer_size, 0.0f);
    }
}

std::size_t
KVCache::FlatIndex (int b, int h, std::size_t t, int d) const
{
  // Shape: [B, H, T, D] in row-major order
  const std::size_t H = static_cast<std::size_t> (spec_.num_kv_heads);
  const std::size_t D = static_cast<std::size_t> (spec_.head_dim);
  return static_cast<std::size_t> (b) * (H * capacity_ * D)
         + static_cast<std::size_t> (h) * (capacity_ * D) + t * D
         + static_cast<std::size_t> (d);
}

void
KVCache::CopySlice (const float *src, std::size_t src_t, float *dst,
                    std::size_t dst_t, std::size_t count)
{
  const std::size_t B = static_cast<std::size_t> (spec_.batch_size);
  const std::size_t H = static_cast<std::size_t> (spec_.num_kv_heads);
  const std::size_t D = static_cast<std::size_t> (spec_.head_dim);

  // Copy count time steps for all batch/head combinations
  for (std::size_t b = 0; b < B; ++b)
    {
      for (std::size_t h = 0; h < H; ++h)
        {
          const float *src_ptr = src + b * (H * count * D) + h * (count * D);
          float *dst_ptr
              = dst + b * (H * capacity_ * D) + h * (capacity_ * D) + dst_t * D;

          for (std::size_t t = 0; t < count; ++t)
            {
              std::memcpy (dst_ptr + t * D, src_ptr + t * D, D * sizeof (float));
            }
        }
    }
}

std::map<std::string, std::vector<float> >
KVCache::PastInputs () const
{
  std::map<std::string, std::vector<float> > past;

  const std::size_t B = static_cast<std::size_t> (spec_.batch_size);
  const std::size_t H = static_cast<std::size_t> (spec_.num_kv_heads);
  const std::size_t D = static_cast<std::size_t> (spec_.head_dim);
  const std::size_t output_size = B * H * length_ * D;

  for (int layer = 0; layer < spec_.num_layers; ++layer)
    {
      std::string key_name
          = "past_key_values." + std::to_string (layer) + ".key";
      std::string value_name
          = "past_key_values." + std::to_string (layer) + ".value";

      // Create views (copies) of the current cache up to length_
      std::vector<float> key_out (output_size);
      std::vector<float> value_out (output_size);

      // Copy [B, H, length_, D] from [B, H, capacity_, D]
      for (std::size_t b = 0; b < B; ++b)
        {
          for (std::size_t h = 0; h < H; ++h)
            {
              const float *src_k
                  = keys_[layer].data () + b * (H * capacity_ * D)
                    + h * (capacity_ * D);
              const float *src_v
                  = values_[layer].data () + b * (H * capacity_ * D)
                    + h * (capacity_ * D);
              float *dst_k
                  = key_out.data () + b * (H * length_ * D) + h * (length_ * D);
              float *dst_v = value_out.data () + b * (H * length_ * D)
                             + h * (length_ * D);

              std::memcpy (dst_k, src_k, length_ * D * sizeof (float));
              std::memcpy (dst_v, src_v, length_ * D * sizeof (float));
            }
        }

      past[key_name] = std::move (key_out);
      past[value_name] = std::move (value_out);
    }

  return past;
}

void
KVCache::InitFromPresent (
    const std::map<std::string, std::vector<float> > &present,
    const std::vector<std::size_t> &shape)
{
  // Shape should be [B, H, seq_len, D]
  if (shape.size () < 4)
    {
      throw std::runtime_error (
          "KVCache::InitFromPresent: invalid shape, expected 4 dimensions");
    }

  const std::size_t seq_len = shape[2];
  EnsureCapacity (seq_len);

  const std::size_t B = static_cast<std::size_t> (spec_.batch_size);
  const std::size_t H = static_cast<std::size_t> (spec_.num_kv_heads);
  const std::size_t D = static_cast<std::size_t> (spec_.head_dim);

  for (int layer = 0; layer < spec_.num_layers; ++layer)
    {
      // Try present_key_values first, fall back to past_key_values
      std::string key_name
          = "present_key_values." + std::to_string (layer) + ".key";
      auto key_it = present.find (key_name);
      if (key_it == present.end ())
        {
          key_name = "past_key_values." + std::to_string (layer) + ".key";
          key_it = present.find (key_name);
        }

      std::string value_name
          = "present_key_values." + std::to_string (layer) + ".value";
      auto value_it = present.find (value_name);
      if (value_it == present.end ())
        {
          value_name = "past_key_values." + std::to_string (layer) + ".value";
          value_it = present.find (value_name);
        }

      if (key_it == present.end () || value_it == present.end ())
        continue;

      // Copy [B, H, seq_len, D] from present to cache [B, H, capacity_, D]
      const float *src_k = key_it->second.data ();
      const float *src_v = value_it->second.data ();

      for (std::size_t b = 0; b < B; ++b)
        {
          for (std::size_t h = 0; h < H; ++h)
            {
              const float *src_k_ptr
                  = src_k + b * (H * seq_len * D) + h * (seq_len * D);
              const float *src_v_ptr
                  = src_v + b * (H * seq_len * D) + h * (seq_len * D);
              float *dst_k = keys_[layer].data () + b * (H * capacity_ * D)
                             + h * (capacity_ * D);
              float *dst_v = values_[layer].data () + b * (H * capacity_ * D)
                             + h * (capacity_ * D);

              std::memcpy (dst_k, src_k_ptr, seq_len * D * sizeof (float));
              std::memcpy (dst_v, src_v_ptr, seq_len * D * sizeof (float));
            }
        }
    }

  length_ = seq_len;
}

void
KVCache::AppendFromPresent (
    const std::map<std::string, std::vector<float> > &present)
{
  EnsureCapacity (length_ + 1);

  const std::size_t B = static_cast<std::size_t> (spec_.batch_size);
  const std::size_t H = static_cast<std::size_t> (spec_.num_kv_heads);
  const std::size_t D = static_cast<std::size_t> (spec_.head_dim);

  for (int layer = 0; layer < spec_.num_layers; ++layer)
    {
      // Try present_key_values first, fall back to past_key_values
      std::string key_name
          = "present_key_values." + std::to_string (layer) + ".key";
      auto key_it = present.find (key_name);
      if (key_it == present.end ())
        {
          key_name = "past_key_values." + std::to_string (layer) + ".key";
          key_it = present.find (key_name);
        }

      std::string value_name
          = "present_key_values." + std::to_string (layer) + ".value";
      auto value_it = present.find (value_name);
      if (value_it == present.end ())
        {
          value_name = "past_key_values." + std::to_string (layer) + ".value";
          value_it = present.find (value_name);
        }

      if (key_it == present.end () || value_it == present.end ())
        continue;

      // Determine sequence length from tensor size
      // Expected shape: [B, H, seq_len, D]
      const std::size_t tensor_size = key_it->second.size ();
      const std::size_t seq_len = tensor_size / (B * H * D);
      if (seq_len == 0)
        continue;

      // Take last time position
      const std::size_t src_idx = seq_len - 1;
      const float *src_k = key_it->second.data ();
      const float *src_v = value_it->second.data ();

      for (std::size_t b = 0; b < B; ++b)
        {
          for (std::size_t h = 0; h < H; ++h)
            {
              const float *src_k_ptr = src_k + b * (H * seq_len * D)
                                       + h * (seq_len * D) + src_idx * D;
              const float *src_v_ptr = src_v + b * (H * seq_len * D)
                                       + h * (seq_len * D) + src_idx * D;
              float *dst_k = keys_[layer].data () + b * (H * capacity_ * D)
                             + h * (capacity_ * D) + length_ * D;
              float *dst_v = values_[layer].data () + b * (H * capacity_ * D)
                             + h * (capacity_ * D) + length_ * D;

              std::memcpy (dst_k, src_k_ptr, D * sizeof (float));
              std::memcpy (dst_v, src_v_ptr, D * sizeof (float));
            }
        }
    }

  length_ += 1;
}

void
KVCache::EnsureCapacity (std::size_t needed)
{
  if (needed <= capacity_)
    return;

  const std::size_t new_capacity
      = std::max (static_cast<std::size_t> (capacity_ * 1.5), needed);
  const std::size_t B = static_cast<std::size_t> (spec_.batch_size);
  const std::size_t H = static_cast<std::size_t> (spec_.num_kv_heads);
  const std::size_t D = static_cast<std::size_t> (spec_.head_dim);
  const std::size_t new_buffer_size = B * H * new_capacity * D;

  for (int layer = 0; layer < spec_.num_layers; ++layer)
    {
      std::vector<float> new_keys (new_buffer_size, 0.0f);
      std::vector<float> new_values (new_buffer_size, 0.0f);

      // Copy existing data [B, H, length_, D]
      for (std::size_t b = 0; b < B; ++b)
        {
          for (std::size_t h = 0; h < H; ++h)
            {
              const float *src_k = keys_[layer].data ()
                                   + b * (H * capacity_ * D)
                                   + h * (capacity_ * D);
              const float *src_v = values_[layer].data ()
                                   + b * (H * capacity_ * D)
                                   + h * (capacity_ * D);
              float *dst_k = new_keys.data () + b * (H * new_capacity * D)
                             + h * (new_capacity * D);
              float *dst_v = new_values.data () + b * (H * new_capacity * D)
                             + h * (new_capacity * D);

              std::memcpy (dst_k, src_k, length_ * D * sizeof (float));
              std::memcpy (dst_v, src_v, length_ * D * sizeof (float));
            }
        }

      keys_[layer] = std::move (new_keys);
      values_[layer] = std::move (new_values);
    }

  capacity_ = new_capacity;
}

void
KVCache::Clear ()
{
  length_ = 0;
  // Note: We don't zero the buffers for efficiency - they'll be overwritten
}

} // namespace cortext
